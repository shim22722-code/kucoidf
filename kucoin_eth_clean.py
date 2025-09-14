#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
쿠코인 선물 자동매매 봇 (ETH 전용) - 로깅 없는 버전
- ETH/USDT 선물 거래
- EMA + RSI 기반 신호
- ATR 기반 리스크 관리
- 텔레그램 연동
"""

import os
import sys
import time
import asyncio
import json
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass

# 필수 패키지 임포트
try:
    import ccxt
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"필수 패키지를 설치하세요: pip install ccxt pandas numpy")
    sys.exit(1)

# TA-Lib 임포트 (선택사항)
try:
    import talib
    TALIB_AVAILABLE = True
    print("TA-Lib 사용 가능")
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib 미설치 - 기본 지표 사용")

# 텔레그램 봇 임포트 (선택사항)
try:
    import telepot
    from telepot.loop import MessageLoop
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("telepot 미설치 - 텔레그램 기능 비활성화")

# ========================================
# 환경변수 및 API 설정
# ========================================

KUCOIN_API_KEY = os.getenv('KUCOIN_API_KEY', 'your_api_key')
KUCOIN_SECRET = os.getenv('KUCOIN_SECRET', 'your_secret')
KUCOIN_PASSPHRASE = os.getenv('KUCOIN_PASSPHRASE', 'your_passphrase')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# .env 파일이 있으면 로드
try:
    from dotenv import load_dotenv
    if os.path.exists('.env'):
        load_dotenv()
        KUCOIN_API_KEY = os.getenv('KUCOIN_API_KEY', KUCOIN_API_KEY)
        KUCOIN_SECRET = os.getenv('KUCOIN_SECRET', KUCOIN_SECRET)
        KUCOIN_PASSPHRASE = os.getenv('KUCOIN_PASSPHRASE', KUCOIN_PASSPHRASE)
        TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', TELEGRAM_BOT_TOKEN)
        TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', TELEGRAM_CHAT_ID)
except ImportError:
    pass

# ========================================
# 출력 함수
# ========================================

def print_info(message):
    """정보 메시지 출력"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] INFO: {message}")

def print_error(message):
    """오류 메시지 출력"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] ERROR: {message}")

def print_warning(message):
    """경고 메시지 출력"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] WARNING: {message}")

# ========================================
# 기본 지표 계산 함수
# ========================================

def calculate_ema(prices, period):
    """EMA 계산"""
    prices = np.array(prices)
    alpha = 2 / (period + 1.0)
    alpha_rev = 1 - alpha
    
    pows = alpha_rev ** (np.arange(len(prices) + 1))
    scale_arr = 1 / pows[:-1]
    offset = prices[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (len(prices) - 1)
    
    mult = prices * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    
    return out

def calculate_rsi(prices, period=14):
    """RSI 계산"""
    prices = np.array(prices)
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)
    
    for i in range(period, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """ATR 계산"""
    high, low, close = np.array(high), np.array(low), np.array(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(tr)
    atr[period-1] = tr[:period].mean()
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

@dataclass
class TradingConfig:
    """트레이딩 설정"""
    symbol = 'ETH/USDT:USDT'
    timeframe = '1h'
    seed_money = 17.0
    risk_per_trade = 0.04
    leverage = 10
    
    ema_fast = 12
    ema_slow = 26
    
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    
    atr_period = 14
    atr_volatility_threshold = 0.002
    
    stop_loss_atr_multiplier = 2.0
    take_profit_levels = [
        {'percentage': 50, 'atr_multiplier': 2.0},
        {'percentage': 25, 'atr_multiplier': 3.0},
        {'percentage': 25, 'atr_multiplier': 4.0}
    ]
    
    check_interval = 60

class Position:
    """포지션 정보 클래스"""
    def __init__(self, symbol: str, side: str, size: float, entry_price: float, 
                 stop_loss: float, take_profits: List[float], leverage: int = 1):
        self.symbol = symbol
        self.side = side
        self.size = size
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profits = take_profits
        self.leverage = leverage
        self.entry_time = datetime.now(timezone.utc)
        self.remaining_size = size

class KucoinETHBot:
    """쿠코인 ETH 선물 자동매매 봇"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.is_running = False
        self.position: Optional[Position] = None
        self.telegram_bot = None
        
        # 쿠코인 거래소 연결
        self.exchange = ccxt.kucoinfutures({
            'apiKey': KUCOIN_API_KEY,
            'secret': KUCOIN_SECRET,
            'password': KUCOIN_PASSPHRASE,
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # 텔레그램 봇 초기화
        if TELEGRAM_AVAILABLE and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            try:
                self.telegram_bot = telepot.Bot(TELEGRAM_BOT_TOKEN)
                MessageLoop(self.telegram_bot, self.handle_telegram_message).run_as_thread()
            except Exception as e:
                print_error(f"텔레그램 봇 초기화 실패: {e}")
                self.telegram_bot = None
        
        print_info("쿠코인 ETH 선물 자동매매 봇 초기화 완료")
    
    async def send_telegram_message(self, message: str):
        """텔레그램 메시지 전송"""
        if self.telegram_bot and TELEGRAM_CHAT_ID:
            try:
                self.telegram_bot.sendMessage(TELEGRAM_CHAT_ID, message)
            except Exception as e:
                print_error(f"텔레그램 메시지 전송 실패: {e}")
    
    def handle_telegram_message(self, msg):
        """텔레그램 명령어 처리"""
        if not TELEGRAM_AVAILABLE:
            return
            
        content_type, chat_type, chat_id = telepot.glance(msg)
        
        if content_type == 'text' and str(chat_id) == TELEGRAM_CHAT_ID:
            command = msg['text'].lower().strip()
            
            if command == '/start':
                self.start_trading()
                self.telegram_bot.sendMessage(chat_id, "🚀 ETH 자동매매를 시작합니다!")
            
            elif command == '/stop':
                self.stop_trading()
                self.telegram_bot.sendMessage(chat_id, "⏹️ ETH 자동매매를 중지합니다!")
            
            elif command == '/balance':
                balance_info = self.get_balance_info()
                self.telegram_bot.sendMessage(chat_id, balance_info)
            
            elif command == '/position':
                position_info = self.get_position_info()
                self.telegram_bot.sendMessage(chat_id, position_info)
            
            elif command == '/pnl':
                pnl_info = self.get_pnl_info()
                self.telegram_bot.sendMessage(chat_id, pnl_info)
    
    def start_trading(self):
        """자동매매 시작"""
        self.is_running = True
        print_info("ETH 자동매매 시작")
    
    def stop_trading(self):
        """자동매매 중지"""
        self.is_running = False
        print_info("ETH 자동매매 중지")
    
    def get_balance_info(self) -> str:
        """잔고 정보 조회"""
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {})
            free = usdt_balance.get('free', 0)
            used = usdt_balance.get('used', 0)
            total = usdt_balance.get('total', 0)
            
            return f"""💰 잔고 정보
Free: {free:.2f} USDT
Used: {used:.2f} USDT
Total: {total:.2f} USDT"""
        except Exception as e:
            return f"❌ 잔고 조회 실패: {e}"
    
    def get_position_info(self) -> str:
        """포지션 정보 조회"""
        if not self.position:
            return "📊 현재 보유 중인 ETH 포지션이 없습니다."
        
        pnl = self.calculate_position_pnl(self.position)
        return f"""📊 현재 ETH 포지션:
Side: {self.position.side.upper()}
Size: {self.position.remaining_size:.4f}
Entry: ${self.position.entry_price:.2f}
Stop Loss: ${self.position.stop_loss:.2f}
PnL: {pnl:.2f} USDT
Entry Time: {self.position.entry_time.strftime('%Y-%m-%d %H:%M:%S')}"""
    
    def get_pnl_info(self) -> str:
        """미실현 손익 조회"""
        if not self.position:
            return "📈 현재 포지션이 없어 손익이 0입니다."
        
        pnl = self.calculate_position_pnl(self.position)
        return f"📈 ETH 미실현 손익: {pnl:.2f} USDT"
    
    def calculate_position_pnl(self, position: Position) -> float:
        """포지션 미실현 손익 계산"""
        try:
            ticker = self.exchange.fetch_ticker(position.symbol)
            current_price = ticker['last']
            
            if position.side == 'long':
                pnl = (current_price - position.entry_price) * position.remaining_size * position.leverage
            else:
                pnl = (position.entry_price - current_price) * position.remaining_size * position.leverage
                
            return pnl
        except:
            return 0.0
    
    def fetch_ohlcv_data(self, limit: int = 100) -> pd.DataFrame:
        """OHLCV 데이터 가져오기"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.config.symbol, self.config.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print_error(f"OHLCV 데이터 가져오기 실패 {self.config.symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        if df.empty or len(df) < max(self.config.ema_slow, self.config.rsi_period, self.config.atr_period):
            return df
        
        if TALIB_AVAILABLE:
            df['ema_fast'] = talib.EMA(df['close'].values, timeperiod=self.config.ema_fast)
            df['ema_slow'] = talib.EMA(df['close'].values, timeperiod=self.config.ema_slow)
            df['rsi'] = talib.RSI(df['close'].values, timeperiod=self.config.rsi_period)
            df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, 
                                 timeperiod=self.config.atr_period)
        else:
            df['ema_fast'] = calculate_ema(df['close'].values, self.config.ema_fast)
            df['ema_slow'] = calculate_ema(df['close'].values, self.config.ema_slow)
            df['rsi'] = calculate_rsi(df['close'].values, self.config.rsi_period)
            df['atr'] = calculate_atr(df['high'].values, df['low'].values, 
                                    df['close'].values, self.config.atr_period)
        
        df['atr_ratio'] = df['atr'] / df['close']
        return df
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[str]:
        """매매 신호 생성"""
        if df.empty or len(df) < 2:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        if pd.isna(current['ema_fast']) or pd.isna(current['ema_slow']) or pd.isna(current['rsi']):
            return None
        
        if current['atr_ratio'] < self.config.atr_volatility_threshold:
            return None
        
        ema_cross_up = (current['ema_fast'] > current['ema_slow'] and 
                       previous['ema_fast'] <= previous['ema_slow'])
        ema_cross_down = (current['ema_fast'] < current['ema_slow'] and 
                         previous['ema_fast'] >= previous['ema_slow'])
        
        rsi_bullish = current['rsi'] < self.config.rsi_overbought
        rsi_bearish = current['rsi'] > self.config.rsi_oversold
        
        if ema_cross_up and rsi_bullish:
            return 'long'
        
        if ema_cross_down and rsi_bearish:
            return 'short'
        
        return None
    
    def calculate_position_size(self, entry_price: float, stop_loss_price: float) -> float:
        """포지션 크기 계산"""
        try:
            balance = self.exchange.fetch_balance()
            available_balance = balance['USDT']['free']
            
            risk_amount = available_balance * self.config.risk_per_trade
            
            if stop_loss_price > 0:
                loss_per_unit = abs(entry_price - stop_loss_price)
                position_size = (risk_amount / loss_per_unit) * self.config.leverage
            else:
                position_size = risk_amount * self.config.leverage / entry_price
            
            min_amount = 0.001
            return max(position_size, min_amount)
            
        except Exception as e:
            print_error(f"포지션 크기 계산 실패: {e}")
            return 0.0
    
    def calculate_stop_loss_take_profit(self, side: str, entry_price: float, 
                                      atr: float) -> Tuple[float, List[float]]:
        """손절가와 익절가 계산"""
        stop_loss_distance = atr * self.config.stop_loss_atr_multiplier
        
        if side == 'long':
            stop_loss = entry_price - stop_loss_distance
            take_profits = [
                entry_price + (atr * tp['atr_multiplier'])
                for tp in self.config.take_profit_levels
            ]
        else:
            stop_loss = entry_price + stop_loss_distance
            take_profits = [
                entry_price - (atr * tp['atr_multiplier'])
                for tp in self.config.take_profit_levels
            ]
        
        return stop_loss, take_profits
    
    async def open_position(self, side: str, df: pd.DataFrame):
        """포지션 진입"""
        try:
            if self.position:
                print_info("ETH에 이미 포지션이 존재합니다.")
                return
            
            current = df.iloc[-1]
            entry_price = current['close']
            atr = current['atr']
            
            if pd.isna(entry_price) or pd.isna(atr):
                print_error("유효하지 않은 가격 또는 ATR 값")
                return
            
            stop_loss, take_profits = self.calculate_stop_loss_take_profit(side, entry_price, atr)
            position_size = self.calculate_position_size(entry_price, stop_loss)
            
            if position_size <= 0:
                print_error("유효하지 않은 포지션 크기")
                return
            
            print_info(f"ETH 포지션 진입 시뮬레이션: {side} {position_size}")
            
            
            
            order = self.exchange.create_market_order(
                symbol=self.config.symbol,
                type='market',
                side='buy' if side == 'long' else 'sell',
                amount=position_size,
                params={
                    'leverage': self.config.leverage,
                    'marginMode': 'isolated'
                }
            )
            
            
            self.position = Position(
                symbol=self.config.symbol,
                side=side,
                size=position_size,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profits=take_profits,
                leverage=self.config.leverage
            )
            
            message = f"""🚀 ETH 포지션 진입 (시뮬레이션)
Side: {side.upper()}
Size: {position_size:.4f} ETH
Entry: ${entry_price:.2f}
Stop Loss: ${stop_loss:.2f}
Take Profits: {[f'${tp:.2f}' for tp in take_profits]}
Leverage: {self.config.leverage}x"""
            
            await self.send_telegram_message(message)
            print_info(f"ETH 포지션 진입 완료: {side}")
            
        except Exception as e:
            error_msg = f"ETH 포지션 진입 실패: {e}"
            print_error(error_msg)
            await self.send_telegram_message(f"❌ {error_msg}")
    
    async def check_and_update_position(self):
        """포지션 상태 확인 및 업데이트"""
        if not self.position:
            return
            
        try:
            elapsed_time = datetime.now(timezone.utc) - self.position.entry_time
            
            # 1시간 후 자동 청산 (시뮬레이션)
            if elapsed_time.total_seconds() > 3600:
                closed_position = self.position
                self.position = None
                
                message = f"""✅ ETH 포지션 청산 (시뮬레이션)
Side: {closed_position.side.upper()}
Entry: ${closed_position.entry_price:.2f}
Duration: {elapsed_time.total_seconds()/60:.1f}분"""
                
                await self.send_telegram_message(message)
                print_info("ETH 포지션 청산 시뮬레이션 완료")
            
        except Exception as e:
            print_error(f"ETH 포지션 상태 확인 실패: {e}")
    
    async def run_trading_cycle(self):
        """트레이딩 사이클 실행"""
        try:
            if self.position:
                await self.check_and_update_position()
                return
            
            df = self.fetch_ohlcv_data()
            if df.empty:
                return
            
            df = self.calculate_indicators(df)
            if df.empty:
                return
            
            signal = self.generate_signal(df)
            if signal:
                await self.open_position(signal, df)
                        
        except Exception as e:
            error_msg = f"ETH 트레이딩 사이클 실행 중 오류: {e}"
            print_error(error_msg)
            await self.send_telegram_message(f"❌ {error_msg}")
    
    async def test_connection(self) -> bool:
        """거래소 연결 테스트"""
        try:
            ticker = self.exchange.fetch_ticker(self.config.symbol)
            if ticker and 'last' in ticker:
                print_info(f"ETH/USDT 연결 성공 - 현재가: ${ticker['last']:.2f}")
                return True
            else:
                print_error("티커 정보를 가져올 수 없습니다")
                return False
        except Exception as e:
            print_error(f"연결 테스트 실패: {e}")
            return False
    
    async def run(self):
        """메인 실행 루프"""
        print_info("ETH 자동매매 봇 시작")
        
        if not await self.test_connection():
            print_error("거래소 연결 실패")
            return
        
        if self.telegram_bot:
            await self.send_telegram_message("🤖 쿠코인 ETH 선물 자동매매 봇이 시작되었습니다!")
        
        self.is_running = True
        
        while True:
            try:
                if self.is_running:
                    await self.run_trading_cycle()
                
                await asyncio.sleep(self.config.check_interval)
                
            except KeyboardInterrupt:
                print_info("사용자에 의해 중단됨")
                break
            except Exception as e:
                error_msg = f"메인 루프 오류: {e}"
                print_error(error_msg)
                if self.telegram_bot:
                    await self.send_telegram_message(f"❌ 메인 루프 오류: {e}")
                await asyncio.sleep(60)
        
        if self.telegram_bot:
            await self.send_telegram_message("🛑 ETH 자동매매 봇이 중지되었습니다.")
        print_info("ETH 자동매매 봇 종료")

def test_kucoin_connection():
    """쿠코인 연결 테스트"""
    print("쿠코인 연결 테스트 중...")
    
    try:
        exchange_test = ccxt.kucoinfutures({'enableRateLimit': True})
        ticker = exchange_test.fetch_ticker('ETH/USDT:USDT')
        
        if ticker and 'last' in ticker:
            print(f"ETH/USDT 연결 성공! 현재가: ${ticker['last']:.2f}")
            return True
        else:
            print("ETH/USDT 티커를 가져올 수 없습니다.")
            return False
            
    except Exception as e:
        print(f"연결 테스트 실패: {e}")
        try:
            exchange_test = ccxt.kucoinfutures({'enableRateLimit': True})
            ticker = exchange_test.fetch_ticker('ETHUSDTM')
            if ticker:
                print(f"ETHUSDTM 연결 성공! 현재가: ${ticker['last']:.2f}")
                return True
        except:
            pass
        return False

def main():
    """메인 함수"""
    print("쿠코인 ETH 선물 자동매매 봇 시작")
    print("=" * 50)
    
    if not test_kucoin_connection():
        print("\n기본 연결 테스트에 실패했습니다.")
        print("인터넷 연결과 방화벽 설정을 확인하세요.")
        sys.exit(1)
    
    if KUCOIN_API_KEY in ['', 'your_api_key'] or \
       KUCOIN_SECRET in ['', 'your_secret'] or \
       KUCOIN_PASSPHRASE in ['', 'your_passphrase']:
        print("쿠코인 API 키가 설정되지 않았습니다.")
        print("환경변수 또는 .env 파일에서 다음을 설정하세요:")
        print("- KUCOIN_API_KEY")
        print("- KUCOIN_SECRET") 
        print("- KUCOIN_PASSPHRASE")
        sys.exit(1)
    
    if TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN not in ['', 'your_telegram_bot_token']:
        print("텔레그램 봇 설정됨")
    else:
        print("텔레그램 미설정 - 콘솔 모드로 실행")
    
    if TALIB_AVAILABLE:
        print("TA-Lib 사용 가능")
    else:
        print("TA-Lib 미설치 - 기본 지표 사용")
    
    config = TradingConfig()
    print(f"ETH 거래 설정:")
    print(f"- 심볼: {config.symbol}")
    print(f"- 시드머니: ${config.seed_money}")
    print(f"- 리스크: {config.risk_per_trade*100:.1f}%")
    print(f"- 레버리지: {config.leverage}x")
    print(f"- 시간프레임: {config.timeframe}")
    print(f"- EMA: {config.ema_fast}/{config.ema_slow}")
    print(f"- RSI: {config.rsi_period}기간, {config.rsi_oversold}-{config.rsi_overbought}")
    print("\n현재 시뮬레이션 모드로 실행됩니다.")
    print("실제 거래를 원한다면 코드의 주문 실행 부분 주석을 해제하세요.")
    
    try:
        bot = KucoinETHBot(config)
        print("ETH 봇 초기화 완료")
        print("Ctrl+C로 중지할 수 있습니다.\n")
        
        asyncio.run(bot.run())
        
    except Exception as e:
        print(f"\n봇 실행 오류: {e}")
        print(f"상세 오류: {traceback.format_exc()}")

if __name__ == "__main__":
    # Oracle 프리티어 환경 체크
    if os.path.exists('/etc/oracle-cloud-agent'):
        print("Oracle Cloud 환경에서 실행 중")
    
    # 필수 패키지 확인
    missing_packages = []
    required_packages = {
        'ccxt': 'ccxt',
        'pandas': 'pandas', 
        'numpy': 'numpy'
    }
    
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"필수 패키지가 설치되지 않았습니다: {missing_packages}")
        print(f"다음 명령어로 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)
    
    main()