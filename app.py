import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import copy
import logging
from typing import Optional, Dict, Any
import hashlib

# Suppress only specific warnings, not all
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="VWV Professional Trading System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        position: relative;
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        overflow: hidden;
        min-height: 200px;
        
        /* Great Wave inspired layered background */
        background: 
            /* Sky gradient */
            linear-gradient(to bottom, #1a4d3a 0%, #2d6b4f 40%, #1e5540 100%),
            /* Main wave shape using clip-path technique */
            radial-gradient(ellipse 120% 60% at 20% 100%, #0a2f1f 0%, #0a2f1f 25%, transparent 26%),
            radial-gradient(ellipse 150% 80% at 80% 95%, #134a35 0%, #134a35 30%, transparent 31%),
            radial-gradient(ellipse 200% 100% at 50% 90%, #0f3b28 0%, #0f3b28 20%, transparent 21%);
        
        /* Wave foam effect */
        background-image:
            /* Small foam dots */
            radial-gradient(circle at 15% 75%, rgba(255,255,255,0.4) 1px, transparent 2px),
            radial-gradient(circle at 25% 78%, rgba(255,255,255,0.5) 1.5px, transparent 2.5px),
            radial-gradient(circle at 35% 72%, rgba(255,255,255,0.3) 1px, transparent 2px),
            radial-gradient(circle at 45% 80%, rgba(255,255,255,0.4) 1.2px, transparent 2.2px),
            radial-gradient(circle at 55% 76%, rgba(255,255,255,0.3) 1px, transparent 2px),
            radial-gradient(circle at 65% 82%, rgba(255,255,255,0.5) 1.3px, transparent 2.3px),
            radial-gradient(circle at 75% 74%, rgba(255,255,255,0.4) 1px, transparent 2px),
            radial-gradient(circle at 85% 79%, rgba(255,255,255,0.3) 1.4px, transparent 2.4px),
            /* Wave crest highlights */
            linear-gradient(135deg, transparent 60%, rgba(255,255,255,0.1) 65%, transparent 70%),
            linear-gradient(45deg, transparent 40%, rgba(255,255,255,0.05) 45%, transparent 50%);
        
        background-size:
            100% 100%,  /* Sky */
            100% 100%,  /* Wave shapes */
            100% 100%,
            100% 100%,
            30px 30px,  /* Foam dots */
            35px 35px,
            40px 40px,
            25px 25px,
            45px 45px,
            50px 50px,
            30px 30px,
            35px 35px,
            100% 100%,  /* Wave highlights */
            100% 100%;
    }
    
    .header-content {
        position: relative;
        z-index: 3;
        background: rgba(0,0,0,0.2);
        padding: 1.5rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .header-content h1 {
        font-size: 2.8rem;
        margin-bottom: 1rem;
        color: #ffffff;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .header-content p {
        color: #f0f8f0;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.7);
        margin: 0.5rem 0;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .signal-good { background-color: #d4edda; border-left: 5px solid #28a745; }
    .signal-strong { background-color: #fff3cd; border-left: 5px solid #ffc107; }
    .signal-very-strong { background-color: #f8d7da; border-left: 5px solid #dc3545; }
    .signal-none { background-color: #e2e3e5; border-left: 5px solid #6c757d; }
    .signal-bearish { background-color: #f8d7da; border-left: 5px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

class DataQualityChecker:
    """Comprehensive data quality validation"""
    
    @staticmethod
    def validate_market_data(data: pd.DataFrame) -> Dict[str, Any]:
        """Validate market data quality and completeness"""
        issues = []
        quality_score = 100
        
        # Check for missing data
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
        if missing_pct > 5:
            issues.append(f"High missing data: {missing_pct:.1f}%")
            quality_score -= 20
            
        # Check for price anomalies
        returns = data['Close'].pct_change().dropna()
        if len(returns) > 0:
            extreme_returns = (abs(returns) > 0.2).sum()  # >20% moves
            if extreme_returns > len(returns) * 0.02:  # More than 2% of data
                issues.append(f"Excessive extreme returns: {extreme_returns}")
                quality_score -= 15
        
        # Check volume consistency
        if 'Volume' in data.columns:
            zero_volume_days = (data['Volume'] == 0).sum()
            if zero_volume_days > len(data) * 0.05:  # More than 5%
                issues.append(f"High zero-volume days: {zero_volume_days}")
                quality_score -= 10
                
        # Check price consistency (High >= Low, etc.)
        price_inconsistencies = ((data['High'] < data['Low']) | 
                                 (data['Close'] > data['High']) | 
                                 (data['Close'] < data['Low'])).sum()
        if price_inconsistencies > 0:
            issues.append(f"Price inconsistencies: {price_inconsistencies}")
            quality_score -= 25
            
        return {
            'quality_score': max(0, quality_score),
            'issues': issues,
            'data_points': len(data),
            'date_range': (data.index[0], data.index[-1]) if len(data) > 0 else None,
            'is_acceptable': quality_score >= 70
        }

def safe_calculation_wrapper(func):
    """Decorator for safe financial calculations"""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if result is None:
                logger.warning(f"Function {func.__name__} returned None")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper

class DataManager:
    """Enhanced data manager with debug control"""

    def __init__(self):
        self._market_data_store = {}
        self._analysis_store = {}

    def store_market_data(self, symbol, market_data, show_debug=False):
        """Data storage with debug control"""
        if not isinstance(market_data, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(market_data)}")

        self._market_data_store[symbol] = market_data.copy(deep=True)
        if show_debug:
            st.write(f"🔒 Stored market data for {symbol}: {market_data.shape}")

    def get_market_data_for_analysis(self, symbol):
        """Get copy for analysis"""
        if symbol not in self._market_data_store:
            return None
        return self._market_data_store[symbol].copy(deep=True)

    def get_market_data_for_chart(self, symbol):
        """Get copy for chart"""
        if symbol not in self._market_data_store:
            return None

        chart_copy = self._market_data_store[symbol].copy(deep=True)

        if not isinstance(chart_copy, pd.DataFrame):
            st.error(f"🚨 Chart data corrupted: {type(chart_copy)}")
            return None

        return chart_copy

    def store_analysis_results(self, symbol, analysis_results):
        """Store analysis results"""
        self._analysis_store[symbol] = copy.deepcopy(analysis_results)

    def get_analysis_results(self, symbol):
        """Get analysis results"""
        return self._analysis_store.get(symbol, {})

# Initialize data manager
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()

@st.cache_data(ttl=300)  # 5-minute cache
def get_cached_market_data(symbol: str, period: str):
    """Cached market data retrieval"""
    return get_market_data_enhanced(symbol, period, show_debug=False)

def generate_cache_key(symbol: str, analysis_config: dict) -> str:
    """Generate unique cache key for analysis results"""
    config_str = str(sorted(analysis_config.items()))
    return hashlib.md5(f"{symbol}_{config_str}".encode()).hexdigest()

def is_etf(symbol):
    """Detect if a symbol is an ETF"""
    try:
        # Known individual stocks that should never be considered ETFs
        known_stocks = {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'JPM', 'JNJ', 'UNH', 'V', 'PG', 'HD', 'MA', 'BAC', 'ABBV', 'PFE',
            'KO', 'ADBE', 'PEP', 'TMO', 'COST', 'AVGO', 'NKE', 'MRK', 'ABT', 'CRM',
            'LLY', 'ACN', 'TXN', 'DHR', 'WMT', 'NEE', 'VZ', 'ORCL', 'CMCSA', 'PM',
            'DIS', 'BMY', 'RTX', 'HON', 'QCOM', 'UPS', 'T', 'AIG', 'LOW', 'MDT'
        }
        
        # If it's a known individual stock, it's definitely not an ETF
        if symbol.upper() in known_stocks:
            return False
        
        # Common ETF patterns and known ETFs
        common_etfs = {
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'AGG', 'BND', 'TLT',
            'GLD', 'SLV', 'USO', 'UNG', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP',
            'XLY', 'XLU', 'XLRE', 'XLB', 'EFA', 'EEM', 'FXI', 'EWJ', 'EWG', 'EWU',
            'ARKK', 'ARKQ', 'ARKW', 'ARKG', 'ARKF', 'FNGU', 'FNGD', 'MAGS', 'SOXX',
            'SMH', 'IBB', 'XBI', 'JETS', 'HACK', 'ESPO', 'ICLN', 'PBW', 'KWEB',
            'SPHB', 'SOXL', 'QQI', 'DIVO', 'URNM', 'GDX', 'FETH'
        }
        
        # Check if symbol is in known ETFs list
        if symbol.upper() in common_etfs:
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"ETF detection error for {symbol}: {e}")
        return False

def get_market_data_enhanced(symbol='SPY', period='1y', show_debug=False):
    """Enhanced market data fetching with debug control"""
    try:
        if show_debug:
            st.write(f"📡 Fetching data for {symbol}...")

        ticker = yf.Ticker(symbol)
        raw_data = ticker.history(period=period)

        if raw_data is None or len(raw_data) == 0:
            st.error(f"❌ No data returned for {symbol}")
            return None

        if show_debug:
            st.write(f"📊 Retrieved {len(raw_data)} rows")

        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = list(raw_data.columns)

        if show_debug:
            st.write(f"📋 Available columns: {available_columns}")

        missing_columns = [col for col in required_columns if col not in available_columns]
        if missing_columns:
            st.error(f"❌ Missing columns: {missing_columns}")
            return None

        # Clean data
        clean_data = raw_data[required_columns].copy()
        clean_data = clean_data.dropna()

        if len(clean_data) == 0:
            st.error(f"❌ No data after cleaning")
            return None

        # Add typical price
        clean_data['Typical_Price'] = (clean_data['High'] + clean_data['Low'] + clean_data['Close']) / 3

        # Data quality check
        quality_check = DataQualityChecker.validate_market_data(clean_data)
        
        if not quality_check['is_acceptable']:
            st.warning(f"⚠️ Data quality issues detected for {symbol}: {quality_check['issues']}")

        if show_debug:
            st.success(f"✅ Data ready: {clean_data.shape} | Quality Score: {quality_check['quality_score']}")
        else:
            st.success(f"✅ Data loaded: {len(clean_data)} periods | Quality: {quality_check['quality_score']}/100")

        return clean_data

    except Exception as e:
        st.error(f"❌ Error fetching {symbol}: {str(e)}")
        return None

@safe_calculation_wrapper
def safe_rsi(prices, period=14):
    """Safe RSI calculation with proper error handling"""
    try:
        if len(prices) < period + 1:
            return pd.Series([50] * len(prices), index=prices.index)
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)
        return rsi
    except Exception as e:
        logger.error(f"RSI calculation error: {e}")
        return pd.Series([50] * len(prices), index=prices.index)

@safe_calculation_wrapper
def calculate_daily_vwap(data):
    """Enhanced daily VWAP calculation"""
    try:
        if not hasattr(data, 'index') or not hasattr(data, 'columns'):
            return float(data['Close'].iloc[-1]) if 'Close' in data else 0.0

        if len(data) < 5:
            return float(data['Close'].iloc[-1])

        recent_data = data.tail(20)

        if 'Typical_Price' in recent_data.columns and 'Volume' in recent_data.columns:
            total_pv = (recent_data['Typical_Price'] * recent_data['Volume']).sum()
            total_volume = recent_data['Volume'].sum()

            if total_volume > 0:
                return float(total_pv / total_volume)

        return float(data['Close'].iloc[-1])
    except Exception:
        try:
            return float(data['Close'].iloc[-1])
        except:
            return 0.0

@safe_calculation_wrapper
def calculate_fibonacci_emas(data):
    """Calculate Fibonacci EMAs (21, 55, 89, 144, 233)"""
    try:
        if len(data) < 21:
            return {}

        close = data['Close']
        fib_periods = [21, 55, 89, 144, 233]
        emas = {}

        for period in fib_periods:
            if len(close) >= period:
                ema_value = close.ewm(span=period).mean().iloc[-1]
                emas[f'EMA_{period}'] = round(float(ema_value), 2)

        return emas
    except Exception:
        return {}

@safe_calculation_wrapper
def calculate_point_of_control_enhanced(data):
    """Enhanced Point of Control with better volume weighting"""
    try:
        if len(data) < 20:
            return None

        # Use recent data for daily POC
        recent_data = data.tail(20)

        # Create price bins and sum volume for each bin
        price_range = recent_data['High'].max() - recent_data['Low'].min()
        bin_size = price_range / 50  # 50 price bins

        if bin_size <= 0:
            return float(recent_data['Close'].iloc[-1])

        # Calculate volume profile with better weighting
        volume_profile = {}

        for idx, row in recent_data.iterrows():
            total_volume = row['Volume']
            open_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            
            # Determine if it's a bullish or bearish bar
            is_bullish = close_price >= open_price
            
            # Enhanced volume distribution weighting
            if is_bullish:
                # For bullish bars: more weight to close and high
                price_weights = {
                    open_price: 0.15,   # 15%
                    high_price: 0.30,   # 30%
                    low_price: 0.10,    # 10%
                    close_price: 0.45   # 45%
                }
            else:
                # For bearish bars: more weight to close and low
                price_weights = {
                    open_price: 0.15,   # 15%
                    high_price: 0.10,   # 10%
                    low_price: 0.30,    # 30%
                    close_price: 0.45   # 45%
                }
            
            # Distribute volume according to weights
            for price, weight in price_weights.items():
                bin_key = round(price / bin_size) * bin_size
                volume_profile[bin_key] = volume_profile.get(bin_key, 0) + (total_volume * weight)

        # Find POC (price with highest volume)
        if volume_profile:
            poc_price = max(volume_profile, key=volume_profile.get)
            return round(float(poc_price), 2)
        else:
            return float(recent_data['Close'].iloc[-1])

    except Exception as e:
        logger.error(f"Enhanced POC calculation error: {e}")
        try:
            return float(data['Close'].iloc[-1])
        except:
            return 0.0

@safe_calculation_wrapper
def calculate_comprehensive_technicals(data):
    """Calculate comprehensive technical indicators for individual symbol analysis"""
    try:
        if len(data) < 50:
            return {}

        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']

        # Previous week high/low
        week_data = data.tail(5)  # Last 5 trading days
        prev_week_high = week_data['High'].max()
        prev_week_low = week_data['Low'].min()

        # RSI (14-period)
        rsi_14 = safe_rsi(close, 14).iloc[-1]

        # Money Flow Index (MFI)
        mfi_14 = calculate_mfi(data, 14)

        # MACD (12, 26, 9)
        macd_data = calculate_macd(close, 12, 26, 9)

        # Average True Range (ATR)
        atr_14 = calculate_atr(data, 14)

        # Bollinger Bands (20, 2)
        bb_data = calculate_bollinger_bands(close, 20, 2)

        # Stochastic Oscillator
        stoch_data = calculate_stochastic(data, 14, 3)

        # Williams %R
        williams_r = calculate_williams_r(data, 14)

        # Volume metrics
        volume_sma_20 = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        current_volume = volume.iloc[-1]
        volume_ratio = (current_volume / volume_sma_20) if volume_sma_20 > 0 else 1

        # Price metrics
        current_price = close.iloc[-1]
        price_change_1d = ((current_price - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) > 1 else 0
        price_change_5d = ((current_price - close.iloc[-6]) / close.iloc[-6] * 100) if len(close) > 5 else 0

        # Volatility (20-day)
        returns = close.pct_change().dropna()
        if len(returns) >= 20:
            volatility_20d = returns.rolling(20).std().iloc[-1] * (252 ** 0.5) * 100  # Annualized
        else:
            volatility_20d = returns.std() * (252 ** 0.5) * 100 if len(returns) > 0 else 20

        return {
            'prev_week_high': round(float(prev_week_high), 2),
            'prev_week_low': round(float(prev_week_low), 2),
            'rsi_14': round(float(rsi_14), 2),
            'mfi_14': round(float(mfi_14), 2),
            'macd': macd_data,
            'atr_14': round(float(atr_14), 2),
            'bollinger_bands': bb_data,
            'stochastic': stoch_data,
            'williams_r': williams_r,
            'volume_sma_20': round(float(volume_sma_20), 0),
            'current_volume': round(float(current_volume), 0),
            'volume_ratio': round(float(volume_ratio), 2),
            'price_change_1d': round(float(price_change_1d), 2),
            'price_change_5d': round(float(price_change_5d), 2),
            'volatility_20d': round(float(volatility_20d), 2)
        }

    except Exception as e:
        logger.error(f"Comprehensive technicals calculation error: {e}")
        return {}

@safe_calculation_wrapper
def calculate_mfi(data, period=14):
    """Calculate Money Flow Index"""
    try:
        if len(data) < period + 1:
            return 50.0
            
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']

        # Positive and negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()

        mfi = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, np.inf)))
        return float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 50.0
    except Exception as e:
        logger.error(f"MFI calculation error: {e}")
        return 50.0

@safe_calculation_wrapper
def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    try:
        if len(close) < slow:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
            
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return {
            'macd': round(float(macd_line.iloc[-1]), 4),
            'signal': round(float(signal_line.iloc[-1]), 4),
            'histogram': round(float(histogram.iloc[-1]), 4)
        }
    except Exception as e:
        logger.error(f"MACD calculation error: {e}")
        return {'macd': 0, 'signal': 0, 'histogram': 0}

@safe_calculation_wrapper
def calculate_atr(data, period=14):
    """Calculate Average True Range"""
    try:
        if len(data) < period + 1:
            return 0.0
            
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift(1)).abs()
        low_close = (data['Low'] - data['Close'].shift(1)).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()

        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
    except Exception as e:
        logger.error(f"ATR calculation error: {e}")
        return 0.0

@safe_calculation_wrapper
def calculate_bollinger_bands(close, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    try:
        if len(close) < period:
            current_close = float(close.iloc[-1])
            return {
                'upper': current_close * 1.02,
                'middle': current_close,
                'lower': current_close * 0.98,
                'position': 50
            }
            
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        current_close = close.iloc[-1]
        
        upper_val = upper_band.iloc[-1]
        lower_val = lower_band.iloc[-1]
        
        if upper_val != lower_val:
            bb_position = ((current_close - lower_val) / (upper_val - lower_val)) * 100
        else:
            bb_position = 50

        return {
            'upper': round(float(upper_val), 2),
            'middle': round(float(sma.iloc[-1]), 2),
            'lower': round(float(lower_val), 2),
            'position': round(float(bb_position), 1)
        }
    except Exception as e:
        logger.error(f"Bollinger Bands calculation error: {e}")
        return {'upper': 0, 'middle': 0, 'lower': 0, 'position': 50}

@safe_calculation_wrapper
def calculate_stochastic(data, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    try:
        if len(data) < k_period:
            return {'k': 50, 'd': 50}
            
        lowest_low = data['Low'].rolling(k_period).min()
        highest_high = data['High'].rolling(k_period).max()

        k_percent = ((data['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
        d_percent = k_percent.rolling(d_period).mean()

        return {
            'k': round(float(k_percent.iloc[-1]), 2),
            'd': round(float(d_percent.iloc[-1]), 2)
        }
    except Exception as e:
        logger.error(f"Stochastic calculation error: {e}")
        return {'k': 50, 'd': 50}

@safe_calculation_wrapper
def calculate_williams_r(data, period=14):
    """Calculate Williams %R"""
    try:
        if len(data) < period:
            return -50.0
            
        highest_high = data['High'].rolling(period).max()
        lowest_low = data['Low'].rolling(period).min()

        williams_r = ((highest_high - data['Close']) / (highest_high - lowest_low)) * -100
        return float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else -50.0
    except Exception as e:
        logger.error(f"Williams %R calculation error: {e}")
        return -50.0

def statistical_normalize(series, lookback_period=252):
    """Simple statistical normalization"""
    try:
        if not hasattr(series, 'rolling') or len(series) < 10:
            if hasattr(series, '__iter__'):
                return 0.5
            else:
                return float(np.clip(series, 0, 1))

        if len(series) < lookback_period:
            lookback_period = len(series)

        percentile = series.rolling(window=lookback_period).rank(pct=True)
        result = percentile.iloc[-1] if not pd.isna(percentile.iloc[-1]) else 0.5
        return float(result)
    except Exception:
        return 0.5

# VWV Trading System with corrected Williams VIX Fix
class VWVTradingSystem:
    def __init__(self, config=None):
        """Initialize enhanced trading system"""
        default_config = {
            'wvf_period': 22,
            'wvf_multiplier': 2.0,  # Standard Bollinger Band multiplier
            'ma_periods': [20, 50, 200],
            'volume_periods': [20, 50],
            'rsi_period': 14,
            'volatility_period': 20,
            'weights': {
                'wvf': 0.8, 'ma': 1.2, 'volume': 0.6,
                'vwap': 0.4, 'momentum': 0.5, 'volatility': 0.3
            },
            'scaling_multiplier': 1.5,
            'signal_thresholds': {'good': 3.5, 'strong': 4.5, 'very_strong': 5.5},
            'stop_loss_pct': 0.022,
            'take_profit_pct': 0.055
        }

        self.config = {**default_config, **(config or {})}
        self.weights = self.config['weights']
        self.scaling_multiplier = self.config['scaling_multiplier']
        self.signal_thresholds = self.config['signal_thresholds']
        self.stop_loss_pct = self.config['stop_loss_pct']
        self.take_profit_pct = self.config['take_profit_pct']

    @safe_calculation_wrapper
    def detect_market_regime(self, data):
        """Detect market regime for dynamic weight adjustment"""
        try:
            if len(data) < 50:
                return {'regime': 'NORMAL', 'volatility_regime': 'NORMAL', 'trend_regime': 'SIDEWAYS'}

            close = data['Close']
            
            # Calculate volatility regime (20-day rolling volatility)
            returns = close.pct_change().dropna()
            current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)  # Annualized
            vol_history = returns.rolling(20).std() * np.sqrt(252)
            vol_percentile = vol_history.rolling(252).rank(pct=True).iloc[-1] if len(vol_history) >= 252 else 0.5
            
            if vol_percentile > 0.8:
                volatility_regime = 'HIGH'
            elif vol_percentile < 0.2:
                volatility_regime = 'LOW'
            else:
                volatility_regime = 'NORMAL'
            
            # Calculate trend regime (EMA slopes and alignment)
            ema_20 = close.ewm(span=20).mean()
            ema_50 = close.ewm(span=50).mean()
            
            # EMA slope strength
            ema20_slope = (ema_20.iloc[-1] - ema_20.iloc[-5]) / ema_20.iloc[-5] * 100
            ema50_slope = (ema_50.iloc[-1] - ema_50.iloc[-10]) / ema_50.iloc[-10] * 100
            
            # EMA alignment
            ema_aligned = ema_20.iloc[-1] > ema_50.iloc[-1]
            
            if abs(ema20_slope) > 2 and abs(ema50_slope) > 1 and ema_aligned:
                if ema20_slope > 0:
                    trend_regime = 'STRONG_UPTREND'
                else:
                    trend_regime = 'STRONG_DOWNTREND'
            elif abs(ema20_slope) > 1:
                if ema20_slope > 0:
                    trend_regime = 'UPTREND'
                else:
                    trend_regime = 'DOWNTREND'
            else:
                trend_regime = 'SIDEWAYS'
            
            # Overall regime classification
            if volatility_regime == 'HIGH' and trend_regime in ['STRONG_UPTREND', 'STRONG_DOWNTREND']:
                overall_regime = 'TRENDING_VOLATILE'
            elif volatility_regime == 'HIGH':
                overall_regime = 'HIGH_VOLATILITY'
            elif trend_regime in ['STRONG_UPTREND', 'STRONG_DOWNTREND']:
                overall_regime = 'TRENDING'
            elif volatility_regime == 'LOW' and trend_regime == 'SIDEWAYS':
                overall_regime = 'LOW_VOLATILITY'
            else:
                overall_regime = 'NORMAL'
            
            return {
                'regime': overall_regime,
                'volatility_regime': volatility_regime,
                'trend_regime': trend_regime,
                'volatility_percentile': vol_percentile,
                'trend_strength': abs(ema20_slope)
            }
            
        except Exception as e:
            logger.error(f"Market regime detection error: {e}")
            return {'regime': 'NORMAL', 'volatility_regime': 'NORMAL', 'trend_regime': 'SIDEWAYS'}

    def get_dynamic_weights(self, market_regime):
        """Get dynamic weights based on market regime"""
        base_weights = self.weights.copy()
        
        regime = market_regime['regime']
        vol_regime = market_regime['volatility_regime']
        trend_regime = market_regime['trend_regime']
        
        # Adjust weights based on regime
        if regime == 'HIGH_VOLATILITY' or vol_regime == 'HIGH':
            # In high volatility, WVF becomes more important
            base_weights['wvf'] *= 1.3
            base_weights['volatility'] *= 1.2
            base_weights['ma'] *= 0.9
            
        elif regime == 'TRENDING' or trend_regime in ['STRONG_UPTREND', 'STRONG_DOWNTREND']:
            # In trending markets, MA confluence becomes more important
            base_weights['ma'] *= 1.4
            base_weights['momentum'] *= 1.2
            base_weights['wvf'] *= 0.8
            
        elif regime == 'LOW_VOLATILITY':
            # In low volatility, volume and VWAP become more important
            base_weights['volume'] *= 1.3
            base_weights['vwap'] *= 1.2
            base_weights['volatility'] *= 0.7
            
        elif regime == 'TRENDING_VOLATILE':
            # In trending volatile markets, balance trend and volatility indicators
            base_weights['wvf'] *= 1.2
            base_weights['ma'] *= 1.3
            base_weights['momentum'] *= 1.1
        
        return base_weights

    def calculate_williams_vix_fix_enhanced(self, data):
        """Enhanced Williams VIX Fix with proper binary signal logic"""
        try:
            period = self.config['wvf_period']  # Default 22
            multiplier = self.config['wvf_multiplier']  # Default 2.0
            
            if len(data) < period * 2:
                return {'binary_signal': 0, 'normalized_strength': 0.0, 'wvf_value': 0, 'upper_band': 0}

            close = data['Close']
            low = data['Low']

            # Original WVF formula: ((Highest Close - Low) / Highest Close) × 100
            highest_close = close.rolling(window=period).max()
            wvf_raw = ((highest_close - low) / highest_close) * 100

            # Apply Bollinger Band to WVF for signals
            wvf_sma = wvf_raw.rolling(window=period).mean()
            wvf_std = wvf_raw.rolling(window=period).std()
            wvf_upper_band = wvf_sma + (wvf_std * multiplier)

            # Get current values
            current_wvf = wvf_raw.iloc[-1] if not pd.isna(wvf_raw.iloc[-1]) else 0
            current_upper = wvf_upper_band.iloc[-1] if not pd.isna(wvf_upper_band.iloc[-1]) else 0

            # Binary signal (true to WVF design)
            binary_signal = 1 if current_wvf > current_upper else 0
            
            # Additional strength measure for confluence (how far above/below)
            if current_upper > 0:
                strength_ratio = (current_wvf - current_upper) / current_upper
                normalized_strength = float(np.clip(strength_ratio, -1, 1))  # Cap at +/-100%
            else:
                normalized_strength = 0.0

            return {
                'binary_signal': binary_signal,
                'normalized_strength': normalized_strength,
                'wvf_value': float(current_wvf),
                'upper_band': float(current_upper)
            }

        except Exception as e:
            logger.error(f"Enhanced Williams VIX Fix calculation error: {e}")
            return {'binary_signal': 0, 'normalized_strength': 0.0, 'wvf_value': 0, 'upper_band': 0}

@safe_calculation_wrapper
    def calculate_ma_confluence(self, data):
        """Moving average confluence"""
        try:
            ma_periods = self.config['ma_periods']
            if len(data) < max(ma_periods):
                return 0.0

            close = data['Close']
            current_price = close.iloc[-1]

            mas = []
            for period in ma_periods:
                if len(close) >= period:
                    ma = close.rolling(window=period).mean().iloc[-1]
                    mas.append(ma)

            if not mas:
                return 0.0

            ma_avg = np.mean(mas)
            deviation_pct = abs((current_price - ma_avg) / ma_avg * 100) if ma_avg > 0 else 0

            # Create deviation series for normalization
            deviation_series = []
            for i in range(min(252, len(close))):
                if i + max(ma_periods) < len(close):
                    subset_close = close.iloc[i:i+max(ma_periods)]
                    subset_mas = []
                    for period in ma_periods:
                        if len(subset_close) >= period:
                            subset_ma = subset_close.rolling(window=period).mean().iloc[-1]
                            subset_mas.append(subset_ma)
                    if subset_mas:
                        subset_avg = np.mean(subset_mas)
                        subset_price = subset_close.iloc[-1]
                        subset_deviation = abs((subset_price - subset_avg) / subset_avg * 100) if subset_avg > 0 else 0
                        deviation_series.append(subset_deviation)

            if deviation_series:
                deviation_df = pd.Series(deviation_series + [deviation_pct])
                return statistical_normalize(deviation_df)
            else:
                return 0.5

        except Exception as e:
            logger.error(f"MA confluence calculation error: {e}")
            return 0.0

    @safe_calculation_wrapper
    def calculate_volume_confluence(self, data):
        """Volume analysis"""
        try:
            periods = self.config['volume_periods']
            if len(data) < max(periods):
                return 0.0

            volume = data['Volume']
            current_vol = volume.iloc[-1]

            vol_mas = []
            for period in periods:
                if len(volume) >= period:
                    vol_ma = volume.rolling(window=period).mean().iloc[-1]
                    vol_mas.append(vol_ma)

            if not vol_mas:
                return 0.0

            avg_vol = np.mean(vol_mas)
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1

            # Create volume ratio series for normalization
            if len(volume) >= periods[0]:
                vol_ratios = volume.tail(252) / volume.rolling(window=periods[0]).mean()
                vol_ratios = vol_ratios.dropna()
                if len(vol_ratios) > 0:
                    return statistical_normalize(vol_ratios)

            return 0.5
        except Exception as e:
            logger.error(f"Volume confluence calculation error: {e}")
            return 0.0

    @safe_calculation_wrapper
    def calculate_vwap_analysis(self, data):
        """VWAP analysis"""
        try:
            if len(data) < 20:
                return 0.0

            current_price = data['Close'].iloc[-1]
            current_vwap = calculate_daily_vwap(data)

            if current_vwap == 0:
                return 0.0

            vwap_deviation_pct = abs(current_price - current_vwap) / current_vwap * 100

            # Create VWAP deviation series for normalization
            vwap_deviations = []
            for i in range(min(252, len(data) - 20)):
                end_idx = len(data) - i
                start_idx = max(0, end_idx - 20)
                subset = data.iloc[start_idx:end_idx]
                if len(subset) >= 5:
                    daily_vwap = calculate_daily_vwap(subset)
                    price = subset['Close'].iloc[-1]
                    if daily_vwap > 0:
                        deviation = abs(price - daily_vwap) / daily_vwap * 100
                        vwap_deviations.append(deviation)

            if vwap_deviations:
                deviation_series = pd.Series(vwap_deviations + [vwap_deviation_pct])
                return statistical_normalize(deviation_series)
            else:
                return 0.5
        except Exception as e:
            logger.error(f"VWAP analysis calculation error: {e}")
            return 0.0

    @safe_calculation_wrapper
    def calculate_momentum(self, data):
        """Momentum calculation"""
        try:
            period = self.config['rsi_period']
            if len(data) < period + 1:
                return 0.0

            close = data['Close']
            rsi = safe_rsi(close, period)
            rsi_value = rsi.iloc[-1]

            # Convert RSI to oversold signal strength (higher values for more oversold)
            oversold_signal = max(0, (50 - rsi_value) / 50) if rsi_value < 50 else 0
            return float(np.clip(oversold_signal, 0, 1))
        except Exception as e:
            logger.error(f"Momentum calculation error: {e}")
            return 0.0

    @safe_calculation_wrapper
    def calculate_volatility_filter(self, data):
        """Volatility filter"""
        try:
            period = self.config['volatility_period']
            if len(data) < period:
                return 0.0

            close = data['Close']
            returns = close.pct_change().dropna()
            
            if len(returns) < period:
                return 0.5
                
            volatility = returns.rolling(window=period).std() * np.sqrt(252)
            
            if len(volatility) == 0:
                return 0.5

            return statistical_normalize(volatility)
        except Exception as e:
            logger.error(f"Volatility filter calculation error: {e}")
            return 0.0

    def calculate_confluence(self, input_data, symbol='SPY', show_debug=False):
        """Enhanced confluence calculation with comprehensive analysis"""
        try:
            if not isinstance(input_data, pd.DataFrame):
                raise ValueError(f"Expected DataFrame, got {type(input_data)}")

            # Work on isolated copy
            working_data = input_data.copy(deep=True)

            # Calculate enhanced technical indicators
            daily_vwap = calculate_daily_vwap(working_data)
            fibonacci_emas = calculate_fibonacci_emas(working_data)
            point_of_control = calculate_point_of_control_enhanced(working_data)
            comprehensive_technicals = calculate_comprehensive_technicals(working_data)

            # Calculate VWV components with corrected WVF
            wvf_result = self.calculate_williams_vix_fix_enhanced(working_data)
            
            # Get market regime
            market_regime = self.detect_market_regime(working_data)
            dynamic_weights = self.get_dynamic_weights(market_regime)
            
            # Ensure all components return numeric values
            components = {
                'wvf': wvf_result.get('binary_signal', 0) if isinstance(wvf_result, dict) else 0,
                'ma': self.calculate_ma_confluence(working_data) or 0,
                'volume': self.calculate_volume_confluence(working_data) or 0,
                'vwap': self.calculate_vwap_analysis(working_data) or 0,
                'momentum': self.calculate_momentum(working_data) or 0,
                'volatility': self.calculate_volatility_filter(working_data) or 0
            }
            
            # Ensure all components are numeric
            for comp_name, comp_value in components.items():
                if not isinstance(comp_value, (int, float)):
                    components[comp_name] = 0
                    logger.warning(f"Component {comp_name} returned non-numeric value, defaulting to 0")

            # Calculate confluence with dynamic weights
            raw_confluence = sum(components[comp] * dynamic_weights[comp] for comp in components)
            base_confluence = raw_confluence * self.scaling_multiplier

            # Signal determination
            abs_confluence = abs(base_confluence)
            signal_direction = 'LONG' if base_confluence >= 0 else 'SHORT'

            signal_type = 'NONE'
            signal_strength = 0
            if abs_confluence >= self.signal_thresholds['very_strong']:
                signal_type, signal_strength = 'VERY_STRONG', 3
            elif abs_confluence >= self.signal_thresholds['strong']:
                signal_type, signal_strength = 'STRONG', 2
            elif abs_confluence >= self.signal_thresholds['good']:
                signal_type, signal_strength = 'GOOD', 1

            if signal_type != 'NONE':
                signal_type = f"{signal_type}_{signal_direction}"

            current_price = round(float(working_data['Close'].iloc[-1]), 2)
            current_date = working_data.index[-1].strftime('%Y-%m-%d')

            # Entry information
            entry_info = {}
            if signal_type != 'NONE':
                if signal_direction == 'LONG':
                    stop_price = round(current_price * (1 - self.stop_loss_pct), 2)
                    target_price = round(current_price * (1 + self.take_profit_pct), 2)
                else:
                    stop_price = round(current_price * (1 + self.stop_loss_pct), 2)
                    target_price = round(current_price * (1 - self.take_profit_pct), 2)

                entry_info = {
                    'entry_price': current_price,
                    'stop_loss': stop_price,
                    'take_profit': target_price,
                    'direction': signal_direction,
                    'position_multiplier': signal_strength,
                    'risk_reward': round(self.take_profit_pct / self.stop_loss_pct, 2)
                }

            return {
                'symbol': symbol,
                'timestamp': current_date,
                'current_price': current_price,
                'components': {k: round(v, 3) for k, v in components.items()},
                'raw_confluence': round(raw_confluence, 3),
                'base_confluence': round(base_confluence, 3),
                'directional_confluence': round(base_confluence, 3),
                'signal_type': signal_type,
                'signal_direction': signal_direction,
                'signal_strength': signal_strength,
                'entry_info': entry_info,
                'enhanced_indicators': {
                    'daily_vwap': daily_vwap,
                    'fibonacci_emas': fibonacci_emas,
                    'point_of_control': point_of_control,
                    'comprehensive_technicals': comprehensive_technicals,
                    'wvf_details': wvf_result,
                    'market_regime': market_regime,
                    'dynamic_weights': dynamic_weights
                },
                'system_status': 'OPERATIONAL'
            }

        except Exception as e:
            logger.error(f"Confluence calculation error for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'system_status': 'ERROR'
            }

def create_enhanced_chart(chart_market_data, analysis_results, symbol):
    """Enhanced chart creation with Fibonacci EMAs and technical levels"""

    if isinstance(chart_market_data, dict):
        st.error(f"❌ CHART RECEIVED DICT INSTEAD OF DATAFRAME!")
        return None

    if not isinstance(chart_market_data, pd.DataFrame):
        st.error(f"❌ Invalid chart data type: {type(chart_market_data)}")
        return None

    if len(chart_market_data) == 0:
        st.error(f"❌ Chart DataFrame is empty")
        return None

    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in chart_market_data.columns]
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        return None

    try:
        chart_data = chart_market_data.tail(100)
        current_price = analysis_results['current_price']
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{symbol} Price Chart with Technical Levels', 'Volume'),
            vertical_spacing=0.08, row_heights=[0.7, 0.3]
        )

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=chart_data.index, open=chart_data['Open'], high=chart_data['High'],
            low=chart_data['Low'], close=chart_data['Close'], name='Price'
        ), row=1, col=1)

        # Fibonacci EMAs
        fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
        ema_colors = {'EMA_21': 'orange', 'EMA_55': 'blue', 'EMA_89': 'purple', 'EMA_144': 'red', 'EMA_233': 'brown'}

        for ema_name, ema_value in fibonacci_emas.items():
            period = int(ema_name.split('_')[1])
            if len(chart_data) >= period:
                ema_series = chart_data['Close'].ewm(span=period).mean()
                color = ema_colors.get(ema_name, 'gray')
                fig.add_trace(go.Scatter(
                    x=chart_data.index, y=ema_series, name=ema_name,
                    line=dict(color=color, width=1.5)
                ), row=1, col=1)

        # Daily VWAP
        daily_vwap = enhanced_indicators.get('daily_vwap')
        if daily_vwap:
            fig.add_hline(y=daily_vwap, line_dash="solid",
                          line_color="cyan", line_width=2, row=1, col=1)

        # Point of Control
        poc = enhanced_indicators.get('point_of_control')
        if poc and poc > 0:
            fig.add_hline(y=poc, line_dash="dot",
                          line_color="magenta", line_width=2, row=1, col=1)

        # Current price line
        fig.add_hline(y=current_price, line_dash="solid", line_color="black",
                      line_width=3, row=1, col=1)

        # Volume chart
        colors = ['green' if close >= open else 'red'
                  for close, open in zip(chart_data['Close'], chart_data['Open'])]
        fig.add_trace(go.Bar(x=chart_data.index, y=chart_data['Volume'],
                             name='Volume', marker_color=colors), row=2, col=1)

        fig.update_layout(
            title=f'{symbol} | Confluence: {analysis_results["directional_confluence"]:.2f} | Signal: {analysis_results["signal_type"]}',
            height=600, showlegend=True, template='plotly_white'
        )

        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        return fig

    except Exception as e:
        st.error(f"❌ Error creating chart: {str(e)}")
        return None

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="header-content">
            <h1>VWV Professional Trading System</h1>
            <p>Advanced market analysis with enhanced technical indicators</p>
            <p><em>Features: Daily VWAP, Fibonacci EMAs, Point of Control, VIX Fix</em></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for recently viewed and watchlist
    if 'recently_viewed' not in st.session_state:
        st.session_state.recently_viewed = []
    if 'custom_watchlist' not in st.session_state:
        st.session_state.custom_watchlist = []

    # Sidebar controls
    st.sidebar.title("📊 Trading Analysis")
    
    # Basic controls
    if 'selected_symbol' in st.session_state:
        default_symbol = st.session_state.selected_symbol
        del st.session_state.selected_symbol
    else:
        default_symbol = "SPY"
        
    symbol = st.sidebar.text_input("Symbol", value=default_symbol, help="Enter stock symbol").upper()
    period = st.sidebar.selectbox("Data Period", ['1mo', '3mo', '6mo', '1y', '2y'], index=3)
    
    # Main analyze button
    analyze_button = st.sidebar.button("📊 Analyze Symbol", type="primary", use_container_width=True)

    # Recently Viewed section
    if len(st.session_state.recently_viewed) > 0:
        with st.sidebar.expander("🕒 Recently Viewed", expanded=False):
            st.write("**Last 6 Analyzed Symbols**")
            
            recent_symbols = st.session_state.recently_viewed[:6]
            
            for i, recent_symbol in enumerate(recent_symbols):
                if st.button(f"{recent_symbol}", key=f"recent_{recent_symbol}_{i}", use_container_width=True):
                    st.session_state.selected_symbol = recent_symbol
                    st.session_state.auto_analyze = True
                    st.rerun()

    # Function to add symbol to recently viewed
    def add_to_recently_viewed(symbol):
        if symbol and symbol != "":
            if symbol in st.session_state.recently_viewed:
                st.session_state.recently_viewed.remove(symbol)
            st.session_state.recently_viewed.insert(0, symbol)
            st.session_state.recently_viewed = st.session_state.recently_viewed[:6]

    # Quick Links section
    with st.sidebar.expander("🔗 Quick Links"):
        st.write("**Popular Symbols**")
        
        quick_symbols = ['TSLA', 'AAPL', 'NVDA', 'QQQ', 'SPY', 'MAGS']
        
        cols = st.columns(2)
        for i, sym in enumerate(quick_symbols):
            col = cols[i % 2]
            with col:
                if st.button(sym, key=f"quick_{sym}", use_container_width=True):
                    st.session_state.selected_symbol = sym
                    st.session_state.auto_analyze = True
                    st.rerun()

    # System parameters
    with st.sidebar.expander("⚙️ System Parameters"):
        wvf_period = st.slider("WVF Period", 10, 50, 22)
        wvf_multiplier = st.slider("WVF Multiplier", 0.5, 3.0, 2.0, step=0.1)
        good_threshold = st.slider("Good Signal", 2.0, 5.0, 3.5, step=0.1)
        strong_threshold = st.slider("Strong Signal", 3.0, 6.0, 4.5, step=0.1)
        very_strong_threshold = st.slider("Very Strong Signal", 4.0, 7.0, 5.5, step=0.1)
    
    # Debug toggle
    show_debug = st.sidebar.checkbox("🐛 Show Debug Info", value=False)

    # Check if auto-analyze was triggered
    auto_analyze = st.session_state.get('auto_analyze', False)
    if auto_analyze:
        st.session_state.auto_analyze = False
        analyze_button = True

    # Create configuration
    custom_config = {
        'wvf_period': wvf_period,
        'wvf_multiplier': wvf_multiplier,
        'signal_thresholds': {
            'good': good_threshold,
            'strong': strong_threshold,
            'very_strong': very_strong_threshold
        }
    }

    # Initialize system
    vwv_system = VWVTradingSystem(custom_config)
    data_manager = st.session_state.data_manager

    # Main logic flow
    if analyze_button and symbol:
        add_to_recently_viewed(symbol)
        
        st.write("## 📊 VWV Trading Analysis")
        
        with st.spinner(f"Analyzing {symbol}..."):
            
            if show_debug:
                st.write("### Step 1: Data Fetching")
            
            market_data = get_market_data_enhanced(symbol, period, show_debug)
            
            if market_data is None:
                st.error(f"❌ Could not fetch data for {symbol}")
                return
            
            data_manager.store_market_data(symbol, market_data, show_debug)
            
            if show_debug:
                st.write("### Step 2: Analysis Processing")
            
            analysis_input = data_manager.get_market_data_for_analysis(symbol)
            
            if analysis_input is None:
                st.error("❌ Could not prepare analysis data")
                return
            
            analysis_results = vwv_system.calculate_confluence(analysis_input, symbol, show_debug)
            
            if 'error' in analysis_results:
                st.error(f"❌ Analysis failed: {analysis_results['error']}")
                return
            
            data_manager.store_analysis_results(symbol, analysis_results)
            
            # Display results
            enhanced_indicators = analysis_results.get('enhanced_indicators', {})
            
            # Technical Analysis Section
            with st.expander(f"📊 {symbol} - Technical Analysis", expanded=True):
                current_price = analysis_results['current_price']
                daily_vwap = enhanced_indicators.get('daily_vwap', 0)
                fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${current_price}")
                with col2:
                    vwap_diff = current_price - daily_vwap if daily_vwap > 0 else 0
                    st.metric("Daily VWAP", f"${daily_vwap:.2f}", f"{vwap_diff:+.2f}")
                with col3:
                    ema_21 = fibonacci_emas.get('EMA_21', 0)
                    ema_diff = current_price - ema_21 if ema_21 > 0 else 0
                    st.metric("EMA 21", f"${ema_21:.2f}", f"{ema_diff:+.2f}")
                with col4:
                    signal_type = analysis_results['signal_type']
                    signal_strength = analysis_results['signal_strength']
                    st.metric("VWV Signal", signal_type.replace('_', ' '), f"Strength: {signal_strength}/3")

                # VWV Signal Analysis
                st.subheader("🎯 VWV Trading Signal")
                
                if signal_type != 'NONE':
                    entry_info = analysis_results['entry_info']
                    direction = entry_info['direction']
                    
                    st.success(f"""
                    🚨 **VWV {direction} SIGNAL DETECTED**
                    
                    **Signal Strength:** {signal_type}  
                    **Entry Price:** ${entry_info['entry_price']}  
                    **Stop Loss:** ${entry_info['stop_loss']}  
                    **Take Profit:** ${entry_info['take_profit']}  
                    **Risk/Reward:** {entry_info['risk_reward']}:1  
                    **Confluence:** {analysis_results['directional_confluence']:.2f}
                    """)
                else:
                    st.info("⚪ **No VWV Signal** - Market conditions do not meet signal criteria")

            # Chart Section
            with st.expander("📈 Technical Analysis Chart", expanded=True):
                chart_market_data = data_manager.get_market_data_for_chart(symbol)
                
                if chart_market_data is None:
                    st.error("❌ Could not get chart data")
                    return
                
                chart = create_enhanced_chart(chart_market_data, analysis_results, symbol)
                
                if chart is not None:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.error("❌ Chart creation failed")

            # Debug information
            if show_debug:
                with st.expander("🐛 Debug Information", expanded=False):
                    st.write("### Analysis Results")
                    st.json(analysis_results)
                    
                    st.write("### Component Details")
                    for component, value in analysis_results['components'].items():
                        st.write(f"**{component.upper()}:** {value:.4f}")

    else:
        st.write("## 🚀 VWV Professional Trading System")
        st.write("Welcome to the enhanced VWV Trading System with advanced technical analysis capabilities.")
        
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.write("### 📊 **Getting Started**")
            st.write("1. **Enter a symbol** in the sidebar (e.g., AAPL, SPY, QQQ)")
            st.write("2. **Select time period** for analysis")
            st.write("3. **Click 'Analyze Symbol'** to run complete analysis")
            st.write("4. **Use Quick Links** for popular symbols")
            
            st.write("### 🎯 **Key Features**")
            st.write("• **Enhanced Williams VIX Fix** - Bottom detection")
            st.write("• **Fibonacci EMAs** - 21, 55, 89, 144, 233 periods")
            st.write("• **Daily VWAP** - Volume-weighted average price")
            st.write("• **Point of Control** - High-volume price levels")
            st.write("• **Dynamic Regime Detection** - Adapts to market conditions")

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("### 📊 System Information")
        st.write(f"**Version:** VWV Professional v3.0")
        st.write(f"**Status:** ✅ Operational")
    
    with col2:
        st.write("### 🎯 Signal Types")
        st.write("**🟢 GOOD** - Moderate confluence")
        st.write("**🟡 STRONG** - High confluence")
        st.write("**🔴 VERY STRONG** - Maximum confluence")
    
    with col3:
        st.write("### ⚠️ Risk Disclaimer")
        st.write("**Educational Purpose Only**")
        st.write("• Not financial advice")
        st.write("• Manage risk appropriately")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.write("Please refresh the page and try again.")
        
        if st.checkbox("Show Error Details"):
            st.exception(e)

