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
        
        if data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100 > 5:
            issues.append(f"High missing data: {data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100:.1f}%")
            quality_score -= 20
            
        returns = data['Close'].pct_change().dropna()
        if len(returns) > 0:
            extreme_returns = (abs(returns) > 0.2).sum()
            if extreme_returns > len(returns) * 0.02:
                issues.append(f"Excessive extreme returns: {extreme_returns}")
                quality_score -= 15
        
        if 'Volume' in data.columns:
            zero_volume_days = (data['Volume'] == 0).sum()
            if zero_volume_days > len(data) * 0.05:
                issues.append(f"High zero-volume days: {zero_volume_days}")
                quality_score -= 10
                
        price_inconsistencies = ((data['High'] < data['Low']) | (data['Close'] > data['High']) | (data['Close'] < data['Low'])).sum()
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
        if not isinstance(market_data, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(market_data)}")
        self._market_data_store[symbol] = market_data.copy(deep=True)
        if show_debug:
            st.write(f"🔒 Stored market data for {symbol}: {market_data.shape}")

    def get_market_data_for_analysis(self, symbol):
        if symbol not in self._market_data_store:
            return None
        return self._market_data_store[symbol].copy(deep=True)

    def get_market_data_for_chart(self, symbol):
        if symbol not in self._market_data_store:
            return None
        chart_copy = self._market_data_store[symbol].copy(deep=True)
        if not isinstance(chart_copy, pd.DataFrame):
            st.error(f"🚨 Chart data corrupted: {type(chart_copy)}")
            return None
        return chart_copy

    def store_analysis_results(self, symbol, analysis_results):
        self._analysis_store[symbol] = copy.deepcopy(analysis_results)

    def get_analysis_results(self, symbol):
        return self._analysis_store.get(symbol, {})

if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()

@st.cache_data(ttl=300)
def get_cached_market_data(symbol: str, period: str):
    return get_market_data_enhanced(symbol, period, show_debug=False)

def generate_cache_key(symbol: str, analysis_config: dict) -> str:
    config_str = str(sorted(analysis_config.items()))
    return hashlib.md5(f"{symbol}_{config_str}".encode()).hexdigest()

def is_etf(symbol):
    try:
        known_stocks = {'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'JPM', 'JNJ', 'UNH', 'V', 'PG', 'HD', 'MA', 'BAC', 'ABBV', 'PFE', 'KO', 'ADBE', 'PEP', 'TMO', 'COST', 'AVGO', 'NKE', 'MRK', 'ABT', 'CRM', 'LLY', 'ACN', 'TXN', 'DHR', 'WMT', 'NEE', 'VZ', 'ORCL', 'CMCSA', 'PM', 'DIS', 'BMY', 'RTX', 'HON', 'QCOM', 'UPS', 'T', 'AIG', 'LOW', 'MDT'}
        if symbol.upper() in known_stocks:
            return False
        
        common_etfs = {'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'AGG', 'BND', 'TLT', 'GLD', 'SLV', 'USO', 'UNG', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLRE', 'XLB', 'EFA', 'EEM', 'FXI', 'EWJ', 'EWG', 'EWU', 'ARKK', 'ARKQ', 'ARKW', 'ARKG', 'ARKF', 'FNGU', 'FNGD', 'MAGS', 'SOXX', 'SMH', 'IBB', 'XBI', 'JETS', 'HACK', 'ESPO', 'ICLN', 'PBW', 'KWEB', 'SPHB', 'SOXL', 'QQI', 'DIVO', 'URNM', 'GDX', 'FETH'}
        if symbol.upper() in common_etfs:
            return True
            
        return False
    except Exception as e:
        logger.error(f"ETF detection error for {symbol}: {e}")
        return False

def get_market_data_enhanced(symbol='SPY', period='1y', show_debug=False):
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
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in raw_data.columns]
        if missing_columns:
            st.error(f"❌ Missing columns: {missing_columns}")
            return None
        clean_data = raw_data[required_columns].copy().dropna()
        if len(clean_data) == 0:
            st.error(f"❌ No data after cleaning")
            return None
        clean_data['Typical_Price'] = (clean_data['High'] + clean_data['Low'] + clean_data['Close']) / 3
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
    try:
        if len(prices) < period + 1:
            return pd.Series([50] * len(prices), index=prices.index)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except Exception as e:
        logger.error(f"RSI calculation error: {e}")
        return pd.Series([50] * len(prices), index=prices.index)

# ... [The rest of your many calculation functions would go here, unchanged] ...
# To save space, I'll put placeholders, but in your real file, you would paste them all.
# Assume all functions from calculate_daily_vwap to statistical_normalize are here.

class VWVTradingSystem:
    def __init__(self, config=None):
        """Initialize enhanced trading system"""
        default_config = {
            'wvf_period': 22,
            'wvf_multiplier': 2.0,
            'ma_periods': [20, 50, 200],
            'volume_periods': [20, 50],
            'rsi_period': 14,
            'volatility_period': 20,
            'weights': {'wvf': 0.8, 'ma': 1.2, 'volume': 0.6, 'vwap': 0.4, 'momentum': 0.5, 'volatility': 0.3},
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
        # ... [Unchanged] ...
        return {'regime': 'NORMAL', 'volatility_regime': 'NORMAL', 'trend_regime': 'SIDEWAYS'}

    def get_dynamic_weights(self, market_regime):
        # ... [Unchanged] ...
        return self.weights.copy()

    def calculate_williams_vix_fix_enhanced(self, data):
        # ... [Unchanged] ...
        return {'binary_signal': 0, 'normalized_strength': 0.0, 'wvf_value': 0, 'upper_band': 0}

    # CORRECTED INDENTATION FOR THIS METHOD
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
                    mas.append(close.rolling(window=period).mean().iloc[-1])
            if not mas:
                return 0.0
            ma_avg = np.mean(mas)
            deviation_series = pd.Series([(ma_avg - p) / ma_avg * 100 for p in close.tail(252)])
            return statistical_normalize(deviation_series.abs())
        except Exception as e:
            logger.error(f"MA confluence calculation error: {e}")
            return 0.0
            
    # ... [The rest of your class methods go here, unchanged] ...

    def calculate_confluence(self, input_data, symbol='SPY', show_debug=False):
        # ... [The entire, long confluence calculation] ...
        return {'symbol': symbol, 'system_status': 'OPERATIONAL', 'components': {}, 'raw_confluence': 0, 'base_confluence': 0, 'directional_confluence': 0, 'signal_type': 'NONE', 'signal_direction': 'NONE', 'signal_strength': 0, 'entry_info': {}, 'enhanced_indicators': {}, 'current_price': 0, 'timestamp': ''}

def create_enhanced_chart(chart_market_data, analysis_results, symbol):
    # ... [Unchanged charting function] ...
    return go.Figure()

def main():
    # ... [The entire main function with the if/elif/else fix applied] ...
    st.markdown("<div class='main-header'><div class='header-content'><h1>VWV Professional Trading System</h1></div></div>", unsafe_allow_html=True)
    
    # ... Sidebar setup ...
    symbol = st.sidebar.text_input("Symbol", "SPY")
    period = st.sidebar.selectbox("Data Period", ['1y'], index=0)
    analyze_button = st.sidebar.button("Analyze Symbol")
    test_button = False # Simplified for clarity

    # --- CORRECTED LOGIC BLOCK ---
    if analyze_button and symbol:
        # ... [Your entire analysis logic] ...
        st.write("Analysis would be displayed here.")
    else:
        st.markdown("## Welcome to the VWV System")

if __name__ == "__main__":
    main()
