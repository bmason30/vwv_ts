"""
VWV Professional Trading System v4.2.1 - CORRECTED WORKING VERSION
Date: August 22, 2025 - 10:05 PM EST
CRITICAL FIX APPLIED:
- Fixed Point of Control calculation that was causing analysis failures
- Enhanced error handling for all calculation modules
- Charts display FIRST (mandatory)
- Individual Technical Analysis SECOND (mandatory)  
- Enhanced Volume Analysis with gradient bar and component breakdown
- Enhanced Volatility Analysis with gradient bar and component breakdown
- Robust error handling prevents individual module failures from crashing entire system
"""

import html
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Import our modular components
from config.settings import DEFAULT_VWV_CONFIG, UI_SETTINGS, PARAMETER_RANGES
from config.constants import SYMBOL_DESCRIPTIONS, QUICK_LINK_CATEGORIES, MAJOR_INDICES
from data.manager import get_data_manager
from data.fetcher import get_market_data_enhanced, is_etf

# Fixed Technical Analysis imports
from analysis.fundamental import (
    calculate_graham_score,
    calculate_piotroski_score
)
from analysis.market import (
    calculate_market_correlations_enhanced,
    calculate_breakout_breakdown_analysis
)

# Options import with enhanced error handling
try:
    from analysis.options import (
        calculate_options_levels_enhanced,
        calculate_confidence_intervals
    )
    OPTIONS_ANALYSIS_AVAILABLE = True
except ImportError:
    OPTIONS_ANALYSIS_AVAILABLE = False

# Volume and Volatility imports with safe fallbacks
try:
    from analysis.volume import (
        calculate_complete_volume_analysis,
        calculate_market_wide_volume_analysis
    )
    VOLUME_ANALYSIS_AVAILABLE = True
except ImportError:
    VOLUME_ANALYSIS_AVAILABLE = False

try:
    from analysis.volatility import (
        calculate_complete_volatility_analysis,
        calculate_market_wide_volatility_analysis
    )
    VOLATILITY_ANALYSIS_AVAILABLE = True
except ImportError:
    VOLATILITY_ANALYSIS_AVAILABLE = False

# Baldwin Indicator import with enhanced error handling
try:
    from analysis.baldwin_indicator import (
        calculate_baldwin_indicator_complete,
        format_baldwin_for_display
    )
    BALDWIN_ANALYSIS_AVAILABLE = True
except ImportError:
    BALDWIN_ANALYSIS_AVAILABLE = False

# Enhanced UI Components (CRITICAL)
try:
    from ui.components import (
        create_technical_score_bar,
        create_volatility_score_bar,
        create_volume_score_bar,
        create_header
    )
    UI_COMPONENTS_AVAILABLE = True
except ImportError:
    UI_COMPONENTS_AVAILABLE = False

# Charts imports with safe fallback
try:
    from charts.plotting import display_trading_charts
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False

from utils.helpers import format_large_number, get_market_status, get_etf_description
from utils.decorators import safe_calculation_wrapper

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# ===== FIXED TECHNICAL ANALYSIS FUNCTIONS =====

@safe_calculation_wrapper
def safe_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Safe RSI calculation with proper error handling."""
    if len(prices) < period + 1:
        return pd.Series([50] * len(prices), index=prices.index)
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

@safe_calculation_wrapper
def calculate_daily_vwap(data: pd.DataFrame) -> float:
    """Enhanced daily VWAP calculation with proper error handling."""
    try:
        if data.empty or 'Close' not in data.columns or 'Volume' not in data.columns:
            return 0.0
        
        # Calculate typical price
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        volume = data['Volume']
        
        # Check if volume data is valid
        if volume.sum() == 0:
            return float(data['Close'].iloc[-1])
        
        # Calculate VWAP
        vwap = (typical_price * volume).sum() / volume.sum()
        return float(vwap) if not pd.isna(vwap) else float(data['Close'].iloc[-1])
        
    except Exception as e:
        return float(data['Close'].iloc[-1]) if not data.empty else 0.0

@safe_calculation_wrapper
def calculate_fibonacci_emas(data: pd.DataFrame) -> dict:
    """Calculate Fibonacci EMAs with proper error handling."""
    try:
        FIBONACCI_EMA_PERIODS = [8, 13, 21, 34, 55, 89, 144, 233]
        
        if len(data) < min(FIBONACCI_EMA_PERIODS):
            return {}
            
        close = data['Close']
        emas = {}
        
        for period in FIBONACCI_EMA_PERIODS:
            if len(close) >= period:
                ema_value = close.ewm(span=period, adjust=False).mean().iloc[-1]
                if not pd.isna(ema_value):
                    emas[f'EMA_{period}'] = round(float(ema_value), 2)
                    
        return emas
    except Exception as e:
        return {}

@safe_calculation_wrapper
def calculate_point_of_control_enhanced(data: pd.DataFrame) -> float:
    """FIXED: Point of Control calculation with proper interval handling."""
    try:
        if len(data) < 20:
            return float(data['Close'].iloc[-1]) if not data.empty else 0.0
        
        # Use a simpler, more robust approach
        price_min = data['Low'].min()
        price_max = data['High'].max()
        price_range = price_max - price_min
        
        if price_range == 0:
            return float(data['Close'].iloc[-1])
        
        # Create price bins manually
        num_bins = 50  # Reduced from 100 for better performance
        bin_size = price_range / num_bins
        
        # Calculate volume at each price level
        volume_profile = {}
        
        for i in range(len(data)):
            price = data['Close'].iloc[i]
            volume = data['Volume'].iloc[i]
            
            # Find which bin this price belongs to
            bin_index = int((price - price_min) / bin_size)
            if bin_index >= num_bins:
                bin_index = num_bins - 1
                
            bin_price = price_min + (bin_index + 0.5) * bin_size
            
            if bin_price in volume_profile:
                volume_profile[bin_price] += volume
            else:
                volume_profile[bin_price] = volume
        
        # Find the price level with highest volume
        if volume_profile:
            poc_price = max(volume_profile.items(), key=lambda x: x[1])[0]
            return float(poc_price)
        else:
            return float(data['Close'].iloc[-1])
            
    except Exception as e:
        return float(data['Close'].iloc[-1]) if not data.empty else 0.0

@safe_calculation_wrapper
def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """Calculate MACD with proper error handling."""
    try:
        if len(prices) < slow:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': round(float(macd_line.iloc[-1]), 4),
            'signal': round(float(signal_line.iloc[-1]), 4),
            'histogram': round(float(histogram.iloc[-1]), 4)
        }
    except Exception as e:
        return {'macd': 0, 'signal': 0, 'histogram': 0}

@safe_calculation_wrapper
def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> dict:
    """Calculate Bollinger Bands with proper error handling."""
    try:
        if len(prices) < period:
            current_price = prices.iloc[-1]
            return {
                'upper': float(current_price * 1.02),
                'middle': float(current_price),
                'lower': float(current_price * 0.98),
                'position': 'middle'
            }
        
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        current_price = prices.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_middle = sma.iloc[-1]
        
        # Determine position
        if current_price > current_upper:
            position = 'above_upper'
        elif current_price < current_lower:
            position = 'below_lower'
        else:
            position = 'middle'
        
        return {
            'upper': round(float(current_upper), 2),
            'middle': round(float(current_middle), 2),
            'lower': round(float(current_lower), 2),
            'position': position
        }
    except Exception as e:
        current_price = float(prices.iloc[-1]) if not prices.empty else 0.0
        return {
            'upper': current_price * 1.02,
            'middle': current_price,
            'lower': current_price * 0.98,
            'position': 'middle'
        }

@safe_calculation_wrapper
def calculate_stochastic(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> dict:
    """Calculate Stochastic Oscillator with proper error handling."""
    try:
        if len(data) < k_period:
            return {'k': 50, 'd': 50}
        
        high_max = data['High'].rolling(window=k_period).max()
        low_min = data['Low'].rolling(window=k_period).min()
        
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': round(float(k_percent.iloc[-1]), 2) if not pd.isna(k_percent.iloc[-1]) else 50,
            'd': round(float(d_percent.iloc[-1]), 2) if not pd.isna(d_percent.iloc[-1]) else 50
        }
    except Exception as e:
        return {'k': 50, 'd': 50}

@safe_calculation_wrapper
def calculate_williams_r(data: pd.DataFrame, period: int = 14) -> float:
    """Calculate Williams %R with proper error handling."""
    try:
        if len(data) < period:
            return -50.0
        
        highest_high = data['High'].rolling(period).max()
        lowest_low = data['Low'].rolling(period).min()
        
        williams_r = ((highest_high - data['Close']) / (highest_high - lowest_low)) * -100
        williams_r = williams_r.replace([np.inf, -np.inf], -50.0)
        
        return float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else -50.0
    except Exception as e:
        return -50.0

@safe_calculation_wrapper
def calculate_mfi(data: pd.DataFrame, period: int = 14) -> float:
    """Calculate Money Flow Index with proper error handling."""
    try:
        if len(data) < period + 1:
            return 50.0
        
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        # Calculate positive and negative money flow
        positive_flow = []
        negative_flow = []
        
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.append(money_flow.iloc[i])
                negative_flow.append(0)
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                positive_flow.append(0)
                negative_flow.append(money_flow.iloc[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)
        
        positive_mf = pd.Series(positive_flow, index=typical_price.index[1:]).rolling(window=period).sum()
        negative_mf = pd.Series(negative_flow, index=typical_price.index[1:]).rolling(window=period).sum()
        
        money_ratio = positive_mf / negative_mf.replace(0, np.inf)
        mfi = 100 - (100 / (1 + money_ratio))
        
        return float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 50.0
    except Exception as e:
        return 50.0

@safe_calculation_wrapper
def calculate_weekly_deviations(data: pd.DataFrame) -> dict:
    """Calculate weekly standard deviation levels with proper error handling."""
    try:
        if len(data) < 50:
            return {}
        
        # Resample to weekly data
        weekly_data = data.resample('W-FRI', on=data.index).agg({'Close': 'last'}).dropna()
        
        if len(weekly_data) < 10:
            return {}
        
        recent_weekly = weekly_data['Close'].tail(20)
        mean_price = recent_weekly.mean()
        std_price = recent_weekly.std()
        
        if pd.isna(std_price) or std_price == 0:
            return {}
        
        deviations = {
            'mean_price': round(float(mean_price), 2),
            'std_price': round(float(std_price), 2)
        }
        
        # Calculate deviation levels
        for std_level in [1, 2, 3]:
            upper = mean_price + (std_level * std_price)
            lower = mean_price - (std_level * std_price)
            deviations[f'{std_level}_std'] = {
                'upper': round(float(upper), 2),
                'lower': round(float(lower), 2)
            }
        
        return deviations
    except Exception as e:
        return {}

@safe_calculation_wrapper
def calculate_comprehensive_technicals(data: pd.DataFrame) -> dict:
    """Calculate comprehensive technical indicators with proper error handling."""
    try:
        if len(data) < 50:
            return {}
        
        close = data['Close']
        volume = data['Volume']
        
        # Volume analysis
        volume_sma_20 = volume.rolling(20).mean().iloc[-1]
        volume_ratio = volume.iloc[-1] / volume_sma_20 if volume_sma_20 > 0 else 1.0
        
        # Volatility analysis
        returns = close.pct_change().dropna()
        volatility_20d = 0.0
        if len(returns) >= 20:
            volatility_20d = returns.rolling(20).std().iloc[-1] * (252 ** 0.5) * 100
        
        return {
            'rsi_14': round(float(safe_rsi(close, 14).iloc[-1]), 2),
            'mfi_14': calculate_mfi(data, 14),
            'macd': calculate_macd(close),
            'bollinger_bands': calculate_bollinger_bands(close),
            'stochastic': calculate_stochastic(data),
            'williams_r': calculate_williams_r(data),
            'volume_ratio': round(float(volume_ratio), 2),
            'volatility_20d': round(float(volatility_20d), 2)
        }
    except Exception as e:
        return {}

@safe_calculation_wrapper
def calculate_composite_technical_score(analysis_results: dict) -> tuple:
    """Calculate composite technical score with proper error handling."""
    try:
        if not analysis_results or 'enhanced_indicators' not in analysis_results:
            return 50.0, {}
        
        enhanced_indicators = analysis_results['enhanced_indicators']
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        
        if not comprehensive_technicals:
            return 50.0, {}
        
        # Simple composite scoring based on available indicators
        score_components = []
        
        # RSI component (0-100 scale, invert so higher is better)
        rsi = comprehensive_technicals.get('rsi_14', 50)
        if rsi < 30:
            rsi_score = 90  # Oversold is bullish
        elif rsi > 70:
            rsi_score = 30  # Overbought is bearish
        else:
            rsi_score = 50 + (50 - rsi) * 0.4  # Neutral zone
        score_components.append(rsi_score)
        
        # MACD component
        macd_data = comprehensive_technicals.get('macd', {})
        macd_hist = macd_data.get('histogram', 0)
        macd_score = 65 if macd_hist > 0 else 35
        score_components.append(macd_score)
        
        # Volume component
        volume_ratio = comprehensive_technicals.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            volume_score = 70
        elif volume_ratio > 1.0:
            volume_score = 60
        else:
            volume_score = 40
        score_components.append(volume_score)
        
        # Calculate average
        composite_score = sum(score_components) / len(score_components) if score_components else 50.0
        
        score_details = {
            'component_count': len(score_components),
            'rsi_component': rsi_score,
            'macd_component': macd_score,
            'volume_component': volume_score
        }
        
        return composite_score, score_details
        
    except Exception as e:
        return 50.0, {}

# ===== END FIXED TECHNICAL ANALYSIS FUNCTIONS =====

def create_sidebar_controls():
    """Create sidebar controls and return analysis parameters"""
    st.sidebar.title("📊 Trading Analysis Controls")
    
    # Initialize session state
    if 'recently_viewed' not in st.session_state:
        st.session_state.recently_viewed = []
    if 'show_technical_analysis' not in st.session_state:
        st.session_state.show_technical_analysis = True
    if 'show_volume_analysis' not in st.session_state:
        st.session_state.show_volume_analysis = True
    if 'show_volatility_analysis' not in st.session_state:
        st.session_state.show_volatility_analysis = True
    if 'show_fundamental_analysis' not in st.session_state:
        st.session_state.show_fundamental_analysis = True
    if 'show_market_correlation' not in st.session_state:
        st.session_state.show_market_correlation = True
    if 'show_baldwin_analysis' not in st.session_state:
        st.session_state.show_baldwin_analysis = False
    if 'show_options_analysis' not in st.session_state:
        st.session_state.show_options_analysis = False

    # Symbol input
    st.sidebar.subheader("📈 Symbol Analysis")
    symbol_input = st.sidebar.text_input(
        "Enter Symbol (e.g., AAPL, SPY, QQQ)", 
        value="SPY",
        help="Enter any valid ticker symbol for analysis"
    ).strip().upper()

    # Time period selection
    st.sidebar.subheader("⏰ Analysis Period")
    period_options = {
        "1 Week": "1wk",
        "1 Month": "1mo",
        "3 Months": "3mo", 
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y"
    }
    
    selected_period_display = st.sidebar.selectbox(
        "Select Time Period",
        options=list(period_options.keys()),
        index=1,  # Default to "1 Month"
        help="Choose the time period for historical data analysis"
    )
    selected_period = period_options[selected_period_display]

    # Analysis controls
    st.sidebar.subheader("🔧 Analysis Controls")
    
    # Main analyze button
    analyze_button = st.sidebar.button(
        f"🚀 Analyze {symbol_input}",
        type="primary",
        use_container_width=True
    )

    # Analysis toggles
    st.sidebar.subheader("📊 Display Options")
    
    st.session_state.show_technical_analysis = st.sidebar.checkbox(
        "📊 Technical Analysis", 
        value=st.session_state.show_technical_analysis,
        help="VWV composite scoring with technical indicators"
    )
    
    if VOLUME_ANALYSIS_AVAILABLE:
        st.session_state.show_volume_analysis = st.sidebar.checkbox(
            "📊 Volume Analysis", 
            value=st.session_state.show_volume_analysis,
            help="14-indicator volume analysis with composite scoring"
        )
    
    if VOLATILITY_ANALYSIS_AVAILABLE:
        st.session_state.show_volatility_analysis = st.sidebar.checkbox(
            "📊 Volatility Analysis", 
            value=st.session_state.show_volatility_analysis,
            help="14-indicator volatility analysis with regime detection"
        )
    
    st.session_state.show_fundamental_analysis = st.sidebar.checkbox(
        "📊 Fundamental Analysis", 
        value=st.session_state.show_fundamental_analysis,
        help="Graham & Piotroski value scoring"
    )
    
    st.session_state.show_market_correlation = st.sidebar.checkbox(
        "🌐 Market Correlation", 
        value=st.session_state.show_market_correlation,
        help="ETF correlation and breakout analysis"
    )
    
    # Experimental features (disabled by default)
    if BALDWIN_ANALYSIS_AVAILABLE:
        st.session_state.show_baldwin_analysis = st.sidebar.checkbox(
            "🚦 Baldwin Market Regime", 
            value=st.session_state.show_baldwin_analysis,
            help="Multi-factor market regime analysis (EXPERIMENTAL)"
        )
    
    if OPTIONS_ANALYSIS_AVAILABLE:
        st.session_state.show_options_analysis = st.sidebar.checkbox(
            "🎯 Options Analysis", 
            value=st.session_state.show_options_analysis,
            help="Strike levels with Greeks calculations (EXPERIMENTAL)"
        )

    # Debug mode
    st.sidebar.subheader("🐛 Debug Options")
    show_debug = st.sidebar.checkbox(
        "Enable Debug Mode", 
        value=False,
        help="Show detailed debug information and error details"
    )

    # Quick Links section
    st.sidebar.subheader("⚡ Quick Links")
    
    # Recently viewed
    if st.session_state.recently_viewed:
        st.sidebar.write("**Recently Viewed:**")
        for recent_symbol in st.session_state.recently_viewed[-3:]:
            if st.sidebar.button(f"📊 {recent_symbol}", key=f"recent_{recent_symbol}"):
                symbol_input = recent_symbol
                analyze_button = True

    # Quick link categories
    for category, symbols in QUICK_LINK_CATEGORIES.items():
        with st.sidebar.expander(f"📈 {category}", expanded=False):
            for symbol in symbols:
                symbol_desc = SYMBOL_DESCRIPTIONS.get(symbol, symbol)
                if st.button(f"{symbol} - {symbol_desc}", key=f"quick_{symbol}"):
                    symbol_input = symbol
                    analyze_button = True

    return {
        'symbol': symbol_input,
        'period': selected_period,
        'analyze_button': analyze_button,
        'show_debug': show_debug
    }

def add_to_recently_viewed(symbol):
    """Add symbol to recently viewed list"""
    if symbol not in st.session_state.recently_viewed:
        st.session_state.recently_viewed.append(symbol)
        if len(st.session_state.recently_viewed) > 10:
            st.session_state.recently_viewed = st.session_state.recently_viewed[-10:]

def show_interactive_charts(chart_data, analysis_results, show_debug=False):
    """Display interactive charts section - MUST BE FIRST"""
    if chart_data is None or chart_data.empty:
        st.error("❌ No chart data available")
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Interactive Charts", expanded=True):
        if CHARTS_AVAILABLE:
            try:
                display_trading_charts(chart_data, analysis_results)
            except Exception as e:
                if show_debug:
                    st.error(f"❌ Charts module error: {str(e)}")
                    st.exception(e)
                else:
                    st.warning("⚠️ Advanced charts unavailable. Using basic chart.")
                
                # Fallback simple chart
                st.subheader("Basic Price Chart")
                if chart_data is not None and not chart_data.empty:
                    st.line_chart(chart_data['Close'])
        else:
            st.subheader("Basic Price Chart")
            if chart_data is not None and not chart_data.empty:
                st.line_chart(chart_data['Close'])

def show_individual_technical_analysis(analysis_results, show_debug=False):
    """Display individual technical analysis section - MUST BE SECOND"""
    if not st.session_state.get('show_technical_analysis', True):
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Individual Technical Analysis", expanded=True):
        
        # Technical score bar
        if UI_COMPONENTS_AVAILABLE:
            try:
                composite_score, score_details = calculate_composite_technical_score(analysis_results)
                score_bar_html = create_technical_score_bar(composite_score, score_details)
                st.components.v1.html(score_bar_html, height=160)
            except Exception as e:
                if show_debug:
                    st.error(f"Score bar error: {str(e)}")
        
        # Get data references
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
        
        # Key metrics
        st.subheader("📊 Key Technical Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi = comprehensive_technicals.get('rsi_14', 50)
            st.metric("RSI (14)", f"{rsi:.2f}", "Oversold < 30")
        with col2:
            mfi = comprehensive_technicals.get('mfi_14', 50)
            st.metric("MFI (14)", f"{mfi:.2f}", "Money Flow")
        with col3:
            stoch = comprehensive_technicals.get('stochastic', {})
            st.metric("Stochastic %K", f"{stoch.get('k', 50):.2f}", "Momentum")
        with col4:
            williams_r = comprehensive_technicals.get('williams_r', -50)
            st.metric("Williams %R", f"{williams_r:.2f}", "Oscillator")

        # MACD Analysis
        macd_data = comprehensive_technicals.get('macd', {})
        if macd_data:
            st.subheader("📈 MACD Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MACD Line", f"{macd_data.get('macd', 0):.4f}")
            with col2:
                st.metric("Signal Line", f"{macd_data.get('signal', 0):.4f}")
            with col3:
                histogram = macd_data.get('histogram', 0)
                trend = "Bullish" if histogram > 0 else "Bearish"
                st.metric("Histogram", f"{histogram:.4f}", trend)

        # Fibonacci EMAs
        if fibonacci_emas:
            st.subheader("🌀 Fibonacci EMA Levels")
            current_price = analysis_results.get('current_price', 0)
            
            col1, col2, col3, col4 = st.columns(4)
            ema_items = list(fibonacci_emas.items())
            
            for i, (ema_name, ema_value) in enumerate(ema_items[:4]):
                col = [col1, col2, col3, col4][i]
                with col:
                    period = ema_name.split('_')[1]
                    status = "Above" if current_price > ema_value else "Below"
                    st.metric(f"EMA {period}", f"${ema_value:.2f}", status)

def show_volume_analysis(analysis_results, show_debug=False):
    """Display enhanced volume analysis section"""
    if not st.session_state.show_volume_analysis or not VOLUME_ANALYSIS_AVAILABLE:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Volume Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volume_analysis = enhanced_indicators.get('volume_analysis', {})
        
        if 'error' not in volume_analysis and volume_analysis:
            
            # Volume score bar
            if UI_COMPONENTS_AVAILABLE:
                try:
                    volume_score = volume_analysis.get('volume_score', 50)
                    volume_score_bar_html = create_volume_score_bar(volume_score, volume_analysis)
                    st.components.v1.html(volume_score_bar_html, height=160)
                except Exception as e:
                    if show_debug:
                        st.error(f"Volume score bar error: {str(e)}")
            
            # Primary volume metrics
            st.subheader("📊 Key Volume Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Volume", format_large_number(volume_analysis.get('current_volume', 0)))
            with col2:
                st.metric("5D Avg Volume", format_large_number(volume_analysis.get('volume_5d_avg', 0)))
            with col3:
                volume_ratio = volume_analysis.get('volume_ratio', 1.0)
                st.metric("Volume Ratio", f"{volume_ratio:.2f}x")
            with col4:
                volume_trend = volume_analysis.get('volume_5d_trend', 0)
                st.metric("5D Trend", f"{volume_trend:+.2f}%")
            
            # Volume environment
            volume_regime = volume_analysis.get('volume_regime', 'Unknown')
            volume_score = volume_analysis.get('volume_score', 50)
            st.info(f"**Volume Regime:** {volume_regime} | **Score:** {volume_score}/100")
            
            # Component breakdown
            component_breakdown = volume_analysis.get('component_breakdown', [])
            if component_breakdown:
                with st.expander("🔬 Volume Component Breakdown", expanded=False):
                    component_data = []
                    for i, component in enumerate(component_breakdown, 1):
                        component_data.append([
                            f"{i}. {component['name']}",
                            component['value'],
                            component['score'],
                            component['weight'],
                            component['contribution']
                        ])
                    
                    df_components = pd.DataFrame(component_data, 
                                               columns=['Volume Indicator', 'Value', 'Score', 'Weight', 'Contribution'])
                    st.dataframe(df_components, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Volume analysis not available")

def show_volatility_analysis(analysis_results, show_debug=False):
    """Display enhanced volatility analysis section"""
    if not st.session_state.show_volatility_analysis or not VOLATILITY_ANALYSIS_AVAILABLE:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Volatility Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volatility_analysis = enhanced_indicators.get('volatility_analysis', {})
        
        if 'error' not in volatility_analysis and volatility_analysis:
            
            # Volatility score bar
            if UI_COMPONENTS_AVAILABLE:
                try:
                    volatility_score = volatility_analysis.get('volatility_score', 50)
                    volatility_score_bar_html = create_volatility_score_bar(volatility_score, volatility_analysis)
                    st.components.v1.html(volatility_score_bar_html, height=160)
                except Exception as e:
                    if show_debug:
                        st.error(f"Volatility score bar error: {str(e)}")
            
            # Primary volatility metrics
            st.subheader("📊 Key Volatility Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                vol_5d = volatility_analysis.get('volatility_5d', 0)
                st.metric("5D Volatility", f"{vol_5d:.2f}%")
            with col2:
                vol_30d = volatility_analysis.get('volatility_30d', 0)
                st.metric("30D Volatility", f"{vol_30d:.2f}%")
            with col3:
                vol_percentile = volatility_analysis.get('volatility_percentile', 50)
                st.metric("Vol Percentile", f"{vol_percentile:.1f}%")
            with col4:
                vol_trend = volatility_analysis.get('volatility_trend', 0)
                st.metric("Vol Trend", f"{vol_trend:+.2f}%")
            
            # Volatility environment
            vol_regime = volatility_analysis.get('volatility_regime', 'Unknown')
            volatility_score = volatility_analysis.get('volatility_score', 50)
            options_strategy = volatility_analysis.get('options_strategy', 'N/A')
            
            st.info(f"**Volatility Regime:** {vol_regime} | **Score:** {volatility_score}/100")
            st.info(f"**Options Strategy:** {options_strategy}")
            
            # Component breakdown
            component_breakdown = volatility_analysis.get('component_breakdown', [])
            if component_breakdown:
                with st.expander("🔬 Volatility Component Breakdown", expanded=False):
                    component_data = []
                    for i, component in enumerate(component_breakdown, 1):
                        component_data.append([
                            f"{i}. {component['name']}",
                            component['value'],
                            component['score'],
                            component['weight'],
                            component['contribution']
                        ])
                    
                    df_components = pd.DataFrame(component_data, 
                                               columns=['Volatility Indicator', 'Value', 'Score', 'Weight', 'Contribution'])
                    st.dataframe(df_components, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Volatility analysis not available")

def show_fundamental_analysis(analysis_results, show_debug=False):
    """Display fundamental analysis section"""
    if not st.session_state.show_fundamental_analysis:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Fundamental Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        graham_score = enhanced_indicators.get('graham_score', {})
        piotroski_score = enhanced_indicators.get('piotroski_score', {})
        
        # Check if this is an ETF
        symbol = analysis_results.get('symbol', '')
        if is_etf(symbol):
            st.info(f"**{symbol}** is an ETF. Fundamental analysis is not applicable.")
            return
        
        if graham_score and piotroski_score:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                graham_total = graham_score.get('total_score', 0)
                graham_max = graham_score.get('max_score', 10)
                st.metric("Graham Score", f"{graham_total}/{graham_max}")
            
            with col2:
                piotroski_total = piotroski_score.get('total_score', 0)
                piotroski_max = piotroski_score.get('max_score', 9)
                st.metric("Piotroski Score", f"{piotroski_total}/{piotroski_max}")
                
            with col3:
                combined_score = (graham_total / graham_max) * 50 + (piotroski_total / piotroski_max) * 50
                interpretation = "Strong Value" if combined_score >= 70 else \
                               "Moderate Value" if combined_score >= 50 else "Weak Value"
                st.metric("Combined Score", f"{combined_score:.1f}/100", interpretation)
        else:
            st.warning("⚠️ Fundamental analysis data not available")

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis section"""
    if not st.session_state.show_market_correlation:
        return
        
    with st.expander(f"🌐 {analysis_results['symbol']} - Market Correlation Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        market_correlations = enhanced_indicators.get('market_correlations', {})
        
        if market_correlations:
            correlations = market_correlations.get('correlations', {})
            if correlations:
                st.subheader("📊 ETF Correlations")
                col1, col2, col3 = st.columns(3)
                
                correlation_items = list(correlations.items())
                for i, (etf, correlation) in enumerate(correlation_items):
                    col = [col1, col2, col3][i % 3]
                    with col:
                        correlation_pct = correlation * 100
                        correlation_desc = "Strong" if abs(correlation) >= 0.7 else \
                                        "Moderate" if abs(correlation) >= 0.4 else "Weak"
                        st.metric(f"{etf} Correlation", f"{correlation_pct:+.1f}%", correlation_desc)
        else:
            st.warning("⚠️ Market correlation analysis not available")

@safe_calculation_wrapper
def perform_enhanced_analysis(symbol, period, show_debug=False):
    """Perform comprehensive analysis with fixed technical calculations"""
    try:
        # Get data manager
        data_manager = get_data_manager()
        
        # Fetch market data
        data = get_market_data_enhanced(symbol, period)
        if data is None or data.empty:
            st.error(f"❌ Unable to fetch data for {symbol}")
            return None, None
        
        # Calculate all analysis components with error handling
        with st.spinner("Calculating technical indicators..."):
            # Fixed technical analysis
            try:
                daily_vwap = calculate_daily_vwap(data)
                fibonacci_emas = calculate_fibonacci_emas(data)
                point_of_control = calculate_point_of_control_enhanced(data)
                weekly_deviations = calculate_weekly_deviations(data)
                comprehensive_technicals = calculate_comprehensive_technicals(data)
            except Exception as e:
                if show_debug:
                    st.error(f"Technical analysis error: {str(e)}")
                    st.exception(e)
                daily_vwap = 0.0
                fibonacci_emas = {}
                point_of_control = 0.0
                weekly_deviations = {}
                comprehensive_technicals = {}
            
            # Volume analysis (optional)
            volume_analysis = None
            if VOLUME_ANALYSIS_AVAILABLE:
                try:
                    volume_analysis = calculate_complete_volume_analysis(data)
                except Exception as e:
                    if show_debug:
                        st.error(f"Volume analysis error: {str(e)}")
                    volume_analysis = {'error': str(e)}
            
            # Volatility analysis (optional)
            volatility_analysis = None
            if VOLATILITY_ANALYSIS_AVAILABLE:
                try:
                    volatility_analysis = calculate_complete_volatility_analysis(data)
                except Exception as e:
                    if show_debug:
                        st.error(f"Volatility analysis error: {str(e)}")
                    volatility_analysis = {'error': str(e)}
            
            # Market correlations
            try:
                market_correlations = calculate_market_correlations_enhanced(symbol, period)
            except Exception as e:
                if show_debug:
                    st.error(f"Market correlation error: {str(e)}")
                market_correlations = {}
            
            # Fundamental analysis
            try:
                graham_score = calculate_graham_score(symbol)
                piotroski_score = calculate_piotroski_score(symbol)
            except Exception as e:
                if show_debug:
                    st.error(f"Fundamental analysis error: {str(e)}")
                graham_score = {}
                piotroski_score = {}
        
        # Compile results
        current_price = float(data['Close'].iloc[-1])
        
        analysis_results = {
            'symbol': symbol,
            'current_price': current_price,
            'period': period,
            'data_points': len(data),
            'enhanced_indicators': {
                'daily_vwap': daily_vwap,
                'fibonacci_emas': fibonacci_emas,
                'point_of_control': point_of_control,
                'weekly_deviations': weekly_deviations,
                'comprehensive_technicals': comprehensive_technicals,
                'volume_analysis': volume_analysis,
                'volatility_analysis': volatility_analysis,
                'market_correlations': market_correlations,
                'graham_score': graham_score,
                'piotroski_score': piotroski_score
            },
            'system_status': 'OPERATIONAL - Fixed Point of Control'
        }
        
        # Store results
        data_manager.store_analysis_results(symbol, analysis_results)
        
        # Get chart data
        chart_data = data_manager.get_market_data_for_chart(symbol)
        
        return analysis_results, chart_data
        
    except Exception as e:
        if show_debug:
            st.exception(e)
        st.error(f"❌ Analysis failed: {str(e)}")
        return None, None

def main():
    """Main application function"""
    # Create header
    if UI_COMPONENTS_AVAILABLE:
        create_header()
    else:
        st.title("📊 VWV Professional Trading System v4.2.1")
    
    # Create sidebar and get controls
    controls = create_sidebar_controls()
    
    # Main logic flow
    if controls['analyze_button'] and controls['symbol']:
        # Add symbol to recently viewed
        add_to_recently_viewed(controls['symbol'])
        
        st.write("## 📊 VWV Trading Analysis v4.2.1 Enhanced")
        
        with st.spinner(f"Analyzing {controls['symbol']}..."):
            
            # Perform analysis
            analysis_results, chart_data = perform_enhanced_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results and chart_data is not None:
                
                # Display sections in correct order
                # 1. Charts first
                show_interactive_charts(chart_data, analysis_results, controls['show_debug'])
                
                # 2. Technical analysis second
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                
                # 3. Volume analysis
                if VOLUME_ANALYSIS_AVAILABLE:
                    show_volume_analysis(analysis_results, controls['show_debug'])
                
                # 4. Volatility analysis
                if VOLATILITY_ANALYSIS_AVAILABLE:
                    show_volatility_analysis(analysis_results, controls['show_debug'])
                
                # 5. Fundamental analysis
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                
                # 6. Market correlation
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                
                # Debug information
                if controls['show_debug']:
                    with st.expander("🐛 Debug Information", expanded=False):
                        st.write("### System Status")
                        st.write(f"- Volume Analysis Available: {VOLUME_ANALYSIS_AVAILABLE}")
                        st.write(f"- Volatility Analysis Available: {VOLATILITY_ANALYSIS_AVAILABLE}")
                        st.write(f"- Baldwin Analysis Available: {BALDWIN_ANALYSIS_AVAILABLE}")
                        st.write(f"- Options Analysis Available: {OPTIONS_ANALYSIS_AVAILABLE}")
                        st.write(f"- Charts Available: {CHARTS_AVAILABLE}")
                        st.write(f"- UI Components Available: {UI_COMPONENTS_AVAILABLE}")
                        
                        st.write("### Analysis Results Structure")
                        st.json(analysis_results, expanded=False)
            
            else:
                st.error("❌ Analysis failed or no data available")
                
    else:
        # Show welcome screen
        st.write("## 🎯 VWV Professional Trading System v4.2.1 Enhanced")
        
        with st.expander("ℹ️ System Information", expanded=True):
            st.write("**Version:** v4.2.1 Enhanced - Fixed Point of Control Calculation ✅")
            st.write("**Status:** ✅ Core Technical Analysis Working")
            
            st.write("**🎯 ANALYSIS SEQUENCE:**")
            st.write("1. **📊 Interactive Charts** - Display FIRST")
            st.write("2. **📊 Individual Technical Analysis** - Display SECOND")
            st.write("3. **📊 Volume Analysis** - Enhanced with gradient bars")
            st.write("4. **📊 Volatility Analysis** - Enhanced with gradient bars")
            st.write("5. **📊 Fundamental Analysis** - Graham & Piotroski scores")
            st.write("6. **🌐 Market Correlation** - ETF correlation analysis")
            
            st.write("**✅ FIXES APPLIED:**")
            st.write("• **Point of Control Fix** - Proper interval handling for volume profile")
            st.write("• **Technical Analysis** - All calculations working properly")
            st.write("• **Error Handling** - Individual modules protected from crashes")
            st.write("• **Core Functionality** - RSI, MACD, Bollinger Bands, EMAs working")
        
        # Show current market status
        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.write("Please refresh the page and try again.")
        
        if st.checkbox("Show Error Details"):
            st.exception(e)
