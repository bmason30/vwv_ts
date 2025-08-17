"""
VWV Professional Trading System v8.0.0 - Volume Analysis Enhancement
NEW FEATURES:
✅ Volume Composite Score Bar (red-green gradient)
✅ Nested expandable container with all calculated values (collapsed by default)
✅ Enhanced volume metrics display
✅ Smart money signals
✅ Fixed pandas FutureWarning
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
from analysis.technical import (
    calculate_daily_vwap, 
    calculate_fibonacci_emas,
    calculate_point_of_control_enhanced,
    calculate_comprehensive_technicals,
    calculate_weekly_deviations,
    calculate_composite_technical_score
)
from analysis.fundamental import (
    calculate_graham_score,
    calculate_piotroski_score
)
from analysis.market import (
    calculate_market_correlations_enhanced,
    calculate_breakout_breakdown_analysis
)
from analysis.options import (
    calculate_options_levels_enhanced,
    calculate_confidence_intervals
)

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

# Baldwin Indicator imports with safe fallback
try:
    from analysis.baldwin_indicator import (
        calculate_baldwin_indicator_complete,
        format_baldwin_for_display
    )
    BALDWIN_ANALYSIS_AVAILABLE = True
except ImportError:
    BALDWIN_ANALYSIS_AVAILABLE = False

try:
    from analysis.options_advanced import (
        calculate_complete_advanced_options,
        format_advanced_options_for_display
    )
    ADVANCED_OPTIONS_AVAILABLE = True
except ImportError:
    ADVANCED_OPTIONS_AVAILABLE = False

try:
    from analysis.vwv_core import (
        calculate_vwv_system_complete,
        get_vwv_signal_interpretation
    )
    VWV_CORE_AVAILABLE = True
except ImportError:
    VWV_CORE_AVAILABLE = False

from ui.components import (
    create_technical_score_bar,
    create_volume_score_bar,  # NEW import for v8.0.0
    create_header
)
from charts.plotting import display_trading_charts
from utils.helpers import format_large_number, get_market_status, get_etf_description
from utils.decorators import safe_calculation_wrapper

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

# Page configuration
st.set_page_config(
    page_title="VWV Professional Trading System v8.0.0",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_sidebar_controls():
    """Create sidebar controls and return analysis parameters"""
    st.sidebar.title("📊 Trading Analysis")
    
    # Initialize session state
    if 'recently_viewed' not in st.session_state:
        st.session_state.recently_viewed = []
    if 'show_vwv_analysis' not in st.session_state:
        st.session_state.show_vwv_analysis = True
    if 'show_fundamental_analysis' not in st.session_state:
        st.session_state.show_fundamental_analysis = True
    if 'show_market_correlation' not in st.session_state:
        st.session_state.show_market_correlation = True
    if 'show_options_analysis' not in st.session_state:
        st.session_state.show_options_analysis = True
    if 'show_confidence_intervals' not in st.session_state:
        st.session_state.show_confidence_intervals = True
    if 'show_volume_analysis' not in st.session_state:
        st.session_state.show_volume_analysis = True
    if 'show_volatility_analysis' not in st.session_state:
        st.session_state.show_volatility_analysis = True
    if 'show_baldwin_analysis' not in st.session_state:
        st.session_state.show_baldwin_analysis = True

    # Quick Links Section
    st.sidebar.subheader("🔗 Quick Links")
    
    for category, symbols in QUICK_LINK_CATEGORIES.items():
        with st.sidebar.expander(f"📈 {category}", expanded=False):
            cols = st.columns(2)
            for i, symbol in enumerate(symbols):
                with cols[i % 2]:
                    if st.button(f"{symbol}", key=f"quick_{symbol}", help=SYMBOL_DESCRIPTIONS.get(symbol, "")):
                        st.session_state.symbol = symbol
                        st.session_state.analyze_on_select = True
                        st.rerun()

    # Symbol Input
    st.sidebar.subheader("🎯 Symbol Analysis")
    symbol = st.sidebar.text_input(
        "Enter Symbol:",
        value=st.session_state.get('symbol', 'SPY'),
        help="Enter stock symbol (e.g., AAPL, MSFT, SPY)"
    ).upper()

    # Period Selection with 1mo default
    period = st.sidebar.selectbox(
        "Time Period:",
        options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=0,  # Default to '1mo' (first option)
        help="Select analysis time period"
    )

    # Analysis Options
    st.sidebar.subheader("📊 Analysis Options")
    
    show_vwv_analysis = st.sidebar.checkbox(
        "🎯 VWV Analysis", 
        value=st.session_state.show_vwv_analysis,
        help="Show VWV proprietary signals"
    )
    
    show_fundamental_analysis = st.sidebar.checkbox(
        "📊 Fundamental Analysis", 
        value=st.session_state.show_fundamental_analysis,
        help="Graham & Piotroski scores"
    )
    
    show_volume_analysis = st.sidebar.checkbox(
        "📊 Volume Analysis", 
        value=st.session_state.show_volume_analysis and VOLUME_ANALYSIS_AVAILABLE,
        disabled=not VOLUME_ANALYSIS_AVAILABLE,
        help="Enhanced volume analysis with smart money detection"
    )
    
    show_volatility_analysis = st.sidebar.checkbox(
        "📊 Volatility Analysis", 
        value=st.session_state.show_volatility_analysis and VOLATILITY_ANALYSIS_AVAILABLE,
        disabled=not VOLATILITY_ANALYSIS_AVAILABLE,
        help="Volatility regime analysis"
    )
    
    show_baldwin_analysis = st.sidebar.checkbox(
        "🚦 Baldwin Market Regime", 
        value=st.session_state.show_baldwin_analysis and BALDWIN_ANALYSIS_AVAILABLE,
        disabled=not BALDWIN_ANALYSIS_AVAILABLE,
        help="Baldwin market regime indicator"
    )
    
    show_market_correlation = st.sidebar.checkbox(
        "🌐 Market Correlation", 
        value=st.session_state.show_market_correlation,
        help="Market correlation analysis"
    )
    
    show_options_analysis = st.sidebar.checkbox(
        "🎯 Options Analysis", 
        value=st.session_state.show_options_analysis,
        help="Options levels and Greeks"
    )
    
    show_confidence_intervals = st.sidebar.checkbox(
        "📊 Confidence Intervals", 
        value=st.session_state.show_confidence_intervals,
        help="Statistical price projections"
    )

    # Advanced Options
    with st.sidebar.expander("⚙️ Advanced Options", expanded=False):
        show_debug = st.checkbox("🐛 Debug Mode", value=False, help="Show debug information")
        use_cache = st.checkbox("💾 Use Cache", value=True, help="Cache data for faster loading")
        
        # VWV Configuration
        st.write("**VWV Configuration:**")
        vwv_config = DEFAULT_VWV_CONFIG.copy()
        
        # Access weights from nested structure
        current_weights = vwv_config.get('weights', {})
        momentum_weight = current_weights.get('momentum', 0.5)
        ma_weight = current_weights.get('ma', 1.2)
        
        vwv_config['weights']['momentum'] = st.slider("Momentum Weight", 0.1, 0.9, momentum_weight, 0.1)
        vwv_config['weights']['ma'] = st.slider("Trend Weight", 0.1, 1.5, ma_weight, 0.1)

    # Update session state
    st.session_state.show_vwv_analysis = show_vwv_analysis
    st.session_state.show_fundamental_analysis = show_fundamental_analysis
    st.session_state.show_market_correlation = show_market_correlation
    st.session_state.show_options_analysis = show_options_analysis
    st.session_state.show_confidence_intervals = show_confidence_intervals
    st.session_state.show_volume_analysis = show_volume_analysis
    st.session_state.show_volatility_analysis = show_volatility_analysis
    st.session_state.show_baldwin_analysis = show_baldwin_analysis

    return {
        'symbol': symbol,
        'period': period,
        'vwv_config': vwv_config,
        'show_debug': show_debug,
        'use_cache': use_cache,
        'show_vwv_analysis': show_vwv_analysis,
        'show_fundamental_analysis': show_fundamental_analysis,
        'show_market_correlation': show_market_correlation,
        'show_options_analysis': show_options_analysis,
        'show_confidence_intervals': show_confidence_intervals,
        'show_volume_analysis': show_volume_analysis,
        'show_volatility_analysis': show_volatility_analysis,
        'show_baldwin_analysis': show_baldwin_analysis
    }

@safe_calculation_wrapper
def perform_complete_analysis(symbol, period, vwv_config, show_debug=False):
    """Perform comprehensive trading analysis"""
    
    if show_debug:
        st.write(f"🔍 Fetching data for {symbol} ({period})")
    
    # Get market data
    data_manager = get_data_manager()
    data = data_manager.get_cached_data(symbol, period)
    
    if data is None:
        data = get_market_data_enhanced(symbol, period)
        if data is not None:
            data_manager.cache_data(symbol, period, data)
    
    if data is None:
        return None, None, None
    
    if show_debug:
        st.write(f"✅ Data fetched: {len(data)} rows")
    
    # Get current price info
    current_price = data['Close'].iloc[-1]
    prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
    price_change = current_price - prev_close
    price_change_pct = (price_change / prev_close) * 100 if prev_close != 0 else 0
    
    # Basic info
    analysis_results = {
        'symbol': symbol,
        'current_price': current_price,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'volume': data['Volume'].iloc[-1],
        'market_cap': None,
        'data_quality': 'Good'
    }
    
    if show_debug:
        st.write(f"💰 Current Price: ${current_price:.2f} ({price_change_pct:+.2f}%)")
    
    # Technical Analysis
    try:
        # Core technical indicators
        daily_vwap = calculate_daily_vwap(data)
        fibonacci_emas = calculate_fibonacci_emas(data)
        poc_data = calculate_point_of_control_enhanced(data)
        comprehensive_technicals = calculate_comprehensive_technicals(data)
        weekly_deviations = calculate_weekly_deviations(data)
        
        # Enhanced technical analysis with volume
        from analysis.technical import calculate_enhanced_technical_analysis
        enhanced_indicators = calculate_enhanced_technical_analysis(data)
        
        analysis_results.update({
            'daily_vwap': daily_vwap,
            'fibonacci_emas': fibonacci_emas,
            'poc_data': poc_data,
            'comprehensive_technicals': comprehensive_technicals,
            'weekly_deviations': weekly_deviations,
            'enhanced_indicators': enhanced_indicators
        })
        
        if show_debug:
            st.write("✅ Technical analysis completed")
            
    except Exception as e:
        if show_debug:
            st.error(f"❌ Technical analysis error: {str(e)}")
        analysis_results['technical_error'] = str(e)
    
    # VWV Core Analysis
    vwv_results = None
    if VWV_CORE_AVAILABLE:
        try:
            vwv_results = calculate_vwv_system_complete(data, vwv_config)
            if show_debug:
                st.write("✅ VWV analysis completed")
        except Exception as e:
            if show_debug:
                st.error(f"❌ VWV analysis error: {str(e)}")
            vwv_results = {'error': str(e)}
    
    # Advanced Options Analysis
    if ADVANCED_OPTIONS_AVAILABLE:
        try:
            advanced_options = calculate_complete_advanced_options(symbol, current_price)
            analysis_results['advanced_options'] = advanced_options
            if show_debug:
                st.write("✅ Advanced options analysis completed")
        except Exception as e:
            if show_debug:
                st.error(f"❌ Advanced options error: {str(e)}")
            analysis_results['advanced_options_error'] = str(e)
    
    return analysis_results, vwv_results, data

def show_technical_analysis(analysis_results, show_debug=False):
    """Display enhanced technical analysis with composite score"""
    
    # Create header
    symbol = analysis_results['symbol']
    current_price = analysis_results['current_price']
    price_change = analysis_results['price_change']
    price_change_pct = analysis_results['price_change_pct']
    
    price_color = "green" if price_change >= 0 else "red"
    change_symbol = "+" if price_change >= 0 else ""
    
    st.markdown(f"""
    ### 🎯 {symbol} - Individual Technical Analysis
    **Current Price:** <span style='color: {price_color}; font-size: 1.2em; font-weight: bold;'>
    ${current_price:.2f} ({change_symbol}{price_change:+.2f} | {price_change_pct:+.2f}%)
    </span>
    """, unsafe_allow_html=True)
    
    # Technical Composite Score
    enhanced_indicators = analysis_results.get('enhanced_indicators', {})
    composite_score = enhanced_indicators.get('composite_score', 50)
    
    # Create and display technical score bar
    score_bar_html = create_technical_score_bar(composite_score)
    st.markdown(score_bar_html, unsafe_allow_html=True)
    
    # Main technical metrics
    comprehensive_technicals = analysis_results.get('comprehensive_technicals', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rsi = comprehensive_technicals.get('rsi', 50)
        rsi_signal = "🔴 Oversold" if rsi < 30 else "🟢 Overbought" if rsi > 70 else "🟡 Neutral"
        st.metric("RSI", f"{rsi:.1f}", rsi_signal)
        
    with col2:
        macd_histogram = comprehensive_technicals.get('macd_histogram', 0)
        macd_signal = "🟢 Bullish" if macd_histogram > 0 else "🔴 Bearish"
        st.metric("MACD", f"{macd_histogram:.3f}", macd_signal)
        
    with col3:
        bb_position = enhanced_indicators.get('bollinger_position', 50)
        bb_signal = "🔴 Oversold" if bb_position < 20 else "🟢 Overbought" if bb_position > 80 else "🟡 Normal"
        st.metric("BB Position", f"{bb_position:.1f}%", bb_signal)
        
    with col4:
        volume_trend = enhanced_indicators.get('volume_analysis', {}).get('volume_5d_trend', 0)
        volume_signal = "🟢 Rising" if volume_trend > 5 else "🔴 Falling" if volume_trend < -5 else "🟡 Stable"
        st.metric("Volume Trend", f"{volume_trend:+.1f}%", volume_signal)
    
    # Fibonacci EMAs
    fibonacci_emas = analysis_results.get('fibonacci_emas', {})
    if fibonacci_emas:
        st.subheader("📈 Fibonacci EMA Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            ema_trend = fibonacci_emas.get('trend_direction', 'Unknown')
            trend_strength = fibonacci_emas.get('trend_strength', 0)
            st.info(f"**EMA Trend:** {ema_trend} (Strength: {trend_strength:.1f})")
            
        with col2:
            ema_signal = fibonacci_emas.get('trading_signal', 'HOLD')
            signal_color = "🟢" if "BUY" in ema_signal else "🔴" if "SELL" in ema_signal else "🟡"
            st.info(f"**EMA Signal:** {signal_color} {ema_signal}")
    
    # Technical signals summary
    if enhanced_indicators:
        st.subheader("📊 Technical Signals Summary")
        
        # Generate technical signals
        from analysis.technical import generate_technical_signals
        signals = generate_technical_signals(analysis_results)
        
        if signals:
            signal_color = "🟢" if "BUY" in signals else "🔴" if "SELL" in signals else "🟡"
            st.success(f"**Overall Signal:** {signal_color} {signals}")
        
        # Show key support/resistance levels
        poc_data = analysis_results.get('poc_data', {})
        if poc_data:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Point of Control", f"${poc_data.get('poc_price', 0):.2f}")
            with col2:
                st.metric("Value Area", f"${poc_data.get('value_area_low', 0):.2f} - ${poc_data.get('value_area_high', 0):.2f}")

def show_volume_analysis(analysis_results, show_debug=False):
    """Enhanced Volume Analysis Display v8.0.0 with composite score bar and nested container"""
    if not st.session_state.show_volume_analysis or not VOLUME_ANALYSIS_AVAILABLE:
        return
        
    enhanced_indicators = analysis_results.get('enhanced_indicators', {})
    volume_analysis = enhanced_indicators.get('volume_analysis', {})
    
    if 'error' in volume_analysis or not volume_analysis:
        st.warning("⚠️ Volume analysis not available - insufficient data")
        return
    
    # Get volume composite score
    volume_composite_score = volume_analysis.get('volume_composite_score', 50)
    
    # Create volume score bar
    volume_score_bar_html = create_volume_score_bar(volume_composite_score)
    
    with st.expander(f"📊 {analysis_results['symbol']} - Volume Analysis", expanded=True):
        
        # Display volume composite score bar
        st.markdown("### 📊 Volume Composite Score")
        st.markdown(volume_score_bar_html, unsafe_allow_html=True)
        
        # Key Volume Metrics (always visible)
        st.markdown("### Key Volume Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_volume = volume_analysis.get('current_volume', 0)
            st.metric("Current Volume", format_large_number(current_volume))
            
        with col2:
            volume_5d_avg = volume_analysis.get('volume_5d_avg', 0)
            st.metric("5D Avg Volume", format_large_number(volume_5d_avg))
            
        with col3:
            volume_ratio = volume_analysis.get('volume_ratio', 1.0)
            st.metric("Volume Ratio", f"{volume_ratio:.2f}x", "vs 5D avg")
            
        with col4:
            volume_trend = volume_analysis.get('volume_5d_trend', 0)
            trend_color = "🟢" if volume_trend > 0 else "🔴" if volume_trend < 0 else "🟡"
            st.metric("Trend", f"{trend_color} {volume_trend:+.1f}%")
        
        # Enhanced Volume Signals (always visible)
        st.markdown("### Enhanced Volume Signals")
        col1, col2 = st.columns(2)
        
        with col1:
            volume_regime = volume_analysis.get('volume_regime', 'Unknown')
            st.info(f"**Volume Regime:** {volume_regime}")
            
            smart_money_signal = volume_analysis.get('smart_money_signal', 'Unknown')
            st.info(f"**Smart Money Signal:** {smart_money_signal}")
            
        with col2:
            volume_quality = volume_analysis.get('volume_quality', 'Unknown')
            st.info(f"**Volume Quality:** {volume_quality}")
            
            institutional_activity = volume_analysis.get('institutional_activity', 'Unknown')
            st.info(f"**Institutional Activity:** {institutional_activity}")
        
        # Nested expandable container with all calculated values (collapsed by default)
        with st.expander("🔍 Detailed Volume Analysis Data", expanded=False):
            
            # Basic Volume Tab
            with st.expander("📊 Basic Volume", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Volume Statistics:**")
                    st.write(f"• Current Volume: {format_large_number(current_volume)}")
                    st.write(f"• 5D Average: {format_large_number(volume_5d_avg)}")
                    st.write(f"• 30D Average: {format_large_number(volume_analysis.get('volume_30d_avg', 0))}")
                    st.write(f"• Volume Ratio (5D): {volume_ratio:.2f}x")
                    st.write(f"• Volume Ratio (30D): {volume_analysis.get('volume_ratio_30d', 1.0):.2f}x")
                    
                with col2:
                    st.write("**Volume Trends:**")
                    st.write(f"• 5D Trend: {volume_trend:+.2f}%")
                    st.write(f"• Volume Z-Score: {volume_analysis.get('volume_zscore', 0):.2f}")
                    st.write(f"• Volume Percentile: {volume_analysis.get('volume_percentile', 50):.1f}%")
                    st.write(f"• Relative Volume: {volume_analysis.get('relative_volume', 1.0):.2f}")
            
            # OBV Analysis Tab
            with st.expander("📈 OBV Analysis", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**On-Balance Volume:**")
                    st.write(f"• Current OBV: {volume_analysis.get('obv_current', 0):,.0f}")
                    st.write(f"• OBV Trend: {volume_analysis.get('obv_trend', 'Unknown')}")
                    st.write(f"• OBV Change: {volume_analysis.get('obv_change_pct', 0):+.2f}%")
                    
                with col2:
                    st.write("**OBV Signals:**")
                    st.write(f"• Price-OBV Divergence: {volume_analysis.get('obv_divergence', 'None')}")
                    st.write(f"• OBV Momentum: {volume_analysis.get('obv_momentum', 'Neutral')}")
                    st.write(f"• Accumulation Signal: {volume_analysis.get('accumulation_signal', 'Unknown')}")
            
            # A/D Line Tab
            with st.expander("📊 A/D Line", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Accumulation/Distribution:**")
                    st.write(f"• Current A/D Line: {volume_analysis.get('ad_line_current', 0):,.0f}")
                    st.write(f"• A/D Trend: {volume_analysis.get('ad_line_trend', 'Unknown')}")
                    st.write(f"• A/D Change: {volume_analysis.get('ad_line_change_pct', 0):+.2f}%")
                    
                with col2:
                    st.write("**Distribution Signals:**")
                    st.write(f"• Distribution Pattern: {volume_analysis.get('distribution_pattern', 'Unknown')}")
                    st.write(f"• Accumulation Phase: {volume_analysis.get('accumulation_phase', 'Unknown')}")
                    st.write(f"• Money Flow Direction: {volume_analysis.get('money_flow_direction', 'Unknown')}")
            
            # VROC & VPA Tab
            with st.expander("🚀 VROC & VPA", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Volume Rate of Change:**")
                    st.write(f"• VROC (14): {volume_analysis.get('vroc_14', 0):.2f}%")
                    st.write(f"• VROC Signal: {volume_analysis.get('vroc_signal', 'Neutral')}")
                    st.write(f"• Volume Momentum: {volume_analysis.get('volume_momentum', 'Unknown')}")
                    
                with col2:
                    st.write("**Volume Price Analysis:**")
                    st.write(f"• VPA Signal: {volume_analysis.get('vpa_signal', 'Unknown')}")
                    st.write(f"• Price-Volume Sync: {volume_analysis.get('price_volume_sync', 'Unknown')}")
                    st.write(f"• Breakout Confirmation: {volume_analysis.get('breakout_confirmation', 'Unknown')}")
            
            # Cluster Analysis Tab
            with st.expander("🔍 Cluster Analysis", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Volume Clusters:**")
                    st.write(f"• High Volume Zones: {volume_analysis.get('high_volume_zones', 'Unknown')}")
                    st.write(f"• Support/Resistance: {volume_analysis.get('volume_support_resistance', 'Unknown')}")
                    st.write(f"• Institutional Footprint: {volume_analysis.get('institutional_footprint', 'Unknown')}")
                    
                with col2:
                    st.write("**Advanced Metrics:**")
                    st.write(f"• Volume Profile: {volume_analysis.get('volume_profile', 'Unknown')}")
                    st.write(f"• Flow Intensity: {volume_analysis.get('flow_intensity', 'Unknown')}")
                    st.write(f"• Market Participation: {volume_analysis.get('market_participation', 'Unknown')}")

def show_volatility_analysis(analysis_results, show_debug=False):
    """Display volatility analysis section"""
    if not st.session_state.show_volatility_analysis or not VOLATILITY_ANALYSIS_AVAILABLE:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Volatility Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volatility_analysis = enhanced_indicators.get('volatility_analysis', {})
        
        if 'error' not in volatility_analysis and volatility_analysis:
            # Primary volatility metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                current_volatility = volatility_analysis.get('current_volatility', 0)
                st.metric("Current Volatility", f"{current_volatility:.2f}%")
            with col2:
                volatility_5d_avg = volatility_analysis.get('volatility_5d_avg', 0)
                st.metric("5D Avg Volatility", f"{volatility_5d_avg:.2f}%")
            with col3:
                volatility_ratio = volatility_analysis.get('volatility_ratio', 1.0)
                st.metric("Volatility Ratio", f"{volatility_ratio:.2f}x")
            with col4:
                volatility_trend = volatility_analysis.get('volatility_5d_trend', 0)
                st.metric("5D Volatility Trend", f"{volatility_trend:+.2f}%")
            
            # Volatility regime and implications
            st.subheader("📊 Volatility Environment")
            volatility_regime = volatility_analysis.get('volatility_regime', 'Unknown')
            trading_implications = volatility_analysis.get('trading_implications', 'No implications available')
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Volatility Regime:** {volatility_regime}")
                st.info(f"**Volatility Score:** {volatility_analysis.get('volatility_score', 50)}/100")
            with col2:
                st.info(f"**Trading Implications:**\n{trading_implications}")
                
        else:
            st.warning("⚠️ Volatility analysis not available - insufficient data")

def show_fundamental_analysis(analysis_results, show_debug=False):
    """Display fundamental analysis section"""
    if not st.session_state.show_fundamental_analysis:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Fundamental Analysis", expanded=True):
        symbol = analysis_results['symbol']
        
        # Skip fundamental analysis for obvious ETFs/indices
        if symbol in ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO'] or symbol.startswith('^'):
            st.info("ℹ️ Fundamental analysis not applicable for ETFs/indices")
            return
        
        # Calculate fundamental scores
        try:
            from data.fetcher import get_market_data_enhanced
            data = get_market_data_enhanced(symbol, '1y')
            
            if data is not None:
                graham_score = calculate_graham_score(symbol)
                piotroski_score = calculate_piotroski_score(symbol)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 Graham Score")
                    if isinstance(graham_score, dict) and 'error' not in graham_score:
                        score = graham_score.get('total_score', 0)
                        max_score = graham_score.get('max_score', 10)
                        st.metric("Graham Score", f"{score}/{max_score}")
                        
                        # Show criteria
                        criteria = graham_score.get('criteria', {})
                        for criterion, passed in criteria.items():
                            icon = "✅" if passed else "❌"
                            st.write(f"{icon} {criterion}")
                    else:
                        st.warning("Graham score not available")
                
                with col2:
                    st.subheader("📊 Piotroski Score")
                    if isinstance(piotroski_score, dict) and 'error' not in piotroski_score:
                        score = piotroski_score.get('total_score', 0)
                        st.metric("Piotroski Score", f"{score}/9")
                        
                        # Show criteria
                        criteria = piotroski_score.get('criteria', {})
                        for criterion, passed in criteria.items():
                            icon = "✅" if passed else "❌"
                            st.write(f"{icon} {criterion}")
                    else:
                        st.warning("Piotroski score not available")
            else:
                st.warning("Unable to fetch fundamental data")
                
        except Exception as e:
            st.error(f"Fundamental analysis error: {str(e)}")

def show_baldwin_analysis(analysis_results, show_debug=False):
    """Display Baldwin Market Regime Analysis - NEW v4.2.1"""
    if not st.session_state.show_baldwin_analysis or not BALDWIN_ANALYSIS_AVAILABLE:
        return
        
    with st.expander("🚦 Baldwin Market Regime Analysis", expanded=True):
        try:
            # Calculate Baldwin indicator
            baldwin_results = calculate_baldwin_indicator_complete(show_debug=show_debug)
            
            if baldwin_results and 'error' not in baldwin_results:
                # Format and display results
                formatted_display = format_baldwin_for_display(baldwin_results)
                st.markdown(formatted_display, unsafe_allow_html=True)
                
                if show_debug:
                    st.write("### 🐛 Baldwin Debug Information")
                    st.json(baldwin_results)
                    
            else:
                st.warning("⚠️ Baldwin analysis not available - insufficient market data")
                if show_debug and baldwin_results:
                    st.error(f"Baldwin error: {baldwin_results.get('error', 'Unknown error')}")
                    
        except Exception as e:
            st.error(f"❌ Baldwin analysis error: {str(e)}")
            if show_debug:
                st.exception(e)

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis section"""
    if not st.session_state.show_market_correlation:
        return
        
    with st.expander(f"🌐 {analysis_results['symbol']} - Market Correlation Analysis", expanded=True):
        
        symbol = analysis_results['symbol']
        
        try:
            # Calculate market correlations
            correlation_data = calculate_market_correlations_enhanced(symbol, show_debug=show_debug)
            
            if correlation_data and 'error' not in correlation_data:
                # Display correlation matrix
                st.subheader("📊 Market Correlations")
                
                correlations = correlation_data.get('correlations', {})
                if correlations:
                    # Create correlation display
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**ETF Correlations:**")
                        for etf, corr in correlations.items():
                            if corr is not None:
                                corr_strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
                                direction = "Positive" if corr > 0 else "Negative"
                                st.write(f"• {etf}: {corr:.3f} ({direction} {corr_strength})")
                    
                    with col2:
                        # Market sentiment
                        market_sentiment = correlation_data.get('market_sentiment', 'Unknown')
                        st.info(f"**Market Sentiment:** {market_sentiment}")
                        
                        # Sector strength
                        sector_strength = correlation_data.get('sector_strength', 'Unknown')
                        st.info(f"**Sector Strength:** {sector_strength}")
            
            # Breakout/Breakdown Analysis
            st.subheader("📈 Breakout/Breakdown Analysis")
            
            try:
                # Use a representative set of symbols for breakout analysis
                breakout_symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLK', 'XLE', 'XLI']
                breakout_data = calculate_breakout_breakdown_analysis(breakout_symbols, show_debug=show_debug)
                
                if breakout_data and 'error' not in breakout_data:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        breakout_pct = breakout_data.get('breakout_percentage', 0)
                        st.metric("Breakouts", f"{breakout_pct:.1f}%")
                    
                    with col2:
                        breakdown_pct = breakout_data.get('breakdown_percentage', 0)
                        st.metric("Breakdowns", f"{breakdown_pct:.1f}%")
                    
                    with col3:
                        neutral_pct = 100 - breakout_pct - breakdown_pct
                        st.metric("Neutral", f"{neutral_pct:.1f}%")
                    
                    # Market bias
                    market_bias = breakout_data.get('market_bias', 'Neutral')
                    st.info(f"**Market Bias:** {market_bias}")
                    
                    # Individual symbol results
                    if show_debug:
                        symbol_results = breakout_data.get('symbol_results', {})
                        if symbol_results:
                            st.write("### Symbol Breakdown:")
                            for sym, result in symbol_results.items():
                                st.write(f"• {sym}: {result}")
                
                else:
                    st.warning("⚠️ Breakout analysis not available")
                    
            except Exception as e:
                st.error(f"Breakout analysis error: {str(e)}")
                if show_debug:
                    st.exception(e)
            
        except Exception as e:
            st.error(f"Market correlation error: {str(e)}")
            if show_debug:
                st.exception(e)

def show_options_analysis(analysis_results, show_debug=False):
    """Display options analysis section"""
    if not st.session_state.show_options_analysis:
        return
        
    with st.expander(f"🎯 {analysis_results['symbol']} - Options Analysis", expanded=True):
        
        symbol = analysis_results['symbol']
        current_price = analysis_results['current_price']
        
        try:
            # Basic options analysis
            options_data = calculate_options_levels_enhanced(symbol, current_price)
            
            if options_data and 'error' not in options_data:
                
                # Key levels
                st.subheader("🎯 Key Options Levels")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    resistance_level = options_data.get('resistance_level')
                    if resistance_level:
                        resistance_distance = ((resistance_level - current_price) / current_price) * 100
                        st.metric("Resistance", f"${resistance_level:.2f}", f"{resistance_distance:+.1f}%")
                
                with col2:
                    support_level = options_data.get('support_level')
                    if support_level:
                        support_distance = ((support_level - current_price) / current_price) * 100
                        st.metric("Support", f"${support_level:.2f}", f"{support_distance:+.1f}%")
                
                with col3:
                    max_pain = options_data.get('max_pain')
                    if max_pain:
                        pain_distance = ((max_pain - current_price) / current_price) * 100
                        st.metric("Max Pain", f"${max_pain:.2f}", f"{pain_distance:+.1f}%")
                
                # Put/Call ratios
                st.subheader("📊 Put/Call Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    put_call_ratio = options_data.get('put_call_ratio')
                    if put_call_ratio:
                        pc_signal = "Bearish" if put_call_ratio > 1.0 else "Bullish"
                        st.metric("Put/Call Ratio", f"{put_call_ratio:.2f}", pc_signal)
                
                with col2:
                    total_volume = options_data.get('total_options_volume')
                    if total_volume:
                        st.metric("Options Volume", format_large_number(total_volume))
                
                # Advanced options analysis
                if ADVANCED_OPTIONS_AVAILABLE:
                    advanced_options = analysis_results.get('advanced_options')
                    if advanced_options and 'error' not in advanced_options:
                        st.subheader("🚀 Advanced Options Analysis")
                        
                        # Format and display advanced analysis
                        formatted_advanced = format_advanced_options_for_display(advanced_options)
                        st.markdown(formatted_advanced, unsafe_allow_html=True)
            
            else:
                st.warning("⚠️ Options data not available")
                if show_debug and options_data:
                    st.error(f"Options error: {options_data.get('error', 'Unknown error')}")
                    
        except Exception as e:
            st.error(f"Options analysis error: {str(e)}")
            if show_debug:
                st.exception(e)

def show_confidence_intervals(analysis_results, show_debug=False):
    """Display statistical confidence intervals"""
    if not st.session_state.show_confidence_intervals:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Statistical Confidence Intervals", expanded=True):
        
        symbol = analysis_results['symbol']
        current_price = analysis_results['current_price']
        
        try:
            # Get weekly deviations for confidence interval calculation
            weekly_deviations = analysis_results.get('weekly_deviations', {})
            
            if weekly_deviations:
                confidence_data = calculate_confidence_intervals(weekly_deviations, current_price)
                
                if confidence_data and 'error' not in confidence_data:
                    
                    st.subheader("📈 Weekly Price Projections")
                    
                    # Display confidence intervals
                    intervals = confidence_data.get('confidence_intervals', {})
                    
                    for confidence_level, bounds in intervals.items():
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**{confidence_level} Confidence:**")
                        
                        with col2:
                            lower_bound = bounds.get('lower')
                            if lower_bound:
                                lower_change = ((lower_bound - current_price) / current_price) * 100
                                st.metric("Lower Bound", f"${lower_bound:.2f}", f"{lower_change:+.1f}%")
                        
                        with col3:
                            upper_bound = bounds.get('upper')
                            if upper_bound:
                                upper_change = ((upper_bound - current_price) / current_price) * 100
                                st.metric("Upper Bound", f"${upper_bound:.2f}", f"{upper_change:+.1f}%")
                    
                    # Trading implications
                    st.subheader("📊 Statistical Summary")
                    
                    expected_move = confidence_data.get('expected_weekly_move')
                    volatility_percentile = confidence_data.get('volatility_percentile')
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if expected_move:
                            st.metric("Expected Weekly Move", f"±{expected_move:.1f}%")
                    
                    with col2:
                        if volatility_percentile:
                            vol_level = "High" if volatility_percentile > 75 else "Low" if volatility_percentile < 25 else "Normal"
                            st.metric("Volatility Percentile", f"{volatility_percentile:.0f}%", vol_level)
                
                else:
                    st.warning("⚠️ Confidence interval calculation failed")
                    
            else:
                st.warning("⚠️ Insufficient data for confidence intervals")
                
        except Exception as e:
            st.error(f"Confidence interval error: {str(e)}")
            if show_debug:
                st.exception(e)

def show_vwv_analysis(analysis_results, vwv_results, show_debug=False):
    """Display VWV proprietary analysis"""
    if not st.session_state.show_vwv_analysis or not VWV_CORE_AVAILABLE:
        return
        
    with st.expander(f"🎯 {analysis_results['symbol']} - VWV Proprietary Analysis", expanded=True):
        
        if vwv_results and 'error' not in vwv_results:
            
            # VWV Signal
            vwv_signal = vwv_results.get('vwv_signal', 'HOLD')
            signal_strength = vwv_results.get('signal_strength', 0)
            
            # Signal interpretation
            signal_interpretation = get_vwv_signal_interpretation(vwv_signal, signal_strength)
            
            col1, col2 = st.columns(2)
            
            with col1:
                signal_color = "🟢" if "BUY" in vwv_signal else "🔴" if "SELL" in vwv_signal else "🟡"
                st.metric("VWV Signal", f"{signal_color} {vwv_signal}", f"Strength: {signal_strength:.1f}")
            
            with col2:
                st.info(f"**Interpretation:** {signal_interpretation}")
            
            # VWV Components
            st.subheader("📊 VWV Signal Components")
            
            momentum_score = vwv_results.get('momentum_score', 50)
            trend_score = vwv_results.get('trend_score', 50)
            volume_score = vwv_results.get('volume_score', 50)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                momentum_signal = "🟢 Strong" if momentum_score > 70 else "🔴 Weak" if momentum_score < 30 else "🟡 Neutral"
                st.metric("Momentum", f"{momentum_score:.1f}", momentum_signal)
            
            with col2:
                trend_signal = "🟢 Strong" if trend_score > 70 else "🔴 Weak" if trend_score < 30 else "🟡 Neutral"
                st.metric("Trend", f"{trend_score:.1f}", trend_signal)
            
            with col3:
                volume_signal = "🟢 Strong" if volume_score > 70 else "🔴 Weak" if volume_score < 30 else "🟡 Neutral"
                st.metric("Volume", f"{volume_score:.1f}", volume_signal)
            
            # Additional VWV metrics
            if show_debug:
                st.subheader("🐛 VWV Debug Information")
                st.json(vwv_results)
        
        else:
            st.warning("⚠️ VWV analysis not available")
            if show_debug and vwv_results:
                st.error(f"VWV error: {vwv_results.get('error', 'Unknown error')}")

def main():
    """Main application function"""
    
    # Create header
    create_header()
    
    # Get sidebar controls
    controls = create_sidebar_controls()
    
    # Handle auto-analysis on quick link selection
    if st.session_state.get('analyze_on_select', False):
        st.session_state.analyze_on_select = False
        # This will trigger the analysis below
    
    symbol = controls['symbol']
    
    if symbol:
        # Add to recently viewed
        if symbol not in st.session_state.recently_viewed:
            st.session_state.recently_viewed.insert(0, symbol)
            st.session_state.recently_viewed = st.session_state.recently_viewed[:10]  # Keep last 10
        
        # Perform analysis
        with st.spinner(f"Analyzing {symbol}..."):
            analysis_results, vwv_results, chart_data = perform_complete_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['vwv_config'],
                controls['show_debug']
            )
            
            if analysis_results and chart_data is not None:
                
                # 1. CHARTS FIRST (Priority #1)
                st.markdown("---")
                display_trading_charts(chart_data, analysis_results, controls.get('show_debug', False))
                
                # 2. INDIVIDUAL TECHNICAL ANALYSIS SECOND (Priority #2)  
                st.markdown("---")
                show_technical_analysis(analysis_results, controls['show_debug'])
                
                # 3. Volume Analysis (NEW v8.0.0 - enhanced)
                st.markdown("---")
                show_volume_analysis(analysis_results, controls['show_debug'])
                
                # 4. Volatility Analysis
                st.markdown("---") 
                show_volatility_analysis(analysis_results, controls['show_debug'])
                
                # 5. Fundamental Analysis
                st.markdown("---")
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                
                # 6. Baldwin Market Regime (Before Market Correlation)
                st.markdown("---")
                show_baldwin_analysis(analysis_results, controls['show_debug'])
                
                # 7. Market Correlation Analysis (After Baldwin)
                st.markdown("---")
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                
                # 8. Options Analysis
                st.markdown("---")
                show_options_analysis(analysis_results, controls['show_debug'])
                
                # 9. Confidence Intervals
                st.markdown("---")
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
                # 10. VWV Analysis (Last)
                st.markdown("---")
                show_vwv_analysis(analysis_results, vwv_results, controls['show_debug'])
                
                # Debug information
                if controls['show_debug']:
                    with st.expander("🐛 Debug Information", expanded=False):
                        st.write("### Analysis Results Structure")
                        st.json(analysis_results, expanded=False)
                        
                        st.write("### VWV Analysis Results")
                        if vwv_results:
                            st.json(vwv_results, expanded=False)
                        else:
                            st.error("❌ VWV results not available")
                        
                        st.write("### Chart Data Information")
                        if chart_data is not None:
                            st.write(f"**Chart data shape:** {chart_data.shape}")
                            st.write(f"**Date range:** {chart_data.index[0]} to {chart_data.index[-1]}")
                            st.write(f"**Columns:** {list(chart_data.columns)}")
                            st.write(f"**Data types:** {chart_data.dtypes.to_dict()}")
                            st.write("**Sample data:**")
                            st.dataframe(chart_data.head(3))
                        else:
                            st.error("❌ Chart data is None")
                        
                        st.write("### Data Manager Summary")
                        data_manager = get_data_manager()
                        summary = data_manager.get_data_summary()
                        st.json(summary)
                        
                        st.write("### System Information")
                        import plotly
                        st.write(f"**Plotly version:** {plotly.__version__}")
                        st.write(f"**Streamlit version:** {st.__version__}")
                        
                        # Test chart creation
                        st.write("### Chart Creation Test")
                        try:
                            import plotly.graph_objects as go
                            test_fig = go.Figure()
                            test_fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
                            st.plotly_chart(test_fig, use_container_width=True)
                            st.success("✅ Chart creation successful")
                        except Exception as e:
                            st.error(f"❌ Chart creation failed: {str(e)}")
                
            else:
                st.error(f"❌ Unable to analyze {symbol}. Please check the symbol and try again.")
    
    else:
        # Welcome screen
        st.title("🚀 VWV Professional Trading System v8.0.0")
        st.markdown("### Volume Analysis Enhancement")
        
        with st.expander("📋 System Overview", expanded=True):
            st.write("**NEW v8.0.0 FEATURES:**")
            st.write("✅ **Volume Composite Score Bar** - Red-green gradient visualization")
            st.write("✅ **Enhanced Volume Intelligence** - Smart money vs retail detection")
            st.write("✅ **Nested Volume Container** - Organized detailed analysis (collapsed by default)")
            st.write("✅ **Fixed Pandas Warning** - Updated for latest pandas compatibility")
            st.write("")
            st.write("**ANALYSIS MODULES:**")
            st.write("1. **📈 Trading Charts** - Candlestick with indicators (PRIORITY FIRST)")
            st.write("2. **🎯 Individual Technical Analysis** - Professional scoring (PRIORITY SECOND)")
            st.write("3. **📊 Volume Analysis** - Enhanced with composite scoring ✨ NEW")
            st.write("4. **📊 Volatility Analysis** - Optional when module available")
            st.write("5. **📊 Fundamental Analysis** - Graham & Piotroski scores")
            st.write("6. **🚦 Baldwin Market Regime** - Before Market Correlation")
            st.write("7. **🌐 Market Correlation** - After Baldwin Indicator")
            st.write("8. **🎯 Options Analysis** - Strike levels with Greeks")
            st.write("9. **📊 Confidence Intervals** - Statistical projections")
            
            st.write("**✅ CRITICAL CORRECTIONS VERIFIED:**")
            st.write("• **Default Period:** 1 month ('1mo') - ✅ CORRECTED")
            st.write("• **Charts Priority:** Display FIRST - ✅ CORRECTED")
            st.write("• **Technical Second:** Individual analysis SECOND - ✅ CORRECTED")
            st.write("• **Baldwin Position:** Before Market Correlation - ✅ CORRECTED")
            st.write("• **Volume Enhancement:** Composite scoring + nested container - ✅ NEW")
        
        # Show current market status
        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.write("1. **Enter a symbol** in the sidebar (e.g., AAPL, SPY, QQQ)")
            st.write("2. **Default period is 1 month** - optimal for most analysis")
            st.write("3. **Charts display FIRST** - immediate visual analysis")
            st.write("4. **Technical analysis SECOND** - professional scoring with Fibonacci EMAs")
            st.write("5. **Enhanced volume analysis** - NEW composite scoring with smart money detection")
            st.write("6. **Baldwin regime indicator** - market-wide assessment")
            st.write("7. **Use Quick Links** for instant analysis of popular symbols")

    # Footer
    st.markdown("---")
    st.write("### 📊 System Information v8.0.0 - Volume Analysis Enhancement")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Version:** VWV Professional v8.0.0")
        st.write(f"**Status:** ✅ Volume Analysis Enhanced")
    with col2:
        st.write(f"**Display Order:** Charts First + Technical Second ✅")
        st.write(f"**Default Period:** 1 month ('1mo') ✅")
    with col3:
        st.write(f"**Volume Enhancement:** 🔥 Composite Score + Smart Money ✅")
        st.write(f"**Enhanced Features:** Volume Intelligence + Nested Container")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.write("Please refresh the page and try again.")
        
        if st.checkbox("Show Error Details"):
            st.exception(e)
