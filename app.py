"""
File: app_redesigned.py
Version: 2.0.0 - Navigation Redesign
VWV Research And Analysis System
Created: 2025-07-15
Updated: 2025-12-04
Purpose: Main Streamlit application with multi-page navigation
System Version: v2.0.0 - Multi-Page Navigation Architecture
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import traceback

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
    calculate_piotroski_score,
    calculate_altman_z_score,
    calculate_roic,
    calculate_key_value_metrics,
    calculate_composite_fundamental_score
)
from analysis.divergence import calculate_divergence_score
from analysis.master_score import calculate_master_score_with_agreement
from analysis.confluence import calculate_signal_confluence, create_confluence_summary
from analysis.backtest import (
    BacktestEngine,
    backtest_buy_and_hold,
    compare_strategies,
    generate_backtest_report
)
from analysis.patterns import (
    detect_all_patterns,
    calculate_pattern_score
)
from analysis.candlestick import (
    scan_all_candlestick_patterns,
    calculate_candlestick_score
)
from analysis.market import (
    calculate_market_correlations_enhanced,
    calculate_breakout_breakdown_analysis
)
from analysis.options import (
    calculate_options_levels_enhanced,
    calculate_confidence_intervals
)
from analysis.scanner import display_scanner_module

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

# Baldwin indicator import with safe fallback
try:
    from analysis.baldwin_indicator import calculate_baldwin_indicator_complete, format_baldwin_for_display
    BALDWIN_INDICATOR_AVAILABLE = True
except ImportError:
    BALDWIN_INDICATOR_AVAILABLE = False
    calculate_baldwin_indicator_complete = None
    format_baldwin_for_display = None

from ui.components import (
    create_technical_score_bar,
    create_fundamental_score_bar,
    create_master_score_bar,
    create_header,
    inject_custom_css,
    create_command_center_header
)
from utils.helpers import format_large_number, get_market_status, get_etf_description
from utils.decorators import safe_calculation_wrapper

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

# Page configuration
st.set_page_config(
    page_title="VWV Research And Analysis System v2.0.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# NAVIGATION SYSTEM
# ============================================================================

def create_navigation():
    """Create navigation menu in sidebar"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Market Sentiment"

    st.sidebar.markdown("""
    <div style="padding: 1.5rem 0 1rem 0; border-bottom: 2px solid rgba(255, 255, 255, 0.15);">
        <div style="text-align: center; font-size: 1.25rem; font-weight: 600; color: white; margin-bottom: 0.5rem;">
            VWV RESEARCH
        </div>
        <div style="text-align: center; font-size: 0.75rem; color: #9ca3af;">
            Multi-Page Analysis Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("### Navigation")

    pages = [
        "Market Sentiment",
        "Equity Research",
        "Derivative Research",
        "Scanner",
        "Strategy Backtest"
    ]

    page = st.sidebar.radio(
        "Select Page",
        pages,
        index=pages.index(st.session_state.current_page),
        label_visibility="collapsed"
    )

    if page != st.session_state.current_page:
        st.session_state.current_page = page
        st.rerun()

    st.sidebar.markdown("---")

    return page

def create_sidebar_controls():
    """Create sidebar controls and return analysis parameters - Command Center Design"""

    # COMPACT BRANDING (replaces emoji-heavy title)
    st.sidebar.markdown("""
    <div style="padding: 1rem 0; border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
        <div class="terminal-badge">
            <div class="terminal-badge-dot"></div>
            <span style="color: white; font-size: 0.875rem;">CONTROLS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for toggles
    if 'recently_viewed' not in st.session_state:
        st.session_state.recently_viewed = []
    if 'last_analyzed_symbol' not in st.session_state:
        st.session_state.last_analyzed_symbol = None
    if 'show_charts' not in st.session_state:
        st.session_state.show_charts = True
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
    if 'show_options_analysis' not in st.session_state:
        st.session_state.show_options_analysis = True
    if 'show_confidence_intervals' not in st.session_state:
        st.session_state.show_confidence_intervals = True
    if 'show_baldwin_indicator' not in st.session_state:
        st.session_state.show_baldwin_indicator = True
    if 'show_master_score' not in st.session_state:
        st.session_state.show_master_score = True
    if 'show_divergence' not in st.session_state:
        st.session_state.show_divergence = True
    if 'show_confluence' not in st.session_state:
        st.session_state.show_confluence = True
    if 'show_backtest' not in st.session_state:
        st.session_state.show_backtest = True
    if 'show_patterns' not in st.session_state:
        st.session_state.show_patterns = True
    if 'show_scanner' not in st.session_state:
        st.session_state.show_scanner = True

    # Initialize pending symbol state for quick links
    if 'pending_symbol' not in st.session_state:
        st.session_state.pending_symbol = "TSLA"

    # Symbol input with Enter key support
    symbol_input = st.sidebar.text_input(
        "Symbol",
        value=st.session_state.pending_symbol,
        help="Enter a stock symbol (e.g., AAPL, TSLA, SPY)"
    ).upper()

    # Update pending_symbol if user typed a new value
    if symbol_input and symbol_input != st.session_state.pending_symbol:
        st.session_state.pending_symbol = symbol_input

    # Data period selection with 3mo as default (optimal for all modules)
    period = st.sidebar.selectbox(
        "Data Period",
        options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=1,  # Default to 3mo (recommended for technical analysis)
        help="Select the historical data period for analysis. 3mo+ recommended for all modules."
    )

    # Analysis sections toggle - HIDDEN (redundant with page navigation)
    # Module visibility controlled by pages now

    # Analyze button - Professional command center style
    analyze_button = st.sidebar.button("RUN ANALYSIS", use_container_width=True, type="primary")
    
    # Recently viewed
    with st.sidebar.expander("RECENTLY VIEWED", expanded=False):
        if st.session_state.recently_viewed:
            for viewed_symbol in st.session_state.recently_viewed[-5:]:
                if st.button(viewed_symbol, key=f"recent_{viewed_symbol}", use_container_width=True):
                    st.session_state.symbol_input = viewed_symbol
                    st.session_state.last_analyzed_symbol = None  # Force re-analysis
                    st.rerun()
        else:
            st.write("No recent symbols")

    # Quick Links
    with st.sidebar.expander("QUICK LINKS", expanded=False):
        for category, symbols in QUICK_LINK_CATEGORIES.items():
            st.write(f"**{category}**")
            cols = st.columns(2)
            for idx, symbol in enumerate(symbols):
                with cols[idx % 2]:
                    if st.button(symbol, key=f"quick_{symbol}", use_container_width=True):
                        st.session_state.pending_symbol = symbol
                        st.session_state.last_analyzed_symbol = None  # Force re-analysis
                        st.rerun()
    
    # FOOTER SECTION (at bottom of sidebar)
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="padding: 1rem 0; display: flex; justify-content: space-between;
                align-items: center; font-size: 0.75rem; color: #6b7280;">
        <span>v1.0.0</span>
        <div style="width: 8px; height: 8px; background-color: #10b981;
                    border-radius: 50%; animation: pulse 2s infinite;"></div>
    </div>
    """, unsafe_allow_html=True)

    # Debug mode in collapsible settings
    with st.sidebar.expander("SETTINGS"):
        show_debug = st.checkbox("Debug Mode", value=False, key="debug_mode")

    return {
        'symbol': symbol_input,
        'period': period,
        'analyze_button': analyze_button,
        'show_debug': show_debug
    }

def add_to_recently_viewed(symbol):
    """Add symbol to recently viewed list"""
    if symbol not in st.session_state.recently_viewed:
        st.session_state.recently_viewed.append(symbol)
    elif symbol in st.session_state.recently_viewed:
        st.session_state.recently_viewed.remove(symbol)
        st.session_state.recently_viewed.append(symbol)

def show_interactive_charts(data, analysis_results, show_debug=False):
    """Display interactive charts section - PRIORITY 1 (FIRST)"""
    if not st.session_state.show_charts:
        return
        
    with st.expander("INTERACTIVE TRADING CHARTS", expanded=True):
        try:
            from charts.plotting import display_trading_charts
            display_trading_charts(data, analysis_results)
        except ImportError as e:
            st.error("📊 Charts module not available")
            if show_debug:
                st.error(f"Import error: {str(e)}")
            # Fallback simple chart
            st.subheader("Basic Price Chart (Fallback)")
            if data is not None and not data.empty:
                st.line_chart(data['Close'])
        except Exception as e:
            if show_debug:
                st.error(f"Chart display error: {str(e)}")
                st.exception(e)
            else:
                st.warning("⚠️ Charts temporarily unavailable")

def format_technical_signal(value, indicator_type="macd"):
    """
    Format technical indicator with correct color/arrow/label alignment

    Args:
        value: Indicator value
        indicator_type: Type of indicator ('macd', 'directional', etc.)

    Returns:
        Formatted HTML string with correct visual signal
    """
    if indicator_type == "macd":
        if value > 0.5:
            return '<span style="color: green; font-weight: bold;">↑ 🟢 Bullish</span>'
        elif value < -0.5:
            return '<span style="color: red; font-weight: bold;">↓ 🔴 Bearish</span>'
        else:
            return '<span style="color: gold; font-weight: bold;">→ 🟡 Neutral</span>'

    return '<span style="color: gray;">Unknown</span>'

def format_grade_signal(score, max_score):
    """
    Format grade with correct color/arrow/letter based on percentage

    Args:
        score: Achieved score
        max_score: Maximum possible score

    Returns:
        Formatted HTML string with grade and visual signal
    """
    percentage = (score / max_score) * 100

    if percentage >= 80:
        grade = "A"
        return f'<span style="color: darkgreen; font-weight: bold;">↑ 🟢 Grade: {grade}</span>'
    elif percentage >= 70:
        grade = "B"
        return f'<span style="color: green; font-weight: bold;">↑ 🟢 Grade: {grade}</span>'
    elif percentage >= 60:
        grade = "C"
        return f'<span style="color: gold; font-weight: bold;">→ 🟡 Grade: {grade}</span>'
    elif percentage >= 50:
        grade = "D"
        return f'<span style="color: orange; font-weight: bold;">↓ 🟠 Grade: {grade}</span>'
    else:
        grade = "F"
        return f'<span style="color: red; font-weight: bold;">↓ 🔴 Grade: {grade}</span>'

def format_directional_signal(plus_di, minus_di):
    """
    Format directional indicator signal based on +DI vs -DI comparison

    Args:
        plus_di: Plus Directional Indicator value
        minus_di: Minus Directional Indicator value

    Returns:
        Formatted HTML string with directional signal
    """
    if plus_di > minus_di:
        return '<span style="color: green; font-weight: bold;">↑ 🟢 Bullish</span>'
    elif minus_di > plus_di:
        return '<span style="color: red; font-weight: bold;">↓ 🔴 Bearish</span>'
    else:
        return '<span style="color: gold; font-weight: bold;">→ 🟡 Neutral</span>'

def show_individual_technical_analysis(analysis_results, show_debug=False):
    """
    Display individual technical analysis section - PRIORITY 2 (SECOND)
    Version: v1.0.9 - Display only fix, calculations unchanged
    """
    if not st.session_state.show_technical_analysis:
        return
    
    symbol = analysis_results.get('symbol', 'Unknown')
    period = analysis_results.get('period', 'Unknown')
    data_points = analysis_results.get('data_points', 0)

    with st.expander(f"TECHNICAL ANALYSIS - {symbol}", expanded=True):

        # Timeframe warning for insufficient data
        if data_points < 50:
            st.warning(
                f"⚠️ **Limited Data Alert:** Only {data_points} data points available. "
                f"Technical analysis requires **50+ data points** for accurate indicators. "
                f"**Use 3mo or longer period** for reliable results. "
                f"Values may show as defaults (50) with current {period} timeframe."
            )

        # Get data - ONLY changed .get() to handle missing keys, NOT calculation logic
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
        current_price = analysis_results.get('current_price', 0)
        daily_vwap = enhanced_indicators.get('daily_vwap', 0)
        point_of_control = enhanced_indicators.get('point_of_control', 0)
        
        # --- 1. COMPOSITE TECHNICAL SCORE BAR ---
        try:
            composite_score, score_details = calculate_composite_technical_score(analysis_results)
            score_bar_html = create_technical_score_bar(composite_score, score_details)
            st.components.v1.html(score_bar_html, height=160)
        except Exception as e:
            if show_debug:
                st.error(f"Score bar error: {str(e)}")
            st.metric("Composite Technical Score", "Calculating...")
        
        # --- 2. KEY MOMENTUM OSCILLATORS ---
        st.subheader("Key Momentum Oscillators")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi = comprehensive_technicals.get('rsi_14', 50)
            if rsi > 70:
                rsi_status = f"Overbought ({rsi:.1f} > 70)"
                rsi_delta = f"+{rsi - 70:.1f}"
            elif rsi < 30:
                rsi_status = f"Oversold ({rsi:.1f} < 30)"
                rsi_delta = f"{rsi - 30:.1f}"
            else:
                rsi_status = "Neutral"
                rsi_delta = None
            st.metric("RSI (14)", f"{rsi:.2f}", rsi_delta, help=rsi_status)

        with col2:
            mfi = comprehensive_technicals.get('mfi_14', 50)
            if mfi > 80:
                mfi_status = f"Overbought ({mfi:.1f} > 80)"
                mfi_delta = f"+{mfi - 80:.1f}"
            elif mfi < 20:
                mfi_status = f"Oversold ({mfi:.1f} < 20)"
                mfi_delta = f"{mfi - 20:.1f}"
            else:
                mfi_status = "Neutral"
                mfi_delta = None
            st.metric("MFI (14)", f"{mfi:.2f}", mfi_delta, help=mfi_status)

        with col3:
            stoch = comprehensive_technicals.get('stochastic', {})
            stoch_k = stoch.get('k', 50) if isinstance(stoch, dict) else 50
            if stoch_k > 80:
                stoch_status = f"Overbought ({stoch_k:.1f} > 80)"
                stoch_delta = f"+{stoch_k - 80:.1f}"
            elif stoch_k < 20:
                stoch_status = f"Oversold ({stoch_k:.1f} < 20)"
                stoch_delta = f"{stoch_k - 20:.1f}"
            else:
                stoch_status = "Neutral"
                stoch_delta = None
            st.metric("Stochastic %K", f"{stoch_k:.2f}", stoch_delta, help=stoch_status)

        with col4:
            williams_r = comprehensive_technicals.get('williams_r', -50)
            if williams_r > -20:
                williams_status = f"Overbought ({williams_r:.1f} > -20)"
                williams_delta = f"+{williams_r + 20:.1f}"
            elif williams_r < -80:
                williams_status = f"Oversold ({williams_r:.1f} < -80)"
                williams_delta = f"{williams_r + 80:.1f}"
            else:
                williams_status = "Neutral"
                williams_delta = None
            st.metric("Williams %R", f"{williams_r:.2f}", williams_delta, help=williams_status)
        
        # --- 3. TREND ANALYSIS ---
        st.subheader("Trend Analysis")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            macd_data = comprehensive_technicals.get('macd', {})
            macd_hist = macd_data.get('histogram', 0) if isinstance(macd_data, dict) else 0
            st.write("**MACD Histogram**")
            st.write(f"{macd_hist:.4f}")
            st.markdown(format_technical_signal(macd_hist, "macd"), unsafe_allow_html=True)

        with col2:
            adx_data = comprehensive_technicals.get('adx', {})
            if isinstance(adx_data, dict):
                adx_value = adx_data.get('adx', 0)
                trend_strength = adx_data.get('trend_strength', 'Unknown')
                st.metric("ADX (14)", f"{adx_value:.2f}",
                         help=f"Trend Strength: {trend_strength}")

        with col3:
            if isinstance(adx_data, dict):
                plus_di = adx_data.get('plus_di', 0)
                st.metric("+DI", f"{plus_di:.2f}",
                         help="Plus Directional Indicator (bullish movement)")

        with col4:
            if isinstance(adx_data, dict):
                minus_di = adx_data.get('minus_di', 0)
                plus_di = adx_data.get('plus_di', 0)
                st.write("**-DI**")
                st.write(f"{minus_di:.2f}")
                st.markdown(format_directional_signal(plus_di, minus_di), unsafe_allow_html=True)
                st.caption("Minus Directional Indicator (bearish movement)")
        
        # --- 4. PRICE-BASED INDICATORS & KEY LEVELS TABLE ---
        st.subheader("Price-Based Indicators & Key Levels")
        
        indicators_data = []
        
        # Current Price
        indicators_data.append({
            'Indicator': 'Current Price',
            'Value': f'${current_price:.2f}',
            'Type': '📍 Reference',
            'Distance': '0.0%',
            'Status': 'Current'
        })
        
        # Daily VWAP
        if daily_vwap and daily_vwap > 0:
            vwap_distance = ((current_price - daily_vwap) / daily_vwap * 100)
            vwap_status = "Above" if current_price > daily_vwap else "Below"
            indicators_data.append({
                'Indicator': 'Daily VWAP',
                'Value': f'${daily_vwap:.2f}',
                'Type': '📊 Volume Weighted',
                'Distance': f'{vwap_distance:+.2f}%',
                'Status': vwap_status
            })
        
        # Point of Control
        if point_of_control and point_of_control > 0:
            poc_distance = ((current_price - point_of_control) / point_of_control * 100)
            poc_status = "Above" if current_price > point_of_control else "Below"
            indicators_data.append({
                'Indicator': 'Point of Control',
                'Value': f'${point_of_control:.2f}',
                'Type': '📊 Volume Profile',
                'Distance': f'{poc_distance:+.2f}%',
                'Status': poc_status
            })
        
        # Fibonacci EMAs
        if fibonacci_emas:
            for ema_name, ema_value in fibonacci_emas.items():
                try:
                    period = str(ema_name).split('_')[1]
                    ema_val = float(ema_value) if ema_value else 0
                    
                    if ema_val > 0:
                        ema_distance = ((current_price - ema_val) / ema_val * 100)
                        ema_status = "Above" if current_price > ema_val else "Below"
                        indicators_data.append({
                            'Indicator': f'EMA {period}',
                            'Value': f'${ema_val:.2f}',
                            'Type': '📈 Trend',
                            'Distance': f'{ema_distance:+.2f}%',
                            'Status': ema_status
                        })
                except:
                    continue
        
        # Display table
        if indicators_data:
            df_technical = pd.DataFrame(indicators_data)
            st.dataframe(df_technical, use_container_width=True, hide_index=True)
        
        # Debug info
        if show_debug:
            with st.expander("TECHNICAL ANALYSIS DEBUG INFO", expanded=True):
                st.write("**Comprehensive Technicals:**")
                st.json(comprehensive_technicals)
                st.write("**All Indicators:**")
                st.json(enhanced_indicators)

def show_volume_analysis(analysis_results, show_debug=False):
    """Display volume analysis with composite score"""
    if not st.session_state.show_volume_analysis or not VOLUME_ANALYSIS_AVAILABLE:
        return

    symbol = analysis_results.get('symbol', 'Unknown')
    period = analysis_results.get('period', 'Unknown')
    data_points = analysis_results.get('data_points', 0)

    with st.expander(f"VOLUME ANALYSIS - {symbol}", expanded=True):

        # Timeframe warning for insufficient data
        if data_points < 30:
            st.warning(
                f"⚠️ **Insufficient Data:** Only {data_points} data points available. "
                f"Volume analysis requires **30+ data points** (about 1.5 months). "
                f"**Use 3mo or longer period** for accurate volume metrics with current {period} timeframe."
            )

        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volume_analysis = enhanced_indicators.get('volume_analysis', {})

        if volume_analysis and 'error' not in volume_analysis:
            # COMPOSITE SCORE BAR (NEW) - Full width for consistency
            composite_score = volume_analysis.get('composite_score', volume_analysis.get('volume_score', 50))
            score_bar_html = create_technical_score_bar(composite_score)
            st.components.v1.html(score_bar_html, height=160)

            # Classification below the score bar
            classification = volume_analysis.get('classification', volume_analysis.get('volume_regime', 'NEUTRAL'))
            st.markdown(f"**Volume Classification:** ", unsafe_allow_html=True)
            if 'STRONG' in str(classification).upper() and 'ACCUM' in str(classification).upper():
                st.success(f"**{classification}**")
            elif 'ACCUM' in str(classification).upper():
                st.info(f"**{classification}**")
            elif 'DISTRIB' in str(classification).upper() and 'STRONG' not in str(classification).upper():
                st.warning(f"**{classification}**")
            elif 'DISTRIB' in str(classification).upper():
                st.error(f"**{classification}**")
            else:
                st.info(f"**{classification}**")

            st.markdown("---")

            # Rest of existing volume analysis display
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                current_vol = volume_analysis.get('current_volume', 0)
                st.metric("Current Volume", format_large_number(current_vol))
            with col2:
                vol_5d = volume_analysis.get('volume_5d_avg', 0)
                st.metric("5D Avg Volume", format_large_number(vol_5d))
            with col3:
                vol_30d = volume_analysis.get('volume_30d_avg', 0)
                st.metric("30D Avg Volume", format_large_number(vol_30d))
            with col4:
                vol_ratio = volume_analysis.get('volume_ratio', 1.0)
                st.metric("Volume Ratio", f"{vol_ratio:.2f}x")
            with col5:
                vol_score = volume_analysis.get('volume_score', 50)
                st.metric("Volume Score", f"{vol_score:.1f}/100")

            # Second row with regime and implications
            col1, col2, col3 = st.columns(3)
            with col1:
                vol_regime = volume_analysis.get('volume_regime', 'Normal')
                st.metric("Volume Regime", vol_regime)
            with col2:
                vol_trend = volume_analysis.get('volume_trend_5d', 0)
                st.metric("5D Trend", f"{vol_trend:+.2f}%")
            with col3:
                vol_zscore = volume_analysis.get('volume_zscore', 0)
                st.metric("Z-Score", f"{vol_zscore:.2f}")

            # Trading implications
            implications = volume_analysis.get('trading_implications', 'N/A')
            st.info(f"**Trading Implications:** {implications}")
        else:
            error_msg = volume_analysis.get('error', 'insufficient data') if volume_analysis else 'insufficient data'
            st.warning(f"⚠️ Volume analysis not available - {error_msg}")

def show_volatility_analysis(analysis_results, show_debug=False):
    """Display volatility analysis with composite score"""
    if not st.session_state.show_volatility_analysis or not VOLATILITY_ANALYSIS_AVAILABLE:
        return

    symbol = analysis_results.get('symbol', 'Unknown')
    period = analysis_results.get('period', 'Unknown')
    data_points = analysis_results.get('data_points', 0)

    with st.expander(f"VOLATILITY ANALYSIS - {symbol}", expanded=True):

        # Timeframe warning for insufficient data
        if data_points < 30:
            st.warning(
                f"⚠️ **Insufficient Data:** Only {data_points} data points available. "
                f"Volatility analysis requires **30+ data points** (about 1.5 months). "
                f"**Use 3mo or longer period** for accurate volatility metrics with current {period} timeframe."
            )

        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volatility_analysis = enhanced_indicators.get('volatility_analysis', {})

        if volatility_analysis and 'error' not in volatility_analysis:
            # COMPOSITE SCORE BAR (NEW) - Full width for consistency
            composite_score = volatility_analysis.get('composite_score', volatility_analysis.get('volatility_score', 50))
            score_bar_html = create_technical_score_bar(composite_score)
            st.components.v1.html(score_bar_html, height=160)

            # Regime classification below the score bar
            regime = volatility_analysis.get('regime', volatility_analysis.get('volatility_regime', 'NORMAL'))
            st.markdown(f"**Volatility Regime:** ", unsafe_allow_html=True)
            if 'LOW' in str(regime).upper():
                st.success(f"**{regime}**")
            elif 'NORMAL' in str(regime).upper():
                st.info(f"**{regime}**")
            elif 'ELEVATED' in str(regime).upper():
                st.warning(f"**{regime}**")
            elif 'HIGH' in str(regime).upper():
                st.error(f"**{regime}**")
            else:
                st.info(f"**{regime}**")

            st.markdown("---")

            # Rest of existing volatility analysis display
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                vol_20d = volatility_analysis.get('volatility_20d', 0)
                st.metric("20D Volatility", f"{vol_20d:.2f}%")
            with col2:
                realized_vol = volatility_analysis.get('realized_volatility', 0)
                st.metric("Realized Vol", f"{realized_vol:.2f}%")
            with col3:
                vol_percentile = volatility_analysis.get('volatility_percentile', 50)
                st.metric("Vol Percentile", f"{vol_percentile:.1f}%")
            with col4:
                vol_score = volatility_analysis.get('volatility_score', 50)
                st.metric("Volatility Score", f"{vol_score:.1f}/100")
            with col5:
                vol_regime = volatility_analysis.get('volatility_regime', 'Normal')
                st.metric("Regime", vol_regime)

            # Additional metrics in a second row
            st.subheader("Trading Implications")
            col1, col2 = st.columns(2)
            with col1:
                options_strategy = volatility_analysis.get('options_strategy', 'N/A')
                st.info(f"**Options Strategy:** {options_strategy}")
            with col2:
                implications = volatility_analysis.get('trading_implications', 'N/A')
                st.info(f"**Implications:** {implications}")
        else:
            error_msg = volatility_analysis.get('error', 'insufficient data') if volatility_analysis else 'insufficient data'
            st.warning(f"⚠️ Volatility analysis not available - {error_msg}")

def show_fundamental_analysis(analysis_results, show_debug=False):
    """Display fundamental analysis section - PRIORITY 5"""
    if not st.session_state.show_fundamental_analysis:
        return

    with st.expander("FUNDAMENTAL ANALYSIS - VALUE INVESTMENT SCORES", expanded=True):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        graham_data = enhanced_indicators.get('graham_score', {})
        piotroski_data = enhanced_indicators.get('piotroski_score', {})
        altman_data = enhanced_indicators.get('altman_z_score', {})

        # Check if symbol is ETF
        is_etf_symbol = ('ETF' in str(graham_data.get('error', '')) or
                         'ETF' in str(piotroski_data.get('error', '')))

        if is_etf_symbol:
            st.info("ℹ️ Fundamental analysis not applicable for ETFs")
            return

        # --- COMPOSITE FUNDAMENTAL SCORE BAR ---
        try:
            composite_score, score_details = calculate_composite_fundamental_score(analysis_results)
            score_bar_html = create_fundamental_score_bar(composite_score, score_details)
            st.components.v1.html(score_bar_html, height=160)
        except Exception as e:
            if show_debug:
                st.error(f"Composite score error: {str(e)}")
            st.metric("Composite Fundamental Score", "Calculating...")

        # Display scores with detailed criteria - 2x2 BALANCED GRID
        # ROW 1: Graham Score and Piotroski F-Score
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Graham Score (Value Investing)")
            if 'error' not in graham_data:
                score = graham_data.get('score', 0)
                total = graham_data.get('total_possible', 10)
                grade = graham_data.get('grade', 'F')
                interpretation = graham_data.get('interpretation', 'Unknown')

                st.write(f"**Score:** {score}/{total}")
                st.markdown(format_grade_signal(score, total), unsafe_allow_html=True)
                st.caption(interpretation)

                # Show criteria details
                criteria = graham_data.get('criteria', [])
                if criteria:
                    with st.expander("DETAILED CRITERIA", expanded=False):
                        for criterion in criteria:
                            if '✓' in criterion:
                                st.success(criterion)
                            else:
                                st.error(criterion)
            else:
                st.metric("Score", "0/10")
                st.error(f"Error: {graham_data.get('error', 'Unknown error')}")

        with col2:
            st.subheader("Piotroski F-Score (Financial Health)")
            if 'error' not in piotroski_data:
                score = piotroski_data.get('score', 0)
                total = piotroski_data.get('total_possible', 9)
                grade = piotroski_data.get('grade', 'F')
                interpretation = piotroski_data.get('interpretation', 'Unknown')

                st.write(f"**Score:** {score}/{total}")
                st.markdown(format_grade_signal(score, total), unsafe_allow_html=True)
                st.caption(interpretation)

                # Show criteria details
                criteria = piotroski_data.get('criteria', [])
                if criteria:
                    with st.expander("DETAILED CRITERIA", expanded=False):
                        for criterion in criteria:
                            if '✓' in criterion:
                                st.success(criterion)
                            else:
                                st.error(criterion)
            else:
                st.metric("Score", "0/9")
                st.error(f"Error: {piotroski_data.get('error', 'Unknown error')}")

        # Visual separator between rows
        st.divider()

        # ROW 2: Altman Z-Score and ROIC
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Altman Z-Score (Bankruptcy Risk)")
            if 'error' not in altman_data:
                z_score = altman_data.get('z_score', 0)
                zone = altman_data.get('zone', 'Unknown')
                interpretation = altman_data.get('interpretation', 'Unknown')

                # Color code based on zone
                if zone == "Safe Zone":
                    zone_color = "🟢"
                elif zone == "Grey Zone":
                    zone_color = "🟡"
                else:  # Distress Zone
                    zone_color = "🔴"

                st.metric("Z-Score", f"{z_score:.2f}", delta=f"{zone_color} {zone}")
                st.caption(interpretation)

                # Show component details
                components = altman_data.get('components', {})
                criteria = altman_data.get('criteria', [])
                if criteria:
                    with st.expander("COMPONENT DETAILS", expanded=False):
                        for criterion in criteria:
                            st.text(criterion)
            else:
                st.metric("Z-Score", "0.00")
                st.error(f"Error: {altman_data.get('error', 'Unknown error')}")

        with col4:
            roic_data = enhanced_indicators.get('roic', {})
            st.subheader("ROIC (Return on Invested Capital)")
            if 'error' not in roic_data:
                roic_percent = roic_data.get('roic_percent', 0)
                grade = roic_data.get('grade', 'F')
                interpretation = roic_data.get('interpretation', 'Unknown')

                # Color code based on grade
                if grade in ['A+', 'A']:
                    grade_color = "🟢"
                elif grade == 'B':
                    grade_color = "🟡"
                elif grade == 'C':
                    grade_color = "🟠"
                else:
                    grade_color = "🔴"

                st.metric("ROIC", f"{roic_percent:.2f}%", delta=f"{grade_color} Grade: {grade}")
                st.caption(interpretation)

                # Show component details
                criteria = roic_data.get('criteria', [])
                if criteria:
                    with st.expander("CALCULATION DETAILS", expanded=False):
                        for criterion in criteria:
                            st.text(criterion)
            else:
                st.metric("ROIC", "0.00%")
                st.error(f"Error: {roic_data.get('error', 'Unknown error')}")

        # --- KEY VALUE METRICS ---
        st.divider()
        st.subheader("Key Value Metrics")

        metrics_data = enhanced_indicators.get('key_value_metrics', {})
        metrics = metrics_data.get('metrics', {})
        company_name = metrics_data.get('company_name', '')

        if company_name:
            st.caption(f"**{company_name}**")

        if not metrics:
            st.warning("⚠️ No metrics data available")
        else:
            # Display metrics in two rows
            # Row 1: P/E, P/B, D/E
            col1, col2, col3 = st.columns(3)

            with col1:
                pe_data = metrics.get('pe_ratio', {})
                st.metric(
                    "P/E Ratio",
                    pe_data.get('display', 'N/A'),
                    help="Price-to-Earnings Ratio: Compares stock price to earnings per share"
                )
                st.caption(pe_data.get('interpretation', ''))

            with col2:
                pb_data = metrics.get('pb_ratio', {})
                st.metric(
                    "P/B Ratio",
                    pb_data.get('display', 'N/A'),
                    help="Price-to-Book Ratio: Compares market price to book value"
                )
                st.caption(pb_data.get('interpretation', ''))

            with col3:
                de_data = metrics.get('de_ratio', {})
                st.metric(
                    "Debt-to-Equity",
                    de_data.get('display', 'N/A'),
                    help="Debt-to-Equity Ratio: Measures financial leverage"
                )
                st.caption(de_data.get('interpretation', ''))

            st.divider()

            # Row 2: Dividend Yield, FCF, ROE
            col1, col2, col3 = st.columns(3)

            with col1:
                div_data = metrics.get('dividend_yield', {})
                st.metric(
                    "Dividend Yield",
                    div_data.get('display', 'N/A'),
                    help="Annual dividend payments relative to stock price"
                )
                st.caption(div_data.get('interpretation', ''))

            with col2:
                fcf_data = metrics.get('free_cash_flow', {})
                st.metric(
                    "Free Cash Flow",
                    fcf_data.get('display', 'N/A'),
                    help="Cash remaining after operating expenses and capital expenditures"
                )
                st.caption(fcf_data.get('interpretation', ''))

            with col3:
                roe_data = metrics.get('roe', {})
                st.metric(
                    "Return on Equity",
                    roe_data.get('display', 'N/A'),
                    help="Profit generated from shareholders' equity"
                )
                st.caption(roe_data.get('interpretation', ''))

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis section - PRIORITY 7 (After Baldwin)"""
    if not st.session_state.show_market_correlation:
        return
        
    with st.expander("MARKET CORRELATION & COMPARISON ANALYSIS", expanded=True):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        market_correlations = enhanced_indicators.get('market_correlations', {})
        
        if market_correlations:
            st.subheader("📊 ETF Correlation Analysis")
            
            correlation_data = []
            for etf, data in market_correlations.items():
                correlation_data.append({
                    'ETF': etf,
                    'Correlation': f"{data.get('correlation', 0):.3f}",
                    'Beta': f"{data.get('beta', 0):.3f}",
                    'Relationship': data.get('relationship', 'Unknown')
                })
            
            if correlation_data:
                df_corr = pd.DataFrame(correlation_data)
                st.dataframe(df_corr, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Market correlation data not available")

def show_baldwin_indicator(show_debug=False):
    """Display Baldwin Market Regime Indicator - Positioned FIRST"""
    if not st.session_state.show_baldwin_indicator or not BALDWIN_INDICATOR_AVAILABLE:
        return

    with st.expander("BALDWIN MARKET REGIME INDICATOR", expanded=True):
        try:
            baldwin_results = calculate_baldwin_indicator_complete(show_debug=show_debug)

            if 'error' in baldwin_results:
                st.error(f"❌ Baldwin Indicator Error: {baldwin_results.get('error', 'Unknown error')}")
                return

            # Main Display Area
            col1, col2, col3 = st.columns([2, 2, 2])

            with col1:
                # Baldwin Score with gradient bar
                baldwin_score = baldwin_results.get('baldwin_score', 50)
                score_bar_html = create_technical_score_bar(baldwin_score)
                st.components.v1.html(score_bar_html, height=160)

            with col2:
                # Regime indicator
                regime = baldwin_results.get('market_regime', 'UNKNOWN')
                regime_color = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(regime, "⚪")

                st.metric(
                    label="Market Regime",
                    value=f"{regime_color} {regime}",
                    help="Current market environment classification"
                )

                # Strategy recommendation
                strategy = baldwin_results.get('strategy', '')
                if strategy:
                    if regime == 'GREEN':
                        st.success(f"🟢 {strategy}")
                    elif regime == 'RED':
                        st.error(f"🔴 {strategy}")
                    else:
                        st.warning(f"🟡 {strategy}")

            with col3:
                # Quick component summary (high-level only)
                st.write("**Component Overview:**")
                components = baldwin_results.get('components', {})
                if components:
                    for name, data in components.items():
                        component_score = data.get('component_score', 50)
                        component_name = name.replace('_', ' & ').title()
                        st.write(f"• {component_name}: {component_score:.1f}/100")
                else:
                    st.write("• No component data available")

            # NESTED EXPANDER FOR COMPONENT BREAKDOWN (collapsed by default)
            with st.expander("COMPONENT BREAKDOWN", expanded=False):
                st.write("### Detailed Component Analysis")

                components = baldwin_results.get('components', {})
                if components:
                    # Create detailed breakdown for each component
                    for name, data in components.items():
                        component_name = name.replace('_', ' & ').title()
                        component_score = data.get('component_score', 50)

                        with st.expander(f"{component_name.upper()} ({component_score:.1f}/100)", expanded=False):
                            # Display sub-components if available
                            if isinstance(data, dict):
                                for key, value in data.items():
                                    if key != 'component_score':
                                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                else:
                    st.info("No detailed component data available")

        except Exception as e:
            st.error(f"❌ Baldwin Indicator Error: {str(e)}")
            if show_debug:
                import traceback
                st.code(traceback.format_exc())

def show_options_analysis(analysis_results, show_debug=False):
    """Display enhanced options analysis section with Black-Scholes pricing - PRIORITY 8"""
    if not st.session_state.show_options_analysis:
        return

    with st.expander("OPTIONS ANALYSIS - BLACK-SCHOLES PRICING & STRIKE QUALITY", expanded=True):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        options_levels = enhanced_indicators.get('options_levels', [])
        current_price = analysis_results.get('current_price', 0)
        volatility = enhanced_indicators.get('comprehensive_technicals', {}).get('volatility', 25)

        if options_levels:
            # Display main options table
            st.subheader("📊 Options Levels with Black-Scholes Pricing")
            df_options = pd.DataFrame(options_levels)
            st.dataframe(df_options, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Add strike quality analysis (Phase 3 enhancement)
            try:
                from ui.components import display_strike_quality_table
                display_strike_quality_table(options_levels, current_price, volatility)
            except Exception as e:
                st.error(f"Strike quality analysis error: {str(e)}")
                if show_debug:
                    import traceback
                    st.error(traceback.format_exc())

            st.markdown("---")

            # Add risk/reward scatter plot (Phase 3 enhancement)
            try:
                from charts.plotting import create_risk_reward_scatter
                st.subheader("📊 Risk/Reward Visualization")

                risk_reward_chart = create_risk_reward_scatter(options_levels, current_price)
                if risk_reward_chart:
                    st.plotly_chart(risk_reward_chart, use_container_width=True)

                    with st.expander("HOW TO READ THIS CHART"):
                        st.write("""
                        **Ideal strikes are in the upper right:**
                        - **High PoP (right)**: Greater probability of profit
                        - **High Premium (top)**: Better income potential
                        - **Larger bubbles**: Longer time to expiration (more premium, more time risk)
                        - **Green zone**: "Sweet spot" strikes with optimal risk/reward (PoP > 65%)

                        **Strategy:**
                        - Target strikes with PoP > 70% for conservative approach
                        - Balance premium vs. probability based on risk tolerance
                        - Longer DTE (bigger bubbles) = more premium but more time risk
                        - Strikes in green zone offer best risk/reward balance
                        """)
            except Exception as e:
                st.error(f"Risk/reward chart error: {str(e)}")
                if show_debug:
                    import traceback
                    st.error(traceback.format_exc())
        else:
            st.warning("⚠️ Options analysis not available - insufficient data")

def show_confidence_intervals(analysis_results, show_debug=False):
    """Display confidence intervals section - PRIORITY 9"""
    if not st.session_state.show_confidence_intervals:
        return

    period = analysis_results.get('period', 'Unknown')
    data_points = analysis_results.get('data_points', 0)
    confidence_analysis = analysis_results.get('confidence_analysis')

    if confidence_analysis:
        with st.expander("STATISTICAL CONFIDENCE INTERVALS", expanded=True):

            # Timeframe warning for insufficient data
            if data_points < 100:
                st.warning(
                    f"⚠️ **Limited Statistical Confidence:** Only {data_points} data points available. "
                    f"Confidence intervals are most reliable with **100+ data points** (about 5 months). "
                    f"**Use 6mo or 1y period** for statistically significant results. "
                    f"Current {period} timeframe may produce wider intervals."
                )
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Weekly Return", f"{confidence_analysis['mean_weekly_return']:.3f}%")
            with col2:
                st.metric("Weekly Volatility", f"{confidence_analysis['weekly_volatility']:.2f}%")
            with col3:
                st.metric("Sample Size", f"{confidence_analysis['sample_size']} weeks")
            
            intervals_data = []
            for level, data in confidence_analysis['confidence_intervals'].items():
                intervals_data.append({
                    'Confidence Level': level,
                    'Upper Bound': f"${data['upper_bound']}",
                    'Lower Bound': f"${data['lower_bound']}",
                    'Expected Move': f"±{data['expected_move_pct']:.2f}%"
                })
            
            df_intervals = pd.DataFrame(intervals_data)
            st.dataframe(df_intervals, use_container_width=True, hide_index=True)

def show_master_score(analysis_results, show_debug=False):
    """
    Display Master Score - unified scoring across all analysis modules.
    Phase 1a implementation.
    """
    if not st.session_state.get('show_master_score', True):
        return

    symbol = analysis_results.get('symbol', 'Unknown')
    master_score_data = analysis_results.get('enhanced_indicators', {}).get('master_score', {})

    if not master_score_data or 'error' in master_score_data:
        return

    with st.expander(f"MASTER SCORE - {symbol}", expanded=True):
        # Main score display
        master_score = master_score_data.get('master_score', 0)
        interpretation = master_score_data.get('interpretation', 'Unknown')
        signal_strength = master_score_data.get('signal_strength', 'Unknown')

        # Create professional score bar matching technical/fundamental style
        score_bar_html = create_master_score_bar(master_score, interpretation, signal_strength)
        st.components.v1.html(score_bar_html, height=160)

        # Component breakdown
        st.subheader("Component Scores")
        components = master_score_data.get('components', {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            tech_score = components.get('technical', {}).get('raw', 0)
            st.metric("Technical", f"{tech_score:.1f}/100",
                     help=f"Weight: {components.get('technical', {}).get('weight', 0):.0%}")

        with col2:
            fund_score = components.get('fundamental', {}).get('raw', 0)
            st.metric("Fundamental", f"{fund_score:.1f}/100",
                     help=f"Weight: {components.get('fundamental', {}).get('weight', 0):.0%}")

        with col3:
            momentum_score = components.get('momentum', {}).get('raw', 0)
            st.metric("Momentum", f"{momentum_score:.1f}/100",
                     help=f"Weight: {components.get('momentum', {}).get('weight', 0):.0%}")

        with col4:
            div_score = components.get('divergence', {}).get('raw', 0)
            st.metric("Divergence", f"{div_score:+.1f}",
                     help=f"Weight: {components.get('divergence', {}).get('weight', 0):.0%}")

        # Agreement analysis
        st.subheader("Component Agreement")
        agreement = master_score_data.get('agreement', {})

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Agreement Level", agreement.get('agreement_level', 'N/A'))

        with col2:
            st.metric("Consensus", agreement.get('consensus', 'N/A'))

        with col3:
            std_dev = agreement.get('std_dev', 0)
            st.metric("Std Deviation", f"{std_dev:.1f}",
                     help="Lower = More agreement between components")

def show_divergence_analysis(analysis_results, show_debug=False):
    """
    Display Divergence Detection analysis.
    Phase 1a implementation - simplified divergence.
    """
    if not st.session_state.get('show_divergence', True):
        return

    symbol = analysis_results.get('symbol', 'Unknown')
    divergence_data = analysis_results.get('enhanced_indicators', {}).get('divergence', {})

    if not divergence_data or 'error' in divergence_data:
        return

    with st.expander(f"DIVERGENCE DETECTION - {symbol}", expanded=False):
        st.subheader("Price/Oscillator Divergence Analysis")

        # Main metrics
        score = divergence_data.get('score', 0)
        status = divergence_data.get('status', 'Unknown')
        total_div = divergence_data.get('total_divergences', 0)
        bullish_count = divergence_data.get('bullish_count', 0)
        bearish_count = divergence_data.get('bearish_count', 0)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Divergence Score", f"{score:+.1f}",
                     help="Positive = Bullish, Negative = Bearish")

        with col2:
            st.metric("Status", status)

        with col3:
            st.metric("Bullish Signals", bullish_count)

        with col4:
            st.metric("Bearish Signals", bearish_count)

        # Detected divergences
        divergences = divergence_data.get('divergences', [])

        if divergences:
            st.subheader("Detected Divergences")

            div_data = []
            for div in divergences:
                div_data.append({
                    'Type': div.get('type', 'Unknown').replace('_', ' ').title(),
                    'Oscillator': div.get('oscillator', 'Unknown').upper(),
                    'Strength': div.get('strength', 'Unknown').title(),
                    'Score': f"{div.get('score', 0):+.1f}",
                    'Description': div.get('description', 'N/A')
                })

            df_div = pd.DataFrame(div_data)
            st.dataframe(df_div, use_container_width=True, hide_index=True)
        else:
            st.info("No divergences detected in current analysis period.")

        # Debug diagnostics
        if show_debug:
            st.subheader("🔍 Divergence Debug Info")

            # Get data for diagnostics
            data_points = analysis_results.get('data_points', 0)
            from config.settings import get_momentum_divergence_config
            config = get_momentum_divergence_config()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Data Points", data_points)
                st.caption(f"Lookback: {config['lookback_period']} bars")

            with col2:
                st.metric("Peak Prominence", f"{config['peak_prominence']*100}%")
                st.caption(f"Min distance: {config['min_swing_distance']} bars")

            with col3:
                regular = divergence_data.get('regular_count', 0)
                hidden = divergence_data.get('hidden_count', 0)
                st.metric("Types Found", f"{regular}R + {hidden}H")
                st.caption("Regular + Hidden")

            # Peak detection info
            try:
                from scipy.signal import find_peaks
                import numpy as np

                # Get recent data (would need to pass this through, but for now show general info)
                st.info(f"""
                **Peak Detection Status:**
                - Using scipy.signal.find_peaks
                - Requires 2+ peaks/troughs for divergence
                - Current settings are conservative (low false positives)

                **No divergences is NORMAL when:**
                - Market trending steadily
                - No recent reversals
                - Price and oscillators in sync

                **To increase sensitivity:** Edit config/settings.py
                - Decrease `peak_prominence` (currently {config['peak_prominence']})
                - Increase `lookback_period` (currently {config['lookback_period']})
                - Decrease `min_swing_distance` (currently {config['min_swing_distance']})
                """)

            except Exception as e:
                st.warning(f"Debug info error: {e}")

        # Info box
        with st.container():
            st.markdown("""
            **About Divergence Detection:**
            - **Bullish Divergence**: Price declining but oscillators rising → Potential reversal up
            - **Bearish Divergence**: Price rising but oscillators declining → Potential reversal down
            - **Hidden Divergence**: Indicates trend continuation rather than reversal

            *Phase 1b now uses advanced peak matching with scipy for improved accuracy.*

            **💡 Tip:** Divergences are rare signals (2-4x/year on stable stocks). Zero divergences is normal!
            Try volatile symbols (COIN, MARA, SOXL) during market reversals for higher detection rates.
            """)


def show_signal_confluence(analysis_results, show_debug=False):
    """
    Display Signal Confluence Dashboard - agreement across all modules.
    Phase 1b implementation.
    """
    if not st.session_state.get('show_confluence', True):
        return

    symbol = analysis_results.get('symbol', 'Unknown')
    confluence_data = analysis_results.get('enhanced_indicators', {}).get('confluence', {})

    if not confluence_data or 'error' in confluence_data:
        return

    with st.expander(f"SIGNAL CONFLUENCE DASHBOARD - {symbol}", expanded=True):
        st.subheader("Cross-Module Signal Agreement")

        # Main metrics
        confluence_score = confluence_data.get('confluence_score', 50)
        confidence = confluence_data.get('confidence', {})
        bullish_count = confluence_data.get('bullish_modules', 0)
        bearish_count = confluence_data.get('bearish_modules', 0)
        neutral_count = confluence_data.get('neutral_modules', 0)
        total_modules = confluence_data.get('total_modules', 0)

        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Confluence Score", f"{confluence_score:.1f}/100",
                     help="Agreement level across all analysis modules")

        with col2:
            st.metric("Confidence Level", confidence.get('level', 'Unknown'),
                     help=f"Score: {confidence.get('score', 0):.1f}")

        with col3:
            st.metric("Bullish Modules", f"{bullish_count}/{total_modules}",
                     delta=f"+{bullish_count - bearish_count}" if bullish_count > bearish_count else None)

        with col4:
            st.metric("Bearish Modules", f"{bearish_count}/{total_modules}",
                     delta=f"+{bearish_count - bullish_count}" if bearish_count > bullish_count else None)

        # Visual confluence bar
        st.subheader("Signal Direction Breakdown")

        # Create a visual representation
        if total_modules > 0:
            bullish_pct = (bullish_count / total_modules) * 100
            bearish_pct = (bearish_count / total_modules) * 100
            neutral_pct = (neutral_count / total_modules) * 100

            col1, col2, col3 = st.columns([bullish_count or 1, neutral_count or 1, bearish_count or 1])

            with col1:
                st.success(f"🟢 Bullish: {bullish_pct:.0f}%")

            with col2:
                st.info(f"⚪ Neutral: {neutral_pct:.0f}%")

            with col3:
                st.error(f"🔴 Bearish: {bearish_pct:.0f}%")

        # Module signals table
        st.subheader("Module Signal Breakdown")

        signals = confluence_data.get('signals', {})
        if signals:
            signal_data = []
            for module_name, signal in signals.items():
                # Format direction with emoji
                direction = signal.get('direction', 'neutral')
                if direction == 'bullish':
                    direction_display = "🟢 Bullish"
                elif direction == 'bearish':
                    direction_display = "🔴 Bearish"
                else:
                    direction_display = "⚪ Neutral"

                signal_data.append({
                    'Module': module_name.replace('_', ' ').title(),
                    'Direction': direction_display,
                    'Strength': f"{signal.get('strength', 0):.1f}",
                    'Score': f"{signal.get('score', 0):.2f}",
                    'Description': signal.get('description', 'N/A')
                })

            df_signals = pd.DataFrame(signal_data)
            st.dataframe(df_signals, use_container_width=True, hide_index=True)

        # Conflicts section
        conflicts = confluence_data.get('conflicts', [])
        if conflicts:
            st.subheader("⚠️ Signal Conflicts Detected")

            for conflict in conflicts:
                conflict_type = conflict.get('type', 'Unknown')
                description = conflict.get('description', 'N/A')

                if conflict_type == 'directional_conflict':
                    st.warning(f"**Directional Conflict**: {description}")
                    col1, col2 = st.columns(2)

                    with col1:
                        bullish_mods = conflict.get('bullish', [])
                        st.write("**Bullish Modules:**")
                        for mod in bullish_mods:
                            st.write(f"  - {mod.replace('_', ' ').title()}")

                    with col2:
                        bearish_mods = conflict.get('bearish', [])
                        st.write("**Bearish Modules:**")
                        for mod in bearish_mods:
                            st.write(f"  - {mod.replace('_', ' ').title()}")

                elif conflict_type == 'strength_mismatch':
                    st.info(f"**Strength Mismatch**: {description}")

        # Summary text
        with st.container():
            st.markdown(f"""
            **Interpretation:**

            {create_confluence_summary(confluence_data)}

            **How to Use:**
            - **High Confluence (70+)**: Strong agreement suggests reliable signals
            - **Medium Confluence (40-70)**: Mixed signals, use additional confirmation
            - **Low Confluence (<40)**: Conflicting signals, wait for clarity
            """)


def show_backtest_analysis(analysis_results, show_debug=False):
    """
    Display Backtest Performance & Strategy Validation
    Phase 2A implementation.
    """
    if not st.session_state.get('show_backtest', True):
        return

    symbol = analysis_results.get('symbol', 'Unknown')
    hist_data = analysis_results.get('hist_data', None)

    if hist_data is None or len(hist_data) < 60:
        return  # Need sufficient data for backtesting

    with st.expander(f"STRATEGY PERFORMANCE (BACKTEST) - {symbol}", expanded=False):
        st.subheader("Historical Performance Validation")

        # Info message
        st.info("""
        **Phase 2A Backtesting** validates signal quality using historical data.
        This shows how the strategy would have performed in the past.
        Note: Past performance does not guarantee future results.
        """)

        # Backtest configuration and controls
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown("**Backtest Configuration:**")
            st.caption("Click 'Run Backtest' to validate strategy performance on historical data")

        with col2:
            run_button = st.button("🔄 Run Backtest", key=f"backtest_run_{symbol}",
                                  type="primary", use_container_width=True)
            if run_button:
                st.session_state[f'run_backtest_{symbol}'] = True
                st.session_state[f'backtest_results_{symbol}'] = None  # Clear old results

        with col3:
            if st.session_state.get(f'run_backtest_{symbol}', False):
                clear_button = st.button("🗑️ Clear Results", key=f"backtest_clear_{symbol}",
                                        use_container_width=True)
                if clear_button:
                    st.session_state[f'run_backtest_{symbol}'] = False
                    st.session_state[f'backtest_results_{symbol}'] = None
                    st.rerun()

        # Run backtest if button clicked
        if st.session_state.get(f'run_backtest_{symbol}', False):
            # Check if we have cached results
            cached_results = st.session_state.get(f'backtest_results_{symbol}')

            if cached_results is None:
                # Run backtest only if no cached results
                with st.spinner("⏳ Running backtest... This may take a moment..."):
                    try:
                        # Run Buy & Hold benchmark
                        benchmark_result = backtest_buy_and_hold(hist_data)

                        # Cache results in session state
                        st.session_state[f'backtest_results_{symbol}'] = benchmark_result

                        st.success("✅ Backtest completed successfully!")

                    except Exception as e:
                        st.error(f"❌ Error running backtest: {str(e)}")
                        if show_debug:
                            st.exception(e)
                        benchmark_result = None
            else:
                # Use cached results
                benchmark_result = cached_results
                st.info("📊 Showing cached backtest results. Click 'Clear Results' and 'Run Backtest' to refresh.")

            # Display benchmark results if available
            if benchmark_result is not None:
                st.subheader("📊 Buy & Hold Strategy (Benchmark)")

                metrics = benchmark_result['metrics']

                if 'error' not in metrics:
                    # Main metrics row
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        total_return = metrics.get('total_return', 0)
                        st.metric("Total Return",
                                 f"{total_return:+.2f}%",
                                 help="Total profit/loss percentage")

                    with col2:
                        win_rate = metrics.get('win_rate', 0)
                        st.metric("Win Rate",
                                 f"{win_rate:.1f}%",
                                 help="Percentage of winning trades")

                    with col3:
                        max_dd = metrics.get('max_drawdown_pct', 0)
                        st.metric("Max Drawdown",
                                 f"{max_dd:.2f}%",
                                 delta=f"{max_dd:.2f}%",
                                 delta_color="inverse",
                                 help="Largest peak-to-trough decline")

                    with col4:
                        sharpe = metrics.get('sharpe_ratio', 0)
                        st.metric("Sharpe Ratio",
                                 f"{sharpe:.2f}",
                                 help="Risk-adjusted return (>1.0 is good)")

                    # Additional metrics row
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        annual_return = metrics.get('annualized_return', 0)
                        st.metric("Annual Return",
                                 f"{annual_return:+.2f}%",
                                 help="Annualized return percentage")

                    with col2:
                        total_trades = metrics.get('total_trades', 0)
                        st.metric("Total Trades",
                                 f"{total_trades}",
                                 help="Number of completed trades")

                    with col3:
                        profit_factor = metrics.get('profit_factor', 0)
                        st.metric("Profit Factor",
                                 f"{profit_factor:.2f}",
                                 help="Gross profit / gross loss (>1.0 is profitable)")

                    with col4:
                        expectancy = metrics.get('expectancy', 0)
                        st.metric("Expectancy/Trade",
                                 f"{expectancy:+.2f}%",
                                 help="Average expected return per trade")

                    # Equity curve chart
                    st.subheader("📈 Equity Curve")

                    equity_curve = benchmark_result.get('equity_curve', pd.DataFrame())
                    if not equity_curve.empty:
                        import plotly.graph_objects as go

                        fig = go.Figure()

                        fig.add_trace(go.Scatter(
                            x=equity_curve['date'],
                            y=equity_curve['equity'],
                            mode='lines',
                            name='Portfolio Value',
                            line=dict(color='#1f77b4', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(31, 119, 180, 0.1)'
                        ))

                        # Add initial capital line
                        fig.add_hline(
                            y=100000,  # Initial capital
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Initial Capital",
                            annotation_position="right"
                        )

                        fig.update_layout(
                            title="Portfolio Equity Over Time",
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value ($)",
                            hovermode='x unified',
                            height=400,
                            showlegend=True
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    # Trades table
                    st.subheader("📋 Trade History")

                    trades_df = benchmark_result.get('trades', pd.DataFrame())
                    if not trades_df.empty:
                        # Format for display
                        display_df = trades_df.copy()
                        display_df['entry_date'] = pd.to_datetime(display_df['entry_date']).dt.strftime('%Y-%m-%d')
                        display_df['exit_date'] = pd.to_datetime(display_df['exit_date']).dt.strftime('%Y-%m-%d')
                        display_df['entry_price'] = display_df['entry_price'].apply(lambda x: f"${x:.2f}")
                        display_df['exit_price'] = display_df['exit_price'].apply(lambda x: f"${x:.2f}")
                        display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:+.2f}")
                        display_df['pnl_pct'] = display_df['pnl_pct'].apply(lambda x: f"{x:+.2f}%")

                        # Rename columns for display
                        display_df = display_df.rename(columns={
                            'entry_date': 'Entry Date',
                            'entry_price': 'Entry Price',
                            'exit_date': 'Exit Date',
                            'exit_price': 'Exit Price',
                            'direction': 'Direction',
                            'holding_days': 'Days Held',
                            'pnl': 'P&L ($)',
                            'pnl_pct': 'Return %',
                            'exit_reason': 'Exit Reason'
                        })

                        st.dataframe(display_df, use_container_width=True, hide_index=True)

                    # Performance summary
                    st.subheader("📝 Performance Summary")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Strengths:**")
                        strengths = []
                        if total_return > 0:
                            strengths.append(f"✅ Positive total return (+{total_return:.2f}%)")
                        if sharpe > 1.0:
                            strengths.append(f"✅ Good risk-adjusted returns (Sharpe: {sharpe:.2f})")
                        if win_rate > 50:
                            strengths.append(f"✅ Above 50% win rate ({win_rate:.1f}%)")
                        if profit_factor > 1.5:
                            strengths.append(f"✅ Strong profit factor ({profit_factor:.2f})")

                        if strengths:
                            for s in strengths:
                                st.write(s)
                        else:
                            st.write("No significant strengths detected")

                    with col2:
                        st.markdown("**Areas for Improvement:**")
                        weaknesses = []
                        if total_return < 0:
                            weaknesses.append(f"⚠️ Negative total return ({total_return:.2f}%)")
                        if sharpe < 1.0:
                            weaknesses.append(f"⚠️ Low risk-adjusted returns (Sharpe: {sharpe:.2f})")
                        if win_rate < 50:
                            weaknesses.append(f"⚠️ Below 50% win rate ({win_rate:.1f}%)")
                        if max_dd > 20:
                            weaknesses.append(f"⚠️ Large drawdown ({max_dd:.1f}%)")

                        if weaknesses:
                            for w in weaknesses:
                                st.write(w)
                        else:
                            st.write("No significant weaknesses detected")

                else:
                    st.error(f"Backtest error: {metrics['error']}")

        # Information box
        with st.container():
            st.markdown("""
            **Understanding Backtest Metrics:**

            - **Total Return**: Overall profit/loss percentage over the entire period
            - **Annualized Return**: Return adjusted to yearly basis for comparison
            - **Max Drawdown**: Largest peak-to-trough decline (risk indicator)
            - **Sharpe Ratio**: Risk-adjusted return (higher is better, >1.0 is good)
            - **Win Rate**: Percentage of profitable trades
            - **Profit Factor**: Ratio of gross profit to gross loss (>1.0 = profitable)
            - **Expectancy**: Average expected return per trade

            **Phase 2A Note:** Currently showing Buy & Hold benchmark. Future updates will add:
            - Master Score strategy backtest
            - Divergence signal strategy
            - Confluence strategy
            - Combined multi-signal strategy
            - Strategy comparison table
            """)


def show_pattern_recognition(analysis_results, show_debug=False):
    """
    Display Pattern Recognition - Chart and Candlestick Patterns
    Phase 2B implementation.
    """
    if not st.session_state.get('show_patterns', True):
        return

    symbol = analysis_results.get('symbol', 'Unknown')
    hist_data = analysis_results.get('hist_data', None)

    if hist_data is None or len(hist_data) < 20:
        return  # Need sufficient data for pattern detection

    with st.expander(f"PATTERN RECOGNITION - {symbol}", expanded=False):
        st.subheader("Chart Patterns & Candlestick Analysis")

        # Info message
        st.info("""
        **Phase 2B Pattern Recognition** detects classic chart and candlestick patterns.
        Patterns provide additional confirmation signals for entries and exits.
        """)

        try:
            # Detect chart patterns
            chart_patterns = detect_all_patterns(hist_data)
            pattern_score_data = calculate_pattern_score(hist_data)

            # Detect candlestick patterns
            candlestick_patterns = scan_all_candlestick_patterns(hist_data, lookback=5)
            candlestick_score_data = calculate_candlestick_score(hist_data)

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                chart_score = pattern_score_data.get('score', 50)
                st.metric("Chart Pattern Score",
                         f"{chart_score:.1f}/100",
                         help="0-50 Bearish, 50-100 Bullish")

            with col2:
                candle_score = candlestick_score_data.get('score', 50)
                st.metric("Candlestick Score",
                         f"{candle_score:.1f}/100",
                         help="Based on recent candlestick patterns")

            with col3:
                total_chart = chart_patterns.get('total_patterns', 0)
                total_candle = candlestick_patterns.get('total_patterns', 0)
                st.metric("Patterns Found",
                         f"{total_chart + total_candle}",
                         help=f"{total_chart} chart + {total_candle} candlestick")

            with col4:
                chart_sentiment = pattern_score_data.get('sentiment', 'neutral')
                candle_sentiment = candlestick_score_data.get('sentiment', 'neutral')
                # Combine sentiments
                if chart_sentiment == candle_sentiment:
                    combined = chart_sentiment.upper()
                else:
                    combined = "MIXED"
                st.metric("Overall Sentiment",
                         combined,
                         help="Combined pattern sentiment")

            # Chart Patterns Section
            st.subheader("📊 Chart Patterns")

            chart_pattern_list = chart_patterns.get('patterns_found', [])

            if chart_pattern_list:
                for pattern in chart_pattern_list:
                    pattern_type = pattern.get('type', 'unknown')
                    direction = pattern.get('direction', 'neutral')
                    confidence = pattern.get('confidence', 0)
                    description = pattern.get('description', 'N/A')
                    status = pattern.get('status', 'unknown')

                    # Color code by direction
                    if direction == 'bullish':
                        st.success(f"**🟢 {pattern_type.replace('_', ' ').title()}** (Confidence: {confidence}%)")
                    elif direction == 'bearish':
                        st.error(f"**🔴 {pattern_type.replace('_', ' ').title()}** (Confidence: {confidence}%)")
                    else:
                        st.info(f"**⚪ {pattern_type.replace('_', ' ').title()}** (Confidence: {confidence}%)")

                    st.caption(f"Status: {status.upper()} | {description}")

                    # Show pattern details
                    if show_debug and 'target_price' in pattern:
                        with st.expander("PATTERN DETAILS"):
                            st.json(pattern)

            else:
                st.info("No significant chart patterns detected in current period.")
                st.caption("Chart patterns (H&S, Double Top/Bottom, Triangles) are relatively rare.")

            # Candlestick Patterns Section
            st.subheader("🕯️ Candlestick Patterns")

            candle_pattern_list = candlestick_patterns.get('patterns_found', [])

            if candle_pattern_list:
                # Group by date
                pattern_df_data = []

                for pattern in candle_pattern_list:
                    pattern_name = pattern.get('name', 'unknown').replace('_', ' ').title()
                    direction = pattern.get('direction', 'neutral')
                    strength = pattern.get('strength', 'unknown')
                    reliability = pattern.get('reliability', 0)
                    description = pattern.get('description', 'N/A')
                    date = pattern.get('date', 'N/A')

                    # Direction emoji
                    if direction == 'bullish':
                        dir_emoji = "🟢"
                    elif direction == 'bearish':
                        dir_emoji = "🔴"
                    else:
                        dir_emoji = "⚪"

                    pattern_df_data.append({
                        'Date': str(date)[:10] if date != 'N/A' else 'N/A',
                        'Pattern': pattern_name,
                        'Direction': f"{dir_emoji} {direction.title()}",
                        'Strength': strength.title(),
                        'Reliability': f"{reliability}%",
                        'Description': description
                    })

                df_patterns = pd.DataFrame(pattern_df_data)
                st.dataframe(df_patterns, use_container_width=True, hide_index=True)

            else:
                st.info("No candlestick patterns detected in last 5 candles.")
                st.caption("Candlestick patterns appear and disappear as new candles form.")

            # Pattern Interpretation Guide
            with st.container():
                st.markdown("""
                **How to Use Pattern Recognition:**

                **Chart Patterns:**
                - **Head & Shoulders**: Strong reversal signal when completed
                - **Double Top/Bottom**: Reversal pattern, wait for confirmation
                - **Triangles**: Continuation patterns, prepare for breakout

                **Candlestick Patterns:**
                - **Engulfing**: Strong reversal signal
                - **Hammer/Shooting Star**: Reversal at support/resistance
                - **Doji**: Indecision, wait for next candle
                - **Morning/Evening Star**: Powerful 3-candle reversal
                - **Three Soldiers/Crows**: Strong continuation

                **Best Practices:**
                - Combine patterns with other analysis (Master Score, Divergence, Confluence)
                - Higher reliability patterns (>70%) are more trustworthy
                - Always wait for confirmation before entering trades
                - Patterns work best at key support/resistance levels
                """)

        except Exception as e:
            st.error(f"Error in pattern detection: {str(e)}")
            if show_debug:
                st.exception(e)


def perform_enhanced_analysis(symbol, period, show_debug=False):
    """
    Perform enhanced analysis - CALCULATION LOGIC UNCHANGED FROM WORKING VERSION
    Version: v1.0.9 - Only display changes, calculations preserved
    """
    try:
        # Step 1: Fetch data (UNCHANGED)
        market_data = get_market_data_enhanced(symbol, period, show_debug)
        
        if market_data is None:
            st.error(f"❌ Could not fetch data for {symbol}")
            return None, None
        
        # Step 2: Store and prepare data (UNCHANGED)
        data_manager = get_data_manager()
        data_manager.store_market_data(symbol, market_data, show_debug)
        analysis_input = data_manager.get_market_data_for_analysis(symbol)
        
        if analysis_input is None:
            st.error("❌ Could not prepare analysis data")
            return None, None
        
        # Step 3: Calculate all indicators (FIXED: Added debug output)
        if show_debug:
            st.write(f"📊 Calculating technical indicators for {len(analysis_input)} data points...")

        daily_vwap = calculate_daily_vwap(analysis_input)
        fibonacci_emas = calculate_fibonacci_emas(analysis_input)
        point_of_control = calculate_point_of_control_enhanced(analysis_input)
        weekly_deviations = calculate_weekly_deviations(analysis_input)
        comprehensive_technicals = calculate_comprehensive_technicals(analysis_input)

        # Step 3.1: Calculate divergence detection
        divergence_result = calculate_divergence_score(analysis_input, comprehensive_technicals)

        if show_debug:
            st.write(f"✓ Daily VWAP: ${daily_vwap:.2f}" if daily_vwap else "✗ Daily VWAP failed")
            st.write(f"✓ Fibonacci EMAs: {len(fibonacci_emas)} calculated")
            st.write(f"✓ Point of Control: ${point_of_control:.2f}" if point_of_control else "✗ POC failed")
            st.write(f"✓ Comprehensive technicals: {list(comprehensive_technicals.keys()) if comprehensive_technicals else 'EMPTY!'}")
            if comprehensive_technicals:
                rsi = comprehensive_technicals.get('rsi_14', 'N/A')
                st.write(f"  - RSI: {rsi}")
                macd = comprehensive_technicals.get('macd', {})
                st.write(f"  - MACD: {macd}")
            else:
                st.error("⚠️ CRITICAL: comprehensive_technicals is EMPTY - data length may be < 50")
            st.write(f"✓ Divergence Score: {divergence_result.get('score', 0)} ({divergence_result.get('status', 'Unknown')})")
        
        # Step 4: Market correlations (FIXED: pass period parameter + debug)
        market_correlations = calculate_market_correlations_enhanced(
            analysis_input, symbol, period=period, show_debug=show_debug
        )
        if show_debug:
            if market_correlations:
                st.write(f"✓ Market correlations calculated: {list(market_correlations.keys())}")
            else:
                st.warning("⚠️ Market correlations returned empty")
        
        # Step 5: Volume analysis (FIXED: Better error handling)
        volume_analysis = None
        if VOLUME_ANALYSIS_AVAILABLE:
            try:
                volume_analysis = calculate_complete_volume_analysis(analysis_input)
                if show_debug and volume_analysis:
                    st.write(f"✓ Volume analysis calculated: {list(volume_analysis.keys())}")
            except Exception as e:
                volume_analysis = {'error': f'Volume calculation failed: {str(e)}'}
                if show_debug:
                    st.warning(f"Volume analysis error: {str(e)}")
                    st.code(traceback.format_exc())
        
        # Step 6: Volatility analysis (FIXED: Better error handling)
        volatility_analysis = None
        if VOLATILITY_ANALYSIS_AVAILABLE:
            try:
                volatility_analysis = calculate_complete_volatility_analysis(analysis_input)
                if show_debug and volatility_analysis:
                    st.write(f"✓ Volatility analysis calculated: {list(volatility_analysis.keys())}")
            except Exception as e:
                volatility_analysis = {'error': f'Volatility calculation failed: {str(e)}'}
                if show_debug:
                    st.warning(f"Volatility analysis error: {str(e)}")
                    st.code(traceback.format_exc())
        
        # Step 7: Fundamental analysis (FIXED: Better error handling and debug)
        is_etf_symbol = is_etf(symbol)

        if is_etf_symbol:
            graham_score = {
                'score': 0,
                'total_possible': 10,
                'error': 'ETF - Fundamental analysis not applicable'
            }
            piotroski_score = {
                'score': 0,
                'total_possible': 9,
                'error': 'ETF - Fundamental analysis not applicable'
            }
            altman_z_score = {
                'z_score': 0,
                'zone': 'Unknown',
                'error': 'ETF - Fundamental analysis not applicable'
            }
            roic_data = {
                'roic': 0,
                'roic_percent': 0,
                'grade': 'F',
                'error': 'ETF - Fundamental analysis not applicable'
            }
            key_value_metrics = {
                'metrics': {},
                'error': 'ETF - Fundamental analysis not applicable'
            }
            if show_debug:
                st.info(f"Symbol {symbol} detected as ETF - skipping fundamental analysis")
        else:
            if show_debug:
                st.write(f"Calculating fundamental scores for {symbol}...")
            graham_score = calculate_graham_score(symbol, show_debug)
            piotroski_score = calculate_piotroski_score(symbol, show_debug)
            altman_z_score = calculate_altman_z_score(symbol, show_debug)
            roic_data = calculate_roic(symbol, show_debug)
            key_value_metrics = calculate_key_value_metrics(symbol, show_debug)
            if show_debug:
                st.write(f"✓ Graham score: {graham_score.get('score', 0)}/10")
                st.write(f"✓ Piotroski score: {piotroski_score.get('score', 0)}/9")
                st.write(f"✓ Altman Z-Score: {altman_z_score.get('z_score', 0):.2f} ({altman_z_score.get('zone', 'Unknown')})")
                st.write(f"✓ ROIC: {roic_data.get('roic_percent', 0):.2f}% (Grade: {roic_data.get('grade', 'F')})")
                st.write(f"✓ Key Value Metrics calculated: {len(key_value_metrics.get('metrics', {}))  } metrics")
                if 'error' in graham_score:
                    st.warning(f"Graham error: {graham_score['error']}")
                if 'error' in piotroski_score:
                    st.warning(f"Piotroski error: {piotroski_score['error']}")
                if 'error' in altman_z_score:
                    st.warning(f"Altman error: {altman_z_score['error']}")
                if 'error' in roic_data:
                    st.warning(f"ROIC error: {roic_data['error']}")
                if 'error' in key_value_metrics:
                    st.warning(f"Key Value Metrics error: {key_value_metrics['error']}")

        # Step 7.1: Calculate Master Score
        # Prepare component scores for master score calculation
        technical_composite_score, _ = calculate_composite_technical_score({
            'enhanced_indicators': {
                'comprehensive_technicals': comprehensive_technicals,
                'fibonacci_emas': fibonacci_emas,
                'daily_vwap': daily_vwap,
                'point_of_control': point_of_control
            },
            'current_price': float(analysis_input['Close'].iloc[-1])
        })

        fundamental_composite_score, _ = calculate_composite_fundamental_score({
            'enhanced_indicators': {
                'graham_score': graham_score,
                'piotroski_score': piotroski_score,
                'altman_z_score': altman_z_score,
                'roic': roic_data,
                'key_value_metrics': key_value_metrics
            }
        })

        # Calculate momentum score from oscillators
        rsi = comprehensive_technicals.get('rsi_14', 50)
        mfi = comprehensive_technicals.get('mfi_14', 50)
        stoch_k = comprehensive_technicals.get('stochastic', {}).get('k', 50) if isinstance(comprehensive_technicals.get('stochastic'), dict) else 50
        williams = comprehensive_technicals.get('williams_r', -50)
        momentum_score = (rsi + mfi + stoch_k + (williams + 100)) / 4  # Normalize to 0-100

        # Prepare master score inputs
        master_score_inputs = {
            'technical_score': technical_composite_score,
            'fundamental_score': fundamental_composite_score,
            'vwv_signal': 0,  # Will be integrated in future phase
            'momentum_score': momentum_score,
            'divergence_score': divergence_result.get('score', 0),
            'volume_score': 0,  # Will be integrated from volume analysis
            'volatility_score': 0  # Will be integrated from volatility analysis
        }

        # Calculate master score with agreement analysis
        master_score_result = calculate_master_score_with_agreement(master_score_inputs)

        if show_debug:
            st.write(f"✓ Master Score: {master_score_result.get('master_score', 0):.1f}/100")
            st.write(f"  - Interpretation: {master_score_result.get('interpretation', 'Unknown')}")
            st.write(f"  - Signal Strength: {master_score_result.get('signal_strength', 'Unknown')}")
            st.write(f"  - Agreement: {master_score_result.get('agreement', {}).get('agreement_level', 'Unknown')}")

        # Step 8: Options levels (UNCHANGED)
        current_price = round(float(analysis_input['Close'].iloc[-1]), 2)
        volatility = comprehensive_technicals.get('volatility_20d', 20)
        underlying_beta = 1.0
        
        if market_correlations:
            for etf in ['SPY', 'QQQ', 'MAGS']:
                if etf in market_correlations and 'beta' in market_correlations[etf]:
                    try:
                        underlying_beta = abs(float(market_correlations[etf]['beta']))
                        break
                    except:
                        continue
        
        options_levels = calculate_options_levels_enhanced(
            current_price, volatility, underlying_beta=underlying_beta
        )
        
        # Step 9: Confidence intervals (UNCHANGED)
        confidence_analysis = calculate_confidence_intervals(analysis_input)
        
        # Step 10: Build results (UNCHANGED)
        current_date = analysis_input.index[-1].strftime('%Y-%m-%d')
        
        analysis_results = {
            'symbol': symbol,
            'timestamp': current_date,
            'current_price': current_price,
            'period': period,  # Store period for timeframe validation in displays
            'data_points': len(analysis_input),  # Store actual data points count
            'hist_data': market_data,  # Historical OHLCV data for backtesting and patterns
            'enhanced_indicators': {
                'daily_vwap': daily_vwap,
                'fibonacci_emas': fibonacci_emas,
                'point_of_control': point_of_control,
                'weekly_deviations': weekly_deviations,
                'comprehensive_technicals': comprehensive_technicals,
                'market_correlations': market_correlations,
                'options_levels': options_levels,
                'graham_score': graham_score,
                'piotroski_score': piotroski_score,
                'altman_z_score': altman_z_score,
                'roic': roic_data,
                'key_value_metrics': key_value_metrics,
                'divergence': divergence_result,
                'master_score': master_score_result
            },
            'confidence_analysis': confidence_analysis,
            'system_status': 'OPERATIONAL v4.2.2'
        }
        
        # Add optional analyses if available (UNCHANGED)
        if volume_analysis:
            analysis_results['enhanced_indicators']['volume_analysis'] = volume_analysis

        if volatility_analysis:
            analysis_results['enhanced_indicators']['volatility_analysis'] = volatility_analysis

        # Calculate signal confluence across all modules
        confluence_result = calculate_signal_confluence(analysis_results)
        analysis_results['enhanced_indicators']['confluence'] = confluence_result

        if show_debug:
            st.write(f"✓ Signal Confluence: {confluence_result.get('confluence_score', 0):.1f}/100")
            st.write(f"  - Confidence: {confluence_result.get('confidence', {}).get('level', 'Unknown')}")
            st.write(f"  - Modules: {confluence_result.get('bullish_modules', 0)} Bullish, {confluence_result.get('bearish_modules', 0)} Bearish")

        # Store results (UNCHANGED)
        data_manager.store_analysis_results(symbol, analysis_results)
        
        # Get chart data (UNCHANGED)
        chart_data = data_manager.get_market_data_for_chart(symbol)
        
        return analysis_results, chart_data
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        if show_debug:
            st.code(traceback.format_exc())
        return None, None

# ============================================================================
# PAGE RENDER FUNCTIONS
# ============================================================================

def render_market_sentiment_page(show_debug=False):
    """
    Page 1: Market Sentiment (Landing Page)
    - Baldwin Market Regime Indicator
    - SPY Chart with Volume + Volatility
    - QQQ Chart with Volume + Volatility
    - IWM Chart with Volume + Volatility

    This page auto-loads these 3 ETFs without requiring symbol search
    """
    st.write("## Market Sentiment")
    st.write("**Overall market analysis across major indices**")
    st.markdown("---")

    # Baldwin Market Regime Indicator (shows overall market regime)
    if BALDWIN_INDICATOR_AVAILABLE:
        show_baldwin_indicator(show_debug)
        st.markdown("---")

    # Analyze each major ETF: SPY, QQQ, IWM
    market_etfs = ['SPY', 'QQQ', 'IWM']

    for etf_symbol in market_etfs:
        st.write(f"### {etf_symbol} Analysis")

        # Analyze this ETF
        with st.spinner(f"Analyzing {etf_symbol}..."):
            etf_results, etf_chart_data = perform_enhanced_analysis(
                etf_symbol,
                '3mo',  # Use 3 month period for market sentiment
                show_debug
            )

        if etf_results and etf_chart_data is not None:
            # Show chart
            show_interactive_charts(etf_chart_data, etf_results, show_debug)

            # Show Volume Analysis below chart
            if VOLUME_ANALYSIS_AVAILABLE:
                show_volume_analysis(etf_results, show_debug)

            # Show Volatility Analysis below volume
            if VOLATILITY_ANALYSIS_AVAILABLE:
                show_volatility_analysis(etf_results, show_debug)
        else:
            st.error(f"Failed to analyze {etf_symbol}")

        st.markdown("---")


def render_equity_research_page(analysis_results, chart_data, show_debug=False):
    """
    Page 2: Equity Research
    - Chart (for searched symbol)
    - Master Score
    - Technical Analysis
    - Volume Analysis
    - Volatility Analysis
    - Fundamental Analysis
    - Market Correlation and Comparison Analysis
    - Divergence Detection
    - Pattern Detection
    - Signal Confluence Dashboard
    """
    st.write("## Equity Research")
    st.write("**Comprehensive equity analysis for searched symbol**")
    st.markdown("---")

    # Chart for searched symbol
    if st.session_state.get('show_charts', True):
        show_interactive_charts(chart_data, analysis_results, show_debug)

    # Master Score
    if st.session_state.get('show_master_score', True):
        show_master_score(analysis_results, show_debug)

    # Technical Analysis
    if st.session_state.get('show_technical_analysis', True):
        show_individual_technical_analysis(analysis_results, show_debug)

    # Volume Analysis
    if VOLUME_ANALYSIS_AVAILABLE and st.session_state.get('show_volume_analysis', True):
        show_volume_analysis(analysis_results, show_debug)

    # Volatility Analysis
    if VOLATILITY_ANALYSIS_AVAILABLE and st.session_state.get('show_volatility_analysis', True):
        show_volatility_analysis(analysis_results, show_debug)

    # Fundamental Analysis
    if st.session_state.get('show_fundamental_analysis', True):
        show_fundamental_analysis(analysis_results, show_debug)

    # Market Correlation and Comparison Analysis
    if st.session_state.get('show_market_correlation', True):
        show_market_correlation_analysis(analysis_results, show_debug)

    # Divergence Detection
    if st.session_state.get('show_divergence', True):
        show_divergence_analysis(analysis_results, show_debug)

    # Pattern Detection
    if st.session_state.get('show_patterns', True):
        show_pattern_recognition(analysis_results, show_debug)

    # Signal Confluence Dashboard
    if st.session_state.get('show_confluence', True):
        show_signal_confluence(analysis_results, show_debug)


def render_derivative_research_page(analysis_results, chart_data, show_debug=False):
    """
    Page 3: Derivative Research
    - Chart (defaulted to Options Levels tab)
    - Options Analysis
    """
    st.write("## Derivative Research")
    st.write("**Options analysis and pricing for searched symbol**")
    st.markdown("---")

    # Chart (will show with Options Levels tab available)
    if st.session_state.get('show_charts', True):
        show_interactive_charts(chart_data, analysis_results, show_debug)

    # Options Analysis
    if st.session_state.get('show_options_analysis', True):
        show_options_analysis(analysis_results, show_debug)


def render_scanner_page(show_debug=False):
    """
    Page 4: Scanner
    - Multi Symbol Master Score Scanner only
    """
    st.write("## Scanner")
    st.write("**Multi-symbol master score scanner**")
    st.markdown("---")

    # Multi-Symbol Scanner
    if st.session_state.get('show_scanner', True):
        display_scanner_module(show_debug)


def render_strategy_backtest_page(analysis_results, chart_data, show_debug=False):
    """
    Page 5: Strategy Backtest
    - Strategy Performance Backtest only
    """
    st.write("## Strategy Backtest")
    st.write("**Strategy performance backtesting for searched symbol**")
    st.markdown("---")

    # Backtest Analysis
    if st.session_state.get('show_backtest', True):
        show_backtest_analysis(analysis_results, show_debug)


def main():
    """Main application function - Multi-Page Navigation Design"""

    # CRITICAL: Inject custom CSS first
    inject_custom_css()

    # Create command center header (replaces old green header)
    create_command_center_header()

    # Create navigation (returns selected page)
    current_page = create_navigation()

    # Create sidebar controls
    controls = create_sidebar_controls()

    # Initialize session state for analysis results persistence
    if 'cached_analysis_results' not in st.session_state:
        st.session_state.cached_analysis_results = None
    if 'cached_chart_data' not in st.session_state:
        st.session_state.cached_chart_data = None

    # Trigger analysis on button click OR when Enter is pressed (symbol changes)
    symbol_changed = (controls['symbol'] and
                     controls['symbol'] != st.session_state.last_analyzed_symbol)
    should_analyze = (controls['analyze_button'] or symbol_changed) and controls['symbol']

    if should_analyze:
        st.session_state.last_analyzed_symbol = controls['symbol']
        add_to_recently_viewed(controls['symbol'])

        st.write(f"## 📊 VWV Research And Analysis v1.0.0 - {controls['symbol']}")

        with st.spinner(f"Analyzing {controls['symbol']}..."):
            analysis_results, chart_data = perform_enhanced_analysis(
                controls['symbol'],
                controls['period'],
                controls['show_debug']
            )

            # Cache results in session state for persistence across reruns
            if analysis_results and chart_data is not None:
                st.session_state.cached_analysis_results = analysis_results
                st.session_state.cached_chart_data = chart_data

    # Route to appropriate page based on navigation
    # Market Sentiment page doesn't need symbol search - always shows SPY/QQQ/IWM
    if current_page == "Market Sentiment":
        render_market_sentiment_page(controls['show_debug'])

    # Scanner page doesn't need specific symbol
    elif current_page == "Scanner":
        render_scanner_page(controls['show_debug'])

    # Other pages require analysis results for searched symbol
    elif (st.session_state.cached_analysis_results is not None and
          st.session_state.cached_chart_data is not None):
        analysis_results = st.session_state.cached_analysis_results
        chart_data = st.session_state.cached_chart_data

        # Show symbol info in header
        symbol = analysis_results.get('symbol', 'Unknown')
        current_price = analysis_results.get('current_price', 0)
        change = analysis_results.get('change_pct', 0)

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"### {symbol}")
        with col2:
            st.metric("Price", f"${current_price:.2f}")
        with col3:
            st.metric("Change", f"{change:+.2f}%")

        st.markdown("---")

        # Route to pages that need analysis results
        if current_page == "Equity Research":
            render_equity_research_page(analysis_results, chart_data, controls['show_debug'])

        elif current_page == "Derivative Research":
            render_derivative_research_page(analysis_results, chart_data, controls['show_debug'])

        elif current_page == "Strategy Backtest":
            render_strategy_backtest_page(analysis_results, chart_data, controls['show_debug'])

        # Debug information (available on all pages)
        if controls['show_debug']:
            with st.expander("DEBUG INFORMATION", expanded=False):
                st.write("### Analysis Results Structure")
                st.json(analysis_results)

    # Show prompt to run analysis for pages that need it
    elif current_page in ["Equity Research", "Derivative Research", "Strategy Backtest"]:
        st.info(f"Please enter a symbol and click 'RUN ANALYSIS' to view {current_page} data.")
    else:
        # Welcome screen (no symbol analyzed yet)
        st.write("## VWV Research And Analysis System v2.0.0")
        st.write("**Multi-Page Navigation • Advanced Technical Analysis • Professional Market Research**")

        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")

        with st.expander("QUICK START GUIDE", expanded=True):
            st.write("### Getting Started")
            st.write("1. **Click 'Market Sentiment'** to see SPY/QQQ/IWM market overview (no symbol needed)")
            st.write("2. **For individual stocks:** Enter symbol and click 'RUN ANALYSIS'")
            st.write("3. **Navigate pages** using the sidebar menu")
            st.write("4. **Period:** 3 months default (optimal for all modules)")
            st.markdown("---")
            st.write("### Page Navigation")
            st.write("- **Market Sentiment** (Landing) - Baldwin + SPY/QQQ/IWM with Volume/Volatility")
            st.write("- **Equity Research** - Complete analysis for searched symbol")
            st.write("- **Derivative Research** - Options analysis for searched symbol")
            st.write("- **Scanner** - Multi-symbol master score scanner")
            st.write("- **Strategy Backtest** - Strategy performance testing")
            st.markdown("---")
            st.write("### Tips")
            st.write("- **Market Sentiment** always visible - no symbol search needed")
            st.write("- Use **Quick Links** for instant symbol analysis")
            st.write("- **Equity Research** has most analysis modules")
            st.write("- Analysis results persist when switching pages")

    # Footer
    st.markdown("---")
    st.write("### System Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(f"**Version:** v2.0.0 - Multi-Page Navigation")
        st.write(f"**Status:** Fully Operational")
    with col2:
        st.write(f"**Architecture:** 5-Page Navigation System")
        st.write(f"**Default Period:** 3 months (3mo)")
    with col3:
        st.write(f"**File:** app_redesigned.py v2.0.0")
        st.write(f"**Platform:** Research And Analysis System")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.code(traceback.format_exc())
