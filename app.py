"""
VWV Trading System v4.2.1 - Main Application
Created: 2025-08-17 05:45:00 UTC
Updated: 2025-08-17 05:45:00 UTC
Purpose: Main Streamlit application with complete session state initialization
Version: v4.2.1
CRITICAL FIX: Completed broken session state initialization that was causing startup crash
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

# Baldwin Indicator import with safe fallback
try:
    from analysis.baldwin_indicator import (
        calculate_baldwin_indicator_complete,
        format_baldwin_for_display
    )
    BALDWIN_INDICATOR_AVAILABLE = True
except ImportError:
    BALDWIN_INDICATOR_AVAILABLE = False

from ui.components import (
    create_technical_score_bar,
    create_header
)
from utils.helpers import format_large_number, get_market_status, get_etf_description
from utils.decorators import safe_calculation_wrapper

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

# Page configuration
st.set_page_config(
    page_title="VWV Professional Trading System v4.2.1",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_sidebar_controls():
    """Create sidebar controls and return analysis parameters - FIXED NAVIGATION"""
    st.sidebar.title("📊 Trading Analysis v4.2.1")
    
    # Initialize session state - FIXED: Completed broken initialization
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
    if 'show_baldwin_indicator' not in st.session_state:
        st.session_state.show_baldwin_indicator = True
    if 'show_market_correlation' not in st.session_state:
        st.session_state.show_market_correlation = True
    if 'show_options_analysis' not in st.session_state:
        st.session_state.show_options_analysis = True
    if 'show_confidence_intervals' not in st.session_state:
        st.session_state.show_confidence_intervals = True
    if 'show_charts' not in st.session_state:
        st.session_state.show_charts = True
    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = ''
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False

    # Handle symbol selection from quick links
    if st.session_state.selected_symbol:
        symbol = st.session_state.selected_symbol
        st.session_state.selected_symbol = ''  # Reset after use
    else:
        symbol = st.sidebar.text_input(
            "📈 Enter Symbol", 
            value="", 
            placeholder="e.g., AAPL, SPY, QQQ",
            help="Enter a stock symbol for analysis"
        )

    # Analysis period selection
    period = st.sidebar.selectbox(
        "📅 Analysis Period",
        options=UI_SETTINGS['periods'],
        index=UI_SETTINGS['periods'].index(UI_SETTINGS['default_period']),
        help="Select the time period for analysis (default: 1 month for optimal accuracy)"
    )

    # Analysis button
    analyze_button = st.sidebar.button("📊 Analyze Now", type="primary", use_container_width=True)

    # Debug mode toggle
    show_debug = st.sidebar.checkbox("🐛 Debug Mode", value=st.session_state.show_debug)
    st.session_state.show_debug = show_debug

    # Quick Links section - FIRST
    st.sidebar.markdown("---")
    with st.sidebar.expander("🔗 Quick Links", expanded=True):
        st.write("**Popular Symbols for Quick Analysis:**")
        
        for category, symbols in QUICK_LINK_CATEGORIES.items():
            st.write(f"**{category}:**")
            
            # Create columns for symbols
            cols = st.columns(len(symbols))
            for idx, symbol_info in enumerate(symbols):
                with cols[idx]:
                    if st.button(symbol_info['symbol'], key=f"quick_{symbol_info['symbol']}", use_container_width=True):
                        st.session_state.selected_symbol = symbol_info['symbol']
                        st.rerun()

    # Recently Viewed section - SECOND
    if len(st.session_state.recently_viewed) > 0:
        with st.sidebar.expander("🕒 Recently Viewed", expanded=False):
            st.write("**Last 9 Analyzed Symbols**")
            
            recent_symbols = st.session_state.recently_viewed[:9]
            
            for row in range(0, len(recent_symbols), 3):
                cols = st.columns(3)
                for col_idx, col in enumerate(cols):
                    symbol_idx = row + col_idx
                    if symbol_idx < len(recent_symbols):
                        recent_symbol = recent_symbols[symbol_idx]
                        with col:
                            if st.button(f"{recent_symbol}", key=f"recent_{recent_symbol}_{symbol_idx}", use_container_width=True):
                                st.session_state.selected_symbol = recent_symbol
                                st.rerun()

    # Analysis Sections Control Panel - THIRD
    with st.sidebar.expander("📋 Analysis Sections", expanded=False):
        st.write("**Toggle Analysis Sections:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.show_charts = st.checkbox(
                "📊 Interactive Charts", 
                value=st.session_state.show_charts,
                key="toggle_charts"
            )
            st.session_state.show_technical_analysis = st.checkbox(
                "Technical Analysis", 
                value=st.session_state.show_technical_analysis,
                key="toggle_technical"
            )
            if VOLUME_ANALYSIS_AVAILABLE:
                st.session_state.show_volume_analysis = st.checkbox(
                    "Volume Analysis", 
                    value=st.session_state.show_volume_analysis,
                    key="toggle_volume"
                )
            if VOLATILITY_ANALYSIS_AVAILABLE:
                st.session_state.show_volatility_analysis = st.checkbox(
                    "Volatility Analysis", 
                    value=st.session_state.show_volatility_analysis,
                    key="toggle_volatility"
                )
            st.session_state.show_fundamental_analysis = st.checkbox(
                "Fundamental Analysis", 
                value=st.session_state.show_fundamental_analysis,
                key="toggle_fundamental"
            )
        
        with col2:
            if BALDWIN_INDICATOR_AVAILABLE:
                st.session_state.show_baldwin_indicator = st.checkbox(
                    "Baldwin Market Regime", 
                    value=st.session_state.show_baldwin_indicator,
                    key="toggle_baldwin"
                )
            st.session_state.show_market_correlation = st.checkbox(
                "Market Correlation", 
                value=st.session_state.show_market_correlation,
                key="toggle_correlation"
            )
            st.session_state.show_options_analysis = st.checkbox(
                "Options Analysis", 
                value=st.session_state.show_options_analysis,
                key="toggle_options"
            )
            st.session_state.show_confidence_intervals = st.checkbox(
                "Confidence Intervals", 
                value=st.session_state.show_confidence_intervals,
                key="toggle_confidence"
            )

    return {
        'symbol': symbol.upper().strip() if symbol else '',
        'period': period,
        'analyze_button': analyze_button,
        'show_debug': show_debug
    }

def add_to_recently_viewed(symbol):
    """Add symbol to recently viewed list"""
    if symbol and symbol not in st.session_state.recently_viewed:
        st.session_state.recently_viewed.insert(0, symbol)
        st.session_state.recently_viewed = st.session_state.recently_viewed[:9]

def show_interactive_charts(data, analysis_results, show_debug=False):
    """Display interactive charts section"""
    if not st.session_state.show_charts:
        return
        
    with st.expander("📊 Interactive Trading Charts", expanded=True):
        try:
            # Check if we have the charts module
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
                else:
                    st.error("No data available for charting")
                    
        except Exception as e:
            if show_debug:
                st.error(f"Chart display error: {str(e)}")
                st.exception(e)
            else:
                st.warning("⚠️ Charts temporarily unavailable. Try refreshing or enable debug mode for details.")
                
                # Fallback simple chart
                st.subheader("Basic Price Chart (Fallback)")
                if data is not None and not data.empty:
                    st.line_chart(data['Close'])
                else:
                    st.error("No data available for charting")

def show_individual_technical_analysis(analysis_results, show_debug=False):
    """Display individual technical analysis section - MANDATORY SECOND"""
    if not st.session_state.show_technical_analysis:
        return
        
    with st.expander(f"🔴 {analysis_results['symbol']} - Individual Technical Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        
        # Technical score and signals
        composite_score = enhanced_indicators.get('composite_technical_score', 50)
        st.write(f"### Technical Composite Score: {composite_score:.1f}/100")
        
        # Create technical score bar
        score_html = create_technical_score_bar(composite_score)
        st.markdown(score_html, unsafe_allow_html=True)
        
        # Main technical indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            daily_vwap = enhanced_indicators.get('daily_vwap', {})
            vwap_price = daily_vwap.get('vwap', 0)
            current_price = analysis_results.get('current_price', 0)
            vwap_position = "Above" if current_price > vwap_price else "Below"
            st.metric("Daily VWAP", f"${vwap_price:.2f}", f"{vwap_position} current price")
        
        with col2:
            fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
            if fibonacci_emas:
                ema_21 = fibonacci_emas.get('ema_21', 0)
                st.metric("EMA 21 (Fibonacci)", f"${ema_21:.2f}")
        
        with col3:
            poc_data = enhanced_indicators.get('point_of_control', {})
            if poc_data:
                poc_price = poc_data.get('poc_price', 0)
                st.metric("Point of Control", f"${poc_price:.2f}")
        
        with col4:
            weekly_dev = enhanced_indicators.get('weekly_deviations', {})
            if weekly_dev:
                std_dev = weekly_dev.get('weekly_std_dev', 0)
                st.metric("Weekly Std Dev", f"{std_dev:.2f}%")

def show_volume_analysis(analysis_results, show_debug=False):
    """Display volume analysis section - NEW v4.2.1"""
    if not st.session_state.show_volume_analysis or not VOLUME_ANALYSIS_AVAILABLE:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Volume Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volume_analysis = enhanced_indicators.get('volume_analysis', {})
        
        if 'error' not in volume_analysis and volume_analysis:
            # Primary volume metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Volume", format_large_number(volume_analysis.get('current_volume', 0)))
            with col2:
                st.metric("5D Avg Volume", format_large_number(volume_analysis.get('volume_5d_avg', 0)))
            with col3:
                volume_ratio = volume_analysis.get('volume_ratio', 1.0)
                st.metric("Volume Ratio", f"{volume_ratio:.2f}x", f"vs 30D avg")
            with col4:
                volume_trend = volume_analysis.get('volume_5d_trend', 0)
                st.metric("5D Volume Trend", f"{volume_trend:+.2f}%")
            
            # Volume regime and implications
            st.subheader("📊 Volume Environment")
            volume_regime = volume_analysis.get('volume_regime', 'Unknown')
            trading_implications = volume_analysis.get('trading_implications', 'No implications available')
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Volume Regime:** {volume_regime}")
                st.info(f"**Volume Score:** {volume_analysis.get('volume_score', 50)}/100")
            with col2:
                st.info(f"**Trading Implications:**\n{trading_implications}")
                
        else:
            st.warning("⚠️ Volume analysis not available - insufficient data")

def show_volatility_analysis(analysis_results, show_debug=False):
    """Display volatility analysis section - NEW v4.2.1"""
    if not st.session_state.show_volatility_analysis or not VOLATILITY_ANALYSIS_AVAILABLE:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Volatility Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volatility_analysis = enhanced_indicators.get('volatility_analysis', {})
        
        if 'error' not in volatility_analysis and volatility_analysis:
            # Primary volatility metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                current_vol = volatility_analysis.get('current_volatility', 0)
                st.metric("Current Volatility", f"{current_vol:.2f}%")
            with col2:
                vol_5d_avg = volatility_analysis.get('volatility_5d_avg', 0)
                st.metric("5D Avg Volatility", f"{vol_5d_avg:.2f}%")
            with col3:
                vol_ratio = volatility_analysis.get('volatility_ratio', 1.0)
                st.metric("Volatility Ratio", f"{vol_ratio:.2f}x", f"vs 30D avg")
            with col4:
                vol_trend = volatility_analysis.get('volatility_5d_trend', 0)
                st.metric("5D Vol Trend", f"{vol_trend:+.2f}%")
            
            # Volatility regime and implications
            st.subheader("🌡️ Volatility Environment")
            vol_regime = volatility_analysis.get('volatility_regime', 'Unknown')
            trading_implications = volatility_analysis.get('trading_implications', 'No implications available')
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Volatility Regime:** {vol_regime}")
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
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        graham_score = enhanced_indicators.get('graham_score', {})
        piotroski_score = enhanced_indicators.get('piotroski_score', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Graham Score")
            if 'error' not in graham_score:
                score = graham_score.get('score', 0)
                total = graham_score.get('total_possible', 10)
                st.metric("Graham Score", f"{score}/{total}", f"{(score/total)*100:.1f}% of criteria met")
                
                criteria = graham_score.get('criteria', [])
                if criteria:
                    st.write("**Criteria Analysis:**")
                    for criterion in criteria[:5]:  # Show first 5
                        status = "✅" if criterion.get('met', False) else "❌"
                        st.write(f"{status} {criterion.get('name', 'Unknown')}")
            else:
                st.info("ℹ️ Graham analysis not available for this symbol")
        
        with col2:
            st.subheader("📊 Piotroski Score")
            if 'error' not in piotroski_score:
                score = piotroski_score.get('score', 0)
                total = piotroski_score.get('total_possible', 9)
                st.metric("Piotroski Score", f"{score}/{total}", f"{(score/total)*100:.1f}% financial strength")
                
                criteria = piotroski_score.get('criteria', [])
                if criteria:
                    st.write("**Financial Strength:**")
                    for criterion in criteria[:5]:  # Show first 5
                        status = "✅" if criterion.get('met', False) else "❌"
                        st.write(f"{status} {criterion.get('name', 'Unknown')}")
            else:
                st.info("ℹ️ Piotroski analysis not available for this symbol")

def show_baldwin_indicator(analysis_results, show_debug=False):
    """Display Baldwin Market Regime Indicator - BEFORE Market Correlation"""
    if not st.session_state.show_baldwin_indicator or not BALDWIN_INDICATOR_AVAILABLE:
        return
        
    with st.expander("🚦 Baldwin Market Regime Indicator", expanded=True):
        try:
            # Calculate Baldwin indicator
            baldwin_results = calculate_baldwin_indicator_complete(show_debug)
            
            if 'error' not in baldwin_results:
                # Display main regime
                regime_color = baldwin_results.get('regime_color', '⚪')
                market_regime = baldwin_results.get('market_regime', 'UNKNOWN')
                baldwin_score = baldwin_results.get('baldwin_score', 50)
                strategy = baldwin_results.get('strategy', 'No strategy available')
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Baldwin Score", f"{baldwin_score:.1f}/100")
                with col2:
                    st.metric("Market Regime", f"{regime_color} {market_regime}")
                with col3:
                    st.write("**Strategy:**")
                    st.write(strategy)
                
                # Component breakdown
                components = baldwin_results.get('components', {})
                if components:
                    st.subheader("📊 Component Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        momentum = components.get('momentum', {})
                        momentum_score = momentum.get('component_score', 50)
                        if momentum_score >= 70:
                            st.success(f"**Momentum (60%)**\n{momentum_score:.1f}/100")
                        elif momentum_score >= 40:
                            st.warning(f"**Momentum (60%)**\n{momentum_score:.1f}/100")
                        else:
                            st.error(f"**Momentum (60%)**\n{momentum_score:.1f}/100")
                    
                    with col2:
                        liquidity = components.get('liquidity', {})
                        liquidity_score = liquidity.get('component_score', 50)
                        if liquidity_score >= 70:
                            st.success(f"**Liquidity (25%)**\n{liquidity_score:.1f}/100")
                        elif liquidity_score >= 40:
                            st.warning(f"**Liquidity (25%)**\n{liquidity_score:.1f}/100")
                        else:
                            st.error(f"**Liquidity (25%)**\n{liquidity_score:.1f}/100")
                    
                    with col3:
                        sentiment = components.get('sentiment', {})
                        sentiment_score = sentiment.get('component_score', 50)
                        if sentiment_score >= 70:
                            st.success(f"**Sentiment (15%)**\n{sentiment_score:.1f}/100")
                        elif sentiment_score >= 40:
                            st.warning(f"**Sentiment (15%)**\n{sentiment_score:.1f}/100")
                        else:
                            st.error(f"**Sentiment (15%)**\n{sentiment_score:.1f}/100")
            else:
                st.warning("⚠️ Baldwin Indicator temporarily unavailable")
                if show_debug:
                    st.error(f"Baldwin error: {baldwin_results.get('error', 'Unknown error')}")
                    
        except Exception as e:
            st.warning("⚠️ Baldwin Indicator calculation failed")
            if show_debug:
                st.error(f"Baldwin exception: {str(e)}")

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis section - AFTER Baldwin Indicator"""
    if not st.session_state.show_market_correlation:
        return
        
    with st.expander(f"🌐 {analysis_results['symbol']} - Market Correlation Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        market_correlations = enhanced_indicators.get('market_correlations', {})
        
        if market_correlations and 'error' not in market_correlations:
            # Display correlations
            st.subheader("📊 ETF Correlations")
            
            col1, col2, col3 = st.columns(3)
            
            correlation_items = list(market_correlations.items())
            for idx, (etf, corr_data) in enumerate(correlation_items):
                with [col1, col2, col3][idx % 3]:
                    if isinstance(corr_data, dict):
                        correlation = corr_data.get('correlation', 0)
                        relationship = corr_data.get('relationship', 'Unknown')
                        
                        # Color code correlation strength
                        if abs(correlation) > 0.7:
                            color = "🔴" if correlation > 0 else "🔵"
                        elif abs(correlation) > 0.3:
                            color = "🟡"
                        else:
                            color = "⚪"
                        
                        st.metric(
                            f"{color} {etf}",
                            f"{correlation:.3f}",
                            relationship
                        )
        else:
            st.warning("⚠️ Market correlation data not available")

def show_options_analysis(analysis_results, show_debug=False):
    """Display options analysis section"""
    if not st.session_state.show_options_analysis:
        return
        
    with st.expander(f"🎯 {analysis_results['symbol']} - Options Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        options_levels = enhanced_indicators.get('options_levels', {})
        
        if options_levels and 'error' not in options_levels:
            current_price = analysis_results.get('current_price', 0)
            
            st.subheader("📊 Options Strike Levels")
            
            # Display key levels
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            
            if 'put_strikes' in options_levels:
                with col2:
                    put_strike = options_levels['put_strikes'].get('conservative', 0)
                    distance = ((current_price - put_strike) / current_price) * 100
                    st.metric("Put Strike", f"${put_strike:.2f}", f"{distance:.1f}% OTM")
            
            if 'call_strikes' in options_levels:
                with col3:
                    call_strike = options_levels['call_strikes'].get('conservative', 0)
                    distance = ((call_strike - current_price) / current_price) * 100
                    st.metric("Call Strike", f"${call_strike:.2f}", f"{distance:.1f}% OTM")
            
            if 'implied_volatility' in options_levels:
                with col4:
                    iv = options_levels['implied_volatility']
                    st.metric("Implied Volatility", f"{iv:.1f}%")
        else:
            st.warning("⚠️ Options analysis not available")

def show_confidence_intervals(analysis_results, show_debug=False):
    """Display confidence intervals section"""
    if not st.session_state.show_confidence_intervals:
        return
        
    confidence_analysis = analysis_results.get('confidence_analysis')
    if confidence_analysis:
        with st.expander("📊 Statistical Confidence Intervals", expanded=True):
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Weekly Return", f"{confidence_analysis['mean_weekly_return']:.3f}%")
            with col2:
                st.metric("Weekly Volatility", f"{confidence_analysis['weekly_volatility']:.2f}%")
            with col3:
                st.metric("Sample Size", f"{confidence_analysis['sample_size']} weeks")
            
            final_intervals_data = []
            for level, level_data in confidence_analysis['confidence_intervals'].items():
                final_intervals_data.append({
                    'Confidence Level': level,
                    'Upper Bound': f"${level_data['upper_bound']}",
                    'Lower Bound': f"${level_data['lower_bound']}",
                    'Expected Move': f"±{level_data['expected_move_pct']:.2f}%"
                })
            
            df_intervals = pd.DataFrame(final_intervals_data)
            st.dataframe(df_intervals, use_container_width=True, hide_index=True)

def perform_enhanced_analysis(symbol, period, show_debug=False):
    """Perform enhanced analysis using modular components - ENHANCED v4.2.1"""
    try:
        # Step 1: Fetch data using modular data fetcher
        market_data = get_market_data_enhanced(symbol, period, show_debug)
        
        if market_data is None:
            st.error(f"❌ Could not fetch data for {symbol}")
            return None, None
        
        # Step 2: Store data using data manager
        data_manager = get_data_manager()
        data_manager.store_market_data(symbol, market_data, show_debug)
        
        # Step 3: Get analysis copy
        analysis_input = data_manager.get_market_data_for_analysis(symbol)
        
        if analysis_input is None:
            st.error("❌ Could not prepare analysis data")
            return None, None
        
        # Step 4: Calculate enhanced indicators using modular analysis
        daily_vwap = calculate_daily_vwap(analysis_input)
        fibonacci_emas = calculate_fibonacci_emas(analysis_input)
        point_of_control = calculate_point_of_control_enhanced(analysis_input)
        weekly_deviations = calculate_weekly_deviations(analysis_input)
        comprehensive_technicals = calculate_comprehensive_technicals(analysis_input)
        
        # Step 5: Calculate Volume Analysis (NEW v4.2.1)
        volume_analysis = {}
        if VOLUME_ANALYSIS_AVAILABLE:
            try:
                volume_analysis = calculate_complete_volume_analysis(analysis_input)
                if show_debug:
                    st.write("✅ Volume analysis completed")
            except Exception as e:
                if show_debug:
                    st.write(f"❌ Volume analysis failed: {e}")
                volume_analysis = {'error': 'Volume analysis failed'}
        
        # Step 6: Calculate Volatility Analysis (NEW v4.2.1)
        volatility_analysis = {}
        if VOLATILITY_ANALYSIS_AVAILABLE:
            try:
                volatility_analysis = calculate_complete_volatility_analysis(analysis_input)
                if show_debug:
                    st.write("✅ Volatility analysis completed")
            except Exception as e:
                if show_debug:
                    st.write(f"❌ Volatility analysis failed: {e}")
                volatility_analysis = {'error': 'Volatility analysis failed'}
        
        # Step 7: Calculate market correlations
        market_correlations = calculate_market_correlations_enhanced(analysis_input, symbol, show_debug=show_debug)
        
        # Step 8: Calculate fundamental analysis (skip for ETFs)
        is_etf_symbol = is_etf(symbol)
        
        if is_etf_symbol:
            graham_score = {'score': 0, 'total_possible': 10, 'criteria': [], 'error': 'ETF - Fundamental analysis not applicable'}
            piotroski_score = {'score': 0, 'total_possible': 9, 'criteria': [], 'error': 'ETF - Fundamental analysis not applicable'}
        else:
            graham_score = calculate_graham_score(symbol, show_debug)
            piotroski_score = calculate_piotroski_score(symbol, show_debug)
        
        # Step 9: Calculate options analysis
        options_levels = calculate_options_levels_enhanced(analysis_input, symbol, show_debug)
        
        # Step 10: Calculate confidence intervals
        confidence_analysis = calculate_confidence_intervals(analysis_input, show_debug)
        
        # Step 11: Compile final results
        analysis_results = {
            'symbol': symbol,
            'current_price': float(analysis_input['Close'].iloc[-1]),
            'period': period,
            'enhanced_indicators': {
                'daily_vwap': daily_vwap,
                'fibonacci_emas': fibonacci_emas,
                'point_of_control': point_of_control,
                'weekly_deviations': weekly_deviations,
                'comprehensive_technicals': comprehensive_technicals,
                'volume_analysis': volume_analysis,  # NEW v4.2.1
                'volatility_analysis': volatility_analysis,  # NEW v4.2.1
                'market_correlations': market_correlations,
                'options_levels': options_levels,
                'graham_score': graham_score,
                'piotroski_score': piotroski_score
            },
            'confidence_analysis': confidence_analysis,
            'system_status': 'OPERATIONAL v4.2.1'
        }
        
        # Store results
        data_manager.store_analysis_results(symbol, analysis_results)
        
        # Get chart data
        chart_data = data_manager.get_market_data_for_chart(symbol)
        
        return analysis_results, chart_data
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        return None, None

def main():
    """Main application function - CORRECTED v4.2.1 with PROPER DISPLAY ORDER"""
    # Create header using modular component
    create_header()
    
    # Create sidebar and get controls
    controls = create_sidebar_controls()
    
    # Main logic flow
    if controls['analyze_button'] and controls['symbol']:
        # Add symbol to recently viewed
        add_to_recently_viewed(controls['symbol'])
        
        st.write("## 📊 VWV Trading Analysis v4.2.1 Enhanced")
        
        with st.spinner(f"Analyzing {controls['symbol']}..."):
            
            # Perform analysis using modular components
            analysis_results, chart_data = perform_enhanced_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results and chart_data is not None:
                
                # CORRECTED DISPLAY ORDER - MANDATORY SEQUENCE:
                
                # 1. CHARTS FIRST (MANDATORY)
                show_interactive_charts(chart_data, analysis_results, controls['show_debug'])
                
                # 2. INDIVIDUAL TECHNICAL ANALYSIS SECOND (MANDATORY)
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                
                # 3. Volume Analysis (Optional - when available)
                if VOLUME_ANALYSIS_AVAILABLE:
                    show_volume_analysis(analysis_results, controls['show_debug'])
                
                # 4. Volatility Analysis (Optional - when available)
                if VOLATILITY_ANALYSIS_AVAILABLE:
                    show_volatility_analysis(analysis_results, controls['show_debug'])
                
                # 5. Fundamental Analysis
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                
                # 6. Baldwin Market Regime (BEFORE Market Correlation)
                if BALDWIN_INDICATOR_AVAILABLE:
                    show_baldwin_indicator(analysis_results, controls['show_debug'])
                
                # 7. Market Correlation (AFTER Baldwin)
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                
                # 8. Options Analysis
                show_options_analysis(analysis_results, controls['show_debug'])
                
                # 9. Confidence Intervals
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
                # Debug information
                if controls['show_debug']:
                    with st.expander("🐛 Debug Information", expanded=False):
                        st.write("### Analysis Results Structure")
                        st.json(analysis_results, expanded=False)
                        st.write("### Chart Data Info")
                        st.write(f"Chart data shape: {chart_data.shape}")
                        st.write(f"Chart data columns: {list(chart_data.columns)}")
                        st.write(f"Chart data date range: {chart_data.index[0]} to {chart_data.index[-1]}")
            else:
                st.error("❌ Analysis failed. Please try a different symbol or check your connection.")
    
    else:
        # Welcome message
        st.write("## 🚀 VWV Professional Trading System v4.2.1")
        st.write("**CORRECTED VERSION:** Charts First + Technical Second + Baldwin Integration")
        
        with st.expander("🎯 Analysis Sequence - v4.2.1 CORRECTED", expanded=True):
            st.write("**✅ CORRECTED MANDATORY DISPLAY ORDER:**")
            st.write("1. **📊 Interactive Charts** - Comprehensive trading visualization")
            st.write("2. **🔴 Individual Technical Analysis** - Composite scoring + indicators")
            st.write("3. **📊 Volume Analysis** - Optional when module available")
            st.write("4. **📊 Volatility Analysis** - Optional when module available")
            st.write("5. **📊
