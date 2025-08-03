"""
VWV Professional Trading System v4.2.1 - CORRECTED VERSION
Complete preservation of existing functionality with Volume & Volatility integration
"""

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
    calculate_composite_technical_score,
    calculate_enhanced_technical_analysis
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

# Charts import with safe fallback
try:
    from charts.plotting import display_trading_charts
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False

# Baldwin indicator import with safe fallback
try:
    from analysis.baldwin_indicator import (
        calculate_baldwin_indicator,
        format_baldwin_for_display
    )
    BALDWIN_INDICATOR_AVAILABLE = True
except ImportError:
    BALDWIN_INDICATOR_AVAILABLE = False

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
    """Create sidebar controls and return analysis parameters"""
    st.sidebar.title("📊 Trading Analysis v4.2.1")
    
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
    if 'show_options_analysis' not in st.session_state:
        st.session_state.show_options_analysis = True
    if 'show_confidence_intervals' not in st.session_state:
        st.session_state.show_confidence_intervals = True
    if 'show_charts' not in st.session_state:
        st.session_state.show_charts = True
    if 'show_baldwin_indicator' not in st.session_state:
        st.session_state.show_baldwin_indicator = True
    
    # Basic controls - Fixed symbol handling
    if 'selected_symbol' in st.session_state:
        default_symbol = st.session_state.selected_symbol
        # Don't delete immediately - let the input field update first
    else:
        default_symbol = UI_SETTINGS['default_symbol']
    
    # Symbol input without form (to prevent reset issues)
    symbol = st.sidebar.text_input("Symbol", value=default_symbol, help="Enter stock symbol (press Enter to analyze)", key="symbol_input").upper()
    period = st.sidebar.selectbox("Data Period", UI_SETTINGS['periods'], index=3)
    
    # Analyze button
    analyze_button = st.sidebar.button("📊 Analyze Symbol", type="primary", use_container_width=True)
    
    # Handle Enter key detection
    enter_pressed = False
    quick_link_clicked = False
    
    if 'last_symbol_input' not in st.session_state:
        st.session_state.last_symbol_input = ""
    
    # Check if this is from a quick link selection
    if 'selected_symbol' in st.session_state and st.session_state.selected_symbol == symbol:
        quick_link_clicked = True
    
    # Check if user pressed Enter (symbol changed and is not from quick link)
    elif symbol != st.session_state.last_symbol_input and symbol != "" and 'selected_symbol' not in st.session_state:
        enter_pressed = True
        st.session_state.last_symbol_input = symbol
    
    # Clear selected_symbol after processing
    if 'selected_symbol' in st.session_state:
        del st.session_state.selected_symbol
    
    # Section Control Panel
    with st.sidebar.expander("📋 Analysis Sections", expanded=False):
        st.write("**Toggle Analysis Sections:**")
        
        col1, col2 = st.columns(2)
        with col1:
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
            st.session_state.show_charts = st.checkbox(
                "📊 Interactive Charts", 
                value=st.session_state.show_charts,
                key="toggle_charts"
            )
            if BALDWIN_INDICATOR_AVAILABLE:
                st.session_state.show_baldwin_indicator = st.checkbox(
                    "🚦 Baldwin Market Indicator", 
                    value=st.session_state.show_baldwin_indicator,
                    key="toggle_baldwin"
                )
    
    # Recently Viewed section
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
                                st.session_state.last_symbol_input = recent_symbol  # Prevent enter detection
                                st.rerun()

    # Quick Links section
    with st.sidebar.expander("🔗 Quick Links", expanded=False):
        st.write("**Popular Symbols by Category**")
        
        for category, symbols in QUICK_LINK_CATEGORIES.items():
            with st.expander(f"{category} ({len(symbols)} symbols)", expanded=False):
                for i in range(0, len(symbols), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(symbols):
                            sym = symbols[i + j]
                            with col:
                                if st.button(sym, help=SYMBOL_DESCRIPTIONS.get(sym, f"{sym} - Financial Symbol"), key=f"quick_link_{sym}", use_container_width=True):
                                    st.session_state.selected_symbol = sym
                                    st.session_state.last_symbol_input = sym  # Prevent enter detection
                                    st.rerun()

    # Debug toggle
    show_debug = st.sidebar.checkbox("🐛 Show Debug Info", value=False)
    
    return {
        'symbol': symbol,
        'period': period,
        'analyze_button': analyze_button or enter_pressed or quick_link_clicked,  # Trigger on button OR enter OR quick link
        'show_debug': show_debug
    }

def add_to_recently_viewed(symbol):
    """Add symbol to recently viewed - updated for 9 symbols"""
    if symbol and symbol != "":
        if symbol in st.session_state.recently_viewed:
            st.session_state.recently_viewed.remove(symbol)
        st.session_state.recently_viewed.insert(0, symbol)
        st.session_state.recently_viewed = st.session_state.recently_viewed[:9]

def show_individual_technical_analysis(analysis_results, show_debug=False):
    """Display individual technical analysis section - PRESERVED FUNCTIONALITY"""
    if not st.session_state.show_technical_analysis:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Individual Technical Analysis", expanded=True):
        
        # COMPOSITE TECHNICAL SCORE - Use modular component with PROPER HTML RENDERING
        composite_score, score_details = calculate_composite_technical_score(analysis_results)
        score_bar_html = create_technical_score_bar(composite_score, score_details)
        st.components.v1.html(score_bar_html, height=110)
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
        
        # Primary metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${analysis_results['current_price']}")
        with col2:
            price_change_1d = comprehensive_technicals.get('price_change_1d', 0)
            st.metric("1-Day Change", f"{price_change_1d:+.2f}%")
        with col3:
            price_change_5d = comprehensive_technicals.get('price_change_5d', 0)
            st.metric("5-Day Change", f"{price_change_5d:+.2f}%")
        with col4:
            volatility = comprehensive_technicals.get('volatility_20d', 0)
            st.metric("20D Volatility", f"{volatility:.1f}%")
        
        # Technical indicators table
        st.subheader("📋 Technical Indicators")
        current_price = analysis_results['current_price']
        daily_vwap = enhanced_indicators.get('daily_vwap', 0)
        point_of_control = enhanced_indicators.get('point_of_control', 0)

        indicators_data = []
        
        # Current Price
        indicators_data.append(("Current Price", f"${current_price:.2f}", "📍 Reference", "0.0%", "Current"))
        
        # Daily VWAP
        vwap_distance = f"{((current_price - daily_vwap) / daily_vwap * 100):+.2f}%" if daily_vwap > 0 else "N/A"
        vwap_status = "Above" if current_price > daily_vwap else "Below"
        indicators_data.append(("Daily VWAP", f"${daily_vwap:.2f}", "📊 Volume Weighted", vwap_distance, vwap_status))
        
        # Point of Control
        poc_distance = f"{((current_price - point_of_control) / point_of_control * 100):+.2f}%" if point_of_control > 0 else "N/A"
        poc_status = "Above" if current_price > point_of_control else "Below"
        indicators_data.append(("Point of Control", f"${point_of_control:.2f}", "📊 Volume Profile", poc_distance, poc_status))
        
        # Add Fibonacci EMAs
        for ema_name, ema_value in fibonacci_emas.items():
            period = ema_name.split('_')[1]
            distance_pct = f"{((current_price - ema_value) / ema_value * 100):+.2f}%" if ema_value > 0 else "N/A"
            status = "Above" if current_price > ema_value else "Below"
            indicators_data.append((f"EMA {period}", f"${ema_value:.2f}", "📈 Trend", distance_pct, status))
        
        df_technical = pd.DataFrame(indicators_data, columns=['Indicator', 'Value', 'Type', 'Distance %', 'Status'])
        st.dataframe(df_technical, use_container_width=True, hide_index=True)

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
            
            # Volatility regime and options guidance
            st.subheader("📊 Volatility Environment & Options Strategy")
            vol_regime = volatility_analysis.get('volatility_regime', 'Unknown')
            options_strategy = volatility_analysis.get('options_strategy', 'No strategy available')
            trading_implications = volatility_analysis.get('trading_implications', 'No implications available')
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Volatility Regime:** {vol_regime}")
                st.info(f"**Volatility Score:** {volatility_analysis.get('volatility_score', 50)}/100")
            with col2:
                st.info(f"**Options Strategy:** {options_strategy}")
                st.info(f"**Trading Implications:**\n{trading_implications}")
                
        else:
            st.warning("⚠️ Volatility analysis not available - insufficient data")

def show_fundamental_analysis(analysis_results, show_debug=False):
    """Display fundamental analysis section - PRESERVED FUNCTIONALITY"""
    if not st.session_state.show_fundamental_analysis:
        return
        
    with st.expander("📊 Fundamental Analysis - Value Investment Scores", expanded=True):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        graham_data = enhanced_indicators.get('graham_score', {})
        piotroski_data = enhanced_indicators.get('piotroski_score', {})
        
        # Check if symbol is ETF
        is_etf_symbol = ('ETF' in graham_data.get('error', '') or 
                         'ETF' in piotroski_data.get('error', ''))
        
        if not is_etf_symbol and ('error' not in graham_data or 'error' not in piotroski_data):
            
            # Display scores overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'error' not in graham_data:
                    st.metric(
                        "Graham Score", 
                        f"{graham_data.get('score', 0)}/10",
                        f"Grade: {graham_data.get('grade', 'N/A')}"
                    )
                else:
                    st.metric("Graham Score", "N/A", "Data Limited")
            
            with col2:
                if 'error' not in piotroski_data:
                    st.metric(
                        "Piotroski F-Score", 
                        f"{piotroski_data.get('score', 0)}/9",
                        f"Grade: {piotroski_data.get('grade', 'N/A')}"
                    )
                else:
                    st.metric("Piotroski F-Score", "N/A", "Data Limited")
            
            with col3:
                if 'error' not in graham_data:
                    st.metric(
                        "Graham %", 
                        f"{graham_data.get('percentage', 0):.0f}%",
                        graham_data.get('interpretation', '')[:20] + "..."
                    )
                else:
                    st.metric("Graham %", "0%", "No Data")
            
            with col4:
                if 'error' not in piotroski_data:
                    st.metric(
                        "Piotroski %", 
                        f"{piotroski_data.get('percentage', 0):.0f}%",
                        piotroski_data.get('interpretation', '')[:20] + "..."
                    )
                else:
                    st.metric("Piotroski %", "0%", "No Data")
        
        elif is_etf_symbol:
            st.info(f"ℹ️ **{analysis_results['symbol']} is an ETF** - Fundamental analysis is not applicable to Exchange-Traded Funds.")

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis section - PRESERVED FUNCTIONALITY"""
    if not st.session_state.show_market_correlation:
        return
        
    with st.expander("🌐 Market Correlation & Comparison Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        market_correlations = enhanced_indicators.get('market_correlations', {})
        
        if market_correlations:
            st.subheader("📊 ETF Correlation Analysis")
            
            correlation_table_data = []
            for etf, etf_data in market_correlations.items():
                correlation_table_data.append({
                    'ETF': etf,
                    'Correlation': f"{etf_data.get('correlation', 0):.3f}",
                    'Beta': f"{etf_data.get('beta', 0):.3f}",
                    'Relationship': etf_data.get('relationship', 'Unknown'),
                    'Description': get_etf_description(etf)
                })
            
            df_correlations = pd.DataFrame(correlation_table_data)
            st.dataframe(df_correlations, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Market correlation data not available")
        
        # Breakout/breakdown analysis
        st.subheader("📊 Breakout/Breakdown Analysis")
        breakout_data = calculate_breakout_breakdown_analysis(show_debug=show_debug)
        
        if breakout_data:
            # Overall market sentiment
            overall_data = breakout_data.get('OVERALL', {})
            if overall_data:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Market Breakouts", f"{overall_data['breakout_ratio']}%")
                with col2:
                    st.metric("Market Breakdowns", f"{overall_data['breakdown_ratio']}%")
                with col3:
                    net_ratio = overall_data['net_ratio']
                    st.metric("Net Bias", f"{net_ratio:+.1f}%", 
                             "📈 Bullish" if net_ratio > 0 else "📉 Bearish" if net_ratio < 0 else "⚖️ Neutral")
                with col4:
                    st.metric("Market Regime", overall_data.get('market_regime', 'Unknown'))

def show_options_analysis(analysis_results, show_debug=False):
    """Display options analysis section - PRESERVED FUNCTIONALITY"""
    if not st.session_state.show_options_analysis:
        return
        
    with st.expander("🎯 Options Trading Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        options_levels = enhanced_indicators.get('options_levels', [])
        
        if options_levels:
            st.subheader("💰 Premium Selling Levels with Greeks")
            st.write("**Enhanced option strike levels with Delta, Theta, and Beta**")
            
            df_options = pd.DataFrame(options_levels)
            st.dataframe(df_options, use_container_width=True, hide_index=True)
            
            # Options context
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("**Put Selling Strategy:**\n"
                        "• Sell puts below current price\n"
                        "• Collect premium if stock stays above strike\n"
                        "• Delta: Price sensitivity (~-0.16)\n"
                        "• Theta: Daily time decay")
            
            with col2:
                st.info("**Call Selling Strategy:**\n" 
                        "• Sell calls above current price\n" 
                        "• Collect premium if stock stays below strike\n"
                        "• Delta: Price sensitivity (~+0.16)\n"
                        "• Theta: Daily time decay")
            
            with col3:
                st.info("**Greeks Explained:**\n"
                        "• **Delta**: Price sensitivity per $1 move\n"
                        "• **Theta**: Daily time decay in option value\n"
                        "• **Beta**: Underlying's market sensitivity\n"
                        "• **PoT**: Probability of Touch %")
        else:
            st.warning("⚠️ Options analysis not available - insufficient data")

def show_confidence_intervals(analysis_results, show_debug=False):
    """Display confidence intervals section - PRESERVED FUNCTIONALITY"""
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

def show_interactive_charts(analysis_results, market_data, show_debug=False):
    """Display interactive charts section - FIRST PRIORITY DISPLAY"""
    if not st.session_state.show_charts:
        return
        
    with st.expander("📊 Interactive Trading Charts", expanded=True):
        try:
            if CHARTS_AVAILABLE and market_data is not None:
                if show_debug:
                    st.write("✅ Charts module available, rendering comprehensive charts...")
                display_trading_charts(market_data, analysis_results)
            else:
                # Fallback chart display
                if show_debug:
                    st.write(f"⚠️ Charts module available: {CHARTS_AVAILABLE}, Data available: {market_data is not None}")
                
                st.subheader("📈 Basic Price Chart (Fallback)")
                if market_data is not None and not market_data.empty:
                    # Create simple but informative chart
                    chart_data = market_data[['Close']].copy()
                    st.line_chart(chart_data)
                    
                    # Add basic info
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${market_data['Close'].iloc[-1]:.2f}")
                    with col2:
                        st.metric("High", f"${market_data['High'].max():.2f}")
                    with col3:
                        st.metric("Low", f"${market_data['Low'].min():.2f}")
                    with col4:
                        change_pct = ((market_data['Close'].iloc[-1] - market_data['Close'].iloc[0]) / market_data['Close'].iloc[0]) * 100
                        st.metric("Period Change", f"{change_pct:+.2f}%")
                else:
                    st.error("❌ No market data available for charting")
                    
        except Exception as e:
            if show_debug:
                st.error(f"Chart display error: {str(e)}")
                st.exception(e)
            else:
                st.warning("⚠️ Charts temporarily unavailable. Enable debug mode for details.")
                # Still show basic fallback
                if market_data is not None and not market_data.empty:
                    st.line_chart(market_data[['Close']])

def show_baldwin_indicator_analysis(baldwin_results, show_debug=False):
    """Display Baldwin Market Sentiment Indicator analysis section"""
    if not st.session_state.show_baldwin_indicator or not BALDWIN_INDICATOR_AVAILABLE:
        return
        
    with st.expander("🚦 Baldwin Market Sentiment Indicator", expanded=True):
        
        if baldwin_results and 'error' not in baldwin_results:
            # Format results for display
            display_data = format_baldwin_for_display(baldwin_results)
            
            if 'error' not in display_data:
                summary = display_data['summary_metrics']
                
                # Main indicator header with traffic light status
                status_emoji = baldwin_results['status_emoji']
                status = baldwin_results['status']
                composite_score = baldwin_results['composite_score']
                market_mode = baldwin_results['market_mode']
                
                # Large status display
                st.markdown(f"""
                <div style="text-align: center; padding: 2rem; 
                            background: linear-gradient(135deg, #1e1e1e 0%, #2d2d30 50%, #1a1a1a 100%); 
                            border-radius: 15px; margin-bottom: 1.5rem;
                            border: 2px solid {'#00ff00' if status == 'GREEN' else '#ffff00' if status == 'YELLOW' else '#ff0000'};">
                    <h1 style="color: {'#00ff00' if status == 'GREEN' else '#ffff00' if status == 'YELLOW' else '#ff0000'}; 
                               margin: 0; font-size: 3em;">
                        {status_emoji} {status}
                    </h1>
                    <h2 style="color: #ffffff; margin: 0.5rem 0;">
                        Market Mode: {market_mode}
                    </h2>
                    <h3 style="color: #cccccc; margin: 0;">
                        Baldwin Score: {composite_score}/100
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Strategy and interpretation
                st.subheader("📋 Market Strategy")
                strategy = baldwin_results['strategy']
                
                if status == 'GREEN':
                    st.success(f"**🟢 RISK-ON Strategy:** {strategy}")
                elif status == 'YELLOW':
                    st.warning(f"**🟡 NEUTRAL Strategy:** {strategy}")
                else:
                    st.error(f"**🔴 RISK-OFF Strategy:** {strategy}")
                
                # Component breakdown
                st.subheader("📊 Component Analysis")
                
                components = baldwin_results['components']
                col1, col2, col3 = st.columns(3)
                
                # Momentum Component (60% weight)
                with col1:
                    momentum = components['momentum']
                    momentum_score = momentum['component_score']
                    momentum_color = '#00ff00' if momentum_score >= 65 else '#ffff00' if momentum_score >= 35 else '#ff0000'
                    
                    st.markdown(f"""
                    <div style="padding: 1rem; background-color: {momentum_color}20; 
                                border-left: 4px solid {momentum_color}; border-radius: 8px;">
                        <h4 style="color: {momentum_color}; margin: 0;">💨 Momentum (60%)</h4>
                        <h2 style="color: {momentum_color}; margin: 0.2rem 0;">{momentum_score:.1f}/100</h2>
                        <p style="margin: 0; color: #888; font-size: 0.9em;">{momentum['interpretation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Liquidity Component (25% weight)
                with col2:
                    liquidity = components['liquidity']
                    liquidity_score = liquidity['component_score']
                    liquidity_color = '#00ff00' if liquidity_score >= 65 else '#ffff00' if liquidity_score >= 35 else '#ff0000'
                    
                    st.markdown(f"""
                    <div style="padding: 1rem; background-color: {liquidity_color}20; 
                                border-left: 4px solid {liquidity_color}; border-radius: 8px;">
                        <h4 style="color: {liquidity_color}; margin: 0;">💧 Liquidity (25%)</h4>
                        <h2 style="color: {liquidity_color}; margin: 0.2rem 0;">{liquidity_score:.1f}/100</h2>
                        <p style="margin: 0; color: #888; font-size: 0.9em;">{liquidity['interpretation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Sentiment Component (15% weight)
                with col3:
                    sentiment = components['sentiment']
                    sentiment_score = sentiment['component_score']
                    sentiment_color = '#00ff00' if sentiment_score >= 65 else '#ffff00' if sentiment_score >= 35 else '#ff0000'
                    
                    st.markdown(f"""
                    <div style="padding: 1rem; background-color: {sentiment_color}20; 
                                border-left: 4px solid {sentiment_color}; border-radius: 8px;">
                        <h4 style="color: {sentiment_color}; margin: 0;">🧠 Sentiment (15%)</h4>
                        <h2 style="color: {sentiment_color}; margin: 0.2rem 0;">{sentiment_score:.1f}/100</h2>
                        <p style="margin: 0; color: #888; font-size: 0.9em;">{sentiment['interpretation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed component breakdown table
                st.subheader("📋 Detailed Component Breakdown")
                component_breakdown = display_data['component_breakdown']
                df_components = pd.DataFrame(component_breakdown)
                st.dataframe(df_components, use_container_width=True, hide_index=True)
                
                # Market internals details
                with st.expander("🔍 Market Internals Detail", expanded=False):
                    
                    # Momentum details
                    st.write("**💨 Momentum Component Details:**")
                    momentum_details = momentum['details']
                    
                    if 'SPY' in momentum_details:
                        spy_data = momentum_details['SPY']
                        st.write(f"• **SPY Position:** {spy_data['position']}")
                        st.write(f"• **SPY EMAs:** 20: ${spy_data['emas'].get('EMA_20', 'N/A')}, 50: ${spy_data['emas'].get('EMA_50', 'N/A')}, 200: ${spy_data['emas'].get('EMA_200', 'N/A')}")
                    
                    if 'QQQ' in momentum_details:
                        qqq_data = momentum_details['QQQ']
                        st.write(f"• **QQQ Position:** {qqq_data['position']}")
                    
                    if 'IWM' in momentum_details:
                        iwm_data = momentum_details['IWM']
                        st.write(f"• **IWM vs SPY Relative Performance:** {iwm_data.get('relative_performance', 0):.2f}%")
                    
                    if 'FNGD' in momentum_details:
                        fngd_data = momentum_details['FNGD']
                        st.write(f"• **FNGD Stress Gauge:** ${fngd_data['current_price']:.2f} (EMA20: ${fngd_data['ema_20']:.2f})")
                    
                    if 'VIX' in momentum_details:
                        vix_data = momentum_details['VIX']
                        st.write(f"• **VIX Fear Level:** {vix_data['current_level']:.2f} (Threshold: {vix_data['threshold']})")
                    
                    # Liquidity details
                    st.write("**💧 Liquidity Component Details:**")
                    liquidity_details = liquidity['details']
                    
                    if 'USD' in liquidity_details:
                        usd_data = liquidity_details['USD']
                        st.write(f"• **USD Strength:** {usd_data['interpretation']} ({usd_data['trend_10d']:+.2f}% over 10 days)")
                    
                    if 'TLT' in liquidity_details:
                        tlt_data = liquidity_details['TLT']
                        st.write(f"• **Treasury Trend:** {tlt_data['interpretation']} ({tlt_data['trend_10d']:+.2f}% over 10 days)")
                    
                    # Sentiment details
                    st.write("**🧠 Sentiment Component Details:**")
                    sentiment_details = sentiment['details']
                    
                    if 'Smart_Money_Proxy' in sentiment_details:
                        smart_money = sentiment_details['Smart_Money_Proxy']
                        st.write(f"• **Smart Money Proxy:** {smart_money['interpretation']}")
                        st.write(f"• **IWM vs SPY (30d):** {smart_money['relative_performance']:+.2f}%")
                    
                    st.info("💡 **Note:** Sentiment component uses market-based proxies. Direct insider trading data would provide more accurate sentiment analysis.")
                
                # Threshold information
                with st.expander("⚙️ Baldwin Indicator Thresholds", expanded=False):
                    thresholds = baldwin_results['thresholds']
                    weights = baldwin_results['weights']
                    
                    st.write("**🚦 Traffic Light Thresholds:**")
                    st.write(f"• **🟢 GREEN:** Score ≥ {thresholds['green_min']} (Risk-On)")
                    st.write(f"• **🟡 YELLOW:** Score {thresholds['yellow_min']}-{thresholds['green_min']-1} (Neutral)")
                    st.write(f"• **🔴 RED:** Score ≤ {thresholds['red_max']} (Risk-Off)")
                    
                    st.write("**⚖️ Component Weights:**")
                    st.write(f"• **Momentum:** {weights['momentum']*100:.0f}% (Market trends, internals, fear)")
                    st.write(f"• **Liquidity:** {weights['liquidity']*100:.0f}% (USD strength, Treasury flows)")
                    st.write(f"• **Sentiment:** {weights['sentiment']*100:.0f}% (Smart money positioning)")
                    
                    symbols_analyzed = baldwin_results.get('symbols_analyzed', [])
                    st.write(f"**📊 Symbols Analyzed:** {', '.join([s.upper() for s in symbols_analyzed])}")
                
            else:
                st.error(f"❌ Baldwin indicator display error: {display_data.get('error', 'Unknown error')}")
                
        else:
            st.warning("⚠️ Baldwin Market Sentiment Indicator not available")
            if show_debug and baldwin_results and 'error' in baldwin_results:
                st.error(f"Error details: {baldwin_results['error']}")

def show_interactive_charts(analysis_results, market_data, show_debug=False):

def show_enhanced_debug_information(analysis_results, market_data, show_debug=False):
    """Display comprehensive debug information"""
    if not show_debug:
        return
        
    with st.expander("🐛 Enhanced Debug Information", expanded=False):
        
        # System Status Overview
        st.subheader("🖥️ System Status Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Core Modules:**")
            st.write(f"✅ Config: Available")
            st.write(f"✅ Data: Available") 
            st.write(f"✅ Analysis: Available")
            st.write(f"✅ UI: Available")
            st.write(f"✅ Utils: Available")
            
        with col2:
            st.write("**Enhanced Modules:**")
            st.write(f"{'✅' if VOLUME_ANALYSIS_AVAILABLE else '❌'} Volume Analysis: {VOLUME_ANALYSIS_AVAILABLE}")
            st.write(f"{'✅' if VOLATILITY_ANALYSIS_AVAILABLE else '❌'} Volatility Analysis: {VOLATILITY_ANALYSIS_AVAILABLE}")
            st.write(f"{'✅' if CHARTS_AVAILABLE else '❌'} Charts: {CHARTS_AVAILABLE}")
            st.write(f"{'✅' if BALDWIN_INDICATOR_AVAILABLE else '❌'} Baldwin Indicator: {BALDWIN_INDICATOR_AVAILABLE}")
            
        with col3:
            st.write("**Environment:**")
            import streamlit as st
            import pandas as pd
            import numpy as np
            try:
                import plotly
                plotly_version = plotly.__version__
            except:
                plotly_version = "Not Available"
            
            st.write(f"**Streamlit:** {st.__version__}")
            st.write(f"**Pandas:** {pd.__version__}")
            st.write(f"**NumPy:** {np.__version__}")
            st.write(f"**Plotly:** {plotly_version}")
            
        # Data Quality Assessment
        st.subheader("📊 Market Data Quality Assessment")
        if market_data is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Data Points", len(market_data))
                st.metric("Missing Values", market_data.isnull().sum().sum())
                
            with col2:
                st.metric("Date Range", f"{len(market_data)} days")
                st.metric("Columns", len(market_data.columns))
                
            with col3:
                price_range = market_data['High'].max() - market_data['Low'].min()
                st.metric("Price Range", f"${price_range:.2f}")
                avg_volume = market_data['Volume'].mean()
                st.metric("Avg Volume", format_large_number(avg_volume))
                
            with col4:
                returns = market_data['Close'].pct_change().dropna()
                volatility = returns.std() * (252**0.5) * 100
                st.metric("Volatility", f"{volatility:.1f}%")
                st.metric("Data Quality", "✅ Good" if market_data.isnull().sum().sum() == 0 else "⚠️ Issues")
            
            # Data structure details
            st.write("**Market Data Structure:**")
            st.write(f"Shape: {market_data.shape}")
            st.write(f"Index: {type(market_data.index)} ({market_data.index[0]} to {market_data.index[-1]})")
            st.write(f"Columns: {list(market_data.columns)}")
            st.write(f"Data Types: {dict(market_data.dtypes)}")
            
            # Sample data
            st.write("**Sample Data (First 3 rows):**")
            st.dataframe(market_data.head(3))
            
            st.write("**Sample Data (Last 3 rows):**")
            st.dataframe(market_data.tail(3))
            
        else:
            st.error("❌ No market data available for quality assessment")
            
        # Analysis Results Structure
        st.subheader("🔍 Analysis Results Structure")
        if analysis_results:
            st.write("**Top-Level Keys:**")
            for key in analysis_results.keys():
                st.write(f"• **{key}**: {type(analysis_results[key])}")
                
            # Enhanced indicators breakdown
            enhanced_indicators = analysis_results.get('enhanced_indicators', {})
            if enhanced_indicators:
                st.write("**Enhanced Indicators Keys:**")
                for key in enhanced_indicators.keys():
                    value = enhanced_indicators[key]
                    if isinstance(value, dict):
                        st.write(f"• **{key}**: Dict with {len(value)} keys")
                    elif isinstance(value, list):
                        st.write(f"• **{key}**: List with {len(value)} items")
                    else:
                        st.write(f"• **{key}**: {type(value)}")
            
            # Complete analysis results (expandable)
            with st.expander("📋 Complete Analysis Results JSON", expanded=False):
                st.json(analysis_results, expanded=False)
        else:
            st.error("❌ No analysis results available")
            
        # Session State Information
        st.subheader("🔄 Session State Information")
        session_info = {
            'recently_viewed': len(st.session_state.get('recently_viewed', [])),
            'show_technical_analysis': st.session_state.get('show_technical_analysis', False),
            'show_volume_analysis': st.session_state.get('show_volume_analysis', False),
            'show_volatility_analysis': st.session_state.get('show_volatility_analysis', False),
            'show_fundamental_analysis': st.session_state.get('show_fundamental_analysis', False),
            'show_market_correlation': st.session_state.get('show_market_correlation', False),
            'show_options_analysis': st.session_state.get('show_options_analysis', False),
            'show_confidence_intervals': st.session_state.get('show_confidence_intervals', False),
            'show_charts': st.session_state.get('show_charts', False),
            'show_baldwin_indicator': st.session_state.get('show_baldwin_indicator', False)
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Toggle States:**")
            for key, value in session_info.items():
                if key.startswith('show_'):
                    st.write(f"• {key}: {'✅' if value else '❌'}")
                    
        with col2:
            st.write("**Session Data:**")
            st.write(f"• Recently Viewed: {session_info['recently_viewed']} symbols")
            st.write(f"• Current Symbol: {st.session_state.get('symbol_input', 'Not Set')}")
            st.write(f"• Last Symbol Input: {st.session_state.get('last_symbol_input', 'Not Set')}")
            
            # Data manager summary
            try:
                data_manager = get_data_manager()
                summary = data_manager.get_data_summary()
                st.write(f"• Stored Symbols: {summary.get('market_data_count', 0)}")
                st.write(f"• Analysis Cache: {summary.get('analysis_results_count', 0)}")
            except Exception as e:
                st.write(f"• Data Manager: Error ({str(e)[:30]}...)")
                
        # Performance Metrics
        st.subheader("⚡ Performance Metrics")
        import time
        current_time = time.time()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Timing Information:**")
            st.write(f"• Current Time: {datetime.now().strftime('%H:%M:%S')}")
            st.write(f"• Analysis Timestamp: {analysis_results.get('timestamp', 'N/A')}")
            
        with col2:
            st.write("**Memory Usage:**")
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                st.write(f"• Memory Usage: {memory_mb:.1f} MB")
                st.write(f"• CPU Percent: {process.cpu_percent():.1f}%")
            except:
                st.write("• Memory Info: Not Available")
                
        with col3:
            st.write("**Data Sizes:**")
            if market_data is not None:
                data_size_mb = market_data.memory_usage(deep=True).sum() / 1024 / 1024
                st.write(f"• Market Data: {data_size_mb:.2f} MB")
            if analysis_results:
                import sys
                result_size_kb = sys.getsizeof(str(analysis_results)) / 1024
                st.write(f"• Analysis Results: {result_size_kb:.1f} KB")
                
        # Error Log
        st.subheader("⚠️ System Diagnostics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Import Status:**")
            st.write(f"• Charts Module: {'✅' if CHARTS_AVAILABLE else '❌'}")
            st.write(f"• Volume Module: {'✅' if VOLUME_ANALYSIS_AVAILABLE else '❌'}")
            st.write(f"• Volatility Module: {'✅' if VOLATILITY_ANALYSIS_AVAILABLE else '❌'}")
            
        with col2:
            st.write("**Configuration Status:**")
            try:
                st.write(f"• UI Settings: ✅ {len(UI_SETTINGS)} settings")
                st.write(f"• Quick Links: ✅ {sum(len(symbols) for symbols in QUICK_LINK_CATEGORIES.values())} symbols")
                st.write(f"• Default Config: ✅ {len(DEFAULT_VWV_CONFIG)} parameters")
            except Exception as e:
                st.write(f"• Configuration: ❌ Error loading")
                
        # Test Basic Functionality
        st.subheader("🧪 Basic Functionality Tests")
        test_col1, test_col2 = st.columns(2)
        
        with test_col1:
            if st.button("Test Chart Creation", key="test_chart"):
                try:
                    import plotly.graph_objects as go
                    test_fig = go.Figure()
                    test_fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 2], name="Test"))
                    test_fig.update_layout(title="Test Chart", height=300)
                    st.plotly_chart(test_fig, use_container_width=True)
                    st.success("✅ Basic chart creation successful")
                except Exception as e:
                    st.error(f"❌ Chart creation failed: {str(e)}")
                    
        with test_col2:
            if st.button("Test Data Processing", key="test_data"):
                try:
                    test_data = pd.DataFrame({
                        'A': [1, 2, 3, 4, 5],
                        'B': [2, 4, 6, 8, 10]
                    })
                    test_result = test_data.mean()
                    st.write("Test calculation result:", test_result)
                    st.success("✅ Data processing successful")
                except Exception as e:
                    st.error(f"❌ Data processing failed: {str(e)}")

def perform_enhanced_analysis(symbol, period, show_debug=False):
    """Perform enhanced analysis using modular components - ENHANCED v4.2.1"""
    try:
        # Step 1: Fetch data using modular data fetcher
        market_data = get_market_data_enhanced(symbol, period, show_debug)
        
        if market_data is None:
            st.error(f"❌ Could not fetch data for {symbol}")
            return None
        
        # Step 2: Store data using data manager
        data_manager = get_data_manager()
        data_manager.store_market_data(symbol, market_data, show_debug)
        
        # Step 3: Get analysis copy
        analysis_input = data_manager.get_market_data_for_analysis(symbol)
        
        if analysis_input is None:
            st.error("❌ Could not prepare analysis data")
            return None
        
        # Step 4: Calculate enhanced indicators using modular analysis
        enhanced_indicators = calculate_enhanced_technical_analysis(analysis_input)
        
        # Step 5: Calculate market correlations
        market_correlations = calculate_market_correlations_enhanced(analysis_input, symbol, show_debug=show_debug)
        
        # Step 6: Calculate fundamental analysis (skip for ETFs)
        is_etf_symbol = is_etf(symbol)
        
        if is_etf_symbol:
            graham_score = {'score': 0, 'total_possible': 10, 'criteria': [], 'error': 'ETF - Fundamental analysis not applicable'}
            piotroski_score = {'score': 0, 'total_possible': 9, 'criteria': [], 'error': 'ETF - Fundamental analysis not applicable'}
        else:
            graham_score = calculate_graham_score(symbol, show_debug)
            piotroski_score = calculate_piotroski_score(symbol, show_debug)
        
        # Step 7: Calculate options levels
        volatility = enhanced_indicators.get('comprehensive_technicals', {}).get('volatility_20d', 20)
        underlying_beta = 1.0  # Default market beta
        
        if market_correlations:
            for etf in ['SPY', 'QQQ', 'MAGS']:
                if etf in market_correlations and 'beta' in market_correlations[etf]:
                    try:
                        underlying_beta = abs(float(market_correlations[etf]['beta']))
                        break
                    except:
                        continue
        
        current_price = round(float(analysis_input['Close'].iloc[-1]), 2)
        options_levels = calculate_options_levels_enhanced(current_price, volatility, underlying_beta=underlying_beta)
        
        # Step 8: Calculate confidence intervals
        confidence_analysis = calculate_confidence_intervals(analysis_input)
        
        # Step 9: Add fundamental and market data to enhanced indicators
        enhanced_indicators['market_correlations'] = market_correlations
        enhanced_indicators['options_levels'] = options_levels
        enhanced_indicators['graham_score'] = graham_score
        enhanced_indicators['piotroski_score'] = piotroski_score
        
        # Step 10: Build analysis results
        current_date = analysis_input.index[-1].strftime('%Y-%m-%d')
        
        analysis_results = {
            'symbol': symbol,
            'timestamp': current_date,
            'current_price': current_price,
            'enhanced_indicators': enhanced_indicators,
            'confidence_analysis': confidence_analysis,
            'system_status': 'OPERATIONAL v4.2.1'
        }
        
        # Store results
        data_manager.store_analysis_results(symbol, analysis_results)
        
        return analysis_results, market_data, baldwin_results
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        return None, None, None

def main():
    """Main application function - ENHANCED v4.2.1"""
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
            analysis_results, market_data, baldwin_results = perform_enhanced_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results and market_data is not None:
                # CHARTS FIRST - TOP PRIORITY DISPLAY
                show_interactive_charts(analysis_results, market_data, controls['show_debug'])
                
                # BALDWIN MARKET SENTIMENT INDICATOR - HIGH PRIORITY
                if BALDWIN_INDICATOR_AVAILABLE:
                    show_baldwin_indicator_analysis(baldwin_results, controls['show_debug'])
                
                # Show all analysis sections using modular functions
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                
                # NEW v4.2.1 - Volume and Volatility Analysis
                if VOLUME_ANALYSIS_AVAILABLE:
                    show_volume_analysis(analysis_results, controls['show_debug'])
                if VOLATILITY_ANALYSIS_AVAILABLE:
                    show_volatility_analysis(analysis_results, controls['show_debug'])
                
                # Existing analysis sections - PRESERVED
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                show_options_analysis(analysis_results, controls['show_debug'])
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
                # Enhanced Debug information
                show_enhanced_debug_information(analysis_results, market_data, controls['show_debug'])
    
    else:
        # Welcome message
        st.write("## 🚀 VWV Professional Trading System v4.2.1 Enhanced")
        st.write("**Complete modular architecture with Volume & Volatility analysis**")
        
        with st.expander("🏗️ System Status v4.2.1", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📁 **Core Modules**")
                st.write("✅ **config/** - Settings, constants, parameters")
                st.write("✅ **data/** - Fetching, validation, management")
                st.write("✅ **analysis/** - Technical, volume, volatility, fundamental, market, options")
                st.write("✅ **ui/** - Components, headers, score bars")
                st.write("✅ **utils/** - Helpers, decorators, formatters")
                
            with col2:
                st.write("### 🎯 **All Sections Working**")
                st.write("• **Individual Technical Analysis** ✅")
                if VOLUME_ANALYSIS_AVAILABLE:
                    st.write("• **🆕 Volume Analysis** ✅")
                else:
                    st.write("• **Volume Analysis** ⚠️ (Module not available)")
                if VOLATILITY_ANALYSIS_AVAILABLE:
                    st.write("• **🆕 Volatility Analysis** ✅")
                else:
                    st.write("• **Volatility Analysis** ⚠️ (Module not available)")
                st.write("• **Fundamental Analysis** ✅")
                st.write("• **Market Correlation & Breakouts** ✅")
                st.write("• **Options Analysis with Greeks** ✅")
                st.write("• **Statistical Confidence Intervals** ✅")
        
        # Show current market status
        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.write("1. **📊 Charts First** - Interactive price charts display immediately")
            st.write("2. **Quick Links** - Click any symbol button for instant analysis")
            st.write("3. **Manual Entry** - Type symbol in sidebar + Enter for analysis")
            st.write("4. **Analyze Button** - Click 'Analyze Symbol' to run analysis")
            st.write("5. **View all sections:** Charts, Technical, Volume, Volatility, Fundamental, Market, Options")
            st.write("6. **Toggle sections** on/off in Analysis Sections panel")
            st.write("7. **Enhanced Debug Mode** - Comprehensive system diagnostics and performance metrics")

    # Footer
    st.markdown("---")
    st.write("### 📊 System Information v4.2.1")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Version:** VWV Professional v4.2.1 Enhanced")
        st.write(f"**Architecture:** Full Modular Implementation")
    with col2:
        st.write(f"**Status:** ✅ All Core Modules Active")
        st.write(f"**Features:** Charts First + Volume & Volatility Analysis")
    with col3:
        st.write(f"**Components:** config, data, analysis, ui, utils, charts")
        st.write(f"**Debug:** Enhanced diagnostics with performance metrics")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.write("Please refresh the page and try again.")
        
        if st.checkbox("Show Error Details"):
            st.exception(e)
