"""
VWV Professional Trading System v4.2.1 - Enhanced Volume Analysis Integration
ENHANCED FEATURES:
- Volume Analysis with 14 comprehensive indicators and composite scoring
- Gradient bar display for volume composite score
- Nested component breakdown showing all volume indicators
- All existing functionality preserved
Date: August 21, 2025 - 3:20 PM EST
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

# Enhanced Volume Analysis imports
try:
    from analysis.volume import (
        calculate_complete_volume_analysis,
        create_volume_score_bar
    )
    VOLUME_ANALYSIS_AVAILABLE = True
except ImportError:
    VOLUME_ANALYSIS_AVAILABLE = False

# Volatility Analysis imports with safe fallbacks
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
    if 'show_baldwin_indicator' not in st.session_state:
        st.session_state.show_baldwin_indicator = True
    if 'show_market_correlation' not in st.session_state:
        st.session_state.show_market_correlation = True
    if 'show_options_analysis' not in st.session_state:
        st.session_state.show_options_analysis = True
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False

    # Symbol input
    symbol = st.sidebar.text_input("🎯 Enter Symbol", value="SPY", help="Enter a stock symbol (e.g., AAPL, SPY, QQQ)")
    
    # Time period selection - CORRECTED DEFAULT
    period = st.sidebar.selectbox(
        "📅 Time Period",
        options=['1mo', '3mo', '6mo', '1y', '2y'],
        index=0,  # CORRECTED: Default to 1mo (first option)
        help="Select the time period for analysis"
    )
    
    # Analysis toggle controls
    st.sidebar.subheader("📊 Analysis Sections")
    
    show_technical = st.sidebar.checkbox("🔴 Technical Analysis", value=st.session_state.show_technical_analysis, key="tech_check")
    show_volume = st.sidebar.checkbox("📊 Volume Analysis", value=st.session_state.show_volume_analysis, key="vol_check")
    show_volatility = st.sidebar.checkbox("📊 Volatility Analysis", value=st.session_state.show_volatility_analysis, key="volat_check")
    show_fundamental = st.sidebar.checkbox("📊 Fundamental Analysis", value=st.session_state.show_fundamental_analysis, key="fund_check")
    show_baldwin = st.sidebar.checkbox("🚦 Baldwin Indicator", value=st.session_state.show_baldwin_indicator, key="bald_check")
    show_market = st.sidebar.checkbox("🌐 Market Correlation", value=st.session_state.show_market_correlation, key="mkt_check")
    show_options = st.sidebar.checkbox("🎯 Options Analysis", value=st.session_state.show_options_analysis, key="opt_check")
    
    # Update session state
    st.session_state.show_technical_analysis = show_technical
    st.session_state.show_volume_analysis = show_volume
    st.session_state.show_volatility_analysis = show_volatility
    st.session_state.show_fundamental_analysis = show_fundamental
    st.session_state.show_baldwin_indicator = show_baldwin
    st.session_state.show_market_correlation = show_market
    st.session_state.show_options_analysis = show_options
    
    # Debug toggle
    show_debug = st.sidebar.checkbox("🐛 Show Debug Info", value=st.session_state.show_debug, key="debug_check")
    st.session_state.show_debug = show_debug
    
    # Analysis button
    analyze_button = st.sidebar.button("📊 Analyze Now", type="primary", use_container_width=True)
    
    # Quick Links section - FIXED STRUCTURE
    with st.sidebar.expander("🔗 Quick Links", expanded=False):
        st.write("**Popular Symbols by Category**")
        for category, symbols in QUICK_LINK_CATEGORIES.items():
            with st.expander(f"{category} ({len(symbols)} symbols)", expanded=False):
                for symbol_key in symbols:
                    symbol_name = SYMBOL_DESCRIPTIONS.get(symbol_key, symbol_key)
                    if st.button(f"{symbol_key} - {symbol_name}", key=f"quick_{symbol_key}", use_container_width=True):
                        st.session_state.quick_symbol = symbol_key
                        st.experimental_rerun()
    
    # Check for quick symbol selection
    if 'quick_symbol' in st.session_state:
        symbol = st.session_state.quick_symbol
        del st.session_state.quick_symbol
        analyze_button = True
    
    return {
        'symbol': symbol.upper() if symbol else '',
        'period': period,
        'analyze_button': analyze_button,
        'show_debug': show_debug
    }

def add_to_recently_viewed(symbol):
    """Add symbol to recently viewed list"""
    if symbol not in st.session_state.recently_viewed:
        st.session_state.recently_viewed.insert(0, symbol)
        st.session_state.recently_viewed = st.session_state.recently_viewed[:10]  # Keep last 10

def show_interactive_charts(chart_data, analysis_results, show_debug=False):
    """PRIORITY 1: Display interactive charts section - MUST BE FIRST"""
    with st.expander(f"📊 {analysis_results['symbol']} - Interactive Charts", expanded=True):
        
        if chart_data is not None and not chart_data.empty:
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Create subplot with secondary y-axis for volume
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3],
                    subplot_titles=('Price & Technical Indicators', 'Volume')
                )
                
                # Candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=chart_data.index,
                        open=chart_data['Open'],
                        high=chart_data['High'],
                        low=chart_data['Low'],
                        close=chart_data['Close'],
                        name="Price"
                    ),
                    row=1, col=1
                )
                
                # Add VWAP if available
                enhanced_indicators = analysis_results.get('enhanced_indicators', {})
                daily_vwap = enhanced_indicators.get('daily_vwap', None)
                if daily_vwap and not pd.isna(daily_vwap):
                    fig.add_hline(y=daily_vwap, line_dash="dash", line_color="blue", 
                                annotation_text=f"VWAP: ${daily_vwap:.2f}", row=1, col=1)
                
                # Volume bars
                fig.add_trace(
                    go.Bar(
                        x=chart_data.index,
                        y=chart_data['Volume'],
                        name="Volume",
                        marker_color='lightblue',
                        opacity=0.7
                    ),
                    row=2, col=1
                )
                
                # Update layout
                fig.update_layout(
                    title=f"{analysis_results['symbol']} - Comprehensive Analysis",
                    xaxis_rangeslider_visible=False,
                    height=600,
                    showlegend=True
                )
                
                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                
            except ImportError:
                st.warning("📊 Plotly not available for advanced charts. Install plotly for enhanced visualizations.")
                # Fallback to simple chart
                st.subheader("Basic Price Chart")
                st.line_chart(chart_data['Close'])
                
        else:
            st.error("❌ No chart data available. Try refreshing or enable debug mode for details.")

def show_individual_technical_analysis(analysis_results, show_debug=False):
    """PRIORITY 2: Display individual technical analysis section - MUST BE SECOND"""
    if not st.session_state.get('show_technical_analysis', True):
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Individual Technical Analysis", expanded=True):
        
        # Composite Technical Score Bar
        composite_score, score_details = calculate_composite_technical_score(analysis_results)
        score_bar_html = create_technical_score_bar(composite_score, score_details)
        st.components.v1.html(score_bar_html, height=160)
        
        # Technical indicators display
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
        
        # Key Momentum Oscillators
        st.subheader("Key Momentum Oscillators")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            rsi = comprehensive_technicals.get('rsi_14', 50)
            st.metric("RSI (14)", f"{rsi:.2f}", "Oversold < 30")
        with col2:
            mfi = comprehensive_technicals.get('mfi_14', 50)
            st.metric("MFI (14)", f"{mfi:.2f}", "Oversold < 20")
        with col3:
            stoch = comprehensive_technicals.get('stochastic', {})
            st.metric("Stochastic %K", f"{stoch.get('k', 50):.2f}", "Oversold < 20")
        with col4:
            williams_r = comprehensive_technicals.get('williams_r', -50)
            st.metric("Williams %R", f"{williams_r:.2f}", "Oversold < -80")

        # Trend Analysis
        st.subheader("Trend Analysis")
        col1, col2 = st.columns(2)
        with col1:
            macd_data = comprehensive_technicals.get('macd', {})
            macd_hist = macd_data.get('histogram', 0)
            macd_delta = "Bullish" if macd_hist > 0 else "Bearish"
            st.metric("MACD Histogram", f"{macd_hist:.4f}", macd_delta)

        # Price-Based Indicators & Key Levels
        st.subheader("Price-Based Indicators & Key Levels")
        current_price = analysis_results['current_price']
        daily_vwap = enhanced_indicators.get('daily_vwap', 0)
        point_of_control = enhanced_indicators.get('point_of_control', 0)

        indicators_data = []
        indicators_data.append(("Current Price", f"${current_price:.2f}", "📍 Reference", "0.0%", "Current"))
        
        vwap_distance = f"{((current_price - daily_vwap) / daily_vwap * 100):+.2f}%" if daily_vwap > 0 else "N/A"
        vwap_status = "Above" if current_price > daily_vwap else "Below"
        indicators_data.append(("Daily VWAP", f"${daily_vwap:.2f}", "📊 Volume Weighted", vwap_distance, vwap_status))
        
        poc_distance = f"{((current_price - point_of_control) / point_of_control * 100):+.2f}%" if point_of_control > 0 else "N/A"
        poc_status = "Above" if current_price > point_of_control else "Below"
        indicators_data.append(("Point of Control", f"${point_of_control:.2f}", "📊 Volume Profile", poc_distance, poc_status))
        
        for ema_name, ema_value in fibonacci_emas.items():
            period = ema_name.split('_')[1]
            distance_pct = f"{((current_price - ema_value) / ema_value * 100):+.2f}%" if ema_value > 0 else "N/A"
            status = "Above" if current_price > ema_value else "Below"
            indicators_data.append((f"EMA {period}", f"${ema_value:.2f}", "📈 Trend", distance_pct, status))
        
        df_technical = pd.DataFrame(indicators_data, columns=['Indicator', 'Value', 'Type', 'Distance %', 'Status'])
        st.dataframe(df_technical, use_container_width=True, hide_index=True)

def show_enhanced_volume_analysis(analysis_results, show_debug=False):
    """Display ENHANCED volume analysis section with composite scoring and component breakdown"""
    if not st.session_state.show_volume_analysis or not VOLUME_ANALYSIS_AVAILABLE:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Enhanced Volume Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volume_analysis = enhanced_indicators.get('volume_analysis', {})
        
        if 'error' not in volume_analysis and volume_analysis:
            
            # 1. COMPOSITE VOLUME SCORE BAR (TOP PRIORITY)
            composite_score = volume_analysis.get('composite_score', 50)
            volume_score_bar_html = create_volume_score_bar(composite_score)
            st.components.v1.html(volume_score_bar_html, height=160)
            
            # 2. KEY VOLUME METRICS
            st.subheader("📊 Key Volume Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Volume", format_large_number(volume_analysis.get('current_volume', 0)))
            with col2:
                st.metric("5D Avg Volume", format_large_number(volume_analysis.get('volume_5d_avg', 0)))
            with col3:
                volume_ratio = volume_analysis.get('volume_ratio', 1.0)
                st.metric("Volume Ratio", f"{volume_ratio:.2f}x", "vs 30D avg")
            with col4:
                st.metric("Volume Regime", volume_analysis.get('volume_regime', 'Normal'))
            
            # 3. TRADING IMPLICATIONS
            st.subheader("📈 Trading Implications")
            trading_implications = volume_analysis.get('trading_implications', 'No implications available')
            st.info(f"**Strategy Guidance:** {trading_implications}")
            
            # 4. COMPONENT BREAKDOWN (NESTED EXPANDER)
            components = volume_analysis.get('components', {})
            if components and 'error' not in components:
                with st.expander("🔍 Volume Component Breakdown", expanded=False):
                    st.write("**Individual volume indicators contributing to the composite score:**")
                    
                    # Create component breakdown table
                    component_data = []
                    for component_name, component_info in components.items():
                        if isinstance(component_info, dict) and 'score' in component_info:
                            # Format component name for display
                            display_name = component_name.replace('_', ' ').title()
                            score = component_info.get('score', 50)
                            weight = component_info.get('weight', 0) * 100  # Convert to percentage
                            contribution = component_info.get('contribution', 0)
                            trend = component_info.get('trend', 'Unknown')
                            value = component_info.get('value', 0)
                            
                            component_data.append([
                                display_name,
                                f"{score:.1f}",
                                f"{weight:.1f}%",
                                f"{contribution:.1f}",
                                trend,
                                f"{value}"
                            ])
                    
                    if component_data:
                        df_components = pd.DataFrame(
                            component_data,
                            columns=['Indicator', 'Score', 'Weight', 'Contribution', 'Trend', 'Value']
                        )
                        st.dataframe(df_components, use_container_width=True, hide_index=True)
                        
                        # Component scoring explanation
                        st.write("**Scoring Guide:**")
                        st.write("- **Score**: Individual indicator score (0-100)")
                        st.write("- **Weight**: Importance in composite calculation")
                        st.write("- **Contribution**: Weighted contribution to final score")
                        st.write("- **Trend**: Current indicator direction/strength")
                    else:
                        st.warning("No component data available for display")
            
            # Debug information for volume analysis
            if show_debug:
                with st.expander("🐛 Volume Analysis Debug", expanded=False):
                    st.write("### Volume Analysis Data Structure")
                    st.json(volume_analysis)
                    
                    st.write("### Component Details")
                    if components:
                        for name, data in components.items():
                            st.write(f"**{name}:**")
                            st.json(data)
                            
        else:
            st.warning("⚠️ Enhanced volume analysis not available - insufficient data or calculation error")
            if show_debug and 'error' in volume_analysis:
                st.error(f"Error details: {volume_analysis['error']}")

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
            if show_debug and 'error' in volatility_analysis:
                st.error(f"Error details: {volatility_analysis['error']}")

def show_fundamental_analysis(analysis_results, show_debug=False):
    """Display fundamental analysis section"""
    if not st.session_state.show_fundamental_analysis:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Fundamental Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        graham_score = enhanced_indicators.get('graham_score', {})
        piotroski_score = enhanced_indicators.get('piotroski_score', {})
        
        if graham_score or piotroski_score:
            col1, col2 = st.columns(2)
            
            with col1:
                if graham_score and 'score' in graham_score:
                    st.subheader("📈 Graham Value Score")
                    score = graham_score['score']
                    st.metric("Graham Score", f"{score}/7", f"{'Strong Value' if score >= 5 else 'Moderate Value' if score >= 3 else 'Weak Value'}")
                    
                    if 'details' in graham_score:
                        with st.expander("Graham Score Details", expanded=False):
                            for criterion, result in graham_score['details'].items():
                                st.write(f"**{criterion}:** {'✅ Pass' if result else '❌ Fail'}")
                else:
                    st.info("Graham analysis not available (requires financial data)")
            
            with col2:
                if piotroski_score and 'score' in piotroski_score:
                    st.subheader("📊 Piotroski F-Score")
                    score = piotroski_score['score']
                    st.metric("Piotroski Score", f"{score}/9", f"{'Strong' if score >= 7 else 'Good' if score >= 5 else 'Weak'}")
                    
                    if 'details' in piotroski_score:
                        with st.expander("Piotroski Score Details", expanded=False):
                            for criterion, result in piotroski_score['details'].items():
                                st.write(f"**{criterion}:** {'✅ Pass' if result else '❌ Fail'}")
                else:
                    st.info("Piotroski analysis not available (requires financial data)")
        else:
            st.warning("⚠️ Fundamental analysis not available - requires financial data")

def show_baldwin_indicator_analysis(show_debug=False):
    """Display Baldwin Indicator analysis section"""
    if not st.session_state.show_baldwin_indicator or not BALDWIN_INDICATOR_AVAILABLE:
        return
        
    with st.expander("🚦 Baldwin Market Regime Indicator", expanded=True):
        try:
            with st.spinner("Calculating Baldwin Market Regime..."):
                baldwin_results = calculate_baldwin_indicator_complete()
                
            if baldwin_results and 'error' not in baldwin_results:
                formatted_results = format_baldwin_for_display(baldwin_results)
                
                # Display formatted results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Market Regime", formatted_results.get('regime', 'Unknown'))
                with col2:
                    st.metric("Regime Score", f"{formatted_results.get('score', 0)}/100")
                with col3:
                    st.metric("Confidence", formatted_results.get('confidence', 'Low'))
                
                # Market implications
                implications = formatted_results.get('implications', 'No implications available')
                st.info(f"**Market Implications:** {implications}")
                
            else:
                st.warning("⚠️ Baldwin indicator analysis not available")
                if show_debug and baldwin_results and 'error' in baldwin_results:
                    st.error(f"Error: {baldwin_results['error']}")
                    
        except Exception as e:
            st.error(f"❌ Baldwin indicator error: {str(e)}")

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis section"""
    if not st.session_state.show_market_correlation:
        return
        
    with st.expander(f"🌐 {analysis_results['symbol']} - Market Correlation Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        market_correlations = enhanced_indicators.get('market_correlations', {})
        
        if market_correlations and 'error' not in market_correlations:
            # Display correlation metrics
            correlations = market_correlations.get('correlations', {})
            if correlations:
                st.subheader("📊 ETF Correlations")
                
                correlation_data = []
                for etf, corr_value in correlations.items():
                    if corr_value is not None:
                        correlation_data.append([etf, f"{corr_value:.3f}", 
                                               "Strong" if abs(corr_value) >= 0.7 else "Moderate" if abs(corr_value) >= 0.3 else "Weak"])
                
                if correlation_data:
                    df_corr = pd.DataFrame(correlation_data, columns=['ETF', 'Correlation', 'Strength'])
                    st.dataframe(df_corr, use_container_width=True, hide_index=True)
            
            # Breakout analysis
            breakout_analysis = market_correlations.get('breakout_analysis', {})
            if breakout_analysis:
                st.subheader("🎯 Breakout Analysis")
                
                breakout_signal = breakout_analysis.get('breakout_signal', 'None')
                breakout_strength = breakout_analysis.get('breakout_strength', 0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Breakout Signal", breakout_signal)
                with col2:
                    st.metric("Breakout Strength", f"{breakout_strength:.2f}")
                    
        else:
            st.warning("⚠️ Market correlation analysis not available")

def show_options_analysis(analysis_results, show_debug=False):
    """Display options analysis section"""
    if not st.session_state.show_options_analysis:
        return
        
    with st.expander(f"🎯 {analysis_results['symbol']} - Options Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        options_levels = enhanced_indicators.get('options_levels', {})
        
        if options_levels and 'error' not in options_levels:
            # Display options levels
            st.subheader("🎯 Options Strike Levels")
            
            put_levels = options_levels.get('put_levels', {})
            call_levels = options_levels.get('call_levels', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Put Options Levels**")
                for level, strike in put_levels.items():
                    st.write(f"**{level}:** ${strike:.2f}")
            
            with col2:
                st.write("**Call Options Levels**")
                for level, strike in call_levels.items():
                    st.write(f"**{level}:** ${strike:.2f}")
                    
        else:
            st.warning("⚠️ Options analysis not available")

def show_confidence_intervals(analysis_results, show_debug=False):
    """Display confidence intervals section"""
    with st.expander(f"📊 {analysis_results['symbol']} - Statistical Confidence Intervals", expanded=True):
        
        confidence_analysis = analysis_results.get('confidence_analysis', {})
        
        if confidence_analysis and 'error' not in confidence_analysis:
            # Display confidence intervals
            intervals = confidence_analysis.get('intervals', {})
            
            if intervals:
                st.subheader("📊 Weekly Price Projections")
                
                interval_data = []
                for confidence_level, bounds in intervals.items():
                    if isinstance(bounds, dict):
                        lower = bounds.get('lower', 0)
                        upper = bounds.get('upper', 0)
                        interval_data.append([confidence_level, f"${lower:.2f}", f"${upper:.2f}"])
                
                if interval_data:
                    df_intervals = pd.DataFrame(interval_data, columns=['Confidence', 'Lower Bound', 'Upper Bound'])
                    st.dataframe(df_intervals, use_container_width=True, hide_index=True)
                    
            # Statistical summary
            stats = confidence_analysis.get('statistics', {})
            if stats:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Expected Return", f"{stats.get('expected_return', 0)*100:.2f}%")
                with col2:
                    st.metric("Volatility", f"{stats.get('volatility', 0)*100:.2f}%")
                with col3:
                    st.metric("Skewness", f"{stats.get('skewness', 0):.3f}")
                with col4:
                    st.metric("Kurtosis", f"{stats.get('kurtosis', 0):.3f}")
                    
        else:
            st.warning("⚠️ Confidence intervals not available")

def perform_enhanced_analysis(symbol, period, show_debug):
    """Perform comprehensive analysis with enhanced volume analysis"""
    try:
        # Get data
        data_manager = get_data_manager()
        data = get_market_data_enhanced(symbol, period)
        
        if data is None or len(data) < 30:
            st.error("❌ Insufficient data for analysis. Try a different symbol or period.")
            return None, None
        
        # Store market data
        data_manager.store_market_data(symbol, data)
        
        # Calculate technical indicators
        daily_vwap = calculate_daily_vwap(data)
        fibonacci_emas = calculate_fibonacci_emas(data)
        point_of_control = calculate_point_of_control_enhanced(data)
        weekly_deviations = calculate_weekly_deviations(data)
        comprehensive_technicals = calculate_comprehensive_technicals(data)
        
        # Enhanced Volume Analysis
        if VOLUME_ANALYSIS_AVAILABLE:
            volume_analysis = calculate_complete_volume_analysis(data)
        else:
            volume_analysis = {'error': 'Volume analysis module not available'}
        
        # Volatility Analysis
        if VOLATILITY_ANALYSIS_AVAILABLE:
            volatility_analysis = calculate_complete_volatility_analysis(data)
        else:
            volatility_analysis = {'error': 'Volatility analysis module not available'}
        
        # Market correlation analysis
        market_correlations = calculate_market_correlations_enhanced(data, symbol)
        
        # Options analysis
        options_levels = calculate_options_levels_enhanced(data)
        
        # Confidence intervals
        confidence_analysis = calculate_confidence_intervals(data)
        
        # Fundamental analysis
        graham_score = calculate_graham_score(symbol)
        piotroski_score = calculate_piotroski_score(symbol)
        
        # Build analysis results
        analysis_results = {
            'symbol': symbol,
            'current_price': float(data['Close'].iloc[-1]),
            'period': period,
            'data_points': len(data),
            'enhanced_indicators': {
                'daily_vwap': daily_vwap,
                'fibonacci_emas': fibonacci_emas,
                'point_of_control': point_of_control,
                'weekly_deviations': weekly_deviations,
                'comprehensive_technicals': comprehensive_technicals,
                'volume_analysis': volume_analysis,  # ENHANCED
                'volatility_analysis': volatility_analysis,
                'market_correlations': market_correlations,
                'options_levels': options_levels,
                'graham_score': graham_score,
                'piotroski_score': piotroski_score
            },
            'confidence_analysis': confidence_analysis,
            'system_status': 'Enhanced Volume Analysis v4.2.1'
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
    """Main application function with enhanced volume analysis integration"""
    # Create header using modular component
    create_header()
    
    # Create sidebar and get controls
    controls = create_sidebar_controls()
    
    # Main logic flow
    if controls['analyze_button'] and controls['symbol']:
        # Add symbol to recently viewed
        add_to_recently_viewed(controls['symbol'])
        
        st.write("## 📊 VWV Trading Analysis v4.2.1 - Enhanced Volume Analysis")
        
        with st.spinner(f"Analyzing {controls['symbol']}..."):
            
            # Perform analysis using modular components
            analysis_results, chart_data = perform_enhanced_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results and chart_data is not None:
                
                # DISPLAY ORDER - MANDATORY SEQUENCE:
                
                # 1. CHARTS FIRST (MANDATORY)
                show_interactive_charts(chart_data, analysis_results, controls['show_debug'])
                
                # 2. INDIVIDUAL TECHNICAL ANALYSIS SECOND (MANDATORY)
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                
                # 3. ENHANCED Volume Analysis (NEW - with composite scoring)
                if VOLUME_ANALYSIS_AVAILABLE:
                    show_enhanced_volume_analysis(analysis_results, controls['show_debug'])
                
                # 4. Volatility Analysis (Optional - when available)
                if VOLATILITY_ANALYSIS_AVAILABLE:
                    show_volatility_analysis(analysis_results, controls['show_debug'])
                
                # 5. Fundamental Analysis
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                
                # 6. BALDWIN INDICATOR (Before Market Correlation)
                if BALDWIN_INDICATOR_AVAILABLE:
                    show_baldwin_indicator_analysis(show_debug=controls['show_debug'])
                
                # 7. Market Correlation Analysis (After Baldwin)
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
                        
                        st.write("### Data Manager Summary")
                        data_manager = get_data_manager()
                        summary = data_manager.get_data_summary()
                        st.json(summary)
                        
                        st.write("### System Status")
                        st.write(f"**Default Period Confirmed:** {controls['period']} (Should be '1mo')")
                        st.write(f"**Volume Analysis Available:** {VOLUME_ANALYSIS_AVAILABLE}")
                        st.write(f"**Volatility Analysis Available:** {VOLATILITY_ANALYSIS_AVAILABLE}")
                        st.write(f"**Baldwin Indicator Available:** {BALDWIN_INDICATOR_AVAILABLE}")
    
    else:
        # Welcome message
        st.write("## 🚀 VWV Professional Trading System v4.2.1 - Enhanced Volume Analysis")
        st.write("**NEW FEATURES:** Comprehensive volume analysis with 14 indicators and composite scoring")
        
        # System capabilities
        with st.expander("✨ Enhanced Volume Analysis Features", expanded=True):
            st.write("""
            **🎯 Comprehensive Volume Indicators (14 Total):**
            - **On-Balance Volume (OBV)** - Accumulation/distribution analysis
            - **Money Flow Index (MFI)** - Volume-weighted RSI
            - **Accumulation/Distribution Line** - Institutional activity
            - **Chaikin Money Flow** - Money flow strength
            - **Volume Rate of Change** - Volume momentum
            - **Relative Volume** - Current vs historical comparison
            - **Force Index** - Price-volume confirmation
            - **Volume Oscillator** - Trend strength analysis
            - **Ease of Movement** - Movement efficiency
            - **Volume Price Trend** - Alternative to OBV
            - **Volume Momentum** - Acceleration analysis
            - **Volume Breakout Analysis** - Spike detection
            - **Volume Trend Analysis** - Direction consistency
            - **Volume Divergence** - Price-volume warnings
            
            **📊 Enhanced Features:**
            - **Composite Volume Score** - Weighted 0-100 scale with research-based weights
            - **Gradient Score Bar** - Professional visual display matching technical analysis
            - **Component Breakdown** - Detailed analysis of all 14 indicators
            - **Trading Implications** - Strategic guidance based on volume regime
            """)
        
        # Baldwin Market Preview (if available)
        if BALDWIN_INDICATOR_AVAILABLE:
            with st.expander("🚦 Live Baldwin Market Regime Preview", expanded=True):
                with st.spinner("Calculating market regime..."):
                    try:
                        baldwin_results = calculate_baldwin_indicator_complete()
                        if baldwin_results and 'error' not in baldwin_results:
                            formatted_results = format_baldwin_for_display(baldwin_results)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Current Market Regime", formatted_results.get('regime', 'Unknown'))
                            with col2:
                                st.metric("Regime Confidence", formatted_results.get('confidence', 'Low'))
                        else:
                            st.info("Baldwin indicator data unavailable")
                    except:
                        st.info("Baldwin indicator temporarily unavailable")

if __name__ == "__main__":
    main()
