"""
VWV Professional Trading System v4.2.1 - CORRECTED VERSION
Complete preservation of existing functionality with Volume & Volatility integration
SYNTAX ERROR FIXED - ALL CHARTS AND BALDWIN FEATURES PRESERVED
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
    from analysis.baldwin_indicator import calculate_baldwin_market_regime
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
    """Create sidebar controls and return analysis parameters"""
    st.sidebar.title("📊 Trading Analysis v4.2.1")
    
    # Initialize session state
    if 'recently_viewed' not in st.session_state:
        st.session_state.recently_viewed = []
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
    if 'show_baldwin_indicator' not in st.session_state:
        st.session_state.show_baldwin_indicator = True
    if 'show_market_correlation' not in st.session_state:
        st.session_state.show_market_correlation = True
    if 'show_options_analysis' not in st.session_state:
        st.session_state.show_options_analysis = True
    if 'show_confidence_intervals' not in st.session_state:
        st.session_state.show_confidence_intervals = True
    
    # Basic controls
    if 'selected_symbol' in st.session_state:
        default_symbol = st.session_state.selected_symbol
        del st.session_state.selected_symbol
    else:
        default_symbol = UI_SETTINGS['default_symbol']
        
    # Create form for Enter key functionality
    with st.sidebar.form(key='symbol_form'):
        symbol = st.text_input("Symbol", value=default_symbol, help="Enter stock symbol").upper()
        
        # CORRECTED: Use 1mo as default (index 0) as per requirements
        period_options = ['1mo', '3mo', '6mo', '1y', '2y']
        period = st.selectbox("Data Period", period_options, index=0)
        
        # Single analyze button in form
        analyze_button = st.form_submit_button("📊 Analyze Symbol", type="primary", use_container_width=True)
    
    # Quick Links section - FIRST POSITION as per requirements
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
                                    st.rerun()

    # Recently Viewed section - SECOND POSITION as per requirements
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

    # Section Control Panel - THIRD POSITION as per requirements
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
                    "🚦 Baldwin Market Regime", 
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

    # Debug toggle
    show_debug = st.sidebar.checkbox("🐛 Show Debug Info", value=False)
    
    return {
        'symbol': symbol,
        'period': period,
        'analyze_button': analyze_button,
        'show_debug': show_debug
    }

def add_to_recently_viewed(symbol):
    """Add symbol to recently viewed - updated for 9 symbols"""
    if symbol and symbol != "":
        if symbol in st.session_state.recently_viewed:
            st.session_state.recently_viewed.remove(symbol)
        st.session_state.recently_viewed.insert(0, symbol)
        st.session_state.recently_viewed = st.session_state.recently_viewed[:9]

def show_interactive_charts(analysis_results, data, show_debug=False):
    """Display interactive charts section - FIRST POSITION - MANDATORY"""
    if not st.session_state.show_charts:
        return
        
    with st.expander("📊 Interactive Trading Charts", expanded=True):
        try:
            # Check if we have the charts module
            try:
                from charts.plotting import display_trading_charts
                display_trading_charts(data, analysis_results)
            except ImportError as e:
                if show_debug:
                    st.error(f"Charts module import error: {str(e)}")
                
                # Fallback to simple chart using Streamlit
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
    """Display individual technical analysis section - SECOND POSITION - MANDATORY"""
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

def show_baldwin_indicator_analysis(show_debug=False):
    """Display Baldwin Market Regime Indicator - BEFORE Market Correlation"""
    if not st.session_state.show_baldwin_indicator or not BALDWIN_INDICATOR_AVAILABLE:
        return
        
    with st.expander("🚦 Baldwin Market Regime Indicator", expanded=True):
        
        try:
            # Calculate Baldwin indicator
            baldwin_results = calculate_baldwin_market_regime(show_debug=show_debug)
            
            if baldwin_results and 'error' not in baldwin_results:
                
                # Main regime display
                regime_score = baldwin_results.get('regime_score', 50)
                regime_color = baldwin_results.get('regime_color', '🟡')
                regime_status = baldwin_results.get('regime_status', 'CAUTION')
                regime_description = baldwin_results.get('regime_description', 'Market regime assessment')
                
                # Header with regime status
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"""
                    <div style="padding: 1rem; background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05)); 
                                border-left: 4px solid {'#00C851' if regime_color == '🟢' else '#FF8800' if regime_color == '🟡' else '#FF4444'}; 
                                border-radius: 8px;">
                        <h3 style="margin: 0;">{regime_color} Market Regime: {regime_status}</h3>
                        <p style="margin: 0.5rem 0 0 0; color: #666;">{regime_description}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Regime Score", f"{regime_score:.1f}/100")
                
                with col3:
                    market_action = baldwin_results.get('market_action', 'Monitor')
                    st.metric("Recommended Action", market_action)
                
                # Component breakdown
                components = baldwin_results.get('components', {})
                if components:
                    st.subheader("📊 Component Analysis")
                    
                    # Momentum component
                    momentum_data = components.get('momentum', {})
                    if momentum_data:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Momentum Component (60% weight):**")
                            momentum_score = momentum_data.get('score', 50)
                            st.progress(momentum_score / 100)
                            st.write(f"Score: {momentum_score:.1f}/100")
                            
                            # Momentum details
                            broad_market = momentum_data.get('broad_market_trend', {})
                            if broad_market:
                                st.write(f"• SPY Trend: {broad_market.get('spy_trend', 'Unknown')}")
                                st.write(f"• QQQ Trend: {broad_market.get('qqq_trend', 'Unknown')}")
                        
                        with col2:
                            # Market internals
                            market_internals = momentum_data.get('market_internals', {})
                            if market_internals:
                                st.write("**Market Internals:**")
                                iwm_signal = market_internals.get('iwm_signal', 'Unknown')
                                st.write(f"• IWM Signal: {iwm_signal}")
                                
                            # Fear indicators
                            fear_indicators = momentum_data.get('fear_indicators', {})
                            if fear_indicators:
                                st.write("**Fear Indicators:**")
                                vix_level = fear_indicators.get('vix_level', 0)
                                fngd_spike = fear_indicators.get('fngd_spike', False)
                                st.write(f"• VIX Level: {vix_level:.1f}")
                                st.write(f"• FNGD Spike: {'Yes' if fngd_spike else 'No'}")
                    
                    # Liquidity and Sentiment components
                    liquidity_data = components.get('liquidity', {})
                    sentiment_data = components.get('sentiment', {})
                    
                    if liquidity_data or sentiment_data:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if liquidity_data:
                                st.write("**Liquidity Component (25% weight):**")
                                liquidity_score = liquidity_data.get('score', 50)
                                st.progress(liquidity_score / 100)
                                st.write(f"Score: {liquidity_score:.1f}/100")
                                
                                dollar_strength = liquidity_data.get('dollar_strength', {})
                                if dollar_strength:
                                    uup_trend = dollar_strength.get('uup_trend', 'Unknown')
                                    st.write(f"• Dollar Trend (UUP): {uup_trend}")
                                
                                treasury_flight = liquidity_data.get('treasury_flight', {})
                                if treasury_flight:
                                    tlt_demand = treasury_flight.get('tlt_demand', 'Unknown')
                                    st.write(f"• Treasury Demand (TLT): {tlt_demand}")
                        
                        with col2:
                            if sentiment_data:
                                st.write("**Sentiment Component (15% weight):**")
                                sentiment_score = sentiment_data.get('score', 50)
                                st.progress(sentiment_score / 100)
                                st.write(f"Score: {sentiment_score:.1f}/100")
                                
                                insider_activity = sentiment_data.get('insider_activity', {})
                                if insider_activity:
                                    buy_sell_ratio = insider_activity.get('buy_sell_ratio', 1.0)
                                    st.write(f"• Insider Buy/Sell Ratio: {buy_sell_ratio:.2f}")
                
                # Trading implications
                implications = baldwin_results.get('trading_implications', [])
                if implications:
                    st.subheader("📋 Trading Implications")
                    for implication in implications:
                        st.write(f"• {implication}")
                
            else:
                error_msg = baldwin_results.get('error', 'Baldwin calculation failed') if baldwin_results else 'Baldwin indicator not available'
                st.warning(f"⚠️ {error_msg}")
                if show_debug and baldwin_results:
                    st.error(f"Debug info: {baldwin_results}")
        
        except Exception as e:
            st.error(f"❌ Baldwin indicator error: {str(e)}")
            if show_debug:
                st.exception(e)

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
    """Display options analysis section - SYNTAX ERROR FIXED"""
    if not st.session_state.show_options_analysis:
        return
        
    with st.expander("🎯 Options Trading Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        options_levels = enhanced_indicators.get('options_levels', [])
        
        if options_levels:
            st.subheader("💰 Premium Selling Levels with Greeks")
            st.write("**Enhanced option strike levels with Delta, Theta, and Beta**")
            
            # SYNTAX ERROR FIXED: Clean up expected move data properly
            cleaned_options_levels = []
            for level in options_levels:
                cleaned_level = level.copy()
                
                # Fix expected move field if it contains problematic characters
                if 'Expected Move' in cleaned_level:
                    expected_move = cleaned_level['Expected Move']
                    
                    # Handle string or numeric expected move values
                    if isinstance(expected_move, str):
                        # Remove problematic characters and clean the string
                        cleaned_move = expected_move.replace('±', '').replace('$', '').strip()
                        try:
                            # Try to convert to float and reformat
                            move_value = float(cleaned_move)
                            cleaned_level['Expected Move'] = f"±${move_value:.2f}"
                        except (ValueError, TypeError):
                            # If conversion fails, keep original or set default
                            cleaned_level['Expected Move'] = "±N/A"
                    elif isinstance(expected_move, (int, float)):
                        # If it's already numeric, format it properly
                        cleaned_level['Expected Move'] = f"±${float(expected_move):.2f}"
                    else:
                        # Fallback for any other type
                        cleaned_level['Expected Move'] = "±N/A"
                
                cleaned_options_levels.append(cleaned_level)
            
            # Display the cleaned options data
            df_options = pd.DataFrame(cleaned_options_levels)
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
        
        # Step 9: Add additional data to enhanced indicators
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
        
        # Get chart data
        chart_data = data_manager.get_market_data_for_chart(symbol)
        
        return analysis_results, chart_data
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        return None, None

def main():
    """Main application function - ENHANCED v4.2.1 - ALL FEATURES PRESERVED"""
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
                # MANDATORY DISPLAY ORDER - DO NOT CHANGE
                
                # 1. INTERACTIVE CHARTS - FIRST POSITION - MANDATORY
                show_interactive_charts(analysis_results, chart_data, controls['show_debug'])
                
                # 2. INDIVIDUAL TECHNICAL ANALYSIS - SECOND POSITION - MANDATORY
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                
                # 3. VOLUME ANALYSIS - NEW v4.2.1
                if VOLUME_ANALYSIS_AVAILABLE:
                    show_volume_analysis(analysis_results, controls['show_debug'])
                
                # 4. VOLATILITY ANALYSIS - NEW v4.2.1
                if VOLATILITY_ANALYSIS_AVAILABLE:
                    show_volatility_analysis(analysis_results, controls['show_debug'])
                
                # 5. FUNDAMENTAL ANALYSIS - PRESERVED
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                
                # 6. BALDWIN MARKET REGIME INDICATOR - BEFORE MARKET CORRELATION
                show_baldwin_indicator_analysis(controls['show_debug'])
                
                # 7. MARKET CORRELATION & BREAKOUTS - PRESERVED
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                
                # 8. OPTIONS ANALYSIS - PRESERVED WITH SYNTAX FIX
                show_options_analysis(analysis_results, controls['show_debug'])
                
                # 9. CONFIDENCE INTERVALS - PRESERVED
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
                        st.write(f"**Volume Analysis Available:** {VOLUME_ANALYSIS_AVAILABLE}")
                        st.write(f"**Volatility Analysis Available:** {VOLATILITY_ANALYSIS_AVAILABLE}")
                        st.write(f"**Baldwin Indicator Available:** {BALDWIN_INDICATOR_AVAILABLE}")
    
    else:
        # Welcome message
        st.write("## 🚀 VWV Professional Trading System v4.2.1 Enhanced")
        st.write("**Complete modular architecture with Charts, Baldwin, Volume & Volatility analysis**")
        
        with st.expander("🏗️ System Status v4.2.1 - ALL FEATURES", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📁 **Core Modules**")
                st.write("✅ **config/** - Settings, constants, parameters")
                st.write("✅ **data/** - Fetching, validation, management")
                st.write("✅ **analysis/** - Technical, volume, volatility, fundamental, market, options, baldwin")
                st.write("✅ **ui/** - Components, headers, score bars")
                st.write("✅ **utils/** - Helpers, decorators, formatters")
                
            with col2:
                st.write("### 🎯 **All Sections Working**")
                st.write("• **📊 Interactive Charts** ✅ FIRST POSITION")
                st.write("• **📊 Individual Technical Analysis** ✅ SECOND POSITION")
                if VOLUME_ANALYSIS_AVAILABLE:
                    st.write("• **🆕 Volume Analysis** ✅")
                else:
                    st.write("• **Volume Analysis** ⚠️ (Module not available)")
                if VOLATILITY_ANALYSIS_AVAILABLE:
                    st.write("• **🆕 Volatility Analysis** ✅")
                else:
                    st.write("• **Volatility Analysis** ⚠️ (Module not available)")
                st.write("• **Fundamental Analysis** ✅")
                if BALDWIN_INDICATOR_AVAILABLE:
                    st.write("• **🚦 Baldwin Market Regime** ✅")
                else:
                    st.write("• **🚦 Baldwin Market Regime** ⚠️ (Module not available)")
                st.write("• **🌐 Market Correlation & Breakouts** ✅")
                st.write("• **🎯 Options Analysis with Greeks** ✅")
                st.write("• **📊 Statistical Confidence Intervals** ✅")
        
        # Show current market status
        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.write("1. **Charts FIRST** - Interactive charts display immediately at the top")
            st.write("2. **Technical Analysis SECOND** - Professional score bar and Fibonacci EMAs")  
            st.write("3. **Baldwin Market Regime** - Traffic-light system for overall market")
            st.write("4. **Complete Analysis Pipeline** - All sections available with toggle control")
            st.write("5. **Use Quick Links** - Instant analysis of popular symbols with all features")
            st.write("6. **1 Month Default** - Optimized for recent market action")

    # Footer
    st.markdown("---")
    st.write("### 📊 System Information v4.2.1 - ALL FEATURES")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Version:** VWV Professional v4.2.1 Complete")
        st.write(f"**Architecture:** Full Modular with Charts + Baldwin")
    with col2:
        st.write(f"**Status:** ✅ All Features Active - Syntax Fixed")
        st.write(f"**Display Order:** Charts → Technical → Baldwin → Options")
    with col3:
        st.write(f"**Default Period:** 1 Month (Optimized)")
        st.write(f"**Comprehensive:** Charts + Baldwin + All Analysis")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.write("Please refresh the page and try again.")
        
        if st.checkbox("Show Error Details"):
            st.exception(e)
