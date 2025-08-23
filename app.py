"""
VWV Trading System v4.2.1 - Main Streamlit Application
FIXED VERSION - Includes Volume and Volatility Analysis integration
Enhanced Professional Trading Analysis Platform with Modular Architecture

Date: August 22, 2025 - 10:15 AM EST
Version: v4.2.1-FIXED
Status: Volume and Volatility Analysis Integration CORRECTED
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="VWV Professional Trading Analysis",
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modular components with safe fallbacks
try:
    from config.settings import *
    from config.constants import QUICK_LINKS, SYMBOL_DESCRIPTIONS
except ImportError as e:
    st.error(f"Configuration import error: {e}")
    st.stop()

try:
    from data.fetcher import get_market_data_enhanced
    from data.manager import get_data_manager
except ImportError as e:
    st.error(f"Data module import error: {e}")
    st.stop()

try:
    from analysis.technical import (
        calculate_daily_vwap, calculate_fibonacci_emas, 
        calculate_point_of_control_enhanced, calculate_weekly_deviations,
        calculate_comprehensive_technicals, calculate_vwv_system_complete, 
        calculate_confidence_intervals
    )
    from analysis.fundamental import calculate_graham_score, calculate_piotroski_score
    from analysis.market import calculate_market_correlations_enhanced
    from analysis.options import calculate_options_levels_enhanced
    
    # NEW v4.2.1 - Volume Analysis
    try:
        from analysis.volume import calculate_complete_volume_analysis
        VOLUME_ANALYSIS_AVAILABLE = True
    except ImportError:
        VOLUME_ANALYSIS_AVAILABLE = False
        st.sidebar.warning("⚠️ Volume Analysis module not available")
    
    # NEW v4.2.1 - Volatility Analysis  
    try:
        from analysis.volatility import calculate_complete_volatility_analysis
        VOLATILITY_ANALYSIS_AVAILABLE = True
    except ImportError:
        VOLATILITY_ANALYSIS_AVAILABLE = False
        st.sidebar.warning("⚠️ Volatility Analysis module not available")
        
except ImportError as e:
    st.error(f"Analysis module import error: {e}")
    st.stop()

try:
    from ui.components import create_header, create_technical_score_bar
    # NEW v4.2.1 - Enhanced UI Components
    try:
        from ui.components import create_volume_score_bar, create_volatility_score_bar
        ENHANCED_UI_AVAILABLE = True
    except ImportError:
        ENHANCED_UI_AVAILABLE = False
except ImportError as e:
    st.error(f"UI components import error: {e}")
    st.stop()

try:
    from utils.helpers import format_large_number, is_etf, add_to_recently_viewed
    from utils.formatters import format_percentage, format_currency
except ImportError as e:
    st.error(f"Utilities import error: {e}")
    st.stop()

# Initialize session state variables
def initialize_session_state():
    """Initialize session state with default values"""
    defaults = {
        'show_charts': True,
        'show_technical_analysis': True,
        'show_volume_analysis': True,  # NEW v4.2.1
        'show_volatility_analysis': True,  # NEW v4.2.1
        'show_fundamental_analysis': True,
        'show_baldwin_indicator': True,
        'show_market_correlation': True,
        'show_options_analysis': True,
        'show_confidence_intervals': True,
        'recently_viewed_symbols': [],
        'current_symbol': None,
        'analysis_results': None,
        'chart_data': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def create_sidebar_controls():
    """Create sidebar controls and return configuration"""
    st.sidebar.title("📊 VWV Trading Analysis v4.2.1")
    
    # Symbol input
    symbol = st.sidebar.text_input(
        "Enter Stock Symbol", 
        value="AAPL",
        placeholder="e.g., AAPL, MSFT, TSLA"
    ).upper().strip()
    
    # Period selection  
    period = st.sidebar.selectbox(
        "Analysis Period",
        options=['1mo', '3mo', '6mo', '1y', '2y'],
        index=0,  # Default to 1 month
        help="Time period for historical data analysis"
    )
    
    # Analysis button
    analyze_button = st.sidebar.button(
        "🚀 Analyze",
        type="primary",
        use_container_width=True
    )
    
    # Section toggles
    st.sidebar.markdown("### 📋 Analysis Sections")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.session_state.show_charts = st.checkbox("📊 Charts", value=st.session_state.show_charts)
        st.session_state.show_technical_analysis = st.checkbox("🔴 Technical", value=st.session_state.show_technical_analysis)
        st.session_state.show_volume_analysis = st.checkbox("📊 Volume", value=st.session_state.show_volume_analysis)  # NEW v4.2.1
        st.session_state.show_volatility_analysis = st.checkbox("📊 Volatility", value=st.session_state.show_volatility_analysis)  # NEW v4.2.1
        st.session_state.show_fundamental_analysis = st.checkbox("📈 Fundamental", value=st.session_state.show_fundamental_analysis)
    
    with col2:
        st.session_state.show_baldwin_indicator = st.checkbox("🚦 Baldwin", value=st.session_state.show_baldwin_indicator)
        st.session_state.show_market_correlation = st.checkbox("🌐 Correlation", value=st.session_state.show_market_correlation)
        st.session_state.show_options_analysis = st.checkbox("🎯 Options", value=st.session_state.show_options_analysis)
        st.session_state.show_confidence_intervals = st.checkbox("📊 Confidence", value=st.session_state.show_confidence_intervals)
    
    # Debug toggle
    show_debug = st.sidebar.checkbox("🔧 Debug Mode", value=False)
    
    # Quick Links
    st.sidebar.markdown("### 🔗 Quick Links")
    for category, symbols in QUICK_LINKS.items():
        with st.sidebar.expander(f"{category}"):
            for symbol_item in symbols:
                if st.button(f"{symbol_item}", key=f"quick_{symbol_item}"):
                    st.session_state['quick_symbol'] = symbol_item
                    st.rerun()
    
    # Recently viewed
    if st.session_state.recently_viewed_symbols:
        st.sidebar.markdown("### 🕒 Recently Viewed")
        for recent_symbol in st.session_state.recently_viewed_symbols[-5:]:
            if st.sidebar.button(f"{recent_symbol}", key=f"recent_{recent_symbol}"):
                st.session_state['quick_symbol'] = recent_symbol
                st.rerun()
    
    # Handle quick symbol selection
    if 'quick_symbol' in st.session_state:
        symbol = st.session_state['quick_symbol']
        del st.session_state['quick_symbol']
    
    return {
        'symbol': symbol,
        'period': period,
        'analyze_button': analyze_button,
        'show_debug': show_debug
    }

def show_individual_technical_analysis(analysis_results, show_debug=False):
    """Display individual technical analysis section with VWV composite scoring"""
    if not st.session_state.show_technical_analysis:
        return
        
    with st.expander(f"🔴 {analysis_results['symbol']} - Individual Technical Analysis", expanded=True):
        
        # Get VWV analysis
        vwv_analysis = analysis_results.get('vwv_analysis', {})
        
        if vwv_analysis and 'composite_score' in vwv_analysis:
            # Create technical score bar
            create_technical_score_bar(vwv_analysis['composite_score'])
            
            # VWV System metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                vix_fix = vwv_analysis.get('vix_fix', {})
                vix_fix_value = vix_fix.get('current_value', 0)
                st.metric("Williams VIX Fix", f"{vix_fix_value:.2f}")
                
            with col2:
                stoch = vwv_analysis.get('stochastic', {})
                stoch_k = stoch.get('k_percent', 0)
                st.metric("Stochastic %K", f"{stoch_k:.1f}")
                
            with col3:
                rsi = vwv_analysis.get('rsi', {})
                rsi_value = rsi.get('current_rsi', 50)
                st.metric("RSI (14)", f"{rsi_value:.1f}")
                
            with col4:
                composite_score = vwv_analysis.get('composite_score', 50)
                st.metric("VWV Score", f"{composite_score:.1f}/100")
            
            # Market signals and conditions
            st.subheader("📊 Market Signals")
            
            col1, col2 = st.columns(2)
            with col1:
                market_condition = vwv_analysis.get('market_condition', 'Neutral')
                st.info(f"**Market Condition:** {market_condition}")
                
                trend_strength = vwv_analysis.get('trend_strength', 'Moderate')
                st.info(f"**Trend Strength:** {trend_strength}")
                
            with col2:
                buy_signals = vwv_analysis.get('buy_signals', [])
                sell_signals = vwv_analysis.get('sell_signals', [])
                
                if buy_signals:
                    st.success(f"**Buy Signals:** {', '.join(buy_signals)}")
                if sell_signals:
                    st.error(f"**Sell Signals:** {', '.join(sell_signals)}")
                if not buy_signals and not sell_signals:
                    st.warning("**No clear signals at this time**")
        
        else:
            st.warning("⚠️ Technical analysis not available - insufficient data")

def show_volume_analysis(analysis_results, show_debug=False):
    """Display volume analysis section - NEW v4.2.1"""
    if not st.session_state.show_volume_analysis or not VOLUME_ANALYSIS_AVAILABLE:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Volume Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volume_analysis = enhanced_indicators.get('volume_analysis', {})
        
        if 'error' not in volume_analysis and volume_analysis:
            
            # Create volume score bar if enhanced UI available
            if ENHANCED_UI_AVAILABLE and 'volume_score' in volume_analysis:
                create_volume_score_bar(volume_analysis['volume_score'])
            
            # Primary volume metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                current_volume = volume_analysis.get('current_volume', 0)
                st.metric("Current Volume", format_large_number(current_volume))
            with col2:
                volume_5d_avg = volume_analysis.get('volume_5d_avg', 0)
                st.metric("5D Avg Volume", format_large_number(volume_5d_avg))
            with col3:
                volume_ratio = volume_analysis.get('volume_ratio', 1.0)
                st.metric("Volume Ratio", f"{volume_ratio:.2f}x", f"vs 30D avg")
            with col4:
                volume_trend = volume_analysis.get('volume_5d_trend', 0)
                st.metric("5D Volume Trend", f"{volume_trend:+.2f}%")
            
            # Volume regime and implications
            st.subheader("📊 Volume Environment")
            volume_regime = volume_analysis.get('volume_regime', 'Unknown')
            volume_score = volume_analysis.get('volume_score', 50)
            trading_implications = volume_analysis.get('trading_implications', 'No implications available')
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Volume Regime:** {volume_regime}")
                st.info(f"**Volume Score:** {volume_score}/100")
            with col2:
                st.info(f"**Trading Implications:**\n{trading_implications}")
            
            # Component breakdown (if available)
            if ENHANCED_UI_AVAILABLE and 'indicators' in volume_analysis:
                with st.expander("🔍 Volume Component Breakdown"):
                    indicators = volume_analysis.get('indicators', {})
                    scores = volume_analysis.get('scores', {})
                    weights = volume_analysis.get('weights', {})
                    
                    breakdown_data = []
                    for indicator, value in indicators.items():
                        score = scores.get(indicator, 0)
                        weight = weights.get(indicator, 0)
                        contribution = score * weight
                        
                        breakdown_data.append({
                            'Indicator': indicator.replace('_', ' ').title(),
                            'Value': f"{value:.2f}" if isinstance(value, (int, float)) else str(value),
                            'Score': f"{score:.1f}",
                            'Weight': f"{weight:.3f}",
                            'Contribution': f"{contribution:.2f}"
                        })
                    
                    if breakdown_data:
                        df_breakdown = pd.DataFrame(breakdown_data)
                        st.dataframe(df_breakdown, use_container_width=True, hide_index=True)
                
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
            
            # Create volatility score bar if enhanced UI available
            if ENHANCED_UI_AVAILABLE and 'volatility_score' in volatility_analysis:
                create_volatility_score_bar(volatility_analysis['volatility_score'])
            
            # Primary volatility metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                vol_20d = volatility_analysis.get('volatility_20d', 0)
                st.metric("20D Volatility", f"{vol_20d:.2f}%")
            with col2:
                vol_10d = volatility_analysis.get('volatility_10d', 0)
                st.metric("10D Volatility", f"{vol_10d:.2f}%")
            with col3:
                vol_percentile = volatility_analysis.get('volatility_percentile', 50)
                st.metric("Vol Percentile", f"{vol_percentile:.1f}%")
            with col4:
                vol_rank = volatility_analysis.get('volatility_rank', 50)
                st.metric("Vol Rank", f"{vol_rank:.1f}%")
            
            # Advanced volatility metrics
            st.subheader("🔬 Advanced Volatility Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                realized_vol = volatility_analysis.get('realized_volatility', 0)
                st.metric("Realized Vol", f"{realized_vol:.2f}%")
            with col2:
                garch_vol = volatility_analysis.get('garch_volatility', 0)
                st.metric("GARCH Vol", f"{garch_vol:.2f}%")
            with col3:
                vol_momentum = volatility_analysis.get('volatility_momentum', 0)
                st.metric("Vol Momentum", f"{vol_momentum:+.2f}%")
            with col4:
                vol_clustering = volatility_analysis.get('volatility_clustering', 0)
                st.metric("Vol Clustering", f"{vol_clustering:.3f}")
            
            # Volatility regime and options guidance
            st.subheader("📊 Volatility Environment & Options Strategy")
            vol_regime = volatility_analysis.get('volatility_regime', 'Unknown')
            vol_score = volatility_analysis.get('volatility_score', 50)
            options_strategy = volatility_analysis.get('options_strategy', 'No strategy available')
            trading_implications = volatility_analysis.get('trading_implications', 'No implications available')
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Volatility Regime:** {vol_regime}")
                st.info(f"**Volatility Score:** {vol_score}/100")
            with col2:
                st.info(f"**Options Strategy:** {options_strategy}")
                st.info(f"**Trading Implications:**\n{trading_implications}")
            
            # Component breakdown (if available)
            if ENHANCED_UI_AVAILABLE and 'indicators' in volatility_analysis:
                with st.expander("🔍 Volatility Component Breakdown"):
                    indicators = volatility_analysis.get('indicators', {})
                    scores = volatility_analysis.get('scores', {})
                    weights = volatility_analysis.get('weights', {})
                    
                    st.markdown("### 📊 Weighting Methodology")
                    st.markdown("""
                    **Research-Based Indicator Weights**: Each volatility indicator is weighted based on academic research 
                    and practical trading effectiveness. Higher weights are assigned to more reliable and 
                    widely-used volatility measures.
                    """)
                    
                    breakdown_data = []
                    for indicator, value in indicators.items():
                        score = scores.get(indicator, 0)
                        weight = weights.get(indicator, 0)
                        contribution = score * weight
                        
                        breakdown_data.append({
                            'Indicator': indicator.replace('_', ' ').title(),
                            'Value': f"{value:.3f}" if isinstance(value, (int, float)) else str(value),
                            'Score': f"{score:.1f}",
                            'Weight': f"{weight:.3f}",
                            'Contribution': f"{contribution:.2f}"
                        })
                    
                    if breakdown_data:
                        df_breakdown = pd.DataFrame(breakdown_data)
                        st.dataframe(df_breakdown, use_container_width=True, hide_index=True)
                        
                        # Show risk-adjusted metrics
                        st.markdown("### 📈 Risk-Adjusted Metrics")
                        volatility_strength_factor = volatility_analysis.get('volatility_strength_factor', 1.0)
                        st.metric("Volatility Strength Factor", f"{volatility_strength_factor:.2f}x", 
                                 help="Multiplier applied to technical analysis based on volatility environment")
                
        else:
            st.warning("⚠️ Volatility analysis not available - insufficient data")

def show_fundamental_analysis(analysis_results, show_debug=False):
    """Display fundamental analysis section"""
    if not st.session_state.show_fundamental_analysis:
        return
        
    enhanced_indicators = analysis_results.get('enhanced_indicators', {})
    graham_score = enhanced_indicators.get('graham_score', {})
    piotroski_score = enhanced_indicators.get('piotroski_score', {})
    
    with st.expander(f"📈 {analysis_results['symbol']} - Fundamental Analysis", expanded=True):
        
        if 'error' not in graham_score and 'error' not in piotroski_score:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Benjamin Graham Score")
                graham_total = graham_score.get('score', 0)
                graham_max = graham_score.get('total_possible', 10)
                st.metric("Graham Score", f"{graham_total}/{graham_max}")
                
                if 'criteria' in graham_score:
                    for criterion in graham_score['criteria']:
                        status = "✅" if criterion['passed'] else "❌"
                        st.write(f"{status} {criterion['name']}: {criterion['value']}")
            
            with col2:
                st.subheader("📊 Piotroski F-Score")
                piotroski_total = piotroski_score.get('score', 0)
                piotroski_max = piotroski_score.get('total_possible', 9)
                st.metric("Piotroski Score", f"{piotroski_total}/{piotroski_max}")
                
                if 'criteria' in piotroski_score:
                    for criterion in piotroski_score['criteria']:
                        status = "✅" if criterion['passed'] else "❌"
                        st.write(f"{status} {criterion['name']}: {criterion['value']}")
        
        else:
            if 'error' in graham_score:
                st.info(f"Graham Analysis: {graham_score['error']}")
            if 'error' in piotroski_score:
                st.info(f"Piotroski Analysis: {piotroski_score['error']}")

def show_baldwin_indicator(analysis_results, show_debug=False):
    """Display Baldwin Market Regime Indicator - PLACEHOLDER"""
    if not st.session_state.show_baldwin_indicator:
        return
        
    with st.expander("🚦 Baldwin Market Regime Indicator", expanded=False):
        st.info("Baldwin Indicator analysis will be available in a future update.")
        # TODO: Implement Baldwin Indicator module

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis section"""
    if not st.session_state.show_market_correlation:
        return
        
    enhanced_indicators = analysis_results.get('enhanced_indicators', {})
    market_correlations = enhanced_indicators.get('market_correlations', {})
    
    with st.expander(f"🌐 {analysis_results['symbol']} - Market Correlation Analysis", expanded=True):
        
        if market_correlations and 'error' not in market_correlations:
            
            # Correlation metrics
            st.subheader("📊 ETF Correlations")
            
            correlation_data = []
            for etf, data in market_correlations.items():
                if isinstance(data, dict) and 'correlation' in data:
                    correlation_data.append({
                        'ETF': etf,
                        'Correlation': f"{data.get('correlation', 0):.3f}",
                        'Beta': f"{data.get('beta', 0):.2f}",
                        'Relationship': data.get('relationship', 'Unknown')
                    })
            
            if correlation_data:
                df_correlations = pd.DataFrame(correlation_data)
                st.dataframe(df_correlations, use_container_width=True, hide_index=True)
            else:
                st.warning("No correlation data available")
                
        else:
            st.warning("⚠️ Market correlation analysis not available")

def show_options_analysis(analysis_results, show_debug=False):
    """Display options analysis section"""
    if not st.session_state.show_options_analysis:
        return
        
    enhanced_indicators = analysis_results.get('enhanced_indicators', {})
    options_levels = enhanced_indicators.get('options_levels', {})
    
    with st.expander(f"🎯 {analysis_results['symbol']} - Options Analysis", expanded=True):
        
        if options_levels and 'error' not in options_levels:
            current_price = analysis_results.get('current_price', 0)
            st.metric("Current Price", f"${current_price:.2f}")
            
            # Options levels by risk category
            for risk_level, levels in options_levels.items():
                if isinstance(levels, dict):
                    st.subheader(f"📊 {risk_level.title()} Risk Options")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Call Levels:**")
                        if 'calls' in levels:
                            for strike, data in levels['calls'].items():
                                st.write(f"${strike}: {data.get('description', 'N/A')}")
                    
                    with col2:
                        st.write("**Put Levels:**")
                        if 'puts' in levels:
                            for strike, data in levels['puts'].items():
                                st.write(f"${strike}: {data.get('description', 'N/A')}")
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

def perform_enhanced_analysis(symbol, period, show_debug=False):
    """Perform enhanced analysis using modular components - FIXED v4.2.1 with Volume/Volatility"""
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
        
        # Step 5: Calculate Volume Analysis (NEW v4.2.1 - FIXED)
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
        else:
            volume_analysis = {'error': 'Volume analysis module not available'}
        
        # Step 6: Calculate Volatility Analysis (NEW v4.2.1 - FIXED)
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
        else:
            volatility_analysis = {'error': 'Volatility analysis module not available'}
        
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
        
        # Step 9: Calculate options levels
        volatility = comprehensive_technicals.get('volatility_20d', 20)
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
        
        # Step 10: Calculate VWV System Analysis
        vwv_results = calculate_vwv_system_complete(analysis_input, symbol, DEFAULT_VWV_CONFIG)
        
        # Step 11: Calculate confidence intervals
        confidence_analysis = calculate_confidence_intervals(analysis_input)
        
        # Step 12: Build analysis results - FIXED to include Volume/Volatility
        current_date = analysis_input.index[-1].strftime('%Y-%m-%d')
        
        analysis_results = {
            'symbol': symbol,
            'timestamp': current_date,
            'current_price': current_price,
            'enhanced_indicators': {
                'daily_vwap': daily_vwap,
                'fibonacci_emas': fibonacci_emas,
                'point_of_control': point_of_control,
                'weekly_deviations': weekly_deviations,
                'comprehensive_technicals': comprehensive_technicals,
                'volume_analysis': volume_analysis,  # NEW v4.2.1 - FIXED
                'volatility_analysis': volatility_analysis,  # NEW v4.2.1 - FIXED
                'market_correlations': market_correlations,
                'options_levels': options_levels,
                'graham_score': graham_score,
                'piotroski_score': piotroski_score
            },
            'vwv_analysis': vwv_results,
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
        if show_debug:
            st.exception(e)
        return None, None

def main():
    """Main application function - CORRECTED v4.2.1 with PROPER DISPLAY ORDER"""
    # Initialize session state
    initialize_session_state()
    
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
                show_volume_analysis(analysis_results, controls['show_debug'])
                
                # 4. Volatility Analysis (Optional - when available)  
                show_volatility_analysis(analysis_results, controls['show_debug'])
                
                # 5. Fundamental Analysis
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                
                # 6. Baldwin Indicator (when available)
                show_baldwin_indicator(analysis_results, controls['show_debug'])
                
                # 7. Market Correlation Analysis
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                
                # 8. Options Analysis
                show_options_analysis(analysis_results, controls['show_debug'])
                
                # 9. Confidence Intervals
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
                # Debug information
                if controls['show_debug']:
                    with st.expander("🔧 Debug Information", expanded=False):
                        st.write("**System Status:**")
                        st.write(f"- Volume Analysis Available: {VOLUME_ANALYSIS_AVAILABLE}")
                        st.write(f"- Volatility Analysis Available: {VOLATILITY_ANALYSIS_AVAILABLE}")
                        st.write(f"- Enhanced UI Available: {ENHANCED_UI_AVAILABLE}")
                        st.write(f"- Analysis Results Keys: {list(analysis_results.keys())}")
                        if 'enhanced_indicators' in analysis_results:
                            st.write(f"- Enhanced Indicators Keys: {list(analysis_results['enhanced_indicators'].keys())}")
            else:
                st.error("❌ Analysis failed - please try a different symbol or time period")
                
    elif not controls['symbol']:
        st.info("👆 Enter a stock symbol in the sidebar to begin analysis")
        
        # Show recent symbols if available
        if st.session_state.recently_viewed_symbols:
            st.subheader("🕒 Recently Analyzed")
            recent_cols = st.columns(min(5, len(st.session_state.recently_viewed_symbols)))
            for i, recent_symbol in enumerate(st.session_state.recently_viewed_symbols[-5:]):
                with recent_cols[i]:
                    if st.button(f"📊 {recent_symbol}", key=f"main_recent_{recent_symbol}"):
                        st.session_state['quick_symbol'] = recent_symbol
                        st.rerun()
    
    else:
        st.info("🚀 Click 'Analyze' to start the analysis")

if __name__ == "__main__":
    main()
