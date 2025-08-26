"""
VWV Professional Trading System - Working Baseline v4.2.1
Version: v4.2.1-SIDEBAR-FIX-2025-08-26-16-18-45-EST
Based on test_imports.py structure + minimal Volume/Volatility additions
Last Updated: August 26, 2025 - 4:18 PM EST
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
from analysis.options_advanced import (
    calculate_complete_advanced_options,
    format_advanced_options_for_display
)
from analysis.vwv_core import (
    calculate_vwv_system_complete,
    get_vwv_signal_interpretation
)
from ui.components import (
    create_technical_score_bar,
    create_header
)
from charts.plotting import display_trading_charts
from utils.helpers import format_large_number, get_market_status, get_etf_description
from utils.decorators import safe_calculation_wrapper

# Volume and Volatility imports with fallbacks
try:
    from analysis.volume import calculate_simple_volume_analysis
    VOLUME_ANALYSIS_AVAILABLE = True
except ImportError:
    VOLUME_ANALYSIS_AVAILABLE = False

try:
    from analysis.volatility import calculate_simple_volatility_analysis
    VOLATILITY_ANALYSIS_AVAILABLE = True
except ImportError:
    VOLATILITY_ANALYSIS_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

# Page configuration
st.set_page_config(
    page_title="VWV Professional Trading System",
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
    if 'show_charts' not in st.session_state:
        st.session_state.show_charts = True
    if 'auto_analyze' not in st.session_state:
        st.session_state.auto_analyze = False
    if 'show_risk_management' not in st.session_state:
        st.session_state.show_risk_management = True
    # Volume and Volatility toggles
    if 'show_volume_analysis' not in st.session_state:
        st.session_state.show_volume_analysis = True
    if 'show_volatility_analysis' not in st.session_state:
        st.session_state.show_volatility_analysis = True
    
    # Handle selected symbol from quicklinks/recents
    if 'selected_symbol' in st.session_state:
        current_symbol = st.session_state.selected_symbol
        st.session_state.auto_analyze = True
        del st.session_state.selected_symbol
    else:
        current_symbol = UI_SETTINGS['default_symbol']
        
    # Symbol input and period selection
    symbol = st.sidebar.text_input("Symbol", value=current_symbol, help="Enter stock symbol").upper()
    
    # Period selection with 1mo default
    period_options = ['1mo', '3mo', '6mo', '1y', '2y']
    period = st.sidebar.selectbox("Data Period", period_options, index=0)
    
    # Analyze button
    analyze_button = st.sidebar.button("📊 Analyze Symbol", type="primary", use_container_width=True)
    
    # Check for auto-analyze trigger
    if st.session_state.auto_analyze:
        st.session_state.auto_analyze = False
        analyze_button = True
    
    # Recently viewed section - FIRST
    if len(st.session_state.recently_viewed) > 0:
        with st.sidebar.expander("🕒 Recently Viewed"):
            recent_cols = st.columns(3)
            for i, recent_symbol in enumerate(st.session_state.recently_viewed[:9]):
                col_idx = i % 3
                with recent_cols[col_idx]:
                    if st.button(recent_symbol, key=f"recent_{recent_symbol}_{i}", help=SYMBOL_DESCRIPTIONS.get(recent_symbol, f"{recent_symbol} - Recently viewed symbol"), use_container_width=True):
                        st.session_state.selected_symbol = recent_symbol
                        st.session_state.auto_analyze = True
                        st.rerun()

    # Quick Links section - SECOND
    with st.sidebar.expander("🔗 Quick Links"):
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
                                    st.session_state.auto_analyze = True
                                    st.rerun()

    # Analysis sections checkboxes - THIRD - In collapsed expander
    with st.sidebar.expander("📋 Analysis Sections", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.show_charts = st.checkbox("📊 Charts", value=st.session_state.show_charts)
            st.session_state.show_vwv_analysis = st.checkbox("🔴 VWV/Tech", value=st.session_state.show_vwv_analysis)
            if VOLUME_ANALYSIS_AVAILABLE:
                st.session_state.show_volume_analysis = st.checkbox("📊 Volume", value=st.session_state.show_volume_analysis)
            if VOLATILITY_ANALYSIS_AVAILABLE:
                st.session_state.show_volatility_analysis = st.checkbox("📊 Volatility", value=st.session_state.show_volatility_analysis)
            st.session_state.show_fundamental_analysis = st.checkbox("📈 Fundamental", value=st.session_state.show_fundamental_analysis)
        
        with col2:
            st.session_state.show_market_correlation = st.checkbox("🌐 Correlation", value=st.session_state.show_market_correlation)
            st.session_state.show_options_analysis = st.checkbox("🎯 Options", value=st.session_state.show_options_analysis)
            st.session_state.show_confidence_intervals = st.checkbox("📊 Confidence", value=st.session_state.show_confidence_intervals)
            st.session_state.show_risk_management = st.checkbox("🎯 Risk Mgmt", value=st.session_state.show_risk_management)

    # Debug toggle
    show_debug = st.sidebar.checkbox("🐛 Show Debug Info", value=False)
    
    return {
        'symbol': symbol,
        'period': period,
        'analyze_button': analyze_button,
        'show_debug': show_debug
    }

def add_to_recently_viewed(symbol):
    """Add symbol to recently viewed"""
    if symbol and symbol != "":
        if symbol in st.session_state.recently_viewed:
            st.session_state.recently_viewed.remove(symbol)
        st.session_state.recently_viewed.insert(0, symbol)
        st.session_state.recently_viewed = st.session_state.recently_viewed[:9]

def show_combined_vwv_technical_analysis(analysis_results, vwv_results, show_debug=False):
    """Display Combined Williams VIX Fix + Technical Composite Score Analysis"""
    if not st.session_state.show_vwv_analysis:
        return
        
    with st.expander(f"🔴 {analysis_results['symbol']} - VWV Signals & Technical Composite Analysis", expanded=True):
        
        if vwv_results and 'error' not in vwv_results:
            # Combined Header with both scores
            col1, col2, col3, col4 = st.columns(4)
            
            # VWV Signal
            signal_strength = vwv_results.get('signal_strength', 'WEAK')
            signal_color = vwv_results.get('signal_color', '⚪')
            vwv_score = vwv_results.get('vwv_score', 0)
            
            # Technical Composite Score
            enhanced_indicators = analysis_results.get('enhanced_indicators', {})
            comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
            composite_score = comprehensive_technicals.get('composite_score', 50)
            
            with col1:
                st.metric("VWV Signal", f"{signal_color} {signal_strength}")
            with col2:
                st.metric("VWV Score", f"{vwv_score:.1f}/100")
            with col3:
                st.metric("Tech Composite", f"{composite_score:.1f}/100")
            with col4:
                current_price = analysis_results.get('current_price', 0)
                st.metric("Current Price", f"${current_price:.2f}")

            # Create technical score bar
            create_technical_score_bar(composite_score)
            
            # Signal analysis
            st.subheader("📊 Signal Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔴 VWV Analysis")
                vwv_interpretation = get_vwv_signal_interpretation(vwv_score, signal_strength)
                st.info(vwv_interpretation)
                
                if 'components' in vwv_results:
                    with st.expander("🔍 VWV Components"):
                        components = vwv_results['components']
                        for component, data in components.items():
                            if isinstance(data, dict):
                                st.write(f"**{component.replace('_', ' ').title()}:**")
                                for key, value in data.items():
                                    if isinstance(value, (int, float)):
                                        st.write(f"  - {key.replace('_', ' ').title()}: {value:.2f}")
            
            with col2:
                st.subheader("📊 Technical Composite")
                
                if composite_score >= 75:
                    st.success("🟢 **Strong Bullish** - Multiple positive technical signals")
                elif composite_score >= 60:
                    st.info("🔵 **Bullish** - Generally positive technical outlook")
                elif composite_score >= 40:
                    st.warning("🟡 **Neutral** - Mixed technical signals")
                elif composite_score >= 25:
                    st.error("🟠 **Bearish** - Generally negative technical outlook")
                else:
                    st.error("🔴 **Strong Bearish** - Multiple negative technical signals")
                    
                if comprehensive_technicals:
                    st.write("**Key Metrics:**")
                    rsi = comprehensive_technicals.get('rsi_14', 0)
                    st.write(f"RSI (14): {rsi:.1f}")
                    volatility = comprehensive_technicals.get('volatility_20d', 0)
                    st.write(f"Volatility (20D): {volatility:.1f}%")
            
            # Risk Management Section
            if st.session_state.show_risk_management and 'risk_management' in vwv_results:
                risk_mgmt = vwv_results['risk_management']
                if risk_mgmt:
                    st.subheader("🎯 Risk Management Levels")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Stop Loss", f"${risk_mgmt.get('stop_loss_price', 0):.2f}", f"-{risk_mgmt.get('stop_loss_pct', 2.2):.1f}%")
                    with col2:
                        st.metric("Take Profit", f"${risk_mgmt.get('take_profit_price', 0):.2f}", f"+{risk_mgmt.get('take_profit_pct', 5.5):.1f}%")
                    with col3:
                        st.metric("Risk/Reward", f"{risk_mgmt.get('risk_reward_ratio', 2.5):.1f}:1")
                    with col4:
                        current_price = vwv_results.get('current_price', 0)
                        potential_profit = risk_mgmt.get('take_profit_price', 0) - current_price
                        st.metric("Potential Profit", f"${potential_profit:.2f}")
        else:
            st.error("❌ VWV/Technical analysis failed")

def show_simple_volume_analysis(analysis_results, show_debug=False):
    """Display simple volume analysis"""
    if not st.session_state.show_volume_analysis or not VOLUME_ANALYSIS_AVAILABLE:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Volume Analysis", expanded=True):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volume_analysis = enhanced_indicators.get('volume_analysis', {})
        
        if volume_analysis and 'error' not in volume_analysis:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                current_volume = volume_analysis.get('current_volume', 0)
                st.metric("Current Volume", format_large_number(current_volume))
            with col2:
                avg_volume = volume_analysis.get('avg_volume_20d', 0)
                st.metric("20D Avg Volume", format_large_number(avg_volume))
            with col3:
                volume_ratio = volume_analysis.get('volume_ratio', 1.0)
                st.metric("Volume Ratio", f"{volume_ratio:.2f}x")
            with col4:
                volume_trend = volume_analysis.get('volume_trend', 0)
                st.metric("Volume Trend", f"{volume_trend:+.1f}%")
            
            volume_regime = volume_analysis.get('volume_regime', 'Normal')
            if volume_ratio > 1.5:
                st.success(f"**High Volume Environment** - {volume_regime}")
            elif volume_ratio > 1.2:
                st.info(f"**Above Average Volume** - {volume_regime}")
            elif volume_ratio < 0.8:
                st.warning(f"**Low Volume Environment** - {volume_regime}")
            else:
                st.info(f"**Normal Volume** - {volume_regime}")
        else:
            st.warning("⚠️ Volume analysis not available")

def show_simple_volatility_analysis(analysis_results, show_debug=False):
    """Display simple volatility analysis"""
    if not st.session_state.show_volatility_analysis or not VOLATILITY_ANALYSIS_AVAILABLE:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Volatility Analysis", expanded=True):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volatility_analysis = enhanced_indicators.get('volatility_analysis', {})
        
        if volatility_analysis and 'error' not in volatility_analysis:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                vol_10d = volatility_analysis.get('volatility_10d', 0)
                st.metric("10D Volatility", f"{vol_10d:.1f}%")
            with col2:
                vol_20d = volatility_analysis.get('volatility_20d', 0)
                st.metric("20D Volatility", f"{vol_20d:.1f}%")
            with col3:
                vol_percentile = volatility_analysis.get('volatility_percentile', 50)
                st.metric("Vol Percentile", f"{vol_percentile:.0f}%")
            with col4:
                vol_regime = volatility_analysis.get('volatility_regime', 'Normal')
                st.metric("Vol Regime", vol_regime)
            
            # Options strategy guidance
            if vol_percentile > 75:
                st.success("**High Volatility** - Consider premium selling strategies")
            elif vol_percentile > 50:
                st.info("**Above Normal Volatility** - Balanced strategies")
            elif vol_percentile < 25:
                st.warning("**Low Volatility** - Consider premium buying strategies")
            else:
                st.info("**Normal Volatility** - Standard strategies apply")
        else:
            st.warning("⚠️ Volatility analysis not available")

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
                        if isinstance(criterion, dict):
                            status = "✅" if criterion.get('passed', False) else "❌"
                            st.write(f"{status} {criterion.get('name', 'Unknown')}: {criterion.get('value', 'N/A')}")
                        else:
                            st.write(f"• {criterion}")
            
            with col2:
                st.subheader("📊 Piotroski F-Score")
                piotroski_total = piotroski_score.get('score', 0)
                piotroski_max = piotroski_score.get('total_possible', 9)
                st.metric("Piotroski Score", f"{piotroski_total}/{piotroski_max}")
                
                if 'criteria' in piotroski_score:
                    for criterion in piotroski_score['criteria']:
                        if isinstance(criterion, dict):
                            status = "✅" if criterion.get('passed', False) else "❌"
                            st.write(f"{status} {criterion.get('name', 'Unknown')}: {criterion.get('value', 'N/A')}")
                        else:
                            st.write(f"• {criterion}")
        else:
            if 'error' in graham_score:
                st.info(f"Graham Analysis: {graham_score['error']}")
            if 'error' in piotroski_score:
                st.info(f"Piotroski Analysis: {piotroski_score['error']}")

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis section"""
    if not st.session_state.show_market_correlation:
        return
        
    enhanced_indicators = analysis_results.get('enhanced_indicators', {})
    market_correlations = enhanced_indicators.get('market_correlations', {})
    
    with st.expander(f"🌐 {analysis_results['symbol']} - Market Correlation Analysis", expanded=True):
        
        if market_correlations and 'error' not in market_correlations:
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

def show_options_analysis(analysis_results, advanced_options_results, show_debug=False):
    """Display options analysis section"""
    if not st.session_state.show_options_analysis:
        return
        
    with st.expander("🎯 Advanced Options Analysis", expanded=True):
        
        if advanced_options_results and 'error' not in advanced_options_results:
            display_data = format_advanced_options_for_display(advanced_options_results)
            
            if 'error' not in display_data:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("5-Day Base Price", f"${display_data['base_price']:.2f}")
                with col2:
                    st.metric("Current Price", f"${display_data['current_price']:.2f}")
                with col3:
                    base_dev = display_data['base_deviation']
                    st.metric("Base Deviation", f"{base_dev:+.2f}%")
                with col4:
                    vol_data = display_data.get('volatility_summary', {})
                    avg_vol = vol_data.get('average_volatility', 20)
                    st.metric("Average Volatility", f"{avg_vol:.1f}%")
                
                st.subheader("💰 Multi-Risk Level Options Strategy")
                display_table = display_data.get('display_table', [])
                if display_table:
                    df_advanced_options = pd.DataFrame(display_table)
                    st.dataframe(df_advanced_options, use_container_width=True, hide_index=True)
                else:
                    st.warning("No options strategy data available")
            else:
                st.error(f"Options formatting error: {display_data.get('error', 'Unknown error')}")
        else:
            # Fallback to basic options
            enhanced_indicators = analysis_results.get('enhanced_indicators', {})
            options_levels = enhanced_indicators.get('options_levels', [])
            
            if options_levels:
                st.subheader("💰 Basic Options Levels")
                if isinstance(options_levels, list):
                    df_options = pd.DataFrame(options_levels)
                    st.dataframe(df_options, use_container_width=True, hide_index=True)
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
                st.warning("⚠️ Charts temporarily unavailable")

def perform_enhanced_analysis(symbol, period, show_debug=False):
    """Perform enhanced analysis using modular components"""
    try:
        # Step 1: Fetch data using modular data fetcher
        market_data = get_market_data_enhanced(symbol, period, show_debug)
        
        if market_data is None:
            st.error(f"❌ Could not fetch data for {symbol}")
            return None, None, None, None
        
        # Step 2: Store data using data manager
        data_manager = get_data_manager()
        data_manager.store_market_data(symbol, market_data, show_debug)
        
        # Step 3: Get analysis copy
        analysis_input = data_manager.get_market_data_for_analysis(symbol)
        
        if analysis_input is None:
            st.error("❌ Could not prepare analysis data")
            return None, None, None, None
        
        # Step 4: Calculate enhanced indicators
        daily_vwap = calculate_daily_vwap(analysis_input)
        fibonacci_emas = calculate_fibonacci_emas(analysis_input)
        point_of_control = calculate_point_of_control_enhanced(analysis_input)
        weekly_deviations = calculate_weekly_deviations(analysis_input)
        comprehensive_technicals = calculate_comprehensive_technicals(analysis_input)
        
        # Step 5: Simple Volume Analysis
        volume_analysis = {}
        if VOLUME_ANALYSIS_AVAILABLE:
            try:
                volume_analysis = calculate_simple_volume_analysis(analysis_input)
            except Exception as e:
                if show_debug:
                    st.write(f"❌ Volume analysis failed: {e}")
                volume_analysis = {'error': 'Volume analysis failed'}
        
        # Step 6: Simple Volatility Analysis
        volatility_analysis = {}
        if VOLATILITY_ANALYSIS_AVAILABLE:
            try:
                volatility_analysis = calculate_simple_volatility_analysis(analysis_input)
            except Exception as e:
                if show_debug:
                    st.write(f"❌ Volatility analysis failed: {e}")
                volatility_analysis = {'error': 'Volatility analysis failed'}
        
        # Step 7: Market correlations
        market_correlations = calculate_market_correlations_enhanced(analysis_input, symbol, show_debug=show_debug)
        
        # Step 8: Fundamental analysis
        is_etf_symbol = is_etf(symbol)
        
        if is_etf_symbol:
            graham_score = {'score': 0, 'total_possible': 10, 'criteria': [], 'error': 'ETF - Fundamental analysis not applicable'}
            piotroski_score = {'score': 0, 'total_possible': 9, 'criteria': [], 'error': 'ETF - Fundamental analysis not applicable'}
        else:
            graham_score = calculate_graham_score(symbol, show_debug)
            piotroski_score = calculate_piotroski_score(symbol, show_debug)
        
        # Step 9: Options levels
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
        
        current_price = round(float(analysis_input['Close'].iloc[-1]), 2)
        options_levels = calculate_options_levels_enhanced(current_price, volatility, underlying_beta=underlying_beta)
        
        # Step 10: Advanced options
        advanced_options_results = calculate_complete_advanced_options(analysis_input, symbol)
        
        # Step 11: VWV System Analysis
        vwv_results = calculate_vwv_system_complete(analysis_input, symbol, DEFAULT_VWV_CONFIG)
        
        # Step 12: Confidence intervals
        confidence_analysis = calculate_confidence_intervals(analysis_input)
        
        # Step 13: Build results
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
                'volume_analysis': volume_analysis,
                'volatility_analysis': volatility_analysis,
                'market_correlations': market_correlations,
                'options_levels': options_levels,
                'graham_score': graham_score,
                'piotroski_score': piotroski_score
            },
            'confidence_analysis': confidence_analysis,
            'system_status': 'OPERATIONAL'
        }
        
        # Store results
        data_manager.store_analysis_results(symbol, analysis_results)
        
        # Get chart data
        chart_data = data_manager.get_market_data_for_chart(symbol)
        
        return analysis_results, chart_data, vwv_results, advanced_options_results
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        if show_debug:
            st.exception(e)
        return None, None, None, None

def main():
    """Main application function"""
    # Create header
    create_header()
    
    # Create sidebar and get controls
    controls = create_sidebar_controls()
    
    # Main logic flow
    if controls['analyze_button'] and controls['symbol']:
        add_to_recently_viewed(controls['symbol'])
        
        st.write("## 📊 VWV Trading Analysis")
        
        with st.spinner(f"Analyzing {controls['symbol']}..."):
            
            analysis_results, chart_data, vwv_results, advanced_options_results = perform_enhanced_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results and chart_data is not None:
                # Show charts FIRST
                show_interactive_charts(chart_data, analysis_results, controls['show_debug'])
                
                # Show analysis sections
                show_combined_vwv_technical_analysis(analysis_results, vwv_results, controls['show_debug'])
                
                # Show volume analysis if available
                if VOLUME_ANALYSIS_AVAILABLE:
                    show_simple_volume_analysis(analysis_results, controls['show_debug'])
                
                # Show volatility analysis if available
                if VOLATILITY_ANALYSIS_AVAILABLE:
                    show_simple_volatility_analysis(analysis_results, controls['show_debug'])
                
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                show_options_analysis(analysis_results, advanced_options_results, controls['show_debug'])
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
                # Debug information
                if controls['show_debug']:
                    with st.expander("🐛 Debug Information", expanded=False):
                        st.write("**System Status:**")
                        st.write(f"- Volume Analysis Available: {VOLUME_ANALYSIS_AVAILABLE}")
                        st.write(f"- Volatility Analysis Available: {VOLATILITY_ANALYSIS_AVAILABLE}")
                        st.json(analysis_results, expanded=False)
            else:
                st.error("❌ Analysis failed. Please try a different symbol or enable debug mode for details.")
    else:
        st.write("## 📊 VWV Professional Trading System")
        
        market_status = get_market_status()
        if market_status:
            st.info(f"🕒 **Market Status:** {market_status}")
        
        st.info("👈 **Enter a stock symbol in the sidebar to begin analysis**")
        
        if len(st.session_state.recently_viewed) > 0:
            st.subheader("🕒 Recently Analyzed Symbols")
            cols = st.columns(min(6, len(st.session_state.recently_viewed)))
            for i, recent_symbol in enumerate(st.session_state.recently_viewed[:6]):
                with cols[i]:
                    if st.button(f"📊 {recent_symbol}", key=f"main_recent_{recent_symbol}", use_container_width=True):
                        st.session_state.selected_symbol = recent_symbol
                        st.session_state.auto_analyze = True
                        st.rerun()

if __name__ == "__main__":
    main()
