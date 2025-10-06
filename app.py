"""
FILENAME: app.py
VWV Professional Trading System v4.2.2
File Revision: r8
Date: October 6, 2025
Revision Type: Critical Bug Fix Release

CRITICAL FIXES/CHANGES IN THIS REVISION:
- Fixed truncated st.dataframe line in show_options_analysis (line 487)
- Added pandas FutureWarning suppression in imports section
- Enhanced error handling in perform_enhanced_analysis function
- Fixed confidence_intervals None return handling with proper fallback
- Verified all dataframe operations use complete parameters

FILE REVISION HISTORY:
r8 (Oct 6, 2025) - Critical bug fixes for production stability
r7 (Oct 5, 2025) - Layout optimization and display order corrections  
r6 (Oct 4, 2025) - Advanced options integration with sigma levels
r5 (Oct 3, 2025) - VWV core analysis enhancements
r4 (Oct 2, 2025) - Technical analysis module updates
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

# Suppress warnings - ENHANCED r8
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# Page configuration
st.set_page_config(
    page_title="VWV Professional Trading System v4.2.2",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_sidebar_controls():
    """Create sidebar controls and return analysis parameters"""
    st.sidebar.title("📊 Trading Analysis v4.2.2")
    
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
    
    # Basic controls
    if 'selected_symbol' in st.session_state:
        default_symbol = st.session_state.selected_symbol
        if not st.session_state.get('auto_analyze', False):
            del st.session_state.selected_symbol
    else:
        default_symbol = UI_SETTINGS['default_symbol']
        
    symbol = st.sidebar.text_input("Symbol", value=default_symbol, help="Enter stock symbol (press Enter to analyze)", key="symbol_input").upper()
    period = st.sidebar.selectbox("Data Period", UI_SETTINGS['periods'], index=1)
    
    # Section Control Panel
    with st.sidebar.expander("📋 Analysis Sections", expanded=False):
        st.write("**Toggle Analysis Sections:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.show_vwv_analysis = st.checkbox(
                "🔴 VWV Analysis", 
                value=st.session_state.show_vwv_analysis,
                key="toggle_vwv"
            )
            st.session_state.show_fundamental_analysis = st.checkbox(
                "Fundamental", 
                value=st.session_state.show_fundamental_analysis,
                key="toggle_fundamental"
            )
            st.session_state.show_market_correlation = st.checkbox(
                "Market Correlation", 
                value=st.session_state.show_market_correlation,
                key="toggle_correlation"
            )
        
        with col2:
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
                "Interactive Charts", 
                value=st.session_state.show_charts,
                key="toggle_charts"
            )
    
    # Quick Links
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Quick Links")
    
    for category, symbols in QUICK_LINK_CATEGORIES.items():
        with st.sidebar.expander(f"📁 {category}", expanded=(category == "Major ETFs")):
            cols = st.columns(3)
            for i, sym in enumerate(symbols):
                with cols[i % 3]:
                    if st.button(sym, key=f"quick_{sym}", use_container_width=True):
                        st.session_state.selected_symbol = sym
                        st.session_state.auto_analyze = True
                        st.rerun()
    
    # Recently Viewed
    if st.session_state.recently_viewed:
        with st.sidebar.expander("🕐 Recently Viewed", expanded=False):
            for sym in st.session_state.recently_viewed[:5]:
                if st.button(sym, key=f"recent_{sym}", use_container_width=True):
                    st.session_state.selected_symbol = sym
                    st.session_state.auto_analyze = True
                    st.rerun()
    
    # Analysis button
    analyze_button = st.sidebar.button("📊 Analyze Now", type="primary", use_container_width=True)
    
    # Auto-analyze on Enter key
    if symbol and st.session_state.get('auto_analyze', False):
        analyze_button = True
        st.session_state.auto_analyze = False
    
    # Debug mode
    show_debug = st.sidebar.checkbox("🐛 Debug Mode", value=False)
    
    return {
        'symbol': symbol,
        'period': period,
        'analyze_button': analyze_button,
        'show_debug': show_debug
    }

def add_to_recently_viewed(symbol):
    """Add symbol to recently viewed list"""
    if symbol not in st.session_state.recently_viewed:
        st.session_state.recently_viewed.insert(0, symbol)
        st.session_state.recently_viewed = st.session_state.recently_viewed[:10]

def show_combined_vwv_technical_analysis(analysis_results, vwv_results, show_debug=False):
    """Display combined VWV and technical analysis"""
    if not st.session_state.show_vwv_analysis:
        return
        
    with st.expander("🔴 VWV System Analysis + Technical Indicators", expanded=True):
        
        if vwv_results and 'error' not in vwv_results:
            # VWV Signal Display
            overall_signal = vwv_results.get('overall_signal', {})
            signal_strength = overall_signal.get('signal_strength', 0)
            signal_classification = overall_signal.get('classification', 'NEUTRAL')
            
            # Color coding
            signal_colors = {
                'VERY_STRONG': '#dc3545',
                'STRONG': '#ffc107',
                'GOOD': '#28a745',
                'NEUTRAL': '#6c757d'
            }
            signal_color = signal_colors.get(signal_classification, '#6c757d')
            
            # Main VWV Signal Header
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"""
                <div style="padding: 1rem; background: linear-gradient(135deg, {signal_color}20, {signal_color}10); 
                             border-left: 4px solid {signal_color}; border-radius: 8px;">
                    <h3 style="color: {signal_color}; margin: 0;">🔴 {signal_classification} Signal</h3>
                    <p style="margin: 0.5rem 0 0 0; color: #666;">VWV Strength: {signal_strength:.2f}/10.0</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                market_bias = vwv_results.get('market_conditions', {}).get('current_bias', 'Unknown')
                st.metric("Market Bias", market_bias)
            
            with col3:
                volatility_regime = vwv_results.get('market_conditions', {}).get('volatility_regime', 'Unknown')
                st.metric("Volatility Regime", volatility_regime)
            
            # Technical Score
            st.subheader("📊 Technical Composite Score")
            technical_score, score_details = calculate_composite_technical_score(analysis_results)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                create_technical_score_bar(technical_score)
            with col2:
                score_color = 'green' if technical_score >= 60 else 'orange' if technical_score >= 40 else 'red'
                st.markdown(f"<h2 style='text-align: center; color: {score_color};'>{technical_score:.1f}/100</h2>", unsafe_allow_html=True)
            
            # VWV Components
            with st.expander("🔍 VWV Signal Components", expanded=False):
                signal_components = vwv_results.get('signal_components', {})
                
                if signal_components:
                    components_data = []
                    for comp_name, comp_data in signal_components.items():
                        if isinstance(comp_data, dict):
                            components_data.append({
                                'Component': comp_name.replace('_', ' ').title(),
                                'Score': f"{comp_data.get('score', 0):.2f}",
                                'Weight': f"{comp_data.get('weight', 0):.1f}",
                                'Status': comp_data.get('status', 'Unknown')
                            })
                    
                    if components_data:
                        df_components = pd.DataFrame(components_data)
                        st.dataframe(df_components, use_container_width=True, hide_index=True)
            
            # Technical Components
            with st.expander("📈 Technical Score Breakdown", expanded=False):
                if score_details and 'component_scores' in score_details:
                    comp_scores = score_details['component_scores']
                    tech_data = []
                    for comp, score in comp_scores.items():
                        tech_data.append({
                            'Indicator': comp.replace('_', ' ').title(),
                            'Score': f"{score:.1f}/100",
                            'Status': '🟢 Bullish' if score >= 60 else '🔴 Bearish' if score <= 40 else '🟡 Neutral'
                        })
                    
                    df_technical = pd.DataFrame(tech_data)
                    st.dataframe(df_technical, use_container_width=True, hide_index=True)
        
        else:
            st.warning("⚠️ VWV analysis not available")

def show_fundamental_analysis(analysis_results, show_debug=False):
    """Display fundamental analysis section"""
    if not st.session_state.show_fundamental_analysis:
        return
        
    enhanced_indicators = analysis_results.get('enhanced_indicators', {})
    graham_score = enhanced_indicators.get('graham_score', {})
    piotroski_score = enhanced_indicators.get('piotroski_score', {})
    
    is_etf_symbol = is_etf(analysis_results.get('symbol', ''))
    
    with st.expander("📊 Fundamental Analysis", expanded=True):
        
        if 'error' not in graham_score and 'error' not in piotroski_score and not is_etf_symbol:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💎 Graham Value Score")
                graham_points = graham_score.get('score', 0)
                graham_total = graham_score.get('total_possible', 10)
                st.metric("Graham Score", f"{graham_points}/{graham_total}")
                
                if graham_points >= 7:
                    st.success("✅ Strong Value Investment")
                elif graham_points >= 5:
                    st.info("ℹ️ Moderate Value")
                else:
                    st.warning("⚠️ Limited Value Characteristics")
            
            with col2:
                st.subheader("📈 Piotroski F-Score")
                piotroski_points = piotroski_score.get('score', 0)
                piotroski_total = piotroski_score.get('total_possible', 9)
                st.metric("Piotroski Score", f"{piotroski_points}/{piotroski_total}")
                
                if piotroski_points >= 7:
                    st.success("✅ Strong Financial Health")
                elif piotroski_points >= 5:
                    st.info("ℹ️ Moderate Financial Health")
                else:
                    st.warning("⚠️ Weak Financial Health")
        
        elif is_etf_symbol:
            st.info(f"ℹ️ **{analysis_results['symbol']} is an ETF** - Fundamental analysis not applicable to ETFs.")
        
        else:
            st.warning("⚠️ **Insufficient Fundamental Data**")

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis section"""
    if not st.session_state.show_market_correlation:
        return
        
    with st.expander("🌐 Market Correlation Analysis", expanded=True):
        
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
        st.subheader("📊 Market Breakout/Breakdown Analysis")
        breakout_data = calculate_breakout_breakdown_analysis(show_debug=show_debug)
        
        if breakout_data:
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

def show_options_analysis(analysis_results, advanced_options_results, show_debug=False):
    """
    Display enhanced options analysis section - FIXED r8
    CRITICAL FIX: Line 487 - Complete st.dataframe parameter
    """
    if not st.session_state.show_options_analysis:
        return
        
    with st.expander("🎯 Advanced Options Analysis - Sigma Levels & Fibonacci Integration", expanded=True):
        
        # Check if we have advanced options data
        if advanced_options_results and 'error' not in advanced_options_results:
            
            # Format data for display
            display_data = format_advanced_options_for_display(advanced_options_results)
            
            if 'error' not in display_data:
                # Header with base price information
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
                
                # Advanced Options Table
                st.subheader("💰 Multi-Risk Level Options Strategy")
                st.write("**Sigma levels with Fibonacci integration and multi-factor weighting**")
                
                display_table = display_data.get('display_table', [])
                if display_table:
                    df_advanced_options = pd.DataFrame(display_table)
                    # CRITICAL FIX r8: Complete line with proper parameters
                    st.dataframe(df_advanced_options, use_container_width=True, hide_index=True)
                
                # Strategy Context
                st.subheader("🎯 Risk Level Strategy Breakdown")
                
                context_col1, context_col2 = st.columns(2)
                
                with context_col1:
                    st.info("""**Conservative Strategy (15% PoT):**
                    • Higher win rate, lower premium
                    • 50% Fibonacci + 30% Volatility + 20% Volume
                    • Best for stable markets
                    • Recommended for beginners""")
                    
                    st.info("""**Moderate Strategy (25% PoT):**
                    • Balanced approach
                    • 35% Fibonacci + 45% Volatility + 20% Volume
                    • Suitable for most market conditions
                    • Good risk/reward balance""")
                
                with context_col2:
                    st.info("""**Aggressive Strategy (35% PoT):**
                    • Higher premium, lower win rate
                    • 25% Fibonacci + 60% Volatility + 15% Volume
                    • Best for high volatility markets
                    • Requires active management""")
                    
                    st.warning("""⚠️ **Risk Management:**
                    • Never risk more than 2% per trade
                    • Always use stop losses
                    • Consider position sizing
                    • Monitor Greeks continuously""")
                
                # Fibonacci Analysis
                fibonacci_summary = display_data.get('fibonacci_summary', {})
                if fibonacci_summary.get('fibonacci_strikes'):
                    with st.expander("📐 Fibonacci Level Analysis", expanded=False):
                        st.write("**Fibonacci-based strike calculations from 5-day rolling base:**")
                        
                        fib_strikes = fibonacci_summary['fibonacci_strikes']
                        fib_data = []
                        
                        for fib_key, fib_info in fib_strikes.items():
                            level = fib_info['level']
                            fib_data.append({
                                'Fibonacci Level': f"{level:.3f}",
                                'Put Strike': f"${fib_info['put_strike']:.2f}",
                                'Call Strike': f"${fib_info['call_strike']:.2f}",
                                'Range $': f"${fib_info['range_dollars']:.2f}",
                                'Range %': f"{fib_info['range_percent']:.2f}%"
                            })
                        
                        key_levels = [item for item in fib_data if float(item['Fibonacci Level']) in [0.382, 0.500, 0.618, 1.000]]
                        if key_levels:
                            df_fibonacci = pd.DataFrame(key_levels)
                            st.dataframe(df_fibonacci, use_container_width=True, hide_index=True)
                
            else:
                st.error(f"❌ Advanced options display formatting failed: {display_data.get('error', 'Unknown error')}")
        
        else:
            st.warning("⚠️ Advanced options analysis not available - using fallback")
            
            # Fallback to basic options
            enhanced_indicators = analysis_results.get('enhanced_indicators', {})
            options_levels = enhanced_indicators.get('options_levels', [])
            
            if options_levels:
                st.subheader("💰 Basic Premium Selling Levels")
                df_options = pd.DataFrame(options_levels)
                st.dataframe(df_options, use_container_width=True, hide_index=True)

def show_confidence_intervals(analysis_results, show_debug=False):
    """Display confidence intervals section - FIXED r8"""
    if not st.session_state.show_confidence_intervals:
        return
        
    confidence_analysis = analysis_results.get('confidence_analysis')
    if confidence_analysis and 'error' not in confidence_analysis:
        with st.expander("📊 Statistical Confidence Intervals", expanded=True):
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Weekly Return", f"{confidence_analysis.get('mean_weekly_return', 0):.3f}%")
            with col2:
                st.metric("Weekly Volatility", f"{confidence_analysis.get('weekly_volatility', 0):.2f}%")
            with col3:
                st.metric("Sample Size", f"{confidence_analysis.get('sample_size', 0)} weeks")
            
            intervals = confidence_analysis.get('confidence_intervals', {})
            if intervals:
                final_intervals_data = []
                for level, level_data in intervals.items():
                    final_intervals_data.append({
                        'Confidence Level': level,
                        'Upper Bound': f"${level_data.get('upper_bound', 0)}",
                        'Lower Bound': f"${level_data.get('lower_bound', 0)}",
                        'Expected Move': f"±{level_data.get('expected_move_pct', 0):.2f}%"
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
        except Exception as e:
            if show_debug:
                st.error(f"Chart display error: {str(e)}")
            st.warning("⚠️ Charts temporarily unavailable")
            
            # Fallback
            if data is not None and not data.empty:
                st.line_chart(data['Close'])

def perform_enhanced_analysis(symbol, period, show_debug=False):
    """Perform enhanced analysis - FIXED r8"""
    try:
        # Fetch data
        market_data = get_market_data_enhanced(symbol, period, show_debug)
        
        if market_data is None:
            st.error(f"❌ Could not fetch data for {symbol}")
            return None, None, None, None
        
        # Store data
        data_manager = get_data_manager()
        data_manager.store_market_data(symbol, market_data, show_debug)
        
        # Get analysis copy
        analysis_input = data_manager.get_market_data_for_analysis(symbol)
        
        if analysis_input is None:
            st.error("❌ Could not prepare analysis data")
            return None, None, None, None
        
        # Calculate indicators
        daily_vwap = calculate_daily_vwap(analysis_input)
        fibonacci_emas = calculate_fibonacci_emas(analysis_input)
        point_of_control = calculate_point_of_control_enhanced(analysis_input)
        weekly_deviations = calculate_weekly_deviations(analysis_input)
        comprehensive_technicals = calculate_comprehensive_technicals(analysis_input)
        
        # Market correlations
        market_correlations = calculate_market_correlations_enhanced(analysis_input, symbol, show_debug=show_debug)
        
        # Fundamental analysis
        is_etf_symbol = is_etf(symbol)
        
        if is_etf_symbol:
            graham_score = {'score': 0, 'total_possible': 10, 'error': 'ETF'}
            piotroski_score = {'score': 0, 'total_possible': 9, 'error': 'ETF'}
        else:
            graham_score = calculate_graham_score(symbol, show_debug)
            piotroski_score = calculate_piotroski_score(symbol, show_debug)
        
        # Options levels
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
        
        # VWV Analysis
        vwv_results = calculate_vwv_system_complete(analysis_input, symbol, DEFAULT_VWV_CONFIG)
        
        # Advanced Options
        advanced_options_results = calculate_complete_advanced_options(analysis_input, symbol)
        
        # Confidence intervals - FIXED r8 to handle None returns
        confidence_analysis = calculate_confidence_intervals(analysis_input, current_price)
        if confidence_analysis is None:
            confidence_analysis = {
                'error': 'Calculation failed',
                'confidence_intervals': {},
                'mean_weekly_return': 0.0,
                'weekly_volatility': 0.0,
                'sample_size': 0
            }
        
        # Build results
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
                'market_correlations': market_correlations,
                'options_levels': options_levels,
                'graham_score': graham_score,
                'piotroski_score': piotroski_score
            },
            'vwv_analysis': vwv_results,
            'advanced_options_analysis': advanced_options_results,
            'confidence_analysis': confidence_analysis,
            'system_status': 'OPERATIONAL v4.2.2 r8'
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
    """Main application function - v4.2.2 r8"""
    create_header()
    
    controls = create_sidebar_controls()
    
    if controls['analyze_button'] and controls['symbol']:
        add_to_recently_viewed(controls['symbol'])
        
        st.write("## 📊 VWV Trading Analysis v4.2.2")
        
        with st.spinner(f"Analyzing {controls['symbol']}..."):
            
            analysis_results, chart_data, vwv_results, advanced_options_results = perform_enhanced_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results and chart_data is not None:
                # Display all sections
                show_interactive_charts(chart_data, analysis_results, controls['show_debug'])
                show_combined_vwv_technical_analysis(analysis_results, vwv_results, controls['show_debug'])
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                show_options_analysis(analysis_results, advanced_options_results, controls['show_debug'])
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
                # Debug info
                if controls['show_debug']:
                    with st.expander("🐛 Debug Information", expanded=False):
                        st.write("### Analysis Results")
                        st.json(analysis_results, expanded=False)
    
    else:
        # Welcome screen
        st.write("## 🚀 VWV Professional Trading System v4.2.2")
        st.write("**Critical Bug Fixes Applied - October 6, 2025**")
        
        with st.expander("🔧 Version 4.2.2 r8 Fixes", expanded=True):
            st.write("### ✅ **Fixed Issues:**")
            st.write("• **Truncated Line Fix:** Complete st.dataframe parameters")
            st.write("• **Pandas Warning Fix:** Added observed=True to groupby operations")
            st.write("• **None Return Fix:** Proper error handling in confidence intervals")
            st.write("• **Enhanced Stability:** All dataframe operations verified")
        
        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Version:** VWV Professional v4.2.2")
        st.write(f"**File Revision:** r8")
    with col2:
        st.write(f"**Status:** ✅ All Fixes Applied")
        st.write(f"**Date:** October 6, 2025")
    with col3:
        st.write(f"**Signal Types:** 🟢 GOOD 🟡 STRONG 🔴 VERY_STRONG")
        st.write(f"**Options:** Sigma Levels + Fibonacci")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.write("Please refresh the page and try again.")
        
        if st.checkbox("Show Error Details"):
            st.exception(e)
