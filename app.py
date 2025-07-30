"""
VWV Professional Trading System - Fixed v3.1
Main application with charts and auto-analysis on quick links
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
    if 'show_technical_analysis' not in st.session_state:
        st.session_state.show_technical_analysis = True
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
        # Clear the selected symbol but keep auto_analyze flag
        if not st.session_state.get('auto_analyze', False):
            del st.session_state.selected_symbol
    else:
        default_symbol = UI_SETTINGS['default_symbol']
        
    symbol = st.sidebar.text_input("Symbol", value=default_symbol, help="Enter stock symbol (press Enter to analyze)", key="symbol_input").upper()
    period = st.sidebar.selectbox("Data Period", UI_SETTINGS['periods'], index=1)  # Default to 3mo
    
    # Section Control Panel
    with st.sidebar.expander("📋 Analysis Sections", expanded=False):
        st.write("**Toggle Analysis Sections:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.show_vwv_analysis = st.checkbox(
                "🔴 VWV Signals", 
                value=st.session_state.show_vwv_analysis,
                key="toggle_vwv"
            )
            st.session_state.show_technical_analysis = st.checkbox(
                "Technical Analysis", 
                value=st.session_state.show_technical_analysis,
                key="toggle_technical"
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
                "Interactive Charts", 
                value=st.session_state.show_charts,
                key="toggle_charts"
            )
    
    # Main analyze button
    analyze_button = st.sidebar.button("📊 Analyze Symbol", type="primary", use_container_width=True)
    
    # Check for Enter key press (when symbol input changes)
    if 'last_symbol' not in st.session_state:
        st.session_state.last_symbol = ""
    
    # Detect if user pressed Enter or changed symbol
    if symbol != st.session_state.last_symbol and symbol != "" and len(symbol) > 0:
        st.session_state.last_symbol = symbol
        st.session_state.auto_analyze = True
        analyze_button = True  # Trigger analysis on Enter/symbol change
    
    # Check for auto-analyze trigger from quick links
    auto_analyze_triggered = st.session_state.get('auto_analyze', False)
    if auto_analyze_triggered:
        st.session_state.auto_analyze = False  # Reset the flag
        analyze_button = True  # Trigger analysis
    
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
                                st.session_state.auto_analyze = True  # Set flag for auto-analysis
                                st.rerun()

    # Quick Links section
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
                                    st.session_state.auto_analyze = True  # Set flag for auto-analysis
                                    st.rerun()

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

def show_vwv_analysis(analysis_results, vwv_results, show_debug=False):
    """Display Williams VIX Fix (VWV) analysis section"""
    if not st.session_state.show_vwv_analysis:
        return
        
    with st.expander(f"🔴 {analysis_results['symbol']} - Williams VIX Fix (VWV) Professional Signals", expanded=True):
        
        if vwv_results and 'error' not in vwv_results:
            # Main VWV Signal Display
            signal_strength = vwv_results.get('signal_strength', 'WEAK')
            signal_color = vwv_results.get('signal_color', '⚪')
            vwv_score = vwv_results.get('vwv_score', 0)
            
            # Color mapping for signal strength
            strength_colors = {
                'VERY_STRONG': '#DC143C',  # Crimson red
                'STRONG': '#FFD700',       # Gold
                'GOOD': '#32CD32',         # Lime green
                'WEAK': '#808080'          # Gray
            }
            
            signal_color_hex = strength_colors.get(signal_strength, '#808080')
            
            # Main signal header
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"""
                <div style="padding: 1rem; background: linear-gradient(135deg, {signal_color_hex}20, {signal_color_hex}10); 
                            border-left: 4px solid {signal_color_hex}; border-radius: 8px; margin-bottom: 1rem;">
                    <h3 style="color: {signal_color_hex}; margin: 0;">
                        {signal_color} VWV Signal: {signal_strength}
                    </h3>
                    <p style="margin: 0.5rem 0 0 0; color: #666;">
                        {vwv_results.get('signal_interpretation', 'Professional market timing signal')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("VWV Score", f"{vwv_score:.1f}/100", f"{signal_strength}")
            
            with col3:
                st.metric("Current Price", f"${vwv_results.get('current_price', 0):.2f}")
            
            # Risk Management Section
            risk_mgmt = vwv_results.get('risk_management', {})
            if risk_mgmt:
                st.subheader("🎯 Risk Management Levels")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Stop Loss", f"${risk_mgmt.get('stop_loss_price', 0):.2f}", 
                             f"-{risk_mgmt.get('stop_loss_pct', 2.2):.1f}%")
                with col2:
                    st.metric("Take Profit", f"${risk_mgmt.get('take_profit_price', 0):.2f}", 
                             f"+{risk_mgmt.get('take_profit_pct', 5.5):.1f}%")
                with col3:
                    st.metric("Risk/Reward", f"{risk_mgmt.get('risk_reward_ratio', 2.5):.1f}:1")
                with col4:
                    # Calculate potential profit/loss
                    current_price = vwv_results.get('current_price', 0)
                    stop_loss = risk_mgmt.get('stop_loss_price', 0)
                    take_profit = risk_mgmt.get('take_profit_price', 0)
                    
                    if current_price > 0:
                        max_loss = ((current_price - stop_loss) / current_price) * 100
                        max_gain = ((take_profit - current_price) / current_price) * 100
                        st.metric("Max Loss", f"-{max_loss:.1f}%", f"Max Gain: +{max_gain:.1f}%")
                    else:
                        st.metric("Max Loss/Gain", "N/A")
            
            # 6-Component Breakdown
            components = vwv_results.get('components', {})
            if components:
                st.subheader("📊 6-Component VWV Analysis")
                
                # Create component summary table
                component_data = []
                
                # Williams VIX Fix
                wvf_comp = components.get('williams_vix_fix', {})
                component_data.append([
                    "1. Williams VIX Fix", 
                    f"{wvf_comp.get('current_value', 0):.2f}", 
                    f"{wvf_comp.get('score', 50):.1f}/100",
                    f"Weight: {wvf_comp.get('weight', 0.8):.1f}",
                    "Fear Gauge"
                ])
                
                # MA Confluence
                ma_comp = components.get('ma_confluence', {})
                component_data.append([
                    "2. MA Confluence", 
                    f"{ma_comp.get('confluence_count', 0)}/3", 
                    f"{ma_comp.get('score', 50):.1f}/100",
                    f"Weight: {ma_comp.get('weight', 1.2):.1f}",
                    "Trend Alignment"
                ])
                
                # Volume Confluence
                vol_comp = components.get('volume_confluence', {})
                component_data.append([
                    "3. Volume Confluence", 
                    vol_comp.get('strength', 'Neutral'), 
                    f"{vol_comp.get('score', 50):.1f}/100",
                    f"Weight: {vol_comp.get('weight', 0.6):.1f}",
                    "Volume Confirmation"
                ])
                
                # VWAP Analysis
                vwap_comp = components.get('vwap_analysis', {})
                component_data.append([
                    "4. Enhanced VWAP", 
                    vwap_comp.get('position', 'Neutral'), 
                    f"{vwap_comp.get('score', 50):.1f}/100",
                    f"Weight: {vwap_comp.get('weight', 0.4):.1f}",
                    "Price/Volume"
                ])
                
                # RSI Momentum
                rsi_comp = components.get('rsi_momentum', {})
                component_data.append([
                    "5. RSI Momentum", 
                    rsi_comp.get('level', 'Neutral'), 
                    f"{rsi_comp.get('score', 50):.1f}/100",
                    f"Weight: {rsi_comp.get('weight', 0.5):.1f}",
                    "Oversold Detection"
                ])
                
                # Volatility Filter
                vol_filter = components.get('volatility_filter', {})
                component_data.append([
                    "6. Volatility Filter", 
                    vol_filter.get('regime', 'Normal'), 
                    f"{vol_filter.get('score', 50):.1f}/100",
                    f"Weight: {vol_filter.get('weight', 0.3):.1f}",
                    "Market Regime"
                ])
                
                # Display component table
                df_components = pd.DataFrame(component_data, 
                                           columns=['Component', 'Current State', 'Score', 'Weight', 'Purpose'])
                st.dataframe(df_components, use_container_width=True, hide_index=True)
                
                # Detailed component expandables
                with st.expander("📋 Detailed Component Analysis", expanded=False):
                    
                    # MA Confluence Details
                    ma_details = ma_comp.get('details', {})
                    if ma_details:
                        st.write("**Moving Average Confluence:**")
                        for ma_name, ma_data in ma_details.items():
                            status = "✅ Above" if ma_data.get('above') else "❌ Below"
                            st.write(f"• {ma_name}: ${ma_data.get('value', 0):.2f} - {status} ({ma_data.get('distance_pct', 0):+.1f}%)")
                    
                    # Volume Details
                    vol_details = vol_comp.get('details', {})
                    if vol_details:
                        st.write("**Volume Analysis:**")
                        for vol_name, vol_data in vol_details.items():
                            ratio = vol_data.get('ratio', 1)
                            status = "✅ Above Average" if vol_data.get('above_average') else "❌ Below Average"
                            st.write(f"• {vol_name}: {ratio:.2f}x - {status}")
                    
                    # VWAP Details
                    if 'vwap_value' in vwap_comp:
                        st.write("**VWAP Analysis:**")
                        st.write(f"• VWAP Level: ${vwap_comp['vwap_value']:.2f}")
                        st.write(f"• Position: {vwap_comp.get('position', 'Neutral')}")
                    
                    # RSI Details
                    if 'rsi_value' in rsi_comp:
                        st.write("**RSI Momentum:**")
                        st.write(f"• RSI(14): {rsi_comp['rsi_value']:.1f}")
                        st.write(f"• Level: {rsi_comp.get('level', 'Neutral')}")
                    
                    # Volatility Details
                    if 'volatility' in vol_filter:
                        st.write("**Volatility Filter:**")
                        st.write(f"• 20-Day Volatility: {vol_filter['volatility']:.1f}%")
                        st.write(f"• Regime: {vol_filter.get('regime', 'Normal')}")
        
        else:
            st.error("❌ VWV analysis failed or data insufficient")
            if show_debug and vwv_results and 'error' in vwv_results:
                st.error(f"Error details: {vwv_results['error']}")

def show_individual_technical_analysis(analysis_results, show_debug=False):
    """Display individual technical analysis section"""
    if not st.session_state.show_technical_analysis:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Individual Technical Analysis", expanded=True):
        
        # COMPOSITE TECHNICAL SCORE - Use modular component
        composite_score, score_details = calculate_composite_technical_score(analysis_results)
        
        # Display score using columns instead of HTML for better reliability
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Technical Composite Score")
            st.write("Aggregated signal from all technical indicators")
        with col2:
            # Determine interpretation and color
            if composite_score >= 80:
                interpretation = "Very Bullish"
                color = "green"
            elif composite_score >= 65:
                interpretation = "Bullish"
                color = "green"
            elif composite_score >= 55:
                interpretation = "Slightly Bullish"
                color = "blue"
            elif composite_score >= 45:
                interpretation = "Neutral"
                color = "orange"
            elif composite_score >= 35:
                interpretation = "Slightly Bearish"
                color = "orange"
            elif composite_score >= 20:
                interpretation = "Bearish"
                color = "red"
            else:
                interpretation = "Very Bearish"
                color = "red"
            
            st.metric("Score", f"{composite_score:.1f}/100", f"{interpretation}")
        
        # Score progress bar
        progress_value = composite_score / 100
        st.progress(progress_value)
        
        # Component breakdown
        if 'component_scores' in score_details:
            st.write("**Component Breakdown:**")
            components = score_details['component_scores']
            
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            with comp_col1:
                st.write(f"• VWAP Position: {components.get('vwap_position', 50):.1f}")
                st.write(f"• POC Position: {components.get('poc_position', 50):.1f}")
            with comp_col2:
                st.write(f"• EMA Confluence: {components.get('ema_confluence', 50):.1f}")
                st.write(f"• RSI Momentum: {components.get('rsi_momentum', 50):.1f}")
            with comp_col3:
                st.write(f"• Volume Strength: {components.get('volume_strength', 50):.1f}")
                st.write(f"• Trend Direction: {components.get('trend_direction', 50):.1f}")
        
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

def show_fundamental_analysis(analysis_results, show_debug=False):
    """Display fundamental analysis section"""
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
            
            # Detailed criteria breakdown
            if 'error' not in graham_data and 'criteria' in graham_data:
                with st.expander("📋 Graham Score Detailed Criteria", expanded=False):
                    st.write("**Benjamin Graham Value Investment Criteria:**")
                    criteria_list = graham_data.get('criteria', [])
                    if criteria_list:
                        for criterion in criteria_list:
                            st.write(f"• {criterion}")
                    else:
                        st.write("Criteria details not available")
            
            if 'error' not in piotroski_data and 'criteria' in piotroski_data:
                with st.expander("📋 Piotroski F-Score Detailed Criteria", expanded=False):
                    st.write("**Piotroski F-Score Quality Metrics:**")
                    criteria_list = piotroski_data.get('criteria', [])
                    if criteria_list:
                        for criterion in criteria_list:
                            st.write(f"• {criterion}")
                    else:
                        st.write("Criteria details not available")
        
        elif is_etf_symbol:
            st.info(f"ℹ️ **{analysis_results['symbol']} is an ETF** - Fundamental analysis is not applicable to Exchange-Traded Funds.")

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis section"""
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
    """Display options analysis section"""
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
                
                # Fallback simple chart
                st.subheader("Basic Price Chart (Fallback)")
                if data is not None and not data.empty:
                    st.line_chart(data['Close'])
                else:
                    st.error("No data available for charting")

def perform_enhanced_analysis(symbol, period, show_debug=False):
    """Perform enhanced analysis using modular components"""
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
        
        # Step 8: Calculate VWV System Analysis
        vwv_results = calculate_vwv_system_complete(analysis_input, symbol, DEFAULT_VWV_CONFIG)
        
        # Step 9: Calculate confidence intervals
        confidence_analysis = calculate_confidence_intervals(analysis_input)
        
        # Step 10: Build analysis results
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
            'vwv_analysis': vwv_results,  # Add VWV results
            'confidence_analysis': confidence_analysis,
            'system_status': 'OPERATIONAL'
        }
        
        # Store results
        data_manager.store_analysis_results(symbol, analysis_results)
        
        # Get chart data
        chart_data = data_manager.get_market_data_for_chart(symbol)
        
        return analysis_results, chart_data, vwv_results
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        return None, None, None

def main():
    """Main application function"""
    # Create header using modular component
    create_header()
    
    # Create sidebar and get controls
    controls = create_sidebar_controls()
    
    # Main logic flow
    if controls['analyze_button'] and controls['symbol']:
        # Add symbol to recently viewed
        add_to_recently_viewed(controls['symbol'])
        
        st.write("## 📊 VWV Trading Analysis")
        
        with st.spinner(f"Analyzing {controls['symbol']}..."):
            
            # Perform analysis using modular components
            analysis_results, chart_data, vwv_results = perform_enhanced_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results and chart_data is not None:
                # Show all analysis sections using modular functions
                show_vwv_analysis(analysis_results, vwv_results, controls['show_debug'])
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                show_options_analysis(analysis_results, controls['show_debug'])
                show_confidence_intervals(analysis_results, controls['show_debug'])
                show_interactive_charts(chart_data, analysis_results, controls['show_debug'])
                
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
                            test_fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3], name="Test"))
                            st.plotly_chart(test_fig, use_container_width=True)
                            st.success("✅ Basic Plotly chart creation successful")
                        except Exception as e:
                            st.error(f"❌ Basic Plotly test failed: {str(e)}")
                            st.exception(e)
    
    else:
        # Welcome message
        st.write("## 🚀 VWV Professional Trading System - v4.0 Enhanced")
        st.write("**All modules active:** VWV Signals, Technical, Fundamental, Market, Options, Charts")
        
        with st.expander("🔧 Latest Enhancements - v4.0", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### ✅ **New Features Added**")
                st.write("🔴 **Williams VIX Fix (VWV) Core System** - Professional market timing")
                st.write("📊 **6-Component Confluence Analysis** - Advanced signal generation") 
                st.write("🎯 **Risk Management Integration** - Stop loss & take profit levels")
                st.write("⚡ **Enhanced Enter Key Support** - Type symbol and press Enter")
                st.write("📅 **Default Period Fixed** - Now correctly defaults to 3mo")
                
            with col2:
                st.write("### 🔴 **VWV Signal System**")
                st.write("• **Williams VIX Fix** - Fear gauge calculation")
                st.write("• **MA Confluence** - 20/50/200 trend alignment")
                st.write("• **Volume Confluence** - Volume confirmation")
                st.write("• **Enhanced VWAP** - Price/volume relationship")
                st.write("• **RSI Momentum** - Oversold detection")
                st.write("• **Volatility Filter** - Market regime analysis")
                st.write("• **Signal Classification** - GOOD/STRONG/VERY_STRONG")
        
        # VWV Signal Classifications
        with st.expander("📊 VWV Signal Classifications", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **🟢 GOOD Signal (≥3.5)**
                - Basic oversold condition
                - Some confluence alignment
                - Standard probability setup
                - Expected frequency: ~40 signals/year
                """)
            
            with col2:
                st.markdown("""
                **🟡 STRONG Signal (≥4.5)**
                - Enhanced oversold condition  
                - Good confluence confirmation
                - Above-average probability
                - Expected frequency: ~20 signals/year
                """)
            
            with col3:
                st.markdown("""
                **🔴 VERY_STRONG Signal (≥5.5)**
                - Extreme oversold condition
                - Full confluence alignment
                - High probability setup
                - Expected frequency: ~8 signals/year
                """)
        
        # Show current market status
        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.write("1. **Use Quick Links** - Click any symbol in the sidebar for instant analysis")
            st.write("2. **Type & Enter** - Enter a symbol and press Enter for immediate analysis")
            st.write("3. **View VWV Signals** - Professional Williams VIX Fix market timing")
            st.write("4. **Check Risk Levels** - Automatic stop loss and take profit calculations")
            st.write("5. **Explore all sections** - Technical, Fundamental, Market, Options, Charts")
            st.write("6. **Toggle sections** on/off in Analysis Sections panel")

    # Footer
    st.markdown("---")
    st.write("### 📊 System Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Version:** VWV Professional v4.0 - Enhanced")
        st.write(f"**Status:** ✅ All Systems Operational")
    with col2:
        st.write(f"**Features:** VWV Signals, Technical, Fundamental, Market, Options, Charts")
        st.write(f"**Core System:** ✅ Williams VIX Fix Integrated")
    with col3:
        st.write(f"**Signal Types:** 🟢 GOOD 🟡 STRONG 🔴 VERY_STRONG")
        st.write(f"**User Experience:** ✅ Enter Key + One-Click Analysis")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.write("Please refresh the page and try again.")
        
        if st.checkbox("Show Error Details"):
            st.exception(e)
