"""
VWV Professional Trading System v4.2.1 Enhanced
Complete Modular Version with Volume & Volatility Analysis
Main application with all sections working properly
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
# NEW: Volume & Volatility Analysis Modules v4.2.1
# CORRECTED CODE
from analysis.volume import (
    analyze_volume_profile,
    interpret_volume_data,
    compare_market_volume
)
from analysis.volatility import (
    calculate_volatility_analysis,
    get_volatility_interpretation,
    calculate_market_volatility_comparison
)
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
    
    # Initialize session state with new Volume & Volatility toggles
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
    
    # Basic controls
    if 'selected_symbol' in st.session_state:
        default_symbol = st.session_state.selected_symbol
        del st.session_state.selected_symbol
    else:
        default_symbol = UI_SETTINGS['default_symbol']
        
    symbol = st.sidebar.text_input("Symbol", value=default_symbol, help="Enter stock symbol").upper()
    period = st.sidebar.selectbox("Data Period", UI_SETTINGS['periods'], index=3)
    
    # Enhanced Section Control Panel with new Volume & Volatility toggles
    with st.sidebar.expander("📋 Analysis Sections", expanded=False):
        st.write("**Toggle Analysis Sections:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.show_technical_analysis = st.checkbox(
                "Technical Analysis", 
                value=st.session_state.show_technical_analysis,
                key="toggle_technical"
            )
            st.session_state.show_volume_analysis = st.checkbox(
                "📊 Volume Analysis", 
                value=st.session_state.show_volume_analysis,
                key="toggle_volume"
            )
            st.session_state.show_volatility_analysis = st.checkbox(
                "🌡️ Volatility Analysis", 
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
    
    # Main analyze button
    analyze_button = st.sidebar.button("📊 Analyze Symbol", type="primary", use_container_width=True)
    
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

def show_individual_technical_analysis(analysis_results, show_debug=False):
    """Display individual technical analysis section with enhanced scoring"""
    if not st.session_state.show_technical_analysis:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Individual Technical Analysis", expanded=True):
        
        # ENHANCED COMPOSITE TECHNICAL SCORE v4.2.1 - Use modular component
        composite_score, score_details = calculate_composite_technical_score(analysis_results)
        score_bar_html = create_technical_score_bar(composite_score, score_details)
        st.markdown(score_bar_html, unsafe_allow_html=True)
        
        # Show enhanced scoring breakdown
        if score_details.get('version') == 'v4.2.1_enhanced':
            st.info("🆕 **Enhanced Scoring v4.2.1:** Now includes Volume Analysis (15%) and Volatility Analysis (15%) for more comprehensive technical assessment")
        
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
    """Display Volume Analysis section - NEW v4.2.1"""
    if not st.session_state.show_volume_analysis:
        return
        
    with st.expander("📊 Volume Analysis - 5D/30D Rolling & Trends", expanded=True):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volume_analysis = enhanced_indicators.get('volume_analysis', {})
        
        if 'error' in volume_analysis:
            st.warning(f"⚠️ Volume analysis not available: {volume_analysis['error']}")
            return
        
        # Volume metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_volume = volume_analysis.get('current_volume', 0)
            st.metric("Current Volume", format_large_number(current_volume))
        
        with col2:
            volume_5d_avg = volume_analysis.get('volume_5d_avg', 0)
            st.metric("5D Avg Volume", format_large_number(volume_5d_avg))
        
        with col3:
            volume_ratio = volume_analysis.get('volume_ratio_5d_30d', 1.0)
            st.metric("5D/30D Ratio", f"{volume_ratio:.2f}x", 
                     "📈 High" if volume_ratio > 1.2 else "📉 Low" if volume_ratio < 0.8 else "⚖️ Normal")
        
        with col4:
            volume_regime = volume_analysis.get('volume_regime', 'Normal')
            regime_score = volume_analysis.get('regime_score', 50)
            st.metric("Volume Regime", volume_regime, f"Score: {regime_score}")
        
        # Volume trend analysis
        st.subheader("📈 Volume Trend Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            trend_pct = volume_analysis.get('volume_5d_trend_pct', 0)
            z_score = volume_analysis.get('volume_z_score', 0)
            
            st.write("**5-Day Volume Trend:**")
            if trend_pct > 10:
                st.success(f"📈 Strong Increase: +{trend_pct:.1f}%")
            elif trend_pct > 0:
                st.info(f"↗️ Moderate Increase: +{trend_pct:.1f}%")
            elif trend_pct > -10:
                st.warning(f"↘️ Moderate Decrease: {trend_pct:.1f}%")
            else:
                st.error(f"📉 Strong Decrease: {trend_pct:.1f}%")
            
            st.write(f"**Volume Z-Score:** {z_score:.2f}")
            if abs(z_score) >= 2.0:
                st.write("🎯 **Significant breakout signal**")
            elif abs(z_score) >= 1.5:
                st.write("📊 **Moderate volume signal**")
            else:
                st.write("😐 **Normal volume range**")
        
        with col2:
            # Volume interpretations
            volume_interpretations = get_volume_interpretation(volume_analysis)
            
            st.write("**Volume Analysis:**")
            st.write(volume_interpretations.get('regime_interpretation', 'N/A'))
            
            st.write("**Trading Implications:**")
            st.write(volume_interpretations.get('trading_implications', 'N/A'))
        
        # Volume breakout analysis
        breakout_data = volume_analysis.get('volume_breakout', {})
        if breakout_data:
            breakout_type = breakout_data.get('type', 'None')
            breakout_strength = breakout_data.get('strength', 50)
            
            if breakout_type != 'None':
                if "Bullish" in breakout_type:
                    st.success(f"🚀 **Volume Breakout:** {breakout_type} (Strength: {breakout_strength})")
                else:
                    st.error(f"🔻 **Volume Breakdown:** {breakout_type} (Strength: {breakout_strength})")

def show_volatility_analysis(analysis_results, show_debug=False):
    """Display Volatility Analysis section - NEW v4.2.1"""
    if not st.session_state.show_volatility_analysis:
        return
        
    with st.expander("🌡️ Volatility Analysis - 5D/30D Rolling & Regime Detection", expanded=True):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volatility_analysis = enhanced_indicators.get('volatility_analysis', {})
        
        if 'error' in volatility_analysis:
            st.warning(f"⚠️ Volatility analysis not available: {volatility_analysis['error']}")
            return
        
        # Volatility metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_5d_vol = volatility_analysis.get('current_5d_vol', 0)
            st.metric("5D Volatility", f"{current_5d_vol:.2f}%")
        
        with col2:
            current_30d_vol = volatility_analysis.get('current_30d_vol', 0)
            st.metric("30D Volatility", f"{current_30d_vol:.2f}%")
        
        with col3:
            vol_percentile = volatility_analysis.get('vol_percentile', 50)
            st.metric("Volatility Percentile", f"{vol_percentile:.1f}%",
                     "📈 High" if vol_percentile > 70 else "📉 Low" if vol_percentile < 30 else "⚖️ Normal")
        
        with col4:
            vol_regime = volatility_analysis.get('vol_regime', 'Normal')
            regime_score = volatility_analysis.get('regime_score', 50)
            st.metric("Volatility Regime", vol_regime, f"Score: {regime_score}")
        
        # Volatility trend analysis
        st.subheader("🌡️ Volatility Trend Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            vol_trend_pct = volatility_analysis.get('vol_5d_trend_pct', 0)
            vol_ratio = volatility_analysis.get('vol_ratio_5d_30d', 1.0)
            
            st.write("**5-Day Volatility Trend:**")
            if vol_trend_pct > 20:
                st.error(f"⚡ Strong Expansion: +{vol_trend_pct:.1f}%")
            elif vol_trend_pct > 10:
                st.warning(f"📈 Moderate Expansion: +{vol_trend_pct:.1f}%")
            elif vol_trend_pct > -10:
                st.info(f"→ Stable: {vol_trend_pct:+.1f}%")
            elif vol_trend_pct > -20:
                st.success(f"📉 Moderate Contraction: {vol_trend_pct:.1f}%")
            else:
                st.success(f"😴 Strong Contraction: {vol_trend_pct:.1f}%")
            
            st.write(f"**5D/30D Ratio:** {vol_ratio:.2f}x")
        
        with col2:
            # Volatility interpretations
            vol_interpretations = get_volatility_interpretation(volatility_analysis)
            
            st.write("**Volatility Analysis:**")
            st.write(vol_interpretations.get('regime_interpretation', 'N/A'))
            
            st.write("**Trading Implications:**")
            st.write(vol_interpretations.get('trading_implications', 'N/A'))
        
        # Options strategy guidance
        options_guidance = volatility_analysis.get('options_guidance', {})
        if options_guidance:
            st.subheader("🎯 Options Strategy Guidance")
            
            strategy = options_guidance.get('strategy', 'Neutral')
            description = options_guidance.get('description', 'Standard approach')
            adjustment = options_guidance.get('adjustment', 'Standard approach')
            risk_level = options_guidance.get('risk_level', 'Moderate')
            
            if "Selling" in strategy:
                st.success(f"📈 **{strategy}:** {description}")
                st.write(f"🎯 **Strategy Adjustment:** {adjustment}")
                st.write(f"⚠️ **Risk Level:** {risk_level}")
            elif "Buying" in strategy:
                st.info(f"📉 **{strategy}:** {description}")
                st.write(f"🎯 **Strategy Adjustment:** {adjustment}")
                st.write(f"⚠️ **Risk Level:** {risk_level}")
            else:
                st.warning(f"⚖️ **{strategy}:** {description}")
                st.write(f"🎯 **Strategy Adjustment:** {adjustment}")

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
        
        elif is_etf_symbol:
            st.info(f"ℹ️ **{analysis_results['symbol']} is an ETF** - Fundamental analysis is not applicable to Exchange-Traded Funds.")

def show_enhanced_market_analysis(analysis_results, show_debug=False):
    """Display enhanced market correlation analysis with Volume/Volatility environment - UPDATED v4.2.1"""
    if not st.session_state.show_market_correlation:
        return
        
    with st.expander("🌊 Enhanced Market Analysis - Correlation & Volume/Volatility Environment", expanded=True):
        
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
        
        # NEW: Market-Wide Volume/Volatility Environment Analysis v4.2.1
        st.subheader("🌊 Market Volume/Volatility Environment")
        
        # Get market-wide volume analysis
        market_volume_data = calculate_market_volume_comparison(['SPY', 'QQQ', 'IWM'], show_debug=show_debug)
        market_vol_data = calculate_market_volatility_comparison(['SPY', 'QQQ', 'IWM'], show_debug=show_debug)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Market Volume Environment:**")
            if market_volume_data and 'error' not in market_volume_data:
                volume_env = market_volume_data.get('market_environment', {})
                volume_description = volume_env.get('description', 'Analysis not available')
                st.write(volume_description)
                
                if 'individual_results' in market_volume_data:
                    st.write("**Individual Volume Results:**")
                    for symbol, data in market_volume_data['individual_results'].items():
                        regime = data.get('volume_regime', 'Unknown')
                        ratio = data.get('volume_ratio_5d_30d', 1.0)
                        st.write(f"• {symbol}: {regime} ({ratio:.1f}x)")
            else:
                st.write("Volume environment analysis not available")
        
        with col2:
            st.write("**🌡️ Market Volatility Environment:**")
            if market_vol_data and 'error' not in market_vol_data:
                vol_env = market_vol_data.get('market_environment', {})
                vol_description = vol_env.get('description', 'Analysis not available')
                st.write(vol_description)
                
                if 'individual_results' in market_vol_data:
                    st.write("**Individual Volatility Results:**")
                    for symbol, data in market_vol_data['individual_results'].items():
                        regime = data.get('vol_regime', 'Unknown')
                        vol_pct = data.get('current_5d_vol', 0)
                        st.write(f"• {symbol}: {regime} ({vol_pct:.1f}%)")
            else:
                st.write("Volatility environment analysis not available")
        
        # Market context classification
        if (market_volume_data and 'error' not in market_volume_data and 
            market_vol_data and 'error' not in market_vol_data):
            
            volume_env = market_volume_data.get('market_environment', {}).get('environment', 'Unknown')
            vol_env = market_vol_data.get('market_environment', {}).get('environment', 'Unknown')
            
            st.subheader("🎯 Market Context Classification")
            
            # Determine market context
            if 'High' in volume_env and 'High' in vol_env:
                context = "🔥 High Activity, High Volatility - Event-driven market"
                context_color = "error"
            elif 'High' in volume_env and 'Low' in vol_env:
                context = "📈 High Volume, Low Volatility - Steady trending market"
                context_color = "success"
            elif 'Low' in volume_env and 'High' in vol_env:
                context = "⚡ Low Volume, High Volatility - Uncertain/choppy market"
                context_color = "warning"
            elif 'Low' in volume_env and 'Low' in vol_env:
                context = "😴 Low Volume, Low Volatility - Quiet/consolidating market"
                context_color = "info"
            else:
                context = "⚖️ Normal Volume & Volatility - Balanced market conditions"
                context_color = "info"
            
            if context_color == "error":
                st.error(context)
            elif context_color == "success":
                st.success(context)
            elif context_color == "warning":
                st.warning(context)
            else:
                st.info(context)

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

def perform_enhanced_analysis(symbol, period, show_debug=False):
    """Perform enhanced analysis using modular components with Volume & Volatility v4.2.1"""
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
        daily_vwap = calculate_daily_vwap(analysis_input)
        fibonacci_emas = calculate_fibonacci_emas(analysis_input)
        point_of_control = calculate_point_of_control_enhanced(analysis_input)
        weekly_deviations = calculate_weekly_deviations(analysis_input)
        comprehensive_technicals = calculate_comprehensive_technicals(analysis_input)
        
        # Step 5: NEW - Calculate Volume Analysis v4.2.1
        volume_analysis = calculate_volume_analysis(analysis_input)
        
        # Step 6: NEW - Calculate Volatility Analysis v4.2.1
        volatility_analysis = calculate_volatility_analysis(analysis_input)
        
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
        
        # Step 10: Calculate confidence intervals
        confidence_analysis = calculate_confidence_intervals(analysis_input)
        
        # Step 11: Build enhanced analysis results v4.2.1
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
                'volume_analysis': volume_analysis,  # NEW v4.2.1
                'volatility_analysis': volatility_analysis,  # NEW v4.2.1
                'market_correlations': market_correlations,
                'options_levels': options_levels,
                'graham_score': graham_score,
                'piotroski_score': piotroski_score
            },
            'confidence_analysis': confidence_analysis,
            'system_status': 'OPERATIONAL',
            'version': 'v4.2.1_enhanced'
        }
        
        # Store results
        data_manager.store_analysis_results(symbol, analysis_results)
        
        return analysis_results
        
    except Exception as e:
        st.error(f"❌ Enhanced analysis failed: {str(e)}")
        return None

def main():
    """Main application function v4.2.1"""
    # Create header using modular component
    create_header()
    
    # Create sidebar and get controls
    controls = create_sidebar_controls()
    
    # Main logic flow
    if controls['analyze_button'] and controls['symbol']:
        # Add symbol to recently viewed
        add_to_recently_viewed(controls['symbol'])
        
        st.write("## 📊 VWV Trading Analysis v4.2.1 Enhanced")
        
        with st.spinner(f"Analyzing {controls['symbol']} with Volume & Volatility..."):
            
            # Perform enhanced analysis using modular components v4.2.1
            analysis_results = perform_enhanced_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results:
                # Show all analysis sections using modular functions
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                show_volume_analysis(analysis_results, controls['show_debug'])  # NEW v4.2.1
                show_volatility_analysis(analysis_results, controls['show_debug'])  # NEW v4.2.1
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                show_enhanced_market_analysis(analysis_results, controls['show_debug'])  # ENHANCED v4.2.1
                show_options_analysis(analysis_results, controls['show_debug'])
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
                # Debug information
                if controls['show_debug']:
                    with st.expander("🐛 Debug Information v4.2.1", expanded=False):
                        st.write("### Enhanced Analysis Results Structure v4.2.1")
                        st.json(analysis_results, expanded=False)
                        
                        st.write("### Data Manager Summary")
                        data_manager = get_data_manager()
                        summary = data_manager.get_data_summary()
                        st.json(summary)
                        
                        # NEW: Volume & Volatility Debug Info
                        volume_analysis = analysis_results.get('enhanced_indicators', {}).get('volume_analysis', {})
                        volatility_analysis = analysis_results.get('enhanced_indicators', {}).get('volatility_analysis', {})
                        
                        if volume_analysis and 'error' not in volume_analysis:
                            st.write("### Volume Analysis Debug")
                            st.json(volume_analysis)
                        
                        if volatility_analysis and 'error' not in volatility_analysis:
                            st.write("### Volatility Analysis Debug")
                            st.json(volatility_analysis)
    
    else:
        # Welcome message with v4.2.1 enhancements
        st.write("## 🚀 VWV Professional Trading System v4.2.1 Enhanced")
        st.write("**All modules active with NEW Volume & Volatility Analysis:** Technical, Volume, Volatility, Fundamental, Market, Options, UI Components")
        
        with st.expander("🏗️ Enhanced Modular Architecture v4.2.1", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📁 **Enhanced Active Modules**")
                st.write("✅ **`config/`** - Settings, constants, parameters")
                st.write("✅ **`data/`** - Fetching, validation, management")
                st.write("✅ **`analysis/`** - Technical, **volume**, **volatility**, fundamental, market, options")
                st.write("✅ **`ui/`** - Components, headers, score bars")
                st.write("✅ **`utils/`** - Helpers, decorators, formatters")
                
            with col2:
                st.write("### 🎯 **All Sections Working + NEW v4.2.1**")
                st.write("• **Individual Technical Analysis** (Enhanced Scoring)")
                st.write("• **🆕 Volume Analysis** (5D/30D Rolling & Trends)")
                st.write("• **🆕 Volatility Analysis** (5D/30D Rolling & Regime)")
                st.write("• **Fundamental Analysis** (Graham & Piotroski)")
                st.write("• **Enhanced Market Analysis** (Correlation & Vol/Vol Environment)")
                st.write("• **Options Analysis with Greeks**")
                st.write("• **Statistical Confidence Intervals**")
        
        # NEW: Enhanced features summary v4.2.1
        with st.expander("🆕 NEW in v4.2.1 - Volume & Volatility Analysis", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📊 **Volume Analysis Features**")
                st.write("• **5-Day Rolling Volume** with trend detection")
                st.write("• **30-Day Volume Comparison** with deviation metrics")
                st.write("• **Volume Regime Classification** (Extreme High → Low)")
                st.write("• **Volume Breakout Detection** using Z-scores")
                st.write("• **Volume Strength Factor** for composite scoring")
                st.write("• **Market-Wide Volume Environment** analysis")
                
            with col2:
                st.write("### 🌡️ **Volatility Analysis Features**")
                st.write("• **5-Day Rolling Volatility** (annualized)")
                st.write("• **30-Day Volatility Comparison** with ratios")
                st.write("• **Volatility Regime Classification** with percentiles")
                st.write("• **Options Strategy Guidance** based on vol regime")
                st.write("• **Volatility Strength Factor** for composite scoring")
                st.write("• **Market-Wide Volatility Environment** analysis")
        
        # Show current market status
        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")
        
        # Enhanced quick start guide v4.2.1
        with st.expander("🚀 Enhanced Quick Start Guide v4.2.1", expanded=True):
            st.write("1. **Enter a symbol** in the sidebar (e.g., AAPL, SPY, QQQ)")
            st.write("2. **Click 'Analyze Symbol'** to run complete enhanced analysis")
            st.write("3. **View all sections:** Technical (Enhanced), **Volume**, **Volatility**, Fundamental, Market, Options")
            st.write("4. **Toggle sections** on/off in Analysis Sections panel")
            st.write("5. **🆕 NEW:** Volume and Volatility analysis now integrated into Technical Scoring")
            st.write("6. **🆕 NEW:** Market-wide Volume/Volatility environment classification")

    # Enhanced Footer v4.2.1
    st.markdown("---")
    st.write("### 📊 Enhanced System Information v4.2.1")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Version:** VWV Professional v4.2.1 Enhanced")
        st.write(f"**Architecture:** Full Modular Implementation + Volume/Volatility")
    with col2:
        st.write(f"**Status:** ✅ All Modules Active + NEW Features")
        st.write(f"**Sections:** Technical+, Volume, Volatility, Fundamental, Market, Options")
    with col3:
        st.write(f"**Components:** config, data, analysis (enhanced), ui, utils")
        st.write(f"**Interface:** Complete multi-section experience with V&V")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error v4.2.1: {str(e)}")
        st.write("Please refresh the page and try again.")
        
        if st.checkbox("Show Enhanced Error Details"):
            st.exception(e)
