"""
VWV Professional Trading System v4.2 - Complete with Volume & Volatility Analysis
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
# NEW: Volume and Volatility Analysis imports
from analysis.volume import (
    calculate_volume_analysis,
    get_volume_interpretation,
    calculate_market_volume_comparison
)
from analysis.volatility import (
    calculate_volatility_analysis,
    get_volatility_interpretation,
    calculate_market_volatility_comparison,
    get_volatility_regime_for_options
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
            st.session_state.show_volume_analysis = st.checkbox(
                "Volume Analysis", 
                value=st.session_state.show_volume_analysis,
                key="toggle_volume"
            )
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
    """Display individual technical analysis section"""
    if not st.session_state.show_technical_analysis:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Individual Technical Analysis", expanded=True):
        
        # COMPOSITE TECHNICAL SCORE - Use modular component
        composite_score, score_details = calculate_composite_technical_score(analysis_results)
        score_bar_html = create_technical_score_bar(composite_score, score_details)
        st.markdown(score_bar_html, unsafe_allow_html=True)
        
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
    """Display volume analysis section"""
    if not st.session_state.show_volume_analysis:
        return
        
    with st.expander("📊 Volume Analysis - 5D/30D Rolling & Trends", expanded=True):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volume_analysis = enhanced_indicators.get('volume_analysis', {})
        
        if volume_analysis.get('calculation_success', False):
            
            # Volume metrics overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                volume_5d = volume_analysis.get('volume_5d_avg', 0)
                st.metric("5-Day Avg Volume", format_large_number(volume_5d))
            with col2:
                volume_30d = volume_analysis.get('volume_30d_avg', 0)
                st.metric("30-Day Avg Volume", format_large_number(volume_30d))
            with col3:
                volume_ratio = volume_analysis.get('volume_ratio_5d_30d', 1.0)
                deviation_pct = volume_analysis.get('volume_deviation_pct', 0)
                st.metric("5D/30D Ratio", f"{volume_ratio:.2f}", f"{deviation_pct:+.1f}%")
            with col4:
                volume_regime = volume_analysis.get('volume_regime', 'Normal')
                regime_score = volume_analysis.get('regime_score', 50)
                st.metric("Volume Regime", volume_regime, f"Score: {regime_score}")
            
            # Volume trend analysis
            st.subheader("📈 Volume Trend Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                trend_direction = volume_analysis.get('volume_5d_trend_direction', 'Unknown')
                trend_strength = volume_analysis.get('volume_5d_trend_strength', 0)
                momentum = volume_analysis.get('volume_momentum', 0)
                
                st.write(f"**Trend Direction:** {trend_direction}")
                st.write(f"**Trend Strength:** {trend_strength:.2f}%")
                st.write(f"**Volume Momentum:** {momentum:+.2f}%")
                
                # Volume breakout detection
                breakout_status = volume_analysis.get('volume_breakout', 'Normal Range')
                z_score = volume_analysis.get('volume_z_score', 0)
                st.write(f"**Breakout Status:** {breakout_status}")
                st.write(f"**Z-Score:** {z_score:.2f}")
                
            with col2:
                # Advanced volume metrics
                current_volume = volume_analysis.get('current_volume', 0)
                acceleration = volume_analysis.get('volume_acceleration', 0)
                consistency = volume_analysis.get('volume_consistency', 50)
                strength_factor = volume_analysis.get('volume_strength_factor', 1.0)
                
                st.write(f"**Current Volume:** {format_large_number(current_volume)}")
                st.write(f"**Volume Acceleration:** {acceleration:+.2f}%")
                st.write(f"**Volume Consistency:** {consistency:.1f}%")
                st.write(f"**Strength Factor:** {strength_factor:.3f}")
            
            # Volume interpretation
            volume_interpretation = get_volume_interpretation(volume_analysis)
            if volume_interpretation:
                st.subheader("🔍 Volume Analysis Interpretation")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Overall:** {volume_interpretation.get('overall', 'N/A')}")
                    st.info(f"**Trend:** {volume_interpretation.get('trend', 'N/A')}")
                with col2:
                    st.info(f"**Strength:** {volume_interpretation.get('strength', 'N/A')}")
                    st.info(f"**Recommendation:** {volume_interpretation.get('recommendation', 'N/A')}")
        
        else:
            st.warning("⚠️ Volume analysis not available - insufficient data or calculation error")
            if 'error' in volume_analysis:
                st.error(f"Error: {volume_analysis['error']}")

def show_volatility_analysis(analysis_results, show_debug=False):
    """Display volatility analysis section"""
    if not st.session_state.show_volatility_analysis:
        return
        
    with st.expander("🌡️ Volatility Analysis - 5D/30D Rolling & Regime Detection", expanded=True):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volatility_analysis = enhanced_indicators.get('volatility_analysis', {})
        
        if volatility_analysis.get('calculation_success', False):
            
            # Volatility metrics overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                vol_5d = volatility_analysis.get('volatility_5d', 0)
                st.metric("5-Day Volatility", f"{vol_5d:.1f}%")
            with col2:
                vol_30d = volatility_analysis.get('volatility_30d', 0)
                st.metric("30-Day Volatility", f"{vol_30d:.1f}%")
            with col3:
                vol_ratio = volatility_analysis.get('vol_ratio_5d_30d', 1.0)
                vol_deviation = volatility_analysis.get('vol_deviation_pct', 0)
                st.metric("5D/30D Ratio", f"{vol_ratio:.2f}", f"{vol_deviation:+.1f}%")
            with col4:
                vol_regime = volatility_analysis.get('vol_regime', 'Normal')
                regime_score = volatility_analysis.get('regime_score', 50)
                st.metric("Vol Regime", vol_regime, f"Score: {regime_score}")
            
            # Volatility trend and cycle analysis
            st.subheader("📊 Volatility Trend & Cycle Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                trend_direction = volatility_analysis.get('vol_5d_trend_direction', 'Unknown')
                trend_strength = volatility_analysis.get('vol_5d_trend_strength', 0)
                vol_momentum = volatility_analysis.get('vol_momentum', 0)
                cycle_position = volatility_analysis.get('cycle_position', 'Unknown')
                
                st.write(f"**Trend Direction:** {trend_direction}")
                st.write(f"**Trend Strength:** {trend_strength:.2f}%")
                st.write(f"**Vol Momentum:** {vol_momentum:+.2f}%")
                st.write(f"**Cycle Position:** {cycle_position}")
                
            with col2:
                # Advanced volatility metrics
                vol_percentile = volatility_analysis.get('vol_percentile', 50)
                vol_breakout = volatility_analysis.get('vol_breakout', 'Normal Range')
                vol_acceleration = volatility_analysis.get('vol_acceleration', 0)
                options_adjustment = volatility_analysis.get('options_adjustment', 'Neutral')
                
                st.write(f"**Vol Percentile:** {vol_percentile:.1f}%")
                st.write(f"**Vol Breakout:** {vol_breakout}")
                st.write(f"**Vol Acceleration:** {vol_acceleration:+.2f}%")
                st.write(f"**Options Strategy:** {options_adjustment}")
            
            # Risk metrics
            st.subheader("⚠️ Risk Assessment")
            col1, col2, col3 = st.columns(3)
            with col1:
                current_return = volatility_analysis.get('current_return', 0)
                st.metric("Current Return", f"{current_return:+.2f}%")
            with col2:
                risk_adjusted_return = volatility_analysis.get('risk_adjusted_return', 0)
                st.metric("Risk-Adj Return", f"{risk_adjusted_return:.2f}")
            with col3:
                vol_of_vol = volatility_analysis.get('vol_of_vol', 0)
                st.metric("Vol of Vol", f"{vol_of_vol:.2f}")
            
            # Volatility interpretation
            vol_interpretation = get_volatility_interpretation(volatility_analysis)
            if vol_interpretation:
                st.subheader("🔍 Volatility Analysis Interpretation")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Overall:** {vol_interpretation.get('overall', 'N/A')}")
                    st.info(f"**Trend:** {vol_interpretation.get('trend', 'N/A')}")
                    st.info(f"**Regime:** {vol_interpretation.get('regime', 'N/A')}")
                with col2:
                    st.info(f"**Options:** {vol_interpretation.get('options', 'N/A')}")
                    st.info(f"**Risk:** {vol_interpretation.get('risk', 'N/A')}")
        
        else:
            st.warning("⚠️ Volatility analysis not available - insufficient data or calculation error")
            if 'error' in volatility_analysis:
                st.error(f"Error: {volatility_analysis['error']}")

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

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis section"""
    if not st.session_state.show_market_correlation:
        return
        
    with st.expander("🌐 Market Correlation & Market-Wide Analysis", expanded=True):
        
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
        
        # Market-Wide Volume & Volatility Analysis
        st.subheader("🌊 Market-Wide Volume & Volatility Environment")
        
        # Get market-wide volume and volatility data
        market_volume_data = calculate_market_volume_comparison(['SPY', 'QQQ', 'IWM'])
        market_vol_data = calculate_market_volatility_comparison(['SPY', 'QQQ', 'IWM'])
        
        if (market_volume_data.get('calculation_success', False) and 
            market_vol_data.get('calculation_success', False)):
            
            # Market context classification
            volume_regime = market_volume_data.get('market_regime', 'Unknown')
            vol_regime = market_vol_data.get('market_regime', 'Unknown')
            
            if 'High' in volume_regime and 'High' in vol_regime:
                market_context = "🔥 High Activity, High Volatility - Event-driven market"
            elif 'High' in volume_regime and 'Low' in vol_regime:
                market_context = "📈 High Volume, Low Volatility - Steady trending market"
            elif 'Low' in volume_regime and 'High' in vol_regime:
                market_context = "⚠️ Low Volume, High Volatility - Unstable/illiquid conditions"
            elif 'Low' in volume_regime and 'Low' in vol_regime:
                market_context = "😴 Low Volume, Low Volatility - Quiet/consolidating market"
            else:
                market_context = "⚖️ Mixed Market Environment"
            
            st.info(f"**Market Context:** {market_context}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔊 Market Volume Environment**")
                avg_volume_ratio = market_volume_data.get('avg_volume_ratio', 1.0)
                avg_regime_score = market_volume_data.get('avg_regime_score', 50)
                
                st.write(f"• **Volume Regime:** {volume_regime}")
                st.write(f"• **Avg Volume Ratio:** {avg_volume_ratio:.2f}")
                st.write(f"• **Regime Score:** {avg_regime_score}/100")
                
                # Individual index breakdown
                individual_volume = market_volume_data.get('individual_metrics', {})
                for symbol, data in individual_volume.items():
                    regime = data.get('volume_regime', 'Unknown')
                    ratio = data.get('volume_ratio_5d_30d', 1.0)
                    st.write(f"  - **{symbol}:** {regime} ({ratio:.2f}x)")
            
            with col2:
                st.write("**🌡️ Market Volatility Environment**")
                avg_volatility = market_vol_data.get('avg_volatility', 0)
                options_env = market_vol_data.get('market_options_env', 'Unknown')
                vix_estimate = market_vol_data.get('vix_estimate', 'Unknown')
                
                st.write(f"• **Volatility Regime:** {vol_regime}")
                st.write(f"• **Avg Volatility:** {avg_volatility:.1f}%")
                st.write(f"• **Options Environment:** {options_env}")
                st.write(f"• **VIX Estimate:** {vix_estimate}")
                
                # Individual index breakdown
                individual_vol = market_vol_data.get('individual_metrics', {})
                for symbol, data in individual_vol.items():
                    regime = data.get('vol_regime', 'Unknown')
                    vol_5d = data.get('volatility_5d', 0)
                    st.write(f"  - **{symbol}:** {regime} ({vol_5d:.1f}%)")
        
        else:
            st.warning("⚠️ Market-wide analysis not available")

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
    """Perform enhanced analysis using modular components"""
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
        
        # NEW: Calculate volume and volatility analysis
        volume_analysis = calculate_volume_analysis(analysis_input)
        volatility_analysis = calculate_volatility_analysis(analysis_input)
        
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
        
        # Step 7: Calculate options levels with volatility regime adjustment
        volatility_regime_info = get_volatility_regime_for_options(volatility_analysis)
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
        
        # Step 8: Calculate confidence intervals
        confidence_analysis = calculate_confidence_intervals(analysis_input)
        
        # Step 9: Build analysis results
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
                'volume_analysis': volume_analysis,  # NEW
                'volatility_analysis': volatility_analysis,  # NEW
                'market_correlations': market_correlations,
                'options_levels': options_levels,
                'graham_score': graham_score,
                'piotroski_score': piotroski_score
            },
            'confidence_analysis': confidence_analysis,
            'volatility_regime_info': volatility_regime_info,  # NEW
            'system_status': 'OPERATIONAL'
        }
        
        # Store results
        data_manager.store_analysis_results(symbol, analysis_results)
        
        return analysis_results
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        return None

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
            analysis_results = perform_enhanced_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results:
                # Show all analysis sections using modular functions
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                show_volume_analysis(analysis_results, controls['show_debug'])  # NEW
                show_volatility_analysis(analysis_results, controls['show_debug'])  # NEW
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                show_options_analysis(analysis_results, controls['show_debug'])
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
    
    else:
        # Welcome message
        st.write("## 🚀 VWV Professional Trading System v4.2 - Volume & Volatility Enhanced")
        st.write("**Enhanced with Volume & Volatility Analysis:** 5D/30D rolling metrics, trend detection, regime classification")
        
        with st.expander("🏗️ Modular Architecture Overview v4.2", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📁 **Active Modules**")
                st.write("✅ **`config/`** - Settings, constants, parameters")
                st.write("✅ **`data/`** - Fetching, validation, management")
                st.write("✅ **`analysis/`** - Technical, fundamental, market, options")
                st.write("✅ **`analysis/volume.py`** - 🆕 Volume analysis module")
                st.write("✅ **`analysis/volatility.py`** - 🆕 Volatility analysis module")
                st.write("✅ **`ui/`** - Components, headers, score bars")
                st.write("✅ **`utils/`** - Helpers, decorators, formatters")
                
            with col2:
                st.write("### 🎯 **All Sections Working**")
                st.write("• **Individual Technical Analysis** (Updated composite scoring)")
                st.write("• **🆕 Volume Analysis** (5D/30D rolling, trends, regime)")
                st.write("• **🆕 Volatility Analysis** (5D/30D rolling, cycle, regime)")
                st.write("• **Fundamental Analysis** (Graham & Piotroski)")
                st.write("• **Market Correlation & Breakouts**")
                st.write("• **Options Analysis with Greeks**")
                st.write("• **Statistical Confidence Intervals**")
        
        # Show current market status
        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide v4.2", expanded=True):
            st.write("1. **Enter a symbol** in the sidebar (e.g., AAPL, SPY, QQQ)")
            st.write("2. **Click 'Analyze Symbol'** to run complete analysis")
            st.write("3. **🆕 Volume Analysis:** 5-day vs 30-day rolling averages, trend detection")
            st.write("4. **🆕 Volatility Analysis:** Regime classification, cycle position, options strategy")
            st.write("5. **🔄 Updated Technical Score:** Now includes Volume (15%) & Volatility (15%) components")
            st.write("6. **🌊 Market-Wide Analysis:** Volume/volatility environment across major indices")
            st.write("7. **Toggle sections** on/off in Analysis Sections panel")

    # Footer
    st.markdown("---")
    st.write("### 📊 System Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Version:** VWV Professional v4.2 - Volume & Volatility Enhanced")
        st.write(f"**Architecture:** Full Modular Implementation + New Analysis Modules")
    with col2:
        st.write(f"**Status:** ✅ All Modules Active")
        st.write(f"**Sections:** Technical, 🆕Volume, 🆕Volatility, Fundamental, Market, Options")
    with col3:
        st.write(f"**New Features:** Volume & Volatility 5D/30D rolling analysis")
        st.write(f"**Enhanced:** Composite technical scoring with new components")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.write("Please refresh the page and try again.")
        
        if st.checkbox("Show Error Details"):
            st.exception(e)
