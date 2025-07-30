"""
VWV Professional Trading System v5.0 - Complete Enhanced Version
Enhanced with Williams VIX Fix, Insider Analysis, Market Divergence, Tech Sentiment, and Sigma Levels
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Import our enhanced modular components
from config.settings import (
    DEFAULT_VWV_CONFIG, UI_SETTINGS, PARAMETER_RANGES,
    VWV_CORE_CONFIG, DIVERGENCE_CONFIG, INSIDER_CONFIG,
    OPTIONS_ENHANCED_CONFIG, TECH_SENTIMENT_CONFIG
)
from config.constants import SYMBOL_DESCRIPTIONS, QUICK_LINK_CATEGORIES, MAJOR_INDICES
from data.manager import get_data_manager
from data.fetcher import get_market_data_enhanced, is_etf

# Enhanced analysis imports
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
    calculate_enhanced_breakout_analysis
)
from analysis.vwv_core import (
    calculate_vwv_confluence_score,
    calculate_vwv_risk_management
)
from analysis.options import (
    calculate_options_levels_enhanced,
    calculate_confidence_intervals,
    calculate_enhanced_sigma_levels
)

# New v5.0 analysis modules
from analysis.vwv_core import (
    calculate_vwv_confluence_score,
    calculate_vwv_risk_management,
    get_vwv_signal_history
)
from analysis.insider import calculate_insider_score
from analysis.divergence import calculate_market_divergence_analysis
from analysis.tech_sentiment import calculate_tech_sector_sentiment_analysis

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
    page_title="VWV Professional Trading System v5.0",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_sidebar_controls():
    """Create enhanced sidebar controls for v5.0 and return analysis parameters"""
    st.sidebar.title("📊 VWV Trading Analysis v5.0")
    
    # Initialize session state for all sections
    section_states = {
        'show_vwv_core_analysis': True,
        'show_technical_analysis': True,
        'show_fundamental_analysis': True,
        'show_market_correlation': True,
        'show_options_analysis': True,
        'show_confidence_intervals': True,
        'show_insider_analysis': True,
        'show_divergence_analysis': True,
        'show_tech_sentiment': True
    }
    
    for key, default_value in section_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    if 'recently_viewed' not in st.session_state:
        st.session_state.recently_viewed = []
    
    # Basic controls
    if 'selected_symbol' in st.session_state:
        default_symbol = st.session_state.selected_symbol
        del st.session_state.selected_symbol
    else:
        default_symbol = UI_SETTINGS['default_symbol']
        
    symbol = st.sidebar.text_input("Symbol", value=default_symbol, help="Enter stock symbol").upper()
    period = st.sidebar.selectbox("Data Period", UI_SETTINGS['periods'], index=1)  # FIXED: Default to 3mo (index 1) (index 1)
    
    # Enhanced Section Control Panel
    with st.sidebar.expander("📋 Analysis Sections", expanded=False):
        st.write("**Core VWV System:**")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.show_vwv_core_analysis = st.checkbox(
                "VWV Core (NEW)", 
                value=st.session_state.get('show_vwv_core_analysis', True),
                key="toggle_vwv_core",
                help="Williams VIX Fix 6-component confluence system"
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
            st.session_state.show_insider_analysis = st.checkbox(
                "Insider Analysis", 
                value=st.session_state.show_insider_analysis,
                key="toggle_insider",
                help="NEW: Insider buying/selling analysis"
            )
            st.session_state.show_divergence_analysis = st.checkbox(
                "Market Divergence", 
                value=st.session_state.show_divergence_analysis,
                key="toggle_divergence",
                help="NEW: Multi-ETF relative strength analysis"
            )
        
        st.write("**Sector Analysis:**")
        st.session_state.show_tech_sentiment = st.checkbox(
            "Tech Sentiment (FNGD/FNGU)", 
            value=st.session_state.show_tech_sentiment,
            key="toggle_tech_sentiment",
            help="NEW: Tech sector sentiment via leveraged ETFs"
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

    # Enhanced debug and settings
    with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
        show_debug = st.checkbox("🐛 Show Debug Info", value=False)
        
        # VWV Core settings
        st.write("**VWV Core Settings:**")
        wvf_multiplier = st.slider("WVF Multiplier", 0.5, 3.0, VWV_CORE_CONFIG['wvf_multiplier'], 0.1)
        
        # Options settings
        st.write("**Options Settings:**")
        options_dte = st.multiselect("Days to Expiry", [7, 14, 21, 30, 45, 60], default=[7, 14, 30, 45])
    
    return {
        'symbol': symbol,
        'period': period,
        'analyze_button': analyze_button,
        'show_debug': show_debug,
        'wvf_multiplier': wvf_multiplier,
        'options_dte': options_dte if options_dte else [7, 14, 30, 45]
    }

def add_to_recently_viewed(symbol):
    """Add symbol to recently viewed - updated for 9 symbols"""
    if symbol and symbol != "":
        if symbol in st.session_state.recently_viewed:
            st.session_state.recently_viewed.remove(symbol)
        st.session_state.recently_viewed.insert(0, symbol)
        st.session_state.recently_viewed = st.session_state.recently_viewed[:9]

def show_vwv_core_analysis(analysis_results, show_debug=False):
    """Display VWV Core Williams VIX Fix analysis section - NEW in v5.0"""
    if not st.session_state.get('show_vwv_core_analysis', True):
        return
        
    with st.expander(f"🎯 {analysis_results['symbol']} - VWV Core Signal Analysis (Williams VIX Fix)", expanded=True):
        
        vwv_data = analysis_results.get('vwv_analysis', {})
        
        if 'error' in vwv_data:
            st.error(f"❌ VWV Core Analysis Error: {vwv_data['error']}")
            return
        
        # VWV Core metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            vwv_score = vwv_data.get('final_score', 0)
            signal_class = vwv_data.get('signal_classification', 'WEAK')
            st.metric("VWV Confluence Score", f"{vwv_score:.2f}", f"Signal: {signal_class}")
        
        with col2:
            wvf_raw = vwv_data.get('component_details', {}).get('wvf_raw', 0)
            st.metric("Williams VIX Fix", f"{wvf_raw:.2f}", "Fear Gauge")
        
        with col3:
            ma_confluence = vwv_data.get('component_details', {}).get('ma_confluence', 0)
            st.metric("MA Confluence", f"{ma_confluence:.2f}", "Trend Alignment")
        
        with col4:
            momentum = vwv_data.get('component_details', {}).get('momentum_component', 0)
            st.metric("Momentum Component", f"{momentum:.2f}", "Oversold Detection")
        
        # Signal interpretation
        if vwv_score >= 5.5:
            st.success("🟢 **VERY STRONG VWV SIGNAL** - Excellent confluence conditions detected")
        elif vwv_score >= 4.5:
            st.success("🟡 **STRONG VWV SIGNAL** - Good confluence conditions")
        elif vwv_score >= 3.5:
            st.info("🔵 **GOOD VWV SIGNAL** - Moderate confluence conditions")
        else:
            st.warning("🟠 **WEAK VWV SIGNAL** - Limited confluence detected")

def show_individual_technical_analysis(analysis_results, show_debug=False):
    """Display individual technical analysis section - ENHANCED in v5.0"""
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

def show_fundamental_analysis(analysis_results, show_debug=False):
    """Display fundamental analysis section - SAME as v3.0"""
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

def show_insider_analysis(analysis_results, show_debug=False):
    """Display insider buying analysis section - NEW in v5.0"""
    if not st.session_state.show_insider_analysis:
        return
        
    with st.expander("💼 Insider Buying Analysis", expanded=True):
        
        insider_data = analysis_results.get('insider_analysis', {})
        
        if 'error' in insider_data:
            st.warning(f"⚠️ Insider Analysis: {insider_data['error']}")
            return
        
        if insider_data.get('sentiment') == 'NO_DATA':
            st.info("ℹ️ **Insider data not available** - This may be due to limited insider activity or data source restrictions.")
            return
        
        # Insider metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            insider_score = insider_data.get('insider_score', 0)
            sentiment = insider_data.get('sentiment', 'NEUTRAL')
            st.metric("Insider Score", f"{insider_score:+.0f}", f"Sentiment: {sentiment}")
        
        with col2:
            transaction_summary = insider_data.get('transaction_summary', {})
            total_transactions = transaction_summary.get('total_transactions', 0)
            st.metric("Recent Transactions", f"{total_transactions}", "30-day period")
        
        with col3:
            buy_transactions = transaction_summary.get('buy_transactions', 0)
            sell_transactions = transaction_summary.get('sell_transactions', 0)
            st.metric("Buy vs Sell", f"{buy_transactions} / {sell_transactions}", "Buys / Sells")
        
        with col4:
            net_flow = transaction_summary.get('net_flow', 0)
            net_flow_formatted = format_large_number(net_flow)
            st.metric("Net Insider Flow", f"${net_flow_formatted}", "Buy - Sell")
        
        # Insider sentiment description
        sentiment_desc = insider_data.get('sentiment_description', 'No description available')
        
        if insider_score > 20:
            st.success(f"🟢 **Bullish Insider Activity**: {sentiment_desc}")
        elif insider_score < -20:
            st.error(f"🔴 **Bearish Insider Activity**: {sentiment_desc}")
        else:
            st.info(f"🟡 **Neutral Insider Activity**: {sentiment_desc}")
        
        # Recent insider activity table
        recent_activity = insider_data.get('recent_activity', [])
        if recent_activity:
            st.subheader("📋 Recent Insider Transactions")
            
            activity_data = []
            for transaction in recent_activity[:5]:  # Show top 5
                activity_data.append({
                    'Date': str(transaction.get('date', 'Unknown'))[:10],
                    'Insider': transaction.get('insider', 'Unknown'),
                    'Title': transaction.get('title', 'Unknown'),
                    'Transaction': transaction.get('transaction', 'Unknown'),
                    'Value': f"${format_large_number(abs(transaction.get('value', 0)))}",
                    'Score Impact': f"{transaction.get('score', 0):+.1f}"
                })
            
            df_insider = pd.DataFrame(activity_data)
            st.dataframe(df_insider, use_container_width=True, hide_index=True)
        
        # Analysis quality indicator
        quality = insider_data.get('analysis_quality', 'Unknown')
        st.caption(f"Analysis Quality: {quality} | Data Source: {insider_data.get('data_sources', 'yfinance')}")

def show_market_divergence_analysis(analysis_results, show_debug=False):
    """Display market divergence analysis section - NEW in v5.0"""
    if not st.session_state.show_divergence_analysis:
        return
        
    with st.expander("🌐 Market Divergence & Expected Moves Analysis", expanded=True):
        
        divergence_data = analysis_results.get('divergence_analysis', {})
        
        if 'error' in divergence_data:
            st.warning(f"⚠️ Divergence Analysis: {divergence_data['error']}")
            return
        
        # Overall divergence metrics
        overall_metrics = divergence_data.get('overall_metrics', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            overall_div_score = overall_metrics.get('overall_divergence_score', 0)
            st.metric("Overall Divergence", f"{overall_div_score:.2f}", "vs Market ETFs")
        
        with col2:
            market_position = overall_metrics.get('market_position', 'UNKNOWN')
            st.metric("Market Position", market_position)
        
        with col3:
            avg_rel_strength = overall_metrics.get('avg_relative_strength', 0)
            st.metric("Avg Relative Strength", f"{avg_rel_strength:+.1f}%", "21-day performance")
        
        with col4:
            benchmarks_analyzed = overall_metrics.get('benchmarks_analyzed', 0)
            st.metric("Benchmarks Analyzed", f"{benchmarks_analyzed}", "ETF Comparisons")
        
        # Technical score breakdown
        tech_score = divergence_data.get('symbol_technical_score', {})
        if tech_score:
            st.subheader("🔍 4-Component Technical Score")
            
            components = tech_score.get('components', {})
            composite = tech_score.get('composite', 0)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Position", f"{components.get('position', 0):.1f}/4", "EMA Relations")
            with col2:
                st.metric("Momentum", f"{components.get('momentum', 0):.1f}/4", "RSI & MACD")
            with col3:
                st.metric("Slope", f"{components.get('slope', 0):.1f}/4", "Trend Direction")
            with col4:
                st.metric("Volume", f"{components.get('volume', 0):.1f}/4", "Flow Analysis")
            with col5:
                st.metric("Composite", f"{composite:.1f}/16", "Total Score")
        
        # Expected moves analysis
        expected_moves = divergence_data.get('expected_moves', {})
        if expected_moves:
            st.subheader("📊 Enhanced Expected Moves")
            
            vol_metrics = expected_moves.get('volatility_metrics', {})
            vol_regime = vol_metrics.get('vol_regime', 'NORMAL_VOLATILITY')
            
            # Volatility regime indicator
            regime_colors = {
                'HIGH_VOLATILITY': '🔴',
                'ELEVATED_VOLATILITY': '🟡', 
                'NORMAL_VOLATILITY': '🟢',
                'LOW_VOLATILITY': '🔵'
            }
            
            st.info(f"{regime_colors.get(vol_regime, '⚪')} **Volatility Regime**: {vol_regime.replace('_', ' ').title()}")
            
            # Expected moves table
            moves_data = expected_moves.get('expected_moves', {})
            if moves_data:
                move_table_data = []
                for period, move_data in moves_data.items():
                    move_table_data.append({
                        'Period': period,
                        'Expected Move': f"±{move_data.get('expected_move_pct', 0):.1f}%",
                        'Upper Level': f"${move_data.get('upper_level', 0):.2f}",
                        'Lower Level': f"${move_data.get('lower_level', 0):.2f}",
                        'Min Move': f"${move_data.get('min_move', 0):.2f}",
                        'Max Move': f"${move_data.get('max_move', 0):.2f}"
                    })
                
                df_moves = pd.DataFrame(move_table_data)
                st.dataframe(df_moves, use_container_width=True, hide_index=True)

def show_tech_sentiment_analysis(analysis_results, show_debug=False):
    """Display tech sector sentiment analysis section - NEW in v5.0"""
    if not st.session_state.show_tech_sentiment:
        return
        
    with st.expander("📈 Tech Sector Sentiment Analysis (FNGD/FNGU)", expanded=True):
        
        tech_sentiment_data = analysis_results.get('tech_sentiment_analysis', {})
        
        if 'error' in tech_sentiment_data:
            st.warning(f"⚠️ Tech Sentiment Analysis: {tech_sentiment_data['error']}")
            return
        
        sentiment_analysis = tech_sentiment_data.get('sentiment_analysis', {})
        
        if 'error' in sentiment_analysis:
            st.warning(f"⚠️ Unable to calculate tech sentiment: {sentiment_analysis['error']}")
            return
        
        # Tech sentiment metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sentiment_score = sentiment_analysis.get('sentiment_score', 0)
            sentiment_class = sentiment_analysis.get('sentiment_classification', 'NEUTRAL')
            st.metric("Tech Sentiment Score", f"{sentiment_score:+.1f}", f"Classification: {sentiment_class}")
        
        with col2:
            market_regime = tech_sentiment_data.get('market_regime', 'UNKNOWN')
            st.metric("Market Regime", market_regime.replace('_', ' ').title())
        
        with col3:
            trading_signal = tech_sentiment_data.get('trading_signal', 'NEUTRAL')
            st.metric("Trading Signal", trading_signal.replace('_', ' ').title())
        
        with col4:
            confidence = sentiment_analysis.get('confidence_level', 'Low')
            st.metric("Confidence Level", confidence)
        
        # Sentiment description
        sentiment_desc = sentiment_analysis.get('sentiment_description', 'No description available')
        
        if sentiment_score > 25:
            st.success(f"🟢 **Bullish Tech Sentiment**: {sentiment_desc}")
        elif sentiment_score < -25:
            st.error(f"🔴 **Bearish Tech Sentiment**: {sentiment_desc}")
        else:
            st.info(f"🟡 **Neutral Tech Sentiment**: {sentiment_desc}")
        
        # Perfect setups detection
        perfect_setups = sentiment_analysis.get('perfect_setups', {})
        if perfect_setups:
            setup_desc = perfect_setups.get('setup_description', 'No setup detected')
            
            if perfect_setups.get('perfect_bull') or perfect_setups.get('perfect_bear'):
                st.success(f"🎯 **Perfect Setup Detected**: {setup_desc}")
            elif perfect_setups.get('strong_bull') or perfect_setups.get('strong_bear'):
                st.info(f"📊 **Strong Setup**: {setup_desc}")
            else:
                st.caption(f"Setup Status: {setup_desc}")
        
        # FNGD/FNGU analysis details
        fngd_analysis = sentiment_analysis.get('fngd_analysis', {})
        fngu_analysis = sentiment_analysis.get('fngu_analysis', {})
        
        if fngd_analysis and fngu_analysis:
            st.subheader("📋 FNGD vs FNGU Analysis")
            
            etf_comparison_data = [
                {
                    'ETF': 'FNGD (3x Bear)',
                    'Current Price': f"${fngd_analysis.get('current_price', 0):.2f}",
                    'vs 20EMA': f"{fngd_analysis.get('price_vs_short_pct', 0):+.1f}%",
                    'vs 50EMA': f"{fngd_analysis.get('price_vs_medium_pct', 0):+.1f}%",
                    'EMA Relationship': 'Above' if fngd_analysis.get('short_above_medium') else 'Below',
                    'Short Slope': f"{fngd_analysis.get('ema_short_slope', 0):+.2f}%"
                },
                {
                    'ETF': 'FNGU (3x Bull)',
                    'Current Price': f"${fngu_analysis.get('current_price', 0):.2f}",
                    'vs 20EMA': f"{fngu_analysis.get('price_vs_short_pct', 0):+.1f}%",
                    'vs 50EMA': f"{fngu_analysis.get('price_vs_medium_pct', 0):+.1f}%",
                    'EMA Relationship': 'Above' if fngu_analysis.get('short_above_medium') else 'Below',
                    'Short Slope': f"{fngu_analysis.get('ema_short_slope', 0):+.2f}%"
                }
            ]
            
            df_etf_comparison = pd.DataFrame(etf_comparison_data)
            st.dataframe(df_etf_comparison, use_container_width=True, hide_index=True)

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis section - ENHANCED in v5.0"""
    if not st.session_state.show_market_correlation:
        return
        
    with st.expander("🌐 Market Correlation & Breakout Analysis", expanded=True):
        
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
        
        # Enhanced Breakout/breakdown analysis - FIXED VERSION
        st.subheader("📊 Enhanced Breakout/Breakdown Analysis (FIXED v5.0)")
        breakout_data = calculate_enhanced_breakout_analysis(['SPY', 'QQQ', 'IWM'], show_debug=show_debug)
        
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
                
                # Individual ETF breakout details
                etf_breakout_data = []
                for symbol in ['SPY', 'QQQ', 'IWM']:
                    if symbol in breakout_data:
                        data = breakout_data[symbol]
                        etf_breakout_data.append({
                            'Symbol': symbol,
                            'Current Price': f"${data.get('current_price', 0):.2f}",
                            'Breakout %': f"{data.get('breakout_ratio', 0):.1f}%",
                            'Breakdown %': f"{data.get('breakdown_ratio', 0):.1f}%",
                            'Net Bias': f"{data.get('net_ratio', 0):+.1f}%",
                            'Signal': data.get('signal_strength', 'Unknown')
                        })
                
                if etf_breakout_data:
                    df_breakouts = pd.DataFrame(etf_breakout_data)
                    st.dataframe(df_breakouts, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Enhanced breakout analysis not available")

def show_options_analysis(analysis_results, show_debug=False):
    """Display enhanced options analysis section - ENHANCED in v5.0"""
    if not st.session_state.show_options_analysis:
        return
        
    with st.expander("🎯 Enhanced Options Trading Analysis with Sigma Levels", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        options_levels = enhanced_indicators.get('options_levels', [])
        sigma_analysis = analysis_results.get('sigma_analysis', {})
        
        # Show enhanced sigma levels if available
        if sigma_analysis:
            st.subheader("🎯 Enhanced Sigma Levels (Fibonacci + Volatility + Volume)")
            
            col1, col2, col3 = st.columns(3)
            
            for i, (risk_level, level_data) in enumerate(sigma_analysis.items()):
                if i < 3:  # Show first 3 risk levels
                    with [col1, col2, col3][i]:
                        st.write(f"**{risk_level.title()} Strategy:**")
                        
                        recommended = level_data.get('recommended', {})
                        st.metric(f"Put Strike", f"${recommended.get('put_strike', 0):.2f}")
                        st.metric(f"Call Strike", f"${recommended.get('call_strike', 0):.2f}")
                        st.metric(f"Probability of Touch", f"{recommended.get('probability_of_touch', 0):.1f}%")
                        st.metric(f"Expected Move", f"±{recommended.get('expected_move_pct', 0):.1f}%")
                        
                        target_pot = level_data.get('target_probability_of_touch', 0)
                        st.caption(f"Target PoT: {target_pot}%")
            
            # Component analysis
            if any('component_analysis' in data for data in sigma_analysis.values()):
                sample_analysis = next(data['component_analysis'] for data in sigma_analysis.values() if 'component_analysis' in data)
                
                st.subheader("🔍 Sigma Components Breakdown")
                component_data = [
                    ('Fibonacci Component', f"${sample_analysis.get('fibonacci_component', 0):.2f}", 'Based on 5-day rolling average'),
                    ('Volatility Component', f"${sample_analysis.get('volatility_component', 0):.2f}", 'Multi-factor volatility calculation'),
                    ('Volume Component', f"${sample_analysis.get('volume_component', 0):.2f}", 'Volume profile analysis'),
                    ('Base Sigma', f"${sample_analysis.get('base_sigma', 0):.2f}", 'Combined weighted sigma')
                ]
                
                component_df_data = []
                for name, value, description in component_data:
                    component_df_data.append({
                        'Component': name,
                        'Value': value,
                        'Description': description
                    })
                
                df_sigma_components = pd.DataFrame(component_df_data)
                st.dataframe(df_sigma_components, use_container_width=True, hide_index=True)
        
        # Show standard options levels
        if options_levels:
            st.subheader("💰 Standard Options Levels with Greeks")
            
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
                st.info("**Enhanced v5.0 Features:**\n"
                        "• **Sigma Levels**: Multi-factor calculations\n"
                        "• **Fibonacci Base**: 5-day rolling average\n"
                        "• **Volume Profile**: POC integration\n"
                        "• **Risk Levels**: Conservative/Moderate/Aggressive")
        else:
            st.warning("⚠️ Options analysis not available - insufficient data")

def show_confidence_intervals(analysis_results, show_debug=False):
    """Display confidence intervals section - SAME as v3.0"""
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

def perform_enhanced_analysis_v5(symbol, period, controls, show_debug=False):
    """
    Perform enhanced analysis for VWV v5.0 using all new modules
    """
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
        
        current_price = round(float(analysis_input['Close'].iloc[-1]), 2)
        current_date = analysis_input.index[-1].strftime('%Y-%m-%d')
        
        # Step 4: Calculate enhanced indicators using modular analysis
        daily_vwap = calculate_daily_vwap(analysis_input)
        fibonacci_emas = calculate_fibonacci_emas(analysis_input)
        point_of_control = calculate_point_of_control_enhanced(analysis_input)
        weekly_deviations = calculate_weekly_deviations(analysis_input)
        comprehensive_technicals = calculate_comprehensive_technicals(analysis_input)
        
        # Step 5: Calculate market correlations
        market_correlations = calculate_market_correlations_enhanced(analysis_input, symbol, show_debug=show_debug)
        
        # Step 6: NEW v5.0 - Calculate VWV Core Analysis
        @safe_calculation_wrapper
        def calculate_vwv_confluence_score_simple(data):
            """Simplified VWV Core calculation"""
            try:
                if len(data) < 50:
                    return 0, {'error': 'Insufficient data'}
                
                # Williams VIX Fix
                lookback = 22
                highest_close = data['Close'].rolling(window=lookback).max()
                wvf = ((highest_close - data['Low']) / highest_close) * 100
                current_wvf = wvf.iloc[-1]
                wvf_normalized = min(current_wvf / 20, 5.0)
                
                # MA Confluence
                current_price = data['Close'].iloc[-1]
                ma_20 = data['Close'].rolling(20).mean().iloc[-1]
                ma_50 = data['Close'].rolling(50).mean().iloc[-1]
                
                ma_score = 0
                if current_price > ma_20:
                    ma_score += 1
                if current_price > ma_50:
                    ma_score += 1
                if ma_20 > ma_50:
                    ma_score += 1
                
                # Volume Analysis
                current_volume = data['Volume'].iloc[-1]
                avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                volume_score = 1 if current_volume > avg_volume * 1.2 else 0
                
                # Momentum (RSI)
                close = data['Close']
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss.replace(0, np.inf)
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
                
                momentum_score = 2 if current_rsi < 30 else 1 if current_rsi < 40 else 0
                
                # Weighted score
                final_score = (wvf_normalized * 1.5 + ma_score * 1.2 + volume_score * 0.8 + momentum_score * 0.7) * 1.2
                
                if final_score >= 5.5:
                    signal_class = "VERY_STRONG"
                elif final_score >= 4.5:
                    signal_class = "STRONG"
                elif final_score >= 3.5:
                    signal_class = "GOOD"
                else:
                    signal_class = "WEAK"
                
                details = {
                    'wvf_raw': round(current_wvf, 2),
                    'wvf_normalized': round(wvf_normalized, 2),
                    'ma_confluence': round(ma_score, 2),
                    'volume_confluence': round(volume_score, 2),
                    'momentum_component': round(momentum_score, 2),
                    'final_score': round(final_score, 2),
                    'signal_classification': signal_class
                }
                
                return final_score, details
                
            except Exception as e:
                return 0, {'error': str(e)}
        
        vwv_score, vwv_details = calculate_vwv_confluence_score_simple(analysis_input)
        
        # Step 7: NEW v5.0 - Calculate Insider Analysis
        insider_analysis = calculate_insider_score(symbol, show_debug)
        
        # Step 8: NEW v5.0 - Calculate Market Divergence Analysis
        divergence_analysis = calculate_market_divergence_analysis(analysis_input, symbol, period, show_debug)
        
        # Step 9: NEW v5.0 - Calculate Tech Sentiment Analysis
        tech_sentiment_analysis = calculate_tech_sector_sentiment_analysis(period, show_debug)
        
        # Step 10: Calculate fundamental analysis (skip for ETFs)
        is_etf_symbol = is_etf(symbol)
        
        if is_etf_symbol:
            graham_score = {'score': 0, 'total_possible': 10, 'criteria': [], 'error': 'ETF - Fundamental analysis not applicable'}
            piotroski_score = {'score': 0, 'total_possible': 9, 'criteria': [], 'error': 'ETF - Fundamental analysis not applicable'}
        else:
            graham_score = calculate_graham_score(symbol, show_debug)
            piotroski_score = calculate_piotroski_score(symbol, show_debug)
        
        # Step 11: ENHANCED v5.0 - Calculate enhanced options levels with sigma
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
        
        # Enhanced options with market data for sigma calculations
        options_levels = calculate_options_levels_enhanced(
            current_price, volatility, 
            controls['options_dte'], 
            underlying_beta=underlying_beta,
            market_data=analysis_input
        )
        
        # Calculate enhanced sigma levels for different risk profiles
        sigma_analysis = {}
        for risk_level in ['conservative', 'moderate', 'aggressive']:
            sigma_levels = calculate_enhanced_sigma_levels(
                analysis_input, current_price, risk_level, 30
            )
            if sigma_levels:
                sigma_analysis[risk_level] = sigma_levels
        
        # Step 12: Calculate confidence intervals
        confidence_analysis = calculate_confidence_intervals(analysis_input)
        
        # Step 13: Build comprehensive analysis results
        analysis_results = {
            'symbol': symbol,
            'timestamp': current_date,
            'current_price': current_price,
            
            # Original indicators
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
            
            # NEW v5.0 analysis sections
            'vwv_analysis': {
                'final_score': vwv_score,
                'component_details': vwv_details,
                'signal_classification': vwv_details.get('signal_classification', 'WEAK')
            },
            'insider_analysis': insider_analysis,
            'divergence_analysis': divergence_analysis,
            'tech_sentiment_analysis': tech_sentiment_analysis,
            'sigma_analysis': sigma_analysis,
            
            'confidence_analysis': confidence_analysis,
            'system_status': 'VWV_v5.0_OPERATIONAL'
        }
        
        # Store results
        data_manager.store_analysis_results(symbol, analysis_results)
        
        return analysis_results
        
    except Exception as e:
        st.error(f"❌ Enhanced v5.0 analysis failed: {str(e)}")
        if show_debug:
            st.exception(e)
        return None

def main():
    """Main application function for VWV v5.0"""
    # Create header using modular component
    create_header()
    
    # Create enhanced sidebar and get controls
    controls = create_sidebar_controls()
    
    # Main logic flow
    if controls['analyze_button'] and controls['symbol']:
        # Add symbol to recently viewed
        add_to_recently_viewed(controls['symbol'])
        
        st.write("## 📊 VWV Trading Analysis v5.0")
        
        with st.spinner(f"Analyzing {controls['symbol']} with enhanced v5.0 capabilities..."):
            
            # Perform enhanced analysis using all v5.0 modules
            analysis_results = perform_enhanced_analysis_v5(
                controls['symbol'], 
                controls['period'], 
                controls,
                controls['show_debug']
            )
            
            if analysis_results:
                # Show all analysis sections using enhanced v5.0 functions
                show_vwv_core_analysis(analysis_results, controls['show_debug'])  # NEW VWV Core section
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                show_insider_analysis(analysis_results, controls['show_debug'])
                show_market_divergence_analysis(analysis_results, controls['show_debug'])
                show_tech_sentiment_analysis(analysis_results, controls['show_debug'])
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                show_options_analysis(analysis_results, controls['show_debug'])
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
                # Debug information
                if controls['show_debug']:
                    with st.expander("🐛 Debug Information", expanded=False):
                        st.write("### v5.0 Analysis Results Structure")
                        st.json(analysis_results, expanded=False)
                        
                        st.write("### Data Manager Summary")
                        data_manager = get_data_manager()
                        summary = data_manager.get_data_summary()
                        st.json(summary)
    
    else:
        # Enhanced welcome message for v5.0
        st.write("## 🚀 VWV Professional Trading System v5.0 - Complete Enhanced Architecture")
        st.write("**Major v5.0 Enhancements:** Williams VIX Fix Core, Insider Analysis, Market Divergence, Tech Sentiment, Enhanced Sigma Levels")
        
        with st.expander("🆕 What's New in v5.0", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 🎯 **NEW Core Features**")
                st.write("✅ **Williams VIX Fix** - 6-component confluence system")
                st.write("✅ **Insider Analysis** - Buying/selling sentiment tracking") 
                st.write("✅ **Market Divergence** - Multi-ETF relative strength")
                st.write("✅ **Tech Sentiment** - FNGD/FNGU leveraged ETF analysis")
                st.write("✅ **Enhanced Sigma Levels** - Fibonacci + Volatility + Volume")
                st.write("✅ **Fixed Breakouts** - Multi-factor breakout detection")
                
            with col2:
                st.write("### 🔧 **Enhanced Analysis**")
                st.write("• **VWV Risk Management** - Dynamic stops & targets")
                st.write("• **Expected Moves** - Volatility regime detection")
                st.write("• **Perfect Setups** - FNGD/FNGU EMA relationships")
                st.write("• **Component Scoring** - 4-factor technical analysis")
                st.write("• **Default Period** - Changed to 3mo for better analysis")
                st.write("• **Probability of Touch** - Enhanced options modeling")
        
        with st.expander("📊 All Active Analysis Sections", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 🎯 **Core VWV System**")
                st.write("• **VWV Core Signals** - Williams VIX Fix confluence")
                st.write("• **Individual Technical** - Enhanced composite scoring")
                st.write("• **Fundamental Analysis** - Graham & Piotroski scores")
                st.write("• **Market Correlation** - ETF relationship analysis")
                st.write("• **Enhanced Breakouts** - Multi-factor detection")
                
            with col2:
                st.write("### 🆕 **New v5.0 Sections**")
                st.write("• **Insider Analysis** - Professional insider tracking")
                st.write("• **Market Divergence** - Expected moves & relative strength")
                st.write("• **Tech Sentiment** - Sector-specific FNGD/FNGU analysis")
                st.write("• **Enhanced Options** - Sigma levels with Fibonacci base")
                st.write("• **Statistical Intervals** - Confidence level calculations")
        
        # Show current market status
        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")
        
        # Enhanced quick start guide
        with st.expander("🚀 Quick Start Guide for v5.0", expanded=True):
            st.write("1. **Enter a symbol** in the sidebar (e.g., AAPL, SPY, QQQ, TSLA)")
            st.write("2. **Select period** - Default is now 3mo for optimal analysis")
            st.write("3. **Click 'Analyze Symbol'** to run complete v5.0 analysis")
            st.write("4. **Explore NEW sections:** VWV Core, Insider, Divergence, Tech Sentiment")
            st.write("5. **Toggle sections** on/off in Analysis Sections panel")
            st.write("6. **Review sigma levels** for enhanced options strategies")

    # Enhanced footer
    st.markdown("---")
    st.write("### 📊 VWV Trading System v5.0 Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Version:** VWV Professional v5.0 - Enhanced")
        st.write(f"**Core System:** Williams VIX Fix (32+ years proven)")
    with col2:
        st.write(f"**Status:** ✅ All Enhanced Modules Active")
        st.write(f"**New Sections:** VWV Core, Insider, Divergence, Tech Sentiment")
    with col3:
        st.write(f"**Architecture:** Complete modular with v5.0 enhancements")
        st.write(f"**Options:** Enhanced with Sigma Levels & Fibonacci integration")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ VWV v5.0 Application Error: {str(e)}")
        st.write("Please refresh the page and try again.")
        
        if st.checkbox("Show Error Details"):
            st.exception(e)
