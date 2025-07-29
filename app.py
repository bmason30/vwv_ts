"""
VWV Professional Trading System - Complete Modular Version
Main application with Technical Score Screener
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time

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

# Screener Configuration
SCREENER_CONFIG = {
    'refresh_minutes': 15,  # Recommended: 15 minutes for good balance
    'extreme_low_threshold': 25,
    'extreme_high_threshold': 75,
    'max_symbols_per_scan': 20,  # Limit to avoid timeouts
    'timeout_per_symbol': 10  # seconds
}

@st.cache_data(ttl=SCREENER_CONFIG['refresh_minutes'] * 60, show_spinner=False)
def scan_extreme_technical_scores():
    """Scan quick links symbols for extreme technical scores with caching"""
    start_time = time.time()
    extreme_scores = {'bearish': [], 'bullish': []}
    scanned_count = 0
    
    # Get all symbols from quick links (limit to prevent timeouts)
    all_symbols = []
    for category, symbols in QUICK_LINK_CATEGORIES.items():
        all_symbols.extend(symbols)
    
    # Remove duplicates and limit
    unique_symbols = list(set(all_symbols))[:SCREENER_CONFIG['max_symbols_per_scan']]
    
    for symbol in unique_symbols:
        if time.time() - start_time > 120:  # 2-minute total timeout
            break
            
        try:
            # Quick analysis for screening (shorter period for speed)
            market_data = get_market_data_enhanced(symbol, period='3mo', show_debug=False)
            
            if market_data is None or len(market_data) < 50:
                continue
                
            # Calculate minimal indicators needed for composite score
            daily_vwap = calculate_daily_vwap(market_data)
            fibonacci_emas = calculate_fibonacci_emas(market_data)
            point_of_control = calculate_point_of_control_enhanced(market_data)
            comprehensive_technicals = calculate_comprehensive_technicals(market_data)
            
            current_price = round(float(market_data['Close'].iloc[-1]), 2)
            
            # Build minimal analysis for composite score
            analysis_for_score = {
                'symbol': symbol,
                'current_price': current_price,
                'enhanced_indicators': {
                    'daily_vwap': daily_vwap,
                    'fibonacci_emas': fibonacci_emas,
                    'point_of_control': point_of_control,
                    'comprehensive_technicals': comprehensive_technicals
                }
            }
            
            # Calculate composite score
            composite_score, score_details = calculate_composite_technical_score(analysis_for_score)
            
            # Check for extreme scores
            if composite_score <= SCREENER_CONFIG['extreme_low_threshold']:
                extreme_scores['bearish'].append({
                    'symbol': symbol,
                    'score': composite_score,
                    'price': current_price,
                    'description': SYMBOL_DESCRIPTIONS.get(symbol, f"{symbol} - Financial Symbol")[:50] + "..."
                })
            elif composite_score >= SCREENER_CONFIG['extreme_high_threshold']:
                extreme_scores['bullish'].append({
                    'symbol': symbol,
                    'score': composite_score,
                    'price': current_price,
                    'description': SYMBOL_DESCRIPTIONS.get(symbol, f"{symbol} - Financial Symbol")[:50] + "..."
                })
                
            scanned_count += 1
            
        except Exception as e:
            continue  # Skip problematic symbols
    
    # Sort by score (most extreme first)
    extreme_scores['bearish'].sort(key=lambda x: x['score'])
    extreme_scores['bullish'].sort(key=lambda x: x['score'], reverse=True)
    
    scan_time = time.time() - start_time
    
    return {
        'results': extreme_scores,
        'scan_time': round(scan_time, 1),
        'scanned_count': scanned_count,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    }

def show_technical_screener():
    """Display the technical score screener section"""
    st.write("### 🎯 Technical Score Screener")
    st.write(f"**Extreme Scores**: < {SCREENER_CONFIG['extreme_low_threshold']} (Very Bearish) or > {SCREENER_CONFIG['extreme_high_threshold']} (Very Bullish) | Auto-refresh: {SCREENER_CONFIG['refresh_minutes']} min")
    
    # Add manual refresh button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔄 Refresh Now", help="Manually refresh screener results"):
            st.cache_data.clear()
            st.rerun()
    
    # Get cached results
    with st.spinner("Scanning symbols for extreme scores..."):
        screen_results = scan_extreme_technical_scores()
    
    with col2:
        st.write(f"**Last Scan:** {screen_results['timestamp']}")
    with col3:
        st.write(f"**Scanned:** {screen_results['scanned_count']} symbols in {screen_results['scan_time']}s")
    
    results = screen_results['results']
    
    # Display results in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### 🔴 Very Bearish (≤25)")
        if results['bearish']:
            bearish_data = []
            for item in results['bearish']:
                bearish_data.append({
                    'Symbol': item['symbol'],
                    'Score': f"{item['score']:.1f}",
                    'Price': f"${item['price']:.2f}",
                    'Description': item['description']
                })
            
            df_bearish = pd.DataFrame(bearish_data)
            st.dataframe(df_bearish, use_container_width=True, hide_index=True)
            
            # Quick analysis buttons
            st.write("**Quick Analysis:**")
            cols = st.columns(min(3, len(results['bearish'])))
            for idx, item in enumerate(results['bearish'][:3]):
                with cols[idx]:
                    if st.button(f"📊 {item['symbol']}", key=f"bearish_{item['symbol']}", use_container_width=True):
                        st.session_state.selected_symbol = item['symbol']
                        st.rerun()
        else:
            st.info("No very bearish signals found")
    
    with col2:
        st.write("#### 🟢 Very Bullish (≥75)")
        if results['bullish']:
            bullish_data = []
            for item in results['bullish']:
                bullish_data.append({
                    'Symbol': item['symbol'],
                    'Score': f"{item['score']:.1f}",
                    'Price': f"${item['price']:.2f}",
                    'Description': item['description']
                })
            
            df_bullish = pd.DataFrame(bullish_data)
            st.dataframe(df_bullish, use_container_width=True, hide_index=True)
            
            # Quick analysis buttons
            st.write("**Quick Analysis:**")
            cols = st.columns(min(3, len(results['bullish'])))
            for idx, item in enumerate(results['bullish'][:3]):
                with cols[idx]:
                    if st.button(f"📊 {item['symbol']}", key=f"bullish_{item['symbol']}", use_container_width=True):
                        st.session_state.selected_symbol = item['symbol']
                        st.rerun()
        else:
            st.info("No very bullish signals found")
    
    # Summary stats
    total_extreme = len(results['bearish']) + len(results['bullish'])
    if total_extreme > 0:
        st.write(f"**Summary:** Found {total_extreme} symbols with extreme scores ({len(results['bearish'])} bearish, {len(results['bullish'])} bullish)")
        
        # Show configuration
        with st.expander("⚙️ Screener Configuration", expanded=False):
            st.write(f"• **Refresh Frequency:** {SCREENER_CONFIG['refresh_minutes']} minutes")
            st.write(f"• **Bearish Threshold:** ≤ {SCREENER_CONFIG['extreme_low_threshold']}")
            st.write(f"• **Bullish Threshold:** ≥ {SCREENER_CONFIG['extreme_high_threshold']}")
            st.write(f"• **Max Symbols per Scan:** {SCREENER_CONFIG['max_symbols_per_scan']}")
            st.write(f"• **Data Period:** 3 months (optimized for speed)")

def initialize_session_state():
    """Initialize session state variables"""
    if 'recently_viewed' not in st.session_state:
        st.session_state.recently_viewed = []
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
    if 'current_symbol' not in st.session_state:
        st.session_state.current_symbol = UI_SETTINGS['default_symbol']

def create_sidebar_controls():
    """Create sidebar controls and return analysis parameters"""
    st.sidebar.title("📊 Trading Analysis")
    
    # Initialize session state
    initialize_session_state()
    
    # Better symbol handling - check for quick link selection first
    if 'selected_symbol' in st.session_state:
        st.session_state.current_symbol = st.session_state.selected_symbol
        del st.session_state.selected_symbol  # Clear the trigger
        
    symbol = st.sidebar.text_input(
        "Symbol", 
        value=st.session_state.current_symbol,
        key="symbol_input",
        help="Enter stock symbol"
    ).upper()
    
    # Update current_symbol when text input changes
    if symbol != st.session_state.current_symbol:
        st.session_state.current_symbol = symbol
    
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
        'symbol': st.session_state.current_symbol,
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
        
        # Use components directly instead of markdown for better rendering
        st.components.v1.html(score_bar_html, height=200)
        
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
    
    # Show current symbol being analyzed
    if controls['symbol']:
        st.write(f"## 📊 VWV Trading Analysis - {controls['symbol']}")
    
    # Main logic flow
    if controls['analyze_button'] and controls['symbol']:
        # Add symbol to recently viewed
        add_to_recently_viewed(controls['symbol'])
        
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
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                show_options_analysis(analysis_results, controls['show_debug'])
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
                # NEW: Technical Score Screener - placed above debug info
                st.markdown("---")
                show_technical_screener()
                
                # Debug information
                if controls['show_debug']:
                    st.markdown("---")
                    with st.expander("🐛 Debug Information", expanded=False):
                        st.write("### Analysis Results Structure")
                        st.json(analysis_results, expanded=False)
                        
                        st.write("### Data Manager Summary")
                        data_manager = get_data_manager()
                        summary = data_manager.get_data_summary()
                        st.json(summary)
                        
                        st.write("### Current Session State")
                        st.json({
                            'current_symbol': st.session_state.get('current_symbol', 'Not Set'),
                            'recently_viewed': st.session_state.get('recently_viewed', []),
                            'selected_symbol': st.session_state.get('selected_symbol', 'Not Set')
                        })
    
    else:
        # Welcome message
        st.write("## 🚀 VWV Professional Trading System - Complete Modular Architecture")
        st.write("**All modules active:** Technical, Fundamental, Market, Options, UI Components, **Technical Screener**")
        
        # NEW: Always show screener on home page
        st.markdown("---")
        show_technical_screener()
        st.markdown("---")
        
        with st.expander("🏗️ Modular Architecture Overview", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📁 **Active Modules**")
                st.write("✅ **`config/`** - Settings, constants, parameters")
                st.write("✅ **`data/`** - Fetching, validation, management")
                st.write("✅ **`analysis/`** - Technical, fundamental, market, options")
                st.write("✅ **`ui/`** - Components, headers, score bars")
                st.write("✅ **`utils/`** - Helpers, decorators, formatters")
                
            with col2:
                st.write("### 🎯 **All Sections Working**")
                st.write("• **Individual Technical Analysis**")
                st.write("• **Fundamental Analysis** (Graham & Piotroski)")
                st.write("• **Market Correlation & Breakouts**")
                st.write("• **Options Analysis with Greeks**")
                st.write("• **Statistical Confidence Intervals**")
                st.write("• **🆕 Technical Score Screener**")
        
        # Show current market status
        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=False):
            st.write("1. **Enter a symbol** in the sidebar (e.g., AAPL, SPY, QQQ)")
            st.write("2. **Click 'Analyze Symbol'** to run complete analysis")
            st.write("3. **View all sections:** Technical, Fundamental, Market, Options")
            st.write("4. **Toggle sections** on/off in Analysis Sections panel")
            st.write("5. **🆕 Use the screener** to find extreme technical scores automatically")

    # Footer
    st.markdown("---")
    st.write("### 📊 System Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Version:** VWV Professional v3.2 - Screener Added")
        st.write(f"**Architecture:** Full Modular + Screening")
    with col2:
        st.write(f"**Status:** ✅ All Modules + Screener Active")
        st.write(f"**Current Symbol:** {st.session_state.get('current_symbol', 'SPY')}")
    with col3:
        st.write(f"**Screener:** Auto-refresh every {SCREENER_CONFIG['refresh_minutes']} min")
        st.write(f"**Thresholds:** ≤{SCREENER_CONFIG['extreme_low_threshold']} | ≥{SCREENER_CONFIG['extreme_high_threshold']}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.write("Please refresh the page and try again.")
        
        if st.checkbox("Show Error Details"):
            st.exception(e)
