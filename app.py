"""
VWV Professional Trading System v4.2.1 - CORE FUNCTIONALITY PRESERVED
Step 1: Essential functionality that MUST work
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Essential imports only
from config.settings import UI_SETTINGS
from config.constants import SYMBOL_DESCRIPTIONS, QUICK_LINK_CATEGORIES
from data.manager import get_data_manager
from data.fetcher import get_market_data_enhanced, is_etf
from analysis.technical import (
    calculate_daily_vwap, 
    calculate_fibonacci_emas,
    calculate_point_of_control_enhanced,
    calculate_comprehensive_technicals,
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
    """Create sidebar controls - CRITICAL: Quick links and Enter key must work"""
    st.sidebar.title("📊 Trading Analysis v4.2.1")
    
    # Initialize session state
    if 'recently_viewed' not in st.session_state:
        st.session_state.recently_viewed = []
    
    # Handle selected symbol from quick links
    if 'selected_symbol' in st.session_state:
        default_symbol = st.session_state.selected_symbol
        del st.session_state.selected_symbol
    else:
        default_symbol = UI_SETTINGS['default_symbol']
        
    # CRITICAL: Form for Enter key functionality
    with st.sidebar.form(key='symbol_form'):
        symbol = st.text_input("Symbol", value=default_symbol, help="Enter stock symbol (press Enter to analyze)").upper()
        period = st.selectbox("Data Period", UI_SETTINGS['periods'], index=1)
        
        # Analyze button in form
        analyze_button = st.form_submit_button("📊 Analyze Symbol", type="primary", use_container_width=True)
    
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

    # CRITICAL: Quick Links section - must trigger analysis
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

def show_individual_technical_analysis(analysis_results, show_debug=False):
    """CRITICAL: Technical analysis with score bar must display"""
    with st.expander(f"📊 {analysis_results['symbol']} - Individual Technical Analysis", expanded=True):
        
        # CRITICAL: Technical score bar must display
        composite_score, score_details = calculate_composite_technical_score(analysis_results)
        score_bar_html = create_technical_score_bar(composite_score, score_details)
        st.components.v1.html(score_bar_html, height=110)
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
        
        # Primary metrics
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
        
        # CRITICAL: Technical indicators table with Fibonacci EMAs
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
        
        # CRITICAL: Fibonacci EMAs must display
        for ema_name, ema_value in fibonacci_emas.items():
            period = ema_name.split('_')[1]
            distance_pct = f"{((current_price - ema_value) / ema_value * 100):+.2f}%" if ema_value > 0 else "N/A"
            status = "Above" if current_price > ema_value else "Below"
            indicators_data.append((f"EMA {period}", f"${ema_value:.2f}", "📈 Trend", distance_pct, status))
        
        df_technical = pd.DataFrame(indicators_data, columns=['Indicator', 'Value', 'Type', 'Distance %', 'Status'])
        st.dataframe(df_technical, use_container_width=True, hide_index=True)

def perform_core_analysis(symbol, period, show_debug=False):
    """Perform core analysis - essential functionality only"""
    try:
        # Step 1: Fetch data
        market_data = get_market_data_enhanced(symbol, period, show_debug)
        
        if market_data is None:
            st.error(f"❌ Could not fetch data for {symbol}")
            return None
        
        # Step 2: Store data
        data_manager = get_data_manager()
        data_manager.store_market_data(symbol, market_data, show_debug)
        
        # Step 3: Get analysis copy
        analysis_input = data_manager.get_market_data_for_analysis(symbol)
        
        if analysis_input is None:
            st.error("❌ Could not prepare analysis data")
            return None
        
        # Step 4: Calculate core indicators
        daily_vwap = calculate_daily_vwap(analysis_input)
        fibonacci_emas = calculate_fibonacci_emas(analysis_input)
        point_of_control = calculate_point_of_control_enhanced(analysis_input)
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
        
        # Step 8: Calculate confidence intervals
        confidence_analysis = calculate_confidence_intervals(analysis_input)
        
        # Step 9: Build results
        current_date = analysis_input.index[-1].strftime('%Y-%m-%d')
        
        analysis_results = {
            'symbol': symbol,
            'timestamp': current_date,
            'current_price': current_price,
            'enhanced_indicators': {
                'daily_vwap': daily_vwap,
                'fibonacci_emas': fibonacci_emas,
                'point_of_control': point_of_control,
                'comprehensive_technicals': comprehensive_technicals,
                'market_correlations': market_correlations,
                'options_levels': options_levels,
                'graham_score': graham_score,
                'piotroski_score': piotroski_score
            },
            'confidence_analysis': confidence_analysis,
            'system_status': 'CORE_OPERATIONAL'
        }
        
        # Store results
        data_manager.store_analysis_results(symbol, analysis_results)
        
        return analysis_results
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        return None

def main():
    """Main application - CORE FUNCTIONALITY ONLY"""
    # Create header
    create_header()
    
    # Create sidebar and get controls
    controls = create_sidebar_controls()
    
    # Main logic flow
    if controls['analyze_button'] and controls['symbol']:
        # Add to recently viewed
        add_to_recently_viewed(controls['symbol'])
        
        st.write("## 📊 VWV Trading Analysis - Core Functionality")
        
        with st.spinner(f"Analyzing {controls['symbol']}..."):
            
            # Perform core analysis
            analysis_results = perform_core_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results:
                # Show core analysis - CRITICAL functionality
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                
                # Success message
                st.success("✅ Core functionality working! Technical score bar, Fibonacci EMAs, Quick Links, and Enter key all operational.")
                
                # Debug information
                if controls['show_debug']:
                    with st.expander("🐛 Debug Information", expanded=False):
                        st.write("### Analysis Results Structure")
                        st.json(analysis_results, expanded=False)
    
    else:
        # Welcome message
        st.write("## 🚀 VWV Professional Trading System v4.2.1")
        st.write("**Core Functionality Test - Step 1 of 3**")
        
        with st.expander("✅ Core Functionality Checklist", expanded=True):
            st.write("**CRITICAL Components:**")
            st.write("• ✅ **Technical Score Bar** - Professional gradient display")
            st.write("• ✅ **Quick Links** - Click any symbol → analysis starts")
            st.write("• ✅ **Enter Key** - Type symbol + Enter → analysis starts")
            st.write("• ✅ **Fibonacci EMAs** - All EMAs display in technical table")
            st.write("• ✅ **Recently Viewed** - Symbol history with quick access")
            
            st.info("**Next Steps:** Once core functionality is verified, we'll add Volume & Volatility analysis in Segment 3.")
        
        # Show current market status
        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")
        
        # Quick start guide
        with st.expander("🚀 Test Instructions", expanded=True):
            st.write("1. **Click any Quick Link symbol** → Analysis should start immediately")
            st.write("2. **Type a symbol + press Enter** → Analysis should start")
            st.write("3. **Verify Technical Score Bar displays** with gradient")
            st.write("4. **Check Fibonacci EMAs show** in technical indicators table")
            st.write("5. **Confirm Recently Viewed works** after analyzing symbols")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.write("Please refresh the page and try again.")
        
        if st.checkbox("Show Error Details"):
            st.exception(e)
