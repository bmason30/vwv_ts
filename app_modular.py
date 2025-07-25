"""
VWV Professional Trading System - Modular Version
Main application entry point with extracted modules
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

# Custom CSS (extracted to ui/styling.py in full implementation)
st.markdown("""
<style>
    .main-header {
        position: relative;
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        overflow: hidden;
        min-height: 200px;
        background: linear-gradient(to bottom, #1a4d3a 0%, #2d6b4f 40%, #1e5540 100%);
    }
    
    .header-content {
        position: relative;
        z-index: 3;
        background: rgba(0,0,0,0.2);
        padding: 1.5rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .header-content h1 {
        font-size: 2.8rem;
        margin-bottom: 1rem;
        color: #ffffff;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .header-content p {
        color: #f0f8f0;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.7);
        margin: 0.5rem 0;
        font-size: 1.1rem;
        font-weight: 400;
    }
</style>
""", unsafe_allow_html=True)

def create_technical_score_bar(score, details=None):
    """Create professional gradient bar for technical score"""
    
    # Determine interpretation and color
    if score >= 80:
        interpretation = "Very Bullish"
        primary_color = "#00A86B"  # Jade green
    elif score >= 65:
        interpretation = "Bullish" 
        primary_color = "#32CD32"  # Lime green
    elif score >= 55:
        interpretation = "Slightly Bullish"
        primary_color = "#9ACD32"  # Yellow green
    elif score >= 45:
        interpretation = "Neutral"
        primary_color = "#FFD700"  # Gold
    elif score >= 35:
        interpretation = "Slightly Bearish"
        primary_color = "#FF8C00"  # Dark orange
    elif score >= 20:
        interpretation = "Bearish"
        primary_color = "#FF4500"  # Orange red
    else:
        interpretation = "Very Bearish"
        primary_color = "#DC143C"  # Crimson
    
    # Create professional gradient bar HTML
    html = f"""
    <div style="margin: 1.5rem 0; padding: 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                border-radius: 12px; border: 1px solid #dee2e6; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div>
                <span style="font-weight: 700; color: #495057; font-size: 1.2em;">Technical Composite Score</span>
                <div style="font-size: 0.9em; color: #6c757d; margin-top: 0.2rem;">
                    Aggregated signal from all technical indicators
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-weight: 700; color: {primary_color}; font-size: 2em;">{score}</div>
                <div style="font-size: 0.9em; color: {primary_color}; font-weight: 600;">{interpretation}</div>
            </div>
        </div>
        
        <div style="position: relative; width: 100%; height: 24px; background: linear-gradient(to right, 
                    #DC143C 0%, #FF4500 15%, #FF8C00 30%, #FFD700 50%, 
                    #9ACD32 70%, #32CD32 85%, #00A86B 100%); 
                    border-radius: 12px; border: 1px solid #ced4da; overflow: hidden;">
            
            <!-- Score indicator -->
            <div style="position: absolute; left: {score}%; top: 50%; transform: translate(-50%, -50%); 
                        width: 6px; height: 30px; background: white; border: 2px solid #343a40; 
                        border-radius: 3px; box-shadow: 0 2px 4px rgba(0,0,0,0.3); z-index: 10;">
            </div>
            
            <!-- Progress fill -->
            <div style="position: absolute; left: 0; top: 0; height: 100%; width: {score}%; 
                        background: linear-gradient(to right, transparent 0%, {primary_color} 100%); 
                        opacity: 0.3; border-radius: 12px;">
            </div>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.75em; color: #6c757d;">
            <span style="font-weight: 600;">Very Bearish</span>
            <span style="font-weight: 600;">Neutral</span>
            <span style="font-weight: 600;">Very Bullish</span>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 0.2rem; font-size: 0.7em; color: #adb5bd;">
            <span>1</span>
            <span>25</span>
            <span>50</span>
            <span>75</span>
            <span>100</span>
        </div>
    </div>
    """
    
    return html

def create_header():
    """Create the main header"""
    st.markdown("""
    <div class="main-header">
        <div class="header-content">
            <h1>VWV Professional Trading System</h1>
            <p>Advanced market analysis with enhanced technical indicators</p>
            <p><em>Modular Architecture: Phase 1 Implementation</em></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar_controls():
    """Create sidebar controls and return analysis parameters"""
    st.sidebar.title("📊 Trading Analysis")
    
    # Initialize session state
    if 'recently_viewed' not in st.session_state:
        st.session_state.recently_viewed = []
    if 'show_technical_analysis' not in st.session_state:
        st.session_state.show_technical_analysis = True
    
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
        
        st.session_state.show_technical_analysis = st.checkbox(
            "Technical Analysis", 
            value=st.session_state.show_technical_analysis,
            key="toggle_technical"
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
        
        # COMPOSITE TECHNICAL SCORE
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
        
        # Comprehensive Technical Analysis Table
        st.subheader("📋 Comprehensive Technical Indicators")
        
        current_price = analysis_results['current_price']
        daily_vwap = enhanced_indicators.get('daily_vwap', 0)
        point_of_control = enhanced_indicators.get('point_of_control', 0)

        def determine_signal(indicator_name, current_price, indicator_value, distance_pct_str):
            """Determine if indicator is Bullish, Neutral, or Bearish"""
            try:
                if distance_pct_str == "N/A" or indicator_value == 0:
                    return "Neutral"
                
                if "%" in distance_pct_str:
                    distance_pct = float(distance_pct_str.replace("+", "").replace("%", ""))
                else:
                    distance_pct = 0
                
                if "Current Price" in indicator_name:
                    return "Neutral"
                elif "VWAP" in indicator_name or "Point of Control" in indicator_name:
                    return "Bullish" if distance_pct > 0 else "Bearish"
                elif "Prev Week High" in indicator_name:
                    if distance_pct > 0:
                        return "Bullish"
                    elif distance_pct > -2:
                        return "Neutral"
                    else:
                        return "Bearish"
                elif "Prev Week Low" in indicator_name:
                    if distance_pct > 5:
                        return "Bullish"
                    elif distance_pct > 0:
                        return "Neutral"
                    else:
                        return "Bearish"
                elif "EMA" in indicator_name:
                    if distance_pct > 1:
                        return "Bullish"
                    elif distance_pct > -1:
                        return "Neutral"
                    else:
                        return "Bearish"
                else:
                    return "Neutral"
                    
            except:
                return "Neutral"

        # Build comprehensive indicators table
        indicators_data = []
        
        # Current Price
        indicators_data.append(("Current Price", f"${current_price:.2f}", "📍 Reference", "0.0%", "Current", 
                                 determine_signal("Current Price", current_price, current_price, "0.0%")))
        
        # Daily VWAP
        vwap_distance = f"{((current_price - daily_vwap) / daily_vwap * 100):+.2f}%" if daily_vwap > 0 else "N/A"
        vwap_status = "Above" if current_price > daily_vwap else "Below"
        indicators_data.append(("Daily VWAP", f"${daily_vwap:.2f}", "📊 Volume Weighted", vwap_distance, vwap_status,
                                 determine_signal("Daily VWAP", current_price, daily_vwap, vwap_distance)))
        
        # Point of Control
        poc_distance = f"{((current_price - point_of_control) / point_of_control * 100):+.2f}%" if point_of_control > 0 else "N/A"
        poc_status = "Above" if current_price > point_of_control else "Below"
        indicators_data.append(("Point of Control", f"${point_of_control:.2f}", "📊 Volume Profile", poc_distance, poc_status,
                                 determine_signal("Point of Control", current_price, point_of_control, poc_distance)))
        
        # Previous Week High/Low
        prev_high = comprehensive_technicals.get('prev_week_high', 0)
        high_distance = f"{((current_price - prev_high) / prev_high * 100):+.2f}%" if prev_high > 0 else "N/A"
        high_status = "Above" if current_price > prev_high else "Below"
        indicators_data.append(("Prev Week High", f"${prev_high:.2f}", "📈 Resistance", high_distance, high_status,
                                 determine_signal("Prev Week High", current_price, prev_high, high_distance)))
        
        prev_low = comprehensive_technicals.get('prev_week_low', 0)
        low_distance = f"{((current_price - prev_low) / prev_low * 100):+.2f}%" if prev_low > 0 else "N/A"
        low_status = "Above" if current_price > prev_low else "Below"
        indicators_data.append(("Prev Week Low", f"${prev_low:.2f}", "📉 Support", low_distance, low_status,
                                 determine_signal("Prev Week Low", current_price, prev_low, low_distance)))
            
        # Add Fibonacci EMAs
        for ema_name, ema_value in fibonacci_emas.items():
            period = ema_name.split('_')[1]
            distance_pct = f"{((current_price - ema_value) / ema_value * 100):+.2f}%" if ema_value > 0 else "N/A"
            status = "Above" if current_price > ema_value else "Below"
            signal = determine_signal(f"EMA {period}", current_price, ema_value, distance_pct)
            indicators_data.append((f"EMA {period}", f"${ema_value:.2f}", "📈 Trend", distance_pct, status, signal))
        
        # Convert to DataFrame with signal emojis
        for i, row in enumerate(indicators_data):
            if len(row) == 6:
                signal = row[5]
                if signal == 'Bullish':
                    indicators_data[i] = row[:5] + ('🟢 Bullish',)
                elif signal == 'Bearish':
                    indicators_data[i] = row[:5] + ('🔴 Bearish',)
                else:
                    indicators_data[i] = row[:5] + ('🟡 Neutral',)
        
        df_technical = pd.DataFrame(indicators_data, columns=['Indicator', 'Value', 'Type', 'Distance %', 'Status', 'Signal'])
        st.dataframe(df_technical, use_container_width=True, hide_index=True)
            
        # Oscillators and Momentum
        st.subheader("📈 Momentum & Oscillator Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            rsi = comprehensive_technicals.get('rsi_14', 50)
            rsi_status = "🔴 Overbought" if rsi > 70 else "🟢 Oversold" if rsi < 30 else "⚪ Neutral"
            st.metric("RSI (14)", f"{rsi:.1f}", rsi_status)
        
        with col2:
            mfi = comprehensive_technicals.get('mfi_14', 50)
            mfi_status = "🔴 Overbought" if mfi > 80 else "🟢 Oversold" if mfi < 20 else "⚪ Neutral"
            st.metric("MFI (14)", f"{mfi:.1f}", mfi_status)
        
        with col3:
            williams_r = comprehensive_technicals.get('williams_r', -50)
            wr_status = "🔴 Overbought" if williams_r > -20 else "🟢 Oversold" if williams_r < -80 else "⚪ Neutral"
            st.metric("Williams %R", f"{williams_r:.1f}", wr_status)
        
        with col4:
            stoch_data = comprehensive_technicals.get('stochastic', {})
            stoch_k = stoch_data.get('k', 50)
            stoch_status = "🔴 Overbought" if stoch_k > 80 else "🟢 Oversold" if stoch_k < 20 else "⚪ Neutral"
            st.metric("Stochastic %K", f"{stoch_k:.1f}", stoch_status)

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
        
        # Step 5: Build analysis results
        current_price = round(float(analysis_input['Close'].iloc[-1]), 2)
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
                'comprehensive_technicals': comprehensive_technicals
            },
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
    # Create header
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
                # Show individual technical analysis
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                
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
        st.write("## 🚀 VWV Professional Trading System - Modular Architecture")
        st.write("**Phase 1 Implementation:** Core modules extracted while maintaining single-page interface.")
        
        with st.expander("🏗️ Modular Architecture Overview", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📁 **Extracted Modules**")
                st.write("✅ **`config/`** - Settings, constants, parameters")
                st.write("✅ **`data/`** - Fetching, validation, management")
                st.write("✅ **`analysis/`** - Technical calculations")
                st.write("✅ **`utils/`** - Helpers, decorators, formatters")
                st.write("🔄 **Remaining:** VWV system, UI components, charts")
                
            with col2:
                st.write("### 🎯 **Benefits Achieved**")
                st.write("• **Cleaner code organization**")
                st.write("• **Easier debugging and testing**")
                st.write("• **Reusable components**")
                st.write("• **Better maintainability**")
                st.write("• **Preserved user experience**")
        
        # Show current market status
        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.write("1. **Enter a symbol** in the sidebar (e.g., AAPL, SPY, QQQ)")
            st.write("2. **Click 'Analyze Symbol'** to run modular analysis")
            st.write("3. **View Technical Composite Score** and detailed indicators")
            st.write("4. **Enable Debug Mode** to see modular architecture in action")

    # Footer
    st.markdown("---")
    st.write("### 📊 System Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Version:** VWV Professional v3.0 - Modular")
        st.write(f"**Architecture:** Phase 1 Implementation")
    with col2:
        st.write(f"**Status:** ✅ Modular Components Active")
        st.write(f"**Modules:** config, data, analysis, utils")
    with col3:
        st.write(f"**Interface:** Single-page with expanders")
        st.write(f"**Next Phase:** VWV system, UI, charts extraction")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.write("Please refresh the page and try again.")
        
        if st.checkbox("Show Error Details"):
            st.exception(e)
