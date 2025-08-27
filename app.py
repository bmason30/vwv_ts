"""
File: app.py  
VWV Professional Trading System v4.2.1 - DIAGNOSTIC VERSION
Version: v4.2.1-DIAGNOSTIC-FORCE-VOLATILITY-2025-08-27-18-40-00-EST
PURPOSE: Force volatility section to appear and diagnose why it's missing
Last Updated: August 27, 2025 - 6:40 PM EST
"""

import html
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

# DIAGNOSTIC: Force availability flags and show import status
st.write("🔍 **DIAGNOSTIC MODE** - Import Status:")

# Volume Analysis Import Diagnostic
try:
    from analysis.volume import calculate_complete_volume_analysis
    VOLUME_ANALYSIS_AVAILABLE = True
    st.write("✅ Volume analysis import: SUCCESS")
except ImportError as e:
    VOLUME_ANALYSIS_AVAILABLE = False
    st.write(f"❌ Volume analysis import: FAILED - {e}")

# Volatility Analysis Import Diagnostic  
try:
    from analysis.volatility import calculate_complete_volatility_analysis
    VOLATILITY_ANALYSIS_AVAILABLE = True
    st.write("✅ Volatility analysis import: SUCCESS")
except ImportError as e:
    VOLATILITY_ANALYSIS_AVAILABLE = False
    st.write(f"❌ Volatility analysis import: FAILED - {e}")

# Display Function Import Diagnostic
try:
    from app_volatility_display import show_volatility_analysis, show_volume_analysis
    DISPLAY_FUNCTIONS_AVAILABLE = True
    st.write("✅ Display functions import: SUCCESS")
except ImportError as e:
    DISPLAY_FUNCTIONS_AVAILABLE = False
    st.write(f"❌ Display functions import: FAILED - {e}")

from ui.components import (
    create_technical_score_bar,
    create_header,
    format_large_number
)
from utils.helpers import get_market_status, get_etf_description
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
    """Create sidebar controls - CORRECTED STRUCTURE"""
    st.sidebar.title("📊 Trading Analysis v4.2.1")
    
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

    # Symbol input
    col1, col2 = st.sidebar.columns([3, 1])
    
    with col1:
        symbol_input = st.text_input("Enter Symbol", value="", placeholder="AAPL, TSLA, SPY...")
    
    with col2:
        analyze_button = st.button("Analyze", use_container_width=True, type="primary")

    # Time period selection - DEFAULT TO 1mo
    period = st.sidebar.selectbox(
        "📅 Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y"],
        index=0,  # Default to 1mo
        help="Select analysis time period"
    )

    # Quick Links PROPERLY GROUPED in single expander
    quick_link_clicked = None
    with st.sidebar.expander("🔗 Quick Links", expanded=False):
        st.write("**Popular Symbols by Category**")
        
        for category, symbols in QUICK_LINK_CATEGORIES.items():
            with st.expander(f"📊 {category}", expanded=False):
                cols = st.columns(2)
                for i, symbol in enumerate(symbols):
                    col = cols[i % 2]
                    with col:
                        description = SYMBOL_DESCRIPTIONS.get(symbol, symbol)
                        if st.button(f"{symbol}", key=f"quick_{symbol}", help=description, use_container_width=True):
                            quick_link_clicked = symbol

    # Recently viewed
    if st.session_state.recently_viewed:
        with st.sidebar.expander("📈 Recently Viewed", expanded=False):
            recent_cols = st.sidebar.columns(2)
            for i, recent_symbol in enumerate(st.session_state.recently_viewed[-6:]):
                col = recent_cols[i % 2]
                with col:
                    if st.button(f"{recent_symbol}", key=f"recent_{recent_symbol}", use_container_width=True):
                        quick_link_clicked = recent_symbol

    # Analysis sections toggle
    with st.sidebar.expander("⚙️ Analysis Sections", expanded=False):
        st.session_state.show_technical_analysis = st.checkbox(
            "📊 Technical Analysis", 
            value=st.session_state.show_technical_analysis
        )
        
        st.session_state.show_volume_analysis = st.checkbox(
            f"🔊 Volume Analysis {'✅' if VOLUME_ANALYSIS_AVAILABLE else '❌'}", 
            value=st.session_state.show_volume_analysis
        )
        
        st.session_state.show_volatility_analysis = st.checkbox(
            f"🌡️ Volatility Analysis {'✅' if VOLATILITY_ANALYSIS_AVAILABLE else '❌'}", 
            value=st.session_state.show_volatility_analysis
        )
        
        st.session_state.show_fundamental_analysis = st.checkbox(
            "📈 Fundamental Analysis", 
            value=st.session_state.show_fundamental_analysis
        )
        
        st.session_state.show_market_correlation = st.checkbox(
            "🌐 Market Correlation", 
            value=st.session_state.show_market_correlation
        )
        
        st.session_state.show_options_analysis = st.checkbox(
            "🎯 Options Analysis", 
            value=st.session_state.show_options_analysis
        )
        
        st.session_state.show_confidence_intervals = st.checkbox(
            "📊 Confidence Intervals", 
            value=st.session_state.show_confidence_intervals
        )

    # Debug toggle
    show_debug = st.sidebar.checkbox("🐛 Debug Mode", value=True)  # DEFAULT TO TRUE

    # Market status
    market_status = get_market_status()
    if market_status:
        st.sidebar.info(f"🏛️ Market: {market_status}")

    # Diagnostic info in sidebar
    st.sidebar.write("**🔍 Module Status:**")
    st.sidebar.write(f"Volume: {'✅' if VOLUME_ANALYSIS_AVAILABLE else '❌'}")
    st.sidebar.write(f"Volatility: {'✅' if VOLATILITY_ANALYSIS_AVAILABLE else '❌'}")
    st.sidebar.write(f"Display Funcs: {'✅' if DISPLAY_FUNCTIONS_AVAILABLE else '❌'}")

    # Determine final symbol
    final_symbol = None
    final_analyze = False

    if quick_link_clicked:
        final_symbol = quick_link_clicked.upper()
        final_analyze = True
    elif analyze_button and symbol_input:
        final_symbol = symbol_input.upper().strip()
        final_analyze = True

    return {
        'symbol': final_symbol,
        'analyze_button': final_analyze,
        'period': period,
        'show_debug': show_debug
    }

def add_to_recently_viewed(symbol):
    """Add symbol to recently viewed list"""
    if symbol and symbol not in st.session_state.recently_viewed:
        st.session_state.recently_viewed.append(symbol)
        if len(st.session_state.recently_viewed) > 10:
            st.session_state.recently_viewed.pop(0)

def show_interactive_charts(chart_data, analysis_results, show_debug=False):
    """Display interactive charts"""
    st.subheader("📊 Interactive Charts")
    
    if chart_data is not None and len(chart_data) > 0:
        try:
            st.line_chart(chart_data['Close'])
            if show_debug:
                st.write(f"📊 Chart data points: {len(chart_data)}")
        except Exception as e:
            st.error(f"❌ Chart display failed: {e}")
    else:
        st.warning("⚠️ No chart data available")

def show_individual_technical_analysis(analysis_results, show_debug=False):
    """Display technical analysis section"""
    if not st.session_state.show_technical_analysis:
        return
        
    symbol = analysis_results.get('symbol', 'Unknown')
    
    with st.expander(f"📊 {symbol} - Individual Technical Analysis", expanded=True):
        try:
            composite_score, score_details = calculate_composite_technical_score(analysis_results)
            create_technical_score_bar(composite_score, score_details)
            
            enhanced_indicators = analysis_results.get('enhanced_indicators', {})
            comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
            
            if comprehensive_technicals:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    current_price = comprehensive_technicals.get('current_price', 0)
                    st.metric("Current Price", f"${current_price:.2f}")
                
                with col2:
                    rsi_value = comprehensive_technicals.get('rsi', 50)
                    rsi_desc = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                    st.metric("RSI(14)", f"{rsi_value:.1f}", rsi_desc)
                
                with col3:
                    vwap_position = comprehensive_technicals.get('vwap_position', 0)
                    vwap_desc = "Above VWAP" if vwap_position > 0 else "Below VWAP" if vwap_position < 0 else "At VWAP"
                    st.metric("VWAP Position", f"{vwap_position:+.2f}%", vwap_desc)
                
                with col4:
                    trend_strength = comprehensive_technicals.get('trend_strength', 50)
                    trend_desc = "Strong" if trend_strength > 70 else "Weak" if trend_strength < 30 else "Moderate"
                    st.metric("Trend Strength", f"{trend_strength:.0f}/100", trend_desc)
                    
        except Exception as e:
            st.error(f"❌ Technical analysis display failed: {e}")

def show_volatility_analysis_forced(analysis_results, show_debug=False):
    """FORCE volatility analysis section to appear - DIAGNOSTIC VERSION"""
    
    # ALWAYS SHOW THIS SECTION FOR DIAGNOSTIC PURPOSES
    symbol = analysis_results.get('symbol', 'Unknown')
    
    with st.expander(f"🌡️ {symbol} - Volatility Analysis (DIAGNOSTIC)", expanded=True):
        
        st.write("🔍 **VOLATILITY ANALYSIS DIAGNOSTIC:**")
        st.write(f"- Session State Enabled: {st.session_state.get('show_volatility_analysis', 'Not Set')}")
        st.write(f"- VOLATILITY_ANALYSIS_AVAILABLE: {VOLATILITY_ANALYSIS_AVAILABLE}")
        st.write(f"- DISPLAY_FUNCTIONS_AVAILABLE: {DISPLAY_FUNCTIONS_AVAILABLE}")
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volatility_analysis = enhanced_indicators.get('volatility_analysis', {})
        
        st.write(f"- Volatility data available: {bool(volatility_analysis)}")
        st.write(f"- Volatility data keys: {list(volatility_analysis.keys()) if volatility_analysis else 'None'}")
        
        if volatility_analysis:
            if 'error' in volatility_analysis:
                st.error(f"❌ Volatility Analysis Error: {volatility_analysis['error']}")
            else:
                st.success("✅ Volatility data found! Attempting display...")
                
                # Try to display volatility data
                try:
                    if DISPLAY_FUNCTIONS_AVAILABLE:
                        from app_volatility_display import show_volatility_analysis
                        show_volatility_analysis(analysis_results, show_debug)
                    else:
                        # Fallback simple display
                        st.write("**Fallback Volatility Display:**")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            vol_20d = volatility_analysis.get('volatility_20d', 0)
                            st.metric("20-Day Volatility", f"{vol_20d:.1f}%")
                            
                        with col2:
                            vol_10d = volatility_analysis.get('volatility_10d', 0)
                            st.metric("10-Day Volatility", f"{vol_10d:.1f}%")
                            
                        with col3:
                            vol_score = volatility_analysis.get('volatility_score', 50)
                            st.metric("Volatility Score", f"{vol_score:.1f}/100")
                            
                        with col4:
                            vol_regime = volatility_analysis.get('volatility_regime', 'Unknown')
                            st.metric("Volatility Regime", vol_regime)
                        
                        # Show some key data
                        if volatility_analysis.get('options_strategy'):
                            st.info(f"**Options Strategy:** {volatility_analysis['options_strategy']}")
                        
                        if volatility_analysis.get('trading_implications'):
                            st.info(f"**Trading Implications:** {volatility_analysis['trading_implications']}")
                        
                        # Show advanced indicators if available
                        if 'indicators' in volatility_analysis:
                            with st.expander("Advanced Volatility Indicators", expanded=False):
                                indicators = volatility_analysis['indicators']
                                for key, value in indicators.items():
                                    st.write(f"• {key.replace('_', ' ').title()}: {value}")
                                
                except Exception as e:
                    st.error(f"❌ Display failed: {e}")
                    if show_debug:
                        st.exception(e)
        else:
            st.warning("⚠️ No volatility analysis data found in results")
            
        # Show raw data for debugging
        if show_debug and volatility_analysis:
            with st.expander("🐛 Raw Volatility Data", expanded=False):
                st.json(volatility_analysis)

def show_fundamental_analysis(analysis_results, show_debug=False):
    """Display fundamental analysis section"""
    if not st.session_state.show_fundamental_analysis:
        return
        
    symbol = analysis_results.get('symbol', 'Unknown')
    
    with st.expander(f"📈 {symbol} - Fundamental Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        graham_score = enhanced_indicators.get('graham_score', {})
        piotroski_score = enhanced_indicators.get('piotroski_score', {})
        
        if graham_score or piotroski_score:
            col1, col2 = st.columns(2)
            
            with col1:
                if graham_score and 'error' not in graham_score:
                    st.subheader("📊 Graham Score")
                    score = graham_score.get('total_score', 0)
                    max_score = graham_score.get('max_possible_score', 10)
                    st.metric("Graham Score", f"{score}/{max_score}")
                
            with col2:
                if piotroski_score and 'error' not in piotroski_score:
                    st.subheader("📊 Piotroski Score")
                    score = piotroski_score.get('total_score', 0)
                    st.metric("Piotroski Score", f"{score}/9")
        else:
            st.warning("⚠️ Fundamental analysis not available - insufficient data")

@safe_calculation_wrapper
def perform_enhanced_analysis(symbol, period, show_debug=False):
    """Perform comprehensive analysis with FORCED VOLATILITY INTEGRATION"""
    try:
        # Get data manager and fetch data
        data_manager = get_data_manager()
        data = get_market_data_enhanced(symbol, period)
        
        if data is None or len(data) == 0:
            raise ValueError(f"No data available for {symbol}")
        
        # Store data
        data_manager.store_market_data(symbol, data, period)
        
        # Initialize analysis results
        enhanced_indicators = {}
        
        # Core technical analysis
        daily_vwap = calculate_daily_vwap(data)
        fibonacci_emas = calculate_fibonacci_emas(data)
        poc_enhanced = calculate_point_of_control_enhanced(data)
        comprehensive_technicals = calculate_comprehensive_technicals(data)
        weekly_deviations = calculate_weekly_deviations(data)
        
        enhanced_indicators.update({
            'daily_vwap': daily_vwap,
            'fibonacci_emas': fibonacci_emas,
            'poc_enhanced': poc_enhanced,
            'comprehensive_technicals': comprehensive_technicals,
            'weekly_deviations': weekly_deviations
        })
        
        # FORCED VOLATILITY ANALYSIS - ALWAYS TRY TO CALCULATE
        st.write("🔍 **Attempting Volatility Analysis...**")
        
        if VOLATILITY_ANALYSIS_AVAILABLE:
            try:
                volatility_analysis = calculate_complete_volatility_analysis(data)
                enhanced_indicators['volatility_analysis'] = volatility_analysis
                st.write(f"✅ Volatility analysis completed successfully")
            except Exception as e:
                error_msg = f'Volatility analysis calculation failed: {e}'
                enhanced_indicators['volatility_analysis'] = {'error': error_msg}
                st.error(f"❌ {error_msg}")
                if show_debug:
                    st.exception(e)
        else:
            enhanced_indicators['volatility_analysis'] = {'error': 'Volatility analysis module not available'}
            st.warning("⚠️ Volatility analysis module not available")
        
        # Market correlation analysis
        market_correlations = calculate_market_correlations_enhanced(symbol, period)
        breakout_analysis = calculate_breakout_breakdown_analysis(data)
        enhanced_indicators.update({
            'market_correlations': market_correlations,
            'breakout_analysis': breakout_analysis
        })
        
        # Options analysis
        options_levels = calculate_options_levels_enhanced(data)
        enhanced_indicators['options_levels'] = options_levels
        
        # Fundamental analysis
        graham_score = calculate_graham_score(symbol)
        piotroski_score = calculate_piotroski_score(symbol) 
        enhanced_indicators.update({
            'graham_score': graham_score,
            'piotroski_score': piotroski_score
        })
        
        # Confidence intervals
        confidence_analysis = calculate_confidence_intervals(data)
        
        # Build complete analysis results
        analysis_results = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'enhanced_indicators': enhanced_indicators,
            'confidence_analysis': confidence_analysis,
            'system_status': 'DIAGNOSTIC v4.2.1'
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
    """Main application function - DIAGNOSTIC VERSION"""
    # Create header using modular component
    create_header()
    
    # Show diagnostic info at top
    st.write("## 🔍 DIAGNOSTIC MODE - Volatility Integration Testing")
    
    # Create sidebar and get controls
    controls = create_sidebar_controls()
    
    # Main logic flow
    if controls['analyze_button'] and controls['symbol']:
        # Add symbol to recently viewed
        add_to_recently_viewed(controls['symbol'])
        
        st.write(f"## 📊 VWV Trading Analysis - {controls['symbol']}")
        
        with st.spinner(f"Analyzing {controls['symbol']}..."):
            
            # Perform analysis using modular components
            analysis_results, chart_data = perform_enhanced_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results and chart_data is not None:
                
                # 1. CHARTS FIRST
                show_interactive_charts(chart_data, analysis_results, controls['show_debug'])
                
                # 2. TECHNICAL ANALYSIS
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                
                # 3. FORCED VOLATILITY ANALYSIS - ALWAYS SHOW
                show_volatility_analysis_forced(analysis_results, controls['show_debug'])
                
                # 4. Fundamental Analysis
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                
                # DEBUG INFO
                if controls['show_debug']:
                    with st.expander("🐛 Full Debug Information", expanded=True):
                        st.write("### System Status")
                        st.write(f"**Volume Analysis Available:** {VOLUME_ANALYSIS_AVAILABLE}")
                        st.write(f"**Volatility Analysis Available:** {VOLATILITY_ANALYSIS_AVAILABLE}")
                        st.write(f"**Display Functions Available:** {DISPLAY_FUNCTIONS_AVAILABLE}")
                        
                        st.write("### Analysis Results Structure")
                        st.json(analysis_results, expanded=False)
    
    else:
        # Welcome message
        st.write("## 🚀 VWV Professional Trading System v4.2.1 - Diagnostic Mode")
        st.write("**PURPOSE:** Identify why volatility analysis section is not appearing")
        
        st.write("**Current Status:**")
        st.write(f"✅ Volume Analysis Available: {VOLUME_ANALYSIS_AVAILABLE}")
        st.write(f"✅ Volatility Analysis Available: {VOLATILITY_ANALYSIS_AVAILABLE}")
        st.write(f"✅ Display Functions Available: {DISPLAY_FUNCTIONS_AVAILABLE}")
        
        st.write("Select a symbol to test volatility integration...")

if __name__ == "__main__":
    main()
