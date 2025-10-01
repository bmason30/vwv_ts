"""
Filename: app.py
VWV Trading System v4.2.1
Created/Updated: 2025-09-04 16:45:00 EDT
Version: 7.0.2 - Definitive restoration of all UI components and debug flag logic
Purpose: Main Streamlit application with all modules integrated.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# --- Safe Module Imports ---
from config.settings import UI_SETTINGS
from config.constants import QUICK_LINK_CATEGORIES, SYMBOL_DESCRIPTIONS
from data.fetcher import get_market_data_enhanced, is_etf
from ui.components import create_header, create_technical_score_bar, create_volume_score_bar, create_volatility_score_bar
from utils.helpers import format_large_number

try:
    from analysis.technical import calculate_comprehensive_technicals, calculate_composite_technical_score
    TECHNICAL_ANALYSIS_AVAILABLE = True
except ImportError: TECHNICAL_ANALYSIS_AVAILABLE = False
try:
    from analysis.volume import calculate_complete_volume_analysis
    VOLUME_ANALYSIS_AVAILABLE = True
except ImportError: VOLUME_ANALYSIS_AVAILABLE = False
try:
    from analysis.volatility import calculate_complete_volatility_analysis
    VOLATILITY_ANALYSIS_AVAILABLE = True
except ImportError: VOLATILITY_ANALYSIS_AVAILABLE = False
try:
    from analysis.fundamental import calculate_graham_score, calculate_piotroski_score
    FUNDAMENTAL_ANALYSIS_AVAILABLE = True
except ImportError: FUNDAMENTAL_ANALYSIS_AVAILABLE = False
try:
    from analysis.baldwin_indicator import calculate_baldwin_indicator_complete, format_baldwin_for_display
    BALDWIN_INDICATOR_AVAILABLE = True
except ImportError: BALDWIN_INDICATOR_AVAILABLE = False

st.set_page_config(page_title="VWV Professional Trading System", page_icon="🚀", layout="wide")
warnings.filterwarnings('ignore', category=FutureWarning)


def setup_session_state():
    """Initializes all necessary keys in Streamlit's session state."""
    defaults = {
        'recently_viewed': [], 'show_technical_analysis': True, 'show_volume_analysis': True,
        'show_volatility_analysis': True, 'show_fundamental_analysis': True,
        'show_baldwin_indicator': True, 'auto_analyze': False
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def create_sidebar_controls():
    """Creates all sidebar UI controls and returns the application state."""
    st.sidebar.title("📊 Trading Analysis v4.2.1")
    if 'selected_symbol' in st.session_state:
        current_symbol = st.session_state.selected_symbol
        st.session_state.auto_analyze = True
        del st.session_state.selected_symbol
    else: current_symbol = UI_SETTINGS['default_symbol']
    symbol = st.sidebar.text_input("Symbol", value=current_symbol, help="Enter stock symbol").upper()
    period = st.sidebar.selectbox("Data Period", ['1mo', '3mo', '6mo', '1y', '2y'], index=3)
    analyze_button = st.sidebar.button("📊 Analyze Symbol", type="primary", use_container_width=True)
    if st.session_state.auto_analyze:
        st.session_state.auto_analyze = False
        analyze_button = True
    with st.sidebar.expander("🔗 Quick Links", expanded=True):
        for category, symbols in QUICK_LINK_CATEGORIES.items():
            with st.expander(f"{category} ({len(symbols)})", expanded=False):
                for i in range(0, len(symbols), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(symbols):
                            sym = symbols[i + j]
                            if col.button(sym, help=SYMBOL_DESCRIPTIONS.get(sym, sym), key=f"ql_{sym}", use_container_width=True):
                                st.session_state.selected_symbol = sym
                                st.rerun()
    def add_to_recently_viewed(symbol):
        if symbol in st.session_state.recently_viewed: st.session_state.recently_viewed.remove(symbol)
        st.session_state.recently_viewed.insert(0, symbol)
        st.session_state.recently_viewed = st.session_state.recently_viewed[:10]
    if st.session_state.recently_viewed:
        with st.sidebar.expander("⏰ Recently Viewed", expanded=False):
            for r_sym in st.session_state.recently_viewed:
                if st.button(f"📊 {r_sym}", key=f"rec_{r_sym}", use_container_width=True):
                    st.session_state.selected_symbol = r_sym
                    st.rerun()
    with st.sidebar.expander("🎛️ Analysis Sections", expanded=True):
        st.session_state.show_technical_analysis = st.checkbox("Technical", st.session_state.show_technical_analysis)
        st.session_state.show_volume_analysis = st.checkbox("Volume", st.session_state.show_volume_analysis)
        st.session_state.show_volatility_analysis = st.checkbox("Volatility", st.session_state.show_volatility_analysis)
        st.session_state.show_fundamental_analysis = st.checkbox("Fundamental", st.session_state.show_fundamental_analysis)
        st.session_state.show_baldwin_indicator = st.checkbox("Baldwin Regime", st.session_state.show_baldwin_indicator)
    show_debug = st.sidebar.checkbox("Show debug info", False)
    return {'symbol': symbol, 'period': period, 'analyze_button': analyze_button, 'show_debug': show_debug, 'add_to_recently_viewed': add_to_recently_viewed}

def perform_full_analysis(symbol, period, show_debug=False):
    analysis_input = get_market_data_enhanced(symbol, period)
    if analysis_input is None: return None
    analysis_results = {'symbol': symbol, 'current_price': analysis_input['Close'].iloc[-1], 'enhanced_indicators': {}}
    if TECHNICAL_ANALYSIS_AVAILABLE:
        analysis_results['enhanced_indicators']['comprehensive_technicals'] = calculate_comprehensive_technicals(analysis_input)
    if VOLUME_ANALYSIS_AVAILABLE:
        analysis_results['enhanced_indicators']['volume_analysis'] = calculate_complete_volume_analysis(analysis_input)
    if VOLATILITY_ANALYSIS_AVAILABLE:
        analysis_results['enhanced_indicators']['volatility_analysis'] = calculate_complete_volatility_analysis(analysis_input)
    if FUNDAMENTAL_ANALYSIS_AVAILABLE:
        if not is_etf(symbol):
            analysis_results['enhanced_indicators']['graham_score'] = calculate_graham_score(symbol, show_debug)
            analysis_results['enhanced_indicators']['piotroski_score'] = calculate_piotroski_score(symbol, show_debug)
        else:
            analysis_results['enhanced_indicators']['graham_score'] = {'error': 'Not applicable for ETFs'}
            analysis_results['enhanced_indicators']['piotroski_score'] = {'error': 'Not applicable for ETFs'}
    return analysis_results

def show_technical_analysis(results, show_debug):
    if not st.session_state.get('show_technical_analysis', True): return
    with st.expander(f"📊 {results['symbol']} - Individual Technical Analysis", expanded=True):
        score, details = calculate_composite_technical_score(results)
        if not details or 'error' in details: return
        create_technical_score_bar(score, "Composite Technical Score")
        divergence = details.get('momentum_divergence', {})
        if divergence and divergence.get('signals'): st.info(f"**Reversal Signal Detected:** {', '.join(divergence['signals'])}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RSI (14)", f"{details.get('rsi_14', 0):.1f}")
        c2.metric("MFI (14)", f"{details.get('mfi_14', 0):.1f}")
        c3.metric("MACD Hist", f"{details.get('macd', {}).get('histogram', 0):.4f}")
        c4.metric("BBand Position", f"{details.get('bollinger_bands', {}).get('position', 0):.1f}%")

def show_volume_analysis(results, show_debug):
    if not st.session_state.get('show_volume_analysis', True): return
    with st.expander(f"📊 {results['symbol']} - Volume Analysis", expanded=True):
        data = results.get('enhanced_indicators', {}).get('volume_analysis', {})
        if not data or 'error' in data: return
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Volume", format_large_number(data.get('current_volume', 0)))
        c2.metric("5D Avg Volume", format_large_number(data.get('volume_5d_avg', 0)))
        c3.metric("Volume Ratio", f"{data.get('volume_ratio', 0):.2f}x")
        create_volume_score_bar(data.get('volume_score', 50), "Volume Score")

def show_volatility_analysis(results, show_debug):
    if not st.session_state.get('show_volatility_analysis', True): return
    with st.expander(f"📊 {results['symbol']} - Volatility Analysis", expanded=True):
        data = results.get('enhanced_indicators', {}).get('volatility_analysis', {})
        if not data or 'error' in data: return
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("20D Volatility", f"{data.get('volatility_20d', 0):.2f}%")
        c2.metric("Vol Percentile", f"{data.get('volatility_percentile', 0):.1f}%")
        c3.metric("Vol Rank", f"{data.get('volatility_rank', 0):.1f}%")
        create_volatility_score_bar(data.get('volatility_score', 50), "Volatility Score")

def show_fundamental_analysis(results, show_debug):
    if not st.session_state.get('show_fundamental_analysis', True): return
    with st.expander(f"📜 {results['symbol']} - Fundamental Analysis", expanded=True):
        graham = results.get('enhanced_indicators', {}).get('graham_score')
        piotroski = results.get('enhanced_indicators', {}).get('piotroski_score')
        if graham is None or piotroski is None:
            st.warning("Fundamental analysis data could not be calculated.")
            return
        if 'error' in graham and 'ETF' in graham['error']:
            st.info("Fundamental analysis is not applicable for ETFs.")
            return
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Graham Score")
            if 'error' not in graham: st.metric(f"Grade: {graham.get('grade', 'N/A')}", f"{graham.get('score', 0)}/{graham.get('total_possible', 10)}", delta=graham.get('interpretation'), delta_color="off")
            else: st.warning(f"Could not calculate: {graham['error']}")
        with c2:
            st.subheader("Piotroski F-Score")
            if 'error' not in piotroski: st.metric(f"Grade: {piotroski.get('grade', 'N/A')}", f"{piotroski.get('score', 0)}/{piotroski.get('total_possible', 9)}", delta=piotroski.get('interpretation'), delta_color="off")
            else: st.warning(f"Could not calculate: {piotroski['error']}")

def show_baldwin_indicator_analysis(show_debug=False):
    if not st.session_state.get('show_baldwin_indicator', True): return
    with st.expander("🚦 Baldwin Market Regime Indicator", expanded=True):
        baldwin_results = calculate_baldwin_indicator_complete(show_debug)
        if baldwin_results and baldwin_results.get('status') == 'OPERATIONAL':
            display_data = format_baldwin_for_display(baldwin_results)
            regime, score, strategy = display_data.get('regime', 'UNKNOWN'), display_data.get('overall_score', 0), display_data.get('strategy', 'N/A')
            color = "green" if regime == "GREEN" else "orange" if regime == "YELLOW" else "red"
            st.header(f"Market Regime: :{color}[{regime}]")
            c1, c2 = st.columns(2)
            c1.metric("Baldwin Composite Score", f"{score:.1f} / 100")
            c2.info(f"**Strategy:** {strategy}")
            st.markdown("---")
            st.subheader("Component Breakdown")
            st.dataframe(pd.DataFrame(display_data.get('component_summary', [])), hide_index=True)
            
            detailed_breakdown = display_data.get('detailed_breakdown', {})
            mom_tab, liq_tab, sen_tab = st.tabs(["Momentum", "Liquidity & Credit", "Sentiment & Entry"])
            with mom_tab:
                if 'Momentum' in detailed_breakdown:
                    details = detailed_breakdown['Momentum']['details']
                    c1, c2 = st.columns(2)
                    with c1:
                        spy_details = details['Broad Market (SPY)']
                        st.metric("Synthesized SPY Score", f"{spy_details['score']:.1f}")
                        st.progress(spy_details['trend']['score'] / 100, text=f"Trend Strength: {spy_details['trend']['score']:.1f}")
                    with c2:
                        iwm_details = details['Market Internals (IWM)']
                        st.metric("Market Internals (IWM) Score", f"{iwm_details['score']:.1f}")
            with liq_tab:
                if 'Liquidity_Credit' in detailed_breakdown:
                    details = detailed_breakdown['Liquidity_Credit']['details']
                    c1, c2 = st.columns(2)
                    with c1:
                        fs_details = details['Flight-to-Safety']
                        st.metric("Flight-to-Safety Score", f"{fs_details['score']:.1f}")
                    with c2:
                        cs_details = details['Credit Spreads']
                        st.metric("Credit Spreads Score", f"{cs_details['score']:.1f}")
            with sen_tab:
                if 'Sentiment_Entry' in detailed_breakdown:
                    details = detailed_breakdown['Sentiment_Entry']['details']
                    c1, c2 = st.columns(2)
                    with c1:
                        se_details = details['Sentiment ETFs']
                        st.metric("Sentiment ETF Score", f"{se_details['score']:.1f}")
                    with c2:
                        ec_details = details['Entry Confirmation']
                        st.metric("Entry Confirmation", "✅ Confirmed" if ec_details['confirmed'] else "⏳ Awaiting")
        else: st.error("Baldwin Indicator calculation failed.")

def main():
    """Main application function - DIAGNOSTIC VERSION"""
    import traceback
    
    st.write("=" * 60)
    st.write("🔬 DIAGNOSTIC MODE ACTIVE")
    st.write("=" * 60)
    
    try:
        st.write("Step 1: Creating header...")
        create_header()
        st.write("✅ Header created")
    except Exception as e:
        st.error(f"❌ Header creation failed: {e}")
        st.code(traceback.format_exc())
    
    try:
        st.write("Step 2: Creating sidebar controls...")
        controls = create_sidebar_controls()
        st.write("✅ Sidebar controls created")
        st.write("**Controls received:**")
        st.json(controls)
    except Exception as e:
        st.error(f"❌ Sidebar creation failed: {e}")
        st.code(traceback.format_exc())
        return
    
    # Check conditions
    st.write("---")
    st.write("Step 3: Checking analysis conditions...")
    st.write(f"  • analyze_button: `{controls.get('analyze_button', 'NOT FOUND')}`")
    st.write(f"  • symbol: `{controls.get('symbol', 'NOT FOUND')}`")
    st.write(f"  • period: `{controls.get('period', 'NOT FOUND')}`")
    st.write(f"  • show_debug: `{controls.get('show_debug', 'NOT FOUND')}`")
    
    condition_check = controls['analyze_button'] and controls['symbol']
    st.write(f"  • **Condition check result: `{condition_check}`**")
    
    if condition_check:
        st.write("---")
        st.success("✅ Conditions met! Entering analysis block...")
        
        try:
            st.write("Step 4: Adding to recently viewed...")
            add_to_recently_viewed(controls['symbol'])
            st.write("✅ Added to recently viewed")
        except Exception as e:
            st.warning(f"⚠️ Recently viewed failed (non-critical): {e}")
        
        st.write("## 📊 VWV Trading Analysis v4.2.1 Enhanced")
        
        st.write("---")
        st.write("Step 5: Calling perform_enhanced_analysis...")
        st.write(f"  Parameters: symbol='{controls['symbol']}', period='{controls['period']}', show_debug={controls['show_debug']}")
        
        try:
            with st.spinner(f"Analyzing {controls['symbol']}..."):
                st.write("⏳ Starting analysis...")
                
                analysis_results, chart_data = perform_enhanced_analysis(
                    controls['symbol'], 
                    controls['period'], 
                    controls['show_debug']
                )
                
                st.write("✅ perform_enhanced_analysis completed")
                st.write(f"  • analysis_results is None: {analysis_results is None}")
                st.write(f"  • chart_data is None: {chart_data is None}")
                
                if analysis_results is not None:
                    st.write(f"  • analysis_results keys: {list(analysis_results.keys())}")
                if chart_data is not None:
                    st.write(f"  • chart_data shape: {chart_data.shape}")
        
        except Exception as e:
            st.error(f"❌ EXCEPTION in perform_enhanced_analysis: {e}")
            st.code(traceback.format_exc())
            return
        
        st.write("---")
        st.write("Step 6: Checking if we have results to display...")
        
        if analysis_results and chart_data is not None:
            st.success("✅ We have results! Displaying analysis...")
            
            # Try to display at least something
            try:
                st.write("**Analysis Results Preview:**")
                st.write(f"Symbol: {analysis_results.get('symbol', 'N/A')}")
                st.write(f"Current Price: ${analysis_results.get('current_price', 'N/A')}")
                st.write(f"Timestamp: {analysis_results.get('timestamp', 'N/A')}")
            except Exception as e:
                st.error(f"Error displaying preview: {e}")
            
            # Try the full display
            try:
                st.write("Attempting to show full analysis sections...")
                
                # Try just the first display function
                try:
                    st.write("Trying show_interactive_charts...")
                    show_interactive_charts(chart_data, analysis_results, controls['show_debug'])
                    st.success("✅ Charts displayed")
                except Exception as e:
                    st.error(f"❌ Charts failed: {e}")
                    st.code(traceback.format_exc())
                
                # Add more sections here if charts work...
                
            except Exception as e:
                st.error(f"❌ Display error: {e}")
                st.code(traceback.format_exc())
        else:
            st.error("❌ No results to display")
            st.write("**Why?**")
            if analysis_results is None:
                st.write("• analysis_results is None")
            if chart_data is None:
                st.write("• chart_data is None")
    else:
        st.warning("⚠️ Analysis block NOT entered")
        st.write("**Reason:** Conditions not met")
        st.write(f"  • Button clicked: {controls.get('analyze_button', False)}")
        st.write(f"  • Symbol provided: '{controls.get('symbol', '')}'")
        
        if not controls.get('analyze_button', False):
            st.info("💡 Click the 'Analyze Symbol' button to start")
        if not controls.get('symbol', ''):
            st.info("💡 Enter a symbol in the sidebar")

if __name__ == "__main__":
    main()
