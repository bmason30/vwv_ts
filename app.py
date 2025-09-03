"""
Filename: app.py
VWV Trading System v4.2.1
Created/Updated: 2025-09-03 17:07:20 EDT
Version: 4.4.2 - Diagnostic build to isolate startup error
Purpose: Main Streamlit application with Baldwin preview disabled for debugging.
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

# Volume and Volatility imports with safe fallbacks
try:
    from analysis.volume import calculate_complete_volume_analysis
    VOLUME_ANALYSIS_AVAILABLE = True
except ImportError:
    VOLUME_ANALYSIS_AVAILABLE = False
try:
    from analysis.volatility import calculate_complete_volatility_analysis
    VOLATILITY_ANALYSIS_AVAILABLE = True
except ImportError:
    VOLATILITY_ANALYSIS_AVAILABLE = False
try:
    from analysis.baldwin_indicator import calculate_baldwin_indicator_complete, format_baldwin_for_display
    BALDWIN_INDICATOR_AVAILABLE = True
except ImportError:
    BALDWIN_INDICATOR_AVAILABLE = False
from ui.components import create_technical_score_bar, create_header
from utils.helpers import format_large_number, get_market_status, get_etf_description
from utils.decorators import safe_calculation_wrapper

st.set_page_config(page_title="VWV Professional Trading System v4.2.1", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")
warnings.filterwarnings('ignore', category=FutureWarning)


def create_sidebar_controls():
    st.sidebar.title("📊 Trading Analysis v4.2.1")
    
    # Initialize session state
    for key, default in [('recently_viewed', []), ('show_charts', True), ('show_technical_analysis', True), 
                         ('show_volume_analysis', True), ('show_volatility_analysis', True), 
                         ('show_fundamental_analysis', True), ('show_baldwin_indicator', True), 
                         ('show_market_correlation', True), ('show_options_analysis', True), 
                         ('show_confidence_intervals', True), ('auto_analyze', False)]:
        if key not in st.session_state:
            st.session_state[key] = default

    if 'selected_symbol' in st.session_state:
        current_symbol = st.session_state.selected_symbol
        st.session_state.auto_analyze = True
        del st.session_state.selected_symbol
    else:
        current_symbol = UI_SETTINGS['default_symbol']
        
    symbol = st.sidebar.text_input("Symbol", value=current_symbol, help="Enter stock symbol").upper()
    period = st.sidebar.selectbox("Data Period", ['1mo', '3mo', '6mo', '1y', '2y'], index=3)
    analyze_button = st.sidebar.button("📊 Analyze Symbol", type="primary", use_container_width=True)
    
    if st.session_state.auto_analyze:
        st.session_state.auto_analyze = False
        analyze_button = True
    
    with st.sidebar.expander("🔗 Quick Links", expanded=False):
        for category, symbols in QUICK_LINK_CATEGORIES.items():
            with st.expander(f"{category} ({len(symbols)})", expanded=False):
                for i in range(0, len(symbols), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(symbols):
                            sym = symbols[i + j]
                            if col.button(sym, key=f"ql_{sym}", use_container_width=True):
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
        st.session_state.show_charts = st.checkbox("Charts", st.session_state.show_charts)
        st.session_state.show_technical_analysis = st.checkbox("Technical", st.session_state.show_technical_analysis)
    
    show_debug = st.sidebar.checkbox("Show debug info", False)
    return {'symbol': symbol, 'period': period, 'analyze_button': analyze_button, 'show_debug': show_debug, 'add_to_recently_viewed': add_to_recently_viewed}

def perform_enhanced_analysis(symbol, period, show_debug=False):
    try:
        market_data = get_market_data_enhanced(symbol, period, show_debug)
        if market_data is None: return None, None
        data_manager = get_data_manager()
        data_manager.store_market_data(symbol, market_data, show_debug)
        analysis_input = data_manager.get_market_data_for_analysis(symbol)
        if analysis_input is None: return None, None
        
        analysis_results = {
            'symbol': symbol.upper(), 'current_price': float(analysis_input['Close'].iloc[-1]), 'period': period,
            'enhanced_indicators': {
                'daily_vwap': calculate_daily_vwap(analysis_input),
                'fibonacci_emas': calculate_fibonacci_emas(analysis_input),
                'point_of_control': calculate_point_of_control_enhanced(analysis_input),
                'comprehensive_technicals': calculate_comprehensive_technicals(analysis_input)
            }
        }
        chart_data = data_manager.get_market_data_for_chart(symbol)
        return analysis_results, chart_data
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return None, None

def show_baldwin_indicator_analysis(show_debug=False):
    if not st.session_state.get('show_baldwin_indicator', True) or not BALDWIN_INDICATOR_AVAILABLE: return
    
    with st.expander("🚦 Baldwin Market Regime Indicator", expanded=True):
        with st.spinner("Synthesizing multi-factor market regime..."):
            try:
                baldwin_results = calculate_baldwin_indicator_complete(show_debug)
                if baldwin_results.get('status') == 'OPERATIONAL':
                    display_data = format_baldwin_for_display(baldwin_results)
                    regime, score, strategy = display_data.get('regime', 'UNKNOWN'), display_data.get('overall_score', 0), display_data.get('strategy', 'N/A')
                    color = "green" if regime == "GREEN" else "orange" if regime == "YELLOW" else "red"
                    
                    st.header(f"Market Regime: :{color}[{regime}]")
                    c1, c2 = st.columns(2)
                    c1.metric("Baldwin Composite Score", f"{score:.1f} / 100")
                    c2.info(f"**Strategy:** {strategy}")
                    st.markdown("---")

                    st.subheader("Component Breakdown")
                    st.dataframe(pd.DataFrame(display_data['component_summary']), use_container_width=True, hide_index=True)

                    detailed_breakdown = display_data.get('detailed_breakdown', {})
                    mom_tab, liq_tab, sen_tab = st.tabs(["Momentum Details", "Liquidity & Credit", "Sentiment & Entry"])
                    
                    with mom_tab:
                        if 'Momentum' in detailed_breakdown:
                            details = detailed_breakdown['Momentum']['details']
                            st.subheader("Momentum Synthesis")
                            c1, c2 = st.columns(2)
                            with c1:
                                spy_details = details['Broad Market (SPY)']
                                st.metric("Synthesized SPY Score", f"{spy_details['score']:.1f}")
                                st.progress(spy_details['trend']['score'] / 100, text=f"Trend Strength: {spy_details['trend']['score']:.1f}")
                                st.progress(spy_details['breakout']['score'] / 100, text=f"Breakout Score: {spy_details['breakout']['score']:.1f} ({spy_details['breakout']['status']})")
                                st.progress(spy_details['roc']['score'] / 100, text=f"ROC Score: {spy_details['roc']['score']:.1f} ({spy_details['roc']['roc_pct']:.2f}%)")
                            with c2:
                                iwm_details = details['Market Internals (IWM)']
                                fear_details = details['Leverage & Fear']
                                st.metric("Market Internals (IWM) Score", f"{iwm_details['score']:.1f}")
                                st.caption(f"IWM Trend Strength: {iwm_details['trend']['score']:.1f}")
                                st.metric("Leverage & Fear Score", f"{fear_details['score']:.1f}")
                                st.caption(f"VIX: {fear_details['vix']:.2f}")

                    with liq_tab:
                        if 'Liquidity_Credit' in detailed_breakdown:
                            details = detailed_breakdown['Liquidity_Credit']['details']
                            st.subheader("Liquidity & Credit Synthesis")
                            c1, c2 = st.columns(2)
                            with c1:
                                fs_details = details['Flight-to-Safety']
                                st.metric("Flight-to-Safety Score", f"{fs_details['score']:.1f}")
                                st.progress(fs_details['uup_strength']['score'] / 100, text=f"Dollar Strength: {fs_details['uup_strength']['score']:.1f}")
                                st.progress(fs_details['tlt_strength']['score'] / 100, text=f"Bond Strength (Risk-Off): {fs_details['tlt_strength']['score']:.1f}")
                            with c2:
                                cs_details = details['Credit Spreads']
                                st.metric("Credit Spreads Score", f"{cs_details['score']:.1f}")
                                status = "Improving" if cs_details['ratio'] > cs_details['ema'] else "Worsening"
                                st.caption(f"HYG/LQD Ratio: {cs_details['ratio']} ({status})")
                    
                    with sen_tab:
                        if 'Sentiment_Entry' in detailed_breakdown:
                            details = detailed_breakdown['Sentiment_Entry']['details']
                            st.subheader("Sentiment & Entry Synthesis")
                            c1, c2 = st.columns(2)
                            with c1:
                                se_details = details['Sentiment ETFs']
                                st.metric("Sentiment ETF Score", f"{se_details['score']:.1f}")
                                st.progress(se_details['insider_avg'] / 100, f"Insider ETF Avg: {se_details['insider_avg']:.1f}")
                                st.progress(se_details['political_avg'] / 100, f"Political ETF Avg: {se_details['political_avg']:.1f}")
                            with c2:
                                ec_details = details['Entry Confirmation']
                                st.metric("Entry Confirmation", "✅ Confirmed" if ec_details['confirmed'] else "⏳ Awaiting")
                                st.caption(f"Sentiment Signal: {'Active' if ec_details['active'] else 'Inactive'}")
                                st.caption(f"Trigger Ticker: {ec_details['ticker']}")
                
                elif 'error' in baldwin_results:
                    st.error(f"Error calculating Baldwin Indicator: {baldwin_results['error']}")
            except Exception as e:
                st.error(f"A critical error occurred while displaying the Baldwin Indicator: {e}")

def main():
    create_header()
    controls = create_sidebar_controls()
    
    if controls['analyze_button'] and controls['symbol']:
        st.write(f"### 📊 Full Analysis for {controls['symbol']}")
        with st.spinner(f"Running VWV analysis for {controls['symbol']}..."):
            analysis_results, chart_data = perform_enhanced_analysis(controls['symbol'], controls['period'], controls['show_debug'])
            if analysis_results:
                # Call all the other show_... functions first
                # show_interactive_charts(chart_data, analysis_results, controls['show_debug'])
                # show_individual_technical_analysis(analysis_results, controls['show_debug'])
                # ... etc ...
                
                # Then show the Baldwin Indicator
                show_baldwin_indicator_analysis(show_debug=controls['show_debug'])
    else:
        st.write("## 🚀 VWV Professional Trading System")
        st.info("Enter a symbol in the sidebar to begin analysis.")
        # DIAGNOSTIC STEP: The call to show_baldwin_indicator_analysis on the home page is temporarily disabled.
        # with st.expander("🚦 Live Baldwin Market Regime Preview", expanded=True):
        #     show_baldwin_indicator_analysis(show_debug=controls['show_debug'])

    st.markdown("---")
    st.write("VWV Professional v4.4.2 (Diagnostic Build)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
