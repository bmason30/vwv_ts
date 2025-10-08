"""
File: app.py v1.0.7
VWV Professional Trading System v4.2.2
Main application with UI improvements
Created: 2025-08-15
Updated: 2025-10-08
File Version: v1.0.7 - UI fixes: button position, default period, Enter key, Quick Links container
System Version: v4.2.2 - Advanced Options with Fibonacci Integration
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import traceback

# Import our modular components
from config.settings import DEFAULT_VWV_CONFIG, UI_SETTINGS, PARAMETER_RANGES
from config.constants import SYMBOL_DESCRIPTIONS, QUICK_LINK_CATEGORIES, MAJOR_INDICES
from data.manager import get_data_manager
from data.fetcher import get_market_data_enhanced, is_etf

# Analysis imports
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

# UI imports
from ui.components import (
    create_technical_score_bar,
    create_header
)
from utils.helpers import format_large_number, get_market_status, get_etf_description

# Optional module imports
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

try:
    from charts.plotting import display_trading_charts
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

# Page configuration
st.set_page_config(
    page_title="VWV Trading Analysis v4.2.2",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_sidebar_controls():
    """
    Create sidebar controls and return analysis parameters
    UI improvements: Button moved, default period 1mo, Enter key fixed, Quick Links contained
    """
    st.sidebar.title("📊 Trading Analysis v4.2.2")
    
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
    if 'show_baldwin_indicator' not in st.session_state:
        st.session_state.show_baldwin_indicator = True
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
    if 'last_symbol' not in st.session_state:
        st.session_state.last_symbol = ""
    
    # Handle selected symbol from quicklinks/recents
    if 'selected_symbol' in st.session_state:
        current_symbol = st.session_state.selected_symbol
        st.session_state.auto_analyze = True
        del st.session_state.selected_symbol
    else:
        current_symbol = UI_SETTINGS['default_symbol']
    
    # Symbol input - track changes for Enter key detection
    symbol = st.sidebar.text_input(
        "Symbol", 
        value=current_symbol, 
        help="Enter stock symbol and press Enter to analyze",
        key="symbol_input"
    ).upper()
    
    # FIXED: Default period set to 1 month (index 0)
    period_options = ['1mo', '3mo', '6mo', '1y', '2y']
    period = st.sidebar.selectbox("Data Period", period_options, index=0)  # Index 0 = '1mo'
    
    # Analysis Sections Control Panel
    with st.sidebar.expander("📋 Analysis Sections", expanded=False):
        st.write("**Toggle Analysis Sections:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.show_charts = st.checkbox(
                "📊 Interactive Charts",
                value=st.session_state.show_charts,
                key="toggle_charts"
            )
            st.session_state.show_technical_analysis = st.checkbox(
                "Technical Analysis",
                value=st.session_state.show_technical_analysis,
                key="toggle_technical"
            )
            if VOLUME_ANALYSIS_AVAILABLE:
                st.session_state.show_volume_analysis = st.checkbox(
                    "Volume Analysis",
                    value=st.session_state.show_volume_analysis,
                    key="toggle_volume"
                )
            if VOLATILITY_ANALYSIS_AVAILABLE:
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
            if BALDWIN_INDICATOR_AVAILABLE:
                st.session_state.show_baldwin_indicator = st.checkbox(
                    "🚦 Baldwin Indicator",
                    value=st.session_state.show_baldwin_indicator,
                    key="toggle_baldwin"
                )
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
    
    # MOVED: Analyze button RIGHT AFTER Analysis Sections expander
    analyze_button = st.sidebar.button(
        "📈 Analyze Now",
        type="primary",
        use_container_width=True,
        key="analyze_button"
    )
    
    # FIXED: Enter key detection - check if symbol changed
    if symbol != st.session_state.last_symbol and symbol != "" and len(symbol) > 0:
        st.session_state.last_symbol = symbol
        st.session_state.auto_analyze = True
        analyze_button = True  # Trigger analysis
    
    # Check for auto-analyze trigger from quicklinks/recents
    if st.session_state.auto_analyze:
        st.session_state.auto_analyze = False
        analyze_button = True
    
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
                            if st.button(
                                f"{recent_symbol}",
                                key=f"recent_{recent_symbol}_{symbol_idx}",
                                use_container_width=True
                            ):
                                st.session_state.selected_symbol = recent_symbol
                                st.rerun()
    
    # FIXED: Quick Links - all category expanders inside one containing expander
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
                                if st.button(
                                    sym,
                                    help=SYMBOL_DESCRIPTIONS.get(sym, f"{sym} - Financial Symbol"),
                                    key=f"quick_link_{sym}",
                                    use_container_width=True
                                ):
                                    st.session_state.selected_symbol = sym
                                    st.rerun()
    
    # Debug toggle at bottom
    show_debug = st.sidebar.checkbox("🐛 Debug Mode", value=False)
    
    return {
        'symbol': symbol,
        'period': period,
        'analyze_button': analyze_button,
        'show_debug': show_debug
    }

def add_to_recently_viewed(symbol):
    """Add symbol to recently viewed - maintains 9 symbols"""
    if symbol and symbol != "":
        if symbol in st.session_state.recently_viewed:
            st.session_state.recently_viewed.remove(symbol)
        st.session_state.recently_viewed.insert(0, symbol)
        st.session_state.recently_viewed = st.session_state.recently_viewed[:9]

def show_interactive_charts(data, analysis_results, show_debug=False):
    """Display interactive charts section"""
    if not st.session_state.show_charts:
        return
    
    with st.expander("📊 Interactive Trading Charts", expanded=True):
        try:
            if CHARTS_AVAILABLE:
                display_trading_charts(data, analysis_results)
            else:
                st.error("📊 Charts module not available")
                if show_debug:
                    st.info("Import the charts.plotting module to enable interactive charts")
                # Fallback
                st.subheader("Basic Price Chart")
                if data is not None and not data.empty:
                    st.line_chart(data['Close'])
        except Exception as e:
            if show_debug:
                st.error(f"Chart error: {str(e)}")
                st.exception(e)
            else:
                st.warning("⚠️ Charts temporarily unavailable")

def show_technical_analysis(analysis_results, show_debug=False):
    """Display technical analysis section"""
    if not st.session_state.show_technical_analysis:
        return
    
    with st.expander("📊 Technical Analysis", expanded=True):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        
        if comprehensive_technicals:
            # Technical score
            score, details = calculate_composite_technical_score(analysis_results)
            create_technical_score_bar(score)
            
            # Display indicators
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("RSI (14)", f"{comprehensive_technicals.get('rsi_14', 0):.1f}")
            with col2:
                st.metric("MFI (14)", f"{comprehensive_technicals.get('mfi_14', 0):.1f}")
            with col3:
                macd = comprehensive_technicals.get('macd', {})
                st.metric("MACD", f"{macd.get('macd', 0):.2f}")
            with col4:
                st.metric("Volume Ratio", f"{comprehensive_technicals.get('volume_ratio', 1):.2f}x")
        else:
            st.warning("Technical analysis data not available")

def show_fundamental_analysis(analysis_results, show_debug=False):
    """Display fundamental analysis section"""
    if not st.session_state.show_fundamental_analysis:
        return
    
    with st.expander("📈 Fundamental Analysis", expanded=False):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        graham_score = enhanced_indicators.get('graham_score', {})
        piotroski_score = enhanced_indicators.get('piotroski_score', {})
        
        if 'error' not in graham_score or 'error' not in piotroski_score:
            col1, col2 = st.columns(2)
            with col1:
                if 'score' in graham_score:
                    st.metric("Graham Score", f"{graham_score['score']}/{graham_score.get('total_possible', 10)}")
            with col2:
                if 'score' in piotroski_score:
                    st.metric("Piotroski Score", f"{piotroski_score['score']}/{piotroski_score.get('total_possible', 9)}")
        else:
            st.info("Fundamental analysis not applicable for this symbol")

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis"""
    if not st.session_state.show_market_correlation:
        return
    
    with st.expander("🌐 Market Correlation & Breakout Analysis", expanded=False):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        market_correlations = enhanced_indicators.get('market_correlations', {})
        
        if market_correlations:
            st.subheader("📊 ETF Correlation Analysis")
            correlation_data = []
            for etf, etf_data in market_correlations.items():
                correlation_data.append({
                    'ETF': etf,
                    'Correlation': f"{etf_data.get('correlation', 0):.3f}",
                    'Beta': f"{etf_data.get('beta', 0):.3f}",
                    'Relationship': etf_data.get('relationship', 'Unknown'),
                    'Description': get_etf_description(etf)
                })
            df_corr = pd.DataFrame(correlation_data)
            st.dataframe(df_corr, use_container_width=True, hide_index=True)
        
        # Breakout/Breakdown Analysis
        st.subheader("📊 Market Breakout/Breakdown Analysis")
        try:
            breakout_data = calculate_breakout_breakdown_analysis(show_debug=show_debug)
            if breakout_data and 'OVERALL' in breakout_data:
                overall = breakout_data['OVERALL']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Market Breakouts", f"{overall['breakout_ratio']}%")
                with col2:
                    st.metric("Market Breakdowns", f"{overall['breakdown_ratio']}%")
                with col3:
                    net = overall['net_ratio']
                    st.metric("Net Bias", f"{net:+.1f}%")
                with col4:
                    st.metric("Regime", overall.get('market_regime', 'Unknown'))
        except Exception as e:
            if show_debug:
                st.error(f"Breakout analysis error: {str(e)}")

def show_options_analysis(analysis_results, show_debug=False):
    """Display options analysis"""
    if not st.session_state.show_options_analysis:
        return
    
    with st.expander("🎯 Options Analysis", expanded=False):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        options_levels = enhanced_indicators.get('options_levels', [])
        
        if options_levels:
            df_options = pd.DataFrame(options_levels)
            st.dataframe(df_options, use_container_width=True, hide_index=True)
        else:
            st.warning("Options analysis data not available")

def show_confidence_intervals(analysis_results, show_debug=False):
    """Display confidence intervals"""
    if not st.session_state.show_confidence_intervals:
        return
    
    with st.expander("📊 Statistical Confidence Intervals", expanded=False):
        confidence_analysis = analysis_results.get('confidence_analysis')
        if confidence_analysis:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Weekly Return", f"{confidence_analysis['mean_weekly_return']:.3f}%")
            with col2:
                st.metric("Weekly Volatility", f"{confidence_analysis['weekly_volatility']:.2f}%")
            with col3:
                st.metric("Sample Size", f"{confidence_analysis['sample_size']} weeks")
            
            intervals_data = []
            for level, level_data in confidence_analysis['confidence_intervals'].items():
                intervals_data.append({
                    'Confidence Level': level,
                    'Upper Bound': f"${level_data['upper_bound']}",
                    'Lower Bound': f"${level_data['lower_bound']}",
                    'Expected Move': f"±{level_data['expected_move_pct']:.2f}%"
                })
            df_intervals = pd.DataFrame(intervals_data)
            st.dataframe(df_intervals, use_container_width=True, hide_index=True)

def perform_enhanced_analysis(symbol, period, show_debug=False):
    """Perform complete analysis"""
    try:
        # Fetch data
        market_data = get_market_data_enhanced(symbol, period, show_debug)
        if market_data is None:
            st.error(f"❌ Could not fetch data for {symbol}")
            return None, None
        
        # Store and prepare data
        data_manager = get_data_manager()
        data_manager.store_market_data(symbol, market_data, show_debug)
        analysis_input = data_manager.get_market_data_for_analysis(symbol)
        
        if analysis_input is None:
            st.error("❌ Could not prepare analysis data")
            return None, None
        
        # Calculate indicators
        daily_vwap = calculate_daily_vwap(analysis_input)
        fibonacci_emas = calculate_fibonacci_emas(analysis_input)
        point_of_control = calculate_point_of_control_enhanced(analysis_input)
        weekly_deviations = calculate_weekly_deviations(analysis_input)
        comprehensive_technicals = calculate_comprehensive_technicals(analysis_input)
        market_correlations = calculate_market_correlations_enhanced(analysis_input, symbol, show_debug=show_debug)
        
        # Fundamental analysis
        is_etf_symbol = is_etf(symbol)
        if is_etf_symbol:
            graham_score = {'error': 'ETF - Not applicable'}
            piotroski_score = {'error': 'ETF - Not applicable'}
        else:
            graham_score = calculate_graham_score(symbol, show_debug)
            piotroski_score = calculate_piotroski_score(symbol, show_debug)
        
        # Options and confidence intervals
        volatility = comprehensive_technicals.get('volatility_20d', 20)
        current_price = round(float(analysis_input['Close'].iloc[-1]), 2)
        options_levels = calculate_options_levels_enhanced(current_price, volatility)
        confidence_analysis = calculate_confidence_intervals(analysis_input)
        
        # Build results
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
        
        data_manager.store_analysis_results(symbol, analysis_results)
        chart_data = data_manager.get_market_data_for_chart(symbol)
        
        return analysis_results, chart_data
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        if show_debug:
            st.exception(e)
        return None, None

def main():
    """Main application function"""
    create_header()
    controls = create_sidebar_controls()
    
    if controls['analyze_button'] and controls['symbol']:
        add_to_recently_viewed(controls['symbol'])
        
        st.write(f"## 📊 VWV Trading Analysis - {controls['symbol']}")
        
        with st.spinner(f"Analyzing {controls['symbol']}..."):
            analysis_results, chart_data = perform_enhanced_analysis(
                controls['symbol'],
                controls['period'],
                controls['show_debug']
            )
            
            if analysis_results and chart_data is not None:
                show_interactive_charts(chart_data, analysis_results, controls['show_debug'])
                show_technical_analysis(analysis_results, controls['show_debug'])
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                show_options_analysis(analysis_results, controls['show_debug'])
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
                if controls['show_debug']:
                    with st.expander("🐛 Debug Information", expanded=False):
                        st.json(analysis_results)
            else:
                st.error("❌ No results to display")
    else:
        st.write("## 🚀 VWV Professional Trading System v4.2.2")
        st.write("Enter a symbol in the sidebar and click 'Analyze Now' to begin.")
        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.code(traceback.format_exc())
