"""
VWV Professional Trading System v4.2.1 - PATCHED
Fixed chart function signature and volume/volatility display logic
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import traceback

# Import modular components
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

# Baldwin Indicator import with safe fallback
try:
    from analysis.baldwin_indicator import calculate_baldwin_indicator_complete, format_baldwin_for_display
    BALDWIN_INDICATOR_AVAILABLE = True
except ImportError:
    BALDWIN_INDICATOR_AVAILABLE = False

from ui.components import create_technical_score_bar, create_header
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

# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
def create_sidebar_controls():
    """Create sidebar controls and return analysis parameters"""
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
    
    # Symbol input
    symbol = st.sidebar.text_input("Symbol", value="AAPL", help="Enter stock symbol").upper()
    
    # Period selection - DEFAULT TO 1 MONTH
    period_options = ['1mo', '3mo', '6mo', '1y', '2y']
    period = st.sidebar.selectbox("Data Period", period_options, index=0)
    
    # Analyze button
    analyze_button = st.sidebar.button("📊 Analyze Symbol", type="primary", use_container_width=True)
    
    # Quick Links section
    with st.sidebar.expander("🔗 Quick Links", expanded=False):
        st.write("**Popular Symbols by Category**")
        
        for category, symbols in QUICK_LINK_CATEGORIES.items():
            with st.expander(f"{category}", expanded=False):
                for i in range(0, len(symbols), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(symbols):
                            sym = symbols[i + j]
                            with col:
                                if st.button(sym, key=f"quick_{sym}", use_container_width=True):
                                    st.session_state.selected_symbol = sym
                                    st.rerun()
    
    # Recently Viewed
    if len(st.session_state.recently_viewed) > 0:
        with st.sidebar.expander("🕒 Recently Viewed", expanded=False):
            recent_symbols = st.session_state.recently_viewed[:9]
            for row in range(0, len(recent_symbols), 3):
                cols = st.columns(3)
                for col_idx, col in enumerate(cols):
                    symbol_idx = row + col_idx
                    if symbol_idx < len(recent_symbols):
                        recent_symbol = recent_symbols[symbol_idx]
                        with col:
                            if st.button(recent_symbol, key=f"recent_{recent_symbol}_{symbol_idx}", use_container_width=True):
                                st.session_state.selected_symbol = recent_symbol
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
    """Add symbol to recently viewed list"""
    if symbol and symbol != "":
        if symbol in st.session_state.recently_viewed:
            st.session_state.recently_viewed.remove(symbol)
        st.session_state.recently_viewed.insert(0, symbol)
        st.session_state.recently_viewed = st.session_state.recently_viewed[:9]

# =============================================================================
# ANALYSIS FUNCTION
# =============================================================================
def perform_enhanced_analysis(symbol, period, show_debug=False):
    """Perform enhanced analysis using modular components"""
    try:
        # Step 1: Fetch data
        if show_debug:
            st.write(f"📡 Fetching {period} of data for {symbol}...")
        
        market_data = get_market_data_enhanced(symbol, period, show_debug)
        
        if market_data is None:
            st.error(f"❌ Could not fetch data for {symbol}")
            return None, None
        
        if show_debug:
            st.success(f"✅ Fetched {len(market_data)} days of data")
        
        # Step 2: Store data
        data_manager = get_data_manager()
        data_manager.store_market_data(symbol, market_data, show_debug)
        
        # Step 3: Get analysis copy
        analysis_input = data_manager.get_market_data_for_analysis(symbol)
        
        if analysis_input is None:
            st.error("❌ Could not prepare analysis data")
            return None, None
        
        # Step 4: Calculate indicators
        if show_debug:
            st.write("⚙️ Calculating technical indicators...")
        
        daily_vwap = calculate_daily_vwap(analysis_input)
        fibonacci_emas = calculate_fibonacci_emas(analysis_input)
        point_of_control = calculate_point_of_control_enhanced(analysis_input)
        weekly_deviations = calculate_weekly_deviations(analysis_input)
        comprehensive_technicals = calculate_comprehensive_technicals(analysis_input)
        
        # Step 5: Volume Analysis (if available)
        volume_analysis = {}
        if VOLUME_ANALYSIS_AVAILABLE:
            try:
                volume_analysis = calculate_complete_volume_analysis(analysis_input)
                if show_debug:
                    st.write("✅ Volume analysis completed")
            except Exception as e:
                if show_debug:
                    st.write(f"❌ Volume analysis failed: {e}")
                volume_analysis = {'error': str(e)}
        
        # Step 6: Volatility Analysis (if available)
        volatility_analysis = {}
        if VOLATILITY_ANALYSIS_AVAILABLE:
            try:
                volatility_analysis = calculate_complete_volatility_analysis(analysis_input)
                if show_debug:
                    st.write("✅ Volatility analysis completed")
            except Exception as e:
                if show_debug:
                    st.write(f"❌ Volatility analysis failed: {e}")
                volatility_analysis = {'error': str(e)}
        
        # Step 7: Market correlations
        if show_debug:
            st.write("⚙️ Calculating market correlations...")
        market_correlations = calculate_market_correlations_enhanced(analysis_input, symbol, show_debug=show_debug)
        
        # Step 8: Fundamental analysis (skip for ETFs)
        is_etf_symbol = is_etf(symbol)
        
        if is_etf_symbol:
            graham_score = {'score': 0, 'total_possible': 10, 'criteria': [], 'error': 'ETF - not applicable'}
            piotroski_score = {'score': 0, 'total_possible': 9, 'criteria': [], 'error': 'ETF - not applicable'}
        else:
            if show_debug:
                st.write("⚙️ Calculating fundamental scores...")
            graham_score = calculate_graham_score(symbol, show_debug)
            piotroski_score = calculate_piotroski_score(symbol, show_debug)
        
        # Step 9: Options levels
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
        
        # Step 10: Confidence intervals
        confidence_analysis = calculate_confidence_intervals(analysis_input)
        
        # Step 11: Build results
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
                'volume_analysis': volume_analysis,
                'volatility_analysis': volatility_analysis,
                'market_correlations': market_correlations,
                'options_levels': options_levels,
                'graham_score': graham_score,
                'piotroski_score': piotroski_score
            },
            'confidence_analysis': confidence_analysis,
            'system_status': 'OPERATIONAL v4.2.1'
        }
        
        # Store results
        data_manager.store_analysis_results(symbol, analysis_results)
        
        # Get chart data
        chart_data = data_manager.get_market_data_for_chart(symbol)
        
        if show_debug:
            st.success("✅ Analysis complete!")
        
        return analysis_results, chart_data
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        if show_debug:
            st.code(traceback.format_exc())
        return None, None

# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================
def show_interactive_charts(chart_data, analysis_results, show_debug=False):
    """Display interactive charts - PRIORITY 1: MUST BE FIRST"""
    if not st.session_state.show_charts:
        return
    
    try:
        from charts.plotting import display_trading_charts
        # FIXED: Only pass 2 arguments (removed show_debug)
        display_trading_charts(chart_data, analysis_results)
    except Exception as e:
        st.error(f"❌ Charts failed: {e}")
        if show_debug:
            st.code(traceback.format_exc())
        # Fallback simple chart
        st.subheader("Basic Price Chart (Fallback)")
        if chart_data is not None and not chart_data.empty:
            st.line_chart(chart_data['Close'])

def show_individual_technical_analysis(analysis_results, show_debug=False):
    """Display technical analysis - PRIORITY 2: MUST BE SECOND"""
    if not st.session_state.show_technical_analysis:
        return
    
    with st.expander(f"📊 {analysis_results['symbol']} - Technical Analysis", expanded=True):
        # Composite score
        composite_score, score_details = calculate_composite_technical_score(analysis_results)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Technical Composite Score")
        with col2:
            if composite_score >= 70:
                st.success(f"**{composite_score:.1f}/100** - Bullish")
            elif composite_score >= 50:
                st.info(f"**{composite_score:.1f}/100** - Neutral")
            else:
                st.warning(f"**{composite_score:.1f}/100** - Bearish")
        
        # Key indicators
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            rsi = comprehensive_technicals.get('rsi_14', 50)
            st.metric("RSI (14)", f"{rsi:.2f}")
        with col2:
            mfi = comprehensive_technicals.get('mfi_14', 50)
            st.metric("MFI (14)", f"{mfi:.2f}")
        with col3:
            stoch = comprehensive_technicals.get('stochastic', {})
            st.metric("Stochastic %K", f"{stoch.get('k', 50):.2f}")
        with col4:
            williams_r = comprehensive_technicals.get('williams_r', -50)
            st.metric("Williams %R", f"{williams_r:.2f}")

def show_volume_analysis(analysis_results, show_debug=False):
    """Display volume analysis"""
    if not st.session_state.show_volume_analysis or not VOLUME_ANALYSIS_AVAILABLE:
        return
    
    with st.expander(f"📊 {analysis_results['symbol']} - Volume Analysis", expanded=True):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volume_analysis = enhanced_indicators.get('volume_analysis', {})
        
        # FIXED: Check for empty dict first, then check for error key
        if volume_analysis and 'error' not in volume_analysis:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Volume", format_large_number(volume_analysis.get('current_volume', 0)))
            with col2:
                st.metric("5D Avg", format_large_number(volume_analysis.get('volume_5d_avg', 0)))
            with col3:
                ratio = volume_analysis.get('volume_ratio', 1.0)
                st.metric("Ratio", f"{ratio:.2f}x")
            
            # Volume regime info
            regime = volume_analysis.get('volume_regime', 'Unknown')
            st.info(f"**Volume Regime:** {regime}")
        else:
            st.warning("⚠️ Volume analysis not available")

def show_volatility_analysis(analysis_results, show_debug=False):
    """Display volatility analysis"""
    if not st.session_state.show_volatility_analysis or not VOLATILITY_ANALYSIS_AVAILABLE:
        return
    
    with st.expander(f"📊 {analysis_results['symbol']} - Volatility Analysis", expanded=True):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volatility_analysis = enhanced_indicators.get('volatility_analysis', {})
        
        # FIXED: Check for empty dict first, then check for error key
        if volatility_analysis and 'error' not in volatility_analysis:
            col1, col2, col3 = st.columns(3)
            with col1:
                vol_5d = volatility_analysis.get('volatility_5d', 0)
                st.metric("5D Volatility", f"{vol_5d:.2f}%")
            with col2:
                vol_30d = volatility_analysis.get('volatility_30d', 0)
                st.metric("30D Volatility", f"{vol_30d:.2f}%")
            with col3:
                regime = volatility_analysis.get('volatility_regime', 'Unknown')
                st.metric("Regime", regime)
            
            # Volatility score
            vol_score = volatility_analysis.get('volatility_score', 50)
            st.info(f"**Volatility Score:** {vol_score}/100")
        else:
            st.warning("⚠️ Volatility analysis not available")

def show_fundamental_analysis(analysis_results, show_debug=False):
    """Display fundamental analysis"""
    if not st.session_state.show_fundamental_analysis:
        return
    
    with st.expander(f"📊 {analysis_results['symbol']} - Fundamental Analysis", expanded=True):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        graham_score = enhanced_indicators.get('graham_score', {})
        piotroski_score = enhanced_indicators.get('piotroski_score', {})
        
        if 'error' in graham_score:
            st.info(f"ℹ️ {graham_score['error']}")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Graham Score", f"{graham_score.get('score', 0)}/{graham_score.get('total_possible', 10)}")
            with col2:
                st.metric("Piotroski Score", f"{piotroski_score.get('score', 0)}/{piotroski_score.get('total_possible', 9)}")

def show_baldwin_indicator_analysis(show_debug=False):
    """Display Baldwin market regime indicator"""
    if not st.session_state.show_baldwin_indicator or not BALDWIN_INDICATOR_AVAILABLE:
        return
    
    with st.expander("🚦 Baldwin Market Regime Indicator", expanded=True):
        with st.spinner("Calculating Baldwin indicator..."):
            try:
                baldwin_results = calculate_baldwin_indicator_complete()
                if baldwin_results and 'error' not in baldwin_results:
                    regime = baldwin_results.get('overall_regime', 'Unknown')
                    score = baldwin_results.get('overall_score', 50)
                    
                    st.subheader(f"Market Regime: {regime}")
                    st.metric("Baldwin Score", f"{score}/100")
                else:
                    st.warning("⚠️ Baldwin indicator not available")
            except Exception as e:
                st.error(f"❌ Baldwin calculation failed: {e}")

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis"""
    if not st.session_state.show_market_correlation:
        return
    
    with st.expander("🌐 Market Correlation Analysis", expanded=True):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        market_correlations = enhanced_indicators.get('market_correlations', {})
        
        if market_correlations:
            correlation_data = []
            for etf, etf_data in market_correlations.items():
                correlation_data.append({
                    'ETF': etf,
                    'Correlation': f"{etf_data.get('correlation', 0):.3f}",
                    'Beta': f"{etf_data.get('beta', 0):.3f}",
                    'Relationship': etf_data.get('relationship', 'Unknown')
                })
            
            df_corr = pd.DataFrame(correlation_data)
            st.dataframe(df_corr, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Market correlation data not available")

def show_options_analysis(analysis_results, show_debug=False):
    """Display options analysis"""
    if not st.session_state.show_options_analysis:
        return
    
    with st.expander("🎯 Options Analysis", expanded=True):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        options_levels = enhanced_indicators.get('options_levels', {})
        
        if options_levels:
            st.subheader("Key Options Levels")
            st.json(options_levels)
        else:
            st.warning("⚠️ Options analysis not available")

def show_confidence_intervals(analysis_results, show_debug=False):
    """Display confidence intervals"""
    if not st.session_state.show_confidence_intervals:
        return
    
    with st.expander("📊 Statistical Confidence Intervals", expanded=True):
        confidence_analysis = analysis_results.get('confidence_analysis')
        if confidence_analysis:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Weekly Return", f"{confidence_analysis.get('mean_weekly_return', 0):.3f}%")
            with col2:
                st.metric("Weekly Volatility", f"{confidence_analysis.get('weekly_volatility', 0):.2f}%")
        else:
            st.warning("⚠️ Confidence intervals not available")

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Main application function"""
    # Create header
    create_header()
    
    # Create sidebar and get controls
    controls = create_sidebar_controls()
    
    # Main logic flow
    if controls['analyze_button'] and controls['symbol']:
        # Add to recently viewed
        add_to_recently_viewed(controls['symbol'])
        
        st.write("## 📊 VWV Trading Analysis v4.2.1")
        
        with st.spinner(f"Analyzing {controls['symbol']}..."):
            # Perform analysis
            analysis_results, chart_data = perform_enhanced_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results and chart_data is not None:
                # DISPLAY ORDER - MANDATORY SEQUENCE:
                
                # 1. CHARTS FIRST
                show_interactive_charts(chart_data, analysis_results, controls['show_debug'])
                
                # 2. TECHNICAL ANALYSIS SECOND
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                
                # 3. VOLUME ANALYSIS (if available)
                if VOLUME_ANALYSIS_AVAILABLE:
                    show_volume_analysis(analysis_results, controls['show_debug'])
                
                # 4. VOLATILITY ANALYSIS (if available)
                if VOLATILITY_ANALYSIS_AVAILABLE:
                    show_volatility_analysis(analysis_results, controls['show_debug'])
                
                # 5. FUNDAMENTAL ANALYSIS
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                
                # 6. BALDWIN INDICATOR (if available)
                if BALDWIN_INDICATOR_AVAILABLE:
                    show_baldwin_indicator_analysis(controls['show_debug'])
                
                # 7. MARKET CORRELATION
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                
                # 8. OPTIONS ANALYSIS
                show_options_analysis(analysis_results, controls['show_debug'])
                
                # 9. CONFIDENCE INTERVALS
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
                # Debug info
                if controls['show_debug']:
                    with st.expander("🐛 Debug Information", expanded=False):
                        st.json(analysis_results)
            else:
                st.error("❌ No results to display")
    
    else:
        # Welcome screen
        st.write("## 🚀 VWV Professional Trading System v4.2.1")
        st.write("Enter a symbol in the sidebar and click 'Analyze Symbol' to begin.")
        
        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.code(traceback.format_exc())
