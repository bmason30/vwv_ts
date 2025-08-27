"""
File: app.py
VWV Professional Trading System v4.2.1 - CORRECTED VERSION  
Version: v4.2.1-CORRECTED-SIDEBAR-VOLATILITY-2025-08-27-18-20-00-EST
FIXES: Restored sidebar structure + Fixed volatility integration + Preserved existing functionality
Last Updated: August 27, 2025 - 6:20 PM EST
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
    from analysis.volume import (
        calculate_complete_volume_analysis,
        calculate_market_wide_volume_analysis
    )
    VOLUME_ANALYSIS_AVAILABLE = True
except ImportError:
    VOLUME_ANALYSIS_AVAILABLE = False

try:
    from analysis.volatility import (
        calculate_complete_volatility_analysis,
        calculate_market_wide_volatility_analysis
    )
    VOLATILITY_ANALYSIS_AVAILABLE = True
except ImportError:
    VOLATILITY_ANALYSIS_AVAILABLE = False

# Import the display functions
try:
    from app_volatility_display import show_volatility_analysis, show_volume_analysis
    VOLATILITY_DISPLAY_AVAILABLE = True
except ImportError:
    VOLATILITY_DISPLAY_AVAILABLE = False
    
    # Fallback display functions if import fails
    def show_volatility_analysis(analysis_results, show_debug=False):
        st.warning("Volatility display function not available")
    
    def show_volume_analysis(analysis_results, show_debug=False):
        st.warning("Volume display function not available")

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
    """Create sidebar controls - CORRECTED STRUCTURE WITH PROPER EXPANDERS"""
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

    # CORRECTED: Quick Links PROPERLY GROUPED in single expander
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

    # Recently viewed - PROPERLY GROUPED
    if st.session_state.recently_viewed:
        with st.sidebar.expander("📈 Recently Viewed", expanded=False):
            recent_cols = st.sidebar.columns(2)
            for i, recent_symbol in enumerate(st.session_state.recently_viewed[-6:]):
                col = recent_cols[i % 2]
                with col:
                    if st.button(f"{recent_symbol}", key=f"recent_{recent_symbol}", use_container_width=True):
                        quick_link_clicked = recent_symbol

    # Analysis sections toggle - PROPERLY GROUPED
    with st.sidebar.expander("⚙️ Analysis Sections", expanded=False):
        st.session_state.show_technical_analysis = st.checkbox(
            "📊 Technical Analysis", 
            value=st.session_state.show_technical_analysis
        )
        
        if VOLUME_ANALYSIS_AVAILABLE:
            st.session_state.show_volume_analysis = st.checkbox(
                "🔊 Volume Analysis", 
                value=st.session_state.show_volume_analysis
            )
        
        if VOLATILITY_ANALYSIS_AVAILABLE:
            st.session_state.show_volatility_analysis = st.checkbox(
                "🌡️ Volatility Analysis", 
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
    show_debug = st.sidebar.checkbox("🐛 Debug Mode", value=False)

    # Market status
    market_status = get_market_status()
    if market_status:
        st.sidebar.info(f"🏛️ Market: {market_status}")

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
    """Display interactive charts - MANDATORY FIRST POSITION"""
    
    st.subheader("📊 Interactive Charts")
    
    if chart_data is not None and len(chart_data) > 0:
        try:
            # Basic chart display - can be enhanced later
            st.line_chart(chart_data['Close'])
            
            if show_debug:
                st.write(f"📊 Chart data points: {len(chart_data)}")
                st.write(f"📅 Date range: {chart_data.index[0]} to {chart_data.index[-1]}")
                
        except Exception as e:
            st.error(f"❌ Chart display failed: {e}")
            if show_debug:
                st.exception(e)
    else:
        st.warning("⚠️ No chart data available")

def show_individual_technical_analysis(analysis_results, show_debug=False):
    """Display individual technical analysis section - MANDATORY SECOND POSITION"""
    if not st.session_state.show_technical_analysis:
        return
        
    symbol = analysis_results.get('symbol', 'Unknown')
    
    with st.expander(f"📊 {symbol} - Individual Technical Analysis", expanded=True):
        
        # COMPOSITE TECHNICAL SCORE - Use modular component
        try:
            composite_score, score_details = calculate_composite_technical_score(analysis_results)
            create_technical_score_bar(composite_score, score_details)
            
            # Technical metrics display
            enhanced_indicators = analysis_results.get('enhanced_indicators', {})
            comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
            
            if comprehensive_technicals:
                # Display key technical metrics
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
            if show_debug:
                st.exception(e)

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

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis section"""
    if not st.session_state.show_market_correlation:
        return
        
    symbol = analysis_results.get('symbol', 'Unknown')
    
    with st.expander(f"🌐 {symbol} - Market Correlation Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        market_correlations = enhanced_indicators.get('market_correlations', {})
        
        if market_correlations and 'error' not in market_correlations:
            correlations = market_correlations.get('correlations', {})
            
            if correlations:
                st.subheader("📊 ETF Correlations")
                
                corr_data = []
                for etf, corr_value in correlations.items():
                    if isinstance(corr_value, (int, float)):
                        strength = "Strong" if abs(corr_value) > 0.7 else "Moderate" if abs(corr_value) > 0.4 else "Weak"
                        direction = "Positive" if corr_value > 0 else "Negative"
                        corr_data.append({
                            'ETF': etf,
                            'Correlation': f"{corr_value:.3f}",
                            'Strength': strength,
                            'Direction': direction
                        })
                
                if corr_data:
                    df_corr = pd.DataFrame(corr_data)
                    st.dataframe(df_corr, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Market correlation analysis not available - insufficient data")

def show_options_analysis(analysis_results, show_debug=False):
    """Display options analysis section"""
    if not st.session_state.show_options_analysis:
        return
        
    symbol = analysis_results.get('symbol', 'Unknown')
    
    with st.expander(f"🎯 {symbol} - Options Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        options_levels = enhanced_indicators.get('options_levels', {})
        
        if options_levels and 'error' not in options_levels:
            # Display key options metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_price = options_levels.get('current_price', 0)
                st.metric("Current Price", f"${current_price:.2f}")
            
            with col2:
                support_level = options_levels.get('support_level', 0)
                st.metric("Support Level", f"${support_level:.2f}")
            
            with col3:
                resistance_level = options_levels.get('resistance_level', 0)  
                st.metric("Resistance Level", f"${resistance_level:.2f}")
                
        else:
            st.warning("⚠️ Options analysis not available - insufficient data")

def show_confidence_intervals(analysis_results, show_debug=False):
    """Display confidence intervals section"""
    if not st.session_state.show_confidence_intervals:
        return
        
    symbol = analysis_results.get('symbol', 'Unknown')
    
    with st.expander(f"📊 {symbol} - Statistical Confidence Intervals", expanded=True):
        
        confidence_analysis = analysis_results.get('confidence_analysis', {})
        
        if confidence_analysis and 'error' not in confidence_analysis:
            weekly_projections = confidence_analysis.get('weekly_volatility_projection', {})
            
            if weekly_projections:
                st.subheader("📊 Weekly Price Projections")
                
                current_price = weekly_projections.get('current_price', 0)
                lower_bound = weekly_projections.get('lower_bound_68', 0)
                upper_bound = weekly_projections.get('upper_bound_68', 0)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                
                with col2:
                    st.metric("Lower Bound (68%)", f"${lower_bound:.2f}")
                
                with col3:
                    st.metric("Upper Bound (68%)", f"${upper_bound:.2f}")
                    
        else:
            st.warning("⚠️ Confidence interval analysis not available - insufficient data")

@safe_calculation_wrapper
def perform_enhanced_analysis(symbol, period, show_debug=False):
    """Perform comprehensive analysis with VOLATILITY INTEGRATION"""
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
        
        # VOLUME ANALYSIS - SAFE INTEGRATION
        if VOLUME_ANALYSIS_AVAILABLE:
            try:
                volume_analysis = calculate_complete_volume_analysis(data)
                enhanced_indicators['volume_analysis'] = volume_analysis
                if show_debug:
                    st.write(f"✅ Volume analysis completed for {symbol}")
            except Exception as e:
                enhanced_indicators['volume_analysis'] = {'error': f'Volume analysis failed: {e}'}
                if show_debug:
                    st.error(f"❌ Volume analysis failed: {e}")
        
        # VOLATILITY ANALYSIS - SAFE INTEGRATION  
        if VOLATILITY_ANALYSIS_AVAILABLE:
            try:
                volatility_analysis = calculate_complete_volatility_analysis(data)
                enhanced_indicators['volatility_analysis'] = volatility_analysis
                if show_debug:
                    st.write(f"✅ Volatility analysis completed for {symbol}")
                    st.write(f"✅ Volatility display available: {VOLATILITY_DISPLAY_AVAILABLE}")
            except Exception as e:
                enhanced_indicators['volatility_analysis'] = {'error': f'Volatility analysis failed: {e}'}
                if show_debug:
                    st.error(f"❌ Volatility analysis failed: {e}")
        
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
            'system_status': 'OPERATIONAL v4.2.1 - CORRECTED'
        }
        
        # Store results
        data_manager.store_analysis_results(symbol, analysis_results)
        
        # Get chart data
        chart_data = data_manager.get_market_data_for_chart(symbol)
        
        return analysis_results, chart_data
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        return None, None

def main():
    """Main application function - CORRECTED VERSION"""
    # Create header using modular component
    create_header()
    
    # Create sidebar and get controls
    controls = create_sidebar_controls()
    
    # Main logic flow
    if controls['analyze_button'] and controls['symbol']:
        # Add symbol to recently viewed
        add_to_recently_viewed(controls['symbol'])
        
        st.write("## 📊 VWV Trading Analysis v4.2.1 - Corrected")
        
        with st.spinner(f"Analyzing {controls['symbol']}..."):
            
            # Perform analysis using modular components
            analysis_results, chart_data = perform_enhanced_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results and chart_data is not None:
                
                # CORRECTED DISPLAY ORDER:
                
                # 1. CHARTS FIRST (MANDATORY)
                show_interactive_charts(chart_data, analysis_results, controls['show_debug'])
                
                # 2. INDIVIDUAL TECHNICAL ANALYSIS SECOND (MANDATORY)
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                
                # 3. VOLUME ANALYSIS (If available and enabled)
                if VOLUME_ANALYSIS_AVAILABLE and VOLATILITY_DISPLAY_AVAILABLE:
                    show_volume_analysis(analysis_results, controls['show_debug'])
                
                # 4. VOLATILITY ANALYSIS (If available and enabled)
                if VOLATILITY_ANALYSIS_AVAILABLE and VOLATILITY_DISPLAY_AVAILABLE:
                    show_volatility_analysis(analysis_results, controls['show_debug'])
                
                # 5. Fundamental Analysis
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                
                # 6. Market Correlation Analysis
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                
                # 7. Options Analysis
                show_options_analysis(analysis_results, controls['show_debug'])
                
                # 8. Confidence Intervals
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
                # Debug information
                if controls['show_debug']:
                    with st.expander("🐛 Debug Information", expanded=False):
                        st.write("### System Status")
                        st.write(f"**Volume Analysis Available:** {VOLUME_ANALYSIS_AVAILABLE}")
                        st.write(f"**Volatility Analysis Available:** {VOLATILITY_ANALYSIS_AVAILABLE}")
                        st.write(f"**Volatility Display Available:** {VOLATILITY_DISPLAY_AVAILABLE}")
                        
                        st.write("### Analysis Results Structure")
                        st.json(analysis_results, expanded=False)
    
    else:
        # Welcome message
        st.write("## 🚀 VWV Professional Trading System v4.2.1 - Corrected")
        st.write("**STATUS:** Sidebar fixed + Volatility integration + Preserved functionality")
        
        # Status indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**📊 Technical Analysis**\nWorking with corrected gradient bars")
            
        with col2:
            if VOLATILITY_ANALYSIS_AVAILABLE:
                st.success("**🌡️ Volatility Analysis**\nModule available and integrated")
            else:
                st.warning("**🌡️ Volatility Analysis**\nModule not available")
                
        with col3:
            if VOLUME_ANALYSIS_AVAILABLE:
                st.success("**🔊 Volume Analysis**\nModule available")
            else:
                st.warning("**🔊 Volume Analysis**\nModule not available")
        
        st.write("**FIXES APPLIED:**")
        st.write("✅ Sidebar quick links properly grouped in expanders")  
        st.write("✅ HTML rendering issues resolved in gradient bars")
        st.write("✅ Volatility analysis integration with safe fallbacks")
        st.write("✅ Preserved all existing functionality")
        
        st.write("Select a symbol from the sidebar or use Quick Links to begin analysis.")

if __name__ == "__main__":
    main()
