"""
VWV Professional Trading System v4.2.1 - Complete Application
Date: August 17, 2025 - 1:00 AM EST  
Enhancement: Charts First + Technical Second + Baldwin Integration + Function Signature Fix
Status: SURGICAL FIX - Confidence Intervals Function Call Corrected
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
from analysis.options_advanced import (
    calculate_complete_advanced_options,
    format_advanced_options_for_display
)
from analysis.vwv_core import (
    calculate_vwv_system_complete,
    get_vwv_signal_interpretation
)
from ui.components import (
    create_technical_score_bar,
    create_header
)
from charts.plotting import display_trading_charts
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

def create_sidebar_controls():
    """Create sidebar controls and return analysis parameters"""
    st.sidebar.title("📊 Trading Analysis")
    
    # Initialize session state
    if 'recently_viewed' not in st.session_state:
        st.session_state.recently_viewed = []
    if 'show_vwv_analysis' not in st.session_state:
        st.session_state.show_vwv_analysis = True
    if 'show_fundamental_analysis' not in st.session_state:
        st.session_state.show_fundamental_analysis = True
    if 'show_market_correlation' not in st.session_state:
        st.session_state.show_market_correlation = True
    if 'show_options_analysis' not in st.session_state:
        st.session_state.show_options_analysis = True
    if 'show_confidence_intervals' not in st.session_state:
        st.session_state.show_confidence_intervals = True
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False

    # Quick Links section FIRST
    st.sidebar.markdown("### 🚀 Quick Links")
    
    # Organize quick links by category
    for category, symbols in QUICK_LINK_CATEGORIES.items():
        with st.sidebar.expander(f"📊 {category}", expanded=(category == "Major Indices")):
            cols = st.columns(2)
            for i, symbol in enumerate(symbols):
                col = cols[i % 2]
                with col:
                    if st.button(symbol, key=f"quick_{symbol}_{category}", use_container_width=True):
                        st.session_state.symbol_input = symbol
                        st.session_state.analyze_clicked = True
                        st.rerun()

    # Recently viewed symbols
    if st.session_state.recently_viewed:
        st.sidebar.markdown("### 🕒 Recently Viewed")
        recent_cols = st.columns(2)
        for i, symbol in enumerate(st.session_state.recently_viewed[-6:]):
            col = recent_cols[i % 2]
            with col:
                if st.button(symbol, key=f"recent_{symbol}_{i}", use_container_width=True):
                    st.session_state.symbol_input = symbol
                    st.session_state.analyze_clicked = True
                    st.rerun()

    # Symbol input
    st.sidebar.markdown("### 🎯 Symbol Analysis")
    
    # Get symbol from session state if available
    default_symbol = st.session_state.get('symbol_input', 'AAPL')
    symbol = st.sidebar.text_input("Enter Stock Symbol:", value=default_symbol, key="symbol_input_widget").upper().strip()
    
    # Time period selection
    period = st.sidebar.selectbox(
        "Analysis Period:",
        options=['1mo', '3mo', '6mo', '1y', '2y'],
        index=0,  # Default to 1mo
        key="period_select"
    )
    
    # Analysis controls
    st.sidebar.markdown("### ⚙️ Analysis Controls")
    
    analyze_button = st.sidebar.button("🔍 **ANALYZE**", type="primary", use_container_width=True)
    
    # Check if analyze was triggered by quick links
    if st.session_state.get('analyze_clicked', False):
        analyze_button = True
        st.session_state.analyze_clicked = False
        symbol = st.session_state.symbol_input

    # Section toggles
    st.sidebar.markdown("### 📊 Display Sections")
    
    show_vwv_analysis = st.sidebar.checkbox("VWV System Analysis", value=st.session_state.show_vwv_analysis, key="vwv_check")
    show_fundamental = st.sidebar.checkbox("Fundamental Analysis", value=st.session_state.show_fundamental_analysis, key="fundamental_check")
    show_market_correlation = st.sidebar.checkbox("Market Correlation", value=st.session_state.show_market_correlation, key="correlation_check")
    show_options = st.sidebar.checkbox("Options Analysis", value=st.session_state.show_options_analysis, key="options_check")
    show_confidence_intervals = st.sidebar.checkbox("Confidence Intervals", value=st.session_state.show_confidence_intervals, key="confidence_check")
    
    # Debug mode
    show_debug = st.sidebar.checkbox("Debug Mode", value=st.session_state.show_debug, key="debug_check")
    
    # Update session state
    st.session_state.show_vwv_analysis = show_vwv_analysis
    st.session_state.show_fundamental_analysis = show_fundamental
    st.session_state.show_market_correlation = show_market_correlation
    st.session_state.show_options_analysis = show_options
    st.session_state.show_confidence_intervals = show_confidence_intervals
    st.session_state.show_debug = show_debug
    
    return {
        'symbol': symbol,
        'period': period,
        'analyze_button': analyze_button,
        'show_debug': show_debug
    }

def add_to_recently_viewed(symbol):
    """Add symbol to recently viewed list"""
    if symbol not in st.session_state.recently_viewed:
        st.session_state.recently_viewed.append(symbol)
    else:
        # Move to end if already exists
        st.session_state.recently_viewed.remove(symbol)
        st.session_state.recently_viewed.append(symbol)
    
    # Keep only last 10
    if len(st.session_state.recently_viewed) > 10:
        st.session_state.recently_viewed = st.session_state.recently_viewed[-10:]

def show_interactive_charts(chart_data, analysis_results, show_debug=False):
    """Display interactive charts section - FIRST PRIORITY"""
    
    if chart_data is not None:
        st.markdown("### 📈 Interactive Trading Charts")
        
        try:
            display_trading_charts(chart_data, analysis_results)
        except Exception as e:
            st.error(f"Charts error: {e}")
            if show_debug:
                st.exception(e)
    else:
        st.warning("⚠️ Chart data not available")

def show_combined_vwv_technical_analysis(analysis_results, vwv_results, show_debug=False):
    """Display VWV system and technical analysis - SECOND PRIORITY"""
    if not st.session_state.show_vwv_analysis:
        return
    
    # VWV System Analysis
    if vwv_results:
        with st.expander("🔴 VWV Professional Trading System", expanded=True):
            
            # VWV Signal and composite score
            vwv_signal = vwv_results.get('vwv_signal', 'NEUTRAL')
            composite_score = vwv_results.get('composite_score', 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("VWV Signal", vwv_signal)
            with col2:
                st.metric("Composite Score", f"{composite_score:.1f}/100")
            with col3:
                confidence = vwv_results.get('signal_confidence', 0)
                st.metric("Signal Confidence", f"{confidence:.1f}%")
            
            # Create technical score bar
            create_technical_score_bar(composite_score)
            
            # VWV interpretation
            interpretation = get_vwv_signal_interpretation(vwv_signal, composite_score)
            if interpretation:
                st.info(f"**VWV Interpretation:** {interpretation}")
            
            # Technical indicators breakdown
            enhanced_indicators = analysis_results.get('enhanced_indicators', {})
            if enhanced_indicators:
                st.markdown("#### 📊 Enhanced Technical Indicators")
                
                tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                
                with tech_col1:
                    if 'daily_vwap' in enhanced_indicators:
                        vwap = enhanced_indicators['daily_vwap']
                        st.metric("Daily VWAP", f"${vwap:.2f}")
                
                with tech_col2:
                    if 'point_of_control' in enhanced_indicators:
                        poc = enhanced_indicators['point_of_control']
                        st.metric("Point of Control", f"${poc:.2f}")
                
                with tech_col3:
                    if 'weekly_deviation' in enhanced_indicators:
                        weekly_dev = enhanced_indicators['weekly_deviation']
                        st.metric("Weekly Deviation", f"{weekly_dev:.2f}%")
                
                with tech_col4:
                    current_price = analysis_results.get('current_price', 0)
                    st.metric("Current Price", f"${current_price:.2f}")

def show_fundamental_analysis(analysis_results, show_debug=False):
    """Display fundamental analysis section"""
    if not st.session_state.show_fundamental_analysis:
        return
    
    graham_score = analysis_results.get('graham_score', {})
    piotroski_score = analysis_results.get('piotroski_score', {})
    
    with st.expander("💰 Fundamental Analysis", expanded=True):
        
        # Check if this is an ETF
        if graham_score.get('error') and 'ETF' in str(graham_score.get('error')):
            st.info("ℹ️ This is an ETF - Fundamental analysis is not applicable for Exchange Traded Funds")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Graham Score")
            if graham_score and 'score' in graham_score:
                score = graham_score['score']
                total = graham_score.get('total_possible', 10)
                st.metric("Graham Score", f"{score}/{total}")
                
                if score >= 7:
                    st.success("Strong fundamental value")
                elif score >= 5:
                    st.warning("Moderate fundamental value")
                else:
                    st.error("Weak fundamental value")
            else:
                st.warning("Graham score data unavailable")
        
        with col2:
            st.markdown("#### 🏆 Piotroski Score")
            if piotroski_score and 'score' in piotroski_score:
                score = piotroski_score['score']
                total = piotroski_score.get('total_possible', 9)
                st.metric("Piotroski Score", f"{score}/{total}")
                
                if score >= 7:
                    st.success("High quality company")
                elif score >= 5:
                    st.warning("Average quality company")
                else:
                    st.error("Low quality company")
            else:
                st.warning("Piotroski score data unavailable")

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis section"""
    if not st.session_state.show_market_correlation:
        return
    
    market_correlations = analysis_results.get('market_correlations', {})
    breakout_analysis = analysis_results.get('breakout_analysis', {})
    
    with st.expander("🌐 Market Correlation Analysis", expanded=True):
        
        if market_correlations:
            st.markdown("#### 📊 ETF Correlations")
            
            correlation_data = []
            for etf, corr_data in market_correlations.items():
                if isinstance(corr_data, dict):
                    correlation_data.append({
                        'ETF': etf,
                        'Correlation': f"{corr_data.get('correlation', 0):.3f}",
                        'Beta': f"{corr_data.get('beta', 0):.2f}",
                        'Description': SYMBOL_DESCRIPTIONS.get(etf, "Market ETF")
                    })
            
            if correlation_data:
                df_corr = pd.DataFrame(correlation_data)
                st.dataframe(df_corr, use_container_width=True, hide_index=True)
        
        if breakout_analysis:
            st.markdown("#### 🚀 Breakout/Breakdown Analysis")
            
            breakout_col1, breakout_col2 = st.columns(2)
            
            with breakout_col1:
                breakout_score = breakout_analysis.get('breakout_score', 0)
                st.metric("Breakout Score", f"{breakout_score:.1f}/100")
                
                if breakout_score >= 70:
                    st.success("Strong breakout potential")
                elif breakout_score >= 40:
                    st.warning("Moderate breakout potential")
                else:
                    st.error("Low breakout potential")
            
            with breakout_col2:
                breakdown_score = breakout_analysis.get('breakdown_score', 0)
                st.metric("Breakdown Score", f"{breakdown_score:.1f}/100")
                
                if breakdown_score >= 70:
                    st.error("High breakdown risk")
                elif breakdown_score >= 40:
                    st.warning("Moderate breakdown risk")
                else:
                    st.success("Low breakdown risk")

def show_options_analysis(analysis_results, show_debug=False):
    """Display options analysis section"""
    if not st.session_state.show_options_analysis:
        return
    
    options_data = analysis_results.get('options_data', [])
    
    with st.expander("🎯 Options Analysis", expanded=True):
        
        if options_data:
            st.markdown("#### 📊 Options Strike Levels")
            
            # Convert to DataFrame for display
            df_options = pd.DataFrame(options_data)
            st.dataframe(df_options, use_container_width=True, hide_index=True)
            
            # Options strategy suggestions
            st.markdown("#### 💡 Strategy Suggestions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info("**Put Selling Strategy:**\n" 
                        "• Sell puts below current price\n" 
                        "• Collect premium if stock stays above strike\n"
                        "• Delta: Price sensitivity (~-0.16)\n"
                        "• Theta: Daily time decay benefit")
            
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
            st.error("❌ No options analysis available - insufficient data")

def show_confidence_intervals(analysis_results, show_debug=False):
    """Display confidence intervals section - PRESERVED FUNCTIONALITY"""
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
    """Perform enhanced analysis using modular components - ENHANCED v4.2.1"""
    try:
        # Step 1: Fetch data using modular data fetcher
        market_data = get_market_data_enhanced(symbol, period, show_debug)
        
        if market_data is None:
            st.error(f"❌ Could not fetch data for {symbol}")
            return None, None
        
        # Step 2: Store data using data manager
        data_manager = get_data_manager()
        data_manager.store_market_data(symbol, market_data)
        
        # Step 3: Get analysis input data
        analysis_input = data_manager.get_market_data_for_analysis(symbol)
        
        # Step 4: Calculate VWV system indicators
        vwv_results = calculate_vwv_system_complete(analysis_input, show_debug)
        
        # Step 5: Calculate enhanced technical indicators  
        daily_vwap = calculate_daily_vwap(analysis_input)
        fibonacci_emas = calculate_fibonacci_emas(analysis_input)
        point_of_control = calculate_point_of_control_enhanced(analysis_input)
        comprehensive_technicals = calculate_comprehensive_technicals(analysis_input, symbol, show_debug)
        weekly_deviation = calculate_weekly_deviations(analysis_input, show_debug)
        
        # Step 6: Calculate market correlations
        market_correlations = calculate_market_correlations_enhanced(analysis_input, symbol, show_debug)
        
        # Step 7: Calculate breakout analysis  
        breakout_analysis = calculate_breakout_breakdown_analysis(analysis_input, symbol, show_debug)
        
        # Step 8: Calculate fundamental analysis (skip for ETFs)
        is_etf_symbol = is_etf(symbol)
        
        if is_etf_symbol:
            graham_score = {'score': 0, 'total_possible': 10, 'criteria': [], 'error': 'ETF - Fundamental analysis not applicable'}
            piotroski_score = {'score': 0, 'total_possible': 9, 'criteria': [], 'error': 'ETF - Fundamental analysis not applicable'}
        else:
            graham_score = calculate_graham_score(symbol, show_debug)
            piotroski_score = calculate_piotroski_score(symbol, show_debug)
        
        # Step 9: Calculate options levels
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
        
        # Step 10: Calculate confidence intervals - FIXED: Only 1 argument
        confidence_analysis = calculate_confidence_intervals(analysis_input)
        
        # Step 11: Build analysis results
        current_date = analysis_input.index[-1].strftime('%Y-%m-%d')
        
        analysis_results = {
            'symbol': symbol,
            'timestamp': current_date,
            'current_price': current_price,
            'enhanced_indicators': {
                'daily_vwap': daily_vwap,
                'fibonacci_emas': fibonacci_emas,
                'point_of_control': point_of_control,
                'weekly_deviation': weekly_deviation
            },
            'comprehensive_technicals': comprehensive_technicals,
            'market_correlations': market_correlations,
            'breakout_analysis': breakout_analysis,
            'graham_score': graham_score,
            'piotroski_score': piotroski_score,
            'options_data': options_levels,
            'confidence_analysis': confidence_analysis,
            'system_status': 'OPERATIONAL'
        }
        
        # Store results
        data_manager.store_analysis_results(symbol, analysis_results)
        
        # Get chart data
        chart_data = data_manager.get_market_data_for_chart(symbol)
        
        return analysis_results, chart_data, vwv_results
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        if show_debug:
            st.exception(e)
        return None, None, None

def main():
    """Main application function"""
    # Create header using modular component
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
            analysis_results, chart_data, vwv_results = perform_enhanced_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results and chart_data is not None:
                # Show charts FIRST at the top
                show_interactive_charts(chart_data, analysis_results, controls['show_debug'])
                
                # Show all analysis sections using modular functions
                show_combined_vwv_technical_analysis(analysis_results, vwv_results, controls['show_debug'])
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                show_options_analysis(analysis_results, controls['show_debug'])
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
                # Debug information
                if controls['show_debug']:
                    with st.expander("🔧 Debug Information", expanded=False):
                        st.json({
                            'symbol': controls['symbol'],
                            'period': controls['period'],
                            'analysis_timestamp': analysis_results.get('timestamp'),
                            'data_points': len(chart_data) if chart_data is not None else 0,
                            'vwv_signal': vwv_results.get('vwv_signal') if vwv_results else 'N/A'
                        })
                
                st.success(f"✅ Analysis complete for {controls['symbol']}")
            else:
                st.error(f"❌ Analysis failed for {controls['symbol']}")
    
    else:
        # Show welcome message
        st.markdown("""
        ## Welcome to VWV Professional Trading System v4.2.1
        
        **Select a symbol from the Quick Links or enter a symbol in the sidebar to begin analysis.**
        
        ### 🚀 Features:
        - **Interactive Charts** with technical overlays
        - **VWV Professional System** with 32+ years of proven logic  
        - **Comprehensive Technical Analysis** with composite scoring
        - **Fundamental Analysis** using Graham & Piotroski methods
        - **Market Correlation Analysis** with ETF comparisons
        - **Options Analysis** with Greeks and strike levels
        - **Statistical Confidence Intervals** for volatility projections
        
        ### 📊 Analysis Sections:
        1. **📈 Interactive Charts** - Price action with indicators
        2. **🔴 VWV System** - Professional trading signals 
        3. **💰 Fundamental** - Value investment scoring
        4. **🌐 Market Correlation** - ETF correlation analysis
        5. **🎯 Options** - Strike levels and Greeks
        6. **📊 Confidence Intervals** - Statistical projections
        """)

if __name__ == "__main__":
    main()
