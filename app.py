"""
VWV Professional Trading System v4.2.1 - Complete Application
Date: August 17, 2025 - Current EST Time
Enhancement: Charts First + Technical Second + All Modules + Fixed Composite Scores + Gradient Bars + Quick Links Structure + Debug Toggle
Status: COMPLETE FIX - Composite Scoring + Gradient Bars + Quick Links Expander + Debug Toggle Restored
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
    get_vwv_signal_interpretation,
    calculate_vwv_composite_score
)

# Volume Analysis imports with safe fallback
try:
    from analysis.volume import (
        calculate_complete_volume_analysis,
        calculate_market_wide_volume_analysis
    )
    VOLUME_ANALYSIS_AVAILABLE = True
except ImportError:
    VOLUME_ANALYSIS_AVAILABLE = False

# Volatility Analysis imports with safe fallback
try:
    from analysis.volatility import (
        calculate_complete_volatility_analysis,
        calculate_market_wide_volatility_analysis
    )
    VOLATILITY_ANALYSIS_AVAILABLE = True
except ImportError:
    VOLATILITY_ANALYSIS_AVAILABLE = False

# Baldwin Indicator import with safe fallback
try:
    from analysis.baldwin_indicator import (
        calculate_baldwin_indicator_complete,
        format_baldwin_for_display
    )
    BALDWIN_INDICATOR_AVAILABLE = True
except ImportError:
    BALDWIN_INDICATOR_AVAILABLE = False

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
    """Create sidebar controls and return analysis parameters - COMPLETE WITH ALL MODULES + DEBUG TOGGLE"""
    st.sidebar.title("📊 Trading Analysis")
    
    # Initialize session state for ALL modules
    if 'recently_viewed' not in st.session_state:
        st.session_state.recently_viewed = []
    if 'show_vwv_analysis' not in st.session_state:
        st.session_state.show_vwv_analysis = True
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
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False

    # Symbol input section
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

    # Quick Links section - FIXED: OVERALL EXPANDER CONTAINING CATEGORIES
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
                                    st.session_state.symbol_input = sym
                                    st.session_state.analyze_clicked = True
                                    st.rerun()

    # Recently viewed symbols
    if st.session_state.recently_viewed:
        st.sidebar.markdown("### 🕒 Recently Viewed")
        with st.sidebar.expander("Recent Symbols", expanded=False):
            recent_cols = st.columns(2)
            for i, symbol_recent in enumerate(st.session_state.recently_viewed[-6:]):
                col = recent_cols[i % 2]
                with col:
                    if st.button(symbol_recent, key=f"recent_{symbol_recent}_{i}", use_container_width=True):
                        st.session_state.symbol_input = symbol_recent
                        st.session_state.analyze_clicked = True
                        st.rerun()

    # Section toggles in expandable section - INCLUDING ALL MODULES
    with st.sidebar.expander("📊 Display Sections", expanded=True):
        show_vwv_analysis = st.checkbox("VWV System Analysis", value=st.session_state.show_vwv_analysis, key="vwv_check")
        
        # Volume Analysis (if available)
        if VOLUME_ANALYSIS_AVAILABLE:
            show_volume_analysis = st.checkbox("📊 Volume Analysis", value=st.session_state.show_volume_analysis, key="volume_check")
        else:
            show_volume_analysis = False
            
        # Volatility Analysis (if available)
        if VOLATILITY_ANALYSIS_AVAILABLE:
            show_volatility_analysis = st.checkbox("📊 Volatility Analysis", value=st.session_state.show_volatility_analysis, key="volatility_check")
        else:
            show_volatility_analysis = False
            
        show_fundamental = st.checkbox("💰 Fundamental Analysis", value=st.session_state.show_fundamental_analysis, key="fundamental_check")
        
        # Baldwin Indicator (if available)
        if BALDWIN_INDICATOR_AVAILABLE:
            show_baldwin_indicator = st.checkbox("🚦 Baldwin Indicator", value=st.session_state.show_baldwin_indicator, key="baldwin_check")
        else:
            show_baldwin_indicator = False
            
        show_market_correlation = st.checkbox("🌐 Market Correlation", value=st.session_state.show_market_correlation, key="correlation_check")
        show_options = st.checkbox("🎯 Options Analysis", value=st.session_state.show_options_analysis, key="options_check")
        show_confidence_intervals = st.checkbox("📊 Confidence Intervals", value=st.session_state.show_confidence_intervals, key="confidence_check")

    # Debug toggle - RESTORED
    show_debug = st.sidebar.checkbox("🐛 Show Debug Info", value=st.session_state.show_debug, key="debug_check")
    
    # Update session state
    st.session_state.show_vwv_analysis = show_vwv_analysis
    st.session_state.show_volume_analysis = show_volume_analysis if VOLUME_ANALYSIS_AVAILABLE else False
    st.session_state.show_volatility_analysis = show_volatility_analysis if VOLATILITY_ANALYSIS_AVAILABLE else False
    st.session_state.show_fundamental_analysis = show_fundamental
    st.session_state.show_baldwin_indicator = show_baldwin_indicator if BALDWIN_INDICATOR_AVAILABLE else False
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
    """Display VWV system and technical analysis - SECOND PRIORITY WITH PROPER SCORING"""
    if not st.session_state.show_vwv_analysis:
        return
    
    # VWV System Analysis
    if vwv_results:
        with st.expander("🔴 VWV Professional Trading System", expanded=True):
            
            # Combined Header with both scores
            col1, col2, col3, col4 = st.columns(4)
            
            # VWV Signal
            signal_strength = vwv_results.get('signal_strength', 'WEAK')
            signal_color = vwv_results.get('signal_color', '⚪')
            vwv_score = vwv_results.get('vwv_score', 0)
            
            # Technical Composite Score - FIXED: Use proper VWV composite scoring
            try:
                # Calculate proper composite score using VWV system
                if 'enhanced_indicators' in analysis_results:
                    chart_data_for_scoring = analysis_results.get('chart_data')
                    if chart_data_for_scoring is None:
                        # Fallback to using analysis input data
                        data_manager = get_data_manager()
                        chart_data_for_scoring = data_manager.get_market_data_for_analysis(analysis_results['symbol'])
                    
                    if chart_data_for_scoring is not None:
                        composite_score, score_details = calculate_vwv_composite_score(chart_data_for_scoring, DEFAULT_VWV_CONFIG)
                    else:
                        composite_score, score_details = 50.0, {}
                else:
                    composite_score, score_details = 50.0, {}
            except Exception as e:
                if show_debug:
                    st.warning(f"Composite score calculation failed: {e}")
                composite_score, score_details = 50.0, {}
            
            # Color mapping for signal strength
            strength_colors = {
                'VERY_STRONG': '#DC143C',  # Crimson red
                'STRONG': '#FFD700',       # Gold
                'GOOD': '#32CD32',         # Lime green
                'WEAK': '#808080'          # Gray
            }
            
            signal_color_hex = strength_colors.get(signal_strength, '#808080')
            
            with col1:
                st.markdown(f"""
                <div style="padding: 1rem; background: linear-gradient(135deg, {signal_color_hex}20, {signal_color_hex}10); 
                            border-left: 4px solid {signal_color_hex}; border-radius: 8px;">
                    <h4 style="color: {signal_color_hex}; margin: 0;">VWV Signal</h4>
                    <h2 style="color: {signal_color_hex}; margin: 0.2rem 0;">{signal_color} {signal_strength}</h2>
                    <p style="margin: 0; color: #666; font-size: 0.9em;">Score: {vwv_score:.1f}/100</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Technical composite score with color
                if composite_score >= 70:
                    tech_color = "#32CD32"  # Green
                    tech_interpretation = "Bullish"
                elif composite_score >= 50:
                    tech_color = "#FFD700"  # Gold
                    tech_interpretation = "Neutral"
                else:
                    tech_color = "#FF6B6B"  # Red
                    tech_interpretation = "Bearish"
                
                st.markdown(f"""
                <div style="padding: 1rem; background: linear-gradient(135deg, {tech_color}20, {tech_color}10); 
                            border-left: 4px solid {tech_color}; border-radius: 8px;">
                    <h4 style="color: {tech_color}; margin: 0;">Technical Score</h4>
                    <h2 style="color: {tech_color}; margin: 0.2rem 0;">{composite_score:.1f}/100</h2>
                    <p style="margin: 0; color: #666; font-size: 0.9em;">{tech_interpretation}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.metric("Current Price", f"${vwv_results.get('current_price', analysis_results.get('current_price', 0)):.2f}")
            
            with col4:
                st.metric("Timestamp", vwv_results.get('timestamp', 'N/A'))
            
            # GRADIENT SCORE BAR - FIXED: Use proper UI component
            st.markdown("#### 📊 Technical Composite Score")
            try:
                # Use the proper gradient score bar component
                score_bar_html = create_technical_score_bar(composite_score, score_details)
                st.components.v1.html(score_bar_html, height=160)
            except Exception as e:
                if show_debug:
                    st.warning(f"Score bar display failed: {e}")
                # Fallback to simple progress bar
                st.progress(composite_score / 100)
                st.write(f"**Technical Score: {composite_score:.1f}/100** - {tech_interpretation}")
            
            # VWV interpretation
            interpretation = get_vwv_signal_interpretation(signal_strength, composite_score)
            if interpretation:
                st.info(f"**VWV Interpretation:** {interpretation}")
            
            # Technical indicators breakdown
            enhanced_indicators = analysis_results.get('enhanced_indicators', {})
            if enhanced_indicators:
                st.markdown("#### 📊 Enhanced Technical Indicators")
                
                tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
                
                with tech_col1:
                    daily_vwap = enhanced_indicators.get('daily_vwap', 0)
                    if daily_vwap:
                        st.metric("Daily VWAP", f"${daily_vwap:.2f}")
                
                with tech_col2:
                    point_of_control = enhanced_indicators.get('point_of_control', 0)
                    if point_of_control:
                        st.metric("Point of Control", f"${point_of_control:.2f}")
                
                with tech_col3:
                    weekly_deviation = enhanced_indicators.get('weekly_deviation', {})
                    if isinstance(weekly_deviation, dict) and 'std_price' in weekly_deviation:
                        std_price = weekly_deviation.get('std_price', 0)
                        st.metric("Weekly Std Dev", f"{std_price:.2f}%")
                
                with tech_col4:
                    current_price = analysis_results.get('current_price', 0)
                    st.metric("Current Price", f"${current_price:.2f}")

def show_volume_analysis(analysis_results, show_debug=False):
    """Display volume analysis section - RESTORED"""
    if not st.session_state.show_volume_analysis or not VOLUME_ANALYSIS_AVAILABLE:
        return
    
    volume_analysis = analysis_results.get('volume_analysis', {})
    
    with st.expander("📊 Volume Analysis", expanded=True):
        
        if volume_analysis:
            st.markdown("#### 📊 Volume Metrics")
            
            vol_col1, vol_col2, vol_col3 = st.columns(3)
            
            with vol_col1:
                volume_strength = volume_analysis.get('volume_strength', 'Normal')
                st.metric("Volume Strength", volume_strength)
            
            with vol_col2:
                current_vs_avg = volume_analysis.get('current_vs_avg', 1.0)
                st.metric("Current vs Average", f"{current_vs_avg:.2f}x")
            
            with vol_col3:
                volume_trend = volume_analysis.get('volume_trend', 'Neutral')
                st.metric("Volume Trend", volume_trend)
            
            # Additional volume data if available
            if 'volume_summary' in volume_analysis:
                volume_summary = volume_analysis['volume_summary']
                st.markdown("#### 📈 Volume Summary")
                st.info(f"**Analysis**: {volume_summary}")
        else:
            st.warning("⚠️ Volume analysis data not available")

def show_volatility_analysis(analysis_results, show_debug=False):
    """Display volatility analysis section - RESTORED"""
    if not st.session_state.show_volatility_analysis or not VOLATILITY_ANALYSIS_AVAILABLE:
        return
    
    volatility_analysis = analysis_results.get('volatility_analysis', {})
    
    with st.expander("📊 Volatility Analysis", expanded=True):
        
        if volatility_analysis:
            st.markdown("#### 📊 Volatility Metrics")
            
            vol_col1, vol_col2, vol_col3 = st.columns(3)
            
            with vol_col1:
                current_volatility = volatility_analysis.get('current_volatility', 0)
                st.metric("Current Volatility", f"{current_volatility:.2f}%")
            
            with vol_col2:
                volatility_regime = volatility_analysis.get('volatility_regime', 'Normal')
                st.metric("Volatility Regime", volatility_regime)
            
            with vol_col3:
                volatility_trend = volatility_analysis.get('volatility_trend', 'Stable')
                st.metric("Volatility Trend", volatility_trend)
            
            # Additional volatility data if available
            if 'volatility_summary' in volatility_analysis:
                volatility_summary = volatility_analysis['volatility_summary']
                st.markdown("#### 📈 Volatility Summary")
                st.info(f"**Analysis**: {volatility_summary}")
        else:
            st.warning("⚠️ Volatility analysis data not available")

def show_baldwin_indicator_analysis(baldwin_results=None, show_debug=False):
    """Display Baldwin Market Regime Indicator - RESTORED"""
    if not st.session_state.show_baldwin_indicator or not BALDWIN_INDICATOR_AVAILABLE:
        return
    
    with st.expander("🚦 Baldwin Market Regime Indicator", expanded=True):
        
        # Calculate Baldwin indicator if not provided
        if baldwin_results is None:
            if show_debug:
                st.write("📊 Calculating Baldwin Market Regime...")
            baldwin_results = calculate_baldwin_indicator_complete(show_debug=show_debug)
        
        if baldwin_results and 'error' not in baldwin_results:
            # Format for display
            display_data = format_baldwin_for_display(baldwin_results)
            
            if 'error' not in display_data:
                # Main regime display
                regime = display_data.get('regime', 'YELLOW')
                regime_score = display_data.get('overall_score', 50)
                regime_description = display_data.get('description', 'Market assessment')
                strategy = display_data.get('strategy', 'Strategy not available')
                
                # Color coding for regime
                regime_colors = {
                    'GREEN': '#32CD32',   # Lime Green
                    'YELLOW': '#FFD700',  # Gold
                    'RED': '#DC143C'      # Crimson
                }
                
                regime_color = regime_colors.get(regime, '#FFD700')
                
                # Main regime header
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Market Regime", f"{display_data.get('regime_color', '🟡')} {regime}")
                with col2:
                    st.metric("Baldwin Score", f"{regime_score:.1f}/100")
                with col3:
                    st.metric("Timestamp", display_data.get('timestamp', 'N/A'))
                
                # Strategy and description
                st.markdown("#### 🎯 Strategy Recommendation")
                st.info(f"**{strategy}**")
                
                st.markdown("#### 📝 Market Assessment")
                st.write(regime_description)
                
                # Component breakdown if available
                if 'component_summary' in display_data:
                    st.markdown("#### 📊 Component Breakdown")
                    component_summary = display_data['component_summary']
                    if component_summary:
                        df_components = pd.DataFrame(component_summary)
                        st.dataframe(df_components, use_container_width=True, hide_index=True)
            else:
                st.error(f"Baldwin formatting error: {display_data.get('error', 'Unknown error')}")
        else:
            st.error(f"Baldwin calculation error: {baldwin_results.get('error', 'Unknown error') if baldwin_results else 'No results'}")

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
            
            # Check if OVERALL data exists in breakout_analysis
            overall_data = breakout_analysis.get('OVERALL', {})
            if overall_data:
                breakout_col1, breakout_col2 = st.columns(2)
                
                with breakout_col1:
                    breakout_ratio = overall_data.get('breakout_ratio', 0)
                    st.metric("Market Breakouts", f"{breakout_ratio:.1f}%")
                
                with breakout_col2:
                    breakdown_ratio = overall_data.get('breakdown_ratio', 0)
                    st.metric("Market Breakdowns", f"{breakdown_ratio:.1f}%")
                
                # Market regime
                market_regime = overall_data.get('market_regime', 'Unknown')
                st.info(f"**Market Regime**: {market_regime}")
            else:
                st.warning("Breakout analysis data not available")

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
    """Perform enhanced analysis using modular components - ALL MODULES INCLUDED"""
    try:
        # Step 1: Fetch data using modular data fetcher
        market_data = get_market_data_enhanced(symbol, period, show_debug)
        
        if market_data is None:
            st.error(f"❌ Could not fetch data for {symbol}")
            return None, None, None
        
        # Step 2: Store data using data manager
        data_manager = get_data_manager()
        data_manager.store_market_data(symbol, market_data)
        
        # Step 3: Get analysis input data
        analysis_input = data_manager.get_market_data_for_analysis(symbol)
        
        # Step 4: Calculate VWV system indicators
        vwv_results = calculate_vwv_system_complete(analysis_input, show_debug)
        
        # Step 5: Calculate enhanced technical indicators - FIXED: Only pass data parameter
        daily_vwap = calculate_daily_vwap(analysis_input)
        fibonacci_emas = calculate_fibonacci_emas(analysis_input)
        point_of_control = calculate_point_of_control_enhanced(analysis_input)
        comprehensive_technicals = calculate_comprehensive_technicals(analysis_input)  # FIXED: Only 1 argument
        weekly_deviation = calculate_weekly_deviations(analysis_input)  # FIXED: Only 1 argument
        
        # Step 6: Calculate volume analysis (if available)
        volume_analysis = None
        if VOLUME_ANALYSIS_AVAILABLE:
            try:
                volume_analysis = calculate_complete_volume_analysis(analysis_input)
            except Exception as e:
                if show_debug:
                    st.warning(f"Volume analysis failed: {e}")
        
        # Step 7: Calculate volatility analysis (if available)
        volatility_analysis = None
        if VOLATILITY_ANALYSIS_AVAILABLE:
            try:
                volatility_analysis = calculate_complete_volatility_analysis(analysis_input)
            except Exception as e:
                if show_debug:
                    st.warning(f"Volatility analysis failed: {e}")
        
        # Step 8: Calculate market correlations
        market_correlations = calculate_market_correlations_enhanced(analysis_input, symbol, period, show_debug)
        
        # Step 9: Calculate breakout analysis - FIXED: Only show_debug parameter
        breakout_analysis = calculate_breakout_breakdown_analysis(show_debug)
        
        # Step 10: Calculate fundamental analysis (skip for ETFs)
        is_etf_symbol = is_etf(symbol)
        
        if is_etf_symbol:
            graham_score = {'score': 0, 'total_possible': 10, 'criteria': [], 'error': 'ETF - Fundamental analysis not applicable'}
            piotroski_score = {'score': 0, 'total_possible': 9, 'criteria': [], 'error': 'ETF - Fundamental analysis not applicable'}
        else:
            graham_score = calculate_graham_score(symbol, show_debug)
            piotroski_score = calculate_piotroski_score(symbol, show_debug)
        
        # Step 11: Calculate options levels
        volatility = 20  # Default fallback
        if comprehensive_technicals and 'volatility_20d' in comprehensive_technicals:
            volatility = comprehensive_technicals.get('volatility_20d', 20)
        
        underlying_beta = 1.0  # Default market beta
        
        if market_correlations:
            for etf in ['SPY', 'QQQ', 'MAGS']:
                if etf in market_correlations and isinstance(market_correlations[etf], dict) and 'beta' in market_correlations[etf]:
                    try:
                        underlying_beta = abs(float(market_correlations[etf]['beta']))
                        break
                    except:
                        continue
        
        current_price = round(float(analysis_input['Close'].iloc[-1]), 2)
        options_levels = calculate_options_levels_enhanced(current_price, volatility, underlying_beta=underlying_beta)
        
        # Step 12: Calculate confidence intervals - FIXED: Only 1 argument
        confidence_analysis = calculate_confidence_intervals(analysis_input)
        
        # Step 13: Build analysis results - SAFE HANDLING OF NONE VALUES + ALL MODULES
        current_date = analysis_input.index[-1].strftime('%Y-%m-%d')
        
        analysis_results = {
            'symbol': symbol,
            'timestamp': current_date,
            'current_price': current_price,
            'chart_data': analysis_input,  # Include for scoring calculations
            'enhanced_indicators': {
                'daily_vwap': daily_vwap if daily_vwap else 0,
                'fibonacci_emas': fibonacci_emas if fibonacci_emas else {},
                'point_of_control': point_of_control if point_of_control else 0,
                'weekly_deviation': weekly_deviation if weekly_deviation else {}
            },
            'comprehensive_technicals': comprehensive_technicals if comprehensive_technicals else {},
            'volume_analysis': volume_analysis if volume_analysis else {},
            'volatility_analysis': volatility_analysis if volatility_analysis else {},
            'market_correlations': market_correlations if market_correlations else {},
            'breakout_analysis': breakout_analysis if breakout_analysis else {},
            'graham_score': graham_score,
            'piotroski_score': piotroski_score,
            'options_data': options_levels if options_levels else [],
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
                
                # Show all analysis sections using modular functions - COMPLETE ORDER
                show_combined_vwv_technical_analysis(analysis_results, vwv_results, controls['show_debug'])
                
                # Volume Analysis (if available)
                if VOLUME_ANALYSIS_AVAILABLE:
                    show_volume_analysis(analysis_results, controls['show_debug'])
                
                # Volatility Analysis (if available)
                if VOLATILITY_ANALYSIS_AVAILABLE:
                    show_volatility_analysis(analysis_results, controls['show_debug'])
                
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                
                # Baldwin Indicator (if available) - BEFORE Market Correlation
                if BALDWIN_INDICATOR_AVAILABLE:
                    show_baldwin_indicator_analysis(show_debug=controls['show_debug'])
                
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                show_options_analysis(analysis_results, controls['show_debug'])
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
                # Debug information - RESTORED
                if controls['show_debug']:
                    with st.expander("🔧 Debug Information", expanded=False):
                        st.json({
                            'symbol': controls['symbol'],
                            'period': controls['period'],
                            'analysis_timestamp': analysis_results.get('timestamp'),
                            'data_points': len(chart_data) if chart_data is not None else 0,
                            'vwv_signal': vwv_results.get('vwv_signal') if vwv_results else 'N/A',
                            'volume_available': VOLUME_ANALYSIS_AVAILABLE,
                            'volatility_available': VOLATILITY_ANALYSIS_AVAILABLE,
                            'baldwin_available': BALDWIN_INDICATOR_AVAILABLE
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
        - **Volume Analysis** - 5D/30D rolling metrics with regime detection
        - **Volatility Analysis** - 5D/30D rolling with market regime classification  
        - **Fundamental Analysis** using Graham & Piotroski methods
        - **Baldwin Market Regime** - Multi-factor market assessment
        - **Market Correlation Analysis** with ETF comparisons
        - **Options Analysis** with Greeks and strike levels
        - **Statistical Confidence Intervals** for volatility projections
        
        ### 📊 Analysis Sections:
        1. **📈 Interactive Charts** - Price action with indicators
        2. **🔴 VWV System** - Professional trading signals 
        3. **📊 Volume Analysis** - Volume strength and trends
        4. **📊 Volatility Analysis** - Market volatility regimes  
        5. **💰 Fundamental** - Value investment scoring
        6. **🚦 Baldwin Indicator** - Market regime analysis
        7. **🌐 Market Correlation** - ETF correlation analysis
        8. **🎯 Options** - Strike levels and Greeks
        9. **📊 Confidence Intervals** - Statistical projections
        """)
        
        # Module availability status
        st.markdown("### 🔧 Module Status")
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            volume_status = "✅ Available" if VOLUME_ANALYSIS_AVAILABLE else "❌ Not Available"
            st.write(f"**Volume Analysis**: {volume_status}")
            
        with status_col2:
            volatility_status = "✅ Available" if VOLATILITY_ANALYSIS_AVAILABLE else "❌ Not Available"
            st.write(f"**Volatility Analysis**: {volatility_status}")
            
        with status_col3:
            baldwin_status = "✅ Available" if BALDWIN_INDICATOR_AVAILABLE else "❌ Not Available"
            st.write(f"**Baldwin Indicator**: {baldwin_status}")

if __name__ == "__main__":
    main()
