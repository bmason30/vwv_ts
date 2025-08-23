"""
VWV Professional Trading System v4.2.1 - WORKING WITH ENHANCEMENTS
Based on working test_imports.py structure + Volume/Volatility Analysis integration
FIXED: Function signature issues and restored all module enhancements
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Import our modular components - CORRECTED IMPORTS
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

# Volume and Volatility Analysis imports with safe fallbacks - NEW v4.2.1
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
    st.sidebar.title("📊 Trading Analysis v4.2.1")
    
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
    if 'show_charts' not in st.session_state:
        st.session_state.show_charts = True
    if 'auto_analyze' not in st.session_state:
        st.session_state.auto_analyze = False
    if 'show_risk_management' not in st.session_state:
        st.session_state.show_risk_management = True
    # NEW v4.2.1 - Volume and Volatility Analysis toggles
    if 'show_volume_analysis' not in st.session_state:
        st.session_state.show_volume_analysis = True
    if 'show_volatility_analysis' not in st.session_state:
        st.session_state.show_volatility_analysis = True
    
    # Handle selected symbol from quicklinks/recents
    if 'selected_symbol' in st.session_state:
        current_symbol = st.session_state.selected_symbol
        st.session_state.auto_analyze = True  # Trigger analysis
        del st.session_state.selected_symbol
    else:
        current_symbol = UI_SETTINGS['default_symbol']
        
    # Symbol input and period selection (NO FORM - this was causing issues)
    symbol = st.sidebar.text_input("Symbol", value=current_symbol, help="Enter stock symbol").upper()
    
    # CORRECTED: Default period set to '1mo' (1 month) instead of 3mo
    period_options = ['1mo', '3mo', '6mo', '1y', '2y']
    period = st.sidebar.selectbox("Data Period", period_options, index=0)  # Index 0 = '1mo'
    
    # Analyze button (outside form to prevent symbol reset)
    analyze_button = st.sidebar.button("📊 Analyze Symbol", type="primary", use_container_width=True)
    
    # Check for auto-analyze trigger from quicklinks/recents
    if st.session_state.auto_analyze:
        st.session_state.auto_analyze = False  # Reset flag
        analyze_button = True  # Force analysis
    
    # Analysis sections checkboxes
    st.sidebar.markdown("### Analysis Sections")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.session_state.show_charts = st.checkbox("📊 Charts", value=st.session_state.show_charts)
        st.session_state.show_vwv_analysis = st.checkbox("🔴 VWV/Tech", value=st.session_state.show_vwv_analysis)
        # NEW v4.2.1 - Volume Analysis toggle
        if VOLUME_ANALYSIS_AVAILABLE:
            st.session_state.show_volume_analysis = st.checkbox("📊 Volume", value=st.session_state.show_volume_analysis)
        # NEW v4.2.1 - Volatility Analysis toggle
        if VOLATILITY_ANALYSIS_AVAILABLE:
            st.session_state.show_volatility_analysis = st.checkbox("📊 Volatility", value=st.session_state.show_volatility_analysis)
        st.session_state.show_fundamental_analysis = st.checkbox("📈 Fundamental", value=st.session_state.show_fundamental_analysis)
    
    with col2:
        st.session_state.show_market_correlation = st.checkbox("🌐 Correlation", value=st.session_state.show_market_correlation)
        st.session_state.show_options_analysis = st.checkbox("🎯 Options", value=st.session_state.show_options_analysis)
        st.session_state.show_confidence_intervals = st.checkbox("📊 Confidence", value=st.session_state.show_confidence_intervals)
        st.session_state.show_risk_management = st.checkbox("🎯 Risk Mgmt", value=st.session_state.show_risk_management)
    
    # Recently viewed section - SECOND
    if len(st.session_state.recently_viewed) > 0:
        with st.sidebar.expander("🕒 Recently Viewed"):
            recent_cols = st.columns(3)
            for i, recent_symbol in enumerate(st.session_state.recently_viewed[:9]):
                col_idx = i % 3
                with recent_cols[col_idx]:
                    if st.button(recent_symbol, key=f"recent_{recent_symbol}_{i}", help=SYMBOL_DESCRIPTIONS.get(recent_symbol, f"{recent_symbol} - Recently viewed symbol"), use_container_width=True):
                        st.session_state.selected_symbol = recent_symbol
                        st.session_state.auto_analyze = True  # Set flag for auto-analysis
                        st.rerun()

    # Quick Links section - THIRD
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
                                    st.session_state.auto_analyze = True  # Set flag for auto-analysis
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

def show_combined_vwv_technical_analysis(analysis_results, vwv_results, show_debug=False):
    """Display Combined Williams VIX Fix + Technical Composite Score Analysis - FIXED"""
    if not st.session_state.show_vwv_analysis:
        return
        
    with st.expander(f"🔴 {analysis_results['symbol']} - VWV Signals & Technical Composite Analysis", expanded=True):
        
        if vwv_results and 'error' not in vwv_results:
            # Combined Header with both scores
            col1, col2, col3, col4 = st.columns(4)
            
            # VWV Signal
            signal_strength = vwv_results.get('signal_strength', 'WEAK')
            signal_color = vwv_results.get('signal_color', '⚪')
            vwv_score = vwv_results.get('vwv_score', 0)
            
            # Technical Composite Score
            enhanced_indicators = analysis_results.get('enhanced_indicators', {})
            comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
            composite_score = comprehensive_technicals.get('composite_score', 50)
            
            with col1:
                st.metric("VWV Signal", f"{signal_color} {signal_strength}")
            with col2:
                st.metric("VWV Score", f"{vwv_score:.1f}/100")
            with col3:
                st.metric("Tech Composite", f"{composite_score:.1f}/100")
            with col4:
                current_price = analysis_results.get('current_price', 0)
                st.metric("Current Price", f"${current_price:.2f}")

            # Create technical score bar
            create_technical_score_bar(composite_score)
            
            # Signal analysis
            st.subheader("📊 Signal Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔴 VWV Analysis")
                # FIXED: Call function with correct parameters (score, signal_strength)
                vwv_interpretation = get_vwv_signal_interpretation(vwv_score, signal_strength)
                st.info(vwv_interpretation)
                
                if 'components' in vwv_results:
                    with st.expander("🔍 VWV Components"):
                        components = vwv_results['components']
                        for component, data in components.items():
                            if isinstance(data, dict):
                                st.write(f"**{component.replace('_', ' ').title()}:**")
                                for key, value in data.items():
                                    if isinstance(value, (int, float)):
                                        st.write(f"  - {key.replace('_', ' ').title()}: {value:.2f}")
                                    else:
                                        st.write(f"  - {key.replace('_', ' ').title()}: {value}")
            
            with col2:
                st.subheader("📊 Technical Composite")
                
                # Technical signal interpretation
                if composite_score >= 75:
                    st.success("🟢 **Strong Bullish** - Multiple positive technical signals")
                elif composite_score >= 60:
                    st.info("🔵 **Bullish** - Generally positive technical outlook")
                elif composite_score >= 40:
                    st.warning("🟡 **Neutral** - Mixed technical signals")
                elif composite_score >= 25:
                    st.error("🟠 **Bearish** - Generally negative technical outlook")
                else:
                    st.error("🔴 **Strong Bearish** - Multiple negative technical signals")
                    
                # Show key technical metrics
                if comprehensive_technicals:
                    st.write("**Key Metrics:**")
                    rsi = comprehensive_technicals.get('rsi_14', 0)
                    st.write(f"RSI (14): {rsi:.1f}")
                    volatility = comprehensive_technicals.get('volatility_20d', 0)
                    st.write(f"Volatility (20D): {volatility:.1f}%")
            
            # Combined signal interpretation
            st.subheader("🎯 Combined Signal Strength")
            if signal_strength in ['VERY_STRONG', 'STRONG'] and composite_score >= 60:
                st.success("🚀 **STRONG CONFLUENCE** - VWV and Technical signals align bullishly")
            elif signal_strength in ['VERY_STRONG', 'STRONG'] or composite_score >= 60:
                st.info("📈 **MODERATE SIGNAL** - One strong signal present")
            else:
                st.warning("⚖️ **MIXED SIGNALS** - Conflicting or weak signals")
                
            # Risk Management Section
            if st.session_state.show_risk_management and 'risk_management' in vwv_results:
                risk_mgmt = vwv_results['risk_management']
                if risk_mgmt:
                    st.subheader("🎯 Risk Management Levels")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Stop Loss", f"${risk_mgmt.get('stop_loss_price', 0):.2f}", 
                                 f"-{risk_mgmt.get('stop_loss_pct', 2.2):.1f}%")
                    with col2:
                        st.metric("Take Profit", f"${risk_mgmt.get('take_profit_price', 0):.2f}", 
                                 f"+{risk_mgmt.get('take_profit_pct', 5.5):.1f}%")
                    with col3:
                        st.metric("Risk/Reward", f"{risk_mgmt.get('risk_reward_ratio', 2.5):.1f}:1")
                    with col4:
                        current_price = vwv_results.get('current_price', 0)
                        potential_profit = risk_mgmt.get('take_profit_price', 0) - current_price
                        st.metric("Potential Profit", f"${potential_profit:.2f}")
        else:
            st.error("❌ VWV/Technical analysis failed")

def show_volume_analysis(analysis_results, show_debug=False):
    """Display Volume Analysis - NEW v4.2.1 ENHANCED"""
    if not st.session_state.show_volume_analysis or not VOLUME_ANALYSIS_AVAILABLE:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Volume Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volume_analysis = enhanced_indicators.get('volume_analysis', {})
        
        if volume_analysis and 'error' not in volume_analysis:
            # Primary volume metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                current_volume = volume_analysis.get('current_volume', 0)
                st.metric("Current Volume", format_large_number(current_volume))
            with col2:
                volume_5d_avg = volume_analysis.get('volume_5d_avg', 0)
                st.metric("5D Avg Volume", format_large_number(volume_5d_avg))
            with col3:
                volume_ratio = volume_analysis.get('volume_ratio', 1.0)
                st.metric("Volume Ratio", f"{volume_ratio:.2f}x", "vs 30D avg")
            with col4:
                volume_score = volume_analysis.get('volume_score', 50)
                st.metric("Volume Score", f"{volume_score:.1f}/100")
            
            # Volume regime and implications
            st.subheader("📊 Volume Environment")
            volume_regime = volume_analysis.get('volume_regime', 'Unknown')
            trading_implications = volume_analysis.get('trading_implications', 'No implications available')
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Volume Regime:** {volume_regime}")
                st.info(f"**Volume Score:** {volume_score}/100")
            with col2:
                st.info(f"**Trading Implications:** {trading_implications}")
            
            # Component breakdown for enhanced analysis
            if 'indicators' in volume_analysis:
                with st.expander("🔍 Volume Component Breakdown"):
                    indicators = volume_analysis.get('indicators', {})
                    scores = volume_analysis.get('scores', {})
                    weights = volume_analysis.get('weights', {})
                    
                    breakdown_data = []
                    for indicator, value in indicators.items():
                        score = scores.get(indicator, 0)
                        weight = weights.get(indicator, 0)
                        contribution = score * weight
                        
                        breakdown_data.append({
                            'Indicator': indicator.replace('_', ' ').title(),
                            'Value': f"{value:.2f}" if isinstance(value, (int, float)) else str(value),
                            'Score': f"{score:.1f}",
                            'Weight': f"{weight:.3f}",
                            'Contribution': f"{contribution:.2f}"
                        })
                    
                    if breakdown_data:
                        df_breakdown = pd.DataFrame(breakdown_data)
                        st.dataframe(df_breakdown, use_container_width=True, hide_index=True)
                
        else:
            st.warning("⚠️ Volume analysis not available - insufficient data")

def show_volatility_analysis(analysis_results, show_debug=False):
    """Display Volatility Analysis - NEW v4.2.1 ENHANCED"""
    if not st.session_state.show_volatility_analysis or not VOLATILITY_ANALYSIS_AVAILABLE:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Volatility Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volatility_analysis = enhanced_indicators.get('volatility_analysis', {})
        
        if volatility_analysis and 'error' not in volatility_analysis:
            # Primary volatility metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                vol_20d = volatility_analysis.get('volatility_20d', 0)
                st.metric("20D Volatility", f"{vol_20d:.2f}%")
            with col2:
                vol_10d = volatility_analysis.get('volatility_10d', 0)
                st.metric("10D Volatility", f"{vol_10d:.2f}%")
            with col3:
                vol_percentile = volatility_analysis.get('volatility_percentile', 50)
                st.metric("Vol Percentile", f"{vol_percentile:.1f}%")
            with col4:
                volatility_score = volatility_analysis.get('volatility_score', 50)
                st.metric("Volatility Score", f"{volatility_score:.1f}/100")
            
            # Advanced volatility metrics
            st.subheader("🔬 Advanced Volatility Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                realized_vol = volatility_analysis.get('realized_volatility', 0)
                st.metric("Realized Vol", f"{realized_vol:.2f}%")
            with col2:
                garch_vol = volatility_analysis.get('garch_volatility', 0)
                st.metric("GARCH Vol", f"{garch_vol:.2f}%")
            with col3:
                vol_momentum = volatility_analysis.get('volatility_momentum', 0)
                st.metric("Vol Momentum", f"{vol_momentum:+.2f}%")
            with col4:
                vol_clustering = volatility_analysis.get('volatility_clustering', 0)
                st.metric("Vol Clustering", f"{vol_clustering:.3f}")
            
            # Volatility regime and options guidance
            st.subheader("📊 Volatility Environment & Options Strategy")
            vol_regime = volatility_analysis.get('volatility_regime', 'Unknown')
            options_strategy = volatility_analysis.get('options_strategy', 'No strategy available')
            trading_implications = volatility_analysis.get('trading_implications', 'No implications available')
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Volatility Regime:** {vol_regime}")
                st.info(f"**Volatility Score:** {volatility_score}/100")
            with col2:
                st.info(f"**Options Strategy:** {options_strategy}")
                st.info(f"**Trading Implications:** {trading_implications}")
            
            # Component breakdown for enhanced analysis
            if 'indicators' in volatility_analysis:
                with st.expander("🔍 Volatility Component Breakdown"):
                    st.markdown("### 📊 Weighting Methodology")
                    st.markdown("""
                    **Research-Based Indicator Weights**: Each volatility indicator is weighted based on academic research 
                    and practical trading effectiveness. Higher weights are assigned to more reliable and widely-used volatility measures.
                    """)
                    
                    indicators = volatility_analysis.get('indicators', {})
                    scores = volatility_analysis.get('scores', {})
                    weights = volatility_analysis.get('weights', {})
                    
                    breakdown_data = []
                    for indicator, value in indicators.items():
                        score = scores.get(indicator, 0)
                        weight = weights.get(indicator, 0)
                        contribution = score * weight
                        
                        breakdown_data.append({
                            'Indicator': indicator.replace('_', ' ').title(),
                            'Value': f"{value:.3f}" if isinstance(value, (int, float)) else str(value),
                            'Score': f"{score:.1f}",
                            'Weight': f"{weight:.3f}",
                            'Contribution': f"{contribution:.2f}"
                        })
                    
                    if breakdown_data:
                        df_breakdown = pd.DataFrame(breakdown_data)
                        st.dataframe(df_breakdown, use_container_width=True, hide_index=True)
                        
                        # Show risk-adjusted metrics
                        st.markdown("### 📈 Risk-Adjusted Metrics")
                        volatility_strength_factor = volatility_analysis.get('volatility_strength_factor', 1.0)
                        st.metric("Volatility Strength Factor", f"{volatility_strength_factor:.2f}x", 
                                 help="Multiplier applied to technical analysis based on volatility environment")
                
        else:
            st.warning("⚠️ Volatility analysis not available - insufficient data")

def show_fundamental_analysis(analysis_results, show_debug=False):
    """Display fundamental analysis section"""
    if not st.session_state.show_fundamental_analysis:
        return
        
    enhanced_indicators = analysis_results.get('enhanced_indicators', {})
    graham_score = enhanced_indicators.get('graham_score', {})
    piotroski_score = enhanced_indicators.get('piotroski_score', {})
    
    with st.expander(f"📈 {analysis_results['symbol']} - Fundamental Analysis", expanded=True):
        
        if 'error' not in graham_score and 'error' not in piotroski_score:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Benjamin Graham Score")
                graham_total = graham_score.get('score', 0)
                graham_max = graham_score.get('total_possible', 10)
                st.metric("Graham Score", f"{graham_total}/{graham_max}")
                
                if 'criteria' in graham_score:
                    for criterion in graham_score['criteria']:
                        status = "✅" if criterion['passed'] else "❌"
                        st.write(f"{status} {criterion['name']}: {criterion['value']}")
            
            with col2:
                st.subheader("📊 Piotroski F-Score")
                piotroski_total = piotroski_score.get('score', 0)
                piotroski_max = piotroski_score.get('total_possible', 9)
                st.metric("Piotroski Score", f"{piotroski_total}/{piotroski_max}")
                
                if 'criteria' in piotroski_score:
                    for criterion in piotroski_score['criteria']:
                        status = "✅" if criterion['passed'] else "❌"
                        st.write(f"{status} {criterion['name']}: {criterion['value']}")
        
        else:
            if 'error' in graham_score:
                st.info(f"Graham Analysis: {graham_score['error']}")
            if 'error' in piotroski_score:
                st.info(f"Piotroski Analysis: {piotroski_score['error']}")

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis section"""
    if not st.session_state.show_market_correlation:
        return
        
    enhanced_indicators = analysis_results.get('enhanced_indicators', {})
    market_correlations = enhanced_indicators.get('market_correlations', {})
    
    with st.expander(f"🌐 {analysis_results['symbol']} - Market Correlation Analysis", expanded=True):
        
        if market_correlations and 'error' not in market_correlations:
            
            # Correlation metrics
            st.subheader("📊 ETF Correlations")
            
            correlation_data = []
            for etf, data in market_correlations.items():
                if isinstance(data, dict) and 'correlation' in data:
                    correlation_data.append({
                        'ETF': etf,
                        'Correlation': f"{data.get('correlation', 0):.3f}",
                        'Beta': f"{data.get('beta', 0):.2f}",
                        'Relationship': data.get('relationship', 'Unknown')
                    })
            
            if correlation_data:
                df_correlations = pd.DataFrame(correlation_data)
                st.dataframe(df_correlations, use_container_width=True, hide_index=True)
            else:
                st.warning("No correlation data available")
                
        else:
            st.warning("⚠️ Market correlation analysis not available")

def show_options_analysis(analysis_results, show_debug=False):
    """Display options analysis section"""
    if not st.session_state.show_options_analysis:
        return
        
    enhanced_indicators = analysis_results.get('enhanced_indicators', {})
    options_levels = enhanced_indicators.get('options_levels', {})
    
    with st.expander(f"🎯 {analysis_results['symbol']} - Options Analysis", expanded=True):
        
        if options_levels and 'error' not in options_levels:
            current_price = analysis_results.get('current_price', 0)
            st.metric("Current Price", f"${current_price:.2f}")
            
            # Options levels by risk category
            for risk_level, levels in options_levels.items():
                if isinstance(levels, dict):
                    st.subheader(f"📊 {risk_level.title()} Risk Options")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Call Levels:**")
                        if 'calls' in levels:
                            for strike, data in levels['calls'].items():
                                st.write(f"${strike}: {data.get('description', 'N/A')}")
                    
                    with col2:
                        st.write("**Put Levels:**")
                        if 'puts' in levels:
                            for strike, data in levels['puts'].items():
                                st.write(f"${strike}: {data.get('description', 'N/A')}")
        else:
            st.warning("⚠️ Options analysis not available")

def show_confidence_intervals(analysis_results, show_debug=False):
    """Display confidence intervals section"""
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

def show_interactive_charts(data, analysis_results, show_debug=False):
    """Display interactive charts section"""
    if not st.session_state.show_charts:
        return
        
    with st.expander("📊 Interactive Trading Charts", expanded=True):
        try:
            # Check if we have the charts module
            try:
                display_trading_charts(data, analysis_results)
            except ImportError as e:
                st.error("📊 Charts module not available")
                if show_debug:
                    st.error(f"Import error: {str(e)}")
                
                # Fallback simple chart
                st.subheader("Basic Price Chart (Fallback)")
                if data is not None and not data.empty:
                    st.line_chart(data['Close'])
                else:
                    st.error("No data available for charting")
                    
        except Exception as e:
            if show_debug:
                st.error(f"Chart display error: {str(e)}")
                st.exception(e)
            else:
                st.warning("⚠️ Charts temporarily unavailable. Try refreshing or enable debug mode for details.")

def perform_enhanced_analysis(symbol, period, show_debug=False):
    """Perform enhanced analysis using modular components - WITH VOLUME/VOLATILITY INTEGRATION"""
    try:
        # Step 1: Fetch data using modular data fetcher
        market_data = get_market_data_enhanced(symbol, period, show_debug)
        
        if market_data is None:
            st.error(f"❌ Could not fetch data for {symbol}")
            return None, None, None
        
        # Step 2: Store data using data manager
        data_manager = get_data_manager()
        data_manager.store_market_data(symbol, market_data, show_debug)
        
        # Step 3: Get analysis copy
        analysis_input = data_manager.get_market_data_for_analysis(symbol)
        
        if analysis_input is None:
            st.error("❌ Could not prepare analysis data")
            return None, None, None
        
        # Step 4: Calculate enhanced indicators using modular analysis
        daily_vwap = calculate_daily_vwap(analysis_input)
        fibonacci_emas = calculate_fibonacci_emas(analysis_input)
        point_of_control = calculate_point_of_control_enhanced(analysis_input)
        weekly_deviations = calculate_weekly_deviations(analysis_input)
        comprehensive_technicals = calculate_comprehensive_technicals(analysis_input)
        
        # Step 5: Calculate Volume Analysis - NEW v4.2.1 WITH ENHANCEMENTS
        volume_analysis = {}
        if VOLUME_ANALYSIS_AVAILABLE:
            try:
                volume_analysis = calculate_complete_volume_analysis(analysis_input)
                if show_debug:
                    st.write("✅ Volume analysis completed")
            except Exception as e:
                if show_debug:
                    st.write(f"❌ Volume analysis failed: {e}")
                volume_analysis = {'error': 'Volume analysis failed'}
        else:
            volume_analysis = {'error': 'Volume analysis module not available'}
        
        # Step 6: Calculate Volatility Analysis - NEW v4.2.1 WITH ENHANCEMENTS
        volatility_analysis = {}
        if VOLATILITY_ANALYSIS_AVAILABLE:
            try:
                volatility_analysis = calculate_complete_volatility_analysis(analysis_input)
                if show_debug:
                    st.write("✅ Volatility analysis completed")
            except Exception as e:
                if show_debug:
                    st.write(f"❌ Volatility analysis failed: {e}")
                volatility_analysis = {'error': 'Volatility analysis failed'}
        else:
            volatility_analysis = {'error': 'Volatility analysis module not available'}
        
        # Step 7: Calculate market correlations
        market_correlations = calculate_market_correlations_enhanced(analysis_input, symbol, show_debug=show_debug)
        
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
        
        # Step 10: Calculate VWV System Analysis
        vwv_results = calculate_vwv_system_complete(analysis_input, symbol, DEFAULT_VWV_CONFIG)
        
        # Step 11: Calculate confidence intervals
        confidence_analysis = calculate_confidence_intervals(analysis_input)
        
        # Step 12: Build analysis results - INCLUDES ENHANCED VOLUME/VOLATILITY
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
                'volume_analysis': volume_analysis,  # NEW v4.2.1 - ENHANCED
                'volatility_analysis': volatility_analysis,  # NEW v4.2.1 - ENHANCED
                'market_correlations': market_correlations,
                'options_levels': options_levels,
                'graham_score': graham_score,
                'piotroski_score': piotroski_score
            },
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
    """Main application function - WITH ENHANCED VOLUME/VOLATILITY ANALYSIS"""
    # Create header using modular component
    create_header()
    
    # Create sidebar and get controls
    controls = create_sidebar_controls()
    
    # Main logic flow
    if controls['analyze_button'] and controls['symbol']:
        # Add symbol to recently viewed
        add_to_recently_viewed(controls['symbol'])
        
        st.write("## 📊 VWV Trading Analysis v4.2.1 Enhanced")
        
        with st.spinner(f"Analyzing {controls['symbol']}..."):
            
            # Perform analysis using modular components
            analysis_results, chart_data, vwv_results = perform_enhanced_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results and chart_data is not None:
                
                # CORRECTED DISPLAY ORDER - Charts First, VWV/Technical Second, Volume/Volatility Third/Fourth
                
                # 1. Show charts FIRST at the top
                show_interactive_charts(chart_data, analysis_results, controls['show_debug'])
                
                # 2. Show VWV/Technical analysis SECOND
                show_combined_vwv_technical_analysis(analysis_results, vwv_results, controls['show_debug'])
                
                # 3. Show Volume Analysis THIRD - NEW v4.2.1 ENHANCED
                if VOLUME_ANALYSIS_AVAILABLE:
                    show_volume_analysis(analysis_results, controls['show_debug'])
                
                # 4. Show Volatility Analysis FOURTH - NEW v4.2.1 ENHANCED
                if VOLATILITY_ANALYSIS_AVAILABLE:
                    show_volatility_analysis(analysis_results, controls['show_debug'])
                
                # 5. Show all other analysis sections using modular functions
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                show_options_analysis(analysis_results, controls['show_debug'])
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
                # Debug information
                if controls['show_debug']:
                    with st.expander("🔧 Debug Information", expanded=False):
                        st.write("**System Status:**")
                        st.write(f"- Volume Analysis Available: {VOLUME_ANALYSIS_AVAILABLE}")
                        st.write(f"- Volatility Analysis Available: {VOLATILITY_ANALYSIS_AVAILABLE}")
                        st.write(f"- Analysis Results Keys: {list(analysis_results.keys())}")
                        if 'enhanced_indicators' in analysis_results:
                            st.write(f"- Enhanced Indicators Keys: {list(analysis_results['enhanced_indicators'].keys())}")
                        st.write("**VWV Results:**")
                        st.json(vwv_results)
            else:
                st.error("❌ Analysis failed. Please try a different symbol or enable debug mode for details.")
    else:
        # Show market status and welcome message
        st.write("## 📊 VWV Professional Trading System v4.2.1 Enhanced")
        
        market_status = get_market_status()
        if market_status:
            st.info(f"🕒 **Market Status:** {market_status}")
        
        st.info("👈 **Enter a stock symbol in the sidebar to begin comprehensive analysis**")
        
        # Show recently viewed symbols if available
        if len(st.session_state.recently_viewed) > 0:
            st.subheader("🕒 Recently Analyzed Symbols")
            
            # Create a grid of recently viewed symbols
            cols = st.columns(min(6, len(st.session_state.recently_viewed)))
            for i, recent_symbol in enumerate(st.session_state.recently_viewed[:6]):
                with cols[i]:
                    if st.button(f"📊 {recent_symbol}", key=f"main_recent_{recent_symbol}", use_container_width=True):
                        st.session_state.selected_symbol = recent_symbol
                        st.session_state.auto_analyze = True
                        st.rerun()

if __name__ == "__main__":
    main()
