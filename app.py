"""
VWV Professional Trading System - Quick Fix Version
Fixes: Default period, Breakout analysis, Adds VWV Core
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
from analysis.options import (
    calculate_options_levels_enhanced,
    calculate_confidence_intervals
)

from ui.components import (
    create_technical_score_bar,
    create_header
)
from utils.helpers import format_large_number, get_market_status, get_etf_description
from utils.decorators import safe_calculation_wrapper

# FIXED MODULES
from analysis.market import calculate_market_correlations_enhanced
from analysis.vwv_core import calculate_vwv_confluence_score, calculate_vwv_risk_management

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

# Page configuration
st.set_page_config(
    page_title="VWV Professional Trading System v5.0",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# FIXED BREAKOUT ANALYSIS FUNCTION
@safe_calculation_wrapper
def calculate_enhanced_breakout_analysis_fixed(symbols=['SPY', 'QQQ', 'IWM'], show_debug=False):
    """FIXED: Enhanced breakout/breakdown analysis"""
    try:
        import yfinance as yf
        results = {}
        
        for symbol in symbols:
            try:
                # Get 3 months of data
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='3mo')
                
                if len(data) < 50:
                    continue
                    
                current_price = data['Close'].iloc[-1]
                current_volume = data['Volume'].iloc[-1]
                
                # Method 1: Moving Average Analysis
                ma_20 = data['Close'].rolling(20).mean().iloc[-1]
                ma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else ma_20
                
                ma_score = 0
                if current_price > ma_20 * 1.005:  # 0.5% above 20MA
                    ma_score += 1
                if current_price > ma_50 * 1.01:   # 1% above 50MA  
                    ma_score += 1
                if ma_20 > ma_50:  # Bullish MA alignment
                    ma_score += 1
                
                # Method 2: Range Analysis
                range_score = 0
                for period in [5, 10, 20]:
                    if len(data) >= period + 2:
                        recent_high = data['High'].iloc[-(period+1):-1].max()
                        recent_low = data['Low'].iloc[-(period+1):-1].min()
                        
                        if current_price > recent_high * 1.002:  # 0.2% above high
                            range_score += 1
                        elif current_price < recent_low * 0.998:  # 0.2% below low
                            range_score -= 1
                
                # Method 3: Volume Confirmation
                avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                volume_score = 0
                
                if current_volume > avg_volume * 1.2:
                    volume_score += 1
                elif current_volume < avg_volume * 0.8:
                    volume_score -= 0.5
                
                # Composite scoring
                composite_raw = (ma_score * 0.4 + range_score * 0.4 + volume_score * 0.2)
                
                # Convert to percentages
                if composite_raw > 0:
                    breakout_ratio = min(100, composite_raw * 25)
                    breakdown_ratio = 0
                elif composite_raw < 0:
                    breakout_ratio = 0
                    breakdown_ratio = min(100, abs(composite_raw) * 25)
                else:
                    breakout_ratio = 0
                    breakdown_ratio = 0
                
                net_ratio = breakout_ratio - breakdown_ratio
                
                if net_ratio > 50:
                    signal_strength = "Very Bullish"
                elif net_ratio > 20:
                    signal_strength = "Bullish"
                elif net_ratio > -20:
                    signal_strength = "Neutral"
                elif net_ratio > -50:
                    signal_strength = "Bearish"
                else:
                    signal_strength = "Very Bearish"
                
                results[symbol] = {
                    'current_price': round(current_price, 2),
                    'breakout_ratio': round(breakout_ratio, 1),
                    'breakdown_ratio': round(breakdown_ratio, 1),
                    'net_ratio': round(net_ratio, 1),
                    'signal_strength': signal_strength,
                    'ma_20': round(ma_20, 2),
                    'ma_50': round(ma_50, 2),
                    'volume_ratio': round(current_volume / avg_volume, 2) if avg_volume > 0 else 1.0
                }
                
            except Exception as e:
                if show_debug:
                    st.write(f"Error analyzing {symbol}: {e}")
                continue
        
        # Overall market sentiment
        if results:
            overall_breakout = sum([results[idx]['breakout_ratio'] for idx in results]) / len(results)
            overall_breakdown = sum([results[idx]['breakdown_ratio'] for idx in results]) / len(results)
            overall_net = overall_breakout - overall_breakdown
            
            if overall_net > 40:
                market_regime = "🚀 Strong Breakout Environment"
            elif overall_net > 15:
                market_regime = "📈 Bullish Breakout Bias"
            elif overall_net > -15:
                market_regime = "⚖️ Balanced Market"
            elif overall_net > -40:
                market_regime = "📉 Bearish Breakdown Bias"
            else:
                market_regime = "🔻 Strong Breakdown Environment"
            
            results['OVERALL'] = {
                'breakout_ratio': round(overall_breakout, 1),
                'breakdown_ratio': round(overall_breakdown, 1),
                'net_ratio': round(overall_net, 1),
                'market_regime': market_regime,
                'sample_size': len(results)
            }
        
        return results
        
    except Exception as e:
        st.error(f"Enhanced breakout analysis error: {e}")
        return {}

def create_sidebar_controls():
    """Create sidebar controls and return analysis parameters"""
    st.sidebar.title("📊 VWV Trading Analysis v5.0")
    
    # Initialize session state
    if 'recently_viewed' not in st.session_state:
        st.session_state.recently_viewed = []
    if 'show_vwv_core_analysis' not in st.session_state:
        st.session_state.show_vwv_core_analysis = True
    if 'show_technical_analysis' not in st.session_state:
        st.session_state.show_technical_analysis = True
    if 'show_fundamental_analysis' not in st.session_state:
        st.session_state.show_fundamental_analysis = True
    if 'show_market_correlation' not in st.session_state:
        st.session_state.show_market_correlation = True
    if 'show_options_analysis' not in st.session_state:
        st.session_state.show_options_analysis = True
    if 'show_confidence_intervals' not in st.session_state:
        st.session_state.show_confidence_intervals = True
    
    # Basic controls
    if 'selected_symbol' in st.session_state:
        default_symbol = st.session_state.selected_symbol
        del st.session_state.selected_symbol
    else:
        default_symbol = UI_SETTINGS['default_symbol']
        
    symbol = st.sidebar.text_input("Symbol", value=default_symbol, help="Enter stock symbol").upper()
    # FIXED: Default to 3mo (index 1, not index 3)
    period = st.sidebar.selectbox("Data Period", UI_SETTINGS['periods'], index=1)
    
    # Section Control Panel
    with st.sidebar.expander("📋 Analysis Sections", expanded=False):
        st.write("**Toggle Analysis Sections:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.show_vwv_core_analysis = st.checkbox(
                "VWV Core (NEW)", 
                value=st.session_state.show_vwv_core_analysis,
                key="toggle_vwv_core"
            )
            st.session_state.show_technical_analysis = st.checkbox(
                "Technical Analysis", 
                value=st.session_state.show_technical_analysis,
                key="toggle_technical"
            )
            st.session_state.show_fundamental_analysis = st.checkbox(
                "Fundamental Analysis", 
                value=st.session_state.show_fundamental_analysis,
                key="toggle_fundamental"
            )
        
        with col2:
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
    """Add symbol to recently viewed"""
    if symbol and symbol != "":
        if symbol in st.session_state.recently_viewed:
            st.session_state.recently_viewed.remove(symbol)
        st.session_state.recently_viewed.insert(0, symbol)
        st.session_state.recently_viewed = st.session_state.recently_viewed[:9]

def show_vwv_core_analysis(analysis_results, show_debug=False):
    """Display VWV Core Williams VIX Fix analysis section - NEW"""
    if not st.session_state.show_vwv_core_analysis:
        return
        
    with st.expander(f"🎯 {analysis_results['symbol']} - VWV Core Signal Analysis (Williams VIX Fix)", expanded=True):
        
        vwv_data = analysis_results.get('vwv_analysis', {})
        
        if 'error' in vwv_data:
            st.error(f"❌ VWV Core Analysis Error: {vwv_data['error']}")
            return
        
        # VWV Core metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            vwv_score = vwv_data.get('final_score', 0)
            signal_class = vwv_data.get('signal_classification', 'WEAK')
            st.metric("VWV Confluence Score", f"{vwv_score:.2f}", f"Signal: {signal_class}")
        
        with col2:
            wvf_raw = vwv_data.get('component_details', {}).get('wvf_raw', 0)
            st.metric("Williams VIX Fix", f"{wvf_raw:.2f}", "Fear Gauge")
        
        with col3:
            ma_confluence = vwv_data.get('component_details', {}).get('ma_confluence', 0)
            st.metric("MA Confluence", f"{ma_confluence:.2f}", "Trend Alignment")
        
        with col4:
            momentum = vwv_data.get('component_details', {}).get('momentum_component', 0)
            st.metric("Momentum Component", f"{momentum:.2f}", "Oversold Detection")
        
        # Signal interpretation
        if vwv_score >= 5.5:
            st.success("🟢 **VERY STRONG VWV SIGNAL** - Excellent confluence conditions detected")
        elif vwv_score >= 4.5:
            st.success("🟡 **STRONG VWV SIGNAL** - Good confluence conditions")
        elif vwv_score >= 3.5:
            st.info("🔵 **GOOD VWV SIGNAL** - Moderate confluence conditions")
        else:
            st.warning("🟠 **WEAK VWV SIGNAL** - Limited confluence detected")

def show_individual_technical_analysis(analysis_results, show_debug=False):
    """Display individual technical analysis section"""
    if not st.session_state.show_technical_analysis:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Individual Technical Analysis", expanded=True):
        
        # COMPOSITE TECHNICAL SCORE - Use modular component
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

def show_fundamental_analysis(analysis_results, show_debug=False):
    """Display fundamental analysis section"""
    if not st.session_state.show_fundamental_analysis:
        return
        
    with st.expander("📊 Fundamental Analysis - Value Investment Scores", expanded=True):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        graham_data = enhanced_indicators.get('graham_score', {})
        piotroski_data = enhanced_indicators.get('piotroski_score', {})
        
        # Check if symbol is ETF
        is_etf_symbol = ('ETF' in graham_data.get('error', '') or 
                         'ETF' in piotroski_data.get('error', ''))
        
        if not is_etf_symbol and ('error' not in graham_data or 'error' not in piotroski_data):
            
            # Display scores overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'error' not in graham_data:
                    st.metric(
                        "Graham Score", 
                        f"{graham_data.get('score', 0)}/10",
                        f"Grade: {graham_data.get('grade', 'N/A')}"
                    )
                else:
                    st.metric("Graham Score", "N/A", "Data Limited")
            
            with col2:
                if 'error' not in piotroski_data:
                    st.metric(
                        "Piotroski F-Score", 
                        f"{piotroski_data.get('score', 0)}/9",
                        f"Grade: {piotroski_data.get('grade', 'N/A')}"
                    )
                else:
                    st.metric("Piotroski F-Score", "N/A", "Data Limited")
            
            with col3:
                if 'error' not in graham_data:
                    st.metric(
                        "Graham %", 
                        f"{graham_data.get('percentage', 0):.0f}%",
                        graham_data.get('interpretation', '')[:20] + "..."
                    )
                else:
                    st.metric("Graham %", "0%", "No Data")
            
            with col4:
                if 'error' not in piotroski_data:
                    st.metric(
                        "Piotroski %", 
                        f"{piotroski_data.get('percentage', 0):.0f}%",
                        piotroski_data.get('interpretation', '')[:20] + "..."
                    )
                else:
                    st.metric("Piotroski %", "0%", "No Data")
        
        elif is_etf_symbol:
            st.info(f"ℹ️ **{analysis_results['symbol']} is an ETF** - Fundamental analysis is not applicable to Exchange-Traded Funds.")

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis section"""
    if not st.session_state.show_market_correlation:
        return
        
    with st.expander("🌐 Market Correlation & Breakout Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        market_correlations = enhanced_indicators.get('market_correlations', {})
        
        if market_correlations:
            st.subheader("📊 ETF Correlation Analysis")
            
            correlation_table_data = []
            for etf, etf_data in market_correlations.items():
                correlation_table_data.append({
                    'ETF': etf,
                    'Correlation': f"{etf_data.get('correlation', 0):.3f}",
                    'Beta': f"{etf_data.get('beta', 0):.3f}",
                    'Relationship': etf_data.get('relationship', 'Unknown'),
                    'Description': get_etf_description(etf)
                })
            
            df_correlations = pd.DataFrame(correlation_table_data)
            st.dataframe(df_correlations, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Market correlation data not available")
        
        # FIXED Breakout/breakdown analysis
        st.subheader("📊 FIXED Breakout/Breakdown Analysis")
        breakout_data = calculate_enhanced_breakout_analysis_fixed(['SPY', 'QQQ', 'IWM'], show_debug=show_debug)
        
        if breakout_data:
            # Overall market sentiment
            overall_data = breakout_data.get('OVERALL', {})
            if overall_data:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Market Breakouts", f"{overall_data['breakout_ratio']}%")
                with col2:
                    st.metric("Market Breakdowns", f"{overall_data['breakdown_ratio']}%")
                with col3:
                    net_ratio = overall_data['net_ratio']
                    st.metric("Net Bias", f"{net_ratio:+.1f}%", 
                             "📈 Bullish" if net_ratio > 0 else "📉 Bearish" if net_ratio < 0 else "⚖️ Neutral")
                with col4:
                    st.metric("Market Regime", overall_data.get('market_regime', 'Unknown'))

def show_options_analysis(analysis_results, show_debug=False):
    """Display options analysis section"""
    if not st.session_state.show_options_analysis:
        return
        
    with st.expander("🎯 Options Trading Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        options_levels = enhanced_indicators.get('options_levels', [])
        
        if options_levels:
            st.subheader("💰 Premium Selling Levels with Greeks")
            st.write("**Enhanced option strike levels with Delta, Theta, and Beta**")
            
            df_options = pd.DataFrame(options_levels)
            st.dataframe(df_options, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Options analysis not available - insufficient data")

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
        
        # Step 5: Calculate market correlations
        market_correlations = calculate_market_correlations_enhanced(analysis_input, symbol, show_debug=show_debug)
        
        # Step 6: NEW - Calculate VWV Core Analysis
        current_price = round(float(analysis_input['Close'].iloc[-1]), 2)
        vwv_score, vwv_details = calculate_vwv_confluence_score(analysis_input)
        vwv_risk_management = calculate_vwv_risk_management(analysis_input, vwv_details.get('signal_classification', 'WEAK'), current_price)
        
        # Step 7: Calculate fundamental analysis (skip for ETFs)
        is_etf_symbol = is_etf(symbol)
        
        if is_etf_symbol:
            graham_score = {'score': 0, 'total_possible': 10, 'criteria': [], 'error': 'ETF - Fundamental analysis not applicable'}
            piotroski_score = {'score': 0, 'total_possible': 9, 'criteria': [], 'error': 'ETF - Fundamental analysis not applicable'}
        else:
            graham_score = calculate_graham_score(symbol, show_debug)
            piotroski_score = calculate_piotroski_score(symbol, show_debug)
        
        # Step 8: Calculate options levels
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
        
        options_levels = calculate_options_levels_enhanced(current_price, volatility, underlying_beta=underlying_beta)
        
        # Step 9: Calculate confidence intervals
        confidence_analysis = calculate_confidence_intervals(analysis_input)
        
        # Step 10: Build analysis results
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
            'vwv_analysis': {
                'final_score': vwv_score,
                'component_details': vwv_details,
                'signal_classification': vwv_details.get('signal_classification', 'WEAK')
            },
            'vwv_risk_management': vwv_risk_management,
            'confidence_analysis': confidence_analysis,
            'system_status': 'VWV_v5.0_OPERATIONAL'
        }
        
        # Store results
        data_manager.store_analysis_results(symbol, analysis_results)
        
        return analysis_results
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        return None

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
        
        st.write("## 📊 VWV Trading Analysis v5.0")
        
        with st.spinner(f"Analyzing {controls['symbol']}..."):
            
            # Perform analysis using modular components
            analysis_results = perform_enhanced_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results:
                # Show all analysis sections using modular functions
                show_vwv_core_analysis(analysis_results, controls['show_debug'])
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                show_options_analysis(analysis_results, controls['show_debug'])
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
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
        st.write("## 🚀 VWV Professional Trading System v5.0 - Quick Fix Release")
        st.write("**Fixed:** Default period (3mo), Breakout analysis (no more 0%), Added VWV Core system")
        
        with st.expander("🔧 v5.0 Quick Fixes Applied", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### ✅ **Fixes Applied**")
                st.write("✅ **Default Period** - Now defaults to 3mo instead of 1y")
                st.write("✅ **Breakout Analysis** - Fixed 0% issue with multi-factor logic")
                st.write("✅ **VWV Core System** - Added Williams VIX Fix 6-component analysis")
                st.write("✅ **Requirements** - Added scipy for enhanced calculations")
                
            with col2:
                st.write("### 🎯 **Working Sections**")
                st.write("• **VWV Core Signals** - Williams VIX Fix confluence system")
                st.write("• **Individual Technical** - Composite scoring with enhanced indicators")
                st.write("• **Fundamental Analysis** - Graham & Piotroski scores")
                st.write("• **Market Correlation** - ETF relationship analysis with FIXED breakouts")
                st.write("• **Options Analysis** - Strike levels with Greeks")
                st.write("• **Statistical Intervals** - Confidence level calculations")
        
        # Show current market status
        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.write("1. **Enter a symbol** in the sidebar (e.g., AAPL, SPY, QQQ)")
            st.write("2. **Period will default to 3mo** - optimal for analysis")
            st.write("3. **Click 'Analyze Symbol'** to run complete analysis")
            st.write("4. **View NEW VWV Core section** - Williams VIX Fix system")
            st.write("5. **Check FIXED breakouts** - now shows actual percentages")

    # Footer
    st.markdown("---")
    st.write("### 📊 System Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Version:** VWV Professional v5.0 - Quick Fix")
        st.write(f"**Status:** ✅ Core Issues Fixed")
    with col2:
        st.write(f"**Default Period:** 3mo (Fixed)")
        st.write(f"**Breakouts:** Multi-factor logic (Fixed)")
    with col3:
        st.write(f"**VWV Core:** Williams VIX Fix (NEW)")
        st.write(f"**Requirements:** scipy added")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.write("Please refresh the page and try again.")
        
        if st.checkbox("Show Error Details"):
            st.exception(e)
