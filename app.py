"""
VWV Professional Trading System v4.2.1 - FIXED NAVIGATION + COMPLETE ANALYSIS
All core functionality preserved with working quick links and analyze button
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Essential imports
from config.settings import UI_SETTINGS
from config.constants import SYMBOL_DESCRIPTIONS, QUICK_LINK_CATEGORIES
from data.manager import get_data_manager
from data.fetcher import get_market_data_enhanced, is_etf
from analysis.technical import (
    calculate_daily_vwap, 
    calculate_fibonacci_emas,
    calculate_point_of_control_enhanced,
    calculate_comprehensive_technicals,
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

from ui.components import (
    create_technical_score_bar,
    create_header
)
from utils.helpers import format_large_number, get_market_status, get_etf_description

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
    """Create sidebar controls - FIXED navigation logic"""
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
    if 'auto_analyze' not in st.session_state:
        st.session_state.auto_analyze = False
    
    # Handle quick link selection - FIXED LOGIC
    if 'selected_symbol' in st.session_state:
        default_symbol = st.session_state.selected_symbol
        st.session_state.auto_analyze = True  # Set flag to trigger analysis
        # Don't delete selected_symbol yet - let the main function handle it
    else:
        default_symbol = UI_SETTINGS['default_symbol']
        
    # Symbol input and period selection
    symbol = st.sidebar.text_input("Symbol", value=default_symbol, help="Enter stock symbol", key="symbol_input").upper()
    period = st.sidebar.selectbox("Data Period", UI_SETTINGS['periods'], index=1)
    
    # Main analyze button
    analyze_button = st.sidebar.button("📊 Analyze Symbol", type="primary", use_container_width=True)
    
    # Check for auto-analyze trigger
    if st.session_state.auto_analyze:
        st.session_state.auto_analyze = False  # Reset flag
        analyze_button = True  # Trigger analysis
        # Now delete selected_symbol after using it
        if 'selected_symbol' in st.session_state:
            del st.session_state.selected_symbol
    
    # Section toggles
    with st.sidebar.expander("📋 Analysis Sections", expanded=False):
        st.write("**Toggle Analysis Sections:**")
        
        col1, col2 = st.columns(2)
        with col1:
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

    # Quick Links section - FIXED to auto-analyze
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

def show_individual_technical_analysis(analysis_results, show_debug=False):
    """Display individual technical analysis section - PRESERVED FUNCTIONALITY"""
    if not st.session_state.show_technical_analysis:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Individual Technical Analysis", expanded=True):
        
        # CRITICAL: Technical score bar must display
        composite_score, score_details = calculate_composite_technical_score(analysis_results)
        score_bar_html = create_technical_score_bar(composite_score, score_details)
        st.components.v1.html(score_bar_html, height=110)
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
        
        # Primary metrics
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
        
        # CRITICAL: Technical indicators table with Fibonacci EMAs
        st.subheader("📋 Technical Indicators")
        current_price = analysis_results['current_price']
        daily_vwap = enhanced_indicators.get('daily_vwap', 0)
        point_of_control = enhanced_indicators.get('point_of_control', 0)

        indicators_data = []
        
        # Current Price
        indicators_data.append(("Current Price", f"${current_price:.2f}", "📍 Reference", "0.0%", "Current"))
        
        # Daily VWAP
        vwap_distance = f"{((current_price - daily_vwap) / daily_vwap * 100):+.2f}%" if daily_vwap > 0 else "N/A"
        vwap_status = "Above" if current_price > daily_vwap else "Below"
        indicators_data.append(("Daily VWAP", f"${daily_vwap:.2f}", "📊 Volume Weighted", vwap_distance, vwap_status))
        
        # Point of Control
        poc_distance = f"{((current_price - point_of_control) / point_of_control * 100):+.2f}%" if point_of_control > 0 else "N/A"
        poc_status = "Above" if current_price > point_of_control else "Below"
        indicators_data.append(("Point of Control", f"${point_of_control:.2f}", "📊 Volume Profile", poc_distance, poc_status))
        
        # CRITICAL: Fibonacci EMAs must display
        for ema_name, ema_value in fibonacci_emas.items():
            period = ema_name.split('_')[1]
            distance_pct = f"{((current_price - ema_value) / ema_value * 100):+.2f}%" if ema_value > 0 else "N/A"
            status = "Above" if current_price > ema_value else "Below"
            indicators_data.append((f"EMA {period}", f"${ema_value:.2f}", "📈 Trend", distance_pct, status))
        
        df_technical = pd.DataFrame(indicators_data, columns=['Indicator', 'Value', 'Type', 'Distance %', 'Status'])
        st.dataframe(df_technical, use_container_width=True, hide_index=True)

def show_volume_analysis(analysis_results, show_debug=False):
    """Display volume analysis section"""
    if not st.session_state.show_volume_analysis or not VOLUME_ANALYSIS_AVAILABLE:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Volume Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volume_analysis = enhanced_indicators.get('volume_analysis', {})
        
        if 'error' not in volume_analysis and volume_analysis:
            # Primary volume metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Volume", format_large_number(volume_analysis.get('current_volume', 0)))
            with col2:
                st.metric("5D Avg Volume", format_large_number(volume_analysis.get('volume_5d_avg', 0)))
            with col3:
                volume_ratio = volume_analysis.get('volume_ratio', 1.0)
                st.metric("Volume Ratio", f"{volume_ratio:.2f}x", f"vs 30D avg")
            with col4:
                volume_trend = volume_analysis.get('volume_5d_trend', 0)
                st.metric("5D Volume Trend", f"{volume_trend:+.2f}%")
            
            # Volume regime and implications
            st.subheader("📊 Volume Environment")
            volume_regime = volume_analysis.get('volume_regime', 'Unknown')
            trading_implications = volume_analysis.get('trading_implications', 'No implications available')
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Volume Regime:** {volume_regime}")
                st.info(f"**Volume Score:** {volume_analysis.get('volume_score', 50)}/100")
            with col2:
                st.info(f"**Trading Implications:**\n{trading_implications}")
                
        else:
            st.warning("⚠️ Volume analysis not available - insufficient data")

def show_volatility_analysis(analysis_results, show_debug=False):
    """Display volatility analysis section"""
    if not st.session_state.show_volatility_analysis or not VOLATILITY_ANALYSIS_AVAILABLE:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Volatility Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volatility_analysis = enhanced_indicators.get('volatility_analysis', {})
        
        if 'error' not in volatility_analysis and volatility_analysis:
            # Primary volatility metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                vol_5d = volatility_analysis.get('volatility_5d', 0)
                st.metric("5D Volatility", f"{vol_5d:.2f}%")
            with col2:
                vol_30d = volatility_analysis.get('volatility_30d', 0)
                st.metric("30D Volatility", f"{vol_30d:.2f}%")
            with col3:
                vol_percentile = volatility_analysis.get('volatility_percentile', 50)
                st.metric("Vol Percentile", f"{vol_percentile:.1f}%")
            with col4:
                vol_trend = volatility_analysis.get('volatility_trend', 0)
                st.metric("Vol Trend", f"{vol_trend:+.2f}%")
            
            # Volatility regime and options guidance
            st.subheader("📊 Volatility Environment & Options Strategy")
            vol_regime = volatility_analysis.get('volatility_regime', 'Unknown')
            options_strategy = volatility_analysis.get('options_strategy', 'No strategy available')
            trading_implications = volatility_analysis.get('trading_implications', 'No implications available')
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Volatility Regime:** {vol_regime}")
                st.info(f"**Volatility Score:** {volatility_analysis.get('volatility_score', 50)}/100")
            with col2:
                st.info(f"**Options Strategy:** {options_strategy}")
                st.info(f"**Trading Implications:**\n{trading_implications}")
                
        else:
            st.warning("⚠️ Volatility analysis not available - insufficient data")

def show_fundamental_analysis(analysis_results, show_debug=False):
    """Display fundamental analysis section - PRESERVED FUNCTIONALITY"""
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
    """Display market correlation analysis section - PRESERVED FUNCTIONALITY"""
    if not st.session_state.show_market_correlation:
        return
        
    with st.expander("🌐 Market Correlation & Comparison Analysis", expanded=True):
        
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
        
        # Breakout/breakdown analysis
        st.subheader("📊 Breakout/Breakdown Analysis")
        breakout_data = calculate_breakout_breakdown_analysis(show_debug=show_debug)
        
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
    """Display options analysis section - PRESERVED FUNCTIONALITY"""
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
            
            # Options context
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("**Put Selling Strategy:**\n"
                        "• Sell puts below current price\n"
                        "• Collect premium if stock stays above strike\n"
                        "• Delta: Price sensitivity (~-0.16)\n"
                        "• Theta: Daily time decay")
            
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
            st.warning("⚠️ Options analysis not available - insufficient data")

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

def perform_complete_analysis(symbol, period, show_debug=False):
    """Perform complete analysis with Volume & Volatility if available"""
    try:
        # Step 1: Fetch data
        market_data = get_market_data_enhanced(symbol, period, show_debug)
        
        if market_data is None:
            st.error(f"❌ Could not fetch data for {symbol}")
            return None
        
        # Step 2: Store data
        data_manager = get_data_manager()
        data_manager.store_market_data(symbol, market_data, show_debug)
        
        # Step 3: Get analysis copy
        analysis_input = data_manager.get_market_data_for_analysis(symbol)
        
        if analysis_input is None:
            st.error("❌ Could not prepare analysis data")
            return None
        
        # Step 4: Calculate core indicators
        daily_vwap = calculate_daily_vwap(analysis_input)
        fibonacci_emas = calculate_fibonacci_emas(analysis_input)
        point_of_control = calculate_point_of_control_enhanced(analysis_input)
        comprehensive_technicals = calculate_comprehensive_technicals(analysis_input)
        
        # Step 5: Calculate Volume Analysis if available
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
        
        # Step 6: Calculate Volatility Analysis if available
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
        
        # Step 10: Calculate confidence intervals
        confidence_analysis = calculate_confidence_intervals(analysis_input)
        
        # Step 11: Build results
        current_date = analysis_input.index[-1].strftime('%Y-%m-%d')
        
        enhanced_indicators = {
            'daily_vwap': daily_vwap,
            'fibonacci_emas': fibonacci_emas,
            'point_of_control': point_of_control,
            'comprehensive_technicals': comprehensive_technicals,
            'market_correlations': market_correlations,
            'options_levels': options_levels,
            'graham_score': graham_score,
            'piotroski_score': piotroski_score
        }
        
        # Add Volume & Volatility if available
        if volume_analysis:
            enhanced_indicators['volume_analysis'] = volume_analysis
        if volatility_analysis:
            enhanced_indicators['volatility_analysis'] = volatility_analysis
        
        analysis_results = {
            'symbol': symbol,
            'timestamp': current_date,
            'current_price': current_price,
            'enhanced_indicators': enhanced_indicators,
            'confidence_analysis': confidence_analysis,
            'system_status': 'COMPLETE_OPERATIONAL'
        }
        
        # Store results
        data_manager.store_analysis_results(symbol, analysis_results)
        
        return analysis_results
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        return None

def main():
    """Main application - COMPLETE with fixed navigation"""
    # Create header
    create_header()
    
    # Create sidebar and get controls
    controls = create_sidebar_controls()
    
    # Main logic flow
    if controls['analyze_button'] and controls['symbol']:
        # Add to recently viewed
        add_to_recently_viewed(controls['symbol'])
        
        st.write(f"## 📊 VWV Trading Analysis - {controls['symbol']}")
        
        with st.spinner(f"Analyzing {controls['symbol']}..."):
            
            # Perform complete analysis
            analysis_results = perform_complete_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results:
                # Show all analysis sections
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                
                # Volume & Volatility if available
                if VOLUME_ANALYSIS_AVAILABLE:
                    show_volume_analysis(analysis_results, controls['show_debug'])
                if VOLATILITY_ANALYSIS_AVAILABLE:
                    show_volatility_analysis(analysis_results, controls['show_debug'])
                
                # Remaining sections
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                show_options_analysis(analysis_results, controls['show_debug'])
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
                # Success message
                st.success("✅ Complete analysis finished! All sections operational.")
                
                # Debug information
                if controls['show_debug']:
                    with st.expander("🐛 Debug Information", expanded=False):
                        st.write("### Analysis Results Structure")
                        st.json(analysis_results, expanded=False)
                        st.write("### System Status")
                        st.write(f"**Volume Analysis Available:** {VOLUME_ANALYSIS_AVAILABLE}")
                        st.write(f"**Volatility Analysis Available:** {VOLATILITY_ANALYSIS_AVAILABLE}")
    
    else:
        # Welcome message
        st.write("## 🚀 VWV Professional Trading System v4.2.1 Enhanced")
        st.write("**Complete modular architecture with Volume & Volatility analysis**")
        
        with st.expander("🏗️ System Status v4.2.1", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📁 **Core Modules**")
                st.write("✅ **config/** - Settings, constants, parameters")
                st.write("✅ **data/** - Fetching, validation, management")
                st.write("✅ **analysis/** - Technical, volume, volatility, fundamental, market, options")
                st.write("✅ **ui/** - Components, headers, score bars")
                st.write("✅ **utils/** - Helpers, decorators, formatters")
                
            with col2:
                st.write("### 🎯 **All Sections Working**")
                st.write("• **Individual Technical Analysis** ✅")
                if VOLUME_ANALYSIS_AVAILABLE:
                    st.write("• **🆕 Volume Analysis** ✅")
                else:
                    st.write("• **Volume Analysis** ⚠️ (Module not available)")
                if VOLATILITY_ANALYSIS_AVAILABLE:
                    st.write("• **🆕 Volatility Analysis** ✅")
                else:
                    st.write("• **Volatility Analysis** ⚠️ (Module not available)")
                st.write("• **Fundamental Analysis** ✅")
                st.write("• **Market Correlation & Breakouts** ✅")
                st.write("• **Options Analysis with Greeks** ✅")
                st.write("• **Statistical Confidence Intervals** ✅")
        
        # Show current market status
        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.write("1. **Click any Quick Link** → Should start analysis immediately")
            st.write("2. **Enter symbol + click Analyze** → Complete analysis")
            st.write("3. **Toggle sections** on/off in Analysis Sections panel")
            st.write("4. **All core functionality** should now work correctly")

    # Footer
    st.markdown("---")
    st.write("### 📊 System Information v4.2.1")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Version:** VWV Professional v4.2.1 Enhanced")
        st.write(f"**Architecture:** Full Modular Implementation")
    with col2:
        st.write(f"**Status:** ✅ Navigation Fixed")
        st.write(f"**Features:** Volume & Volatility Analysis")
    with col3:
        st.write(f"**Components:** config, data, analysis, ui, utils")
        st.write(f"**Interface:** Complete multi-section experience")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.write("Please refresh the page and try again.")
        
        if st.checkbox("Show Error Details"):
            st.exception(e)
