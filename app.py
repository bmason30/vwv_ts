"""
File: app.py v1.0.9
VWV Professional Trading System v4.2.2
Main Streamlit application - Technical analysis display fix (calculations preserved)
Created: 2025-07-15
Updated: 2025-10-08
File Version: v1.0.9 - Fixed display only, preserved working calculation pipeline
System Version: v4.2.2 - Technical Analysis Display Error Resolution
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
    calculate_piotroski_score,
    calculate_altman_z_score,
    calculate_roic,
    calculate_key_value_metrics,
    calculate_composite_fundamental_score
)
from analysis.divergence import calculate_divergence_score
from analysis.master_score import calculate_master_score_with_agreement
from analysis.confluence import calculate_signal_confluence, create_confluence_summary
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

# Baldwin indicator import with safe fallback
try:
    from analysis.baldwin_indicator import calculate_baldwin_indicator_complete, format_baldwin_for_display
    BALDWIN_INDICATOR_AVAILABLE = True
except ImportError:
    BALDWIN_INDICATOR_AVAILABLE = False
    calculate_baldwin_indicator_complete = None
    format_baldwin_for_display = None

from ui.components import create_technical_score_bar, create_fundamental_score_bar, create_header
from utils.helpers import format_large_number, get_market_status, get_etf_description
from utils.decorators import safe_calculation_wrapper

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

# Page configuration
st.set_page_config(
    page_title="VWV Professional Trading System v4.2.2",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_sidebar_controls():
    """Create sidebar controls and return analysis parameters"""
    st.sidebar.title("📊 Trading Analysis v4.2.2")
    
    # Initialize session state for toggles
    if 'recently_viewed' not in st.session_state:
        st.session_state.recently_viewed = []
    if 'last_analyzed_symbol' not in st.session_state:
        st.session_state.last_analyzed_symbol = None
    if 'show_charts' not in st.session_state:
        st.session_state.show_charts = True
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
    if 'show_baldwin_indicator' not in st.session_state:
        st.session_state.show_baldwin_indicator = True
    if 'show_master_score' not in st.session_state:
        st.session_state.show_master_score = True
    if 'show_divergence' not in st.session_state:
        st.session_state.show_divergence = True
    if 'show_confluence' not in st.session_state:
        st.session_state.show_confluence = True
    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = None

    # Symbol input with Enter key support
    # Use selected_symbol if it was set by quicklinks/recent, otherwise use default
    default_symbol = st.session_state.selected_symbol if st.session_state.selected_symbol else "tsla"
    symbol_input = st.sidebar.text_input(
        "Symbol",
        value=default_symbol,
        key="symbol_input",
        help="Enter a stock symbol (e.g., AAPL, TSLA, SPY)"
    ).upper()

    # Clear selected_symbol after it's been used
    if st.session_state.selected_symbol:
        st.session_state.selected_symbol = None
    
    # Data period selection with 3mo as default (optimal for all modules)
    period = st.sidebar.selectbox(
        "Data Period",
        options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
        index=1,  # Default to 3mo (recommended for technical analysis)
        help="Select the historical data period for analysis. 3mo+ recommended for all modules."
    )
    
    # Analysis sections toggle
    with st.sidebar.expander("📊 Analysis Sections", expanded=False):
        st.session_state.show_charts = st.checkbox("Show Charts", value=st.session_state.show_charts)
        st.session_state.show_master_score = st.checkbox("Show Master Score", value=st.session_state.show_master_score)
        st.session_state.show_confluence = st.checkbox("Show Signal Confluence", value=st.session_state.show_confluence)
        st.session_state.show_technical_analysis = st.checkbox("Show Technical Analysis", value=st.session_state.show_technical_analysis)
        st.session_state.show_divergence = st.checkbox("Show Divergence Detection", value=st.session_state.show_divergence)
        if VOLUME_ANALYSIS_AVAILABLE:
            st.session_state.show_volume_analysis = st.checkbox("Show Volume Analysis", value=st.session_state.show_volume_analysis)
        if VOLATILITY_ANALYSIS_AVAILABLE:
            st.session_state.show_volatility_analysis = st.checkbox("Show Volatility Analysis", value=st.session_state.show_volatility_analysis)
        st.session_state.show_fundamental_analysis = st.checkbox("Show Fundamental Analysis", value=st.session_state.show_fundamental_analysis)
        st.session_state.show_market_correlation = st.checkbox("Show Market Correlation", value=st.session_state.show_market_correlation)
        st.session_state.show_options_analysis = st.checkbox("Show Options Analysis", value=st.session_state.show_options_analysis)
        st.session_state.show_confidence_intervals = st.checkbox("Show Confidence Intervals", value=st.session_state.show_confidence_intervals)
        if BALDWIN_INDICATOR_AVAILABLE:
            st.session_state.show_baldwin_indicator = st.checkbox("Show Baldwin Indicator", value=st.session_state.show_baldwin_indicator)
    
    # Analyze button
    analyze_button = st.sidebar.button("🔍 Analyze Now", use_container_width=True, type="primary")
    
    # Recently viewed
    with st.sidebar.expander("🕒 Recently Viewed", expanded=False):
        if st.session_state.recently_viewed:
            for viewed_symbol in st.session_state.recently_viewed[-5:]:
                if st.button(viewed_symbol, key=f"recent_{viewed_symbol}", use_container_width=True):
                    st.session_state.selected_symbol = viewed_symbol
                    st.rerun()
        else:
            st.write("No recent symbols")
    
    # Quick Links
    with st.sidebar.expander("🔗 Quick Links", expanded=False):
        for category, symbols in QUICK_LINK_CATEGORIES.items():
            st.write(f"**{category}**")
            cols = st.columns(2)
            for idx, symbol in enumerate(symbols):
                with cols[idx % 2]:
                    if st.button(symbol, key=f"quick_{symbol}", use_container_width=True):
                        st.session_state.selected_symbol = symbol
                        st.rerun()
    
    # Debug mode
    show_debug = st.sidebar.checkbox("🐛 Debug Mode", value=False)
    
    return {
        'symbol': symbol_input,
        'period': period,
        'analyze_button': analyze_button,
        'show_debug': show_debug
    }

def add_to_recently_viewed(symbol):
    """Add symbol to recently viewed list"""
    if symbol not in st.session_state.recently_viewed:
        st.session_state.recently_viewed.append(symbol)
    elif symbol in st.session_state.recently_viewed:
        st.session_state.recently_viewed.remove(symbol)
        st.session_state.recently_viewed.append(symbol)

def show_interactive_charts(data, analysis_results, show_debug=False):
    """Display interactive charts section - PRIORITY 1 (FIRST)"""
    if not st.session_state.show_charts:
        return
        
    with st.expander("📊 Interactive Trading Charts", expanded=True):
        try:
            from charts.plotting import display_trading_charts
            display_trading_charts(data, analysis_results)
        except ImportError as e:
            st.error("📊 Charts module not available")
            if show_debug:
                st.error(f"Import error: {str(e)}")
            # Fallback simple chart
            st.subheader("Basic Price Chart (Fallback)")
            if data is not None and not data.empty:
                st.line_chart(data['Close'])
        except Exception as e:
            if show_debug:
                st.error(f"Chart display error: {str(e)}")
                st.exception(e)
            else:
                st.warning("⚠️ Charts temporarily unavailable")

def show_individual_technical_analysis(analysis_results, show_debug=False):
    """
    Display individual technical analysis section - PRIORITY 2 (SECOND)
    Version: v1.0.9 - Display only fix, calculations unchanged
    """
    if not st.session_state.show_technical_analysis:
        return
    
    symbol = analysis_results.get('symbol', 'Unknown')
    period = analysis_results.get('period', 'Unknown')
    data_points = analysis_results.get('data_points', 0)

    with st.expander(f"📊 Technical Analysis - {symbol}", expanded=True):

        # Timeframe warning for insufficient data
        if data_points < 50:
            st.warning(
                f"⚠️ **Limited Data Alert:** Only {data_points} data points available. "
                f"Technical analysis requires **50+ data points** for accurate indicators. "
                f"**Use 3mo or longer period** for reliable results. "
                f"Values may show as defaults (50) with current {period} timeframe."
            )

        # Get data - ONLY changed .get() to handle missing keys, NOT calculation logic
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
        current_price = analysis_results.get('current_price', 0)
        daily_vwap = enhanced_indicators.get('daily_vwap', 0)
        point_of_control = enhanced_indicators.get('point_of_control', 0)
        
        # --- 1. COMPOSITE TECHNICAL SCORE BAR ---
        try:
            composite_score, score_details = calculate_composite_technical_score(analysis_results)
            score_bar_html = create_technical_score_bar(composite_score, score_details)
            st.components.v1.html(score_bar_html, height=160)
        except Exception as e:
            if show_debug:
                st.error(f"Score bar error: {str(e)}")
            st.metric("Composite Technical Score", "Calculating...")
        
        # --- 2. KEY MOMENTUM OSCILLATORS ---
        st.subheader("Key Momentum Oscillators")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi = comprehensive_technicals.get('rsi_14', 50)
            if rsi > 70:
                rsi_status = f"Overbought ({rsi:.1f} > 70)"
                rsi_delta = f"+{rsi - 70:.1f}"
            elif rsi < 30:
                rsi_status = f"Oversold ({rsi:.1f} < 30)"
                rsi_delta = f"{rsi - 30:.1f}"
            else:
                rsi_status = "Neutral"
                rsi_delta = None
            st.metric("RSI (14)", f"{rsi:.2f}", rsi_delta, help=rsi_status)

        with col2:
            mfi = comprehensive_technicals.get('mfi_14', 50)
            if mfi > 80:
                mfi_status = f"Overbought ({mfi:.1f} > 80)"
                mfi_delta = f"+{mfi - 80:.1f}"
            elif mfi < 20:
                mfi_status = f"Oversold ({mfi:.1f} < 20)"
                mfi_delta = f"{mfi - 20:.1f}"
            else:
                mfi_status = "Neutral"
                mfi_delta = None
            st.metric("MFI (14)", f"{mfi:.2f}", mfi_delta, help=mfi_status)

        with col3:
            stoch = comprehensive_technicals.get('stochastic', {})
            stoch_k = stoch.get('k', 50) if isinstance(stoch, dict) else 50
            if stoch_k > 80:
                stoch_status = f"Overbought ({stoch_k:.1f} > 80)"
                stoch_delta = f"+{stoch_k - 80:.1f}"
            elif stoch_k < 20:
                stoch_status = f"Oversold ({stoch_k:.1f} < 20)"
                stoch_delta = f"{stoch_k - 20:.1f}"
            else:
                stoch_status = "Neutral"
                stoch_delta = None
            st.metric("Stochastic %K", f"{stoch_k:.2f}", stoch_delta, help=stoch_status)

        with col4:
            williams_r = comprehensive_technicals.get('williams_r', -50)
            if williams_r > -20:
                williams_status = f"Overbought ({williams_r:.1f} > -20)"
                williams_delta = f"+{williams_r + 20:.1f}"
            elif williams_r < -80:
                williams_status = f"Oversold ({williams_r:.1f} < -80)"
                williams_delta = f"{williams_r + 80:.1f}"
            else:
                williams_status = "Neutral"
                williams_delta = None
            st.metric("Williams %R", f"{williams_r:.2f}", williams_delta, help=williams_status)
        
        # --- 3. TREND ANALYSIS ---
        st.subheader("Trend Analysis")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            macd_data = comprehensive_technicals.get('macd', {})
            macd_hist = macd_data.get('histogram', 0) if isinstance(macd_data, dict) else 0
            macd_delta = "Bullish" if macd_hist > 0 else "Bearish"
            st.metric("MACD Histogram", f"{macd_hist:.4f}", macd_delta)

        with col2:
            adx_data = comprehensive_technicals.get('adx', {})
            if isinstance(adx_data, dict):
                adx_value = adx_data.get('adx', 0)
                trend_strength = adx_data.get('trend_strength', 'Unknown')
                st.metric("ADX (14)", f"{adx_value:.2f}",
                         help=f"Trend Strength: {trend_strength}")

        with col3:
            if isinstance(adx_data, dict):
                plus_di = adx_data.get('plus_di', 0)
                st.metric("+DI", f"{plus_di:.2f}",
                         help="Plus Directional Indicator (bullish movement)")

        with col4:
            if isinstance(adx_data, dict):
                minus_di = adx_data.get('minus_di', 0)
                trend_dir = adx_data.get('trend_direction', 'Unknown')
                st.metric("-DI", f"{minus_di:.2f}",
                         delta=trend_dir if trend_dir != 'Unknown' else None,
                         help="Minus Directional Indicator (bearish movement)")
        
        # --- 4. PRICE-BASED INDICATORS & KEY LEVELS TABLE ---
        st.subheader("Price-Based Indicators & Key Levels")
        
        indicators_data = []
        
        # Current Price
        indicators_data.append({
            'Indicator': 'Current Price',
            'Value': f'${current_price:.2f}',
            'Type': '📍 Reference',
            'Distance': '0.0%',
            'Status': 'Current'
        })
        
        # Daily VWAP
        if daily_vwap and daily_vwap > 0:
            vwap_distance = ((current_price - daily_vwap) / daily_vwap * 100)
            vwap_status = "Above" if current_price > daily_vwap else "Below"
            indicators_data.append({
                'Indicator': 'Daily VWAP',
                'Value': f'${daily_vwap:.2f}',
                'Type': '📊 Volume Weighted',
                'Distance': f'{vwap_distance:+.2f}%',
                'Status': vwap_status
            })
        
        # Point of Control
        if point_of_control and point_of_control > 0:
            poc_distance = ((current_price - point_of_control) / point_of_control * 100)
            poc_status = "Above" if current_price > point_of_control else "Below"
            indicators_data.append({
                'Indicator': 'Point of Control',
                'Value': f'${point_of_control:.2f}',
                'Type': '📊 Volume Profile',
                'Distance': f'{poc_distance:+.2f}%',
                'Status': poc_status
            })
        
        # Fibonacci EMAs
        if fibonacci_emas:
            for ema_name, ema_value in fibonacci_emas.items():
                try:
                    period = str(ema_name).split('_')[1]
                    ema_val = float(ema_value) if ema_value else 0
                    
                    if ema_val > 0:
                        ema_distance = ((current_price - ema_val) / ema_val * 100)
                        ema_status = "Above" if current_price > ema_val else "Below"
                        indicators_data.append({
                            'Indicator': f'EMA {period}',
                            'Value': f'${ema_val:.2f}',
                            'Type': '📈 Trend',
                            'Distance': f'{ema_distance:+.2f}%',
                            'Status': ema_status
                        })
                except:
                    continue
        
        # Display table
        if indicators_data:
            df_technical = pd.DataFrame(indicators_data)
            st.dataframe(df_technical, use_container_width=True, hide_index=True)
        
        # Debug info
        if show_debug:
            with st.expander("🐛 Technical Analysis Debug Info", expanded=True):
                st.write("**Comprehensive Technicals:**")
                st.json(comprehensive_technicals)
                st.write("**All Indicators:**")
                st.json(enhanced_indicators)

def show_volume_analysis(analysis_results, show_debug=False):
    """Display volume analysis section - PRIORITY 3 (Optional)"""
    if not st.session_state.show_volume_analysis or not VOLUME_ANALYSIS_AVAILABLE:
        return

    symbol = analysis_results.get('symbol', 'Unknown')
    period = analysis_results.get('period', 'Unknown')
    data_points = analysis_results.get('data_points', 0)

    with st.expander(f"📊 Volume Analysis - {symbol}", expanded=True):

        # Timeframe warning for insufficient data
        if data_points < 30:
            st.warning(
                f"⚠️ **Insufficient Data:** Only {data_points} data points available. "
                f"Volume analysis requires **30+ data points** (about 1.5 months). "
                f"**Use 3mo or longer period** for accurate volume metrics with current {period} timeframe."
            )

        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volume_analysis = enhanced_indicators.get('volume_analysis', {})
        
        if volume_analysis and 'error' not in volume_analysis:
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                current_vol = volume_analysis.get('current_volume', 0)
                st.metric("Current Volume", format_large_number(current_vol))
            with col2:
                vol_5d = volume_analysis.get('volume_5d_avg', 0)
                st.metric("5D Avg Volume", format_large_number(vol_5d))
            with col3:
                vol_30d = volume_analysis.get('volume_30d_avg', 0)
                st.metric("30D Avg Volume", format_large_number(vol_30d))
            with col4:
                vol_ratio = volume_analysis.get('volume_ratio', 1.0)
                st.metric("Volume Ratio", f"{vol_ratio:.2f}x")
            with col5:
                vol_score = volume_analysis.get('volume_score', 50)
                st.metric("Volume Score", f"{vol_score:.1f}/100")

            # Second row with regime and implications
            col1, col2, col3 = st.columns(3)
            with col1:
                vol_regime = volume_analysis.get('volume_regime', 'Normal')
                st.metric("Volume Regime", vol_regime)
            with col2:
                vol_trend = volume_analysis.get('volume_trend_5d', 0)
                st.metric("5D Trend", f"{vol_trend:+.2f}%")
            with col3:
                vol_zscore = volume_analysis.get('volume_zscore', 0)
                st.metric("Z-Score", f"{vol_zscore:.2f}")

            # Trading implications
            implications = volume_analysis.get('trading_implications', 'N/A')
            st.info(f"**Trading Implications:** {implications}")
        else:
            error_msg = volume_analysis.get('error', 'insufficient data') if volume_analysis else 'insufficient data'
            st.warning(f"⚠️ Volume analysis not available - {error_msg}")

def show_volatility_analysis(analysis_results, show_debug=False):
    """Display volatility analysis section - PRIORITY 4 (Optional)"""
    if not st.session_state.show_volatility_analysis or not VOLATILITY_ANALYSIS_AVAILABLE:
        return

    symbol = analysis_results.get('symbol', 'Unknown')
    period = analysis_results.get('period', 'Unknown')
    data_points = analysis_results.get('data_points', 0)

    with st.expander(f"📊 Volatility Analysis - {symbol}", expanded=True):

        # Timeframe warning for insufficient data
        if data_points < 30:
            st.warning(
                f"⚠️ **Insufficient Data:** Only {data_points} data points available. "
                f"Volatility analysis requires **30+ data points** (about 1.5 months). "
                f"**Use 3mo or longer period** for accurate volatility metrics with current {period} timeframe."
            )

        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volatility_analysis = enhanced_indicators.get('volatility_analysis', {})
        
        if volatility_analysis and 'error' not in volatility_analysis:
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                vol_20d = volatility_analysis.get('volatility_20d', 0)
                st.metric("20D Volatility", f"{vol_20d:.2f}%")
            with col2:
                realized_vol = volatility_analysis.get('realized_volatility', 0)
                st.metric("Realized Vol", f"{realized_vol:.2f}%")
            with col3:
                vol_percentile = volatility_analysis.get('volatility_percentile', 50)
                st.metric("Vol Percentile", f"{vol_percentile:.1f}%")
            with col4:
                vol_score = volatility_analysis.get('volatility_score', 50)
                st.metric("Volatility Score", f"{vol_score:.1f}/100")
            with col5:
                vol_regime = volatility_analysis.get('volatility_regime', 'Normal')
                st.metric("Regime", vol_regime)

            # Additional metrics in a second row
            st.subheader("Trading Implications")
            col1, col2 = st.columns(2)
            with col1:
                options_strategy = volatility_analysis.get('options_strategy', 'N/A')
                st.info(f"**Options Strategy:** {options_strategy}")
            with col2:
                implications = volatility_analysis.get('trading_implications', 'N/A')
                st.info(f"**Implications:** {implications}")
        else:
            error_msg = volatility_analysis.get('error', 'insufficient data') if volatility_analysis else 'insufficient data'
            st.warning(f"⚠️ Volatility analysis not available - {error_msg}")

def show_fundamental_analysis(analysis_results, show_debug=False):
    """Display fundamental analysis section - PRIORITY 5"""
    if not st.session_state.show_fundamental_analysis:
        return

    with st.expander("📊 Fundamental Analysis - Value Investment Scores", expanded=True):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        graham_data = enhanced_indicators.get('graham_score', {})
        piotroski_data = enhanced_indicators.get('piotroski_score', {})
        altman_data = enhanced_indicators.get('altman_z_score', {})

        # Check if symbol is ETF
        is_etf_symbol = ('ETF' in str(graham_data.get('error', '')) or
                         'ETF' in str(piotroski_data.get('error', '')))

        if is_etf_symbol:
            st.info("ℹ️ Fundamental analysis not applicable for ETFs")
            return

        # --- COMPOSITE FUNDAMENTAL SCORE BAR ---
        try:
            composite_score, score_details = calculate_composite_fundamental_score(analysis_results)
            score_bar_html = create_fundamental_score_bar(composite_score, score_details)
            st.components.v1.html(score_bar_html, height=160)
        except Exception as e:
            if show_debug:
                st.error(f"Composite score error: {str(e)}")
            st.metric("Composite Fundamental Score", "Calculating...")

        # Display scores with detailed criteria
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Graham Score (Value Investing)")
            if 'error' not in graham_data:
                score = graham_data.get('score', 0)
                total = graham_data.get('total_possible', 10)
                grade = graham_data.get('grade', 'F')
                interpretation = graham_data.get('interpretation', 'Unknown')

                st.metric("Score", f"{score}/{total}", delta=f"Grade: {grade}")
                st.caption(interpretation)

                # Show criteria details
                criteria = graham_data.get('criteria', [])
                if criteria:
                    with st.expander("📋 Detailed Criteria", expanded=False):
                        for criterion in criteria:
                            if '✓' in criterion:
                                st.success(criterion)
                            else:
                                st.error(criterion)
            else:
                st.metric("Score", "0/10")
                st.error(f"Error: {graham_data.get('error', 'Unknown error')}")

        with col2:
            st.subheader("Piotroski F-Score (Financial Health)")
            if 'error' not in piotroski_data:
                score = piotroski_data.get('score', 0)
                total = piotroski_data.get('total_possible', 9)
                grade = piotroski_data.get('grade', 'F')
                interpretation = piotroski_data.get('interpretation', 'Unknown')

                st.metric("Score", f"{score}/{total}", delta=f"Grade: {grade}")
                st.caption(interpretation)

                # Show criteria details
                criteria = piotroski_data.get('criteria', [])
                if criteria:
                    with st.expander("📋 Detailed Criteria", expanded=False):
                        for criterion in criteria:
                            if '✓' in criterion:
                                st.success(criterion)
                            else:
                                st.error(criterion)
            else:
                st.metric("Score", "0/9")
                st.error(f"Error: {piotroski_data.get('error', 'Unknown error')}")

        with col3:
            st.subheader("Altman Z-Score (Bankruptcy Risk)")
            if 'error' not in altman_data:
                z_score = altman_data.get('z_score', 0)
                zone = altman_data.get('zone', 'Unknown')
                interpretation = altman_data.get('interpretation', 'Unknown')

                # Color code based on zone
                if zone == "Safe Zone":
                    zone_color = "🟢"
                elif zone == "Grey Zone":
                    zone_color = "🟡"
                else:  # Distress Zone
                    zone_color = "🔴"

                st.metric("Z-Score", f"{z_score:.2f}", delta=f"{zone_color} {zone}")
                st.caption(interpretation)

                # Show component details
                components = altman_data.get('components', {})
                criteria = altman_data.get('criteria', [])
                if criteria:
                    with st.expander("📋 Component Details", expanded=False):
                        for criterion in criteria:
                            st.text(criterion)
            else:
                st.metric("Z-Score", "0.00")
                st.error(f"Error: {altman_data.get('error', 'Unknown error')}")

        # Add ROIC in a second row
        st.divider()
        roic_data = enhanced_indicators.get('roic', {})

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.subheader("ROIC (Return on Invested Capital)")
            if 'error' not in roic_data:
                roic_percent = roic_data.get('roic_percent', 0)
                grade = roic_data.get('grade', 'F')
                interpretation = roic_data.get('interpretation', 'Unknown')

                # Color code based on grade
                if grade in ['A+', 'A']:
                    grade_color = "🟢"
                elif grade == 'B':
                    grade_color = "🟡"
                elif grade == 'C':
                    grade_color = "🟠"
                else:
                    grade_color = "🔴"

                st.metric("ROIC", f"{roic_percent:.2f}%", delta=f"{grade_color} Grade: {grade}")
                st.caption(interpretation)

                # Show component details
                criteria = roic_data.get('criteria', [])
                if criteria:
                    with st.expander("📋 Calculation Details", expanded=False):
                        for criterion in criteria:
                            st.text(criterion)
            else:
                st.metric("ROIC", "0.00%")
                st.error(f"Error: {roic_data.get('error', 'Unknown error')}")

        # --- KEY VALUE METRICS ---
        st.divider()
        st.subheader("Key Value Metrics")

        metrics_data = enhanced_indicators.get('key_value_metrics', {})
        metrics = metrics_data.get('metrics', {})
        company_name = metrics_data.get('company_name', '')

        if company_name:
            st.caption(f"**{company_name}**")

        if not metrics:
            st.warning("⚠️ No metrics data available")
        else:
            # Display metrics in two rows
            # Row 1: P/E, P/B, D/E
            col1, col2, col3 = st.columns(3)

            with col1:
                pe_data = metrics.get('pe_ratio', {})
                st.metric(
                    "P/E Ratio",
                    pe_data.get('display', 'N/A'),
                    help="Price-to-Earnings Ratio: Compares stock price to earnings per share"
                )
                st.caption(pe_data.get('interpretation', ''))

            with col2:
                pb_data = metrics.get('pb_ratio', {})
                st.metric(
                    "P/B Ratio",
                    pb_data.get('display', 'N/A'),
                    help="Price-to-Book Ratio: Compares market price to book value"
                )
                st.caption(pb_data.get('interpretation', ''))

            with col3:
                de_data = metrics.get('de_ratio', {})
                st.metric(
                    "Debt-to-Equity",
                    de_data.get('display', 'N/A'),
                    help="Debt-to-Equity Ratio: Measures financial leverage"
                )
                st.caption(de_data.get('interpretation', ''))

            st.divider()

            # Row 2: Dividend Yield, FCF, ROE
            col1, col2, col3 = st.columns(3)

            with col1:
                div_data = metrics.get('dividend_yield', {})
                st.metric(
                    "Dividend Yield",
                    div_data.get('display', 'N/A'),
                    help="Annual dividend payments relative to stock price"
                )
                st.caption(div_data.get('interpretation', ''))

            with col2:
                fcf_data = metrics.get('free_cash_flow', {})
                st.metric(
                    "Free Cash Flow",
                    fcf_data.get('display', 'N/A'),
                    help="Cash remaining after operating expenses and capital expenditures"
                )
                st.caption(fcf_data.get('interpretation', ''))

            with col3:
                roe_data = metrics.get('roe', {})
                st.metric(
                    "Return on Equity",
                    roe_data.get('display', 'N/A'),
                    help="Profit generated from shareholders' equity"
                )
                st.caption(roe_data.get('interpretation', ''))

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis section - PRIORITY 7 (After Baldwin)"""
    if not st.session_state.show_market_correlation:
        return
        
    with st.expander("🌐 Market Correlation & Comparison Analysis", expanded=True):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        market_correlations = enhanced_indicators.get('market_correlations', {})
        
        if market_correlations:
            st.subheader("📊 ETF Correlation Analysis")
            
            correlation_data = []
            for etf, data in market_correlations.items():
                correlation_data.append({
                    'ETF': etf,
                    'Correlation': f"{data.get('correlation', 0):.3f}",
                    'Beta': f"{data.get('beta', 0):.3f}",
                    'Relationship': data.get('relationship', 'Unknown')
                })
            
            if correlation_data:
                df_corr = pd.DataFrame(correlation_data)
                st.dataframe(df_corr, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Market correlation data not available")

def show_baldwin_indicator(show_debug=False):
    """Display Baldwin Market Regime Indicator - PRIORITY 6"""
    if not st.session_state.show_baldwin_indicator or not BALDWIN_INDICATOR_AVAILABLE:
        return

    with st.expander("🚦 Baldwin Market Regime Indicator", expanded=True):
        try:
            baldwin_results = calculate_baldwin_indicator_complete(show_debug=show_debug)

            if 'error' in baldwin_results:
                st.error(f"❌ Baldwin Indicator Error: {baldwin_results.get('error', 'Unknown error')}")
                return

            # Display overall score and regime
            col1, col2, col3 = st.columns(3)

            with col1:
                score = baldwin_results.get('baldwin_score', 0)
                st.metric("Baldwin Score", f"{score:.1f}/100")

            with col2:
                regime = baldwin_results.get('market_regime', 'UNKNOWN')
                regime_color = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(regime, "⚪")
                st.metric("Market Regime", f"{regime_color} {regime}")

            with col3:
                timestamp = baldwin_results.get('timestamp', 'N/A')
                st.metric("Updated", timestamp.split(' ')[1] if ' ' in timestamp else timestamp)

            # Display strategy
            st.info(f"**Strategy:** {baldwin_results.get('strategy', 'N/A')}")

            # Display component breakdown
            components = baldwin_results.get('components', {})
            if components:
                st.subheader("Component Breakdown")

                component_data = []
                for name, data in components.items():
                    component_data.append({
                        'Component': name.replace('_', ' & '),
                        'Score': f"{data.get('component_score', 0):.1f}/100"
                    })

                if component_data:
                    df_components = pd.DataFrame(component_data)
                    st.dataframe(df_components, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"❌ Baldwin Indicator Error: {str(e)}")
            if show_debug:
                import traceback
                st.code(traceback.format_exc())

def show_options_analysis(analysis_results, show_debug=False):
    """Display options analysis section - PRIORITY 8"""
    if not st.session_state.show_options_analysis:
        return
        
    with st.expander("🎯 Options Analysis - Strike Levels with Greeks", expanded=True):
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        options_levels = enhanced_indicators.get('options_levels', [])
        
        if options_levels:
            df_options = pd.DataFrame(options_levels)
            st.dataframe(df_options, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Options analysis not available - insufficient data")

def show_confidence_intervals(analysis_results, show_debug=False):
    """Display confidence intervals section - PRIORITY 9"""
    if not st.session_state.show_confidence_intervals:
        return

    period = analysis_results.get('period', 'Unknown')
    data_points = analysis_results.get('data_points', 0)
    confidence_analysis = analysis_results.get('confidence_analysis')

    if confidence_analysis:
        with st.expander("📊 Statistical Confidence Intervals", expanded=True):

            # Timeframe warning for insufficient data
            if data_points < 100:
                st.warning(
                    f"⚠️ **Limited Statistical Confidence:** Only {data_points} data points available. "
                    f"Confidence intervals are most reliable with **100+ data points** (about 5 months). "
                    f"**Use 6mo or 1y period** for statistically significant results. "
                    f"Current {period} timeframe may produce wider intervals."
                )
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Weekly Return", f"{confidence_analysis['mean_weekly_return']:.3f}%")
            with col2:
                st.metric("Weekly Volatility", f"{confidence_analysis['weekly_volatility']:.2f}%")
            with col3:
                st.metric("Sample Size", f"{confidence_analysis['sample_size']} weeks")
            
            intervals_data = []
            for level, data in confidence_analysis['confidence_intervals'].items():
                intervals_data.append({
                    'Confidence Level': level,
                    'Upper Bound': f"${data['upper_bound']}",
                    'Lower Bound': f"${data['lower_bound']}",
                    'Expected Move': f"±{data['expected_move_pct']:.2f}%"
                })
            
            df_intervals = pd.DataFrame(intervals_data)
            st.dataframe(df_intervals, use_container_width=True, hide_index=True)

def show_master_score(analysis_results, show_debug=False):
    """
    Display Master Score - unified scoring across all analysis modules.
    Phase 1a implementation.
    """
    if not st.session_state.get('show_master_score', True):
        return

    symbol = analysis_results.get('symbol', 'Unknown')
    master_score_data = analysis_results.get('enhanced_indicators', {}).get('master_score', {})

    if not master_score_data or 'error' in master_score_data:
        return

    with st.expander(f"🎯 Master Score - {symbol}", expanded=True):
        st.subheader("Unified Analysis Score (0-100)")

        # Main score display
        master_score = master_score_data.get('master_score', 0)
        interpretation = master_score_data.get('interpretation', 'Unknown')
        signal_strength = master_score_data.get('signal_strength', 'Unknown')

        # Color-coded score bar similar to technical and fundamental scores
        from ui.components import create_technical_score_bar

        # Create score bar HTML (reuse technical score bar with custom text)
        score_html = f"""
        <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 10px; margin: 10px 0;">
            <div style="text-align: center; color: white;">
                <h2 style="margin: 0; font-size: 3em; font-weight: bold;">{master_score:.1f}/100</h2>
                <p style="margin: 5px 0; font-size: 1.2em;">{interpretation}</p>
                <p style="margin: 5px 0; font-size: 1em;">Signal Strength: {signal_strength}</p>
            </div>
            <div style="background: rgba(255,255,255,0.2); height: 20px; border-radius: 10px; margin-top: 15px;">
                <div style="background: white; height: 100%; width: {master_score}%; border-radius: 10px;"></div>
            </div>
        </div>
        """
        st.components.v1.html(score_html, height=180)

        # Component breakdown
        st.subheader("Component Scores")
        components = master_score_data.get('components', {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            tech_score = components.get('technical', {}).get('raw', 0)
            st.metric("Technical", f"{tech_score:.1f}/100",
                     help=f"Weight: {components.get('technical', {}).get('weight', 0):.0%}")

        with col2:
            fund_score = components.get('fundamental', {}).get('raw', 0)
            st.metric("Fundamental", f"{fund_score:.1f}/100",
                     help=f"Weight: {components.get('fundamental', {}).get('weight', 0):.0%}")

        with col3:
            momentum_score = components.get('momentum', {}).get('raw', 0)
            st.metric("Momentum", f"{momentum_score:.1f}/100",
                     help=f"Weight: {components.get('momentum', {}).get('weight', 0):.0%}")

        with col4:
            div_score = components.get('divergence', {}).get('raw', 0)
            st.metric("Divergence", f"{div_score:+.1f}",
                     help=f"Weight: {components.get('divergence', {}).get('weight', 0):.0%}")

        # Agreement analysis
        st.subheader("Component Agreement")
        agreement = master_score_data.get('agreement', {})

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Agreement Level", agreement.get('agreement_level', 'N/A'))

        with col2:
            st.metric("Consensus", agreement.get('consensus', 'N/A'))

        with col3:
            std_dev = agreement.get('std_dev', 0)
            st.metric("Std Deviation", f"{std_dev:.1f}",
                     help="Lower = More agreement between components")

def show_divergence_analysis(analysis_results, show_debug=False):
    """
    Display Divergence Detection analysis.
    Phase 1a implementation - simplified divergence.
    """
    if not st.session_state.get('show_divergence', True):
        return

    symbol = analysis_results.get('symbol', 'Unknown')
    divergence_data = analysis_results.get('enhanced_indicators', {}).get('divergence', {})

    if not divergence_data or 'error' in divergence_data:
        return

    with st.expander(f"🔄 Divergence Detection - {symbol}", expanded=False):
        st.subheader("Price/Oscillator Divergence Analysis")

        # Main metrics
        score = divergence_data.get('score', 0)
        status = divergence_data.get('status', 'Unknown')
        total_div = divergence_data.get('total_divergences', 0)
        bullish_count = divergence_data.get('bullish_count', 0)
        bearish_count = divergence_data.get('bearish_count', 0)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Divergence Score", f"{score:+.1f}",
                     help="Positive = Bullish, Negative = Bearish")

        with col2:
            st.metric("Status", status)

        with col3:
            st.metric("Bullish Signals", bullish_count)

        with col4:
            st.metric("Bearish Signals", bearish_count)

        # Detected divergences
        divergences = divergence_data.get('divergences', [])

        if divergences:
            st.subheader("Detected Divergences")

            div_data = []
            for div in divergences:
                div_data.append({
                    'Type': div.get('type', 'Unknown').replace('_', ' ').title(),
                    'Oscillator': div.get('oscillator', 'Unknown').upper(),
                    'Strength': div.get('strength', 'Unknown').title(),
                    'Score': f"{div.get('score', 0):+.1f}",
                    'Description': div.get('description', 'N/A')
                })

            df_div = pd.DataFrame(div_data)
            st.dataframe(df_div, use_container_width=True, hide_index=True)
        else:
            st.info("No divergences detected in current analysis period.")

        # Debug diagnostics
        if show_debug:
            st.subheader("🔍 Divergence Debug Info")

            # Get data for diagnostics
            data_points = analysis_results.get('data_points', 0)
            from config.settings import get_momentum_divergence_config
            config = get_momentum_divergence_config()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Data Points", data_points)
                st.caption(f"Lookback: {config['lookback_period']} bars")

            with col2:
                st.metric("Peak Prominence", f"{config['peak_prominence']*100}%")
                st.caption(f"Min distance: {config['min_swing_distance']} bars")

            with col3:
                regular = divergence_data.get('regular_count', 0)
                hidden = divergence_data.get('hidden_count', 0)
                st.metric("Types Found", f"{regular}R + {hidden}H")
                st.caption("Regular + Hidden")

            # Peak detection info
            try:
                from scipy.signal import find_peaks
                import numpy as np

                # Get recent data (would need to pass this through, but for now show general info)
                st.info(f"""
                **Peak Detection Status:**
                - Using scipy.signal.find_peaks
                - Requires 2+ peaks/troughs for divergence
                - Current settings are conservative (low false positives)

                **No divergences is NORMAL when:**
                - Market trending steadily
                - No recent reversals
                - Price and oscillators in sync

                **To increase sensitivity:** Edit config/settings.py
                - Decrease `peak_prominence` (currently {config['peak_prominence']})
                - Increase `lookback_period` (currently {config['lookback_period']})
                - Decrease `min_swing_distance` (currently {config['min_swing_distance']})
                """)

            except Exception as e:
                st.warning(f"Debug info error: {e}")

        # Info box
        with st.container():
            st.markdown("""
            **About Divergence Detection:**
            - **Bullish Divergence**: Price declining but oscillators rising → Potential reversal up
            - **Bearish Divergence**: Price rising but oscillators declining → Potential reversal down
            - **Hidden Divergence**: Indicates trend continuation rather than reversal

            *Phase 1b now uses advanced peak matching with scipy for improved accuracy.*

            **💡 Tip:** Divergences are rare signals (2-4x/year on stable stocks). Zero divergences is normal!
            Try volatile symbols (COIN, MARA, SOXL) during market reversals for higher detection rates.
            """)


def show_signal_confluence(analysis_results, show_debug=False):
    """
    Display Signal Confluence Dashboard - agreement across all modules.
    Phase 1b implementation.
    """
    if not st.session_state.get('show_confluence', True):
        return

    symbol = analysis_results.get('symbol', 'Unknown')
    confluence_data = analysis_results.get('enhanced_indicators', {}).get('confluence', {})

    if not confluence_data or 'error' in confluence_data:
        return

    with st.expander(f"📊 Signal Confluence Dashboard - {symbol}", expanded=True):
        st.subheader("Cross-Module Signal Agreement")

        # Main metrics
        confluence_score = confluence_data.get('confluence_score', 50)
        confidence = confluence_data.get('confidence', {})
        bullish_count = confluence_data.get('bullish_modules', 0)
        bearish_count = confluence_data.get('bearish_modules', 0)
        neutral_count = confluence_data.get('neutral_modules', 0)
        total_modules = confluence_data.get('total_modules', 0)

        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Confluence Score", f"{confluence_score:.1f}/100",
                     help="Agreement level across all analysis modules")

        with col2:
            st.metric("Confidence Level", confidence.get('level', 'Unknown'),
                     help=f"Score: {confidence.get('score', 0):.1f}")

        with col3:
            st.metric("Bullish Modules", f"{bullish_count}/{total_modules}",
                     delta=f"+{bullish_count - bearish_count}" if bullish_count > bearish_count else None)

        with col4:
            st.metric("Bearish Modules", f"{bearish_count}/{total_modules}",
                     delta=f"+{bearish_count - bullish_count}" if bearish_count > bullish_count else None)

        # Visual confluence bar
        st.subheader("Signal Direction Breakdown")

        # Create a visual representation
        if total_modules > 0:
            bullish_pct = (bullish_count / total_modules) * 100
            bearish_pct = (bearish_count / total_modules) * 100
            neutral_pct = (neutral_count / total_modules) * 100

            col1, col2, col3 = st.columns([bullish_count or 1, neutral_count or 1, bearish_count or 1])

            with col1:
                st.success(f"🟢 Bullish: {bullish_pct:.0f}%")

            with col2:
                st.info(f"⚪ Neutral: {neutral_pct:.0f}%")

            with col3:
                st.error(f"🔴 Bearish: {bearish_pct:.0f}%")

        # Module signals table
        st.subheader("Module Signal Breakdown")

        signals = confluence_data.get('signals', {})
        if signals:
            signal_data = []
            for module_name, signal in signals.items():
                # Format direction with emoji
                direction = signal.get('direction', 'neutral')
                if direction == 'bullish':
                    direction_display = "🟢 Bullish"
                elif direction == 'bearish':
                    direction_display = "🔴 Bearish"
                else:
                    direction_display = "⚪ Neutral"

                signal_data.append({
                    'Module': module_name.replace('_', ' ').title(),
                    'Direction': direction_display,
                    'Strength': f"{signal.get('strength', 0):.1f}",
                    'Score': f"{signal.get('score', 0):.2f}",
                    'Description': signal.get('description', 'N/A')
                })

            df_signals = pd.DataFrame(signal_data)
            st.dataframe(df_signals, use_container_width=True, hide_index=True)

        # Conflicts section
        conflicts = confluence_data.get('conflicts', [])
        if conflicts:
            st.subheader("⚠️ Signal Conflicts Detected")

            for conflict in conflicts:
                conflict_type = conflict.get('type', 'Unknown')
                description = conflict.get('description', 'N/A')

                if conflict_type == 'directional_conflict':
                    st.warning(f"**Directional Conflict**: {description}")
                    col1, col2 = st.columns(2)

                    with col1:
                        bullish_mods = conflict.get('bullish', [])
                        st.write("**Bullish Modules:**")
                        for mod in bullish_mods:
                            st.write(f"  - {mod.replace('_', ' ').title()}")

                    with col2:
                        bearish_mods = conflict.get('bearish', [])
                        st.write("**Bearish Modules:**")
                        for mod in bearish_mods:
                            st.write(f"  - {mod.replace('_', ' ').title()}")

                elif conflict_type == 'strength_mismatch':
                    st.info(f"**Strength Mismatch**: {description}")

        # Summary text
        with st.container():
            st.markdown(f"""
            **Interpretation:**

            {create_confluence_summary(confluence_data)}

            **How to Use:**
            - **High Confluence (70+)**: Strong agreement suggests reliable signals
            - **Medium Confluence (40-70)**: Mixed signals, use additional confirmation
            - **Low Confluence (<40)**: Conflicting signals, wait for clarity
            """)

def perform_enhanced_analysis(symbol, period, show_debug=False):
    """
    Perform enhanced analysis - CALCULATION LOGIC UNCHANGED FROM WORKING VERSION
    Version: v1.0.9 - Only display changes, calculations preserved
    """
    try:
        # Step 1: Fetch data (UNCHANGED)
        market_data = get_market_data_enhanced(symbol, period, show_debug)
        
        if market_data is None:
            st.error(f"❌ Could not fetch data for {symbol}")
            return None, None
        
        # Step 2: Store and prepare data (UNCHANGED)
        data_manager = get_data_manager()
        data_manager.store_market_data(symbol, market_data, show_debug)
        analysis_input = data_manager.get_market_data_for_analysis(symbol)
        
        if analysis_input is None:
            st.error("❌ Could not prepare analysis data")
            return None, None
        
        # Step 3: Calculate all indicators (FIXED: Added debug output)
        if show_debug:
            st.write(f"📊 Calculating technical indicators for {len(analysis_input)} data points...")

        daily_vwap = calculate_daily_vwap(analysis_input)
        fibonacci_emas = calculate_fibonacci_emas(analysis_input)
        point_of_control = calculate_point_of_control_enhanced(analysis_input)
        weekly_deviations = calculate_weekly_deviations(analysis_input)
        comprehensive_technicals = calculate_comprehensive_technicals(analysis_input)

        # Step 3.1: Calculate divergence detection
        divergence_result = calculate_divergence_score(analysis_input, comprehensive_technicals)

        if show_debug:
            st.write(f"✓ Daily VWAP: ${daily_vwap:.2f}" if daily_vwap else "✗ Daily VWAP failed")
            st.write(f"✓ Fibonacci EMAs: {len(fibonacci_emas)} calculated")
            st.write(f"✓ Point of Control: ${point_of_control:.2f}" if point_of_control else "✗ POC failed")
            st.write(f"✓ Comprehensive technicals: {list(comprehensive_technicals.keys()) if comprehensive_technicals else 'EMPTY!'}")
            if comprehensive_technicals:
                rsi = comprehensive_technicals.get('rsi_14', 'N/A')
                st.write(f"  - RSI: {rsi}")
                macd = comprehensive_technicals.get('macd', {})
                st.write(f"  - MACD: {macd}")
            else:
                st.error("⚠️ CRITICAL: comprehensive_technicals is EMPTY - data length may be < 50")
            st.write(f"✓ Divergence Score: {divergence_result.get('score', 0)} ({divergence_result.get('status', 'Unknown')})")
        
        # Step 4: Market correlations (FIXED: pass period parameter + debug)
        market_correlations = calculate_market_correlations_enhanced(
            analysis_input, symbol, period=period, show_debug=show_debug
        )
        if show_debug:
            if market_correlations:
                st.write(f"✓ Market correlations calculated: {list(market_correlations.keys())}")
            else:
                st.warning("⚠️ Market correlations returned empty")
        
        # Step 5: Volume analysis (FIXED: Better error handling)
        volume_analysis = None
        if VOLUME_ANALYSIS_AVAILABLE:
            try:
                volume_analysis = calculate_complete_volume_analysis(analysis_input)
                if show_debug and volume_analysis:
                    st.write(f"✓ Volume analysis calculated: {list(volume_analysis.keys())}")
            except Exception as e:
                volume_analysis = {'error': f'Volume calculation failed: {str(e)}'}
                if show_debug:
                    st.warning(f"Volume analysis error: {str(e)}")
                    st.code(traceback.format_exc())
        
        # Step 6: Volatility analysis (FIXED: Better error handling)
        volatility_analysis = None
        if VOLATILITY_ANALYSIS_AVAILABLE:
            try:
                volatility_analysis = calculate_complete_volatility_analysis(analysis_input)
                if show_debug and volatility_analysis:
                    st.write(f"✓ Volatility analysis calculated: {list(volatility_analysis.keys())}")
            except Exception as e:
                volatility_analysis = {'error': f'Volatility calculation failed: {str(e)}'}
                if show_debug:
                    st.warning(f"Volatility analysis error: {str(e)}")
                    st.code(traceback.format_exc())
        
        # Step 7: Fundamental analysis (FIXED: Better error handling and debug)
        is_etf_symbol = is_etf(symbol)

        if is_etf_symbol:
            graham_score = {
                'score': 0,
                'total_possible': 10,
                'error': 'ETF - Fundamental analysis not applicable'
            }
            piotroski_score = {
                'score': 0,
                'total_possible': 9,
                'error': 'ETF - Fundamental analysis not applicable'
            }
            altman_z_score = {
                'z_score': 0,
                'zone': 'Unknown',
                'error': 'ETF - Fundamental analysis not applicable'
            }
            roic_data = {
                'roic': 0,
                'roic_percent': 0,
                'grade': 'F',
                'error': 'ETF - Fundamental analysis not applicable'
            }
            key_value_metrics = {
                'metrics': {},
                'error': 'ETF - Fundamental analysis not applicable'
            }
            if show_debug:
                st.info(f"Symbol {symbol} detected as ETF - skipping fundamental analysis")
        else:
            if show_debug:
                st.write(f"Calculating fundamental scores for {symbol}...")
            graham_score = calculate_graham_score(symbol, show_debug)
            piotroski_score = calculate_piotroski_score(symbol, show_debug)
            altman_z_score = calculate_altman_z_score(symbol, show_debug)
            roic_data = calculate_roic(symbol, show_debug)
            key_value_metrics = calculate_key_value_metrics(symbol, show_debug)
            if show_debug:
                st.write(f"✓ Graham score: {graham_score.get('score', 0)}/10")
                st.write(f"✓ Piotroski score: {piotroski_score.get('score', 0)}/9")
                st.write(f"✓ Altman Z-Score: {altman_z_score.get('z_score', 0):.2f} ({altman_z_score.get('zone', 'Unknown')})")
                st.write(f"✓ ROIC: {roic_data.get('roic_percent', 0):.2f}% (Grade: {roic_data.get('grade', 'F')})")
                st.write(f"✓ Key Value Metrics calculated: {len(key_value_metrics.get('metrics', {}))  } metrics")
                if 'error' in graham_score:
                    st.warning(f"Graham error: {graham_score['error']}")
                if 'error' in piotroski_score:
                    st.warning(f"Piotroski error: {piotroski_score['error']}")
                if 'error' in altman_z_score:
                    st.warning(f"Altman error: {altman_z_score['error']}")
                if 'error' in roic_data:
                    st.warning(f"ROIC error: {roic_data['error']}")
                if 'error' in key_value_metrics:
                    st.warning(f"Key Value Metrics error: {key_value_metrics['error']}")

        # Step 7.1: Calculate Master Score
        # Prepare component scores for master score calculation
        technical_composite_score, _ = calculate_composite_technical_score({
            'enhanced_indicators': {
                'comprehensive_technicals': comprehensive_technicals,
                'fibonacci_emas': fibonacci_emas,
                'daily_vwap': daily_vwap,
                'point_of_control': point_of_control
            },
            'current_price': float(analysis_input['Close'].iloc[-1])
        })

        fundamental_composite_score, _ = calculate_composite_fundamental_score({
            'enhanced_indicators': {
                'graham_score': graham_score,
                'piotroski_score': piotroski_score,
                'altman_z_score': altman_z_score,
                'roic': roic_data,
                'key_value_metrics': key_value_metrics
            }
        })

        # Calculate momentum score from oscillators
        rsi = comprehensive_technicals.get('rsi_14', 50)
        mfi = comprehensive_technicals.get('mfi_14', 50)
        stoch_k = comprehensive_technicals.get('stochastic', {}).get('k', 50) if isinstance(comprehensive_technicals.get('stochastic'), dict) else 50
        williams = comprehensive_technicals.get('williams_r', -50)
        momentum_score = (rsi + mfi + stoch_k + (williams + 100)) / 4  # Normalize to 0-100

        # Prepare master score inputs
        master_score_inputs = {
            'technical_score': technical_composite_score,
            'fundamental_score': fundamental_composite_score,
            'vwv_signal': 0,  # Will be integrated in future phase
            'momentum_score': momentum_score,
            'divergence_score': divergence_result.get('score', 0),
            'volume_score': 0,  # Will be integrated from volume analysis
            'volatility_score': 0  # Will be integrated from volatility analysis
        }

        # Calculate master score with agreement analysis
        master_score_result = calculate_master_score_with_agreement(master_score_inputs)

        if show_debug:
            st.write(f"✓ Master Score: {master_score_result.get('master_score', 0):.1f}/100")
            st.write(f"  - Interpretation: {master_score_result.get('interpretation', 'Unknown')}")
            st.write(f"  - Signal Strength: {master_score_result.get('signal_strength', 'Unknown')}")
            st.write(f"  - Agreement: {master_score_result.get('agreement', {}).get('agreement_level', 'Unknown')}")

        # Step 8: Options levels (UNCHANGED)
        current_price = round(float(analysis_input['Close'].iloc[-1]), 2)
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
        
        options_levels = calculate_options_levels_enhanced(
            current_price, volatility, underlying_beta=underlying_beta
        )
        
        # Step 9: Confidence intervals (UNCHANGED)
        confidence_analysis = calculate_confidence_intervals(analysis_input)
        
        # Step 10: Build results (UNCHANGED)
        current_date = analysis_input.index[-1].strftime('%Y-%m-%d')
        
        analysis_results = {
            'symbol': symbol,
            'timestamp': current_date,
            'current_price': current_price,
            'period': period,  # Store period for timeframe validation in displays
            'data_points': len(analysis_input),  # Store actual data points count
            'enhanced_indicators': {
                'daily_vwap': daily_vwap,
                'fibonacci_emas': fibonacci_emas,
                'point_of_control': point_of_control,
                'weekly_deviations': weekly_deviations,
                'comprehensive_technicals': comprehensive_technicals,
                'market_correlations': market_correlations,
                'options_levels': options_levels,
                'graham_score': graham_score,
                'piotroski_score': piotroski_score,
                'altman_z_score': altman_z_score,
                'roic': roic_data,
                'key_value_metrics': key_value_metrics,
                'divergence': divergence_result,
                'master_score': master_score_result
            },
            'confidence_analysis': confidence_analysis,
            'system_status': 'OPERATIONAL v4.2.2'
        }
        
        # Add optional analyses if available (UNCHANGED)
        if volume_analysis:
            analysis_results['enhanced_indicators']['volume_analysis'] = volume_analysis

        if volatility_analysis:
            analysis_results['enhanced_indicators']['volatility_analysis'] = volatility_analysis

        # Calculate signal confluence across all modules
        confluence_result = calculate_signal_confluence(analysis_results)
        analysis_results['enhanced_indicators']['confluence'] = confluence_result

        if show_debug:
            st.write(f"✓ Signal Confluence: {confluence_result.get('confluence_score', 0):.1f}/100")
            st.write(f"  - Confidence: {confluence_result.get('confidence', {}).get('level', 'Unknown')}")
            st.write(f"  - Modules: {confluence_result.get('bullish_modules', 0)} Bullish, {confluence_result.get('bearish_modules', 0)} Bearish")

        # Store results (UNCHANGED)
        data_manager.store_analysis_results(symbol, analysis_results)
        
        # Get chart data (UNCHANGED)
        chart_data = data_manager.get_market_data_for_chart(symbol)
        
        return analysis_results, chart_data
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        if show_debug:
            st.code(traceback.format_exc())
        return None, None

def main():
    """Main application function"""
    create_header()
    controls = create_sidebar_controls()
    
    # Trigger analysis on button click OR when Enter is pressed (symbol changes)
    symbol_changed = (controls['symbol'] and
                     controls['symbol'] != st.session_state.last_analyzed_symbol)
    should_analyze = (controls['analyze_button'] or symbol_changed) and controls['symbol']

    if should_analyze:
        st.session_state.last_analyzed_symbol = controls['symbol']
        add_to_recently_viewed(controls['symbol'])
        
        st.write(f"## 📊 VWV Trading Analysis v4.2.2 - {controls['symbol']}")
        
        with st.spinner(f"Analyzing {controls['symbol']}..."):
            analysis_results, chart_data = perform_enhanced_analysis(
                controls['symbol'],
                controls['period'],
                controls['show_debug']
            )
            
            if analysis_results and chart_data is not None:
                # MANDATORY DISPLAY ORDER
                show_interactive_charts(chart_data, analysis_results, controls['show_debug'])
                show_master_score(analysis_results, controls['show_debug'])
                show_signal_confluence(analysis_results, controls['show_debug'])
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                
                if VOLUME_ANALYSIS_AVAILABLE:
                    show_volume_analysis(analysis_results, controls['show_debug'])
                
                if VOLATILITY_ANALYSIS_AVAILABLE:
                    show_volatility_analysis(analysis_results, controls['show_debug'])

                show_divergence_analysis(analysis_results, controls['show_debug'])
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                show_market_correlation_analysis(analysis_results, controls['show_debug'])

                if BALDWIN_INDICATOR_AVAILABLE:
                    show_baldwin_indicator(controls['show_debug'])

                show_options_analysis(analysis_results, controls['show_debug'])
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
                if controls['show_debug']:
                    with st.expander("🐛 Debug Information", expanded=False):
                        st.write("### Analysis Results Structure")
                        st.json(analysis_results)
            else:
                st.error("❌ No results to display")
    else:
        st.write("## VWV Professional Trading System v4.2.2")
        st.write("**Advanced Technical Analysis • Volatility Analysis • Professional Trading Signals**")
        
        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")
        
        with st.expander("📖 Quick Start Guide", expanded=True):
            st.write("1. **Enter a symbol** in the sidebar (e.g., AAPL, TSLA, SPY)")
            st.write("2. **Default period is 3 months** - optimal for all analysis modules")
            st.write("3. **⚠️ Use 3mo+ for best results** - 1mo period has limited data")
            st.write("4. **Charts display FIRST** - immediate visual analysis")
            st.write("5. **Technical analysis SECOND** - professional scoring")
            st.write("6. **Use Quick Links** for instant analysis")
    
    # Footer
    st.markdown("---")
    st.write("### 📊 System Information v4.2.2")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Version:** VWV Professional v4.2.2")
        st.write(f"**Status:** ✅ Calculations Preserved")
    with col2:
        st.write(f"**Display Order:** Charts → Technical → Analysis")
        st.write(f"**Default Period:** 1 month (1mo)")
    with col3:
        st.write(f"**File Version:** app.py v1.0.9")
        st.write(f"**Fix:** Display only, calculations intact")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.code(traceback.format_exc())
