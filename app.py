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
    calculate_altman_z_score
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

# Baldwin indicator import with safe fallback
try:
    from analysis.baldwin_indicator import calculate_baldwin_indicator_complete, format_baldwin_for_display
    BALDWIN_INDICATOR_AVAILABLE = True
except ImportError:
    BALDWIN_INDICATOR_AVAILABLE = False
    calculate_baldwin_indicator_complete = None
    format_baldwin_for_display = None

from ui.components import create_technical_score_bar, create_header
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

    # Symbol input with Enter key support
    symbol_input = st.sidebar.text_input(
        "Symbol",
        value="tsla",
        key="symbol_input",
        help="Enter a stock symbol (e.g., AAPL, TSLA, SPY)"
    ).upper()
    
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
        st.session_state.show_technical_analysis = st.checkbox("Show Technical Analysis", value=st.session_state.show_technical_analysis)
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
                    st.session_state.symbol_input = viewed_symbol
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
                        st.session_state.symbol_input = symbol
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
        col1, col2 = st.columns(2)
        
        with col1:
            macd_data = comprehensive_technicals.get('macd', {})
            macd_hist = macd_data.get('histogram', 0) if isinstance(macd_data, dict) else 0
            macd_delta = "Bullish" if macd_hist > 0 else "Bearish"
            st.metric("MACD Histogram", f"{macd_hist:.4f}", macd_delta)
        
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
            if show_debug:
                st.info(f"Symbol {symbol} detected as ETF - skipping fundamental analysis")
        else:
            if show_debug:
                st.write(f"Calculating fundamental scores for {symbol}...")
            graham_score = calculate_graham_score(symbol, show_debug)
            piotroski_score = calculate_piotroski_score(symbol, show_debug)
            altman_z_score = calculate_altman_z_score(symbol, show_debug)
            if show_debug:
                st.write(f"✓ Graham score: {graham_score.get('score', 0)}/10")
                st.write(f"✓ Piotroski score: {piotroski_score.get('score', 0)}/9")
                st.write(f"✓ Altman Z-Score: {altman_z_score.get('z_score', 0):.2f} ({altman_z_score.get('zone', 'Unknown')})")
                if 'error' in graham_score:
                    st.warning(f"Graham error: {graham_score['error']}")
                if 'error' in piotroski_score:
                    st.warning(f"Piotroski error: {piotroski_score['error']}")
                if 'error' in altman_z_score:
                    st.warning(f"Altman error: {altman_z_score['error']}")
        
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
                'altman_z_score': altman_z_score
            },
            'confidence_analysis': confidence_analysis,
            'system_status': 'OPERATIONAL v4.2.2'
        }
        
        # Add optional analyses if available (UNCHANGED)
        if volume_analysis:
            analysis_results['enhanced_indicators']['volume_analysis'] = volume_analysis
        
        if volatility_analysis:
            analysis_results['enhanced_indicators']['volatility_analysis'] = volatility_analysis
        
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
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                
                if VOLUME_ANALYSIS_AVAILABLE:
                    show_volume_analysis(analysis_results, controls['show_debug'])
                
                if VOLATILITY_ANALYSIS_AVAILABLE:
                    show_volatility_analysis(analysis_results, controls['show_debug'])
                
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
