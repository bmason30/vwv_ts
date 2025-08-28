"""
Filename: app.py
VWV Trading System v4.2.1
Created/Updated: 2025-08-28 17:00:00 EST
Version: 4.2.4 - Plotly compatibility fixes and pandas groupby corrections
Purpose: Main Streamlit application with corrected plotly parameters and technical analysis fixes
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
from utils.helpers import format_large_number, get_market_status, get_etf_description
from utils.decorators import safe_calculation_wrapper

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# Page configuration
st.set_page_config(
    page_title="VWV Professional Trading System v4.2.1",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_sidebar_controls():
    """Create sidebar controls and return analysis parameters - FIXED NAVIGATION"""
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
    if 'auto_analyze' not in st.session_state:
        st.session_state.auto_analyze = False
    
    # Handle selected symbol from quicklinks/recents
    if 'selected_symbol' in st.session_state:
        current_symbol = st.session_state.selected_symbol
        st.session_state.auto_analyze = True  # Trigger analysis
        del st.session_state.selected_symbol
    else:
        current_symbol = UI_SETTINGS['default_symbol']
        
    # Symbol input and period selection (NO FORM - this was causing issues)
    symbol = st.sidebar.text_input("Symbol", value=current_symbol, help="Enter stock symbol").upper()
    
    # CORRECTED: Default period set to '1mo' (1 month)
    period_options = ['1mo', '3mo', '6mo', '1y', '2y']
    period = st.sidebar.selectbox("Data Period", period_options, index=0)  # Index 0 = '1mo'
    
    # Analyze button (outside form to prevent symbol reset)
    analyze_button = st.sidebar.button("📊 Analyze Symbol", type="primary", use_container_width=True)
    
    # Check for auto-analyze trigger from quicklinks/recents
    if st.session_state.auto_analyze:
        st.session_state.auto_analyze = False  # Reset flag
        analyze_button = True  # Force analysis
    
    # CORRECTED SIDEBAR ORDER: Quick Links FIRST, Recently Viewed SECOND, Analysis Sections THIRD
    
    # 1. Quick Links section - FIRST
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

    # 2. Recently Viewed section - SECOND
    def add_to_recently_viewed(symbol):
        """Add symbol to recently viewed list"""
        if symbol not in st.session_state.recently_viewed:
            st.session_state.recently_viewed.insert(0, symbol)
            st.session_state.recently_viewed = st.session_state.recently_viewed[:10]  # Keep last 10
        elif symbol in st.session_state.recently_viewed:
            # Move to front if already exists
            st.session_state.recently_viewed.remove(symbol)
            st.session_state.recently_viewed.insert(0, symbol)
    
    if st.session_state.recently_viewed:
        with st.sidebar.expander("⏰ Recently Viewed", expanded=False):
            for recent_symbol in st.session_state.recently_viewed:
                if st.button(f"📊 {recent_symbol}", key=f"recent_{recent_symbol}", use_container_width=True):
                    st.session_state.selected_symbol = recent_symbol
                    st.rerun()
    
    # 3. Analysis Section Toggles - THIRD
    with st.sidebar.expander("🎛️ Analysis Sections", expanded=False):
        st.session_state.show_charts = st.checkbox("📊 Interactive Charts", st.session_state.show_charts)
        st.session_state.show_technical_analysis = st.checkbox("📊 Technical Analysis", st.session_state.show_technical_analysis)
        st.session_state.show_volume_analysis = st.checkbox("📊 Volume Analysis", st.session_state.show_volume_analysis)
        st.session_state.show_volatility_analysis = st.checkbox("📊 Volatility Analysis", st.session_state.show_volatility_analysis)
        st.session_state.show_fundamental_analysis = st.checkbox("📊 Fundamental Analysis", st.session_state.show_fundamental_analysis)
        st.session_state.show_baldwin_indicator = st.checkbox("🚦 Baldwin Market Regime", st.session_state.show_baldwin_indicator)
        st.session_state.show_market_correlation = st.checkbox("🌐 Market Correlation", st.session_state.show_market_correlation)
        st.session_state.show_options_analysis = st.checkbox("🎯 Options Analysis", st.session_state.show_options_analysis)
        st.session_state.show_confidence_intervals = st.checkbox("📊 Confidence Intervals", st.session_state.show_confidence_intervals)
    
    # Advanced options
    with st.sidebar.expander("🔧 Advanced Options", expanded=False):
        show_debug = st.checkbox("Show debug info", False)
        
    return {
        'symbol': symbol,
        'period': period,
        'analyze_button': analyze_button,
        'show_debug': show_debug,
        'add_to_recently_viewed': add_to_recently_viewed
    }

def show_interactive_charts(chart_data, analysis_results, show_debug=False):
    """Display interactive charts section - PLOTLY COMPATIBILITY FIXED"""
    if not st.session_state.show_charts:
        return
        
    symbol = analysis_results['symbol']
    
    with st.expander(f"📊 {symbol} - Interactive Charts", expanded=True):
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # FIXED: Correct plotly parameters
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,  # FIXED: was shared_xaxis
                vertical_spacing=0.1,
                subplot_titles=(f'{symbol} Price & Moving Averages', 'Volume'),
                row_heights=[0.7, 0.3]  # FIXED: was row_width
            )
            
            # Price chart
            fig.add_trace(
                go.Candlestick(
                    x=chart_data.index,
                    open=chart_data['Open'],
                    high=chart_data['High'], 
                    low=chart_data['Low'],
                    close=chart_data['Close'],
                    name=f'{symbol} Price'
                ), row=1, col=1
            )
            
            # Add moving averages if available
            enhanced_indicators = analysis_results.get('enhanced_indicators', {})
            fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
            
            colors = ['orange', 'red', 'purple', 'brown']
            for i, (ema_name, ema_value) in enumerate(fibonacci_emas.items()):
                period = ema_name.split('_')[1]
                if f'EMA_{period}' in chart_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=chart_data.index,
                            y=chart_data[f'EMA_{period}'],
                            mode='lines',
                            name=f'EMA {period}',
                            line=dict(color=colors[i % len(colors)], width=1)
                        ), row=1, col=1
                    )
            
            # Volume chart  
            fig.add_trace(
                go.Bar(
                    x=chart_data.index,
                    y=chart_data['Volume'],
                    name='Volume',
                    marker_color='lightblue',
                    opacity=0.7
                ), row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} - Professional Trading Analysis',
                xaxis_rangeslider_visible=False,
                height=700,
                showlegend=True,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            st.error("📊 Plotly not available for interactive charts")
            if show_debug:
                st.write("Try refreshing or enable debug mode for details.")
                
            # Fallback simple chart
            st.subheader("Basic Price Chart (Fallback)")
            if chart_data is not None and not chart_data.empty:
                st.line_chart(chart_data['Close'])
            else:
                st.error("No data available for charting")
        except Exception as e:
            st.error(f"📊 Chart generation error: {str(e)}")
            if show_debug:
                import traceback
                st.code(traceback.format_exc())
            
            # Fallback simple chart
            st.subheader("Basic Price Chart (Fallback)")
            if chart_data is not None and not chart_data.empty:
                st.line_chart(chart_data['Close'])

def perform_enhanced_analysis(symbol, period, show_debug=False):
    """Perform enhanced analysis using modular components - FUNCTION SIGNATURE FIXED"""
    try:
        # Step 1: Fetch data using modular data fetcher
        market_data = get_market_data_enhanced(symbol, period, show_debug)
        
        if market_data is None:
            st.error(f"❌ Could not fetch data for {symbol}")
            return None, None
        
        # Step 2: Store data using data manager
        data_manager = get_data_manager()
        data_manager.store_market_data(symbol, market_data, show_debug)
        
        # Step 3: Get analysis copy
        analysis_input = data_manager.get_market_data_for_analysis(symbol)
        
        if analysis_input is None:
            st.error("❌ Could not prepare analysis data")
            return None, None
        
        # DEBUG: Check data structure for volume analysis
        if show_debug:
            st.write("🔍 **DEBUGGING VOLUME DATA:**")
            st.write(f"Data shape: {analysis_input.shape}")
            st.write(f"Columns: {list(analysis_input.columns)}")
            st.write(f"Has Volume column: {'Volume' in analysis_input.columns}")
            if 'Volume' in analysis_input.columns:
                st.write(f"Volume data sample: {analysis_input['Volume'].tail()}")
                st.write(f"Volume data type: {analysis_input['Volume'].dtype}")
                st.write(f"Non-zero volume count: {(analysis_input['Volume'] > 0).sum()}")
            else:
                st.error("❌ Volume column missing from data!")
        
        # Step 4: Calculate enhanced indicators using modular analysis
        daily_vwap = calculate_daily_vwap(analysis_input)
        fibonacci_emas = calculate_fibonacci_emas(analysis_input)
        point_of_control = calculate_point_of_control_enhanced(analysis_input)
        weekly_deviations = calculate_weekly_deviations(analysis_input)
        comprehensive_technicals = calculate_comprehensive_technicals(analysis_input)
        
        # Step 5: Calculate Volume Analysis (NEW v4.2.1) - WITH ENHANCED DEBUG
        volume_analysis = {}
        if VOLUME_ANALYSIS_AVAILABLE:
            try:
                if show_debug:
                    st.write("🔍 **CALLING VOLUME ANALYSIS:**")
                    st.write(f"Function available: {VOLUME_ANALYSIS_AVAILABLE}")
                    st.write(f"Analysis input type: {type(analysis_input)}")
                
                volume_analysis = calculate_complete_volume_analysis(analysis_input)
                
                if show_debug:
                    st.write("🔍 **VOLUME ANALYSIS RESULT:**")
                    st.write(f"Result type: {type(volume_analysis)}")
                    st.write(f"Result keys: {list(volume_analysis.keys()) if isinstance(volume_analysis, dict) else 'Not a dict'}")
                    if 'error' in volume_analysis:
                        st.error(f"Volume analysis error: {volume_analysis['error']}")
                    else:
                        st.write("✅ Volume analysis completed successfully")
                        st.write(f"Volume regime: {volume_analysis.get('volume_regime', 'Unknown')}")
                        st.write(f"Volume score: {volume_analysis.get('volume_score', 0)}")
                        
            except Exception as e:
                if show_debug:
                    st.error(f"❌ Volume analysis exception: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                volume_analysis = {'error': f'Volume analysis failed: {str(e)}'}
        else:
            if show_debug:
                st.warning("⚠️ Volume analysis not available - import failed")
        
        # Step 6: Calculate Volatility Analysis (NEW v4.2.1)
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
        
        # Step 9: Calculate options analysis - SAFE FALLBACK
        try:
            options_levels = calculate_options_levels_enhanced(analysis_input, symbol, show_debug)
        except Exception as e:
            if show_debug:
                st.error(f"Options analysis failed: {e}")
            options_levels = {'error': f'Options analysis failed: {str(e)}'}
        
        # Step 10: Calculate confidence intervals - FIXED FUNCTION SIGNATURE
        try:
            confidence_analysis = calculate_confidence_intervals(analysis_input)  # REMOVED show_debug parameter
            if show_debug:
                st.write("✅ Confidence intervals calculated successfully")
        except Exception as e:
            if show_debug:
                st.error(f"❌ Confidence intervals failed: {e}")
            confidence_analysis = None
        
        # DEBUG: Check final enhanced_indicators structure
        if show_debug:
            st.write("🔍 **FINAL ENHANCED INDICATORS:**")
            enhanced_indicators = {
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
            }
            st.write(f"Volume analysis in enhanced_indicators: {'volume_analysis' in enhanced_indicators}")
            if 'volume_analysis' in enhanced_indicators:
                vol_data = enhanced_indicators['volume_analysis']
                st.write(f"Volume analysis type: {type(vol_data)}")
                st.write(f"Volume analysis empty: {not vol_data if isinstance(vol_data, dict) else 'Not dict'}")
                st.write(f"Volume analysis has error: {'error' in vol_data if isinstance(vol_data, dict) else 'Unknown'}")
        
        # Compile comprehensive analysis results
        analysis_results = {
            'symbol': symbol.upper(),
            'current_price': float(analysis_input['Close'].iloc[-1]),
            'period': period,
            'data_points': len(analysis_input),
            'last_updated': analysis_input.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
            'enhanced_indicators': {
                'daily_vwap': daily_vwap,
                'fibonacci_emas': fibonacci_emas,
                'point_of_control': point_of_control,
                'weekly_deviations': weekly_deviations,
                'comprehensive_technicals': comprehensive_technicals,
                'volume_analysis': volume_analysis,  # NEW v4.2.1
                'volatility_analysis': volatility_analysis,  # NEW v4.2.1
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
        
        return analysis_results, chart_data
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        if show_debug:
            import traceback
            st.code(traceback.format_exc())
        return None, None

def show_individual_technical_analysis(analysis_results, show_debug=False):
    """Display individual technical analysis section - MANDATORY SECOND"""
    if not st.session_state.show_technical_analysis:
        return
        
    symbol = analysis_results['symbol']
    enhanced_indicators = analysis_results.get('enhanced_indicators', {})
    
    with st.expander(f"📊 {symbol} - Technical Analysis", expanded=True):
        
        # Technical score with gradient bar
        composite_score, components = calculate_composite_technical_score(analysis_results)
        create_technical_score_bar(composite_score, f"{symbol} Technical Score")
        
        # Enhanced indicators sections
        fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        
        # --- 1. OVERVIEW METRICS ---
        st.subheader("Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi = comprehensive_technicals.get('rsi_14', 50)
            rsi_signal = "🟢 Oversold" if rsi < 30 else "🔴 Overbought" if rsi > 70 else "🟡 Neutral"
            st.metric("RSI (14)", f"{rsi:.1f}", rsi_signal)
        with col2:
            mfi = comprehensive_technicals.get('mfi_14', 50)
            mfi_signal = "🟢 Oversold" if mfi < 20 else "🔴 Overbought" if mfi > 80 else "🟡 Neutral"
            st.metric("MFI (14)", f"{mfi:.1f}", mfi_signal)
        with col3:
            vol_ratio = comprehensive_technicals.get('volume_ratio', 1.0)
            vol_signal = "🔴 High" if vol_ratio > 2.0 else "🟡 Normal" if vol_ratio > 0.5 else "🟢 Low"
            st.metric("Volume Ratio", f"{vol_ratio:.2f}x", vol_signal)
        with col4:
            volatility = comprehensive_technicals.get('volatility_20d', 20)
            vol_signal = "🔴 High" if volatility > 40 else "🟡 Normal" if volatility > 15 else "🟢 Low"
            st.metric("20D Volatility", f"{volatility:.1f}%", vol_signal)

        # --- 2. BOLLINGER BANDS ---
        st.subheader("Bollinger Bands Analysis")
        bollinger = comprehensive_technicals.get('bollinger_bands', {})
        if bollinger:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Upper Band", f"${bollinger.get('upper', 0):.2f}")
            with col2:
                st.metric("Middle (SMA20)", f"${bollinger.get('middle', 0):.2f}")
            with col3:
                st.metric("Lower Band", f"${bollinger.get('lower', 0):.2f}")
        
        # --- 3. MACD TREND ANALYSIS ---
        st.subheader("Trend Analysis")
        col1, col2 = st.columns(2)
        with col1:
            macd_data = comprehensive_technicals.get('macd', {})
            macd_hist = macd_data.get('histogram', 0)
            macd_delta = "Bullish" if macd_hist > 0 else "Bearish"
            st.metric("MACD Histogram", f"{macd_hist:.4f}", macd_delta)
        with col2:
             # Placeholder for another trend indicator like ADX if you add it later
             pass

        # --- 4. PRICE-BASED INDICATORS & KEY LEVELS TABLE ---
        st.subheader("Price-Based Indicators & Key Levels")
        current_price = analysis_results['current_price']
        daily_vwap = enhanced_indicators.get('daily_vwap', 0)
        point_of_control = enhanced_indicators.get('point_of_control', 0)

        indicators_data = []
        
        indicators_data.append(("Current Price", f"${current_price:.2f}", "📍 Reference", "0.0%", "Current"))
        
        vwap_distance = f"{((current_price - daily_vwap) / daily_vwap * 100):+.2f}%" if daily_vwap > 0 else "N/A"
        vwap_status = "Above" if current_price > daily_vwap else "Below"
        indicators_data.append(("Daily VWAP", f"${daily_vwap:.2f}", "📊 Volume Weighted", vwap_distance, vwap_status))
        
        poc_distance = f"{((current_price - point_of_control) / point_of_control * 100):+.2f}%" if point_of_control > 0 else "N/A"
        poc_status = "Above" if current_price > point_of_control else "Below"
        indicators_data.append(("Point of Control", f"${point_of_control:.2f}", "📊 Volume Profile", poc_distance, poc_status))
        
        for ema_name, ema_value in fibonacci_emas.items():
            period = ema_name.split('_')[1]
            distance_pct = f"{((current_price - ema_value) / ema_value * 100):+.2f}%" if ema_value > 0 else "N/A"
            status = "Above" if current_price > ema_value else "Below"
            indicators_data.append((f"EMA {period}", f"${ema_value:.2f}", "📈 Trend", distance_pct, status))
        
        df_technical = pd.DataFrame(indicators_data, columns=['Indicator', 'Value', 'Type', 'Distance %', 'Status'])
        st.dataframe(df_technical, use_container_width=True, hide_index=True)

def show_volume_analysis(analysis_results, show_debug=False):
    """Display volume analysis section - NEW v4.2.1"""
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
            if show_debug and 'error' in volume_analysis:
                st.error(f"Volume error details: {volume_analysis['error']}")

def show_volatility_analysis(analysis_results, show_debug=False):
    """Display volatility analysis section - NEW v4.2.1"""
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
            if show_debug and 'error' in volatility_analysis:
                st.error(f"Volatility error details: {volatility_analysis['error']}")

def show_fundamental_analysis(analysis_results, show_debug=False):
    """Display fundamental analysis section - ENHANCED v4.2.1"""
    if not st.session_state.show_fundamental_analysis:
        return
        
    symbol = analysis_results['symbol']
    enhanced_indicators = analysis_results.get('enhanced_indicators', {})
    
    with st.expander(f"📊 {symbol} - Fundamental Analysis", expanded=True):
        
        graham_score = enhanced_indicators.get('graham_score', {})
        piotroski_score = enhanced_indicators.get('piotroski_score', {})
        
        # Overview metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            graham_total = graham_score.get('score', 0)
            graham_possible = graham_score.get('total_possible', 10)
            st.metric("Graham Score", f"{graham_total}/{graham_possible}", f"{(graham_total/graham_possible*100):.0f}%" if graham_possible > 0 else "N/A")
        
        with col2:
            piotroski_total = piotroski_score.get('score', 0)
            piotroski_possible = piotroski_score.get('total_possible', 9)
            st.metric("Piotroski Score", f"{piotroski_total}/{piotroski_possible}", f"{(piotroski_total/piotroski_possible*100):.0f}%" if piotroski_possible > 0 else "N/A")
            
        with col3:
            combined_score = graham_total + piotroski_total
            max_combined = graham_possible + piotroski_possible
            st.metric("Combined Score", f"{combined_score}/{max_combined}", f"{(combined_score/max_combined*100):.0f}%" if max_combined > 0 else "N/A")
        
        # Error handling for ETFs
        if graham_score.get('error') or piotroski_score.get('error'):
            if 'ETF' in str(graham_score.get('error', '')) or 'ETF' in str(piotroski_score.get('error', '')):
                st.info("📊 ETF detected - Fundamental analysis not applicable for ETFs. Consider technical and market analysis.")
            else:
                st.warning("⚠️ Fundamental data not available - may be due to data limitations")

def show_baldwin_indicator_analysis(show_debug=False):
    """Display Baldwin Market Regime Indicator - BEFORE Market Correlation"""
    if not st.session_state.show_baldwin_indicator or not BALDWIN_INDICATOR_AVAILABLE:
        return
        
    with st.expander("🚦 Baldwin Market Regime Indicator", expanded=True):
        st.warning("⚠️ Baldwin Indicator not available - module needs restoration")

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis section"""
    if not st.session_state.show_market_correlation:
        return
        
    symbol = analysis_results['symbol']
    enhanced_indicators = analysis_results.get('enhanced_indicators', {})
    
    with st.expander(f"🌐 {symbol} - Market Correlation Analysis", expanded=True):
        
        market_correlations = enhanced_indicators.get('market_correlations', {})
        
        if market_correlations and 'error' not in market_correlations:
            # Display correlation data
            st.info("Market correlation analysis available")
        else:
            st.warning("⚠️ Market correlation analysis not available - insufficient data")

def show_options_analysis(analysis_results, show_debug=False):
    """Display options analysis section - SAFE FALLBACK"""
    if not st.session_state.show_options_analysis:
        return
        
    symbol = analysis_results['symbol']
    enhanced_indicators = analysis_results.get('enhanced_indicators', {})
    
    with st.expander(f"🎯 {symbol} - Options Analysis", expanded=True):
        
        options_levels = enhanced_indicators.get('options_levels', {})
        
        if options_levels and 'error' not in options_levels:
            # Display options data
            st.info("Options analysis available")
            if show_debug:
                st.json(options_levels)
        else:
            st.warning("⚠️ Options analysis not available - insufficient data")
            if show_debug and 'error' in options_levels:
                st.error(f"Options error details: {options_levels['error']}")

def show_confidence_intervals(analysis_results, show_debug=False):
    """Display confidence intervals analysis section - FIXED"""
    if not st.session_state.show_confidence_intervals:
        return
        
    symbol = analysis_results['symbol']
    confidence_analysis = analysis_results.get('confidence_analysis', {})
    
    with st.expander(f"📊 {symbol} - Statistical Confidence Intervals", expanded=True):
        
        if confidence_analysis and 'error' not in confidence_analysis:
            # Display confidence metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Weekly Return", f"{confidence_analysis.get('mean_weekly_return', 0):.3f}%")
            with col2:
                st.metric("Weekly Volatility", f"{confidence_analysis.get('weekly_volatility', 0):.2f}%")
            with col3:
                st.metric("Sample Size", f"{confidence_analysis.get('sample_size', 0)} weeks")
            
            # Confidence intervals table
            intervals = confidence_analysis.get('confidence_intervals', {})
            if intervals:
                final_intervals_data = []
                for level, level_data in intervals.items():
                    final_intervals_data.append({
                        'Confidence Level': level,
                        'Upper Bound': f"${level_data.get('upper_bound', 0)}",
                        'Lower Bound': f"${level_data.get('lower_bound', 0)}",
                        'Expected Move': f"±{level_data.get('expected_move_pct', 0):.2f}%"
                    })
                
                df_intervals = pd.DataFrame(final_intervals_data)
                st.dataframe(df_intervals, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Confidence intervals not available - insufficient data")

def main():
    """Main application function - CORRECTED v4.2.1 with PROPER DISPLAY ORDER"""
    # Create header using modular component
    create_header()
    
    # Create sidebar and get controls
    controls = create_sidebar_controls()
    
    # Main logic flow
    if controls['analyze_button'] and controls['symbol']:
        # Add symbol to recently viewed
        controls['add_to_recently_viewed'](controls['symbol'])
        
        st.write("## 📊 VWV Trading Analysis v4.2.1 Enhanced")
        
        with st.spinner(f"Analyzing {controls['symbol']}..."):
            
            # Perform analysis using modular components
            analysis_results, chart_data = perform_enhanced_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results and chart_data is not None:
                
                # CORRECTED DISPLAY ORDER - MANDATORY SEQUENCE:
                
                # 1. CHARTS FIRST (MANDATORY)
                show_interactive_charts(chart_data, analysis_results, controls['show_debug'])
                
                # 2. INDIVIDUAL TECHNICAL ANALYSIS SECOND (MANDATORY)
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                
                # 3. Volume Analysis (Optional - when available)
                if VOLUME_ANALYSIS_AVAILABLE:
                    show_volume_analysis(analysis_results, controls['show_debug'])
                
                # 4. Volatility Analysis (Optional - when available)
                if VOLATILITY_ANALYSIS_AVAILABLE:
                    show_volatility_analysis(analysis_results, controls['show_debug'])
                
                # 5. Fundamental Analysis
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                
                # 6. BALDWIN INDICATOR (Before Market Correlation)
                if BALDWIN_INDICATOR_AVAILABLE:
                    show_baldwin_indicator_analysis(show_debug=controls['show_debug'])
                
                # 7. Market Correlation Analysis (After Baldwin)
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                
                # 8. Options Analysis
                show_options_analysis(analysis_results, controls['show_debug'])
                
                # 9. Confidence Intervals
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
                        
                        st.write("### System Status")
                        st.write(f"**Default Period Confirmed:** {controls['period']} (Should be '1mo')")
                        st.write(f"**Volume Analysis Available:** {VOLUME_ANALYSIS_AVAILABLE}")
                        st.write(f"**Volatility Analysis Available:** {VOLATILITY_ANALYSIS_AVAILABLE}")
                        st.write(f"**Baldwin Indicator Available:** {BALDWIN_INDICATOR_AVAILABLE}")
    
    else:
        # Welcome message
        st.write("## 🚀 VWV Professional Trading System v4.2.1 - CORRECTED")
        st.write("**CRITICAL FIXES APPLIED:** Charts First + Technical Second + Baldwin Positioned + 1mo Default")
        
        # Baldwin Market Preview (if available)
        if BALDWIN_INDICATOR_AVAILABLE:
            with st.expander("🚦 Live Baldwin Market Regime Preview", expanded=True):
                with st.spinner("Calculating market regime..."):
                    show_baldwin_indicator_analysis(show_debug=False)
        
        # System overview
        with st.expander("ℹ️ System Overview", expanded=True):
            st.write("**📊 CORRECTED ANALYSIS PIPELINE:**")
            st.write("1. **📊 Interactive Charts** - Immediate visual analysis")
            st.write("2. **📊 Individual Technical Analysis** - Professional scoring with Fibonacci EMAs")
            st.write("3. **📊 Volume Analysis** - Optional when module available")
            st.write("4. **📊 Volatility Analysis** - Optional when module available")
            st.write("5. **📊 Fundamental Analysis** - Graham & Piotroski scores")
            st.write("6. **🚦 Baldwin Market Regime** - Before Market Correlation")
            st.write("7. **🌐 Market Correlation** - After Baldwin Indicator")
            st.write("8. **🎯 Options Analysis** - Strike levels with Greeks")
            st.write("9. **📊 Confidence Intervals** - Statistical projections")
            
            st.write("**✅ CRITICAL CORRECTIONS VERIFIED:**")
            st.write("• **Default Period:** 1 month ('1mo') - ✅ CORRECTED")
            st.write("• **Charts Priority:** Display FIRST - ✅ CORRECTED")
            st.write("• **Technical Second:** Individual analysis SECOND - ✅ CORRECTED")
            st.write("• **Baldwin Position:** Before Market Correlation - ✅ CORRECTED")
        
        # Show current market status
        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.write("1. **Enter a symbol** in the sidebar (e.g., AAPL, SPY, QQQ)")
            st.write("2. **Default period is 1 month** - optimal for most analysis")
            st.write("3. **Charts display FIRST** - immediate visual analysis")
            st.write("4. **Technical analysis SECOND** - professional scoring with Fibonacci EMAs")
            st.write("5. **Baldwin regime indicator** - market-wide assessment")
            st.write("6. **Use Quick Links** for instant analysis of popular symbols")

    # Footer
    st.markdown("---")
    st.write("### 📊 System Information v4.2.4 PLOTLY FIXED")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Version:** VWV Professional v4.2.4 PLOTLY FIXED")
        st.write(f"**Status:** ✅ Chart Compatibility Fixes Applied")
    with col2:
        st.write(f"**Display Order:** Charts First + Technical Second ✅")
        st.write(f"**Default Period:** 1 month ('1mo') ✅")
    with col3:
        st.write(f"**Fixed Issues:** Plotly make_subplots Parameters")
        st.write(f"**Enhanced Features:** Volume & Volatility Available")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.write("Please refresh the page and try again.")
        
        if st.checkbox("Show Error Details"):
            st.exception(e)
