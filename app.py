"""
VWV Professional Trading System v4.2.2
Complete Trading Analysis Platform with Modular Architecture
Created: 2025-10-02
Updated: 2025-10-02
Version: v4.2.2
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

# Page configuration
st.set_page_config(
    page_title="VWV Professional Trading System v4.2.2",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_sidebar_controls():
    """Create sidebar controls and return analysis parameters"""
    st.sidebar.title("📊 Trading Analysis v4.2.2")
    
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
    
    # Symbol input with quick links
    st.sidebar.subheader("🎯 Symbol Selection")
    symbol = st.sidebar.text_input(
        "Enter Symbol",
        value="",
        placeholder="e.g., AAPL, SPY, QQQ",
        key="symbol_input"
    ).upper()
    
    # Analysis period
    period_options = {
        '1 Month': '1mo',
        '3 Months': '3mo',
        '6 Months': '6mo',
        '1 Year': '1y',
        '2 Years': '2y'
    }
    selected_period = st.sidebar.selectbox(
        "Analysis Period",
        options=list(period_options.keys()),
        index=0
    )
    period = period_options[selected_period]
    
    # Analyze button
    analyze_button = st.sidebar.button("🔍 Analyze Symbol", type="primary", use_container_width=True)
    
    # Quick Links
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Quick Links")
    
    for category, symbols in QUICK_LINK_CATEGORIES.items():
        with st.sidebar.expander(f"📊 {category}", expanded=False):
            cols = st.columns(2)
            for idx, sym in enumerate(symbols):
                description = SYMBOL_DESCRIPTIONS.get(sym, sym)
                col_idx = idx % 2
                with cols[col_idx]:
                    if st.button(
                        sym,
                        key=f"quick_{sym}",
                        help=description,
                        use_container_width=True
                    ):
                        st.session_state['symbol_input'] = sym
                        st.rerun()
    
    # Recently viewed
    if st.session_state.recently_viewed:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🕐 Recently Viewed")
        cols = st.sidebar.columns(3)
        for idx, recent_sym in enumerate(st.session_state.recently_viewed[:6]):
            col_idx = idx % 3
            with cols[col_idx]:
                if st.button(
                    recent_sym,
                    key=f"recent_{recent_sym}_{idx}",
                    use_container_width=True
                ):
                    st.session_state['symbol_input'] = recent_sym
                    st.rerun()
    
    # Analysis sections toggle
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Analysis Sections")
    
    with st.sidebar.expander("Toggle Sections", expanded=False):
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
        if BALDWIN_INDICATOR_AVAILABLE:
            st.session_state.show_baldwin_indicator = st.checkbox(
                "Baldwin Market Regime",
                value=st.session_state.show_baldwin_indicator,
                key="toggle_baldwin"
            )
        st.session_state.show_market_correlation = st.checkbox(
            "Market Correlation",
            value=st.session_state.show_market_correlation,
            key="toggle_market"
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
    
    # Debug mode
    st.sidebar.markdown("---")
    show_debug = st.sidebar.checkbox("🐛 Debug Mode", value=False)
    
    return {
        'symbol': symbol,
        'period': period,
        'analyze_button': analyze_button,
        'show_debug': show_debug
    }

def add_to_recently_viewed(symbol):
    """Add symbol to recently viewed list"""
    if symbol not in st.session_state.recently_viewed:
        st.session_state.recently_viewed.insert(0, symbol)
        st.session_state.recently_viewed = st.session_state.recently_viewed[:10]

def show_interactive_charts(data, analysis_results, show_debug=False):
    """Display interactive charts section"""
    with st.expander(f"📊 {analysis_results['symbol']} - Interactive Charts", expanded=True):
        try:
            # Try to import and use plotly for charts
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=('Price & Indicators', 'Volume', 'RSI')
            )
            
            # Add candlestick
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add Fibonacci EMAs
            enhanced_indicators = analysis_results.get('enhanced_indicators', {})
            fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
            
            colors = ['#FF6692', '#4ECDC4', '#FFD93D', '#6BCF7F', '#A8E6CF']
            for idx, (ema_name, ema_value) in enumerate(fibonacci_emas.items()):
                period = ema_name.split('_')[1]
                # Calculate EMA series for plotting
                ema_series = data['Close'].ewm(span=int(period), adjust=False).mean()
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=ema_series,
                        mode='lines',
                        name=f'EMA {period}',
                        line=dict(color=colors[idx % len(colors)], width=1.5)
                    ),
                    row=1, col=1
                )
            
            # Add volume
            colors_vol = ['red' if row['Close'] < row['Open'] else 'green' 
                         for idx, row in data.iterrows()]
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color=colors_vol,
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Add RSI
            comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
            if 'rsi_series' in comprehensive_technicals:
                rsi_series = comprehensive_technicals['rsi_series']
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=rsi_series,
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple', width=2)
                    ),
                    row=3, col=1
                )
                
                # Add RSI reference lines
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, opacity=0.5)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, opacity=0.5)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1, opacity=0.3)
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                template='plotly_dark'
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            if show_debug:
                st.error(f"Chart generation error: {str(e)}")
                st.code(traceback.format_exc())
            
            # Fallback simple chart
            st.subheader("Basic Price Chart (Fallback)")
            if data is not None and not data.empty:
                st.line_chart(data['Close'])
            else:
                st.error("No data available for charting")

def show_individual_technical_analysis(analysis_results, show_debug=False):
    """Display individual technical analysis section"""
    if not st.session_state.get('show_technical_analysis', True):
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Technical Analysis", expanded=True):
        
        # Composite Technical Score Bar
        composite_score, score_details = calculate_composite_technical_score(analysis_results)
        score_bar_html = create_technical_score_bar(composite_score, score_details)
        st.components.v1.html(score_bar_html, height=160)
        
        # Prepare data references
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
        
        # Key Momentum Oscillators
        st.subheader("Key Momentum Oscillators")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            rsi = comprehensive_technicals.get('rsi_14', 50)
            st.metric("RSI (14)", f"{rsi:.2f}", "Oversold < 30")
        with col2:
            mfi = comprehensive_technicals.get('mfi_14', 50)
            st.metric("MFI (14)", f"{mfi:.2f}", "Oversold < 20")
        with col3:
            stoch = comprehensive_technicals.get('stochastic', {})
            st.metric("Stochastic %K", f"{stoch.get('k', 50):.2f}", "Oversold < 20")
        with col4:
            williams_r = comprehensive_technicals.get('williams_r', -50)
            st.metric("Williams %R", f"{williams_r:.2f}", "Oversold < -80")

        # Trend Analysis
        st.subheader("Trend Analysis")
        col1, col2 = st.columns(2)
        with col1:
            macd_data = comprehensive_technicals.get('macd', {})
            macd_hist = macd_data.get('histogram', 0)
            macd_delta = "Bullish" if macd_hist > 0 else "Bearish"
            st.metric("MACD Histogram", f"{macd_hist:.4f}", macd_delta)
        with col2:
            adx = comprehensive_technicals.get('adx', 25)
            if adx:
                st.metric("ADX", f"{adx:.2f}", "Strong trend > 25")

        # Price-Based Indicators & Key Levels
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
                st.metric("Volume Ratio", f"{volume_ratio:.2f}x", "vs 30D avg")
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
                st.metric("5D Volatility", f"{volatility_analysis.get('volatility_5d', 0):.2f}%", "Annualized")
            with col2:
                st.metric("30D Volatility", f"{volatility_analysis.get('volatility_30d', 0):.2f}%", "Annualized")
            with col3:
                vol_ratio = volatility_analysis.get('volatility_ratio', 1.0)
                st.metric("Vol Ratio", f"{vol_ratio:.2f}x", "5D vs 30D")
            with col4:
                vol_trend = volatility_analysis.get('volatility_trend', 0)
                st.metric("Vol Trend", f"{vol_trend:+.2f}%", "5D Change")
            
            # Volatility regime and implications
            st.subheader("📊 Volatility Environment")
            vol_regime = volatility_analysis.get('volatility_regime', 'Unknown')
            trading_implications = volatility_analysis.get('trading_implications', 'No implications available')
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Volatility Regime:** {vol_regime}")
                st.info(f"**Volatility Score:** {volatility_analysis.get('volatility_score', 50)}/100")
            with col2:
                st.info(f"**Trading Implications:**\n{trading_implications}")
                
        else:
            st.warning("⚠️ Volatility analysis not available - insufficient data")

def show_fundamental_analysis(analysis_results, show_debug=False):
    """Display fundamental analysis section"""
    if not st.session_state.show_fundamental_analysis:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Fundamental Analysis", expanded=True):
        
        fundamental_data = analysis_results.get('fundamental_analysis', {})
        graham_data = fundamental_data.get('graham_score', {})
        piotroski_data = fundamental_data.get('piotroski_score', {})
        
        # Summary metrics
        st.subheader("Investment Quality Scores")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'error' not in graham_data:
                st.metric(
                    "Graham Score", 
                    f"{graham_data.get('total_score', 0)}/10",
                    graham_data.get('grade', 'N/A')
                )
            else:
                st.metric("Graham Score", "N/A", "Insufficient Data")
        
        with col2:
            if 'error' not in graham_data:
                st.metric(
                    "Graham %", 
                    f"{graham_data.get('percentage', 0):.0f}%",
                    graham_data.get('interpretation', '')[:20] + "..."
                )
            else:
                st.metric("Graham %", "0%", "No Data")
        
        with col3:
            if 'error' not in piotroski_data:
                st.metric(
                    "Piotroski Score", 
                    f"{piotroski_data.get('total_score', 0)}/9",
                    piotroski_data.get('grade', 'N/A')
                )
            else:
                st.metric("Piotroski Score", "N/A", "Insufficient Data")
        
        with col4:
            if 'error' not in piotroski_data:
                st.metric(
                    "Piotroski %", 
                    f"{piotroski_data.get('percentage', 0):.0f}%",
                    piotroski_data.get('interpretation', '')[:20] + "..."
                )
            else:
                st.metric("Piotroski %", "0%", "No Data")
        
        # Detailed interpretations
        st.subheader("📈 Investment Analysis Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'error' not in graham_data:
                graham_interp = graham_data.get('interpretation', 'No interpretation available')
                graham_grade = graham_data.get('grade', 'N/A')
                
                if graham_grade in ['A', 'B']:
                    st.success(f"**Graham Analysis: {graham_grade} Grade**\n\n{graham_interp}")
                elif graham_grade in ['C', 'D']:
                    st.warning(f"**Graham Analysis: {graham_grade} Grade**\n\n{graham_interp}")
                else:
                    st.error(f"**Graham Analysis: {graham_grade} Grade**\n\n{graham_interp}")
            else:
                st.info("**Graham Analysis:** Data insufficient for comprehensive analysis")
        
        with col2:
            if 'error' not in piotroski_data:
                piotroski_interp = piotroski_data.get('interpretation', 'No interpretation available')
                piotroski_grade = piotroski_data.get('grade', 'N/A')
                
                if piotroski_grade in ['A', 'B+', 'B']:
                    st.success(f"**Piotroski Analysis: {piotroski_grade} Grade**\n\n{piotroski_interp}")
                elif piotroski_grade in ['B-', 'C']:
                    st.warning(f"**Piotroski Analysis: {piotroski_grade} Grade**\n\n{piotroski_interp}")
                else:
                    st.error(f"**Piotroski Analysis: {piotroski_grade} Grade**\n\n{piotroski_interp}")
            else:
                st.info("**Piotroski Analysis:** Data insufficient for comprehensive analysis")

def show_baldwin_indicator_analysis(show_debug=False):
    """Display Baldwin market regime indicator"""
    if not st.session_state.show_baldwin_indicator or not BALDWIN_INDICATOR_AVAILABLE:
        return
        
    with st.expander("🚦 Baldwin Market Regime Indicator", expanded=True):
        try:
            with st.spinner("Calculating Baldwin market regime..."):
                baldwin_data = calculate_baldwin_indicator_complete()
                
                if baldwin_data and 'error' not in baldwin_data:
                    display_data = format_baldwin_for_display(baldwin_data)
                    
                    if 'error' not in display_data:
                        # Main regime display
                        regime = display_data.get('overall_regime', 'UNKNOWN')
                        composite_score = display_data.get('composite_score', 50)
                        
                        st.subheader("🚦 Current Market Regime")
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        
                        with col1:
                            if regime == 'GREEN':
                                st.success(f"# 🟢 GREEN")
                                st.write("**Risk-On Mode**")
                            elif regime == 'RED':
                                st.error(f"# 🔴 RED")
                                st.write("**Risk-Off Mode**")
                            else:
                                st.warning(f"# 🟡 YELLOW")
                                st.write("**Caution Mode**")
                        
                        with col2:
                            st.metric("Composite Score", f"{composite_score:.1f}/100")
                            
                            interpretation = display_data.get('interpretation', '')
                            st.info(interpretation)
                        
                        with col3:
                            components = display_data.get('components', {})
                            if components:
                                momentum = components.get('momentum', {}).get('score', 50)
                                liquidity = components.get('liquidity', {}).get('score', 50)
                                sentiment = components.get('sentiment', {}).get('score', 50)
                                
                                st.write("**Component Scores:**")
                                st.write(f"• Momentum: {momentum:.1f}/100")
                                st.write(f"• Liquidity: {liquidity:.1f}/100")
                                st.write(f"• Sentiment: {sentiment:.1f}/100")
                        
                        # Trading recommendations
                        st.subheader("💡 Trading Recommendations")
                        recommendations = display_data.get('recommendations', [])
                        for rec in recommendations:
                            st.write(f"• {rec}")
                    else:
                        st.warning("⚠️ Baldwin indicator display data unavailable")
                else:
                    st.warning("⚠️ Baldwin indicator calculation failed")
        except Exception as e:
            if show_debug:
                st.error(f"Baldwin indicator error: {str(e)}")
                st.code(traceback.format_exc())

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis section"""
    if not st.session_state.show_market_correlation:
        return
        
    with st.expander(f"🌐 {analysis_results['symbol']} - Market Correlation Analysis", expanded=True):
        
        market_correlations = analysis_results.get('market_analysis', {}).get('market_correlations', {})
        
        if 'error' not in market_correlations and market_correlations:
            # Correlation summary
            st.subheader("Market Index Correlations")
            
            correlations = market_correlations.get('correlations', {})
            
            if correlations:
                corr_data = []
                for index, corr_value in correlations.items():
                    corr_data.append({
                        'Index': index,
                        'Correlation': f"{corr_value:.3f}",
                        'Strength': 'Strong' if abs(corr_value) > 0.7 else 'Moderate' if abs(corr_value) > 0.4 else 'Weak',
                        'Direction': 'Positive' if corr_value > 0 else 'Negative'
                    })
                
                df_corr = pd.DataFrame(corr_data)
                st.dataframe(df_corr, use_container_width=True, hide_index=True)
            
            # Breakout/Breakdown analysis
            breakout_data = market_correlations.get('breakout_breakdown', {})
            if breakout_data:
                st.subheader("📈 Breakout/Breakdown Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Breakout Candidates:**")
                    breakouts = breakout_data.get('breakouts', [])
                    if breakouts:
                        for symbol in breakouts[:5]:
                            st.write(f"• {symbol}")
                    else:
                        st.write("None identified")
                
                with col2:
                    st.write("**Breakdown Candidates:**")
                    breakdowns = breakout_data.get('breakdowns', [])
                    if breakdowns:
                        for symbol in breakdowns[:5]:
                            st.write(f"• {symbol}")
                    else:
                        st.write("None identified")
        else:
            st.warning("⚠️ Market correlation analysis not available")

def show_options_analysis(analysis_results, show_debug=False):
    """Display options analysis section"""
    if not st.session_state.show_options_analysis:
        return
        
    with st.expander(f"🎯 {analysis_results['symbol']} - Options Analysis", expanded=True):
        
        options_data = analysis_results.get('options_analysis', {})
        
        if 'error' not in options_data and options_data:
            current_price = analysis_results['current_price']
            
            # Options levels summary
            st.subheader("Options Strike Levels")
            
            levels = options_data.get('levels', {})
            
            if levels:
                options_table_data = []
                
                for level_name, level_data in levels.items():
                    put_strike = level_data.get('put_strike', 0)
                    call_strike = level_data.get('call_strike', 0)
                    
                    options_table_data.append({
                        'Level': level_name,
                        'Put Strike': f"${put_strike:.2f}",
                        'Put Distance': f"{((current_price - put_strike) / current_price * 100):.2f}%",
                        'Call Strike': f"${call_strike:.2f}",
                        'Call Distance': f"{((call_strike - current_price) / current_price * 100):.2f}%"
                    })
                
                df_options = pd.DataFrame(options_table_data)
                st.dataframe(df_options, use_container_width=True, hide_index=True)
            
            # Greeks summary
            greeks = options_data.get('greeks', {})
            if greeks:
                st.subheader("Options Greeks")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Delta", f"{greeks.get('delta', 0):.3f}")
                with col2:
                    st.metric("Gamma", f"{greeks.get('gamma', 0):.4f}")
                with col3:
                    st.metric("Theta", f"{greeks.get('theta', 0):.4f}")
                with col4:
                    st.metric("Vega", f"{greeks.get('vega', 0):.4f}")
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
            
            # Confidence intervals table
            intervals_data = []
            for level, level_data in confidence_analysis['confidence_intervals'].items():
                intervals_data.append({
                    'Confidence Level': level,
                    'Upper Bound': f"${level_data['upper_bound']}",
                    'Lower Bound': f"${level_data['lower_bound']}",
                    'Expected Move': f"±{level_data['expected_move_pct']:.2f}%"
                })
            
            df_intervals = pd.DataFrame(intervals_data)
            st.dataframe(df_intervals, use_container_width=True, hide_index=True)

def perform_enhanced_analysis(symbol, period, show_debug=False):
    """Perform enhanced analysis using modular components"""
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
        
        # Step 4: Calculate enhanced indicators
        daily_vwap = calculate_daily_vwap(analysis_input)
        fibonacci_emas = calculate_fibonacci_emas(analysis_input)
        point_of_control = calculate_point_of_control_enhanced(analysis_input)
        weekly_deviations = calculate_weekly_deviations(analysis_input)
        comprehensive_technicals = calculate_comprehensive_technicals(analysis_input)
        
        # Step 5: Calculate Volume Analysis (if available)
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
        
        # Step 6: Calculate Volatility Analysis (if available)
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
        market_correlations = calculate_market_correlations_enhanced(analysis_input, symbol, show_debug)
        
        # Step 8: Calculate options levels
        options_levels = calculate_options_levels_enhanced(analysis_input, symbol)
        
        # Step 9: Calculate fundamental scores
        graham_score = calculate_graham_score(symbol)
        piotroski_score = calculate_piotroski_score(symbol)
        
        # Step 10: Calculate confidence intervals
        confidence_analysis = calculate_confidence_intervals(analysis_input, analysis_input['Close'].iloc[-1])
        
        # Compile results
        analysis_results = {
            'symbol': symbol,
            'current_price': float(analysis_input['Close'].iloc[-1]),
            'enhanced_indicators': {
                'daily_vwap': daily_vwap,
                'fibonacci_emas': fibonacci_emas,
                'point_of_control': point_of_control,
                'weekly_deviations': weekly_deviations,
                'comprehensive_technicals': comprehensive_technicals,
                'volume_analysis': volume_analysis,
                'volatility_analysis': volatility_analysis
            },
            'market_analysis': {
                'market_correlations': market_correlations
            },
            'options_analysis': options_levels,
            'fundamental_analysis': {
                'graham_score': graham_score,
                'piotroski_score': piotroski_score
            },
            'confidence_analysis': confidence_analysis,
            'system_status': 'OPERATIONAL v4.2.2'
        }
        
        # Store results
        data_manager.store_analysis_results(symbol, analysis_results)
        
        # Get chart data
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
    
    if controls['analyze_button'] and controls['symbol']:
        add_to_recently_viewed(controls['symbol'])
        
        st.write("## 📊 VWV Trading Analysis v4.2.2")
        
        with st.spinner(f"Analyzing {controls['symbol']}..."):
            analysis_results, chart_data = perform_enhanced_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results and chart_data is not None:
                # MANDATORY DISPLAY ORDER
                # 1. Charts First
                show_interactive_charts(chart_data, analysis_results, controls['show_debug'])
                
                # 2. Individual Technical Analysis
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                
                # 3. Volume Analysis (Optional)
                if VOLUME_ANALYSIS_AVAILABLE:
                    show_volume_analysis(analysis_results, controls['show_debug'])
                
                # 4. Volatility Analysis (Optional)
                if VOLATILITY_ANALYSIS_AVAILABLE:
                    show_volatility_analysis(analysis_results, controls['show_debug'])
                
                # 5. Fundamental Analysis
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                
                # 6. Baldwin Indicator (Before Market Correlation)
                if BALDWIN_INDICATOR_AVAILABLE:
                    show_baldwin_indicator_analysis(show_debug=controls['show_debug'])
                
                # 7. Market Correlation Analysis
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                
                # 8. Options Analysis
                show_options_analysis(analysis_results, controls['show_debug'])
                
                # 9. Confidence Intervals
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
                # Debug information
                if controls['show_debug']:
                    with st.expander("🐛 Debug Information", expanded=False):
                        st.write("### Analysis Results Structure")
                        st.json(analysis_results)
                        
                        st.write("### Data Manager Summary")
                        data_manager = get_data_manager()
                        summary = data_manager.get_data_summary()
                        st.json(summary)
                        
                        st.write("### System Status")
                        st.write(f"**Volume Analysis Available:** {VOLUME_ANALYSIS_AVAILABLE}")
                        st.write(f"**Volatility Analysis Available:** {VOLATILITY_ANALYSIS_AVAILABLE}")
                        st.write(f"**Baldwin Indicator Available:** {BALDWIN_INDICATOR_AVAILABLE}")
            else:
                st.error("❌ No results to display")
    else:
        st.write("## 🚀 VWV Professional Trading System v4.2.2")
        st.write("Enter a symbol in the sidebar and click 'Analyze Symbol' to begin.")
        
        # Market status
        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")
        
        # Display order information
        with st.expander("✅ Analysis Display Order", expanded=True):
            st.write("**MANDATORY SEQUENCE:**")
            st.write("1. **📊 Interactive Charts** - Comprehensive trading visualization")
            st.write("2. **📊 Technical Analysis** - Professional score bar with Fibonacci EMAs")
            st.write("3. **📊 Volume Analysis** - Optional when module available")
            st.write("4. **📊 Volatility Analysis** - Optional when module available")
            st.write("5. **📊 Fundamental Analysis** - Graham & Piotroski scores")
            st.write("6. **🚦 Baldwin Market Regime** - Market-wide analysis")
            st.write("7. **🌐 Market Correlation** - ETF correlation & breakout analysis")
            st.write("8. **🎯 Options Analysis** - Strike levels with Greeks")
            st.write("9. **📊 Confidence Intervals** - Statistical projections")
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.write("1. **Enter a symbol** in the sidebar (e.g., AAPL, SPY, QQQ)")
            st.write("2. **Select time period** - default is 1 month")
            st.write("3. **Click 'Analyze Symbol'** or press Enter")
            st.write("4. **Use Quick Links** for instant analysis of popular symbols")
            st.write("5. **Toggle sections** in Analysis Sections panel")
            st.write("6. **Enable Debug Mode** to see detailed system information")

    # Footer
    st.markdown("---")
    st.write("### 📊 System Information v4.2.2")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Version:** VWV Professional v4.2.2")
        st.write(f"**Status:** ✅ All Systems Operational")
    with col2:
        st.write(f"**Modules:** Technical, Fundamental, Market, Options")
        st.write(f"**Optional:** Volume, Volatility, Baldwin")
    with col3:
        st.write(f"**Architecture:** Modular & Extensible")
        st.write(f"**Display:** Optimized Order & Professional UI")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.write("Please refresh the page and try again.")
        
        if st.checkbox("Show Error Details"):
            st.exception(e)
