"""
VWV Professional Trading System - Complete Version with Interactive Charts
Main application with enhanced indicators, charts, and collapsible screener
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    page_title="VWV Professional Trading System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Screener Configuration - UPDATED 4-TIER SYSTEM
SCREENER_CONFIG = {
    'refresh_minutes': 15,  # Recommended: 15 minutes for good balance
    'very_bearish_threshold': 20,     # 0-20: Very Bearish
    'bearish_threshold': 30,          # 20-30: Bearish  
    'bullish_threshold': 70,          # 70-80: Bullish
    'very_bullish_threshold': 80,     # 80-100: Very Bullish
    'max_symbols_per_scan': 20,  # Limit to avoid timeouts
    'timeout_per_symbol': 10  # seconds
}

def create_interactive_chart(data: pd.DataFrame, analysis_results: dict, symbol: str) -> go.Figure:
    """
    Create comprehensive interactive chart with all technical indicators
    Optimized for Streamlit Cloud free tier
    """
    try:
        # Create subplots: Main chart + Volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} - Price & Technical Analysis', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # 1. CANDLESTICK CHART
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff3366',
                increasing_fillcolor='#00ff88',
                decreasing_fillcolor='#ff3366'
            ),
            row=1, col=1
        )
        
        # 2. FIBONACCI EMAs
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
        
        # EMA colors and styling
        ema_colors = {
            'EMA_21': '#FFD700',   # Gold
            'EMA_55': '#FF6B6B',   # Red
            'EMA_89': '#4ECDC4',   # Teal
            'EMA_144': '#45B7D1',  # Blue
            'EMA_233': '#FFA07A'   # Light Salmon
        }
        
        # Calculate EMAs for the chart
        for ema_period in [21, 55, 89, 144, 233]:
            if len(data) >= ema_period:
                ema_values = data['Close'].ewm(span=ema_period).mean()
                ema_name = f'EMA_{ema_period}'
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=ema_values,
                        mode='lines',
                        name=f'EMA {ema_period}',
                        line=dict(
                            color=ema_colors.get(ema_name, '#CCCCCC'),
                            width=2
                        ),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
        
        # 3. VWAP LINE
        daily_vwap = enhanced_indicators.get('daily_vwap', 0)
        if daily_vwap > 0:
            fig.add_hline(
                y=daily_vwap,
                line_dash="dash",
                line_color="#FFF700",
                annotation_text=f"VWAP: ${daily_vwap:.2f}",
                annotation_position="bottom right",
                row=1, col=1
            )
        
        # 4. POINT OF CONTROL
        point_of_control = enhanced_indicators.get('point_of_control', 0)
        if point_of_control > 0:
            fig.add_hline(
                y=point_of_control,
                line_dash="dot",
                line_color="#FF69B4",
                annotation_text=f"POC: ${point_of_control:.2f}",
                annotation_position="top right",
                row=1, col=1
            )
        
        # 5. SUPPORT AND RESISTANCE LEVELS
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        
        # Previous week high/low
        prev_week_high = comprehensive_technicals.get('prev_week_high', 0)
        prev_week_low = comprehensive_technicals.get('prev_week_low', 0)
        
        if prev_week_high > 0:
            fig.add_hline(
                y=prev_week_high,
                line_dash="dash",
                line_color="#FF4500",
                annotation_text=f"Week High: ${prev_week_high:.2f}",
                annotation_position="top left",
                opacity=0.7,
                row=1, col=1
            )
        
        if prev_week_low > 0:
            fig.add_hline(
                y=prev_week_low,
                line_dash="dash",
                line_color="#32CD32",
                annotation_text=f"Week Low: ${prev_week_low:.2f}",
                annotation_position="bottom left",
                opacity=0.7,
                row=1, col=1
            )
        
        # 6. VOLUME CHART
        volume_colors = [
            '#00ff88' if close >= open else '#ff3366'
            for close, open in zip(data['Close'], data['Open'])
        ]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=volume_colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Add volume moving average
        if len(data) >= 20:
            volume_ma = data['Volume'].rolling(20).mean()
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=volume_ma,
                    mode='lines',
                    name='Volume MA(20)',
                    line=dict(color='#FFA500', width=2),
                    opacity=0.8
                ),
                row=2, col=1
            )
        
        # 7. CURRENT PRICE MARKER
        current_price = analysis_results.get('current_price', data['Close'].iloc[-1])
        current_date = data.index[-1]
        
        fig.add_trace(
            go.Scatter(
                x=[current_date],
                y=[current_price],
                mode='markers',
                name='Current Price',
                marker=dict(
                    size=12,
                    color='#FFFF00',
                    symbol='diamond',
                    line=dict(width=2, color='#000000')
                )
            ),
            row=1, col=1
        )
        
        # 8. CHART LAYOUT AND STYLING
        fig.update_layout(
            title=dict(
                text=f"{symbol} - Professional Trading Analysis",
                font=dict(size=20, color='white'),
                x=0.5
            ),
            template='plotly_dark',
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01,
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1
            ),
            margin=dict(l=0, r=100, t=50, b=0),
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(
            title_text="Date",
            gridcolor='rgba(128,128,128,0.2)',
            row=2, col=1
        )
        
        fig.update_yaxes(
            title_text="Price ($)",
            gridcolor='rgba(128,128,128,0.2)',
            row=1, col=1
        )
        
        fig.update_yaxes(
            title_text="Volume",
            gridcolor='rgba(128,128,128,0.2)',
            row=2, col=1
        )
        
        # Remove rangeslider for cleaner look
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return create_simple_fallback_chart(data, symbol)

def create_simple_fallback_chart(data: pd.DataFrame, symbol: str) -> go.Figure:
    """
    Fallback chart in case main chart creation fails
    """
    fig = go.Figure()
    
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=symbol
        )
    )
    
    fig.update_layout(
        title=f"{symbol} - Basic Chart",
        template='plotly_dark',
        height=600
    )
    
    return fig

def display_interactive_charts(analysis_results: dict, market_data: pd.DataFrame):
    """
    Display all interactive charts in the Streamlit app
    """
    symbol = analysis_results.get('symbol', 'Unknown')
    
    # Main price chart
    with st.container():
        st.subheader("📈 Interactive Price Chart")
        
        with st.spinner("Generating interactive chart..."):
            main_chart = create_interactive_chart(market_data, analysis_results, symbol)
            st.plotly_chart(main_chart, use_container_width=True, config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
            })

@st.cache_data(ttl=SCREENER_CONFIG['refresh_minutes'] * 60, show_spinner=False)
def scan_extreme_technical_scores(analysis_period='1y'):
    """Scan quick links symbols for extreme technical scores with caching"""
    start_time = time.time()
    scores_by_category = {
        'very_bearish': [],  # 0-20
        'bearish': [],       # 20-30
        'bullish': [],       # 70-80
        'very_bullish': []   # 80-100
    }
    scanned_count = 0
    
    # Get all symbols from quick links (limit to prevent timeouts)
    all_symbols = []
    for category, symbols in QUICK_LINK_CATEGORIES.items():
        all_symbols.extend(symbols)
    
    # Remove duplicates and limit
    unique_symbols = list(set(all_symbols))[:SCREENER_CONFIG['max_symbols_per_scan']]
    
    for symbol in unique_symbols:
        if time.time() - start_time > 120:  # 2-minute total timeout
            break
            
        try:
            # Quick analysis for screening - USE SAME PERIOD AS INDIVIDUAL ANALYSIS
            market_data = get_market_data_enhanced(symbol, period=analysis_period, show_debug=False)
            
            if market_data is None or len(market_data) < 50:
                continue
                
            # Calculate minimal indicators needed for composite score
            daily_vwap = calculate_daily_vwap(market_data)
            fibonacci_emas = calculate_fibonacci_emas(market_data)
            point_of_control = calculate_point_of_control_enhanced(market_data)
            comprehensive_technicals = calculate_comprehensive_technicals(market_data)
            
            current_price = round(float(market_data['Close'].iloc[-1]), 2)
            
            # Build minimal analysis for composite score
            analysis_for_score = {
                'symbol': symbol,
                'current_price': current_price,
                'enhanced_indicators': {
                    'daily_vwap': daily_vwap,
                    'fibonacci_emas': fibonacci_emas,
                    'point_of_control': point_of_control,
                    'comprehensive_technicals': comprehensive_technicals
                }
            }
            
            # Calculate composite score
            composite_score, score_details = calculate_composite_technical_score(analysis_for_score)
            
            # Categorize scores into 4 tiers
            score_data = {
                'symbol': symbol,
                'score': composite_score,
                'price': current_price,
                'description': SYMBOL_DESCRIPTIONS.get(symbol, f"{symbol} - Financial Symbol")[:50] + "..."
            }
            
            if composite_score <= SCREENER_CONFIG['very_bearish_threshold']:
                scores_by_category['very_bearish'].append(score_data)
            elif composite_score <= SCREENER_CONFIG['bearish_threshold']:
                scores_by_category['bearish'].append(score_data)
            elif composite_score >= SCREENER_CONFIG['very_bullish_threshold']:
                scores_by_category['very_bullish'].append(score_data)
            elif composite_score >= SCREENER_CONFIG['bullish_threshold']:
                scores_by_category['bullish'].append(score_data)
                
            scanned_count += 1
            
        except Exception as e:
            continue  # Skip problematic symbols
    
    # Sort by score (most extreme first)
    scores_by_category['very_bearish'].sort(key=lambda x: x['score'])
    scores_by_category['bearish'].sort(key=lambda x: x['score'])
    scores_by_category['bullish'].sort(key=lambda x: x['score'], reverse=True)
    scores_by_category['very_bullish'].sort(key=lambda x: x['score'], reverse=True)
    
    scan_time = time.time() - start_time
    
    return {
        'results': scores_by_category,
        'scan_time': round(scan_time, 1),
        'scanned_count': scanned_count,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    }

def create_technical_chart(market_data, analysis_results):
    """Create comprehensive technical analysis chart with indicators"""
    try:
        if market_data is None or len(market_data) < 50:
            return None
            
        # Create subplots with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{analysis_results["symbol"]} - Price & Technical Indicators', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=market_data.index,
                open=market_data['Open'],
                high=market_data['High'],
                low=market_data['Low'],
                close=market_data['Close'],
                name='Price',
                increasing_line_color='#00A86B',
                decreasing_line_color='#DC143C'
            ),
            row=1, col=1
        )
        
        # Add technical indicators
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        
        # EMAs
        fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
        ema_colors = {'EMA_21': '#FF6B6B', 'EMA_55': '#4ECDC4', 'EMA_89': '#45B7D1', 'EMA_144': '#96CEB4', 'EMA_233': '#FFEAA7'}
        
        for ema_name, ema_value in fibonacci_emas.items():
            period = ema_name.split('_')[1]
            if len(market_data) >= int(period):
                ema_series = market_data['Close'].ewm(span=int(period)).mean()
                fig.add_trace(
                    go.Scatter(
                        x=market_data.index,
                        y=ema_series,
                        mode='lines',
                        name=f'EMA {period}',
                        line=dict(color=ema_colors.get(ema_name, '#95A5A6'), width=1.5),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
        
        # VWAP
        daily_vwap = enhanced_indicators.get('daily_vwap', 0)
        if daily_vwap > 0:
            fig.add_hline(
                y=daily_vwap,
                line_dash="dash",
                line_color="#E74C3C",
                annotation_text=f"VWAP: ${daily_vwap:.2f}",
                row=1, col=1
            )
        
        # Point of Control
        poc = enhanced_indicators.get('point_of_control', 0)
        if poc > 0:
            fig.add_hline(
                y=poc,
                line_dash="dot",
                line_color="#9B59B6",
                annotation_text=f"POC: ${poc:.2f}",
                row=1, col=1
            )
        
        # Previous Week High/Low
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        prev_week_high = comprehensive_technicals.get('prev_week_high', 0)
        prev_week_low = comprehensive_technicals.get('prev_week_low', 0)
        
        if prev_week_high > 0:
            fig.add_hline(
                y=prev_week_high,
                line_dash="dashdot",
                line_color="#27AE60",
                annotation_text=f"Prev Week High: ${prev_week_high:.2f}",
                row=1, col=1
            )
        
        if prev_week_low > 0:
            fig.add_hline(
                y=prev_week_low,
                line_dash="dashdot", 
                line_color="#E67E22",
                annotation_text=f"Prev Week Low: ${prev_week_low:.2f}",
                row=1, col=1
            )
        
        # Volume
        colors = ['#00A86B' if close >= open else '#DC143C' 
                 for close, open in zip(market_data['Close'], market_data['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=market_data.index,
                y=market_data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"{analysis_results['symbol']} - Technical Analysis Chart",
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            template='plotly_dark',
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Chart creation error: {e}")
        return None

def show_technical_screener(analysis_period='1y'):
    """Display the technical score screener section - NOW COLLAPSIBLE"""
    if not st.session_state.show_technical_screener:
        return
        
    with st.expander("🎯 Technical Score Screener", expanded=True):
        st.write(f"**Score Categories**: 0-{SCREENER_CONFIG['very_bearish_threshold']} (Very Bearish) | {SCREENER_CONFIG['very_bearish_threshold']}-{SCREENER_CONFIG['bearish_threshold']} (Bearish) | {SCREENER_CONFIG['bullish_threshold']}-{SCREENER_CONFIG['very_bullish_threshold']} (Bullish) | {SCREENER_CONFIG['very_bullish_threshold']}-100 (Very Bullish) | Auto-refresh: {SCREENER_CONFIG['refresh_minutes']} min")
        
        # Add manual refresh button and summary info
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔄 Refresh Now", help="Manually refresh screener results"):
                st.cache_data.clear()
                st.rerun()
        
        # Get cached results with consolidated spinner
        with st.spinner("Scanning symbols for technical scores..."):
            screen_results = scan_extreme_technical_scores(analysis_period)
        
        with col2:
            st.write(f"**Last Scan:** {screen_results['timestamp']}")
        with col3:
            st.write(f"**Scanned:** {screen_results['scanned_count']} symbols in {screen_results['scan_time']}s")
        
        results = screen_results['results']
        
        # Consolidated data quality summary
        total_signals = sum(len(category) for category in results.values())
        if total_signals > 0:
            st.success(f"✅ **Data Quality:** All symbols loaded successfully | **Signals Found:** {total_signals} total")
        else:
            st.info("ℹ️ **Status:** All symbols in normal range (30-70) - No extreme signals detected")
        
        # Display results in four columns with new categories
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write("#### 🔴 Very Bearish (0-20)")
            if results['very_bearish']:
                very_bearish_data = []
                for item in results['very_bearish']:
                    very_bearish_data.append({
                        'Symbol': item['symbol'],
                        'Score': f"{item['score']:.1f}",
                        'Price': f"${item['price']:.2f}"
                    })
                
                df_very_bearish = pd.DataFrame(very_bearish_data)
                st.dataframe(df_very_bearish, use_container_width=True, hide_index=True)
                
                # Quick analysis buttons - AUTO ANALYZE
                for idx, item in enumerate(results['very_bearish'][:2]):
                    if st.button(f"📊 {item['symbol']}", key=f"very_bearish_{item['symbol']}", use_container_width=True):
                        st.session_state.selected_symbol = item['symbol']
                        st.session_state.auto_analyze = True
                        st.rerun()
            else:
                st.info("No very bearish signals")
        
        with col2:
            st.write("#### 🟠 Bearish (20-30)")
            if results['bearish']:
                bearish_data = []
                for item in results['bearish']:
                    bearish_data.append({
                        'Symbol': item['symbol'],
                        'Score': f"{item['score']:.1f}",
                        'Price': f"${item['price']:.2f}"
                    })
                
                df_bearish = pd.DataFrame(bearish_data)
                st.dataframe(df_bearish, use_container_width=True, hide_index=True)
                
                # Quick analysis buttons - AUTO ANALYZE
                for idx, item in enumerate(results['bearish'][:2]):
                    if st.button(f"📊 {item['symbol']}", key=f"bearish_{item['symbol']}", use_container_width=True):
                        st.session_state.selected_symbol = item['symbol']
                        st.session_state.auto_analyze = True
                        st.rerun()
            else:
                st.info("No bearish signals")
        
        with col3:
            st.write("#### 🟡 Bullish (70-80)")
            if results['bullish']:
                bullish_data = []
                for item in results['bullish']:
                    bullish_data.append({
                        'Symbol': item['symbol'],
                        'Score': f"{item['score']:.1f}",
                        'Price': f"${item['price']:.2f}"
                    })
                
                df_bullish = pd.DataFrame(bullish_data)
                st.dataframe(df_bullish, use_container_width=True, hide_index=True)
                
                # Quick analysis buttons - AUTO ANALYZE
                for idx, item in enumerate(results['bullish'][:2]):
                    if st.button(f"📊 {item['symbol']}", key=f"bullish_{item['symbol']}", use_container_width=True):
                        st.session_state.selected_symbol = item['symbol']
                        st.session_state.auto_analyze = True
                        st.rerun()
            else:
                st.info("No bullish signals")
        
        with col4:
            st.write("#### 🟢 Very Bullish (80-100)")
            if results['very_bullish']:
                very_bullish_data = []
                for item in results['very_bullish']:
                    very_bullish_data.append({
                        'Symbol': item['symbol'],
                        'Score': f"{item['score']:.1f}",
                        'Price': f"${item['price']:.2f}"
                    })
                
                df_very_bullish = pd.DataFrame(very_bullish_data)
                st.dataframe(df_very_bullish, use_container_width=True, hide_index=True)
                
                # Quick analysis buttons - AUTO ANALYZE
                for idx, item in enumerate(results['very_bullish'][:2]):
                    if st.button(f"📊 {item['symbol']}", key=f"very_bullish_{item['symbol']}", use_container_width=True):
                        st.session_state.selected_symbol = item['symbol']
                        st.session_state.auto_analyze = True
                        st.rerun()
            else:
                st.info("No very bullish signals")
        
        # Enhanced summary stats
        total_extreme = len(results['very_bearish']) + len(results['bearish']) + len(results['bullish']) + len(results['very_bullish'])
        if total_extreme > 0:
            summary_text = f"**Summary:** {total_extreme} signals detected: "
            signal_parts = []
            if results['very_bearish']:
                signal_parts.append(f"{len(results['very_bearish'])} very bearish")
            if results['bearish']:
                signal_parts.append(f"{len(results['bearish'])} bearish")
            if results['bullish']:
                signal_parts.append(f"{len(results['bullish'])} bullish")
            if results['very_bullish']:
                signal_parts.append(f"{len(results['very_bullish'])} very bullish")
            
            summary_text += ", ".join(signal_parts)
            st.write(summary_text)
            
            # Show configuration
            with st.expander("⚙️ Screener Configuration", expanded=False):
                st.write(f"• **Refresh Frequency:** {SCREENER_CONFIG['refresh_minutes']} minutes")
                st.write(f"• **Very Bearish:** 0-{SCREENER_CONFIG['very_bearish_threshold']}")
                st.write(f"• **Bearish:** {SCREENER_CONFIG['very_bearish_threshold']}-{SCREENER_CONFIG['bearish_threshold']}")
                st.write(f"• **Bullish:** {SCREENER_CONFIG['bullish_threshold']}-{SCREENER_CONFIG['very_bullish_threshold']}")
                st.write(f"• **Very Bullish:** {SCREENER_CONFIG['very_bullish_threshold']}-100")
                st.write(f"• **Max Symbols per Scan:** {SCREENER_CONFIG['max_symbols_per_scan']}")
                st.write(f"• **Data Period:** {analysis_period} (matches individual analysis)")

def initialize_session_state():
    """Initialize session state variables"""
    if 'recently_viewed' not in st.session_state:
        st.session_state.recently_viewed = []
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
    if 'show_technical_screener' not in st.session_state:  # NEW TOGGLE
        st.session_state.show_technical_screener = True
    if 'show_interactive_charts' not in st.session_state:
        st.session_state.show_interactive_charts = True
    if 'current_symbol' not in st.session_state:
        st.session_state.current_symbol = UI_SETTINGS['default_symbol']
    if 'auto_analyze' not in st.session_state:
        st.session_state.auto_analyze = False
    if 'previous_symbol' not in st.session_state:
        st.session_state.previous_symbol = UI_SETTINGS['default_symbol']

def create_sidebar_controls():
    """Create sidebar controls and return analysis parameters"""
    st.sidebar.title("📊 Trading Analysis")
    
    # Initialize session state
    initialize_session_state()
    
    # Better symbol handling - check for quick link selection first
    if 'selected_symbol' in st.session_state:
        st.session_state.current_symbol = st.session_state.selected_symbol
        del st.session_state.selected_symbol  # Clear the trigger
        
    symbol = st.sidebar.text_input(
        "Symbol", 
        value=st.session_state.current_symbol,
        key="symbol_input",
        help="Enter stock symbol and press Enter to analyze"
    ).upper()
    
    # ENTER KEY BEHAVIOR: Check if symbol changed to trigger auto-analysis
    symbol_changed = False
    if symbol != st.session_state.current_symbol:
        st.session_state.current_symbol = symbol
        symbol_changed = True
        if symbol and symbol != st.session_state.previous_symbol:
            st.session_state.auto_analyze = True
            st.session_state.previous_symbol = symbol
    
    period = st.sidebar.selectbox("Data Period", UI_SETTINGS['periods'], index=3)
    
    # Section Control Panel
    with st.sidebar.expander("📋 Analysis Sections", expanded=False):
        st.write("**Toggle Analysis Sections:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.show_interactive_charts = st.checkbox(
                "Interactive Charts", 
                value=st.session_state.show_interactive_charts,
                key="toggle_charts"
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
                            # AUTO ANALYZE FOR RECENTLY VIEWED
                            if st.button(f"{recent_symbol}", key=f"recent_{recent_symbol}_{symbol_idx}", use_container_width=True):
                                st.session_state.selected_symbol = recent_symbol
                                st.session_state.auto_analyze = True
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
                                # AUTO ANALYZE FOR QUICK LINKS
                                if st.button(sym, help=SYMBOL_DESCRIPTIONS.get(sym, f"{sym} - Financial Symbol"), key=f"quick_link_{sym}", use_container_width=True):
                                    st.session_state.selected_symbol = sym
                                    st.session_state.auto_analyze = True
                                    st.rerun()

    # Debug toggle
    show_debug = st.sidebar.checkbox("🐛 Show Debug Info", value=False)
    
    return {
        'symbol': st.session_state.current_symbol,
        'period': period,
        'analyze_button': analyze_button,
        'show_debug': show_debug,
        'auto_analyze': st.session_state.auto_analyze,
        'symbol_changed': symbol_changed
    }

def add_to_recently_viewed(symbol):
    """Add symbol to recently viewed - updated for 9 symbols"""
    if symbol and symbol != "":
        if symbol in st.session_state.recently_viewed:
            st.session_state.recently_viewed.remove(symbol)
        st.session_state.recently_viewed.insert(0, symbol)
        st.session_state.recently_viewed = st.session_state.recently_viewed[:9]

def show_interactive_charts_section(analysis_results, show_debug=False):
    """Display interactive charts section - NEW FEATURE"""
    if not st.session_state.get('show_interactive_charts', True):
        return
        
    # Get market data for charts
    data_manager = get_data_manager()
    symbol = analysis_results['symbol']
    chart_data = data_manager.get_market_data_for_chart(symbol)
    
    if chart_data is not None and len(chart_data) > 0:
        with st.expander(f"📈 {symbol} - Interactive Charts & Visualization", expanded=True):
            try:
                # Display the interactive charts
                display_interactive_charts(analysis_results, chart_data)
                
                # Chart information panel
                with st.container():
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.info("""
                        **📈 Price Chart Features:**
                        • Candlestick price data
                        • Fibonacci EMAs (21, 55, 89, 144, 233)
                        • VWAP & Point of Control levels
                        • Support/Resistance levels
                        • Volume analysis
                        """)
                    
                    with col2:
                        st.info("""
                        **📊 Volume Analysis:**
                        • Color-coded volume bars
                        • 20-period volume moving average
                        • Volume relationship to price
                        • Intraday volume patterns
                        """)
                    
                    with col3:
                        st.info("""
                        **🎯 Interactive Features:**
                        • Zoom and pan functionality
                        • Hover for detailed data
                        • Download chart as PNG
                        • Mobile-responsive design
                        """)
                
                if show_debug:
                    st.write(f"📊 Chart data points: {len(chart_data)}")
                    st.write(f"📅 Date range: {chart_data.index[0]} to {chart_data.index[-1]}")
                
            except Exception as e:
                st.error(f"❌ Error displaying charts: {str(e)}")
                if show_debug:
                    st.exception(e)
    else:
        st.warning("⚠️ Chart data not available for visualization")

def show_individual_technical_analysis(analysis_results, show_debug=False):
    """Display individual technical analysis section with ENHANCED indicators"""
    if not st.session_state.show_technical_analysis:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Individual Technical Analysis", expanded=True):
        
        # COMPOSITE TECHNICAL SCORE - Use modular component
        composite_score, score_details = calculate_composite_technical_score(analysis_results)
        score_bar_html = create_technical_score_bar(composite_score, score_details)
        
        # Use components directly instead of markdown for better rendering
        st.components.v1.html(score_bar_html, height=200)
        
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
        
        # ENHANCED CHART SECTION
        data_manager = get_data_manager()
        chart_data = data_manager.get_market_data_for_chart(analysis_results['symbol'])
        
        if chart_data is not None:
            st.subheader("📈 Technical Analysis Chart")
            chart_fig = create_technical_chart(chart_data, analysis_results)
            if chart_fig:
                st.plotly_chart(chart_fig, use_container_width=True)
        
        # ENHANCED Technical indicators table - ALL EMAs
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
        
        # ALL Fibonacci EMAs (21, 55, 89, 144, 233)
        for ema_name, ema_value in fibonacci_emas.items():
            period = ema_name.split('_')[1]
            distance_pct = f"{((current_price - ema_value) / ema_value * 100):+.2f}%" if ema_value > 0 else "N/A"
            status = "Above" if current_price > ema_value else "Below"
            indicators_data.append((f"EMA {period}", f"${ema_value:.2f}", "📈 Trend", distance_pct, status))
        
        # ENHANCED - Previous Week High/Low
        prev_week_high = comprehensive_technicals.get('prev_week_high', 0)
        prev_week_low = comprehensive_technicals.get('prev_week_low', 0)
        if prev_week_high > 0:
            distance_pct = f"{((current_price - prev_week_high) / prev_week_high * 100):+.2f}%"
            status = "Above" if current_price > prev_week_high else "Below"
            indicators_data.append(("Prev Week High", f"${prev_week_high:.2f}", "📅 Weekly", distance_pct, status))
        
        if prev_week_low > 0:
            distance_pct = f"{((current_price - prev_week_low) / prev_week_low * 100):+.2f}%"
            status = "Above" if current_price > prev_week_low else "Below"
            indicators_data.append(("Prev Week Low", f"${prev_week_low:.2f}", "📅 Weekly", distance_pct, status))
        
        df_technical = pd.DataFrame(indicators_data, columns=['Indicator', 'Value', 'Type', 'Distance %', 'Status'])
        st.dataframe(df_technical, use_container_width=True, hide_index=True)

def show_fundamental_analysis(analysis_results, show_debug=False):
    """Display ENHANCED fundamental analysis section"""
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
            
            # ENHANCED - Detailed Criteria Breakdown
            if 'error' not in graham_data and graham_data.get('criteria'):
                with st.expander("📋 Graham Score Detailed Criteria", expanded=False):
                    for criterion in graham_data['criteria']:
                        if '✅' in criterion:
                            st.success(criterion)
                        else:
                            st.error(criterion)
            
            if 'error' not in piotroski_data and piotroski_data.get('criteria'):
                with st.expander("📋 Piotroski F-Score Detailed Criteria", expanded=False):
                    for criterion in piotroski_data['criteria']:
                        if '✅' in criterion:
                            st.success(criterion)
                        else:
                            st.error(criterion)
        
        elif is_etf_symbol:
            st.info(f"ℹ️ **{analysis_results['symbol']} is an ETF** - Fundamental analysis is not applicable to Exchange-Traded Funds.")

def show_market_correlation_analysis(analysis_results, show_debug=False):
    """Display market correlation analysis section"""
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
        
        # Step 6: Calculate fundamental analysis (skip for ETFs)
        is_etf_symbol = is_etf(symbol)
        
        if is_etf_symbol:
            graham_score = {'score': 0, 'total_possible': 10, 'criteria': [], 'error': 'ETF - Fundamental analysis not applicable'}
            piotroski_score = {'score': 0, 'total_possible': 9, 'criteria': [], 'error': 'ETF - Fundamental analysis not applicable'}
        else:
            graham_score = calculate_graham_score(symbol, show_debug)
            piotroski_score = calculate_piotroski_score(symbol, show_debug)
        
        # Step 7: Calculate options levels
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
        
        # Step 8: Calculate confidence intervals
        confidence_analysis = calculate_confidence_intervals(analysis_input)
        
        # Step 9: Build analysis results
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
            'confidence_analysis': confidence_analysis,
            'system_status': 'OPERATIONAL'
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
    
    # Show current symbol being analyzed
    if controls['symbol']:
        st.write(f"## 📊 VWV Trading Analysis - {controls['symbol']}")
    
    # AUTO ANALYSIS LOGIC: Run analysis if button clicked, auto_analyze flag set, or symbol changed via Enter
    should_analyze = (
        (controls['analyze_button'] and controls['symbol']) or 
        (controls['auto_analyze'] and controls['symbol']) or
        (controls['symbol_changed'] and controls['symbol'] and len(controls['symbol']) > 0)
    )
    
    if should_analyze:
        # Clear auto_analyze flag
        st.session_state.auto_analyze = False
        
        # Add symbol to recently viewed
        add_to_recently_viewed(controls['symbol'])
        
        with st.spinner(f"Analyzing {controls['symbol']}..."):
            
            # Perform analysis using modular components
            analysis_results = perform_enhanced_analysis(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results:
                # Show all analysis sections - UPDATED ORDER
                show_interactive_charts_section(analysis_results, controls['show_debug'])  # NEW - FIRST!
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                show_options_analysis(analysis_results, controls['show_debug'])
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
                # Debug information
                if controls['show_debug']:
                    st.markdown("---")
                    with st.expander("🐛 Debug Information", expanded=False):
                        st.write("### Analysis Results Structure")
                        st.json(analysis_results, expanded=False)
                        
                        st.write("### Data Manager Summary")
                        data_manager = get_data_manager()
                        summary = data_manager.get_data_summary()
                        st.json(summary)
                        
                        st.write("### Current Session State")
                        st.json({
                            'current_symbol': st.session_state.get('current_symbol', 'Not Set'),
                            'recently_viewed': st.session_state.get('recently_viewed', []),
                            'selected_symbol': st.session_state.get('selected_symbol', 'Not Set'),
                            'auto_analyze': st.session_state.get('auto_analyze', False),
                            'previous_symbol': st.session_state.get('previous_symbol', 'Not Set')
                        })
    
    else:
        # Welcome message
        st.write("## 🚀 VWV Professional Trading System - Complete Enhanced Architecture")
        st.write("**All modules active:** Technical + **Chart**, Fundamental + **Detailed Criteria**, Market, Options, UI Components, **Collapsible 4-Tier Screener**")
        
        # Always show screener on home page (if enabled)
        if st.session_state.show_technical_screener:
            st.markdown("---")
            show_technical_screener('1y')  # Default to 1 year on home page
            st.markdown("---")
        
        with st.expander("🏗️ Enhanced Architecture Overview", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📁 **Active Modules**")
                st.write("✅ **`config/`** - Settings, constants, parameters")
                st.write("✅ **`data/`** - Fetching, validation, management")
                st.write("✅ **`analysis/`** - Technical, fundamental, market, options")
                st.write("✅ **`ui/`** - Components, headers, score bars")
                st.write("✅ **`utils/`** - Helpers, decorators, formatters")
                
            with col2:
                st.write("### 🎯 **Enhanced Features**")
                st.write("• **📈 Interactive Charts** with all technical overlays")
                st.write("• **📊 All Fibonacci EMAs** (21,55,89,144,233)")
                st.write("• **📅 Previous Week High/Low** levels")
                st.write("• **📋 Detailed Fundamental** criteria breakdown")
                st.write("• **🎯 Collapsible Technical Screener**")
                st.write("• **⚙️ Toggleable sections** in sidebar")
        
        # Show current market status
        market_status = get_market_status()
        st.info(f"**Market Status:** {market_status}")
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=False):
            st.write("1. **Enter a symbol** and press Enter for instant analysis")
            st.write("2. **Click any Quick Link** for immediate analysis")
            st.write("3. **View interactive charts** with technical indicators")
            st.write("4. **Toggle sections** on/off in Analysis Sections panel")
            st.write("5. **🆕 Interactive charts** - Full Plotly integration")

    # Footer
    st.markdown("---")
    st.write("### 📊 System Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Version:** VWV Professional v4.0 - Interactive Charts")
        st.write(f"**Architecture:** Full Modular + Interactive Charts")
    with col2:
        st.write(f"**Status:** ✅ All Modules + Interactive Charts Active")
        st.write(f"**Current Symbol:** {st.session_state.get('current_symbol', 'SPY')}")
    with col3:
        st.write(f"**Charts:** Candlestick, EMAs, VWAP, Volume, POC")
        st.write(f"**Compatible:** Streamlit Cloud Free Tier")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.write("Please refresh the page and try again.")
        
        if st.checkbox("Show Error Details"):
            st.exception(e)
