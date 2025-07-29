"""
UI components for the VWV Trading System
Updated with better color scheme for score bar
"""
import streamlit as st

def create_technical_score_bar(score, details=None):
    """Create professional gradient bar for technical score with dark theme"""
    
    # Determine interpretation and color
    if score >= 80:
        interpretation = "Very Bullish"
        primary_color = "#00A86B"  # Jade green
    elif score >= 65:
        interpretation = "Bullish" 
        primary_color = "#32CD32"  # Lime green
    elif score >= 55:
        interpretation = "Slightly Bullish"
        primary_color = "#9ACD32"  # Yellow green
    elif score >= 45:
        interpretation = "Neutral"
        primary_color = "#FFD700"  # Gold
    elif score >= 35:
        interpretation = "Slightly Bearish"
        primary_color = "#FF8C00"  # Dark orange
    elif score >= 20:
        interpretation = "Bearish"
        primary_color = "#FF4500"  # Orange red
    else:
        interpretation = "Very Bearish"
        primary_color = "#DC143C"  # Crimson
    
    # Create professional gradient bar HTML with dark theme
    html = f"""
    <div style="margin: 1.5rem 0; padding: 1.5rem; 
                background: linear-gradient(135deg, #1e1e1e 0%, #2d2d30 50%, #1a1a1a 100%); 
                border-radius: 15px; 
                border: 1px solid #404040; 
                box-shadow: 0 4px 12px rgba(0,0,0,0.3),
                           inset 0 1px 0 rgba(255,255,255,0.1);">
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div>
                <span style="font-weight: 700; color: #ffffff; font-size: 1.3em; text-shadow: 0 1px 2px rgba(0,0,0,0.5);">
                    Technical Composite Score
                </span>
                <div style="font-size: 0.95em; color: #b0b0b0; margin-top: 0.3rem; text-shadow: 0 1px 1px rgba(0,0,0,0.5);">
                    Aggregated signal from all technical indicators
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-weight: 700; color: {primary_color}; font-size: 2.2em; text-shadow: 0 2px 4px rgba(0,0,0,0.4);">
                    {score}
                </div>
                <div style="font-size: 0.95em; color: {primary_color}; font-weight: 600; text-shadow: 0 1px 2px rgba(0,0,0,0.4);">
                    {interpretation}
                </div>
            </div>
        </div>
        
        <div style="position: relative; width: 100%; height: 28px; 
                    background: linear-gradient(to right, 
                        #DC143C 0%, #FF4500 15%, #FF8C00 30%, #FFD700 50%, 
                        #9ACD32 70%, #32CD32 85%, #00A86B 100%); 
                    border-radius: 14px; 
                    border: 2px solid #404040; 
                    overflow: hidden;
                    box-shadow: inset 0 2px 4px rgba(0,0,0,0.4);">
            
            <!-- Score indicator -->
            <div style="position: absolute; left: {score}%; top: 50%; transform: translate(-50%, -50%); 
                        width: 8px; height: 36px; 
                        background: linear-gradient(to bottom, #ffffff 0%, #f0f0f0 100%); 
                        border: 2px solid #1a1a1a; 
                        border-radius: 4px; 
                        box-shadow: 0 3px 6px rgba(0,0,0,0.5), 
                                   inset 0 1px 0 rgba(255,255,255,0.8); 
                        z-index: 10;">
            </div>
            
            <!-- Progress fill with glow effect -->
            <div style="position: absolute; left: 0; top: 0; height: 100%; width: {score}%; 
                        background: linear-gradient(to right, 
                            transparent 0%, 
                            {primary_color}40 70%, 
                            {primary_color}60 100%); 
                        border-radius: 14px;
                        box-shadow: inset 0 0 8px {primary_color}80;">
            </div>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 0.7rem; font-size: 0.8em; color: #a0a0a0; text-shadow: 0 1px 1px rgba(0,0,0,0.5);">
            <span style="font-weight: 600;">Very Bearish</span>
            <span style="font-weight: 600;">Neutral</span>
            <span style="font-weight: 600;">Very Bullish</span>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 0.3rem; font-size: 0.75em; color: #808080;">
            <span>1</span>
            <span>25</span>
            <span>50</span>
            <span>75</span>
            <span>100</span>
        </div>
    </div>
    """
    
    return html

def create_header():
    """Create the main header with enhanced dark theme"""
    st.markdown("""
    <div style="position: relative; padding: 3rem 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center; 
                overflow: hidden; min-height: 200px; 
                background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
                border: 1px solid #404040;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
        <div style="position: relative; z-index: 3; 
                    background: linear-gradient(135deg, rgba(0,0,0,0.4) 0%, rgba(0,0,0,0.2) 100%); 
                    padding: 2rem; border-radius: 12px; 
                    backdrop-filter: blur(10px); 
                    border: 1px solid rgba(255,255,255,0.1);
                    box-shadow: inset 0 1px 0 rgba(255,255,255,0.1);">
            <h1 style="font-size: 2.8rem; margin-bottom: 1rem; color: #ffffff; 
                       text-shadow: 2px 2px 8px rgba(0,0,0,0.8); 
                       font-weight: 700; letter-spacing: 1px;">
                VWV Professional Trading System
            </h1>
            <p style="color: #e0f0e0; text-shadow: 1px 1px 4px rgba(0,0,0,0.7); 
                      margin: 0.5rem 0; font-size: 1.1rem; font-weight: 400;">
                Advanced market analysis with enhanced technical indicators
            </p>
            <p style="color: #c0d0c0; font-style: italic; font-size: 1rem;">
                <em>Modular Architecture: Complete Implementation</em>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Optional

def create_interactive_chart(data: pd.DataFrame, analysis_results: Dict[str, Any], symbol: str) -> go.Figure:
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

def display_interactive_charts(analysis_results: Dict[str, Any], market_data: pd.DataFrame):
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
