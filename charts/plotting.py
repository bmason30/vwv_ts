"""
Advanced charting functionality for VWV Trading System
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def create_comprehensive_trading_chart(data: pd.DataFrame, analysis_results: Dict[str, Any], height: int = 800) -> go.Figure:
    """Create comprehensive trading chart with multiple indicators"""
    try:
        if data is None or len(data) == 0:
            st.error("No data available for charting")
            return None
            
        # Create subplot with secondary y-axis for volume
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price & Technical Indicators', 'Volume', 'RSI & Momentum'),
            row_heights=[0.6, 0.2, 0.2],
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}], 
                   [{"secondary_y": True}]]
        )
        
        # Main price candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'], 
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ),
            row=1, col=1
        )
        
        # Add technical indicators to price chart
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        
        # Daily VWAP
        daily_vwap = enhanced_indicators.get('daily_vwap')
        if daily_vwap:
            vwap_line = [daily_vwap] * len(data)
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=vwap_line,
                    mode='lines',
                    name='VWAP',
                    line=dict(color='purple', width=2)
                ),
                row=1, col=1
            )
        
        # Point of Control
        poc = enhanced_indicators.get('point_of_control')
        if poc:
            poc_line = [poc] * len(data)
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=poc_line,
                    mode='lines',
                    name='Point of Control',
                    line=dict(color='orange', width=2, dash='dot')
                ),
                row=1, col=1
            )
        
        # Fibonacci EMAs
        fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
        ema_colors = ['blue', 'green', 'red', 'purple', 'brown']
        
        for i, (ema_name, ema_value) in enumerate(fibonacci_emas.items()):
            if ema_value and i < len(ema_colors):
                period = ema_name.split('_')[1]
                ema_series = data['Close'].ewm(span=int(period)).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=ema_series,
                        mode='lines',
                        name=f'EMA {period}',
                        line=dict(color=ema_colors[i], width=1)
                    ),
                    row=1, col=1
                )
        
        # Bollinger Bands
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        bb_data = comprehensive_technicals.get('bollinger_bands', {})
        
        if bb_data and 'upper' in bb_data:
            # Calculate BB series (simplified for display)
            bb_period = 20
            if len(data) >= bb_period:
                bb_sma = data['Close'].rolling(bb_period).mean()
                bb_std = data['Close'].rolling(bb_period).std()
                bb_upper = bb_sma + (bb_std * 2)
                bb_lower = bb_sma - (bb_std * 2)
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=bb_upper,
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='gray', width=1, dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=bb_lower,
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)',
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # Volume chart
        colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
                 for i in range(len(data))]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Volume SMA
        volume_sma = comprehensive_technicals.get('volume_sma_20')
        if volume_sma:
            volume_sma_series = data['Volume'].rolling(20).mean()
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=volume_sma_series,
                    mode='lines',
                    name='Volume SMA(20)',
                    line=dict(color='orange', width=2)
                ),
                row=2, col=1
            )
        
        # RSI
        rsi_14 = comprehensive_technicals.get('rsi_14', 50)
        if len(data) >= 14:
            # Calculate RSI series for display
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.inf)
            rsi_series = 100 - (100 / (1 + rs))
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=rsi_series,
                    mode='lines',
                    name='RSI(14)',
                    line=dict(color='blue', width=2)
                ),
                row=3, col=1, secondary_y=False
            )
            
            # RSI overbought/oversold levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        
        # MACD on secondary y-axis
        macd_data = comprehensive_technicals.get('macd', {})
        if macd_data and len(data) >= 26:
            # Calculate MACD series
            ema12 = data['Close'].ewm(span=12).mean()
            ema26 = data['Close'].ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=macd_line,
                    mode='lines',
                    name='MACD',
                    line=dict(color='red', width=1)
                ),
                row=3, col=1, secondary_y=True
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=signal_line,
                    mode='lines',
                    name='Signal',
                    line=dict(color='orange', width=1)
                ),
                row=3, col=1, secondary_y=True
            )
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=histogram,
                    name='MACD Histogram',
                    opacity=0.6
                ),
                row=3, col=1, secondary_y=True
            )
        
        # Update layout
        fig.update_layout(
            title=f"{analysis_results.get('symbol', 'Stock')} - Comprehensive Trading Analysis",
            height=height,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template='plotly_white',
            hovermode='x unified'
        )
        
        # Update y-axes
        fig.update_yaxis(title_text="Price ($)", row=1, col=1)
        fig.update_yaxis(title_text="Volume", row=2, col=1)
        fig.update_yaxis(title_text="RSI", row=3, col=1, secondary_y=False, range=[0, 100])
        fig.update_yaxis(title_text="MACD", row=3, col=1, secondary_y=True)
        
        # Update x-axes
        fig.update_xaxis(title_text="Date", row=3, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Chart creation error: {e}")
        st.error(f"Failed to create chart: {str(e)}")
        return None

def create_options_levels_chart(data: pd.DataFrame, analysis_results: Dict[str, Any]) -> Optional[go.Figure]:
    """Create options levels overlay chart"""
    try:
        current_price = analysis_results.get('current_price', 0)
        options_levels = analysis_results.get('enhanced_indicators', {}).get('options_levels', [])
        
        if not options_levels:
            return None
            
        fig = go.Figure()
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Price',
                line=dict(color='black', width=2)
            )
        )
        
        # Add option strike levels
        colors = ['red', 'orange', 'yellow', 'lightblue']
        
        for i, level in enumerate(options_levels[:4]):  # Show first 4 DTEs
            dte = level.get('DTE', 0)
            put_strike = level.get('Put Strike', 0)
            call_strike = level.get('Call Strike', 0)
            
            color = colors[i % len(colors)]
            
            # Put strike line
            fig.add_hline(
                y=put_strike,
                line_dash="dash",
                line_color=color,
                opacity=0.7,
                annotation_text=f"{dte}D Put: ${put_strike}",
                annotation_position="left"
            )
            
            # Call strike line  
            fig.add_hline(
                y=call_strike,
                line_dash="dash", 
                line_color=color,
                opacity=0.7,
                annotation_text=f"{dte}D Call: ${call_strike}",
                annotation_position="right"
            )
        
        # Current price line
        fig.add_hline(
            y=current_price,
            line_color="blue",
            line_width=3,
            annotation_text=f"Current: ${current_price}",
            annotation_position="top left"
        )
        
        fig.update_layout(
            title="Options Levels - Premium Selling Strikes",
            height=400,
            yaxis_title="Price ($)",
            xaxis_title="Date",
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Options chart creation error: {e}")
        return None

def create_technical_score_chart(analysis_results: Dict[str, Any]) -> Optional[go.Figure]:
    """Create technical score breakdown chart"""
    try:
        from analysis.technical import calculate_composite_technical_score
        
        score, details = calculate_composite_technical_score(analysis_results)
        component_scores = details.get('component_scores', {})
        
        if not component_scores:
            return None
            
        # Create bar chart of component scores
        components = list(component_scores.keys())
        scores = list(component_scores.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=components,
                y=scores,
                marker_color=['green' if s >= 50 else 'red' for s in scores],
                text=[f'{s:.1f}' for s in scores],
                textposition='auto'
            )
        ])
        
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title=f"Technical Score Breakdown - Composite: {score}",
            yaxis_title="Score",
            yaxis_range=[0, 100],
            height=300,
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Technical score chart error: {e}")
        return None

def display_trading_charts(data: pd.DataFrame, analysis_results: Dict[str, Any]):
    """Display all trading charts in organized tabs"""
    try:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Main Chart", "🎯 Options Levels", "📈 Technical Breakdown", "📋 Data Table"])
        
        with tab1:
            st.subheader("Comprehensive Trading Chart")
            main_chart = create_comprehensive_trading_chart(data, analysis_results)
            if main_chart:
                st.plotly_chart(main_chart, use_container_width=True)
            else:
                st.error("Unable to create main trading chart")
        
        with tab2:
            st.subheader("Options Premium Selling Levels")
            options_chart = create_options_levels_chart(data, analysis_results)
            if options_chart:
                st.plotly_chart(options_chart, use_container_width=True)
            else:
                st.info("Options levels chart not available")
        
        with tab3:
            st.subheader("Technical Score Components")
            score_chart = create_technical_score_chart(analysis_results)
            if score_chart:
                st.plotly_chart(score_chart, use_container_width=True)
            else:
                st.info("Technical score chart not available")
        
        with tab4:
            st.subheader("Raw Market Data")
            st.dataframe(data.tail(50), use_container_width=True)
    
    except Exception as e:
        logger.error(f"Chart display error: {e}")
        st.error(f"Error displaying charts: {str(e)}")
