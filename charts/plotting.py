"""
File: plotting.py v1.0.3
VWV Professional Trading System v4.2.2
Advanced charting functionality with type-safe data handling
Created: 2025-08-15
Updated: 2025-10-07
File Version: v1.0.3 - Added type-safe number extraction for options charts
System Version: v4.2.2 - Advanced Options with Fibonacci Integration
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)

def safe_extract_number(value, default=0):
    """
    Safely extract a numeric value from various formats
    Handles: numbers, strings with $ or %, formatted strings, etc.
    
    Args:
        value: Input value (int, float, str, or None)
        default: Default value to return if extraction fails
        
    Returns:
        float: Extracted numeric value or default
    """
    if value is None:
        return default
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Remove $ and % symbols and convert to float
        cleaned = re.sub(r'[^\d.-]', '', value)
        try:
            return float(cleaned) if cleaned else default
        except ValueError:
            return default
    
    return default

def create_comprehensive_trading_chart(data: pd.DataFrame, analysis_results: Dict[str, Any], height: int = 800) -> Optional[go.Figure]:
    """
    Create comprehensive trading chart with multiple indicators
    
    Args:
        data: OHLC price data with DatetimeIndex
        analysis_results: Dictionary containing all analysis results
        height: Chart height in pixels
        
    Returns:
        Plotly Figure object or None if creation fails
    """
    try:
        if data is None or len(data) == 0:
            logger.error("No data available for charting")
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
        if daily_vwap and daily_vwap > 0:
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
        if poc and poc > 0:
            poc_line = [poc] * len(data)
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=poc_line,
                    mode='lines',
                    name='POC',
                    line=dict(color='orange', width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # Fibonacci EMAs
        fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
        ema_colors = {'EMA_21': 'red', 'EMA_34': 'blue', 'EMA_55': 'green', 'EMA_89': 'purple'}
        
        for ema_name, ema_value in fibonacci_emas.items():
            if ema_value and ema_value > 0 and len(data) >= 21:
                ema_line = [ema_value] * len(data)
                color = ema_colors.get(ema_name, 'gray')
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=ema_line,
                        mode='lines',
                        name=ema_name,
                        line=dict(color=color, width=1, dash='dot')
                    ),
                    row=1, col=1
                )
        
        # Bollinger Bands
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        bb = comprehensive_technicals.get('bollinger_bands', {})
        
        if bb:
            upper = bb.get('upper')
            lower = bb.get('lower')
            middle = bb.get('middle')
            
            if upper and lower and middle:
                bb_upper = [upper] * len(data)
                bb_lower = [lower] * len(data)
                bb_middle = [middle] * len(data)
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=bb_upper,
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='gray', width=1, dash='dot'),
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
                        line=dict(color='gray', width=1, dash='dot'),
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
        if len(data) >= 20:
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
        if len(data) >= 26:
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
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1, secondary_y=False, range=[0, 100])
        fig.update_yaxes(title_text="MACD", row=3, col=1, secondary_y=True)
        
        # Update x-axes
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Chart creation error: {e}")
        st.error(f"Failed to create chart: {str(e)}")
        return None

def create_options_levels_chart(data: pd.DataFrame, analysis_results: Dict[str, Any]) -> Optional[go.Figure]:
    """
    Create options levels overlay chart with TYPE-SAFE number extraction
    
    CRITICAL FIX: Handles both numeric and formatted string values to prevent
    "unsupported operand type(s) for +: 'int' and 'str'" errors
    
    Args:
        data: OHLC price data with DatetimeIndex
        analysis_results: Dictionary containing analysis results
        
    Returns:
        Plotly Figure object or None if creation fails
    """
    try:
        current_price = safe_extract_number(analysis_results.get('current_price', 0))
        options_levels = analysis_results.get('enhanced_indicators', {}).get('options_levels', [])
        
        if not options_levels or current_price == 0:
            logger.warning("No options levels data available")
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
        
        # Add option strike levels with type-safe extraction
        colors = ['red', 'orange', 'yellow', 'lightblue']
        
        for i, level in enumerate(options_levels[:4]):  # Show first 4 DTEs
            # CRITICAL: Use safe_extract_number for all numeric values
            dte = int(safe_extract_number(level.get('DTE', 0), 0))
            put_strike = safe_extract_number(level.get('Put Strike', 0), 0)
            call_strike = safe_extract_number(level.get('Call Strike', 0), 0)
            
            if dte == 0 or put_strike == 0 or call_strike == 0:
                logger.warning(f"Skipping invalid options level {i}: DTE={dte}, Put={put_strike}, Call={call_strike}")
                continue
            
            color = colors[i % len(colors)]
            
            # Put strike line
            fig.add_hline(
                y=put_strike,
                line_dash="dash",
                line_color=color,
                opacity=0.7,
                annotation_text=f"{dte}D Put: ${put_strike:.2f}",
                annotation_position="left"
            )
            
            # Call strike line  
            fig.add_hline(
                y=call_strike,
                line_dash="dash", 
                line_color=color,
                opacity=0.7,
                annotation_text=f"{dte}D Call: ${call_strike:.2f}",
                annotation_position="right"
            )
        
        # Current price line
        fig.add_hline(
            y=current_price,
            line_color="blue",
            line_width=3,
            annotation_text=f"Current: ${current_price:.2f}",
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
        st.error(f"Failed to create options chart: {str(e)}")
        return None

def create_technical_score_chart(analysis_results: Dict[str, Any]) -> Optional[go.Figure]:
    """
    Create technical score breakdown chart
    
    Args:
        analysis_results: Dictionary containing analysis results
        
    Returns:
        Plotly Figure object or None if creation fails
    """
    try:
        from analysis.technical import calculate_composite_technical_score
        
        score, details = calculate_composite_technical_score(analysis_results)
        component_scores = details.get('component_scores', {})
        
        if not component_scores:
            logger.warning("No component scores available")
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
            title=f"Technical Score Breakdown - Composite: {score:.1f}",
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
    """
    Display all trading charts in organized tabs
    
    Args:
        data: OHLC price data with DatetimeIndex
        analysis_results: Dictionary containing all analysis results
    """
    try:
        # Validate inputs first
        if data is None or data.empty:
            st.error("❌ No data available for charting")
            return
            
        if not analysis_results:
            st.error("❌ No analysis results available for charting")
            return
            
        # Check if required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required data columns: {missing_columns}")
            return
            
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Main Chart", "🎯 Options Levels", "📈 Technical Breakdown", "📋 Data Table"])
        
        with tab1:
            st.subheader("Comprehensive Trading Chart")
            try:
                main_chart = create_comprehensive_trading_chart(data, analysis_results)
                if main_chart:
                    st.plotly_chart(main_chart, use_container_width=True)
                else:
                    st.error("Unable to create main trading chart")
                    # Fallback to simple line chart
                    st.subheader("Fallback: Basic Price Chart")
                    st.line_chart(data['Close'])
            except Exception as e:
                st.error(f"Main chart error: {str(e)}")
                # Fallback to simple line chart
                st.subheader("Fallback: Basic Price Chart")
                st.line_chart(data['Close'])
        
        with tab2:
            st.subheader("Options Premium Selling Levels")
            try:
                options_chart = create_options_levels_chart(data, analysis_results)
                if options_chart:
                    st.plotly_chart(options_chart, use_container_width=True)
                else:
                    st.info("Options levels chart not available")
            except Exception as e:
                st.error(f"Options chart error: {str(e)}")
        
        with tab3:
            st.subheader("Technical Score Components")
            try:
                score_chart = create_technical_score_chart(analysis_results)
                if score_chart:
                    st.plotly_chart(score_chart, use_container_width=True)
                else:
                    st.info("Technical score chart not available")
            except Exception as e:
                st.error(f"Technical score chart error: {str(e)}")
        
        with tab4:
            st.subheader("Raw Market Data")
            try:
                # Show data info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Data Points", len(data))
                with col2:
                    st.metric("Date Range", f"{len(data)} days")
                with col3:
                    st.metric("Columns", len(data.columns))
                
                # Show actual data
                st.dataframe(data.tail(50), use_container_width=True)
            except Exception as e:
                st.error(f"Data table error: {str(e)}")
    
    except Exception as e:
        logger.error(f"Chart display error: {e}")
        st.error(f"Error displaying charts: {str(e)}")
        
        # Ultimate fallback
        st.subheader("Emergency Fallback: Basic Data View")
        if data is not None and not data.empty:
            st.line_chart(data['Close'])
            st.write("Data shape:", data.shape)
            st.write("Columns:", list(data.columns))
        else:
            st.error("No data available at all")
