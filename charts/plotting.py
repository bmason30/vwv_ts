"""
File: plotting.py v1.4.0
VWV Research And Analysis System v4.3.0
Advanced charting functionality with type-safe data handling
Created: 2025-08-15
Updated: 2025-11-19
File Version: v1.4.0 - Added risk/reward scatter plot and Phase 3 enhancements
Changes in this version:
    - v1.3.0: Improved color scheme with clear put/call differentiation
    - v1.3.0: Added expected move zone visualization
    - v1.3.0: Enhanced annotations with Greeks and probabilities
    - v1.3.0: Added interactive filtering parameters
    - v1.4.0: Added risk/reward scatter plot visualization
    - v1.4.0: Added "sweet spot" analysis zone
System Version: v4.3.0 - Black-Scholes Options Pricing
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

def get_command_center_layout():
    """
    Get command center dark theme layout settings for all charts
    Consistent with institutional terminal design
    """
    return {
        'template': 'plotly_dark',
        'paper_bgcolor': '#0f0f0f',
        'plot_bgcolor': '#0f0f0f',
        'font': dict(
            family='JetBrains Mono, monospace',
            size=11,
            color='#9ca3af'
        ),
        'title': dict(
            font=dict(
                family='Inter, sans-serif',
                size=14,
                color='#ffffff'
            ),
            x=0.02,
            xanchor='left'
        ),
        'xaxis': dict(
            gridcolor='rgba(255, 255, 255, 0.05)',
            linecolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.1)'
        ),
        'yaxis': dict(
            gridcolor='rgba(255, 255, 255, 0.05)',
            linecolor='rgba(255, 255, 255, 0.1)',
            zerolinecolor='rgba(255, 255, 255, 0.1)'
        ),
        'hovermode': 'x unified',
        'hoverlabel': dict(
            bgcolor='#151515',
            font=dict(
                family='JetBrains Mono, monospace',
                color='#ffffff'
            ),
            bordercolor='#3b82f6'
        )
    }

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
        # Apply command center theme
        layout = get_command_center_layout()
        layout.update({
            'title': f"{analysis_results.get('symbol', 'Stock')} - Comprehensive Trading Analysis",
            'height': height,
            'showlegend': True,
            'xaxis_rangeslider_visible': False
        })
        fig.update_layout(**layout)
        
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

def create_options_levels_chart(
    data: pd.DataFrame,
    analysis_results: Dict[str, Any],
    show_puts: bool = True,
    show_calls: bool = True,
    show_expected_move: bool = True,
    selected_dtes: list = None,
    show_annotations: bool = True
) -> Optional[go.Figure]:
    """
    Create enhanced options levels overlay chart with improved visuals

    VERSION 1.3.0 ENHANCEMENTS:
    - Clear color differentiation (red gradient for puts, green gradient for calls)
    - Expected move zone visualization (±1 standard deviation)
    - Enhanced annotations with Greeks and probabilities
    - Interactive filtering for puts/calls/DTEs

    Args:
        data: OHLC price data with DatetimeIndex
        analysis_results: Dictionary containing analysis results
        show_puts: Show put strike levels
        show_calls: Show call strike levels
        show_expected_move: Show expected move zone
        selected_dtes: List of DTEs to display (None = all)
        show_annotations: Show detailed annotations

    Returns:
        Plotly Figure object or None if creation fails
    """
    try:
        current_price = safe_extract_number(analysis_results.get('current_price', 0))
        options_levels = analysis_results.get('enhanced_indicators', {}).get('options_levels', [])
        volatility = safe_extract_number(analysis_results.get('enhanced_indicators', {}).get('comprehensive_technicals', {}).get('volatility', 25), 25)

        if not options_levels or current_price == 0:
            logger.warning("No options levels data available")
            return None

        fig = go.Figure()

        # Enhanced color schemes
        PUT_COLORS = {
            7: 'rgba(255, 59, 48, 0.8)',    # Bright red - nearest expiration
            14: 'rgba(255, 99, 71, 0.7)',   # Tomato red
            30: 'rgba(255, 140, 105, 0.6)', # Lighter red
            45: 'rgba(255, 160, 122, 0.5)', # Lightest red
            60: 'rgba(255, 182, 142, 0.4)'  # Very light red
        }

        CALL_COLORS = {
            7: 'rgba(52, 199, 89, 0.8)',    # Bright green - nearest expiration
            14: 'rgba(52, 199, 120, 0.7)',  # Medium green
            30: 'rgba(52, 199, 150, 0.6)',  # Lighter green
            45: 'rgba(52, 199, 180, 0.5)',  # Lightest green
            60: 'rgba(52, 199, 200, 0.4)'   # Very light green
        }

        # Line width based on DTE (nearer = thicker)
        line_width_map = {7: 3, 14: 2.5, 30: 2, 45: 1.8, 60: 1.5}

        # Add expected move zone if requested
        if show_expected_move and len(options_levels) > 0:
            # Get minimum DTE for expected move calculation
            min_dte = min([safe_extract_number(level.get('DTE', 30), 30) for level in options_levels])

            # Calculate expected move (±1 standard deviation)
            vol_annual = volatility / 100.0
            import math
            expected_move_upper = current_price * (1 + vol_annual * math.sqrt(min_dte / 365.0))
            expected_move_lower = current_price * (1 - vol_annual * math.sqrt(min_dte / 365.0))

            # Add shaded expected move zone
            fig.add_shape(
                type="rect",
                x0=data.index[0],
                x1=data.index[-1],
                y0=expected_move_lower,
                y1=expected_move_upper,
                fillcolor="rgba(100, 150, 255, 0.1)",
                line=dict(width=1, color="rgba(100, 150, 255, 0.3)", dash="dot"),
                layer="below",
                name="Expected Move Zone"
            )

            # Add expected move annotations
            if show_annotations:
                fig.add_annotation(
                    x=data.index[len(data)//2],
                    y=expected_move_upper,
                    text=f"Expected Move +1σ (${expected_move_upper:.2f})",
                    showarrow=False,
                    font=dict(size=9, color="blue"),
                    yshift=10
                )
                fig.add_annotation(
                    x=data.index[len(data)//2],
                    y=expected_move_lower,
                    text=f"Expected Move -1σ (${expected_move_lower:.2f})",
                    showarrow=False,
                    font=dict(size=9, color="blue"),
                    yshift=-10
                )

        # Price line - bright cyan for visibility on dark backgrounds
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Price',
                line=dict(color='rgba(0, 255, 255, 0.9)', width=2.5)  # Bright cyan
            )
        )

        # Filter DTEs if specified
        if selected_dtes:
            options_levels = [level for level in options_levels
                            if int(safe_extract_number(level.get('DTE', 0), 0)) in selected_dtes]

        # Add option strike levels with enhanced visualization
        for level in options_levels:
            dte = int(safe_extract_number(level.get('DTE', 0), 0))
            put_strike = safe_extract_number(level.get('Put Strike', 0), 0)
            call_strike = safe_extract_number(level.get('Call Strike', 0), 0)

            if dte == 0 or (put_strike == 0 and call_strike == 0):
                continue

            # Get colors and line width for this DTE
            put_color = PUT_COLORS.get(dte, 'rgba(255, 100, 100, 0.5)')
            call_color = CALL_COLORS.get(dte, 'rgba(100, 200, 100, 0.5)')
            line_width = line_width_map.get(dte, 2.0)

            # Put strike line
            if show_puts and put_strike > 0:
                # Extract Greeks for annotation
                put_delta = level.get('Put Delta', 'N/A')
                put_theta = level.get('Put Theta', 'N/A')
                put_pot = level.get('Put PoT', 'N/A')

                fig.add_hline(
                    y=put_strike,
                    line_dash="solid",
                    line_color=put_color,
                    line_width=line_width,
                    annotation_text=(f"<b>{dte}D Put</b><br>${put_strike:.2f}<br>Δ:{put_delta} θ:{put_theta}<br>PoT:{put_pot}"
                                   if show_annotations else f"{dte}D Put: ${put_strike:.2f}"),
                    annotation_position="left",
                    annotation_font=dict(size=9, color=put_color.replace('0.', '1.'))
                )

            # Call strike line
            if show_calls and call_strike > 0:
                # Extract Greeks for annotation
                call_delta = level.get('Call Delta', 'N/A')
                call_theta = level.get('Call Theta', 'N/A')
                call_pot = level.get('Call PoT', 'N/A')

                fig.add_hline(
                    y=call_strike,
                    line_dash="solid",
                    line_color=call_color,
                    line_width=line_width,
                    annotation_text=(f"<b>{dte}D Call</b><br>${call_strike:.2f}<br>Δ:{call_delta} θ:{call_theta}<br>PoT:{call_pot}"
                                   if show_annotations else f"{dte}D Call: ${call_strike:.2f}"),
                    annotation_position="right",
                    annotation_font=dict(size=9, color=call_color.replace('0.', '1.'))
                )

        # Current price line
        fig.add_hline(
            y=current_price,
            line_color="rgba(0, 100, 255, 0.9)",
            line_width=3,
            line_dash="dot",
            annotation_text=f"<b>Current Price: ${current_price:.2f}</b>",
            annotation_position="top right",
            annotation_font=dict(size=11, color="blue")
        )

        # Apply command center theme
        layout = get_command_center_layout()
        layout.update({
            'title': "Options Levels - Premium Selling Strikes (Enhanced Visualization)",
            'height': 500,
            'yaxis_title': "Price ($)",
            'xaxis_title': "Date",
            'showlegend': True
        })
        fig.update_layout(**layout)

        return fig

    except Exception as e:
        logger.error(f"Options chart creation error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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
        
        # Apply command center theme
        layout = get_command_center_layout()
        layout.update({
            'title': f"Technical Score Breakdown - Composite: {score:.1f}",
            'yaxis_title': "Score",
            'yaxis_range': [0, 100],
            'height': 300
        })
        fig.update_layout(**layout)
        
        return fig

    except Exception as e:
        logger.error(f"Technical score chart error: {e}")
        return None

def create_risk_reward_scatter(options_data: list, current_price: float) -> go.Figure:
    """
    Create interactive scatter plot showing risk/reward profile of all strikes

    VERSION 1.4.0 - Advanced risk/reward visualization

    X-axis: Probability of Profit (PoP)
    Y-axis: Premium collected
    Size: Days to expiration (larger = longer DTE)
    Color: Put (red) vs Call (green)

    Parameters:
    -----------
    options_data : list - Options data with strikes, premiums, and probabilities
    current_price : float - Current stock price

    Returns:
    --------
    Plotly Figure object
    """
    try:
        fig = go.Figure()

        # Prepare data
        puts_data = []
        calls_data = []

        for opt in options_data:
            # Put data
            try:
                put_pop_str = opt.get('Put PoP', '0%')
                put_premium_str = opt.get('Put Premium', '$0')
                put_dte = opt['DTE']
                put_strike = opt['Put Strike']

                put_pop = float(safe_extract_number(put_pop_str, 0))
                put_premium = float(safe_extract_number(put_premium_str, 0))

                puts_data.append({
                    'pop': put_pop,
                    'premium': put_premium,
                    'dte': put_dte,
                    'strike': put_strike,
                    'label': f"Put ${put_strike:.2f} - {put_dte}D"
                })
            except:
                pass

            # Call data
            try:
                call_pop_str = opt.get('Call PoP', '0%')
                call_premium_str = opt.get('Call Premium', '$0')
                call_dte = opt['DTE']
                call_strike = opt['Call Strike']

                call_pop = float(safe_extract_number(call_pop_str, 0))
                call_premium = float(safe_extract_number(call_premium_str, 0))

                calls_data.append({
                    'pop': call_pop,
                    'premium': call_premium,
                    'dte': call_dte,
                    'strike': call_strike,
                    'label': f"Call ${call_strike:.2f} - {call_dte}D"
                })
            except:
                pass

        # Plot puts
        if puts_data:
            fig.add_trace(go.Scatter(
                x=[p['pop'] for p in puts_data],
                y=[p['premium'] for p in puts_data],
                mode='markers',
                name='Put Strikes',
                marker=dict(
                    size=[p['dte'] * 0.7 for p in puts_data],  # Scale bubble size
                    sizemode='diameter',
                    sizemin=8,
                    color='rgba(255, 59, 48, 0.7)',
                    line=dict(width=2, color='white')
                ),
                text=[p['label'] for p in puts_data],
                hovertemplate='<b>%{text}</b><br>PoP: %{x:.1f}%<br>Premium: $%{y:.2f}<extra></extra>'
            ))

        # Plot calls
        if calls_data:
            fig.add_trace(go.Scatter(
                x=[c['pop'] for c in calls_data],
                y=[c['premium'] for c in calls_data],
                mode='markers',
                name='Call Strikes',
                marker=dict(
                    size=[c['dte'] * 0.7 for c in calls_data],  # Scale bubble size
                    sizemode='diameter',
                    sizemin=8,
                    color='rgba(52, 199, 89, 0.7)',
                    line=dict(width=2, color='white')
                ),
                text=[c['label'] for c in calls_data],
                hovertemplate='<b>%{text}</b><br>PoP: %{x:.1f}%<br>Premium: $%{y:.2f}<extra></extra>'
            ))

        # Add "sweet spot" annotation
        all_premiums = [p['premium'] for p in puts_data + calls_data]
        if all_premiums:
            max_premium = max(all_premiums)

            # Sweet spot zone (PoP > 65%, good premium)
            fig.add_shape(
                type="rect",
                x0=65, x1=95,
                y0=0, y1=max_premium * 1.1,
                fillcolor="rgba(52, 199, 89, 0.1)",
                line=dict(width=0),
                layer="below"
            )

            fig.add_annotation(
                x=80, y=max_premium * 1.05,
                text="Sweet Spot<br>(High PoP + Good Premium)",
                showarrow=False,
                font=dict(size=10, color="green", family="Arial Black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="green",
                borderwidth=1
            )

        # Apply command center theme
        layout = get_command_center_layout()
        layout.update({
            'title': "Risk/Reward Analysis - Strike Selection",
            'xaxis_title': "Probability of Profit (%)",
            'yaxis_title': "Premium Collected ($)",
            'height': 500,
            'hovermode': 'closest',
            'showlegend': True,
            'xaxis': dict(range=[0, 100], gridcolor='rgba(255, 255, 255, 0.05)'),
            'yaxis': dict(range=[0, max(all_premiums) * 1.15 if all_premiums else 10], gridcolor='rgba(255, 255, 255, 0.05)')
        })
        fig.update_layout(**layout)

        return fig

    except Exception as e:
        logger.error(f"Risk/reward scatter plot error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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
                    st.plotly_chart(main_chart, width='stretch')
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
            st.subheader("💰 Options Premium Selling Levels")
            try:
                # Add interactive chart controls (v1.3.0 enhancement)
                from ui.components import create_options_chart_controls
                # Use symbol as key prefix to avoid duplicate keys when multiple charts on same page
                symbol = analysis_results.get('symbol', 'unknown')
                chart_controls = create_options_chart_controls(key_prefix=f"{symbol}_")

                # Create chart with user-selected options
                options_chart = create_options_levels_chart(
                    data,
                    analysis_results,
                    show_puts=chart_controls['show_puts'],
                    show_calls=chart_controls['show_calls'],
                    show_expected_move=chart_controls['show_expected_move'],
                    selected_dtes=chart_controls['selected_dtes'],
                    show_annotations=chart_controls['show_annotations']
                )

                if options_chart:
                    st.plotly_chart(options_chart, width='stretch')

                    # Add legend explanation
                    with st.expander("📖 Chart Legend & Interpretation", expanded=False):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown("**🔴 Put Strikes (Red)**")
                            st.write("• Darker = Sooner expiration")
                            st.write("• Sell puts below current price")
                            st.write("• Bullish to neutral strategy")
                            st.write("• Collect premium if stock stays above strike")

                        with col2:
                            st.markdown("**🟢 Call Strikes (Green)**")
                            st.write("• Darker = Sooner expiration")
                            st.write("• Sell calls above current price")
                            st.write("• Bearish to neutral strategy")
                            st.write("• Collect premium if stock stays below strike")

                        with col3:
                            st.markdown("**💙 Expected Move Zone**")
                            st.write("• Blue shaded area")
                            st.write("• ±1 standard deviation range")
                            st.write("• ~68% probability range")
                            st.write("• Strikes outside zone = higher safety")
                else:
                    st.info("Options levels chart not available")
            except Exception as e:
                logger.error(f"Options chart error: {e}")
                st.error(f"Options chart error: {str(e)}")
        
        with tab3:
            st.subheader("Technical Score Components")
            try:
                score_chart = create_technical_score_chart(analysis_results)
                if score_chart:
                    st.plotly_chart(score_chart, width='stretch')
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
                st.dataframe(data.tail(50), width='stretch')
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
