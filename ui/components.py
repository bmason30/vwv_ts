"""
File: ui/components.py v1.2.0
VWV Professional Trading System v4.2.2
UI Components Module - Professional score bars and headers
Created: 2025-10-02
Updated: 2025-11-19
File Version: v1.2.0 - Added interactive options chart controls
Changes in this version:
    - Added create_options_chart_controls() for interactive filtering
    - Added display functions for enhanced options analysis
System Version: v4.2.2 - Advanced Options with Fibonacci Integration
"""

import streamlit as st
from typing import Dict, Any

def create_technical_score_bar(composite_score: float, score_details: Dict[str, Any] = None) -> str:
    """
    Create professional HTML score bar for technical analysis
    
    Args:
        composite_score: Overall technical score (0-100)
        score_details: Dictionary of component scores (not displayed in main bar)
    
    Returns:
        HTML string for the score bar
    """
    # Determine color based on score
    if composite_score >= 70:
        color = "#4CAF50"  # Green
        label = "Bullish"
    elif composite_score >= 50:
        color = "#FFC107"  # Yellow/Amber
        label = "Neutral"
    else:
        color = "#F44336"  # Red
        label = "Bearish"
    
    # Create clean, professional HTML score bar (NO raw data dump)
    html = f"""
    <style>
        .score-container {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        .score-header {{
            color: #ffffff;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            text-align: center;
        }}
        .score-bar-background {{
            background: #2a2a3e;
            border-radius: 8px;
            height: 40px;
            position: relative;
            overflow: hidden;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
        }}
        .score-bar-fill {{
            background: linear-gradient(90deg, {color} 0%, {color}dd 100%);
            height: 100%;
            width: {composite_score}%;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: width 0.3s ease;
            position: relative;
        }}
        .score-text {{
            color: #ffffff;
            font-size: 20px;
            font-weight: 700;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
        }}
        .score-label {{
            color: #cccccc;
            font-size: 14px;
            text-align: center;
            margin-top: 10px;
        }}
    </style>
    <div class="score-container">
        <div class="score-header">Composite Technical Score</div>
        <div class="score-bar-background">
            <div class="score-bar-fill">
                <span class="score-text">{composite_score:.1f}</span>
            </div>
        </div>
        <div class="score-label">{label} Signal</div>
    </div>
    """
    
    return html

def create_fundamental_score_bar(composite_score: float, score_details: Dict[str, Any] = None) -> str:
    """
    Create professional HTML score bar for fundamental analysis

    Args:
        composite_score: Overall fundamental score (0-100)
        score_details: Dictionary of component scores (not displayed in main bar)

    Returns:
        HTML string for the score bar
    """
    # Determine color and label based on score
    if composite_score >= 70:
        color = "#4CAF50"  # Green
        label = "Strong"
    elif composite_score >= 50:
        color = "#FFC107"  # Yellow/Amber
        label = "Moderate"
    else:
        color = "#F44336"  # Red
        label = "Weak"

    # Create clean, professional HTML score bar
    html = f"""
    <style>
        .fund-score-container {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        .fund-score-header {{
            color: #ffffff;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            text-align: center;
        }}
        .fund-score-bar-background {{
            background: #2a2a3e;
            border-radius: 8px;
            height: 40px;
            position: relative;
            overflow: hidden;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
        }}
        .fund-score-bar-fill {{
            background: linear-gradient(90deg, {color} 0%, {color}dd 100%);
            height: 100%;
            width: {composite_score}%;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: width 0.3s ease;
            position: relative;
        }}
        .fund-score-text {{
            color: #ffffff;
            font-size: 20px;
            font-weight: 700;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
        }}
        .fund-score-label {{
            color: #cccccc;
            font-size: 14px;
            text-align: center;
            margin-top: 10px;
        }}
    </style>
    <div class="fund-score-container">
        <div class="fund-score-header">Composite Fundamental Score</div>
        <div class="fund-score-bar-background">
            <div class="fund-score-bar-fill">
                <span class="fund-score-text">{composite_score:.1f}</span>
            </div>
        </div>
        <div class="fund-score-label">{label} Fundamentals</div>
    </div>
    """

    return html

def create_master_score_bar(composite_score: float, interpretation: str = None, signal_strength: str = None) -> str:
    """
    Create professional HTML score bar for master score (unified analysis)

    Args:
        composite_score: Overall master score (0-100)
        interpretation: Score interpretation text
        signal_strength: Signal strength description

    Returns:
        HTML string for the score bar
    """
    # Determine color and label based on score
    if composite_score >= 70:
        color = "#4CAF50"  # Green
        label = "Strong Bullish"
    elif composite_score >= 60:
        color = "#8BC34A"  # Light Green
        label = "Bullish"
    elif composite_score >= 50:
        color = "#FFC107"  # Yellow/Amber
        label = "Neutral"
    elif composite_score >= 40:
        color = "#FF9800"  # Orange
        label = "Bearish"
    else:
        color = "#F44336"  # Red
        label = "Strong Bearish"

    # Use interpretation if provided, otherwise use label
    display_label = interpretation if interpretation else label

    # Create clean, professional HTML score bar matching technical/fundamental style
    html = f"""
    <style>
        .master-score-container {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        .master-score-header {{
            color: #ffffff;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            text-align: center;
        }}
        .master-score-bar-background {{
            background: #2a2a3e;
            border-radius: 8px;
            height: 40px;
            position: relative;
            overflow: hidden;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
        }}
        .master-score-bar-fill {{
            background: linear-gradient(90deg, {color} 0%, {color}dd 100%);
            height: 100%;
            width: {composite_score}%;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: width 0.3s ease;
            position: relative;
        }}
        .master-score-text {{
            color: #ffffff;
            font-size: 20px;
            font-weight: 700;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
        }}
        .master-score-label {{
            color: #cccccc;
            font-size: 14px;
            text-align: center;
            margin-top: 10px;
        }}
    </style>
    <div class="master-score-container">
        <div class="master-score-header">Unified Master Score</div>
        <div class="master-score-bar-background">
            <div class="master-score-bar-fill">
                <span class="master-score-text">{composite_score:.1f}</span>
            </div>
        </div>
        <div class="master-score-label">{display_label}</div>
    </div>
    """

    return html

def create_header() -> None:
    """
    Create professional header for the trading system
    """
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='color: #4CAF50; margin: 0;'>
                VWV Professional Trading System v4.2.2
            </h1>
            <p style='color: #888; margin: 5px 0;'>
                Advanced Technical Analysis • Volatility Analysis • Professional Trading Signals
            </p>
        </div>
        <hr style='border: 1px solid #333; margin: 20px 0;'>
    """, unsafe_allow_html=True)

def create_options_chart_controls() -> Dict[str, Any]:
    """
    Create interactive controls for options chart filtering

    VERSION 1.2.0 - New function for enhanced options visualization

    Returns:
        dict: User selections for chart customization including:
            - show_puts: bool
            - show_calls: bool
            - show_expected_move: bool
            - show_annotations: bool
            - selected_dtes: list
    """
    st.subheader("📊 Chart Display Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        show_puts = st.checkbox("Show Put Strikes", value=True, key="opt_show_puts",
                               help="Display put option strike levels (red lines)")
        show_calls = st.checkbox("Show Call Strikes", value=True, key="opt_show_calls",
                                help="Display call option strike levels (green lines)")

    with col2:
        show_expected_move = st.checkbox("Show Expected Move Zone", value=True, key="opt_expected_move",
                                        help="Display ±1 standard deviation price range (blue shaded area)")
        show_annotations = st.checkbox("Show Strike Details", value=True, key="opt_annotations",
                                      help="Show detailed Greeks and probabilities for each strike")

    with col3:
        selected_dtes = st.multiselect(
            "Select DTEs (Days to Expiration)",
            options=[7, 14, 30, 45, 60],
            default=[7, 14, 30, 45],
            key="opt_selected_dtes",
            help="Filter which expiration dates to display"
        )

    return {
        'show_puts': show_puts,
        'show_calls': show_calls,
        'show_expected_move': show_expected_move,
        'show_annotations': show_annotations,
        'selected_dtes': selected_dtes if selected_dtes else [7, 14, 30, 45, 60]
    }
