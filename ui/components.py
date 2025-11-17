"""
File: ui/components.py v1.0.1
VWV Professional Trading System v4.2.2
UI Components Module - Professional score bars and headers
Created: 2025-10-02
Updated: 2025-10-03
File Version: v1.0.1 - Removed raw dictionary dump from technical score display
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
