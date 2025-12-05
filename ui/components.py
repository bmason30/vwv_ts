"""
File: ui/components.py v1.3.0
VWV Research And Analysis System v4.3.0
UI Components Module - Professional score bars and headers
Created: 2025-10-02
Updated: 2025-11-19
File Version: v1.3.0 - Added strike quality analysis display
Changes in this version:
    - v1.2.0: Added create_options_chart_controls() for interactive filtering
    - v1.2.0: Added display functions for enhanced options analysis
    - v1.3.0: Added display_strike_quality_table() for quality scoring
System Version: v4.3.0 - Black-Scholes Options Pricing
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List

def inject_custom_css():
    """
    Inject custom CSS for institutional command center design
    This is the maximum customization achievable in Streamlit
    """
    st.markdown("""
    <style>
    /* ========================================
       PHASE 1: GLOBAL RESETS & BASE THEME
       ======================================== */

    /* Import Professional Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Root Variables - Dark Card-Based Theme */
    :root {
        /* Updated Background Colors - Darker Theme */
        --bg-primary: #0a0a0a;
        --bg-secondary: #0d0d0d;
        --bg-tertiary: #151515;
        --bg-card: #1a1a1a;
        --bg-input: #1a1a1a;
        --bg-hover: #202020;

        /* Updated Borders */
        --border-subtle: #2a2a2a;
        --border-card: #303030;
        --border-focus: rgba(0, 212, 255, 0.5);

        /* Text Colors */
        --text-primary: #e0e0e0;
        --text-secondary: #a0a0a0;
        --text-tertiary: #707070;
        --text-label: #888888;

        /* Accent Colors - Cyan Blue Theme */
        --accent-cyan: #00d4ff;
        --accent-cyan-hover: #20c5e8;
        --accent-cyan-glow: rgba(0, 212, 255, 0.3);
        --accent-green: #32CD32;
        --accent-yellow: #FFD700;
        --accent-red: #DC143C;

        --font-ui: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', 'Courier New', monospace;
    }

    /* Global Background - Deep Black */
    .stApp {
        background: linear-gradient(to bottom right, var(--bg-primary), var(--bg-secondary)) !important;
        font-family: var(--font-ui) !important;
        color: var(--text-secondary) !important;
    }

    /* Main Content Area */
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }

    /* Remove Default Streamlit Padding */
    .main {
        padding: 0 !important;
    }

    /* ========================================
       PHASE 2: REMOVE VISUAL CLUTTER
       ======================================== */

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ========================================
       PHASE 3: SIDEBAR - THE CONTROL DECK
       ======================================== */

    /* Sidebar Base Styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-subtle) !important;
        padding-top: 0 !important;
    }

    /* Sidebar Content */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--text-secondary) !important;
    }

    /* Sidebar Title - Compact Brand */
    [data-testid="stSidebar"] h1 {
        font-size: 0.875rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em !important;
        color: var(--text-primary) !important;
        padding: 1rem 0 0.5rem 0 !important;
        margin: 0 !important;
        border-bottom: 1px solid var(--border-subtle) !important;
    }

    /* Sidebar Inputs - Compact Style */
    [data-testid="stSidebar"] input {
        background-color: var(--bg-input) !important;
        border: 1px solid var(--border-subtle) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.875rem !important;
        padding: 0.5rem !important;
        border-radius: 0.375rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    [data-testid="stSidebar"] input:focus {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 0 1px var(--border-focus) !important;
        background-color: var(--bg-hover) !important;
    }

    /* Sidebar Select Boxes */
    [data-testid="stSidebar"] [data-baseweb="select"] {
        background-color: var(--bg-input) !important;
        border: 1px solid var(--border-subtle) !important;
        font-size: 0.875rem !important;
    }

    /* Analyze Button - Cyan Blue with Glow */
    [data-testid="stSidebar"] button[kind="primary"] {
        background: var(--accent-cyan) !important;
        border: none !important;
        color: #0a0a0a !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        padding: 0.625rem 1.25rem !important;
        border-radius: 0.375rem !important;
        box-shadow: 0 0 15px var(--accent-cyan-glow) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stSidebar"] button[kind="primary"]:hover {
        background: var(--accent-cyan-hover) !important;
        box-shadow: 0 0 20px var(--accent-cyan-glow) !important;
        transform: translateY(-1px) !important;
    }

    /* Sidebar Checkboxes - Minimal Design */
    [data-testid="stSidebar"] [data-testid="stCheckbox"] {
        font-size: 0.813rem !important;
        color: var(--text-secondary) !important;
    }

    /* Sidebar Expanders - Flat Design */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background-color: transparent !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 0.375rem !important;
        margin: 0.5rem 0 !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        font-size: 0.813rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* ========================================
       PHASE 4: MAIN CONTENT - PRECISION LAYOUT
       ======================================== */

    /* Section Headers - Subtle, Professional */
    h2, h3 {
        font-family: var(--font-ui) !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        font-size: 1.25rem !important;
        margin-top: 2rem !important;
        margin-bottom: 0.75rem !important;
        letter-spacing: -0.025em !important;
    }

    h3 {
        font-size: 0.875rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: var(--text-tertiary) !important;
        font-weight: 500 !important;
    }

    /* Expanders - Card-Based Design with Shadow */
    [data-testid="stExpander"] {
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border-card) !important;
        border-radius: 10px !important;
        margin: 16px 0 !important;
        overflow: hidden !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
    }

    [data-testid="stExpander"] summary {
        background-color: var(--bg-tertiary) !important;
        border-bottom: 1px solid var(--border-card) !important;
        padding: 16px 20px !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    [data-testid="stExpander"] summary:hover {
        background-color: var(--bg-hover) !important;
    }

    /* Expander Content */
    [data-testid="stExpander"] > div:last-child {
        background-color: var(--bg-card) !important;
        padding: 20px !important;
    }

    /* ========================================
       PHASE 5: DATA DISPLAY - MONOSPACE PRECISION
       ======================================== */

    /* All Numbers and Financial Data */
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"],
    .stMetric {
        font-family: var(--font-mono) !important;
    }

    /* Metric Values */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 500 !important;
        color: var(--text-primary) !important;
    }

    /* Metric Labels */
    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: var(--text-tertiary) !important;
        font-weight: 500 !important;
    }

    /* DataFrames - Clean Grid */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-subtle) !important;
        border-radius: 0.5rem !important;
        overflow: hidden !important;
    }

    [data-testid="stDataFrame"] table {
        background-color: var(--bg-tertiary) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.813rem !important;
    }

    [data-testid="stDataFrame"] thead {
        background-color: var(--bg-secondary) !important;
        border-bottom: 1px solid var(--border-subtle) !important;
    }

    [data-testid="stDataFrame"] th {
        color: var(--text-tertiary) !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.05em !important;
        padding: 0.75rem !important;
    }

    [data-testid="stDataFrame"] td {
        color: var(--text-secondary) !important;
        padding: 0.625rem 0.75rem !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* ========================================
       PHASE 6: CHARTS - MAXIMUM CONTRAST
       ======================================== */

    /* Plotly Charts */
    .js-plotly-plot {
        border: 1px solid var(--border-subtle) !important;
        border-radius: 0.75rem !important;
        overflow: hidden !important;
        background-color: var(--bg-tertiary) !important;
    }

    /* Chart Modebar (Plotly controls) */
    .modebar {
        background-color: var(--bg-secondary) !important;
        border-radius: 0.375rem !important;
        padding: 0.25rem !important;
    }

    /* ========================================
       PHASE 7: ALERTS & NOTIFICATIONS
       ======================================== */

    /* Success Messages */
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 0.5rem !important;
        color: var(--accent-green) !important;
        font-size: 0.875rem !important;
    }

    /* Error Messages */
    .stError {
        background-color: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 0.5rem !important;
        color: var(--accent-red) !important;
        font-size: 0.875rem !important;
    }

    /* Info Messages */
    .stInfo {
        background-color: rgba(0, 212, 255, 0.1) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 0.5rem !important;
        color: var(--accent-cyan) !important;
        font-size: 0.875rem !important;
    }

    /* Warning Messages */
    .stWarning {
        background-color: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid rgba(245, 158, 11, 0.3) !important;
        border-radius: 0.5rem !important;
        color: #f59e0b !important;
        font-size: 0.875rem !important;
    }

    /* ========================================
       PHASE 8: RESPONSIVE ADJUSTMENTS
       ======================================== */

    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }

    /* ========================================
       PHASE 9: ANIMATION & INTERACTIONS
       ======================================== */

    /* Smooth Transitions */
    * {
        transition: background-color 0.2s ease, border-color 0.2s ease !important;
    }

    /* Hover States for Interactive Elements */
    button:hover,
    [data-testid="stExpander"] summary:hover {
        cursor: pointer !important;
    }

    /* Focus States */
    input:focus,
    select:focus,
    textarea:focus {
        outline: none !important;
    }

    /* Loading Indicator */
    .stSpinner > div {
        border-color: var(--accent-cyan) transparent transparent transparent !important;
    }

    /* ========================================
       PHASE 10: CUSTOM UTILITY CLASSES
       ======================================== */

    /* For custom HTML injections */
    .terminal-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.875rem;
        font-weight: 700;
        color: white;
    }

    .terminal-badge-dot {
        width: 0.75rem;
        height: 0.75rem;
        background-color: var(--accent-cyan);
        border-radius: 0.125rem;
        box-shadow: 0 0 10px var(--accent-cyan-glow);
    }

    .section-divider {
        height: 1px;
        background: linear-gradient(to right, transparent, var(--border-subtle), transparent);
        margin: 2rem 0;
    }

    /* Accent Border Top for Cards */
    .accent-border-top {
        border-top: 2px solid transparent;
        border-image: linear-gradient(to right, var(--accent-cyan), var(--accent-cyan-hover)) 1;
    }

    /* Navigation Radio Buttons - Cyan Accent */
    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        background-color: transparent !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 0.375rem !important;
        padding: 0.5rem 1rem !important;
        margin: 0.25rem 0 !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
        background-color: var(--bg-hover) !important;
        border-color: var(--accent-cyan) !important;
    }

    [data-testid="stSidebar"] [data-baseweb="radio"] input:checked + label {
        background-color: var(--accent-cyan) !important;
        border-color: var(--accent-cyan) !important;
        color: #0a0a0a !important;
        font-weight: 600 !important;
    }

    /* Pulse animation for live indicators */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    </style>
    """, unsafe_allow_html=True)

def display_vwv_logo():
    """
    Display VWV logo in sidebar

    PLACEHOLDER: Replace with actual logo image when available
    Options to replace:
    1. Use st.sidebar.image("path/to/logo.png", width=240)
    2. Use base64 encoded image in HTML
    3. Use hosted URL: st.sidebar.image("https://example.com/logo.png", width=240)
    """
    # OPTION 1: Professional text-based placeholder (current)
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 2rem 1rem 1.5rem 1rem;">
        <div style="font-size: 2.5rem; font-weight: 700;
                    background: linear-gradient(135deg, #00d4ff 0%, #20c5e8 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    margin-bottom: 0.5rem;
                    letter-spacing: 0.1em;">
            VWV
        </div>
        <div style="font-size: 0.75rem; color: #888888; font-weight: 500; letter-spacing: 0.1em;">
            RESEARCH & ANALYSIS
        </div>
    </div>
    """, unsafe_allow_html=True)

    # OPTION 2: To use actual image file (uncomment and modify when ready):
    # st.sidebar.image("assets/vwv_logo.png", width=240)

    # OPTION 3: To use base64 encoded image (uncomment and add base64 string):
    # st.sidebar.markdown(f"""
    # <div style="text-align: center; padding: 1rem;">
    #     <img src="data:image/png;base64,{BASE64_STRING_HERE}" width="240"/>
    # </div>
    # """, unsafe_allow_html=True)

def create_command_center_header():
    """
    Create institutional command center header
    Replaces the large green centered title
    """
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between;
                padding: 1rem 0; margin-bottom: 2rem;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);">
        <div class="terminal-badge">
            <div class="terminal-badge-dot"></div>
            <span style="color: white;">VWV</span>
            <span style="color: #6b7280; font-weight: 400;">TERMINAL</span>
        </div>
        <div style="font-size: 0.75rem; color: #6b7280; font-family: 'JetBrains Mono', monospace;">
            RESEARCH &amp; ANALYSIS SYSTEM
        </div>
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <div style="width: 8px; height: 8px; background-color: #10b981;
                            border-radius: 50%; animation: pulse 2s infinite;"></div>
                <span style="font-size: 0.75rem; color: #10b981; font-weight: 500;">LIVE</span>
            </div>
            <span style="font-size: 0.75rem; color: #6b7280;">v1.0.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
    
    # Create enhanced professional HTML score bar with modern aesthetics
    html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        .score-container {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 16px;
            padding: 24px;
            margin: 12px 0;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4), 0 2px 4px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.05);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        .score-header {{
            color: #ffffff;
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 18px;
            text-align: center;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            font-size: 14px;
            opacity: 0.95;
        }}
        .score-bar-background {{
            background: linear-gradient(180deg, #1f1f2e 0%, #2a2a3e 100%);
            border-radius: 12px;
            height: 48px;
            position: relative;
            overflow: hidden;
            box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.4), inset 0 -1px 2px rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(0, 0, 0, 0.3);
        }}
        .score-bar-fill {{
            background: linear-gradient(135deg, {color} 0%, {color}ee 50%, {color} 100%);
            height: 100%;
            width: {composite_score}%;
            border-radius: 11px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            box-shadow: 0 2px 8px rgba({color.replace('#', '')}, 0.3);
        }}
        .score-bar-fill::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 50%;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.2) 0%, transparent 100%);
            border-radius: 11px 11px 0 0;
        }}
        .score-text {{
            color: #ffffff;
            font-size: 24px;
            font-weight: 700;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.6), 0 1px 2px rgba(0, 0, 0, 0.8);
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            letter-spacing: 0.5px;
            z-index: 10;
        }}
        .score-label {{
            color: #e0e0e0;
            font-size: 15px;
            font-weight: 600;
            text-align: center;
            margin-top: 14px;
            letter-spacing: 0.3px;
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

    # Create enhanced professional HTML score bar with modern aesthetics
    html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        .fund-score-container {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 16px;
            padding: 24px;
            margin: 12px 0;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4), 0 2px 4px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.05);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        .fund-score-header {{
            color: #ffffff;
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 18px;
            text-align: center;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            font-size: 14px;
            opacity: 0.95;
        }}
        .fund-score-bar-background {{
            background: linear-gradient(180deg, #1f1f2e 0%, #2a2a3e 100%);
            border-radius: 12px;
            height: 48px;
            position: relative;
            overflow: hidden;
            box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.4), inset 0 -1px 2px rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(0, 0, 0, 0.3);
        }}
        .fund-score-bar-fill {{
            background: linear-gradient(135deg, {color} 0%, {color}ee 50%, {color} 100%);
            height: 100%;
            width: {composite_score}%;
            border-radius: 11px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            box-shadow: 0 2px 8px rgba({color.replace('#', '')}, 0.3);
        }}
        .fund-score-bar-fill::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 50%;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.2) 0%, transparent 100%);
            border-radius: 11px 11px 0 0;
        }}
        .fund-score-text {{
            color: #ffffff;
            font-size: 24px;
            font-weight: 700;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.6), 0 1px 2px rgba(0, 0, 0, 0.8);
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            letter-spacing: 0.5px;
            z-index: 10;
        }}
        .fund-score-label {{
            color: #e0e0e0;
            font-size: 15px;
            font-weight: 600;
            text-align: center;
            margin-top: 14px;
            letter-spacing: 0.3px;
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

    # Create enhanced professional HTML score bar with modern aesthetics
    html = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        .master-score-container {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 16px;
            padding: 24px;
            margin: 12px 0;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4), 0 2px 4px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.05);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}
        .master-score-header {{
            color: #ffffff;
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 18px;
            text-align: center;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            font-size: 14px;
            opacity: 0.95;
        }}
        .master-score-bar-background {{
            background: linear-gradient(180deg, #1f1f2e 0%, #2a2a3e 100%);
            border-radius: 12px;
            height: 48px;
            position: relative;
            overflow: hidden;
            box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.4), inset 0 -1px 2px rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(0, 0, 0, 0.3);
        }}
        .master-score-bar-fill {{
            background: linear-gradient(135deg, {color} 0%, {color}ee 50%, {color} 100%);
            height: 100%;
            width: {composite_score}%;
            border-radius: 11px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            box-shadow: 0 2px 8px rgba({color.replace('#', '')}, 0.3);
        }}
        .master-score-bar-fill::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 50%;
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.2) 0%, transparent 100%);
            border-radius: 11px 11px 0 0;
        }}
        .master-score-text {{
            color: #ffffff;
            font-size: 24px;
            font-weight: 700;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.6), 0 1px 2px rgba(0, 0, 0, 0.8);
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            letter-spacing: 0.5px;
            z-index: 10;
        }}
        .master-score-label {{
            color: #e0e0e0;
            font-size: 15px;
            font-weight: 600;
            text-align: center;
            margin-top: 14px;
            letter-spacing: 0.3px;
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
                VWV Research And Analysis System v1.0.0
            </h1>
            <p style='color: #888; margin: 5px 0;'>
                Advanced Technical Analysis • Volatility Analysis • Professional Market Analysis
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

def display_strike_quality_table(options_data: List[Dict], current_price: float, volatility: float):
    """
    Display comprehensive strike quality analysis with scores and recommendations

    VERSION 1.3.0 - Advanced strike quality scoring display

    Parameters:
    -----------
    options_data : list - Options data with strikes and Greeks
    current_price : float - Current stock price
    volatility : float - Current volatility (for IV rank estimation)
    """

    from analysis.options import calculate_strike_quality_score

    st.subheader("⭐ Strike Quality Analysis")
    st.write("**Comprehensive scoring based on premium, probability, Greeks, and position**")

    # Prepare data for display
    analysis_data = []

    for opt in options_data:
        dte = opt['DTE']

        # Analyze put
        put_data = {
            'Strike': opt['Put Strike'],
            'PoP': opt.get('Put PoP', '0%'),
            'Delta': opt.get('Put Delta', '0'),
            'Theta': opt.get('Put Theta', '$0'),
            'Premium': opt.get('Put Premium', '$0')
        }
        put_score = calculate_strike_quality_score(put_data, current_price, volatility)

        # Analyze call
        call_data = {
            'Strike': opt['Call Strike'],
            'PoP': opt.get('Call PoP', '0%'),
            'Delta': opt.get('Call Delta', '0'),
            'Theta': opt.get('Call Theta', '$0'),
            'Premium': opt.get('Call Premium', '$0')
        }
        call_score = calculate_strike_quality_score(call_data, current_price, volatility)

        # Put row
        analysis_data.append({
            'Type': '🔻 PUT',
            'DTE': dte,
            'Strike': f"${opt['Put Strike']:.2f}",
            'Premium': opt.get('Put Premium', 'N/A'),
            'PoP': opt.get('Put PoP', 'N/A'),
            'Score': put_score['total_score'],
            'Rating': put_score['rating'],
            'Stars': '⭐' * put_score['stars'],
            'Recommendation': put_score['recommendation']
        })

        # Call row
        analysis_data.append({
            'Type': '🔺 CALL',
            'DTE': dte,
            'Strike': f"${opt['Call Strike']:.2f}",
            'Premium': opt.get('Call Premium', 'N/A'),
            'PoP': opt.get('Call PoP', 'N/A'),
            'Score': call_score['total_score'],
            'Rating': call_score['rating'],
            'Stars': '⭐' * call_score['stars'],
            'Recommendation': call_score['recommendation']
        })

    # Convert to DataFrame and sort by score
    df_analysis = pd.DataFrame(analysis_data)
    df_analysis = df_analysis.sort_values('Score', ascending=False)

    # Style the DataFrame
    def highlight_score(val):
        """Apply color coding based on score column"""
        if val >= 80:
            return 'background-color: rgba(52, 199, 89, 0.2)'
        elif val >= 65:
            return 'background-color: rgba(52, 199, 150, 0.15)'
        elif val >= 50:
            return 'background-color: rgba(255, 204, 0, 0.1)'
        else:
            return 'background-color: rgba(255, 59, 48, 0.1)'

    # Display with styling
    styled_df = df_analysis.style.applymap(highlight_score, subset=['Score'])

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        height=400
    )

    # Top recommendations
    st.markdown("---")
    st.subheader("🎯 Top 3 Recommendations")
    top_3 = df_analysis.head(3)

    for idx, row in top_3.iterrows():
        col1, col2, col3 = st.columns([2, 1, 3])

        with col1:
            st.metric(f"{row['Type']} - {row['DTE']}D", row['Strike'])

        with col2:
            st.write(f"**{row['Stars']}**")
            st.write(f"Score: {row['Score']}/100")

        with col3:
            # Color the recommendation based on score
            score = row['Score']
            if score >= 80:
                st.success(row['Recommendation'])
            elif score >= 65:
                st.info(row['Recommendation'])
            elif score >= 50:
                st.warning(row['Recommendation'])
            else:
                st.error(row['Recommendation'])
