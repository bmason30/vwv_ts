"""
File: ui/components.py
FIXED UI Components for VWV Trading System v4.2.1 - HTML Rendering Corrected  
Version: v4.2.1-FIXED-HTML-2025-08-27-18-15-00-EST
Corrected HTML rendering issues and restored working gradient bars
Last Updated: August 27, 2025 - 6:15 PM EST
"""
import streamlit as st
import pandas as pd

def create_technical_score_bar(score, details=None):
    """Create professional gradient bar for technical score - FIXED HTML RENDERING"""
    
    score = round(float(score), 1)

    # Determine interpretation and color based on the score
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
    
    # FIXED: Use columns instead of HTML for better compatibility
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Technical Composite Score")
        st.caption("Aggregated signal from all technical indicators")
        
    with col2:
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 2.5em; font-weight: bold; color: {primary_color};">
                {score}
            </div>
            <div style="color: {primary_color}; font-weight: 600;">
                {interpretation}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Progress bar as fallback for gradient
    st.progress(score / 100)
    
    # Score interpretation below
    score_cols = st.columns(5)
    score_labels = ["Very Bearish", "Bearish", "Neutral", "Bullish", "Very Bullish"]
    for i, (col, label) in enumerate(zip(score_cols, score_labels)):
        with col:
            if (i * 20) <= score < ((i + 1) * 20):
                st.markdown(f"**{label}**")
            else:
                st.markdown(f"<small>{label}</small>", unsafe_allow_html=True)

def create_volatility_score_bar(score, regime):
    """Create professional display for volatility score - SIMPLIFIED FOR COMPATIBILITY"""
    
    score = round(float(score), 1)

    # Determine interpretation and color based on volatility score
    if score >= 85:
        interpretation = "Extreme Volatility"
        primary_color = "#DC143C"  # Crimson - extreme volatility
    elif score >= 70:
        interpretation = "High Volatility" 
        primary_color = "#FF4500"  # Orange red - high volatility
    elif score >= 55:
        interpretation = "Above Normal"
        primary_color = "#FF8C00"  # Dark orange - above normal
    elif score >= 45:
        interpretation = "Normal Volatility"
        primary_color = "#FFD700"  # Gold - normal
    elif score >= 30:
        interpretation = "Low Volatility"
        primary_color = "#9ACD32"  # Yellow green - low
    else:
        interpretation = "Very Low Volatility"
        primary_color = "#32CD32"  # Lime green - very low
    
    # Use columns for better compatibility
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Volatility Composite Score")
        st.caption("14 advanced volatility indicators with weighted scoring")
        
    with col2:
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 2.5em; font-weight: bold; color: {primary_color};">
                {score}
            </div>
            <div style="color: {primary_color}; font-weight: 600;">
                {interpretation}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Progress bar for score
    st.progress(score / 100)
    
    # Regime display
    st.info(f"**Current Regime:** {regime}")

def create_volume_score_bar(score, regime):
    """Create professional display for volume score - SIMPLIFIED FOR COMPATIBILITY"""
    
    score = round(float(score), 1)

    # Determine interpretation and color based on volume score
    if score >= 85:
        interpretation = "Extreme Volume"
        primary_color = "#DC143C"  # Crimson - extreme volume
    elif score >= 70:
        interpretation = "High Volume" 
        primary_color = "#FF4500"  # Orange red - high volume
    elif score >= 55:
        interpretation = "Above Normal"
        primary_color = "#FF8C00"  # Dark orange - above normal
    elif score >= 45:
        interpretation = "Normal Volume"
        primary_color = "#FFD700"  # Gold - normal
    elif score >= 30:
        interpretation = "Low Volume"
        primary_color = "#9ACD32"  # Yellow green - low
    else:
        interpretation = "Very Low Volume"
        primary_color = "#32CD32"  # Lime green - very low
    
    # Use columns for better compatibility
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Volume Composite Score")
        st.caption("Volume strength analysis with multi-timeframe confirmation")
        
    with col2:
        st.markdown(f"""
        <div style="text-align: center;">
            <div style="font-size: 2.5em; font-weight: bold; color: {primary_color};">
                {score}
            </div>
            <div style="color: {primary_color}; font-weight: 600;">
                {interpretation}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Progress bar for score
    st.progress(score / 100)
    
    # Regime display
    st.info(f"**Volume Regime:** {regime}")

def create_header():
    """Create VWV system header - SIMPLIFIED"""
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #1e3c72, #2a5298); border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0;">🚀 VWV Professional Trading System v4.2.1</h1>
        <p style="color: #b0c4de; margin: 0.5rem 0 0 0;">Advanced Technical Analysis • Volatility Analysis • Professional Trading Signals</p>
    </div>
    """, unsafe_allow_html=True)

def format_large_number(num):
    """Format large numbers with K, M, B suffixes"""
    if num == 0:
        return "0"
    
    try:
        num = float(num)
        if abs(num) >= 1e9:
            return f"{num/1e9:.1f}B"
        elif abs(num) >= 1e6:
            return f"{num/1e6:.1f}M" 
        elif abs(num) >= 1e3:
            return f"{num/1e3:.1f}K"
        else:
            return f"{num:.0f}"
    except:
        return str(num)

def create_component_breakdown_table(indicators, scores, weights, contributions, title="Component Breakdown"):
    """Create professional component breakdown table for analysis modules"""
    
    # Create component data for display
    component_data = []
    
    for indicator_name in indicators.keys():
        if indicator_name in scores and indicator_name in weights:
            component_data.append({
                'Indicator': indicator_name.replace('_', ' ').title(),
                'Value': f"{indicators[indicator_name]:.2f}",
                'Score': f"{scores[indicator_name]:.1f}/100",
                'Weight': f"{weights[indicator_name]:.3f}",
                'Contribution': f"{contributions.get(indicator_name, 0):.2f}"
            })
    
    if component_data:
        df = pd.DataFrame(component_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("No component data available for breakdown")

def display_regime_classification(score, regime, options_strategy, trading_implications):
    """Display regime classification with professional styling"""
    
    # Determine color scheme based on score
    if score >= 75:
        color = "#DC143C"  # High intensity
    elif score >= 60:
        color = "#FF8C00"  # Medium-high
    elif score >= 40:
        color = "#FFD700"  # Medium
    else:
        color = "#32CD32"  # Lower intensity
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Current Environment:** {regime}")
        
        if options_strategy:
            st.success(f"**Options Strategy:** {options_strategy}")
    
    with col2:
        if trading_implications:
            st.info(f"**Trading Implications:** {trading_implications}")
