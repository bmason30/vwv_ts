"""
UI components for the VWV Trading System v4.2.1
ENHANCED: Added volatility score bar for comprehensive analysis display
Date: August 21, 2025 - 3:50 PM EST
"""
import streamlit as st

def create_technical_score_bar(score, details=None):
    """Create professional gradient bar for technical score - CRITICAL FUNCTIONALITY"""
    
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
    
    # Create professional gradient bar HTML
    html = f"""
    <div style="margin-bottom: 1rem;">
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div>
                <span style="font-weight: 700; color: #ffffff; font-size: 1.3em;">
                    Technical Composite Score
                </span>
                <div style="font-size: 0.95em; color: #b0b0b0; margin-top: 0.3rem;">
                    Aggregated signal from all technical indicators
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-weight: 700; color: {primary_color}; font-size: 2.2em; line-height: 1;">
                    {score}
                </div>
                <div style="font-size: 0.95em; color: {primary_color}; font-weight: 600;">
                    {interpretation}
                </div>
            </div>
        </div>
        
        <div style="position: relative; width: 100%; height: 28px; 
                    background: linear-gradient(to right, 
                        #DC143C 0%, #FF4500 15%, #FF8C00 30%, #FFD700 50%, 
                        #9ACD32 70%, #32CD32 85%, #00A86B 100%); 
                    border-radius: 14px; 
                    border: 1px solid #404040; 
                    overflow: hidden;
                    box-shadow: inset 0 2px 4px rgba(0,0,0,0.4);">
            
            <div style="position: absolute; left: {score}%; top: 50%; 
                        transform: translateX(-50%) translateY(-50%); 
                        width: 4px; height: 36px; 
                        background: #ffffff; 
                        border-radius: 2px; 
                        box-shadow: 0 0 6px rgba(0,0,0,0.8);
                        border: 1px solid #000000;"></div>
        </div>
        
        <div style="display: flex; justify-content: space-between; 
                    margin-top: 0.5rem; font-size: 0.75em; color: #999;">
            <span>0</span>
            <span>25</span>
            <span>50</span>
            <span>75</span>
            <span>100</span>
        </div>
    </div>
    """
    
    return html

def create_volatility_score_bar(score, details=None):
    """Create professional gradient bar for volatility score"""
    
    score = round(float(score), 1)

    # Determine interpretation and color based on volatility score
    if score >= 80:
        interpretation = "Extreme Volatility"
        primary_color = "#DC143C"  # Crimson for extreme volatility
    elif score >= 65:
        interpretation = "High Volatility" 
        primary_color = "#FF4500"  # Orange red
    elif score >= 55:
        interpretation = "Elevated Volatility"
        primary_color = "#FF8C00"  # Dark orange
    elif score >= 45:
        interpretation = "Normal Volatility"
        primary_color = "#FFD700"  # Gold
    elif score >= 35:
        interpretation = "Low Volatility"
        primary_color = "#9ACD32"  # Yellow green
    elif score >= 20:
        interpretation = "Very Low Volatility"
        primary_color = "#32CD32"  # Lime green
    else:
        interpretation = "Extremely Low Volatility"
        primary_color = "#00A86B"  # Jade green
    
    # Create professional gradient bar HTML
    html = f"""
    <div style="margin-bottom: 1rem;">
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div>
                <span style="font-weight: 700; color: #ffffff; font-size: 1.3em;">
                    Volatility Composite Score
                </span>
                <div style="font-size: 0.95em; color: #b0b0b0; margin-top: 0.3rem;">
                    Weighted analysis from 14 volatility indicators
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-weight: 700; color: {primary_color}; font-size: 2.2em; line-height: 1;">
                    {score}
                </div>
                <div style="font-size: 0.95em; color: {primary_color}; font-weight: 600;">
                    {interpretation}
                </div>
            </div>
        </div>
        
        <div style="position: relative; width: 100%; height: 28px; 
                    background: linear-gradient(to right, 
                        #00A86B 0%, #32CD32 15%, #9ACD32 30%, #FFD700 50%, 
                        #FF8C00 70%, #FF4500 85%, #DC143C 100%); 
                    border-radius: 14px; 
                    border: 1px solid #404040; 
                    overflow: hidden;
                    box-shadow: inset 0 2px 4px rgba(0,0,0,0.4);">
            
            <div style="position: absolute; left: {score}%; top: 50%; 
                        transform: translateX(-50%) translateY(-50%); 
                        width: 4px; height: 36px; 
                        background: #ffffff; 
                        border-radius: 2px; 
                        box-shadow: 0 0 6px rgba(0,0,0,0.8);
                        border: 1px solid #000000;"></div>
        </div>
        
        <div style="display: flex; justify-content: space-between; 
                    margin-top: 0.5rem; font-size: 0.75em; color: #999;">
            <span>Very Low</span>
            <span>Low</span>
            <span>Normal</span>
            <span>High</span>
            <span>Extreme</span>
        </div>
    </div>
    """
    
    return html

def create_volume_score_bar(score, details=None):
    """Create professional gradient bar for volume score"""
    
    score = round(float(score), 1)

    # Determine interpretation and color based on volume score
    if score >= 80:
        interpretation = "Extreme Volume"
        primary_color = "#00A86B"  # Jade green for high volume
    elif score >= 65:
        interpretation = "High Volume" 
        primary_color = "#32CD32"  # Lime green
    elif score >= 55:
        interpretation = "Above Normal Volume"
        primary_color = "#9ACD32"  # Yellow green
    elif score >= 45:
        interpretation = "Normal Volume"
        primary_color = "#FFD700"  # Gold
    elif score >= 35:
        interpretation = "Below Normal Volume"
        primary_color = "#FF8C00"  # Dark orange
    elif score >= 20:
        interpretation = "Low Volume"
        primary_color = "#FF4500"  # Orange red
    else:
        interpretation = "Very Low Volume"
        primary_color = "#DC143C"  # Crimson
    
    # Create professional gradient bar HTML
    html = f"""
    <div style="margin-bottom: 1rem;">
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div>
                <span style="font-weight: 700; color: #ffffff; font-size: 1.3em;">
                    Volume Composite Score
                </span>
                <div style="font-size: 0.95em; color: #b0b0b0; margin-top: 0.3rem;">
                    Weighted analysis from 14 volume indicators
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-weight: 700; color: {primary_color}; font-size: 2.2em; line-height: 1;">
                    {score}
                </div>
                <div style="font-size: 0.95em; color: {primary_color}; font-weight: 600;">
                    {interpretation}
                </div>
            </div>
        </div>
        
        <div style="position: relative; width: 100%; height: 28px; 
                    background: linear-gradient(to right, 
                        #DC143C 0%, #FF4500 15%, #FF8C00 30%, #FFD700 50%, 
                        #9ACD32 70%, #32CD32 85%, #00A86B 100%); 
                    border-radius: 14px; 
                    border: 1px solid #404040; 
                    overflow: hidden;
                    box-shadow: inset 0 2px 4px rgba(0,0,0,0.4);">
            
            <div style="position: absolute; left: {score}%; top: 50%; 
                        transform: translateX(-50%) translateY(-50%); 
                        width: 4px; height: 36px; 
                        background: #ffffff; 
                        border-radius: 2px; 
                        box-shadow: 0 0 6px rgba(0,0,0,0.8);
                        border: 1px solid #000000;"></div>
        </div>
        
        <div style="display: flex; justify-content: space-between; 
                    margin-top: 0.5rem; font-size: 0.75em; color: #999;">
            <span>Very Low</span>
            <span>Low</span>
            <span>Normal</span>
            <span>High</span>
            <span>Extreme</span>
        </div>
    </div>
    """
    
    return html

def create_header():
    """Create application header - WORKING VERSION"""
    st.set_page_config(
        page_title="VWV Professional Trading System v4.2.1",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem; margin-bottom: 2rem; 
                background: linear-gradient(90deg, #1f4e79 0%, #2d5aa0 50%, #1f4e79 100%); 
                border-radius: 10px; border: 1px solid #4CAF50;">
        <h1 style="color: white; margin: 0; font-size: 2.5rem; font-weight: 700;">
            📊 VWV Professional Trading System
        </h1>
        <p style="color: #b8c6db; margin: 0.5rem 0 0 0; font-size: 1.1rem; font-weight: 500;">
            v4.2.1 Enhanced • Advanced Technical Analysis • Volume & Volatility Analysis • Options Strategies
        </p>
    </div>
    """, unsafe_allow_html=True)
