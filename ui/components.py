"""
UI components for the VWV Trading System v8.0.0
ENHANCED: Added Volume Composite Score Bar
NEW FEATURES: Module-specific gradient score bars
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
            
            <div style="position: absolute; left: {score}%; top: 50%; transform: translate(-50%, -50%); 
                        width: 4px; height: 32px; 
                        background: #ffffff; 
                        border: 1px solid #1a1a1a; 
                        border-radius: 4px; 
                        box-shadow: 0 2px 5px rgba(0,0,0,0.5); 
                        z-index: 10;">
            </div>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.8em; color: #a0a0a0;">
            <span>0 (Bearish)</span>
            <span>50 (Neutral)</span>
            <span>100 (Bullish)</span>
        </div>
    </div>
    """
    
    return html

def create_volume_score_bar(score, details=None):
    """
    NEW FEATURE v8.0.0: Create professional gradient bar for volume composite score
    
    Similar to technical score bar but optimized for volume analysis
    """
    
    score = round(float(score), 1)

    # Volume-specific interpretation and color mapping
    if score >= 85:
        interpretation = "Extreme Activity"
        primary_color = "#FF1493"  # Deep pink - extreme activity
    elif score >= 75:
        interpretation = "High Activity" 
        primary_color = "#00A86B"  # Jade green - high positive activity
    elif score >= 65:
        interpretation = "Above Normal"
        primary_color = "#32CD32"  # Lime green - above normal
    elif score >= 55:
        interpretation = "Slightly Elevated"
        primary_color = "#9ACD32"  # Yellow green - slightly elevated
    elif score >= 45:
        interpretation = "Normal Volume"
        primary_color = "#FFD700"  # Gold - normal/neutral
    elif score >= 35:
        interpretation = "Below Normal"
        primary_color = "#FF8C00"  # Dark orange - below normal
    elif score >= 25:
        interpretation = "Low Activity"
        primary_color = "#FF4500"  # Orange red - low activity
    else:
        interpretation = "Very Low Activity"
        primary_color = "#DC143C"  # Crimson - very low activity
    
    # Create volume-specific gradient bar HTML
    html = f"""
    <div style="margin-bottom: 1rem;">
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div>
                <span style="font-weight: 700; color: #ffffff; font-size: 1.3em;">
                    📊 Volume Composite Score
                </span>
                <div style="font-size: 0.95em; color: #b0b0b0; margin-top: 0.3rem;">
                    Comprehensive volume analysis with smart money signals
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
                        #9ACD32 70%, #32CD32 85%, #00A86B 95%, #FF1493 100%); 
                    border-radius: 14px; 
                    border: 1px solid #404040; 
                    overflow: hidden;
                    box-shadow: inset 0 2px 4px rgba(0,0,0,0.4);">
            
            <div style="position: absolute; left: {score}%; top: 50%; transform: translate(-50%, -50%); 
                        width: 4px; height: 32px; 
                        background: #ffffff; 
                        border: 1px solid #1a1a1a; 
                        border-radius: 4px; 
                        box-shadow: 0 2px 5px rgba(0,0,0,0.5); 
                        z-index: 10;">
            </div>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.8em; color: #a0a0a0;">
            <span>0 (Low)</span>
            <span>50 (Normal)</span>
            <span>85+ (Extreme)</span>
        </div>
    </div>
    """
    
    return html

def create_header():
    """Create the main header"""
    st.markdown("""
    <div style="padding: 2rem 1rem; border-radius: 15px; margin-bottom: 2rem; text-align: center; 
                background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
                border: 1px solid #404040;">
        <h1 style="font-size: 2.5rem; margin-bottom: 1rem; color: #ffffff; font-weight: 700;">
            VWV Professional Trading System v8.0.0
        </h1>
        <p style="color: #c0d0c0; margin: 0.5rem 0; font-size: 1.1rem;">
            A comprehensive tool for multi-factor market analysis with enhanced volume intelligence.
        </p>
    </div>
    """, unsafe_allow_html=True)
