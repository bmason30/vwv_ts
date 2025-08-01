"""
UI components for the VWV Trading System v4.2.1
CRITICAL: Technical score bar must display properly
"""
import streamlit as st

def create_technical_score_bar(score, details=None):
    """Create professional gradient bar for technical score - CRITICAL FUNCTIONALITY"""
    
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
    
    # Create professional gradient bar HTML
    html = f"""
    <div style="margin: 1.5rem 0; padding: 1.5rem; 
                background: linear-gradient(135deg, #1e1e1e 0%, #2d2d30 50%, #1a1a1a 100%); 
                border-radius: 15px; 
                border: 1px solid #404040; 
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div>
                <span style="font-weight: 700; color: #ffffff; font-size: 1.3em;">
                    Technical Composite Score
                </span>
                <div style="font-size: 0.95em; color: #b0b0b0; margin-top: 0.3rem;">
                    Enhanced with Volume & Volatility Analysis
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-weight: 700; color: {primary_color}; font-size: 2.2em;">
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
                    border: 2px solid #404040; 
                    overflow: hidden;">
            
            <div style="position: absolute; left: {score}%; top: 50%; transform: translate(-50%, -50%); 
                        width: 8px; height: 36px; 
                        background: linear-gradient(to bottom, #ffffff 0%, #f0f0f0 100%); 
                        border: 2px solid #1a1a1a; 
                        border-radius: 4px; 
                        box-shadow: 0 3px 6px rgba(0,0,0,0.5); 
                        z-index: 10;">
            </div>
            
            <div style="position: absolute; left: 0; top: 0; height: 100%; width: {score}%; 
                        background: linear-gradient(to right, 
                            transparent 0%, 
                            {primary_color}40 70%, 
                            {primary_color}60 100%); 
                        border-radius: 14px;">
            </div>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 0.7rem; font-size: 0.8em; color: #a0a0a0;">
            <span style="font-weight: 600;">Very Bearish</span>
            <span style="font-weight: 600;">Neutral</span>
            <span style="font-weight: 600;">Very Bullish</span>
        </div>
    </div>
    """
    
    return html

def create_header():
    """Create the main header"""
    st.markdown("""
    <div style="padding: 3rem 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center; 
                background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
                border: 1px solid #404040;">
        <div style="background: rgba(0,0,0,0.3); padding: 2rem; border-radius: 12px;">
            <h1 style="font-size: 2.8rem; margin-bottom: 1rem; color: #ffffff; font-weight: 700;">
                VWV Professional Trading System v4.2.1
            </h1>
            <p style="color: #e0f0e0; margin: 0.5rem 0; font-size: 1.1rem;">
                Enhanced with Volume & Volatility Analysis
            </p>
            <p style="color: #c0d0c0; font-style: italic; font-size: 1rem;">
                <em>Complete Modular Implementation</em>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
