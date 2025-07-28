"""
UI components for the VWV Trading System
"""
import streamlit as st

def create_technical_score_bar(score, details=None):
    """Create professional gradient bar for technical score"""
    
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
    <div style="margin: 1.5rem 0; padding: 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                border-radius: 12px; border: 1px solid #dee2e6; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div>
                <span style="font-weight: 700; color: #495057; font-size: 1.2em;">Technical Composite Score</span>
                <div style="font-size: 0.9em; color: #6c757d; margin-top: 0.2rem;">
                    Aggregated signal from all technical indicators
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-weight: 700; color: {primary_color}; font-size: 2em;">{score}</div>
                <div style="font-size: 0.9em; color: {primary_color}; font-weight: 600;">{interpretation}</div>
            </div>
        </div>
        
        <div style="position: relative; width: 100%; height: 24px; background: linear-gradient(to right, 
                    #DC143C 0%, #FF4500 15%, #FF8C00 30%, #FFD700 50%, 
                    #9ACD32 70%, #32CD32 85%, #00A86B 100%); 
                    border-radius: 12px; border: 1px solid #ced4da; overflow: hidden;">
            
            <!-- Score indicator -->
            <div style="position: absolute; left: {score}%; top: 50%; transform: translate(-50%, -50%); 
                        width: 6px; height: 30px; background: white; border: 2px solid #343a40; 
                        border-radius: 3px; box-shadow: 0 2px 4px rgba(0,0,0,0.3); z-index: 10;">
            </div>
            
            <!-- Progress fill -->
            <div style="position: absolute; left: 0; top: 0; height: 100%; width: {score}%; 
                        background: linear-gradient(to right, transparent 0%, {primary_color} 100%); 
                        opacity: 0.3; border-radius: 12px;">
            </div>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.75em; color: #6c757d;">
            <span style="font-weight: 600;">Very Bearish</span>
            <span style="font-weight: 600;">Neutral</span>
            <span style="font-weight: 600;">Very Bullish</span>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 0.2rem; font-size: 0.7em; color: #adb5bd;">
            <span>1</span>
            <span>25</span>
            <span>50</span>
            <span>75</span>
            <span>100</span>
        </div>
    </div>
    """
    
    return html

def create_header():
    """Create the main header"""
    st.markdown("""
    <div style="position: relative; padding: 3rem 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center; 
                overflow: hidden; min-height: 200px; background: linear-gradient(to bottom, #1a4d3a 0%, #2d6b4f 40%, #1e5540 100%);">
        <div style="position: relative; z-index: 3; background: rgba(0,0,0,0.2); padding: 1.5rem; border-radius: 10px; 
                    backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1);">
            <h1 style="font-size: 2.8rem; margin-bottom: 1rem; color: #ffffff; text-shadow: 2px 2px 8px rgba(0,0,0,0.8); 
                       font-weight: 700; letter-spacing: 1px;">VWV Professional Trading System</h1>
            <p style="color: #f0f8f0; text-shadow: 1px 1px 4px rgba(0,0,0,0.7); margin: 0.5rem 0; font-size: 1.1rem; font-weight: 400;">
                Advanced market analysis with enhanced technical indicators</p>
            <p style="color: #e0f0e0; font-style: italic; font-size: 1rem;">
                <em>Modular Architecture: Complete Implementation</em></p>
        </div>
    </div>
    """, unsafe_allow_html=True)
