"""
File: ui/components.py
Enhanced UI Components for VWV Trading System v4.2.1 with Volatility Support
Version: v4.2.1-VOLATILITY-ENHANCED-2025-08-27-17-30-00-EST
Professional gradient score bars and display components for all analysis types
Last Updated: August 27, 2025 - 5:30 PM EST
"""
import streamlit as st
import pandas as pd

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
            
            <div style="position: absolute; top: 0; left: {score}%; 
                        width: 3px; height: 100%; 
                        background: white; 
                        box-shadow: 0 0 6px rgba(255,255,255,0.8);
                        transform: translateX(-1.5px);">
            </div>
            
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.8em; color: #888;">
            <span>0</span>
            <span>50</span>
            <span>100</span>
        </div>
        
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

def create_volatility_score_bar(score, regime):
    """Create professional gradient bar for volatility score - NEW VOLATILITY SUPPORT"""
    
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
    
    # Create professional volatility gradient bar HTML
    html = f"""
    <div style="margin-bottom: 1rem;">
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div>
                <span style="font-weight: 700; color: #ffffff; font-size: 1.3em;">
                    Volatility Composite Score
                </span>
                <div style="font-size: 0.95em; color: #b0b0b0; margin-top: 0.3rem;">
                    14 advanced volatility indicators with weighted scoring
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
                        #32CD32 0%, #9ACD32 20%, #FFD700 40%, 
                        #FF8C00 60%, #FF4500 80%, #DC143C 100%); 
                    border-radius: 14px; 
                    border: 1px solid #404040; 
                    overflow: hidden;
                    box-shadow: inset 0 2px 4px rgba(0,0,0,0.4);">
            
            <div style="position: absolute; top: 0; left: {score}%; 
                        width: 3px; height: 100%; 
                        background: white; 
                        box-shadow: 0 0 6px rgba(255,255,255,0.8);
                        transform: translateX(-1.5px);">
            </div>
            
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.8em; color: #888;">
            <span>Very Low</span>
            <span>Normal</span>
            <span>High</span>
            <span>Extreme</span>
        </div>
        
        <div style="text-align: center; margin-top: 0.5rem; padding: 0.5rem; background: {primary_color}20; border-radius: 8px; border-left: 4px solid {primary_color};">
            <strong style="color: {primary_color};">{regime}</strong>
        </div>
        
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

def create_volume_score_bar(score, regime):
    """Create professional gradient bar for volume score - VOLUME SUPPORT"""
    
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
    
    # Create professional volume gradient bar HTML
    html = f"""
    <div style="margin-bottom: 1rem;">
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div>
                <span style="font-weight: 700; color: #ffffff; font-size: 1.3em;">
                    Volume Composite Score
                </span>
                <div style="font-size: 0.95em; color: #b0b0b0; margin-top: 0.3rem;">
                    Volume strength analysis with multi-timeframe confirmation
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
                        #32CD32 0%, #9ACD32 20%, #FFD700 40%, 
                        #FF8C00 60%, #FF4500 80%, #DC143C 100%); 
                    border-radius: 14px; 
                    border: 1px solid #404040; 
                    overflow: hidden;
                    box-shadow: inset 0 2px 4px rgba(0,0,0,0.4);">
            
            <div style="position: absolute; top: 0; left: {score}%; 
                        width: 3px; height: 100%; 
                        background: white; 
                        box-shadow: 0 0 6px rgba(255,255,255,0.8);
                        transform: translateX(-1.5px);">
            </div>
            
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.8em; color: #888;">
            <span>Very Low</span>
            <span>Normal</span>
            <span>High</span>
            <span>Extreme</span>
        </div>
        
        <div style="text-align: center; margin-top: 0.5rem; padding: 0.5rem; background: {primary_color}20; border-radius: 8px; border-left: 4px solid {primary_color};">
            <strong style="color: {primary_color};">{regime}</strong>
        </div>
        
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)

def create_header():
    """Create VWV system header"""
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
        st.markdown(f"""
        <div style="padding: 1rem; background: {color}20; border-left: 4px solid {color}; border-radius: 8px; margin-bottom: 1rem;">
            <h4 style="color: {color}; margin: 0 0 0.5rem 0;">Current Environment</h4>
            <p style="margin: 0; font-size: 1.1em;"><strong>{regime}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        if options_strategy:
            st.markdown(f"""
            <div style="padding: 1rem; background: {color}15; border: 1px solid {color}40; border-radius: 8px;">
                <h4 style="color: {color}; margin: 0 0 0.5rem 0;">Options Strategy</h4>
                <p style="margin: 0;">{options_strategy}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if trading_implications:
            st.markdown(f"""
            <div style="padding: 1rem; background: {color}15; border: 1px solid {color}40; border-radius: 8px; height: 100%;">
                <h4 style="color: {color}; margin: 0 0 0.5rem 0;">Trading Implications</h4>
                <p style="margin: 0; line-height: 1.5;">{trading_implications}</p>
            </div>
            """, unsafe_allow_html=True)
