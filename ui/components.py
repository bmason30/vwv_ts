"""
Filename: ui/components.py
VWV Trading System v4.2.1
Created/Updated: 2025-09-04 09:40:11 EDT
Version: 1.1.0 - Refactored to use local imports for deployment compatibility
Purpose: UI components for the Streamlit application.
"""

def create_technical_score_bar(score, name):
    """Creates a progress bar to represent a technical score."""
    import streamlit as st
    
    if score is None:
        st.warning(f"Score for {name} not available.")
        return

    # Determine color based on score
    if score >= 70:
        color = "green"
    elif score >= 40:
        color = "orange"
    else:
        color = "red"

    st.markdown(f"**{name}:**")
    st.markdown(f"""
    <div style="background-color: #262730; border-radius: 5px; padding: 5px; border: 1px solid #333;">
        <div style="background-color: {color}; width: {score}%; height: 20px; border-radius: 5px; text-align: center; color: white; font-weight: bold;">
            {score:.1f}
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("") # Add some space

def create_header():
    """Creates the main header for the application."""
    import streamlit as st
    
    st.markdown("""
        <div style="background-color:#0e1117;padding:10px;border-radius:10px;text-align:center;border: 1px solid #262730;">
            <h1 style="color:#ffffff;font-size:2em;">🚀 VWV Professional Trading System v4</h1>
            <p style="color:#a0a0a0;">Advanced Technical Analysis • Volatility Analysis • Professional Trading Signals</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
