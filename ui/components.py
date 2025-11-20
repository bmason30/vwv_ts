"""
File: ui/components.py v1.3.0
VWV Professional Trading System v4.3.0
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
