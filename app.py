# Debug information
            if show_debug:
                with st.expander("🐛 Debug Information", expanded=False):
                    st.write("### Analysis Results Structure")
                    st.json(analysis_results, expanded=False)
                    
                    st.write("### Market Data Info")
                    st.write(f"**Data Shape:** {market_data.shape}")
                    st.write(f"**Date Range:** {market_data.index[0]} to {market_data.index[-1]}")
                    st.write(f"**Columns:** {list(market_data.columns)}")
                    
                    st.write("### Sample Data")
                    st.dataframe(market_data.tail(5), use_container_width=True)
                    
                    st.write("### Component Details")
                    for component, value in analysis_results['components'].items():
                        st.write(f"**{component.upper()}:** {value:.4f}")
                    
                    st.write("### Enhanced Indicators Summary")
                    enhanced_summary = {}
                    for key, value in enhanced_indicators.items():
                        if isinstance(value, (int, float)):
                            enhanced_summary[key] = round(value, 4)
                        elif isinstance(value, dict) and len(value) < 10:
                            enhanced_summary[key] = value
                        else:
                            enhanced_summary[key] = f"{type(value).__name__} with {len(value) if hasattr(value, '__len__') else 'N/A'} items"
                    
                    st.json(enhanced_summary)

    # ============================================================
    # NO ANALYSIS STATE - WELCOME MESSAGE
    # ============================================================
    else:
        st.write("## 🚀 VWV Professional Trading System")
        st.write("Welcome to the enhanced VWV Trading System with advanced technical analysis capabilities.")
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📊 **Getting Started**")
                st.write("1. **Enter a symbol** in the sidebar (e.g., AAPL, SPY, QQQ)")
                st.write("2. **Select time period** for analysis")
                st.write("3. **Click 'Analyze Symbol'** to run complete analysis")
                st.write("4. **Use Quick Links** for popular symbols")
                st.write("5. **Customize sections** in Analysis Sections panel")
                
                st.write("### 🎯 **What You'll Get**")
                st.write("• **Technical Composite Score** - Aggregated signal strength")
                st.write("• **Individual Technical Analysis** - 20+ indicators")
                st.write("• **Fundamental Analysis** - Graham & Piotroski scores")
                st.write("• **Market Correlation Analysis** - ETF comparisons")
                st.write("• **Options Trading Levels** - Greeks & probabilities")
                st.write("• **Interactive Charts** - Multi-timeframe view")
            
            with col2:
                st.write("### 📈 **Key Features**")
                st.write("• **Enhanced Williams VIX Fix** - Bottom detection")
                st.write("• **Fibonacci EMAs** - 21, 55, 89, 144, 233 periods")
                st.write("• **Daily VWAP** - Volume-weighted average price")
                st.write("• **Point of Control** - High-volume price levels")
                st.write("• **Weekly Deviations** - Statistical support/resistance")
                st.write("• **Dynamic Regime Detection** - Adapts to market conditions")
                
                st.write("### ⚡ **Signal Types**")
                st.write("• **🟢 GOOD** - Moderate confluence signal")
                st.write("• **🟡 STRONG** - High confluence signal")
                st.write("• **🔴 VERY STRONG** - Maximum confluence signal")
                st.write("• **📊 Directional bias** - Long/Short positioning")
                st.write("• **🎯 Entry/Exit levels** - Risk management")
        
        # Market overview
        with st.expander("🌍 Current Market Overview", expanded=True):
            st.write("### 📊 Market Indices Quick View")
            
            # Get quick market data for major indices
            try:
                major_indices = ['SPY', 'QQQ', 'IWM']
                market_overview_data = []
                
                for index in major_indices:
                    try:
                        # Get minimal data for overview
                        index_data = get_cached_market_data(index, '5d')
                        if index_data is not None and len(index_data) > 1:
                            current_price = index_data['Close'].iloc[-1]
                            prev_price = index_data['Close'].iloc[-2]
                            change_pct = ((current_price - prev_price) / prev_price) * 100
                            
                            market_overview_data.append({
                                'Index': index,
                                'Price': f"${current_price:.2f}",
                                'Change': f"{change_pct:+.2f}%",
                                'Status': "🟢" if change_pct > 0 else "🔴" if change_pct < 0 else "⚪"
                            })
                    except:
                        continue
                
                if market_overview_data:
                    df_market = pd.DataFrame(market_overview_data)
                    
                    # Display in columns
                    cols = st.columns(len(market_overview_data))
                    for i, (col, data) in enumerate(zip(cols, market_overview_data)):
                        with col:
                            st.metric(
                                f"{data['Status']} {data['Index']}", 
                                data['Price'], 
                                data['Change']
                            )
                else:
                    st.info("📊 Enter a symbol above to begin analysis")
                    
            except Exception as e:
                st.info("📊 Enter a symbol above to begin comprehensive market analysis")
        
        # Sample analysis showcase
        with st.expander("🎯 Sample Analysis Preview", expanded=False):
            st.write("### 🔍 What a Complete Analysis Includes:")
            
            # Mock data for demonstration
            st.write("#### 📊 Technical Composite Score")
            sample_score_html = create_technical_score_bar(73.2)
            st.markdown(sample_score_html, unsafe_allow_html=True)
            
            st.write("#### 📋 Technical Indicators Sample")
            sample_tech_data = {
                'Indicator': ['Current Price', 'Daily VWAP', 'EMA 21', 'RSI (14)', 'Volume Ratio'],
                'Value': ['$150.25', '$149.80', '$148.50', '45.2', '1.34x'],
                'Signal': ['🟡 Neutral', '🟢 Bullish', '🟢 Bullish', '🟡 Neutral', '🟢 Bullish']
            }
            st.dataframe(sample_tech_data, use_container_width=True, hide_index=True)
            
            st.write("#### 🎯 Options Levels Sample")
            sample_options_data = {
                'DTE': [7, 14, 30, 45],
                'Put Strike': ['$145.20', '$142.85', '$138.50', '$135.75'],
                'Call Strike': ['$155.40', '$158.20', '$163.80', '$167.25'],
                'Expected Move': ['±$3.20', '±$4.50', '±$7.30', '±$9.10']
            }
            st.dataframe(sample_options_data, use_container_width=True, hide_index=True)
            
            st.info("💡 **This is just a preview!** Run a real analysis to see comprehensive results with live market data.")

    # ============================================================
    # FOOTER AND ADDITIONAL INFORMATION
    # ============================================================
    
    # Footer section
    st.markdown("---")
    
    # System information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("### 📊 System Information")
        st.write(f"**Version:** VWV Professional v3.0")
        st.write(f"**Status:** ✅ Operational")
        st.write(f"**Last Update:** Enhanced Technical Analysis")
    
    with col2:
        st.write("### 🎯 Signal Methodology")
        st.write("**Williams VIX Fix:** Market fear indicator")
        st.write("**VWAP Analysis:** Volume-weighted pricing")
        st.write("**Fibonacci EMAs:** Multi-timeframe trends")
        st.write("**Dynamic Weighting:** Regime-adaptive signals")
    
    with col3:
        st.write("### ⚠️ Risk Disclaimer")
        st.write("**Educational Purpose Only**")
        st.write("• Not financial advice")
        st.write("• Past performance ≠ future results")
        st.write("• Always manage risk appropriately")
        st.write("• Consider professional advice")
    
    # Recently viewed footer
    if len(st.session_state.recently_viewed) > 0:
        st.write("### 🕒 Recently Analyzed")
        recent_chips = " • ".join([f"**{sym}**" for sym in st.session_state.recently_viewed[:5]])
        st.write(recent_chips)
    
    # Performance note
    st.info("🚀 **Performance Note:** First analysis may take longer due to data fetching. Subsequent analyses use caching for faster results.")

def display_signal_card(signal_type, signal_strength, directional_confluence):
    """Display signal information in a styled card"""
    
    if signal_type == 'NONE':
        card_color = "#e2e3e5"
        icon = "⚪"
        title = "No Signal"
        text_color = "#6c757d"
    elif "GOOD" in signal_type:
        card_color = "#d4edda"
        icon = "🟢"
        title = f"Good {signal_type.split('_')[1]} Signal"
        text_color = "#155724"
    elif "STRONG" in signal_type:
        card_color = "#fff3cd"
        icon = "🟡"
        title = f"Strong {signal_type.split('_')[1]} Signal"
        text_color = "#856404"
    elif "VERY_STRONG" in signal_type:
        card_color = "#f8d7da"
        icon = "🔴"
        title = f"Very Strong {signal_type.split('_')[1]} Signal"
        text_color = "#721c24"
    else:
        card_color = "#e2e3e5"
        icon = "⚪"
        title = "Unknown Signal"
        text_color = "#6c757d"
    
    card_html = f"""
    <div style="
        background-color: {card_color};
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid {text_color};
        margin: 1rem 0;
    ">
        <h3 style="color: {text_color}; margin: 0 0 0.5rem 0;">
            {icon} {title}
        </h3>
        <p style="color: {text_color}; margin: 0; font-size: 1.1em;">
            <strong>Confluence Score:</strong> {directional_confluence:.2f}<br>
            <strong>Signal Strength:</strong> {signal_strength}/3
        </p>
    </div>
    """
    
    return card_html

# Add any additional utility functions here
def format_large_number(num):
    """Format large numbers with appropriate suffixes"""
    try:
        num = float(num)
        if abs(num) >= 1e12:
            return f"{num/1e12:.1f}T"
        elif abs(num) >= 1e9:
            return f"{num/1e9:.1f}B"
        elif abs(num) >= 1e6:
            return f"{num/1e6:.1f}M"
        elif abs(num) >= 1e3:
            return f"{num/1e3:.1f}K"
        else:
            return f"{num:.0f}"
    except:
        return str(num)

def get_market_status():
    """Get current market status"""
    try:
        from datetime import datetime
        import pytz
        
        # US Eastern Time
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        
        # Market hours: 9:30 AM - 4:00 PM ET, Mon-Fri
        if now.weekday() < 5:  # Monday = 0, Friday = 4
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            if market_open <= now <= market_close:
                return "🟢 Market Open"
            elif now < market_open:
                return "🟡 Pre-Market"
            else:
                return "🔴 After Hours"
        else:
            return "🔴 Market Closed"
    except:
        return "🟡 Status Unknown"

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.write("Please refresh the page and try again.")
        
        # Show debug info if error occurs
        if st.checkbox("Show Error Details"):
            st.exception(e)
            
        # Emergency reset button
        if st.button("🔄 Reset Application State"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Application state cleared. Please refresh the page.")
