"""
Enhanced Technical Analysis Display with Gradient Score Bar
Add this to your app.py or create as ui/technical_display.py
"""
import streamlit as st
import pandas as pd

def create_gradient_score_bar(score, title="Score", height=80):
    """
    Create a horizontal gradient bar (red to green) with pointer
    Score should be 0-100
    """
    # Ensure score is within bounds
    score = max(0, min(100, score))
    
    # Calculate color at score position
    # Red (255,0,0) to Yellow (255,255,0) to Green (0,255,0)
    if score <= 50:
        # Red to Yellow
        ratio = score / 50
        r = 255
        g = int(255 * ratio)
        b = 0
    else:
        # Yellow to Green
        ratio = (score - 50) / 50
        r = int(255 * (1 - ratio))
        g = 255
        b = 0
    
    color = f"rgb({r},{g},{b})"
    
    # Create HTML for gradient bar with pointer
    html = f"""
    <div style="margin: 20px 0;">
        <div style="position: relative; width: 100%; height: {height}px;">
            <!-- Gradient Bar -->
            <div style="
                width: 100%;
                height: 40px;
                background: linear-gradient(to right, 
                    rgb(220,20,60) 0%,      /* Crimson Red */
                    rgb(255,69,0) 20%,      /* Red-Orange */
                    rgb(255,165,0) 40%,     /* Orange */
                    rgb(255,215,0) 50%,     /* Gold */
                    rgb(173,255,47) 60%,    /* Yellow-Green */
                    rgb(50,205,50) 80%,     /* Lime Green */
                    rgb(0,255,0) 100%       /* Bright Green */
                );
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                position: relative;
            ">
                <!-- Score markers -->
                <div style="
                    position: absolute;
                    bottom: -20px;
                    left: 0;
                    width: 100%;
                    display: flex;
                    justify-content: space-between;
                    font-size: 11px;
                    color: #888;
                    padding: 0 5px;
                ">
                    <span>0</span>
                    <span>25</span>
                    <span>50</span>
                    <span>75</span>
                    <span>100</span>
                </div>
            </div>
            
            <!-- Pointer and Score Display -->
            <div style="
                position: absolute;
                top: -25px;
                left: {score}%;
                transform: translateX(-50%);
                text-align: center;
            ">
                <!-- Score Value -->
                <div style="
                    font-size: 24px;
                    font-weight: bold;
                    color: {color};
                    text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
                    margin-bottom: 5px;
                ">
                    {score:.1f}
                </div>
                
                <!-- Pointer Arrow -->
                <div style="
                    width: 0;
                    height: 0;
                    border-left: 8px solid transparent;
                    border-right: 8px solid transparent;
                    border-top: 12px solid {color};
                    margin: 0 auto;
                    filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.3));
                ">
                </div>
            </div>
        </div>
    </div>
    """
    
    return html

def show_enhanced_technical_analysis(analysis_results, show_debug=False):
    """
    Enhanced Technical Analysis Display with expandable sections
    """
    if not st.session_state.get('show_technical_analysis', True):
        return
    
    with st.expander(f"📊 {analysis_results['symbol']} - Technical Analysis", expanded=True):
        
        # Calculate composite score
        from analysis.technical import calculate_composite_technical_score
        composite_score, score_details = calculate_composite_technical_score(analysis_results)
        
        # Get enhanced indicators
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        
        # === SECTION 1: COMPOSITE SCORE WITH GRADIENT BAR ===
        st.markdown("### Technical Composite Score")
        
        # Create gradient bar
        gradient_html = create_gradient_score_bar(composite_score, "Technical Score")
        st.components.v1.html(gradient_html, height=120)
        
        # Interpretation text
        if composite_score >= 75:
            interpretation = "🟢 **Strong Bullish** - Multiple bullish indicators aligned"
        elif composite_score >= 65:
            interpretation = "🟢 **Bullish** - Moderately positive technical setup"
        elif composite_score >= 55:
            interpretation = "🟡 **Neutral-Bullish** - Slight bullish bias"
        elif composite_score >= 45:
            interpretation = "⚪ **Neutral** - Mixed signals, no clear direction"
        elif composite_score >= 35:
            interpretation = "🟡 **Neutral-Bearish** - Slight bearish bias"
        elif composite_score >= 25:
            interpretation = "🔴 **Bearish** - Moderately negative technical setup"
        else:
            interpretation = "🔴 **Strong Bearish** - Multiple bearish indicators aligned"
        
        st.markdown(f"<div style='text-align: center; font-size: 16px; margin: 10px 0;'>{interpretation}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # === SECTION 2: KEY MOMENTUM OSCILLATORS (Quick Glance) ===
        st.markdown("### 🎯 Key Momentum Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi = comprehensive_technicals.get('rsi_14', 50)
            rsi_signal = "🔴 Overbought" if rsi > 70 else "🟢 Oversold" if rsi < 30 else "⚪ Neutral"
            st.metric("RSI (14)", f"{rsi:.2f}", rsi_signal)
        
        with col2:
            mfi = comprehensive_technicals.get('mfi_14', 50)
            mfi_signal = "🔴 Overbought" if mfi > 80 else "🟢 Oversold" if mfi < 20 else "⚪ Neutral"
            st.metric("MFI (14)", f"{mfi:.2f}", mfi_signal)
        
        with col3:
            stoch = comprehensive_technicals.get('stochastic', {})
            stoch_k = stoch.get('k', 50)
            stoch_signal = "🔴 Overbought" if stoch_k > 80 else "🟢 Oversold" if stoch_k < 20 else "⚪ Neutral"
            st.metric("Stochastic %K", f"{stoch_k:.2f}", stoch_signal)
        
        with col4:
            williams = comprehensive_technicals.get('williams_r', -50)
            williams_signal = "🔴 Overbought" if williams > -20 else "🟢 Oversold" if williams < -80 else "⚪ Neutral"
            st.metric("Williams %R", f"{williams:.2f}", williams_signal)
        
        # === SECTION 3: EXPANDABLE DETAILED ANALYSIS ===
        
        # 3A: Trend Analysis
        with st.expander("📈 Trend Analysis (Detailed)", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### MACD Analysis")
                macd = comprehensive_technicals.get('macd', {})
                macd_line = macd.get('macd', 0)
                signal_line = macd.get('signal', 0)
                histogram = macd.get('histogram', 0)
                
                st.write(f"**MACD Line:** {macd_line:.4f}")
                st.write(f"**Signal Line:** {signal_line:.4f}")
                st.write(f"**Histogram:** {histogram:.4f}")
                
                if histogram > 0:
                    st.success("✅ Bullish crossover - upward momentum")
                else:
                    st.warning("❌ Bearish crossover - downward momentum")
            
            with col2:
                st.markdown("#### Moving Averages")
                fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
                
                if fibonacci_emas:
                    current_price = analysis_results.get('current_price', 0)
                    
                    for ema_name, ema_value in sorted(fibonacci_emas.items()):
                        period = ema_name.split('_')[1]
                        distance = ((current_price - ema_value) / ema_value * 100) if ema_value > 0 else 0
                        status = "🟢 Above" if distance > 0 else "🔴 Below"
                        st.write(f"**EMA {period}:** ${ema_value:.2f} ({status}, {distance:+.2f}%)")
        
        # 3B: Volatility & Range
        with st.expander("📊 Volatility & Price Range", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Bollinger Bands")
                bollinger = comprehensive_technicals.get('bollinger_bands', {})
                bb_upper = bollinger.get('upper', 0)
                bb_middle = bollinger.get('middle', 0)
                bb_lower = bollinger.get('lower', 0)
                bb_width = bollinger.get('width', 0)
                
                st.write(f"**Upper Band:** ${bb_upper:.2f}")
                st.write(f"**Middle Band:** ${bb_middle:.2f}")
                st.write(f"**Lower Band:** ${bb_lower:.2f}")
                st.write(f"**Band Width:** {bb_width:.4f}")
                
                current_price = analysis_results.get('current_price', 0)
                if current_price > bb_upper:
                    st.warning("🔴 Price above upper band - potentially overbought")
                elif current_price < bb_lower:
                    st.success("🟢 Price below lower band - potentially oversold")
                else:
                    st.info("⚪ Price within bands - normal range")
            
            with col2:
                st.markdown("#### Average True Range (ATR)")
                atr = comprehensive_technicals.get('atr_14', 0)
                volatility_20d = comprehensive_technicals.get('volatility_20d', 0)
                
                st.write(f"**ATR (14):** ${atr:.2f}")
                st.write(f"**20-Day Volatility:** {volatility_20d:.2f}%")
                
                if volatility_20d > 30:
                    st.warning("🔴 High volatility - increased risk")
                elif volatility_20d < 15:
                    st.info("🟢 Low volatility - stable conditions")
                else:
                    st.info("⚪ Normal volatility")
        
        # 3C: Volume Analysis
        with st.expander("📊 Volume Indicators", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Volume Metrics")
                current_volume = comprehensive_technicals.get('current_volume', 0)
                avg_volume = comprehensive_technicals.get('volume_sma_20', 0)
                volume_ratio = (current_volume / avg_volume) if avg_volume > 0 else 1
                
                st.write(f"**Current Volume:** {current_volume:,.0f}")
                st.write(f"**20-Day Avg:** {avg_volume:,.0f}")
                st.write(f"**Volume Ratio:** {volume_ratio:.2f}x")
                
                if volume_ratio > 2:
                    st.success("🟢 Very high volume - strong interest")
                elif volume_ratio > 1.5:
                    st.info("🟢 Above average volume")
                elif volume_ratio < 0.5:
                    st.warning("🔴 Low volume - weak interest")
                else:
                    st.info("⚪ Normal volume")
            
            with col2:
                st.markdown("#### VWAP & Price Levels")
                daily_vwap = enhanced_indicators.get('daily_vwap', 0)
                poc = enhanced_indicators.get('point_of_control', 0)
                current_price = analysis_results.get('current_price', 0)
                
                vwap_distance = ((current_price - daily_vwap) / daily_vwap * 100) if daily_vwap > 0 else 0
                poc_distance = ((current_price - poc) / poc * 100) if poc > 0 else 0
                
                st.write(f"**Daily VWAP:** ${daily_vwap:.2f}")
                st.write(f"**Distance:** {vwap_distance:+.2f}%")
                st.write(f"**Point of Control:** ${poc:.2f}")
                st.write(f"**Distance:** {poc_distance:+.2f}%")
        
        # 3D: Score Breakdown
        if show_debug and score_details:
            with st.expander("🔬 Score Calculation Breakdown", expanded=False):
                st.markdown("#### Component Scores")
                st.json(score_details)

# Example usage in app.py:
# Replace your current show_individual_technical_analysis() with:
# show_enhanced_technical_analysis(analysis_results, controls['show_debug'])
