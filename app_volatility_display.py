"""
File: app_volatility_display.py
Professional Volatility Analysis Display Functions for VWV Trading System
Version: v4.2.1-VOLATILITY-DISPLAY-2025-08-27-17-35-00-EST
Complete UI integration for advanced volatility analysis with 14 indicators
Last Updated: August 27, 2025 - 5:35 PM EST
"""

import streamlit as st
import pandas as pd
from ui.components import (
    create_volatility_score_bar,
    create_component_breakdown_table,
    display_regime_classification,
    format_large_number
)

def show_volatility_analysis(analysis_results, show_debug=False):
    """Display advanced volatility analysis section with professional UI - NEW v4.2.1"""
    
    # Check if volatility analysis should be displayed
    if not st.session_state.get('show_volatility_analysis', True):
        return
    
    # Check if volatility analysis is available
    try:
        from analysis.volatility import VOLATILITY_ANALYSIS_AVAILABLE
        if not VOLATILITY_ANALYSIS_AVAILABLE:
            return
    except ImportError:
        return
        
    symbol = analysis_results.get('symbol', 'Unknown')
    
    with st.expander(f"🌡️ {symbol} - Advanced Volatility Analysis", expanded=True):
        
        # Get volatility analysis data
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volatility_analysis = enhanced_indicators.get('volatility_analysis', {})
        
        if 'error' in volatility_analysis:
            st.error(f"❌ Volatility analysis failed: {volatility_analysis['error']}")
            return
            
        if not volatility_analysis or not volatility_analysis.get('analysis_success', False):
            st.warning("⚠️ Volatility analysis not available - insufficient data")
            return
        
        # MAIN VOLATILITY COMPOSITE SCORE BAR
        volatility_score = volatility_analysis.get('volatility_score', 50.0)
        volatility_regime = volatility_analysis.get('volatility_regime', 'Normal Volatility')
        create_volatility_score_bar(volatility_score, volatility_regime)
        
        # PRIMARY VOLATILITY METRICS
        st.subheader("📊 Core Volatility Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            vol_20d = volatility_analysis.get('volatility_20d', 20.0)
            st.metric("20-Day Volatility", f"{vol_20d:.1f}%")
            
        with col2:
            vol_10d = volatility_analysis.get('volatility_10d', 20.0)
            st.metric("10-Day Volatility", f"{vol_10d:.1f}%")
            
        with col3:
            vol_percentile = volatility_analysis.get('volatility_percentile', 50.0)
            st.metric("Volatility Percentile", f"{vol_percentile:.0f}%")
            
        with col4:
            vol_rank = volatility_analysis.get('volatility_rank', 50.0)
            st.metric("Volatility Rank", f"{vol_rank:.0f}%")
        
        # ADVANCED VOLATILITY ESTIMATORS
        st.subheader("🔬 Advanced Volatility Estimators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            garch_vol = volatility_analysis.get('garch_volatility', vol_20d)
            st.metric("GARCH Volatility", f"{garch_vol:.1f}%", "Advanced Modeling")
            
        with col2:
            parkinson_vol = volatility_analysis.get('parkinson_volatility', vol_20d)
            st.metric("Parkinson Est.", f"{parkinson_vol:.1f}%", "High-Low Range")
            
        with col3:
            yang_zhang_vol = volatility_analysis.get('yang_zhang_volatility', vol_20d)
            st.metric("Yang-Zhang Est.", f"{yang_zhang_vol:.1f}%", "Combined Approach")
            
        with col4:
            realized_vol = volatility_analysis.get('realized_volatility', vol_20d)
            st.metric("Realized Vol", f"{realized_vol:.1f}%", "Actual Movements")
        
        # VOLATILITY DYNAMICS
        st.subheader("📈 Volatility Dynamics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            vol_momentum = volatility_analysis.get('volatility_momentum', 0.0)
            momentum_delta = "↗️" if vol_momentum > 2 else "↘️" if vol_momentum < -2 else "➡️"
            st.metric("Vol Momentum", f"{vol_momentum:+.1f}%", momentum_delta)
            
        with col2:
            vol_acceleration = volatility_analysis.get('volatility_acceleration', 0.0)
            accel_delta = "🚀" if vol_acceleration > 1 else "🛑" if vol_acceleration < -1 else "➡️"
            st.metric("Vol Acceleration", f"{vol_acceleration:+.2f}", accel_delta)
            
        with col3:
            vol_of_vol = volatility_analysis.get('volatility_of_volatility', 2.0)
            st.metric("Vol of Vol", f"{vol_of_vol:.2f}", "2nd Order")
            
        with col4:
            vol_clustering = volatility_analysis.get('volatility_clustering', 0.5)
            cluster_level = "High" if vol_clustering > 0.7 else "Low" if vol_clustering < 0.3 else "Moderate"
            st.metric("Vol Clustering", f"{vol_clustering:.3f}", cluster_level)
        
        # REGIME CLASSIFICATION & STRATEGY RECOMMENDATIONS  
        st.subheader("🎯 Market Environment & Strategy")
        
        options_strategy = volatility_analysis.get('options_strategy', 'No strategy available')
        trading_implications = volatility_analysis.get('trading_implications', 'No implications available')
        
        display_regime_classification(
            volatility_score,
            volatility_regime, 
            options_strategy,
            trading_implications
        )
        
        # RISK METRICS
        st.subheader("⚠️ Risk & Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            vol_strength_factor = volatility_analysis.get('volatility_strength_factor', 1.0)
            strength_desc = "Very High" if vol_strength_factor >= 1.2 else "High" if vol_strength_factor >= 1.1 else "Normal" if vol_strength_factor >= 0.9 else "Low"
            st.metric("Vol Strength Factor", f"{vol_strength_factor:.2f}x", strength_desc)
            
        with col2:
            risk_adj_return = volatility_analysis.get('risk_adjusted_return', 0.0)
            st.metric("Risk-Adj Return", f"{risk_adj_return:.2f}", "Sharpe-like")
            
        with col3:
            vol_consistency = volatility_analysis.get('volatility_consistency', 0.0)
            consistency_desc = "Stable" if vol_consistency < 20 else "Volatile" if vol_consistency > 40 else "Moderate"
            st.metric("Vol Consistency", f"{vol_consistency:.1f}%", consistency_desc)
        
        # COMPONENT BREAKDOWN EXPANDER
        if volatility_analysis.get('indicators') and volatility_analysis.get('scores'):
            with st.expander("📋 14-Component Volatility Breakdown", expanded=False):
                st.write("**Advanced Volatility Indicator Analysis**")
                st.write("Showing all 14 volatility indicators with research-based weightings")
                
                indicators = volatility_analysis.get('indicators', {})
                scores = volatility_analysis.get('scores', {})
                weights = volatility_analysis.get('weights', {})
                contributions = volatility_analysis.get('contributions', {})
                
                if indicators and scores and weights:
                    create_component_breakdown_table(
                        indicators, scores, weights, contributions,
                        title="Volatility Components"
                    )
                    
                    # Methodology explanation
                    st.write("**📚 Methodology Overview:**")
                    st.write("• **Historical Volatility (20D/10D)**: Primary volatility measures using rolling standard deviation")
                    st.write("• **GARCH/Parkinson/Yang-Zhang**: Advanced volatility estimators using OHLC data")  
                    st.write("• **Volatility Percentile/Rank**: Relative positioning vs historical volatility ranges")
                    st.write("• **Vol of Vol/Clustering**: Second-order volatility and persistence analysis")
                    st.write("• **Momentum/Mean Reversion**: Volatility trend analysis and mean reversion tendency")
                    
                else:
                    st.warning("Component breakdown data not available")
        
        # MARKET-WIDE VOLATILITY CONTEXT
        if show_debug:
            with st.expander("📊 Market-Wide Volatility Context", expanded=False):
                st.write("**Analyzing market-wide volatility environment...**")
                
                try:
                    from analysis.volatility import calculate_market_wide_volatility_analysis
                    market_vol_data = calculate_market_wide_volatility_analysis(
                        symbols=['SPY', 'QQQ', 'IWM'], 
                        show_debug=True
                    )
                    
                    if market_vol_data.get('market_analysis_success', False):
                        market_indices = market_vol_data.get('market_indices', {})
                        
                        if market_indices:
                            st.write("**Market Index Volatility Comparison:**")
                            
                            market_data = []
                            for symbol, data in market_indices.items():
                                market_data.append({
                                    'Index': symbol,
                                    'Volatility': f"{data.get('volatility_20d', 0):.1f}%",
                                    'Score': f"{data.get('volatility_score', 0):.0f}/100",
                                    'Regime': data.get('volatility_regime', 'Unknown')
                                })
                            
                            if market_data:
                                df_market = pd.DataFrame(market_data)
                                st.dataframe(df_market, use_container_width=True, hide_index=True)
                            
                            # Overall market environment
                            market_env = market_vol_data.get('market_volatility_environment', 'Unknown')
                            avg_vol = market_vol_data.get('average_volatility', 0)
                            
                            st.info(f"**Market Environment:** {market_env} (Avg: {avg_vol:.1f}%)")
                        
                    else:
                        st.warning("Market-wide volatility analysis not available")
                        
                except Exception as e:
                    st.error(f"Market volatility analysis failed: {e}")
        
        # DEBUG INFORMATION
        if show_debug:
            with st.expander("🐛 Volatility Debug Information", expanded=False):
                st.write("**Raw Volatility Analysis Results:**")
                st.json(volatility_analysis, expanded=False)
                
                if volatility_analysis.get('weights'):
                    st.write("**Indicator Weights Verification:**")
                    total_weight = sum(volatility_analysis['weights'].values())
                    st.write(f"Total Weight Sum: {total_weight:.6f} (Should be close to 1.0)")
                    
                    for indicator, weight in volatility_analysis['weights'].items():
                        st.write(f"• {indicator}: {weight:.3f}")

def show_volume_analysis(analysis_results, show_debug=False):
    """Display volume analysis section - ENHANCED with Professional UI"""
    
    # Check if volume analysis should be displayed
    if not st.session_state.get('show_volume_analysis', True):
        return
    
    # Check if volume analysis is available
    try:
        from analysis.volume import VOLUME_ANALYSIS_AVAILABLE
        if not VOLUME_ANALYSIS_AVAILABLE:
            return
    except ImportError:
        return
        
    symbol = analysis_results.get('symbol', 'Unknown')
    
    with st.expander(f"🔊 {symbol} - Volume Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volume_analysis = enhanced_indicators.get('volume_analysis', {})
        
        if 'error' in volume_analysis:
            st.error(f"❌ Volume analysis failed: {volume_analysis['error']}")
            return
            
        if not volume_analysis:
            st.warning("⚠️ Volume analysis not available - insufficient data")
            return

        # VOLUME COMPOSITE SCORE BAR (if available)
        volume_score = volume_analysis.get('volume_score', 50)
        volume_regime = volume_analysis.get('volume_regime', 'Normal')
        
        if volume_score and volume_regime:
            from ui.components import create_volume_score_bar
            create_volume_score_bar(volume_score, volume_regime)
        
        # PRIMARY VOLUME METRICS
        st.subheader("📊 Core Volume Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_volume = volume_analysis.get('current_volume', 0)
            st.metric("Current Volume", format_large_number(current_volume))
            
        with col2:
            volume_5d_avg = volume_analysis.get('volume_5d_avg', 0)
            st.metric("5-Day Avg", format_large_number(volume_5d_avg))
            
        with col3:
            volume_ratio = volume_analysis.get('volume_ratio', 1.0)
            ratio_desc = "🔥" if volume_ratio > 2.0 else "📈" if volume_ratio > 1.5 else "➡️" if volume_ratio > 0.8 else "📉"
            st.metric("Volume Ratio", f"{volume_ratio:.2f}x", ratio_desc)
            
        with col4:
            volume_trend = volume_analysis.get('volume_5d_trend', 0)
            trend_desc = "↗️" if volume_trend > 10 else "↘️" if volume_trend < -10 else "➡️"
            st.metric("5D Trend", f"{volume_trend:+.1f}%", trend_desc)
        
        # VOLUME ENVIRONMENT & IMPLICATIONS
        st.subheader("🎯 Volume Environment")
        
        trading_implications = volume_analysis.get('trading_implications', 'Volume analysis provides insight into market participation and conviction behind price movements.')
        
        display_regime_classification(
            volume_score,
            volume_regime,
            None,  # Volume doesn't have options strategy
            trading_implications
        )
        
        # DEBUG INFORMATION
        if show_debug:
            with st.expander("🐛 Volume Debug Information", expanded=False):
                st.write("**Raw Volume Analysis Results:**")
                st.json(volume_analysis, expanded=False)
