# Enhanced show_volatility_analysis function for app.py v4.2.1
# This replaces the existing show_volatility_analysis function
# Add this import at the top of your app.py file:
from ui.components import create_technical_score_bar, create_volatility_score_bar, create_volume_score_bar, create_header

def show_volatility_analysis(analysis_results, show_debug=False):
    """
    Display volatility analysis section - ENHANCED v4.2.1
    NOW INCLUDES: Gradient score bar + Component breakdown expander
    """
    if not st.session_state.show_volatility_analysis or not VOLATILITY_ANALYSIS_AVAILABLE:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Volatility Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volatility_analysis = enhanced_indicators.get('volatility_analysis', {})
        
        if 'error' not in volatility_analysis and volatility_analysis:
            
            # --- 1. VOLATILITY COMPOSITE SCORE BAR (NEW) ---
            volatility_score = volatility_analysis.get('volatility_score', 50)
            volatility_score_bar_html = create_volatility_score_bar(volatility_score, volatility_analysis)
            st.components.v1.html(volatility_score_bar_html, height=160)
            
            # --- 2. PRIMARY VOLATILITY METRICS ---
            st.subheader("📊 Key Volatility Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                vol_5d = volatility_analysis.get('volatility_5d', 0)
                st.metric("5D Volatility", f"{vol_5d:.2f}%")
            with col2:
                vol_30d = volatility_analysis.get('volatility_30d', 0)
                st.metric("30D Volatility", f"{vol_30d:.2f}%")
            with col3:
                vol_percentile = volatility_analysis.get('volatility_percentile', 50)
                st.metric("Vol Percentile", f"{vol_percentile:.1f}%")
            with col4:
                vol_trend = volatility_analysis.get('volatility_trend', 0)
                st.metric("Vol Trend", f"{vol_trend:+.2f}%")
            
            # --- 3. VOLATILITY ENVIRONMENT & OPTIONS STRATEGY ---
            st.subheader("📊 Volatility Environment & Options Strategy")
            vol_regime = volatility_analysis.get('volatility_regime', 'Unknown')
            options_strategy = volatility_analysis.get('options_strategy', 'No strategy available')
            trading_implications = volatility_analysis.get('trading_implications', 'No implications available')
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Volatility Regime:** {vol_regime}")
                st.info(f"**Volatility Score:** {volatility_score}/100")
            with col2:
                st.info(f"**Options Strategy:** {options_strategy}")
                st.info(f"**Trading Implications:**\n{trading_implications}")
            
            # --- 4. COMPONENT BREAKDOWN EXPANDER (NEW) ---
            component_breakdown = volatility_analysis.get('component_breakdown', [])
            if component_breakdown:
                with st.expander("🔬 14-Indicator Volatility Component Breakdown", expanded=False):
                    
                    st.write("**Comprehensive breakdown of all volatility indicators contributing to the composite score:**")
                    
                    # Create component summary table
                    component_data = []
                    for i, component in enumerate(component_breakdown, 1):
                        component_data.append([
                            f"{i}. {component['name']}",
                            component['value'],
                            component['score'],
                            component['weight'],
                            component['contribution']
                        ])
                    
                    # Display component table
                    df_components = pd.DataFrame(component_data, 
                                               columns=['Volatility Indicator', 'Current Value', 'Score', 'Weight', 'Contribution'])
                    st.dataframe(df_components, use_container_width=True, hide_index=True)
                    
                    # Advanced volatility metrics
                    st.subheader("📊 Advanced Volatility Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        vol_ratio = volatility_analysis.get('volatility_ratio', 1.0)
                        st.metric("Vol Ratio (5D/30D)", f"{vol_ratio:.2f}")
                    with col2:
                        vol_rank = volatility_analysis.get('volatility_rank', 50)
                        st.metric("Volatility Rank", f"{vol_rank:.1f}%")
                    with col3:
                        vol_clustering = volatility_analysis.get('volatility_clustering', 0.5)
                        st.metric("Vol Clustering", f"{vol_clustering:.3f}")
                    with col4:
                        vol_mean_reversion = volatility_analysis.get('volatility_mean_reversion', 0.5)
                        st.metric("Mean Reversion", f"{vol_mean_reversion:.3f}")
                    
                    # Risk metrics
                    st.subheader("⚖️ Risk & Performance Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        risk_adjusted_return = volatility_analysis.get('risk_adjusted_return', 0)
                        st.metric("Risk-Adj Return", f"{risk_adjusted_return:.2f}")
                    with col2:
                        vol_of_vol = volatility_analysis.get('volatility_of_volatility', 5.0)
                        st.metric("Vol of Vol", f"{vol_of_vol:.2f}")
                    with col3:
                        vol_strength_factor = volatility_analysis.get('volatility_strength_factor', 1.0)
                        st.metric("Vol Strength Factor", f"{vol_strength_factor:.2f}x")
                    
                    # Component weighting explanation
                    st.subheader("🎯 Indicator Weighting Methodology")
                    st.write("""
                    **Research-Based Weighting System:**
                    - **Historical Volatility (20D)**: 15% - Primary volatility measure
                    - **Historical Volatility (10D)**: 12% - Short-term volatility
                    - **Realized Volatility**: 13% - Actual price movements
                    - **Volatility Percentile**: 11% - Relative positioning
                    - **Volatility Rank**: 9% - Historical ranking
                    - **GARCH Volatility**: 8% - Advanced modeling
                    - **Parkinson/Garman-Klass/Rogers-Satchell/Yang-Zhang**: 22% combined - Advanced estimators
                    - **Volatility Metrics**: 10% combined - VoV, Momentum, Mean Reversion, Clustering
                    
                    **Total Weight**: 100% across all 14 indicators
                    """)
            
            else:
                st.warning("⚠️ Component breakdown not available - using basic volatility calculation")
                
        else:
            error_msg = volatility_analysis.get('error', 'Unknown error')
            st.warning(f"⚠️ Volatility analysis not available - {error_msg}")
            
        # Debug information for volatility analysis
        if show_debug and volatility_analysis:
            with st.expander("🐛 Volatility Analysis Debug", expanded=False):
                st.write("**Raw Volatility Analysis Data:**")
                st.json(volatility_analysis, expanded=True)

# Enhanced show_volume_analysis function for app.py v4.2.1
# This also adds the missing gradient bar for volume analysis
def show_volume_analysis(analysis_results, show_debug=False):
    """
    Display volume analysis section - ENHANCED v4.2.1  
    NOW INCLUDES: Gradient score bar + Component breakdown expander
    """
    if not st.session_state.show_volume_analysis or not VOLUME_ANALYSIS_AVAILABLE:
        return
        
    with st.expander(f"📊 {analysis_results['symbol']} - Volume Analysis", expanded=True):
        
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        volume_analysis = enhanced_indicators.get('volume_analysis', {})
        
        if 'error' not in volume_analysis and volume_analysis:
            
            # --- 1. VOLUME COMPOSITE SCORE BAR (NEW) ---
            volume_score = volume_analysis.get('volume_score', 50)
            volume_score_bar_html = create_volume_score_bar(volume_score, volume_analysis)
            st.components.v1.html(volume_score_bar_html, height=160)
            
            # --- 2. PRIMARY VOLUME METRICS ---
            st.subheader("📊 Key Volume Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Volume", format_large_number(volume_analysis.get('current_volume', 0)))
            with col2:
                st.metric("5D Avg Volume", format_large_number(volume_analysis.get('volume_5d_avg', 0)))
            with col3:
                volume_ratio = volume_analysis.get('volume_ratio', 1.0)
                st.metric("Volume Ratio", f"{volume_ratio:.2f}x", f"vs 30D avg")
            with col4:
                volume_trend = volume_analysis.get('volume_5d_trend', 0)
                st.metric("5D Volume Trend", f"{volume_trend:+.2f}%")
            
            # --- 3. VOLUME ENVIRONMENT ---
            st.subheader("📊 Volume Environment")
            volume_regime = volume_analysis.get('volume_regime', 'Unknown')
            trading_implications = volume_analysis.get('trading_implications', 'No implications available')
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Volume Regime:** {volume_regime}")
                st.info(f"**Volume Score:** {volume_score}/100")
            with col2:
                st.info(f"**Trading Implications:**\n{trading_implications}")
            
            # --- 4. COMPONENT BREAKDOWN EXPANDER (NEW) ---
            component_breakdown = volume_analysis.get('component_breakdown', [])
            if component_breakdown:
                with st.expander("🔬 14-Indicator Volume Component Breakdown", expanded=False):
                    
                    st.write("**Comprehensive breakdown of all volume indicators contributing to the composite score:**")
                    
                    # Create component summary table
                    component_data = []
                    for i, component in enumerate(component_breakdown, 1):
                        component_data.append([
                            f"{i}. {component['name']}",
                            component['value'],
                            component['score'],
                            component['weight'],
                            component['contribution']
                        ])
                    
                    # Display component table
                    df_components = pd.DataFrame(component_data, 
                                               columns=['Volume Indicator', 'Current Value', 'Score', 'Weight', 'Contribution'])
                    st.dataframe(df_components, use_container_width=True, hide_index=True)
                    
                    # Advanced volume metrics
                    st.subheader("📊 Advanced Volume Metrics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        volume_zscore = volume_analysis.get('volume_zscore', 0)
                        st.metric("Volume Z-Score", f"{volume_zscore:.2f}")
                    with col2:
                        volume_breakout = volume_analysis.get('volume_breakout', 'None')
                        st.metric("Volume Breakout", volume_breakout)
                    with col3:
                        volume_acceleration = volume_analysis.get('volume_acceleration', 0)
                        st.metric("Vol Acceleration", f"{volume_acceleration:+.2f}%")
                    with col4:
                        volume_strength_factor = volume_analysis.get('volume_strength_factor', 1.0)
                        st.metric("Vol Strength Factor", f"{volume_strength_factor:.2f}x")
            
            else:
                st.warning("⚠️ Component breakdown not available - using basic volume calculation")
                
        else:
            error_msg = volume_analysis.get('error', 'Unknown error')
            st.warning(f"⚠️ Volume analysis not available - {error_msg}")
            
        # Debug information for volume analysis
        if show_debug and volume_analysis:
            with st.expander("🐛 Volume Analysis Debug", expanded=False):
                st.write("**Raw Volume Analysis Data:**")
                st.json(volume_analysis, expanded=True)
