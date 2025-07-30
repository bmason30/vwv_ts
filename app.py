# NEW SECTIONS TO ADD TO app.py

def show_volume_analysis(analysis_results, show_debug=False):
    """Display volume analysis section"""
    if not st.session_state.get('show_volume_analysis', True):
        return
        
    enhanced_indicators = analysis_results.get('enhanced_indicators', {})
    volume_analysis = enhanced_indicators.get('volume_analysis', {})
    
    if 'error' in volume_analysis:
        st.warning(f"⚠️ Volume Analysis: {volume_analysis.get('error', 'Unknown error')}")
        return
    
    with st.expander("📊 Volume Analysis - 5d/30d Rolling & Trend Detection", expanded=True):
        
        # Format data for display
        from analysis.volume import format_volume_analysis_for_display
        display_data = format_volume_analysis_for_display(volume_analysis)
        
        if 'error' in display_data:
            st.error(f"Volume analysis display error: {display_data['error']}")
            return
        
        summary_metrics = display_data.get('summary_metrics', {})
        detailed_analysis = display_data.get('detailed_analysis', {})
        
        # Volume composite score with visual bar
        composite_score = summary_metrics.get('composite_score', 50)
        interpretation = summary_metrics.get('interpretation', 'Normal Volume')
        
        st.subheader(f"📊 Volume Composite Score: {composite_score}/100")
        
        # Create volume score bar (similar to technical score bar)
        if composite_score >= 80:
            score_color = "#00A86B"  # Green
        elif composite_score >= 60:
            score_color = "#9ACD32"  # Yellow-green
        elif composite_score >= 40:
            score_color = "#FFD700"  # Gold
        elif composite_score >= 20:
            score_color = "#FF8C00"  # Orange
        else:
            score_color = "#DC143C"  # Red
        
        volume_bar_html = f"""
        <div style="margin: 1rem 0; padding: 0.5rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                    border-radius: 8px; border: 1px solid #dee2e6;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-weight: 600; color: #495057;">Volume Strength</span>
                <span style="color: {score_color}; font-weight: 600;">{interpretation}</span>
            </div>
            <div style="width: 100%; height: 20px; background: linear-gradient(to right, 
                        #DC143C 0%, #FF8C00 25%, #FFD700 50%, #9ACD32 75%, #00A86B 100%); 
                        border-radius: 10px; position: relative;">
                <div style="position: absolute; left: {composite_score}%; top: 50%; transform: translate(-50%, -50%); 
                            width: 4px; height: 24px; background: white; border: 1px solid #333; border-radius: 2px;">
                </div>
            </div>
        </div>
        """
        st.markdown(volume_bar_html, unsafe_allow_html=True)
        
        # Primary volume metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_vol = summary_metrics.get('current_volume', 0)
            st.metric("Current Volume", f"{current_vol:,.0f}")
        
        with col2:
            vol_5d = summary_metrics.get('volume_5d_avg', 0)
            st.metric("5-Day Average", f"{vol_5d:,.0f}")
        
        with col3:
            vol_30d = summary_metrics.get('volume_30d_avg', 0)
            st.metric("30-Day Average", f"{vol_30d:,.0f}")
        
        with col4:
            relative_ratio = summary_metrics.get('relative_ratio', 1.0)
            deviation_pct = summary_metrics.get('deviation_pct', 0)
            st.metric("5d vs 30d Ratio", f"{relative_ratio:.2f}x", f"{deviation_pct:+.1f}%")
        
        # Volume trend and regime
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trend_direction = summary_metrics.get('trend_direction', 'Unknown')
            st.info(f"**Volume Trend:**\n{trend_direction}")
        
        with col2:
            regime = summary_metrics.get('regime_classification', 'Normal')
            st.info(f"**Volume Regime:**\n{regime}")
        
        with col3:
            breakout_type = summary_metrics.get('breakout_type', 'Normal Volume')
            st.info(f"**Breakout Status:**\n{breakout_type}")
        
        # Detailed breakdowns (expandable)
        with st.expander("📈 5-Day Rolling Volume Analysis", expanded=False):
            rolling_5d = detailed_analysis.get('5_day_rolling', {})
            if rolling_5d and 'error' not in rolling_5d:
                
                st.write("**Trend Analysis:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"• **Trend Direction:** {rolling_5d.get('trend_direction', 'N/A')}")
                    st.write(f"• **Trend Strength:** {rolling_5d.get('trend_strength', 'N/A')}")
                    st.write(f"• **Trend Slope:** {rolling_5d.get('trend_slope_pct', 0):.2f}%")
                
                with col2:
                    st.write(f"• **Volume Momentum:** {rolling_5d.get('volume_momentum', 0):.2f}%")
                    st.write(f"• **Current vs 5d Avg:** {rolling_5d.get('volume_vs_5d_pct', 0):+.1f}%")
                    st.write(f"• **5-Day Average:** {rolling_5d.get('current_5d_average', 0):,.0f}")
            else:
                st.warning("5-day analysis data not available")
        
        with st.expander("📊 30-Day Comparison Analysis", expanded=False):
            comparison_30d = detailed_analysis.get('30_day_comparison', {})
            if comparison_30d and 'error' not in comparison_30d:
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Comparison Metrics:**")
                    st.write(f"• **5-Day Average:** {comparison_30d.get('volume_5d_avg', 0):,.0f}")
                    st.write(f"• **30-Day Average:** {comparison_30d.get('volume_30d_avg', 0):,.0f}")
                    st.write(f"• **Relative Ratio:** {comparison_30d.get('relative_ratio', 0):.2f}x")
                
                with col2:
                    st.write("**Classification:**")
                    st.write(f"• **Regime:** {comparison_30d.get('regime_classification', 'N/A')}")
                    st.write(f"• **Significance:** {comparison_30d.get('significance', 'N/A')}")
                    st.write(f"• **Above 30d Avg:** {'Yes' if comparison_30d.get('above_30d_average', False) else 'No'}")
            else:
                st.warning("30-day comparison data not available")
        
        with st.expander("🚨 Breakout Detection Analysis", expanded=False):
            breakout = detailed_analysis.get('breakout_detection', {})
            if breakout and 'error' not in breakout:
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Breakout Metrics:**")
                    st.write(f"• **Z-Score:** {breakout.get('volume_zscore', 0):.2f}")
                    st.write(f"• **Percentile:** {breakout.get('volume_percentile', 0):.1f}%")
                    st.write(f"• **Is Breakout:** {'Yes' if breakout.get('is_breakout', False) else 'No'}")
                
                with col2:
                    st.write("**Classification:**")
                    st.write(f"• **Type:** {breakout.get('breakout_type', 'N/A')}")
                    st.write(f"• **Strength:** {breakout.get('breakout_strength', 'N/A')}")
                    st.write(f"• **Score:** {breakout.get('breakout_score', 0)}/100")
            else:
                st.warning("Breakout detection data not available")

def show_volatility_analysis(analysis_results, show_debug=False):
    """Display volatility analysis section"""
    if not st.session_state.get('show_volatility_analysis', True):
        return
        
    enhanced_indicators = analysis_results.get('enhanced_indicators', {})
    volatility_analysis = enhanced_indicators.get('volatility_analysis', {})
    
    if 'error' in volatility_analysis:
        st.warning(f"⚠️ Volatility Analysis: {volatility_analysis.get('error', 'Unknown error')}")
        return
    
    with st.expander("🌡️ Volatility Analysis - 5d/30d Rolling & Regime Detection", expanded=True):
        
        # Format data for display
        from analysis.volatility import format_volatility_analysis_for_display
        display_data = format_volatility_analysis_for_display(volatility_analysis)
        
        if 'error' in display_data:
            st.error(f"Volatility analysis display error: {display_data['error']}")
            return
        
        summary_metrics = display_data.get('summary_metrics', {})
        detailed_analysis = display_data.get('detailed_analysis', {})
        
        # Volatility composite score with visual bar
        composite_score = summary_metrics.get('composite_score', 50)
        interpretation = summary_metrics.get('interpretation', 'Normal Volatility')
        
        st.subheader(f"🌡️ Volatility Composite Score: {composite_score}/100")
        
        # Create volatility score bar
        if composite_score >= 85:
            score_color = "#DC143C"  # Red for extreme volatility
        elif composite_score >= 75:
            score_color = "#FF4500"  # Orange-red for high volatility
        elif composite_score >= 65:
            score_color = "#FF8C00"  # Orange for elevated volatility
        elif composite_score >= 35:
            score_color = "#FFD700"  # Gold for normal
        elif composite_score >= 25:
            score_color = "#9ACD32"  # Yellow-green for low volatility
        else:
            score_color = "#00A86B"  # Green for very low volatility
        
        volatility_bar_html = f"""
        <div style="margin: 1rem 0; padding: 0.5rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                    border-radius: 8px; border: 1px solid #dee2e6;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-weight: 600; color: #495057;">Volatility Environment</span>
                <span style="color: {score_color}; font-weight: 600;">{interpretation}</span>
            </div>
            <div style="width: 100%; height: 20px; background: linear-gradient(to right, 
                        #00A86B 0%, #9ACD32 20%, #FFD700 40%, #FF8C00 65%, #FF4500 80%, #DC143C 100%); 
                        border-radius: 10px; position: relative;">
                <div style="position: absolute; left: {composite_score}%; top: 50%; transform: translate(-50%, -50%); 
                            width: 4px; height: 24px; background: white; border: 1px solid #333; border-radius: 2px;">
                </div>
            </div>
        </div>
        """
        st.markdown(volatility_bar_html, unsafe_allow_html=True)
        
        # Primary volatility metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_5d_vol = summary_metrics.get('current_5d_volatility', 0)
            st.metric("5-Day Volatility", f"{current_5d_vol:.1f}%")
        
        with col2:
            current_30d_vol = summary_metrics.get('current_30d_volatility', 0)
            st.metric("30-Day Volatility", f"{current_30d_vol:.1f}%")
        
        with col3:
            relative_ratio = summary_metrics.get('relative_ratio', 1.0)
            deviation_pct = summary_metrics.get('deviation_pct', 0)
            st.metric("5d vs 30d Ratio", f"{relative_ratio:.2f}x", f"{deviation_pct:+.1f}%")
        
        with col4:
            market_env = summary_metrics.get('market_environment', 'Stable')
            st.metric("Market Environment", market_env)
        
        # Volatility trend and regime
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trend_direction = summary_metrics.get('trend_direction', 'Unknown')
            cycle_phase = summary_metrics.get('cycle_phase', 'Unknown')
            st.info(f"**Volatility Trend:**\n{trend_direction}\n\n**Cycle Phase:**\n{cycle_phase}")
        
        with col2:
            regime = summary_metrics.get('regime_classification', 'Normal Volatility')
            st.info(f"**Volatility Regime:**\n{regime}")
        
        with col3:
            options_implication = summary_metrics.get('options_implication', 'Normal Premium Environment')
            st.info(f"**Options Impact:**\n{options_implication}")
        
        # Detailed breakdowns (expandable)
        with st.expander("📈 5-Day Rolling Volatility Analysis", expanded=False):
            rolling_5d = detailed_analysis.get('5_day_rolling', {})
            if rolling_5d and 'error' not in rolling_5d:
                
                st.write("**Volatility Trend Analysis:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"• **Trend Direction:** {rolling_5d.get('trend_direction', 'N/A')}")
                    st.write(f"• **Trend Strength:** {rolling_5d.get('trend_strength', 'N/A')}")
                    st.write(f"• **Cycle Phase:** {rolling_5d.get('cycle_phase', 'N/A')}")
                
                with col2:
                    st.write(f"• **Volatility Momentum:** {rolling_5d.get('volatility_momentum', 0):.2f}%")
                    st.write(f"• **Volatility Acceleration:** {rolling_5d.get('volatility_acceleration', 0):.3f}")
                    st.write(f"• **Current Daily Vol:** {rolling_5d.get('current_daily_volatility', 0):.2f}%")
            else:
                st.warning("5-day volatility analysis data not available")
        
        with st.expander("📊 30-Day Volatility Comparison", expanded=False):
            comparison_30d = detailed_analysis.get('30_day_comparison', {})
            if comparison_30d and 'error' not in comparison_30d:
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Comparison Metrics:**")
                    st.write(f"• **5-Day Volatility:** {comparison_30d.get('volatility_5d', 0):.2f}%")
                    st.write(f"• **30-Day Volatility:** {comparison_30d.get('volatility_30d', 0):.2f}%")
                    st.write(f"• **Relative Ratio:** {comparison_30d.get('relative_ratio', 0):.2f}x")
                
                with col2:
                    st.write("**Environment Classification:**")
                    st.write(f"• **Regime:** {comparison_30d.get('regime_classification', 'N/A')}")
                    st.write(f"• **Market Environment:** {comparison_30d.get('market_environment', 'N/A')}")
                    st.write(f"• **Significance:** {comparison_30d.get('significance', 'N/A')}")
            else:
                st.warning("30-day volatility comparison data not available")
        
        with st.expander("🔍 Volatility Regime Detection", expanded=False):
            regime = detailed_analysis.get('regime_detection', {})
            if regime and 'error' not in regime:
                
                st.write("**Multi-Timeframe Volatility:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"• **10-Day Volatility:** {regime.get('current_vol_10d', 0):.2f}%")
                    st.write(f"• **20-Day Volatility:** {regime.get('current_vol_20d', 0):.2f}%")
                    st.write(f"• **50-Day Volatility:** {regime.get('current_vol_50d', 0):.2f}%")
                
                with col2:
                    st.write(f"• **20d Percentile:** {regime.get('vol_percentile_20d', 0):.1f}%")
                    st.write(f"• **50d Percentile:** {regime.get('vol_percentile_50d', 0):.1f}%")
                    st.write(f"• **Regime Score:** {regime.get('regime_score', 0)}/100")
                
                st.write("**Volatility Clustering:**")
                st.write(f"• **Pattern:** {regime.get('volatility_clustering', 'N/A')}")
                st.write(f"• **Recent High Vol Days:** {regime.get('high_vol_days_recent', 0)}/10")
                st.write(f"• **Recent Low Vol Days:** {regime.get('low_vol_days_recent', 0)}/10")
            else:
                st.warning("Volatility regime detection data not available")

# UPDATE TO SIDEBAR CONTROLS (add these toggles)
def update_sidebar_controls():
    """Add volume and volatility toggles to sidebar"""
    # Add to existing section control panel
    if 'show_volume_analysis' not in st.session_state:
        st.session_state.show_volume_analysis = True
    if 'show_volatility_analysis' not in st.session_state:
        st.session_state.show_volatility_analysis = True
    
    # Add to existing checkbox section
    with st.sidebar.expander("📋 Analysis Sections", expanded=False):
        st.write("**Toggle Analysis Sections:**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.show_technical_analysis = st.checkbox(
                "Technical Analysis", 
                value=st.session_state.show_technical_analysis,
                key="toggle_technical"
            )
            st.session_state.show_volume_analysis = st.checkbox(
                "Volume Analysis", 
                value=st.session_state.show_volume_analysis,
                key="toggle_volume"
            )
            st.session_state.show_volatility_analysis = st.checkbox(
                "Volatility Analysis", 
                value=st.session_state.show_volatility_analysis,
                key="toggle_volatility"
            )
            st.session_state.show_fundamental_analysis = st.checkbox(
                "Fundamental Analysis", 
                value=st.session_state.show_fundamental_analysis,
                key="toggle_fundamental"
            )
        
        with col2:
            st.session_state.show_market_correlation = st.checkbox(
                "Market Correlation", 
                value=st.session_state.show_market_correlation,
                key="toggle_correlation"
            )
            st.session_state.show_options_analysis = st.checkbox(
                "Options Analysis", 
                value=st.session_state.show_options_analysis,
                key="toggle_options"
            )
            st.session_state.show_confidence_intervals = st.checkbox(
                "Confidence Intervals", 
                value=st.session_state.show_confidence_intervals,
                key="toggle_confidence"
            )

# UPDATE TO MAIN ANALYSIS PIPELINE
def perform_enhanced_analysis_updated(symbol, period, show_debug=False):
    """Updated analysis pipeline with volume and volatility integration"""
    try:
        # Step 1-3: Existing data fetching and storage (unchanged)
        market_data = get_market_data_enhanced(symbol, period, show_debug)
        
        if market_data is None:
            st.error(f"❌ Could not fetch data for {symbol}")
            return None
        
        data_manager = get_data_manager()
        data_manager.store_market_data(symbol, market_data, show_debug)
        analysis_input = data_manager.get_market_data_for_analysis(symbol)
        
        if analysis_input is None:
            st.error("❌ Could not prepare analysis data")
            return None
        
        # Step 4: Calculate enhanced indicators using NEW technical analysis
        from analysis.technical import calculate_enhanced_technical_analysis
        enhanced_indicators = calculate_enhanced_technical_analysis(analysis_input)
        
        if 'error' in enhanced_indicators:
            st.error(f"❌ Technical analysis failed: {enhanced_indicators['error']}")
            return None
        
        # Steps 5-8: Existing market correlations, fundamental analysis, etc. (unchanged)
        market_correlations = calculate_market_correlations_enhanced(analysis_input, symbol, show_debug=show_debug)
        
        is_etf_symbol = is_etf(symbol)
        if is_etf_symbol:
            graham_score = {'score': 0, 'total_possible': 10, 'criteria': [], 'error': 'ETF - Fundamental analysis not applicable'}
            piotroski_score = {'score': 0, 'total_possible': 9, 'criteria': [], 'error': 'ETF - Fundamental analysis not applicable'}
        else:
            graham_score = calculate_graham_score(symbol, show_debug)
            piotroski_score = calculate_piotroski_score(symbol, show_debug)
        
        # Options levels calculation (unchanged)
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        volatility = comprehensive_technicals.get('volatility_20d', 20)
        underlying_beta = 1.0
        
        if market_correlations:
            for etf in ['SPY', 'QQQ', 'MAGS']:
                if etf in market_correlations and 'beta' in market_correlations[etf]:
                    try:
                        underlying_beta = abs(float(market_correlations[etf]['beta']))
                        break
                    except:
                        continue
        
        current_price = round(float(analysis_input['Close'].iloc[-1]), 2)
        options_levels = calculate_options_levels_enhanced(current_price, volatility, underlying_beta=underlying_beta)
        
        confidence_analysis = calculate_confidence_intervals(analysis_input)
        
        # Step 9: Build analysis results with enhanced indicators
        current_date = analysis_input.index[-1].strftime('%Y-%m-%d')
        
        # Update enhanced_indicators structure
        enhanced_indicators.update({
            'market_correlations': market_correlations,
            'options_levels': options_levels,
            'graham_score': graham_score,
            'piotroski_score': piotroski_score
        })
        
        analysis_results = {
            'symbol': symbol,
            'timestamp': current_date,
            'current_price': current_price,
            'enhanced_indicators': enhanced_indicators,
            'confidence_analysis': confidence_analysis,
            'system_status': 'OPERATIONAL - Volume/Volatility Integrated'
        }
        
        data_manager.store_analysis_results(symbol, analysis_results)
        return analysis_results
        
    except Exception as e:
        st.error(f"❌ Enhanced analysis failed: {str(e)}")
        return None

# UPDATE TO MAIN DISPLAY FUNCTION
def main_updated():
    """Updated main function with new analysis sections"""
    create_header()
    controls = create_sidebar_controls()
    
    if controls['analyze_button'] and controls['symbol']:
        add_to_recently_viewed(controls['symbol'])
        st.write("## 📊 VWV Trading Analysis - Enhanced with Volume & Volatility")
        
        with st.spinner(f"Analyzing {controls['symbol']} with new Volume & Volatility modules..."):
            
            analysis_results = perform_enhanced_analysis_updated(
                controls['symbol'], 
                controls['period'], 
                controls['show_debug']
            )
            
            if analysis_results:
                # Show all analysis sections (updated order)
                show_individual_technical_analysis(analysis_results, controls['show_debug'])
                show_volume_analysis(analysis_results, controls['show_debug'])  # NEW
                show_volatility_analysis(analysis_results, controls['show_debug'])  # NEW
                show_fundamental_analysis(analysis_results, controls['show_debug'])
                show_market_correlation_analysis(analysis_results, controls['show_debug'])
                show_options_analysis(analysis_results, controls['show_debug'])
                show_confidence_intervals(analysis_results, controls['show_debug'])
                
                if controls['show_debug']:
                    with st.expander("🐛 Debug Information", expanded=False):
                        st.write("### Enhanced Analysis Results (with Volume/Volatility)")
                        st.json(analysis_results, expanded=False)
    else:
        # Updated welcome message
        st.write("## 🚀 VWV Professional Trading System - Volume & Volatility Enhanced")
        st.write("**NEW:** Volume Analysis & Volatility Analysis sections with comprehensive 5d/30d rolling analysis")
        
        with st.expander("🆕 New Volume & Volatility Analysis Features", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### 📊 **Volume Analysis**")
                st.write("✅ **5-Day Rolling Volume** - Trend detection and momentum")
                st.write("✅ **30-Day Comparison** - Regime classification and deviations")
                st.write("✅ **Breakout Detection** - Z-score analysis and percentiles")
                st.write("✅ **Composite Scoring** - 15% weight in technical score")
                
            with col2:
                st.write("### 🌡️ **Volatility Analysis**")
                st.write("✅ **5-Day Rolling Volatility** - Expansion/contraction cycles")
                st.write("✅ **30-Day Comparison** - Market environment classification")
                st.write("✅ **Regime Detection** - Multi-timeframe analysis")
                st.write("✅ **Options Integration** - Premium environment analysis")
