"""
File: core/module_loader.py
Module Integration System for VWV Research And Analysis System  
Version: v4.2.2-MODULE-LOADER-2025-08-27-19-35-00-EST
PURPOSE: Safe module loading and registration with isolation guarantees
Last Updated: August 27, 2025 - 7:35 PM EST
"""

import logging
from typing import Dict, Any
import streamlit as st
from core.module_registry import get_module_registry, safe_module_import, ModuleStatus

logger = logging.getLogger(__name__)

class ModuleLoader:
    """Loads and registers all analysis modules safely"""
    
    def __init__(self):
        self.registry = get_module_registry()
        self.loaded_modules = set()
    
    def load_all_modules(self):
        """Load all available analysis modules with safe isolation"""
        
        # Load core modules (always available)
        self._load_technical_analysis()
        self._load_fundamental_analysis()
        
        # Load optional modules with safe fallbacks
        self._load_volatility_analysis()
        self._load_volume_analysis()
        self._load_baldwin_indicator()
        self._load_market_correlation()
        self._load_options_analysis()
        
        # Initialize session state for all registered modules
        self.registry.initialize_session_state()
        
        logger.info(f"Module loading complete. Loaded: {len(self.loaded_modules)} modules")
    
    def _load_technical_analysis(self):
        """Load technical analysis module - core functionality"""
        try:
            # Import functions safely
            functions = safe_module_import('analysis.technical', [
                'calculate_comprehensive_technicals',
                'calculate_composite_technical_score'
            ])
            
            if functions['calculate_comprehensive_technicals'] and functions['calculate_composite_technical_score']:
                # Create analysis wrapper
                def technical_analysis(data):
                    comprehensive = functions['calculate_comprehensive_technicals'](data)
                    score, details = functions['calculate_composite_technical_score']({'enhanced_indicators': {'comprehensive_technicals': comprehensive}})
                    return {
                        'comprehensive_technicals': comprehensive,
                        'composite_score': score,
                        'score_details': details,
                        'analysis_success': True
                    }
                
                # Create display wrapper
                def technical_display(analysis_results, show_debug=False):
                    if not st.session_state.get('show_technical_analysis', True):
                        return
                    
                    symbol = analysis_results.get('symbol', 'Unknown')
                    with st.expander(f"Technical Analysis - {symbol}", expanded=True):
                        tech_data = analysis_results.get('enhanced_indicators', {}).get('technical_analysis', {})
                        
                        if tech_data and tech_data.get('analysis_success'):
                            from ui.components import create_technical_score_bar
                            create_technical_score_bar(tech_data.get('composite_score', 50), tech_data.get('score_details'))
                            
                            # Display key metrics
                            comprehensive = tech_data.get('comprehensive_technicals', {})
                            if comprehensive:
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    current_price = comprehensive.get('current_price', 0)
                                    st.metric("Current Price", f"${current_price:.2f}")
                                with col2:
                                    rsi_value = comprehensive.get('rsi', 50)
                                    rsi_desc = "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                                    st.metric("RSI(14)", f"{rsi_value:.1f}", rsi_desc)
                                with col3:
                                    vwap_position = comprehensive.get('vwap_position', 0)
                                    vwap_desc = "Above VWAP" if vwap_position > 0 else "Below VWAP" if vwap_position < 0 else "At VWAP"
                                    st.metric("VWAP Position", f"{vwap_position:+.2f}%", vwap_desc)
                                with col4:
                                    trend_strength = comprehensive.get('trend_strength', 50)
                                    trend_desc = "Strong" if trend_strength > 70 else "Weak" if trend_strength < 30 else "Moderate"
                                    st.metric("Trend Strength", f"{trend_strength:.0f}/100", trend_desc)
                        else:
                            st.warning("Technical analysis not available")
                
                # Register module
                self.registry.register_module(
                    'technical_analysis',
                    'Technical Analysis',
                    'Comprehensive technical analysis with composite scoring',
                    'v4.2.2',
                    technical_analysis,
                    technical_display
                )
                self.loaded_modules.add('technical_analysis')
                
        except Exception as e:
            logger.error(f"Failed to load technical analysis: {e}")
    
    def _load_fundamental_analysis(self):
        """Load fundamental analysis module"""
        try:
            functions = safe_module_import('analysis.fundamental', [
                'calculate_graham_score',
                'calculate_piotroski_score'
            ])
            
            if functions['calculate_graham_score'] or functions['calculate_piotroski_score']:
                def fundamental_analysis(symbol):
                    result = {}
                    if functions['calculate_graham_score']:
                        result['graham_score'] = functions['calculate_graham_score'](symbol)
                    if functions['calculate_piotroski_score']:
                        result['piotroski_score'] = functions['calculate_piotroski_score'](symbol)
                    result['analysis_success'] = True
                    return result
                
                def fundamental_display(analysis_results, show_debug=False):
                    if not st.session_state.get('show_fundamental_analysis', True):
                        return
                    
                    symbol = analysis_results.get('symbol', 'Unknown')
                    with st.expander(f"Fundamental Analysis - {symbol}", expanded=True):
                        fund_data = analysis_results.get('enhanced_indicators', {}).get('fundamental_analysis', {})
                        
                        if fund_data and fund_data.get('analysis_success'):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                graham = fund_data.get('graham_score', {})
                                if graham and 'error' not in graham:
                                    st.subheader("Graham Score")
                                    score = graham.get('total_score', 0)
                                    max_score = graham.get('max_possible_score', 10)
                                    st.metric("Graham Score", f"{score}/{max_score}")
                            
                            with col2:
                                piotroski = fund_data.get('piotroski_score', {})
                                if piotroski and 'error' not in piotroski:
                                    st.subheader("Piotroski Score")
                                    score = piotroski.get('total_score', 0)
                                    st.metric("Piotroski Score", f"{score}/9")
                        else:
                            st.warning("Fundamental analysis not available")
                
                self.registry.register_module(
                    'fundamental_analysis',
                    'Fundamental Analysis',
                    'Graham and Piotroski fundamental scoring',
                    'v4.2.2',
                    fundamental_analysis,
                    fundamental_display
                )
                self.loaded_modules.add('fundamental_analysis')
                
        except Exception as e:
            logger.error(f"Failed to load fundamental analysis: {e}")
    
    def _load_volatility_analysis(self):
        """Load volatility analysis module with enhanced display"""
        try:
            functions = safe_module_import('analysis.volatility', [
                'calculate_complete_volatility_analysis'
            ])
            
            if functions['calculate_complete_volatility_analysis']:
                def volatility_display(analysis_results, show_debug=False):
                    if not st.session_state.get('show_volatility_analysis', True):
                        return
                    
                    symbol = analysis_results.get('symbol', 'Unknown')
                    with st.expander(f"Advanced Volatility Analysis - {symbol}", expanded=True):
                        vol_data = analysis_results.get('enhanced_indicators', {}).get('volatility_analysis', {})
                        
                        if vol_data and vol_data.get('analysis_success'):
                            # Main metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                vol_20d = vol_data.get('volatility_20d', 0)
                                st.metric("20-Day Volatility", f"{vol_20d:.1f}%")
                            with col2:
                                vol_10d = vol_data.get('volatility_10d', 0)
                                st.metric("10-Day Volatility", f"{vol_10d:.1f}%")
                            with col3:
                                vol_score = vol_data.get('volatility_score', 50)
                                st.metric("Volatility Score", f"{vol_score:.0f}/100")
                            with col4:
                                vol_regime = vol_data.get('volatility_regime', 'Normal')
                                st.metric("Volatility Regime", vol_regime)
                            
                            # Advanced metrics
                            st.subheader("Advanced Volatility Estimators")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                garch_vol = vol_data.get('garch_volatility', vol_20d)
                                st.metric("GARCH Volatility", f"{garch_vol:.1f}%", "Advanced Modeling")
                            with col2:
                                parkinson_vol = vol_data.get('parkinson_volatility', vol_20d)
                                st.metric("Parkinson Est.", f"{parkinson_vol:.1f}%", "High-Low Range")
                            with col3:
                                yang_zhang_vol = vol_data.get('yang_zhang_volatility', vol_20d)
                                st.metric("Yang-Zhang Est.", f"{yang_zhang_vol:.1f}%", "Combined Approach")
                            with col4:
                                realized_vol = vol_data.get('realized_volatility', vol_20d)
                                st.metric("Realized Vol", f"{realized_vol:.1f}%", "Actual Movements")
                            
                            # Regime and strategy
                            st.subheader("Volatility Environment & Strategy")
                            col1, col2 = st.columns(2)
                            with col1:
                                options_strategy = vol_data.get('options_strategy', 'Directional Strategies')
                                st.info(f"**Options Strategy:** {options_strategy}")
                            with col2:
                                trading_implications = vol_data.get('trading_implications', 'Monitor volatility for position sizing.')
                                st.info(f"**Trading Implications:** {trading_implications}")
                            
                            # Component breakdown
                            if vol_data.get('indicators') and vol_data.get('scores') and show_debug:
                                with st.expander("14-Component Volatility Breakdown", expanded=False):
                                    st.write("**All Volatility Indicators with Weights:**")
                                    
                                    indicators = vol_data.get('indicators', {})
                                    scores = vol_data.get('scores', {})
                                    weights = vol_data.get('weights', {})
                                    contributions = vol_data.get('contributions', {})
                                    
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
                                        import pandas as pd
                                        df = pd.DataFrame(component_data)
                                        st.dataframe(df, use_container_width=True, hide_index=True)
                                        
                                        st.write("**Methodology:**")
                                        st.write("• Weighted composite score from 14 research-based indicators")
                                        st.write("• Advanced estimators: GARCH, Yang-Zhang, Parkinson, Garman-Klass")
                                        st.write("• Volatility dynamics: momentum, clustering, mean reversion")
                                        st.write("• Score range: 0-100 with regime classification")
                        else:
                            st.warning("Volatility analysis not available - insufficient data")
                
                self.registry.register_module(
                    'volatility_analysis',
                    'Volatility Analysis',
                    'Advanced volatility analysis with 14 indicators and weighted scoring',
                    'v4.2.2',
                    functions['calculate_complete_volatility_analysis'],
                    volatility_display
                )
                self.loaded_modules.add('volatility_analysis')
                
        except Exception as e:
            logger.error(f"Failed to load volatility analysis: {e}")
    
    def _load_volume_analysis(self):
        """Load volume analysis module"""
        try:
            functions = safe_module_import('analysis.volume', [
                'calculate_complete_volume_analysis'
            ])
            
            if functions['calculate_complete_volume_analysis']:
                def volume_display(analysis_results, show_debug=False):
                    if not st.session_state.get('show_volume_analysis', True):
                        return
                    
                    symbol = analysis_results.get('symbol', 'Unknown')
                    with st.expander(f"Volume Analysis - {symbol}", expanded=True):
                        vol_data = analysis_results.get('enhanced_indicators', {}).get('volume_analysis', {})
                        
                        if vol_data and vol_data.get('analysis_success'):
                            from ui.components import format_large_number
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                current_volume = vol_data.get('current_volume', 0)
                                st.metric("Current Volume", format_large_number(current_volume))
                            with col2:
                                volume_5d_avg = vol_data.get('volume_5d_avg', 0)
                                st.metric("5-Day Avg", format_large_number(volume_5d_avg))
                            with col3:
                                volume_ratio = vol_data.get('volume_ratio', 1.0)
                                ratio_desc = "High" if volume_ratio > 2.0 else "Above Normal" if volume_ratio > 1.5 else "Normal" if volume_ratio > 0.8 else "Low"
                                st.metric("Volume Ratio", f"{volume_ratio:.2f}x", ratio_desc)
                            with col4:
                                volume_trend = vol_data.get('volume_5d_trend', 0)
                                trend_desc = "Rising" if volume_trend > 10 else "Falling" if volume_trend < -10 else "Stable"
                                st.metric("5D Trend", f"{volume_trend:+.1f}%", trend_desc)
                            
                            # Volume environment
                            volume_regime = vol_data.get('volume_regime', 'Normal')
                            trading_implications = vol_data.get('trading_implications', 'Monitor volume for confirmation.')
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.info(f"**Volume Regime:** {volume_regime}")
                            with col2:
                                st.info(f"**Implications:** {trading_implications}")
                        else:
                            st.warning("Volume analysis not available")
                
                self.registry.register_module(
                    'volume_analysis',
                    'Volume Analysis',
                    'Volume strength analysis with multi-timeframe confirmation',
                    'v4.2.2',
                    functions['calculate_complete_volume_analysis'],
                    volume_display
                )
                self.loaded_modules.add('volume_analysis')
                
        except Exception as e:
            logger.error(f"Failed to load volume analysis: {e}")
    
    def _load_baldwin_indicator(self):
        """Load Baldwin Market Regime Indicator - PRIORITY RESTORATION"""
        try:
            functions = safe_module_import('analysis.baldwin_indicator', [
                'calculate_baldwin_indicator_complete',
                'format_baldwin_for_display'
            ])
            
            if functions['calculate_baldwin_indicator_complete']:
                def baldwin_display(analysis_results=None, show_debug=False):
                    if not st.session_state.get('show_baldwin_indicator', True):
                        return
                    
                    with st.expander("Baldwin Market Regime Indicator", expanded=True):
                        # Calculate Baldwin if not provided in analysis results
                        baldwin_results = None
                        if analysis_results:
                            baldwin_results = analysis_results.get('enhanced_indicators', {}).get('baldwin_indicator', {})
                        
                        if not baldwin_results:
                            baldwin_results = functions['calculate_baldwin_indicator_complete'](show_debug=show_debug)
                        
                        if baldwin_results and 'error' not in baldwin_results:
                            # Format for display
                            if functions['format_baldwin_for_display']:
                                display_data = functions['format_baldwin_for_display'](baldwin_results)
                            else:
                                display_data = baldwin_results
                            
                            if 'error' not in display_data:
                                # Main regime display
                                regime = display_data.get('regime', baldwin_results.get('market_regime', 'YELLOW'))
                                regime_score = display_data.get('overall_score', baldwin_results.get('baldwin_score', 50))
                                strategy = display_data.get('strategy', baldwin_results.get('strategy', 'Monitor conditions'))
                                
                                # Color coding
                                regime_colors = {
                                    'GREEN': '#32CD32',
                                    'YELLOW': '#FFD700', 
                                    'RED': '#DC143C'
                                }
                                regime_color = regime_colors.get(regime, '#FFD700')
                                
                                # Display regime
                                col1, col2, col3 = st.columns([2, 1, 1])
                                with col1:
                                    st.markdown(f"""
                                    <div style="padding: 1rem; background: {regime_color}20; 
                                                 border-left: 4px solid {regime_color}; border-radius: 8px;">
                                        <h3 style="color: {regime_color}; margin: 0;">
                                            Market Regime: {regime}
                                        </h3>
                                        <p style="margin: 0.5rem 0 0 0; font-weight: 600;">
                                            {strategy}
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.metric("Baldwin Score", f"{regime_score:.1f}/100")
                                
                                with col3:
                                    confidence = "High" if abs(regime_score - 50) > 25 else "Medium" if abs(regime_score - 50) > 15 else "Low"
                                    st.metric("Confidence", confidence)
                                
                                # Component breakdown if available
                                if 'components' in baldwin_results:
                                    with st.expander("Baldwin Component Analysis", expanded=False):
                                        components = baldwin_results['components']
                                        
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            momentum = components.get('momentum', {})
                                            st.metric("Momentum (60%)", f"{momentum.get('component_score', 50):.1f}/100")
                                        with col2:
                                            liquidity = components.get('liquidity', {})
                                            st.metric("Liquidity (25%)", f"{liquidity.get('component_score', 50):.1f}/100")
                                        with col3:
                                            sentiment = components.get('sentiment', {})
                                            st.metric("Sentiment (15%)", f"{sentiment.get('component_score', 50):.1f}/100")
                                        
                                        st.write("**Baldwin Methodology:**")
                                        st.write("• **Momentum Analysis (60%)**: Broad market trends and internal strength")
                                        st.write("• **Liquidity Analysis (25%)**: Dollar strength and safe-haven demand")
                                        st.write("• **Sentiment Analysis (15%)**: Smart money positioning signals")
                            else:
                                st.error(f"Baldwin display error: {display_data.get('error', 'Unknown error')}")
                        else:
                            st.warning("Baldwin Market Regime Indicator not available")
                            error_msg = baldwin_results.get('error', 'Unknown error') if baldwin_results else 'No data'
                            if show_debug:
                                st.error(f"Baldwin error: {error_msg}")
                            
                            # Show methodology even when unavailable
                            st.info("""
                            **About Baldwin Market Regime Indicator:**
                            
                            Comprehensive market assessment combining momentum, liquidity, and sentiment 
                            analysis to provide clear GREEN/YELLOW/RED signals for risk positioning.
                            """)
                
                self.registry.register_module(
                    'baldwin_indicator',
                    'Baldwin Market Regime',
                    'Market regime indicator with GREEN/YELLOW/RED signals',
                    'v4.2.2',
                    functions['calculate_baldwin_indicator_complete'],
                    baldwin_display
                )
                self.loaded_modules.add('baldwin_indicator')
                
        except Exception as e:
            logger.error(f"Failed to load Baldwin indicator: {e}")
    
    def _load_market_correlation(self):
        """Load market correlation analysis"""
        try:
            functions = safe_module_import('analysis.market', [
                'calculate_market_correlations_enhanced',
                'calculate_breakout_breakdown_analysis'
            ])
            
            if functions['calculate_market_correlations_enhanced']:
                def market_analysis(symbol, period):
                    correlations = functions['calculate_market_correlations_enhanced'](symbol, period)
                    return {
                        'market_correlations': correlations,
                        'analysis_success': True
                    }
                
                def market_display(analysis_results, show_debug=False):
                    if not st.session_state.get('show_market_correlation', True):
                        return
                    
                    symbol = analysis_results.get('symbol', 'Unknown')
                    with st.expander(f"Market Correlation Analysis - {symbol}", expanded=True):
                        market_data = analysis_results.get('enhanced_indicators', {}).get('market_analysis', {})
                        
                        if market_data and market_data.get('analysis_success'):
                            correlations = market_data.get('market_correlations', {})
                            if correlations and 'error' not in correlations:
                                corr_data = correlations.get('correlations', {})
                                
                                if corr_data:
                                    st.subheader("ETF Correlations")
                                    
                                    corr_list = []
                                    for etf, corr_value in corr_data.items():
                                        if isinstance(corr_value, (int, float)):
                                            strength = "Strong" if abs(corr_value) > 0.7 else "Moderate" if abs(corr_value) > 0.4 else "Weak"
                                            direction = "Positive" if corr_value > 0 else "Negative"
                                            corr_list.append({
                                                'ETF': etf,
                                                'Correlation': f"{corr_value:.3f}",
                                                'Strength': strength,
                                                'Direction': direction
                                            })
                                    
                                    if corr_list:
                                        import pandas as pd
                                        df_corr = pd.DataFrame(corr_list)
                                        st.dataframe(df_corr, use_container_width=True, hide_index=True)
                            else:
                                st.warning("Market correlation data not available")
                        else:
                            st.warning("Market correlation analysis not available")
                
                self.registry.register_module(
                    'market_analysis',
                    'Market Correlation',
                    'ETF correlation analysis and market positioning',
                    'v4.2.2',
                    market_analysis,
                    market_display
                )
                self.loaded_modules.add('market_analysis')
                
        except Exception as e:
            logger.error(f"Failed to load market correlation: {e}")
    
    def _load_options_analysis(self):
        """Load options analysis module"""
        try:
            functions = safe_module_import('analysis.options', [
                'calculate_options_levels_enhanced',
                'calculate_confidence_intervals'
            ])
            
            if functions['calculate_options_levels_enhanced']:
                def options_analysis(data):
                    options_levels = functions['calculate_options_levels_enhanced'](data)
                    confidence_intervals = None
                    if functions['calculate_confidence_intervals']:
                        confidence_intervals = functions['calculate_confidence_intervals'](data)
                    
                    return {
                        'options_levels': options_levels,
                        'confidence_intervals': confidence_intervals,
                        'analysis_success': True
                    }
                
                def options_display(analysis_results, show_debug=False):
                    if not st.session_state.get('show_options_analysis', True):
                        return
                    
                    symbol = analysis_results.get('symbol', 'Unknown')
                    with st.expander(f"Options Analysis - {symbol}", expanded=True):
                        options_data = analysis_results.get('enhanced_indicators', {}).get('options_analysis', {})
                        
                        if options_data and options_data.get('analysis_success'):
                            options_levels = options_data.get('options_levels', {})
                            
                            if options_levels and 'error' not in options_levels:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    current_price = options_levels.get('current_price', 0)
                                    st.metric("Current Price", f"${current_price:.2f}")
                                with col2:
                                    support_level = options_levels.get('support_level', 0)
                                    st.metric("Support Level", f"${support_level:.2f}")
                                with col3:
                                    resistance_level = options_levels.get('resistance_level', 0)
                                    st.metric("Resistance Level", f"${resistance_level:.2f}")
                            else:
                                st.warning("Options levels not available")
                        else:
                            st.warning("Options analysis not available")
                
                self.registry.register_module(
                    'options_analysis',
                    'Options Analysis',
                    'Options levels and Greeks analysis with confidence intervals',
                    'v4.2.2',
                    options_analysis,
                    options_display
                )
                self.loaded_modules.add('options_analysis')
                
        except Exception as e:
            logger.error(f"Failed to load options analysis: {e}")
    
    def get_registry(self):
        """Get the module registry"""
        return self.registry

# Global module loader instance
_module_loader = None

def get_module_loader():
    """Get the global module loader instance"""
    global _module_loader
    if _module_loader is None:
        _module_loader = ModuleLoader()
        _module_loader.load_all_modules()
    return _module_loader
