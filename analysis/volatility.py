"""
Enhanced Volatility analysis module for VWV Trading System v4.2.1
COMPREHENSIVE VOLATILITY ANALYSIS with 14 sophisticated indicators and composite scoring
Date: August 21, 2025 - 3:40 PM EST
Enhancement: Complete volatility analysis with weighted composite scoring system
Status: Production Ready - Enhanced from basic placeholder to comprehensive analysis
"""
import pandas as pd
import numpy as np
import streamlit as st
import logging
from typing import Dict, Any, Optional, Tuple
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

# Research-based weights for volatility composite scoring
VOLATILITY_INDICATOR_WEIGHTS = {
    'historical_volatility_20d': 0.15,      # Primary volatility measure
    'historical_volatility_10d': 0.12,      # Short-term volatility
    'realized_volatility': 0.13,            # Actual price movements
    'volatility_percentile': 0.11,          # Relative positioning
    'volatility_rank': 0.09,                # Ranking system
    'garch_volatility': 0.08,               # Advanced modeling
    'parkinson_volatility': 0.07,           # High-low based
    'garman_klass_volatility': 0.06,        # OHLC estimator
    'rogers_satchell_volatility': 0.05,     # Drift-independent
    'yang_zhang_volatility': 0.04,          # Combined estimator
    'volatility_of_volatility': 0.03,       # Second-order vol
    'volatility_momentum': 0.03,            # Rate of change
    'volatility_mean_reversion': 0.02,      # Mean reversion tendency
    'volatility_clustering': 0.02           # Persistence analysis
}

@safe_calculation_wrapper
def calculate_historical_volatility(returns: pd.Series, window: int = 20) -> float:
    """Calculate historical volatility over specified window"""
    try:
        if len(returns) < window:
            return 20.0  # Default fallback
        vol = returns.rolling(window).std().iloc[-1] * np.sqrt(252) * 100
        return float(vol) if not pd.isna(vol) else 20.0
    except Exception as e:
        logger.warning(f"Historical volatility calculation error: {e}")
        return 20.0

@safe_calculation_wrapper
def calculate_realized_volatility(data: pd.DataFrame) -> float:
    """Calculate realized volatility from intraday movements"""
    try:
        if len(data) < 5:
            return 20.0
        # Use high-low range as proxy for intraday movements
        hl_returns = np.log(data['High'] / data['Low'])
        realized_vol = hl_returns.tail(20).std() * np.sqrt(252) * 100
        return float(realized_vol) if not pd.isna(realized_vol) else 20.0
    except Exception as e:
        logger.warning(f"Realized volatility calculation error: {e}")
        return 20.0

@safe_calculation_wrapper
def calculate_volatility_percentile(returns: pd.Series, current_vol: float, lookback: int = 252) -> float:
    """Calculate current volatility percentile vs historical distribution"""
    try:
        if len(returns) < lookback:
            return 50.0
        
        # Calculate rolling volatilities
        rolling_vols = returns.rolling(20).std() * np.sqrt(252) * 100
        valid_vols = rolling_vols.dropna().tail(lookback)
        
        if len(valid_vols) == 0:
            return 50.0
            
        percentile = (valid_vols < current_vol).sum() / len(valid_vols) * 100
        return float(percentile)
    except Exception as e:
        logger.warning(f"Volatility percentile calculation error: {e}")
        return 50.0

@safe_calculation_wrapper
def calculate_volatility_rank(returns: pd.Series, current_vol: float, lookback: int = 252) -> float:
    """Calculate volatility rank (0-100 scale)"""
    try:
        if len(returns) < lookback:
            return 50.0
        
        rolling_vols = returns.rolling(20).std() * np.sqrt(252) * 100
        valid_vols = rolling_vols.dropna().tail(lookback)
        
        if len(valid_vols) == 0:
            return 50.0
            
        min_vol = valid_vols.min()
        max_vol = valid_vols.max()
        
        if max_vol == min_vol:
            return 50.0
            
        rank = (current_vol - min_vol) / (max_vol - min_vol) * 100
        return float(np.clip(rank, 0, 100))
    except Exception as e:
        logger.warning(f"Volatility rank calculation error: {e}")
        return 50.0

@safe_calculation_wrapper
def calculate_garch_volatility(returns: pd.Series) -> float:
    """Simple GARCH(1,1) volatility estimate"""
    try:
        if len(returns) < 30:
            return 20.0
        
        # Simplified GARCH implementation
        returns_clean = returns.dropna().tail(252)  # Last year of data
        if len(returns_clean) < 30:
            return 20.0
        
        # Initial variance estimate
        variance = returns_clean.var()
        alpha, beta, omega = 0.1, 0.8, variance * 0.1  # Standard parameters
        
        # Simple GARCH iteration
        for ret in returns_clean.tail(30):
            variance = omega + alpha * (ret ** 2) + beta * variance
        
        garch_vol = np.sqrt(variance * 252) * 100
        return float(garch_vol) if not pd.isna(garch_vol) else 20.0
    except Exception as e:
        logger.warning(f"GARCH volatility calculation error: {e}")
        return 20.0

@safe_calculation_wrapper
def calculate_parkinson_volatility(data: pd.DataFrame, window: int = 20) -> float:
    """Parkinson volatility estimator using high-low range"""
    try:
        if len(data) < window:
            return 20.0
        
        hl_ratio = np.log(data['High'] / data['Low'])
        parkinson_var = (hl_ratio ** 2).rolling(window).mean() / (4 * np.log(2))
        parkinson_vol = np.sqrt(parkinson_var.iloc[-1] * 252) * 100
        
        return float(parkinson_vol) if not pd.isna(parkinson_vol) else 20.0
    except Exception as e:
        logger.warning(f"Parkinson volatility calculation error: {e}")
        return 20.0

@safe_calculation_wrapper
def calculate_garman_klass_volatility(data: pd.DataFrame, window: int = 20) -> float:
    """Garman-Klass volatility estimator using OHLC"""
    try:
        if len(data) < window:
            return 20.0
        
        # Garman-Klass formula components
        hl_component = 0.5 * (np.log(data['High'] / data['Low']) ** 2)
        co_component = (2 * np.log(2) - 1) * (np.log(data['Close'] / data['Open']) ** 2)
        
        gk_var = (hl_component - co_component).rolling(window).mean()
        gk_vol = np.sqrt(gk_var.iloc[-1] * 252) * 100
        
        return float(gk_vol) if not pd.isna(gk_vol) else 20.0
    except Exception as e:
        logger.warning(f"Garman-Klass volatility calculation error: {e}")
        return 20.0

@safe_calculation_wrapper
def calculate_rogers_satchell_volatility(data: pd.DataFrame, window: int = 20) -> float:
    """Rogers-Satchell drift-independent volatility estimator"""
    try:
        if len(data) < window:
            return 20.0
        
        # Rogers-Satchell formula
        rs_component = (np.log(data['High'] / data['Close']) * np.log(data['High'] / data['Open']) +
                       np.log(data['Low'] / data['Close']) * np.log(data['Low'] / data['Open']))
        
        rs_var = rs_component.rolling(window).mean()
        rs_vol = np.sqrt(rs_var.iloc[-1] * 252) * 100
        
        return float(rs_vol) if not pd.isna(rs_vol) else 20.0
    except Exception as e:
        logger.warning(f"Rogers-Satchell volatility calculation error: {e}")
        return 20.0

@safe_calculation_wrapper
def calculate_yang_zhang_volatility(data: pd.DataFrame, window: int = 20) -> float:
    """Yang-Zhang volatility estimator combining multiple approaches"""
    try:
        if len(data) < window:
            return 20.0
        
        # Yang-Zhang combines overnight, open-to-close, and Rogers-Satchell
        overnight = np.log(data['Open'] / data['Close'].shift(1))
        open_to_close = np.log(data['Close'] / data['Open'])
        
        # Rogers-Satchell component
        rs = (np.log(data['High'] / data['Close']) * np.log(data['High'] / data['Open']) +
              np.log(data['Low'] / data['Close']) * np.log(data['Low'] / data['Open']))
        
        # Yang-Zhang combination (simplified)
        yz_var = (overnight.rolling(window).var() + 
                  0.164 * open_to_close.rolling(window).var() + 
                  rs.rolling(window).mean())
        
        yz_vol = np.sqrt(yz_var.iloc[-1] * 252) * 100
        return float(yz_vol) if not pd.isna(yz_vol) else 20.0
    except Exception as e:
        logger.warning(f"Yang-Zhang volatility calculation error: {e}")
        return 20.0

@safe_calculation_wrapper
def calculate_volatility_of_volatility(returns: pd.Series, window: int = 20) -> float:
    """Calculate volatility of volatility (second-order volatility)"""
    try:
        if len(returns) < window * 2:
            return 5.0
        
        # Rolling volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
        
        # Volatility of the rolling volatility
        vol_of_vol = rolling_vol.rolling(window).std()
        
        return float(vol_of_vol.iloc[-1]) if not pd.isna(vol_of_vol.iloc[-1]) else 5.0
    except Exception as e:
        logger.warning(f"Volatility of volatility calculation error: {e}")
        return 5.0

@safe_calculation_wrapper
def calculate_volatility_momentum(returns: pd.Series, window: int = 20) -> float:
    """Calculate rate of change in volatility"""
    try:
        if len(returns) < window * 2:
            return 0.0
        
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
        vol_momentum = rolling_vol.pct_change(window).iloc[-1] * 100
        
        return float(vol_momentum) if not pd.isna(vol_momentum) else 0.0
    except Exception as e:
        logger.warning(f"Volatility momentum calculation error: {e}")
        return 0.0

@safe_calculation_wrapper
def calculate_volatility_mean_reversion(returns: pd.Series, window: int = 60) -> float:
    """Calculate volatility mean reversion tendency"""
    try:
        if len(returns) < window:
            return 0.5
        
        rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100
        vol_series = rolling_vol.dropna().tail(window)
        
        if len(vol_series) < 30:
            return 0.5
        
        # Simple mean reversion measure: correlation with lagged deviation from mean
        vol_mean = vol_series.mean()
        vol_deviation = vol_series - vol_mean
        lagged_deviation = vol_deviation.shift(1)
        
        # Mean reversion coefficient (negative correlation indicates mean reversion)
        correlation = vol_deviation.corr(lagged_deviation)
        mean_reversion = -correlation if not pd.isna(correlation) else 0.0
        
        # Scale to 0-1 range
        mean_reversion_scaled = (mean_reversion + 1) / 2
        return float(np.clip(mean_reversion_scaled, 0, 1))
    except Exception as e:
        logger.warning(f"Volatility mean reversion calculation error: {e}")
        return 0.5

@safe_calculation_wrapper
def calculate_volatility_clustering(returns: pd.Series, window: int = 60) -> float:
    """Calculate volatility clustering coefficient"""
    try:
        if len(returns) < window:
            return 0.5
        
        # Calculate squared returns as proxy for volatility
        squared_returns = returns ** 2
        recent_squared = squared_returns.tail(window)
        
        if len(recent_squared) < 30:
            return 0.5
        
        # Autocorrelation in squared returns indicates clustering
        clustering = recent_squared.autocorr(lag=1)
        
        # Scale to 0-1 range
        clustering_scaled = (clustering + 1) / 2 if not pd.isna(clustering) else 0.5
        return float(np.clip(clustering_scaled, 0, 1))
    except Exception as e:
        logger.warning(f"Volatility clustering calculation error: {e}")
        return 0.5

@safe_calculation_wrapper
def determine_volatility_regime(vol_percentile: float, vol_rank: float) -> Tuple[str, str]:
    """Determine volatility regime and options strategy"""
    try:
        # Combine percentile and rank for regime classification
        combined_metric = (vol_percentile + vol_rank) / 2
        
        if combined_metric >= 80:
            regime = "Extreme High Volatility"
            strategy = "Sell Premium (Strangles/Straddles)"
        elif combined_metric >= 65:
            regime = "High Volatility"
            strategy = "Sell Premium (Iron Condors)"
        elif combined_metric >= 35:
            regime = "Normal Volatility"
            strategy = "Directional Strategies"
        elif combined_metric >= 20:
            regime = "Low Volatility"
            strategy = "Buy Premium (Long Options)"
        else:
            regime = "Extremely Low Volatility"
            strategy = "Buy Premium (Calendar Spreads)"
        
        return regime, strategy
    except Exception as e:
        logger.warning(f"Volatility regime determination error: {e}")
        return "Normal Volatility", "Directional Strategies"

@safe_calculation_wrapper
def get_volatility_trading_implications(regime: str, percentile: float) -> str:
    """Get detailed trading implications based on volatility analysis"""
    try:
        implications = []
        
        if "Extreme High" in regime:
            implications.extend([
                "• Volatility at extreme levels - Premium selling opportunities",
                "• Consider iron condors or strangles",
                "• Reduce position sizes due to high risk",
                "• Monitor for volatility contraction"
            ])
        elif "High" in regime:
            implications.extend([
                "• Elevated volatility favors premium selling",
                "• Iron condors and covered calls attractive",
                "• Avoid buying premium unless strong directional conviction",
                "• Watch for mean reversion opportunities"
            ])
        elif "Low" in regime:
            implications.extend([
                "• Low volatility favors premium buying",
                "• Long options and calendar spreads attractive",
                "• Avoid premium selling strategies",
                "• Look for volatility expansion catalysts"
            ])
        else:
            implications.extend([
                "• Normal volatility environment",
                "• Directional strategies most suitable",
                "• Monitor regime changes carefully",
                "• Balanced approach to premium strategies"
            ])
        
        # Add percentile-specific insights
        if percentile >= 90:
            implications.append("• Volatility in top 10% historically - expect reversion")
        elif percentile <= 10:
            implications.append("• Volatility in bottom 10% historically - expect expansion")
        
        return "\n".join(implications)
    except Exception as e:
        logger.warning(f"Trading implications generation error: {e}")
        return "• Monitor volatility regime changes\n• Adjust strategies accordingly"

@safe_calculation_wrapper
def calculate_comprehensive_volatility_indicators(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate all 14 comprehensive volatility indicators"""
    try:
        if len(data) < 30:
            logger.warning("Insufficient data for comprehensive volatility analysis")
            return {}
        
        # Calculate returns
        returns = data['Close'].pct_change().dropna()
        
        if len(returns) < 20:
            logger.warning("Insufficient return data for volatility calculations")
            return {}
        
        # Calculate all 14 volatility indicators
        indicators = {}
        
        # 1. Historical Volatility (20-day)
        indicators['historical_volatility_20d'] = calculate_historical_volatility(returns, 20)
        
        # 2. Historical Volatility (10-day)
        indicators['historical_volatility_10d'] = calculate_historical_volatility(returns, 10)
        
        # 3. Realized Volatility
        indicators['realized_volatility'] = calculate_realized_volatility(data)
        
        # 4. Volatility Percentile
        indicators['volatility_percentile'] = calculate_volatility_percentile(
            returns, indicators['historical_volatility_20d'])
        
        # 5. Volatility Rank
        indicators['volatility_rank'] = calculate_volatility_rank(
            returns, indicators['historical_volatility_20d'])
        
        # 6. GARCH Volatility
        indicators['garch_volatility'] = calculate_garch_volatility(returns)
        
        # 7. Parkinson Volatility
        indicators['parkinson_volatility'] = calculate_parkinson_volatility(data)
        
        # 8. Garman-Klass Volatility
        indicators['garman_klass_volatility'] = calculate_garman_klass_volatility(data)
        
        # 9. Rogers-Satchell Volatility
        indicators['rogers_satchell_volatility'] = calculate_rogers_satchell_volatility(data)
        
        # 10. Yang-Zhang Volatility
        indicators['yang_zhang_volatility'] = calculate_yang_zhang_volatility(data)
        
        # 11. Volatility of Volatility
        indicators['volatility_of_volatility'] = calculate_volatility_of_volatility(returns)
        
        # 12. Volatility Momentum
        indicators['volatility_momentum'] = calculate_volatility_momentum(returns)
        
        # 13. Volatility Mean Reversion
        indicators['volatility_mean_reversion'] = calculate_volatility_mean_reversion(returns)
        
        # 14. Volatility Clustering
        indicators['volatility_clustering'] = calculate_volatility_clustering(returns)
        
        return indicators
        
    except Exception as e:
        logger.error(f"Comprehensive volatility indicators calculation error: {e}")
        return {}

@safe_calculation_wrapper
def calculate_weighted_volatility_score(indicators: Dict[str, Any]) -> float:
    """Calculate weighted composite volatility score (0-100 scale)"""
    try:
        if not indicators:
            return 50.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for indicator_name, weight in VOLATILITY_INDICATOR_WEIGHTS.items():
            if indicator_name in indicators:
                indicator_value = indicators[indicator_name]
                
                # Normalize each indicator to 0-100 scale
                if indicator_name in ['historical_volatility_20d', 'historical_volatility_10d', 
                                     'realized_volatility', 'garch_volatility', 'parkinson_volatility',
                                     'garman_klass_volatility', 'rogers_satchell_volatility', 
                                     'yang_zhang_volatility']:
                    # For volatility measures, higher values = higher score
                    normalized_score = min(100, max(0, (indicator_value / 50.0) * 100))
                elif indicator_name in ['volatility_percentile', 'volatility_rank']:
                    # Already 0-100 scale
                    normalized_score = indicator_value
                elif indicator_name == 'volatility_of_volatility':
                    # Higher VoV = higher score (more volatile)
                    normalized_score = min(100, max(0, (indicator_value / 10.0) * 100))
                elif indicator_name == 'volatility_momentum':
                    # Convert momentum to 0-100 scale (centered at 50)
                    normalized_score = min(100, max(0, 50 + indicator_value))
                elif indicator_name in ['volatility_mean_reversion', 'volatility_clustering']:
                    # Already 0-1 scale, convert to 0-100
                    normalized_score = indicator_value * 100
                else:
                    normalized_score = 50.0  # Default fallback
                
                total_score += normalized_score * weight
                total_weight += weight
        
        # Calculate weighted average
        if total_weight > 0:
            composite_score = total_score / total_weight
        else:
            composite_score = 50.0
        
        return float(np.clip(composite_score, 0, 100))
        
    except Exception as e:
        logger.error(f"Weighted volatility score calculation error: {e}")
        return 50.0

@safe_calculation_wrapper
def calculate_complete_volatility_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """
    ENHANCED: Calculate complete volatility analysis with 14 comprehensive indicators
    and weighted composite scoring system
    """
    try:
        if len(data) < 30:
            return {
                'error': 'Insufficient data for volatility analysis',
                'volatility_regime': 'Unknown',
                'volatility_score': 50,
                'volatility_strength_factor': 1.0
            }

        # Calculate all comprehensive volatility indicators
        indicators = calculate_comprehensive_volatility_indicators(data)
        
        if not indicators:
            return {
                'error': 'Failed to calculate volatility indicators',
                'volatility_regime': 'Unknown',
                'volatility_score': 50,
                'volatility_strength_factor': 1.0
            }
        
        # Calculate weighted composite score
        composite_score = calculate_weighted_volatility_score(indicators)
        
        # Determine volatility regime and strategy
        vol_percentile = indicators.get('volatility_percentile', 50)
        vol_rank = indicators.get('volatility_rank', 50)
        vol_regime, options_strategy = determine_volatility_regime(vol_percentile, vol_rank)
        
        # Calculate derived metrics
        current_vol_20d = indicators.get('historical_volatility_20d', 20.0)
        current_vol_10d = indicators.get('historical_volatility_10d', 20.0)
        
        # Volatility ratio (10D vs 20D)
        vol_ratio = current_vol_10d / current_vol_20d if current_vol_20d > 0 else 1.0
        
        # Volatility trend (momentum)
        vol_trend = indicators.get('volatility_momentum', 0.0)
        
        # Risk-adjusted return proxy
        returns = data['Close'].pct_change().dropna()
        if len(returns) >= 20:
            avg_return = returns.tail(20).mean() * 252 * 100  # Annualized %
            risk_adjusted_return = avg_return / current_vol_20d if current_vol_20d > 0 else 0
        else:
            risk_adjusted_return = 0.0
        
        # Volatility strength factor for technical scoring (0.85 to 1.3)
        if composite_score >= 80:
            vol_strength_factor = 1.3
        elif composite_score >= 65:
            vol_strength_factor = 1.15  
        elif composite_score >= 35:
            vol_strength_factor = 1.0
        else:
            vol_strength_factor = 0.85
        
        # Prepare component breakdown for UI display
        component_breakdown = []
        for indicator_name, weight in VOLATILITY_INDICATOR_WEIGHTS.items():
            if indicator_name in indicators:
                indicator_value = indicators[indicator_name]
                
                # Format display name
                display_name = indicator_name.replace('_', ' ').title()
                
                # Calculate individual contribution to composite score
                if indicator_name in ['historical_volatility_20d', 'historical_volatility_10d', 
                                     'realized_volatility', 'garch_volatility', 'parkinson_volatility',
                                     'garman_klass_volatility', 'rogers_satchell_volatility', 
                                     'yang_zhang_volatility']:
                    normalized_score = min(100, max(0, (indicator_value / 50.0) * 100))
                    value_display = f"{indicator_value:.2f}%"
                elif indicator_name in ['volatility_percentile', 'volatility_rank']:
                    normalized_score = indicator_value
                    value_display = f"{indicator_value:.1f}%"
                elif indicator_name == 'volatility_of_volatility':
                    normalized_score = min(100, max(0, (indicator_value / 10.0) * 100))
                    value_display = f"{indicator_value:.2f}"
                elif indicator_name == 'volatility_momentum':
                    normalized_score = min(100, max(0, 50 + indicator_value))
                    value_display = f"{indicator_value:+.2f}%"
                elif indicator_name in ['volatility_mean_reversion', 'volatility_clustering']:
                    normalized_score = indicator_value * 100
                    value_display = f"{indicator_value:.3f}"
                else:
                    normalized_score = 50.0
                    value_display = f"{indicator_value:.2f}"
                
                contribution = normalized_score * weight
                
                component_breakdown.append({
                    'name': display_name,
                    'value': value_display,
                    'score': f"{normalized_score:.1f}/100",
                    'weight': f"{weight:.3f}",
                    'contribution': f"{contribution:.2f}"
                })
        
        return {
            # Main volatility metrics
            'volatility_5d': round(current_vol_10d, 2),  # Using 10D as short-term
            'volatility_30d': round(current_vol_20d, 2),  # Using 20D as medium-term
            'volatility_ratio': round(vol_ratio, 2),
            'volatility_trend': round(vol_trend, 2),
            'volatility_percentile': round(vol_percentile, 1),
            'volatility_rank': round(vol_rank, 1),
            
            # Regime and scoring
            'volatility_regime': vol_regime,
            'volatility_score': int(round(composite_score)),
            'options_strategy': options_strategy,
            
            # Advanced metrics
            'risk_adjusted_return': round(risk_adjusted_return, 2),
            'volatility_of_volatility': round(indicators.get('volatility_of_volatility', 5.0), 2),
            'volatility_clustering': round(indicators.get('volatility_clustering', 0.5), 3),
            'volatility_mean_reversion': round(indicators.get('volatility_mean_reversion', 0.5), 3),
            'volatility_strength_factor': vol_strength_factor,
            
            # All individual indicators for reference
            'all_indicators': indicators,
            
            # Component breakdown for UI
            'component_breakdown': component_breakdown,
            
            # Trading guidance
            'trading_implications': get_volatility_trading_implications(vol_regime, vol_percentile)
        }
        
    except Exception as e:
        logger.error(f"Complete volatility analysis calculation error: {e}")
        return {
            'error': f'Volatility analysis failed: {str(e)}',
            'volatility_regime': 'Unknown',
            'volatility_score': 50,
            'volatility_strength_factor': 1.0
        }

@safe_calculation_wrapper        
def calculate_market_wide_volatility_analysis(show_debug=False) -> Dict[str, Any]:
    """Calculate market-wide volatility environment across SPY, QQQ, IWM"""
    try:
        import yfinance as yf
        
        major_indices = ['SPY', 'QQQ', 'IWM']
        market_volatility = {}
        
        for symbol in major_indices:
            try:
                # Suppress yfinance messages during bulk analysis
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period='6mo', interval='1d')
                
                if len(data) >= 30:
                    vol_analysis = calculate_complete_volatility_analysis(data)
                    if 'error' not in vol_analysis:
                        market_volatility[symbol] = {
                            'volatility_20d': vol_analysis.get('volatility_30d', 20.0),
                            'volatility_regime': vol_analysis.get('volatility_regime', 'Normal'),
                            'volatility_score': vol_analysis.get('volatility_score', 50),
                            'volatility_percentile': vol_analysis.get('volatility_percentile', 50)
                        }
                    else:
                        market_volatility[symbol] = {
                            'volatility_20d': 20.0,
                            'volatility_regime': 'Unknown',
                            'volatility_score': 50,
                            'volatility_percentile': 50
                        }
                else:
                    if show_debug:
                        st.warning(f"Insufficient data for {symbol} volatility analysis")
                    
            except Exception as e:
                if show_debug:
                    st.error(f"Failed to fetch volatility data for {symbol}: {e}")
                market_volatility[symbol] = {
                    'volatility_20d': 20.0,
                    'volatility_regime': 'Unknown',
                    'volatility_score': 50,
                    'volatility_percentile': 50
                }
        
        # Calculate market-wide metrics
        if market_volatility:
            avg_volatility = np.mean([data['volatility_20d'] for data in market_volatility.values()])
            avg_score = np.mean([data['volatility_score'] for data in market_volatility.values()])
            avg_percentile = np.mean([data['volatility_percentile'] for data in market_volatility.values()])
            
            # Determine overall market volatility regime
            if avg_percentile >= 80:
                market_regime = "High Volatility Market"
            elif avg_percentile >= 60:
                market_regime = "Elevated Volatility"
            elif avg_percentile >= 40:
                market_regime = "Normal Volatility"
            elif avg_percentile >= 20:
                market_regime = "Low Volatility"
            else:
                market_regime = "Very Low Volatility"
            
            return {
                'individual_analysis': market_volatility,
                'market_average_volatility': round(avg_volatility, 2),
                'market_volatility_score': int(round(avg_score)),
                'market_volatility_percentile': round(avg_percentile, 1),
                'market_regime': market_regime,
                'analysis_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S EST')
            }
        else:
            return {
                'error': 'Failed to analyze market-wide volatility',
                'market_regime': 'Unknown'
            }
            
    except Exception as e:
        logger.error(f"Market-wide volatility analysis error: {e}")
        return {
            'error': f'Market volatility analysis failed: {str(e)}',
            'market_regime': 'Unknown'
        }
