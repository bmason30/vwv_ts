"""
Volatility Analysis Module for VWV Trading System v4.2.1
Advanced 5-day and 30-day rolling volatility analysis with regime detection
Fixed circular dependencies with local imports
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
import functools

logger = logging.getLogger(__name__)

def safe_calculation_wrapper(func):
    """Decorator for safe financial calculations"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if result is None:
                logger.warning(f"Function {func.__name__} returned None")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper

@safe_calculation_wrapper
def analyze_volatility_profile(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive 5-day and 30-day rolling volatility analysis
    
    Args:
        data: DataFrame with OHLCV data
    
    Returns:
        Dictionary with complete volatility analysis results
    """
    try:
        if len(data) < 30:
            return {
                'error': 'Insufficient data for volatility analysis (need ≥30 periods)',
                'data_points': len(data)
            }
        
        close = data['Close'].copy()
        
        # Calculate returns
        returns = close.pct_change().dropna()
        
        if len(returns) < 25:
            return {
                'error': 'Insufficient return data for volatility calculation',
                'data_points': len(returns)
            }
        
        # 5-day rolling volatility (annualized)
        volatility_5d_rolling = returns.rolling(window=5).std() * np.sqrt(252) * 100
        current_5d_vol = volatility_5d_rolling.iloc[-1]
        
        # 30-day rolling volatility (annualized)
        volatility_30d_rolling = returns.rolling(window=30).std() * np.sqrt(252) * 100
        current_30d_vol = volatility_30d_rolling.iloc[-1]
        
        # Volatility ratios and comparisons
        vol_ratio_5d_30d = current_5d_vol / current_30d_vol if current_30d_vol > 0 else 1.0
        
        # Volatility trend analysis (5-day trend)
        if len(volatility_5d_rolling) >= 10:
            prev_5d_vol = volatility_5d_rolling.iloc[-6]  # 5 days ago
            vol_5d_trend = ((current_5d_vol - prev_5d_vol) / prev_5d_vol * 100) if prev_5d_vol > 0 else 0
        else:
            vol_5d_trend = 0
        
        # Volatility cycle analysis (percentile positioning)
        vol_30d_values = volatility_30d_rolling.dropna().tail(60)  # Last 60 periods
        if len(vol_30d_values) >= 30:
            vol_percentile = (vol_30d_values <= current_5d_vol).sum() / len(vol_30d_values) * 100
        else:
            vol_percentile = 50
        
        # Volatility regime classification
        vol_regime, regime_score = classify_volatility_regime(current_5d_vol)
        
        # Volatility acceleration analysis
        vol_acceleration = calculate_volatility_acceleration(volatility_5d_rolling)
        
        # Risk-adjusted returns
        risk_adjusted_return = calculate_risk_adjusted_returns(returns, current_5d_vol)
        
        # Volatility consistency score
        vol_consistency = calculate_volatility_consistency(volatility_5d_rolling.tail(20))
        
        # Volatility strength factor for composite scoring
        vol_strength_factor = calculate_volatility_strength_factor(
            current_5d_vol, vol_ratio_5d_30d, vol_percentile
        )
        
        # Options strategy guidance based on volatility
        options_guidance = determine_volatility_regime_for_options({
            'current_5d_vol': current_5d_vol,
            'vol_regime': vol_regime,
            'vol_percentile': vol_percentile,
            'vol_trend': vol_5d_trend
        })
        
        return {
            'current_5d_vol': round(float(current_5d_vol), 2),
            'current_30d_vol': round(float(current_30d_vol), 2),
            'vol_ratio_5d_30d': round(float(vol_ratio_5d_30d), 2),
            'vol_5d_trend_pct': round(float(vol_5d_trend), 2),
            'vol_percentile': round(float(vol_percentile), 1),
            'vol_regime': vol_regime,
            'regime_score': regime_score,
            'vol_acceleration': round(float(vol_acceleration), 3),
            'risk_adjusted_return': round(float(risk_adjusted_return), 4),
            'vol_consistency': round(float(vol_consistency), 2),
            'vol_strength_factor': round(float(vol_strength_factor), 3),
            'options_guidance': options_guidance,
            'analysis_quality': 'High' if len(data) >= 60 else 'Moderate',
            'data_periods': len(data)
        }
        
    except Exception as e:
        logger.error(f"Volatility analysis calculation error: {e}")
        return {
            'error': f'Volatility analysis failed: {str(e)}',
            'data_points': len(data) if hasattr(data, '__len__') else 0
        }

def classify_volatility_regime(volatility: float) -> tuple:
    """Classify volatility regime and assign score"""
    if volatility >= 50:
        return "Extreme High", 95
    elif volatility >= 35:
        return "High", 80
    elif volatility >= 25:
        return "Above Normal", 65
    elif volatility >= 15:
        return "Normal", 50
    elif volatility >= 10:
        return "Below Normal", 35
    else:
        return "Low", 20

def calculate_volatility_acceleration(vol_5d_series: pd.Series) -> float:
    """Calculate volatility acceleration (rate of change in volatility trend)"""
    try:
        if len(vol_5d_series) < 10:
            return 0.0
        
        # Calculate the acceleration of volatility changes
        vol_changes = vol_5d_series.pct_change().dropna()
        if len(vol_changes) < 5:
            return 0.0
        
        # Recent acceleration vs historical
        recent_acceleration = vol_changes.tail(5).mean()
        historical_acceleration = vol_changes.head(-5).mean() if len(vol_changes) > 5 else 0
        
        acceleration = recent_acceleration - historical_acceleration
        return float(acceleration)
        
    except Exception:
        return 0.0

def calculate_risk_adjusted_returns(returns: pd.Series, current_vol: float) -> float:
    """Calculate risk-adjusted returns (Sharpe-like ratio)"""
    try:
        if len(returns) < 20 or current_vol <= 0:
            return 0.0
        
        # Use recent returns (last 30 days)
        recent_returns = returns.tail(30)
        avg_return = recent_returns.mean() * 252  # Annualized
        
        # Risk-adjusted return (simplified Sharpe ratio)
        risk_adjusted = (avg_return * 100) / current_vol if current_vol > 0 else 0
        
        return float(risk_adjusted)
        
    except Exception:
        return 0.0

def calculate_volatility_consistency(vol_series: pd.Series) -> float:
    """Calculate volatility consistency score (0-100)"""
    try:
        if len(vol_series) < 10:
            return 50.0
        
        # Calculate coefficient of variation (lower = more consistent)
        mean_vol = vol_series.mean()
        std_vol = vol_series.std()
        
        if mean_vol <= 0:
            return 50.0
        
        cv = std_vol / mean_vol
        
        # Convert to consistency score (0-100, higher = more consistent)
        # CV of 0.2 = score of 80, CV of 0.5 = score of 50
        consistency_score = max(0, min(100, 100 - (cv * 100)))
        
        return float(consistency_score)
        
    except Exception:
        return 50.0

def calculate_volatility_strength_factor(current_vol: float, vol_ratio: float, vol_percentile: float) -> float:
    """
    Calculate volatility strength factor for composite technical scoring
    Returns multiplier between 0.85 (low vol) and 1.15 (high vol)
    """
    try:
        # Base score from volatility level (moderate vol is neutral)
        if 15 <= current_vol <= 25:
            vol_score = 50  # Neutral zone
        elif current_vol > 25:
            # Higher volatility gets higher score (more active market)
            vol_score = min(100, 50 + (current_vol - 25) * 1.5)
        else:
            # Lower volatility gets lower score (less active market)
            vol_score = max(0, 50 - (25 - current_vol) * 2)
        
        # Adjust for ratio (recent vs historical)
        ratio_adjustment = min(20, max(-20, (vol_ratio - 1) * 30))
        
        # Adjust for percentile position
        percentile_adjustment = min(15, max(-15, (vol_percentile - 50) * 0.3))
        
        # Combine scores
        total_score = vol_score + ratio_adjustment + percentile_adjustment
        total_score = max(0, min(100, total_score))
        
        # Convert to multiplier (0.85 to 1.15)
        multiplier = 0.85 + (total_score / 100) * 0.30
        
        return float(multiplier)
        
    except Exception:
        return 1.0  # Neutral multiplier

def determine_volatility_regime_for_options(vol_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate options strategy guidance based on volatility regime"""
    try:
        current_vol = vol_data.get('current_5d_vol', 20)
        vol_regime = vol_data.get('vol_regime', 'Normal')
        vol_percentile = vol_data.get('vol_percentile', 50)
        vol_trend = vol_data.get('vol_trend', 0)
        
        # Base strategy from volatility level
        if current_vol >= 40:
            base_strategy = "Aggressive Selling"
            base_description = "Very high volatility - sell premium aggressively"
        elif current_vol >= 30:
            base_strategy = "Premium Selling"
            base_description = "High volatility - favor selling strategies"
        elif current_vol >= 20:
            base_strategy = "Neutral"
            base_description = "Moderate volatility - balanced approach"
        elif current_vol >= 12:
            base_strategy = "Slight Buying Bias"
            base_description = "Below average volatility - consider buying strategies"
        else:
            base_strategy = "Premium Buying"
            base_description = "Low volatility - buy premium before expansion"
        
        # Adjust based on percentile and trend
        if vol_percentile > 80 and vol_trend > 10:
            adjustment = "🔥 Extreme: Sell premium immediately"
        elif vol_percentile > 70:
            adjustment = "📈 High: Strong selling opportunity"
        elif vol_percentile < 20 and vol_trend < -10:
            adjustment = "❄️ Extreme: Buy premium before expansion"
        elif vol_percentile < 30:
            adjustment = "📉 Low: Good buying opportunity"
        else:
            adjustment = "⚖️ Moderate: Standard approach"
        
        # Risk level recommendation
        if "Selling" in base_strategy:
            risk_level = "Conservative" if current_vol > 35 else "Moderate"
        else:
            risk_level = "Moderate" if current_vol < 15 else "Conservative"
        
        return {
            'strategy': base_strategy,
            'description': base_description,
            'adjustment': adjustment,
            'risk_level': risk_level,
            'vol_percentile': vol_percentile,
            'trend_factor': "Rising" if vol_trend > 5 else "Falling" if vol_trend < -5 else "Stable"
        }
        
    except Exception as e:
        logger.error(f"Options guidance error: {e}")
        return {
            'strategy': 'Neutral',
            'description': 'Analysis error - use standard approach',
            'adjustment': 'Exercise caution',
            'risk_level': 'Conservative'
        }

@safe_calculation_wrapper
def interpret_volatility_data(vol_data: Dict[str, Any]) -> Dict[str, str]:
    """Generate volatility analysis interpretation"""
    try:
        if 'error' in vol_data:
            return {
                'regime_interpretation': 'Analysis not available',
                'trend_interpretation': 'No trend data',
                'trading_implications': 'Use alternative analysis methods',
                'options_implications': 'No options guidance available'
            }
        
        # Regime interpretation
        regime = vol_data.get('vol_regime', 'Unknown')
        current_vol = vol_data.get('current_5d_vol', 0)
        percentile = vol_data.get('vol_percentile', 50)
        
        if regime == "Extreme High":
            regime_interp = f"🔥 Extreme volatility ({current_vol:.1f}%) - Major market stress or event"
        elif regime == "High":
            regime_interp = f"⚡ High volatility ({current_vol:.1f}%) - Increased market uncertainty"
        elif regime == "Above Normal":
            regime_interp = f"📈 Above normal volatility ({current_vol:.1f}%) - Elevated market activity"
        elif regime == "Normal":
            regime_interp = f"⚖️ Normal volatility ({current_vol:.1f}%) - Typical market conditions"
        elif regime == "Below Normal":
            regime_interp = f"📉 Below normal volatility ({current_vol:.1f}%) - Calm market conditions"
        else:
            regime_interp = f"😴 Low volatility ({current_vol:.1f}%) - Very calm market"
        
        # Add percentile context
        if percentile > 80:
            regime_interp += f" (Top {100-percentile:.0f}% of recent range)"
        elif percentile < 20:
            regime_interp += f" (Bottom {percentile:.0f}% of recent range)"
        
        # Trend interpretation
        trend = vol_data.get('vol_5d_trend_pct', 0)
        if trend > 20:
            trend_interp = f"📈 Strong volatility expansion (+{trend:.1f}%) - Increasing uncertainty"
        elif trend > 10:
            trend_interp = f"↗️ Moderate volatility rise (+{trend:.1f}%) - Building tension"
        elif trend > -10:
            trend_interp = f"→ Stable volatility ({trend:+.1f}%) - Consistent conditions"
        elif trend > -20:
            trend_interp = f"↘️ Moderate volatility decline ({trend:.1f}%) - Calming market"
        else:
            trend_interp = f"📉 Strong volatility contraction ({trend:.1f}%) - Market settling"
        
        # Trading implications
        vol_ratio = vol_data.get('vol_ratio_5d_30d', 1.0)
        if current_vol > 30 and vol_ratio > 1.2:
            trading_impl = "🚨 High risk environment - Reduce position sizes, tight stops"
        elif current_vol > 25:
            trading_impl = "⚠️ Elevated risk - Standard position sizing with enhanced monitoring"
        elif current_vol < 12 and vol_ratio < 0.8:
            trading_impl = "😴 Low risk environment - Consider larger positions, expansion pending"
        else:
            trading_impl = "⚖️ Normal risk environment - Standard risk management"
        
        # Options implications
        options_guidance = vol_data.get('options_guidance', {})
        strategy = options_guidance.get('strategy', 'Neutral')
        adjustment = options_guidance.get('adjustment', 'Standard approach')
        
        options_impl = f"🎯 {strategy}: {adjustment}"
        
        return {
            'regime_interpretation': regime_interp,
            'trend_interpretation': trend_interp,
            'trading_implications': trading_impl,
            'options_implications': options_impl
        }
        
    except Exception as e:
        logger.error(f"Volatility interpretation error: {e}")
        return {
            'regime_interpretation': 'Analysis error',
            'trend_interpretation': 'Analysis error',
            'trading_implications': 'Use caution - analysis incomplete',
            'options_implications': 'Use standard options approach'
        }

@safe_calculation_wrapper
def compare_market_volatility(symbols: List[str] = None, period: str = '3mo', show_debug: bool = False) -> Dict[str, Any]:
    """
    Calculate market-wide volatility comparison across major symbols
    Uses local imports to avoid circular dependencies
    """
    try:
        # Local imports to avoid circular dependencies
        import streamlit as st
        import yfinance as yf
        
        if symbols is None:
            symbols = ['SPY', 'QQQ', 'IWM']
        
        if show_debug:
            st.write(f"🌡️ Analyzing market volatility for: {', '.join(symbols)}")
        
        # Get market data with local processing
        market_data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if len(data) > 30:
                    market_data[symbol] = data
                    
            except Exception as e:
                logger.error(f"Error fetching volatility data for {symbol}: {e}")
                continue
        
        if not market_data:
            return {'error': 'No market data available'}
        
        market_vol_results = {}
        
        for symbol, data in market_data.items():
            try:
                vol_analysis = analyze_volatility_profile(data)
                if 'error' not in vol_analysis:
                    market_vol_results[symbol] = {
                        'vol_regime': vol_analysis.get('vol_regime'),
                        'current_5d_vol': vol_analysis.get('current_5d_vol'),
                        'regime_score': vol_analysis.get('regime_score'),
                        'vol_5d_trend_pct': vol_analysis.get('vol_5d_trend_pct'),
                        'vol_percentile': vol_analysis.get('vol_percentile'),
                        'vol_strength_factor': vol_analysis.get('vol_strength_factor'),
                        'options_guidance': vol_analysis.get('options_guidance')
                    }
                
            except Exception as e:
                if show_debug:
                    st.write(f"  • {symbol}: Error - {str(e)}")
                continue
        
        if not market_vol_results:
            return {'error': 'No valid volatility analysis results'}
        
        # Calculate market volatility environment
        environment_analysis = analyze_market_volatility_environment(market_vol_results)
        
        return {
            'individual_results': market_vol_results,
            'market_environment': environment_analysis,
            'analysis_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbols_analyzed': list(market_vol_results.keys())
        }
        
    except Exception as e:
        logger.error(f"Market volatility comparison error: {e}")
        return {'error': f'Market volatility analysis failed: {str(e)}'}

def analyze_market_volatility_environment(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze overall market volatility environment"""
    try:
        if not results:
            return {'environment': 'Unknown', 'description': 'No data available'}
        
        # Calculate average metrics
        avg_vol = np.mean([data['current_5d_vol'] for data in results.values()])
        avg_score = np.mean([data['regime_score'] for data in results.values()])
        avg_trend = np.mean([data['vol_5d_trend_pct'] for data in results.values()])
        avg_percentile = np.mean([data['vol_percentile'] for data in results.values()])
        
        # Count regime classifications
        regime_counts = {}
        for data in results.values():
            regime = data['vol_regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Determine dominant regime
        dominant_regime = max(regime_counts, key=regime_counts.get)
        
        # Market environment classification
        if avg_vol >= 35:
            environment = "High Volatility"
            description = f"⚡ High volatility environment - {dominant_regime} conditions dominate (avg: {avg_vol:.1f}%)"
        elif avg_vol >= 25:
            environment = "Above Normal Volatility"
            description = f"📈 Above normal volatility - {dominant_regime} conditions (avg: {avg_vol:.1f}%)"
        elif avg_vol >= 15:
            environment = "Normal Volatility"
            description = f"⚖️ Normal volatility environment - {dominant_regime} conditions (avg: {avg_vol:.1f}%)"
        else:
            environment = "Low Volatility"
            description = f"😴 Low volatility environment - {dominant_regime} conditions (avg: {avg_vol:.1f}%)"
        
        # Overall options guidance
        strategy_counts = {}
        for data in results.values():
            strategy = data.get('options_guidance', {}).get('strategy', 'Neutral')
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        dominant_strategy = max(strategy_counts, key=strategy_counts.get) if strategy_counts else 'Neutral'
        
        return {
            'environment': environment,
            'description': description,
            'avg_volatility': round(avg_vol, 2),
            'avg_regime_score': round(avg_score, 1),
            'avg_trend_pct': round(avg_trend, 2),
            'avg_percentile': round(avg_percentile, 1),
            'dominant_regime': dominant_regime,
            'dominant_strategy': dominant_strategy,
            'regime_distribution': regime_counts,
            'strategy_distribution': strategy_counts
        }
        
    except Exception as e:
        logger.error(f"Market volatility environment analysis error: {e}")
        return {
            'environment': 'Analysis Error',
            'description': 'Unable to determine market volatility environment'
        }

# Aliases for the expected function names to maintain compatibility
calculate_volatility_analysis = analyze_volatility_profile
get_volatility_interpretation = interpret_volatility_data
calculate_market_volatility_comparison = compare_market_volatility
get_volatility_regime_for_options = determine_volatility_regime_for_options
