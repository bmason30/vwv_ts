"""
Volume Analysis Module for VWV Trading System v4.2.1
Advanced 5-day and 30-day rolling volume analysis with regime detection
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
def analyze_volume_profile(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive 5-day and 30-day rolling volume analysis
    
    Args:
        data: DataFrame with OHLCV data
    
    Returns:
        Dictionary with complete volume analysis results
    """
    try:
        if len(data) < 30:
            return {
                'error': 'Insufficient data for volume analysis (need ≥30 periods)',
                'data_points': len(data)
            }
        
        volume = data['Volume'].copy()
        
        # Remove zero volume days for calculation
        volume_clean = volume[volume > 0]
        if len(volume_clean) < 20:
            return {
                'error': 'Insufficient non-zero volume data',
                'data_points': len(volume_clean)
            }
        
        # Current volume (most recent day)
        current_volume = volume.iloc[-1]
        
        # 5-day rolling volume analysis
        volume_5d_rolling = volume.rolling(window=5).mean()
        current_5d_avg = volume_5d_rolling.iloc[-1]
        
        # 30-day rolling volume analysis  
        volume_30d_rolling = volume.rolling(window=30).mean()
        current_30d_avg = volume_30d_rolling.iloc[-1]
        
        # Volume ratios and comparisons
        volume_ratio_5d_30d = current_5d_avg / current_30d_avg if current_30d_avg > 0 else 1.0
        volume_ratio_current_5d = current_volume / current_5d_avg if current_5d_avg > 0 else 1.0
        volume_ratio_current_30d = current_volume / current_30d_avg if current_30d_avg > 0 else 1.0
        
        # Volume trend analysis (5-day trend)
        if len(volume_5d_rolling) >= 10:
            prev_5d_avg = volume_5d_rolling.iloc[-6]  # 5 days ago
            volume_5d_trend = ((current_5d_avg - prev_5d_avg) / prev_5d_avg * 100) if prev_5d_avg > 0 else 0
        else:
            volume_5d_trend = 0
        
        # Volume standard deviation and Z-score analysis
        volume_std_30d = volume.rolling(window=30).std().iloc[-1]
        volume_z_score = (current_5d_avg - current_30d_avg) / volume_std_30d if volume_std_30d > 0 else 0
        
        # Volume regime classification
        volume_regime, regime_score = classify_volume_regime(volume_ratio_5d_30d)
        
        # Volume breakout detection
        volume_breakout = detect_volume_breakout(volume_z_score, volume_ratio_5d_30d)
        
        # Volume acceleration analysis
        volume_acceleration = calculate_volume_acceleration(volume_5d_rolling)
        
        # Volume consistency score
        volume_consistency = calculate_volume_consistency(volume.tail(20))
        
        # Volume momentum indicator
        volume_momentum = calculate_volume_momentum(volume, volume_5d_rolling)
        
        # Volume strength factor for composite scoring
        volume_strength_factor = calculate_volume_strength_factor(
            volume_ratio_5d_30d, volume_z_score, volume_acceleration
        )
        
        return {
            'current_volume': round(float(current_volume), 0),
            'volume_5d_avg': round(float(current_5d_avg), 0),
            'volume_30d_avg': round(float(current_30d_avg), 0),
            'volume_ratio_5d_30d': round(float(volume_ratio_5d_30d), 2),
            'volume_ratio_current_5d': round(float(volume_ratio_current_5d), 2),
            'volume_ratio_current_30d': round(float(volume_ratio_current_30d), 2),
            'volume_5d_trend_pct': round(float(volume_5d_trend), 2),
            'volume_std_30d': round(float(volume_std_30d), 0),
            'volume_z_score': round(float(volume_z_score), 2),
            'volume_regime': volume_regime,
            'regime_score': regime_score,
            'volume_breakout': volume_breakout,
            'volume_acceleration': round(float(volume_acceleration), 3),
            'volume_consistency': round(float(volume_consistency), 2),
            'volume_momentum': round(float(volume_momentum), 2),
            'volume_strength_factor': round(float(volume_strength_factor), 3),
            'analysis_quality': 'High' if len(data) >= 60 else 'Moderate',
            'data_periods': len(data)
        }
        
    except Exception as e:
        logger.error(f"Volume analysis calculation error: {e}")
        return {
            'error': f'Volume analysis failed: {str(e)}',
            'data_points': len(data) if hasattr(data, '__len__') else 0
        }

def classify_volume_regime(volume_ratio: float) -> tuple:
    """Classify volume regime and assign score"""
    if volume_ratio >= 2.0:
        return "Extreme High", 95
    elif volume_ratio >= 1.5:
        return "High", 80
    elif volume_ratio >= 1.2:
        return "Above Normal", 65
    elif volume_ratio >= 0.8:
        return "Normal", 50
    elif volume_ratio >= 0.5:
        return "Below Normal", 35
    else:
        return "Low", 20

def detect_volume_breakout(z_score: float, volume_ratio: float) -> Dict[str, Any]:
    """Detect volume breakouts using Z-score and ratio analysis"""
    breakout_type = "None"
    breakout_strength = 0
    
    if z_score >= 2.0 and volume_ratio >= 1.5:
        breakout_type = "Strong Bullish"
        breakout_strength = 90
    elif z_score >= 1.5 and volume_ratio >= 1.3:
        breakout_type = "Moderate Bullish"
        breakout_strength = 70
    elif z_score <= -2.0 and volume_ratio <= 0.5:
        breakout_type = "Strong Bearish"
        breakout_strength = 10
    elif z_score <= -1.5 and volume_ratio <= 0.7:
        breakout_type = "Moderate Bearish"
        breakout_strength = 30
    else:
        breakout_strength = 50
    
    return {
        'type': breakout_type,
        'strength': breakout_strength,
        'z_score': round(z_score, 2)
    }

def calculate_volume_acceleration(volume_5d_series: pd.Series) -> float:
    """Calculate volume acceleration (rate of change in volume trend)"""
    try:
        if len(volume_5d_series) < 10:
            return 0.0
        
        # Calculate the acceleration of volume changes
        volume_changes = volume_5d_series.pct_change().dropna()
        if len(volume_changes) < 5:
            return 0.0
        
        # Recent acceleration vs historical
        recent_acceleration = volume_changes.tail(5).mean()
        historical_acceleration = volume_changes.head(-5).mean() if len(volume_changes) > 5 else 0
        
        acceleration = recent_acceleration - historical_acceleration
        return float(acceleration)
        
    except Exception:
        return 0.0

def calculate_volume_consistency(volume_series: pd.Series) -> float:
    """Calculate volume consistency score (0-100)"""
    try:
        if len(volume_series) < 10:
            return 50.0
        
        # Calculate coefficient of variation (lower = more consistent)
        mean_volume = volume_series.mean()
        std_volume = volume_series.std()
        
        if mean_volume <= 0:
            return 50.0
        
        cv = std_volume / mean_volume
        
        # Convert to consistency score (0-100, higher = more consistent)
        # CV of 0.3 = score of 70, CV of 1.0 = score of 30
        consistency_score = max(0, min(100, 100 - (cv * 70)))
        
        return float(consistency_score)
        
    except Exception:
        return 50.0

def calculate_volume_momentum(volume: pd.Series, volume_5d: pd.Series) -> float:
    """Calculate volume momentum indicator"""
    try:
        if len(volume) < 20 or len(volume_5d) < 10:
            return 0.0
        
        # Current vs recent trend
        current_volume = volume.iloc[-1]
        avg_5d_volume = volume_5d.iloc[-5:].mean()
        
        # Volume momentum as percentage difference
        if avg_5d_volume > 0:
            momentum = ((current_volume - avg_5d_volume) / avg_5d_volume) * 100
        else:
            momentum = 0.0
        
        # Cap extreme values
        momentum = max(-200, min(200, momentum))
        
        return float(momentum)
        
    except Exception:
        return 0.0

def calculate_volume_strength_factor(volume_ratio: float, z_score: float, acceleration: float) -> float:
    """
    Calculate volume strength factor for composite technical scoring
    Returns multiplier between 0.85 (weak) and 1.3 (strong)
    """
    try:
        # Base score from volume ratio
        ratio_score = min(100, max(0, (volume_ratio - 0.5) * 40 + 50))
        
        # Adjust for Z-score (breakout strength)
        z_adjustment = min(20, max(-20, z_score * 5))
        
        # Adjust for acceleration
        accel_adjustment = min(10, max(-10, acceleration * 100))
        
        # Combine scores
        total_score = ratio_score + z_adjustment + accel_adjustment
        total_score = max(0, min(100, total_score))
        
        # Convert to multiplier (0.85 to 1.3)
        multiplier = 0.85 + (total_score / 100) * 0.45
        
        return float(multiplier)
        
    except Exception:
        return 1.0  # Neutral multiplier

@safe_calculation_wrapper
def interpret_volume_data(volume_data: Dict[str, Any]) -> Dict[str, str]:
    """Generate volume analysis interpretation"""
    try:
        if 'error' in volume_data:
            return {
                'regime_interpretation': 'Analysis not available',
                'trend_interpretation': 'No trend data',
                'trading_implications': 'Use alternative analysis methods',
                'breakout_interpretation': 'No breakout analysis'
            }
        
        # Regime interpretation
        regime = volume_data.get('volume_regime', 'Unknown')
        ratio = volume_data.get('volume_ratio_5d_30d', 1.0)
        
        if regime == "Extreme High":
            regime_interp = f"🔥 Exceptional volume activity ({ratio:.1f}x normal) - Major market event or strong interest"
        elif regime == "High":
            regime_interp = f"📈 High volume activity ({ratio:.1f}x normal) - Increased market participation"
        elif regime == "Above Normal":
            regime_interp = f"↗️ Above normal volume ({ratio:.1f}x normal) - Moderate increased interest"
        elif regime == "Normal":
            regime_interp = f"⚖️ Normal volume activity ({ratio:.1f}x normal) - Typical market participation"
        elif regime == "Below Normal":
            regime_interp = f"↘️ Below normal volume ({ratio:.1f}x normal) - Reduced market interest"
        else:
            regime_interp = f"📉 Low volume activity ({ratio:.1f}x normal) - Minimal market participation"
        
        # Trend interpretation
        trend = volume_data.get('volume_5d_trend_pct', 0)
        if trend > 15:
            trend_interp = f"📈 Strong volume increase trend (+{trend:.1f}%) - Building momentum"
        elif trend > 5:
            trend_interp = f"↗️ Moderate volume increase (+{trend:.1f}%) - Positive momentum"
        elif trend > -5:
            trend_interp = f"→ Stable volume trend ({trend:+.1f}%) - Consistent participation"
        elif trend > -15:
            trend_interp = f"↘️ Moderate volume decrease ({trend:.1f}%) - Weakening interest"
        else:
            trend_interp = f"📉 Strong volume decrease ({trend:.1f}%) - Declining participation"
        
        # Trading implications
        breakout = volume_data.get('volume_breakout', {})
        breakout_type = breakout.get('type', 'None')
        
        if "Strong Bullish" in breakout_type:
            trading_impl = "🚀 Strong conviction signals - Consider larger position sizes"
        elif "Moderate Bullish" in breakout_type:
            trading_impl = "📈 Positive volume confirmation - Standard position sizing"
        elif "Strong Bearish" in breakout_type:
            trading_impl = "🔻 Weak volume environment - Reduce position sizes or wait"
        elif "Moderate Bearish" in breakout_type:
            trading_impl = "📉 Below average participation - Exercise caution"
        else:
            trading_impl = "⚖️ Normal volume environment - Standard risk management"
        
        # Breakout interpretation
        z_score = volume_data.get('volume_z_score', 0)
        if abs(z_score) >= 2.0:
            breakout_interp = f"🎯 Significant volume breakout (Z-score: {z_score:.1f}) - High probability move"
        elif abs(z_score) >= 1.5:
            breakout_interp = f"📊 Moderate volume signal (Z-score: {z_score:.1f}) - Worth monitoring"
        else:
            breakout_interp = f"😐 Normal volume range (Z-score: {z_score:.1f}) - No strong signals"
        
        return {
            'regime_interpretation': regime_interp,
            'trend_interpretation': trend_interp,
            'trading_implications': trading_impl,
            'breakout_interpretation': breakout_interp
        }
        
    except Exception as e:
        logger.error(f"Volume interpretation error: {e}")
        return {
            'regime_interpretation': 'Analysis error',
            'trend_interpretation': 'Analysis error',
            'trading_implications': 'Use caution - analysis incomplete',
            'breakout_interpretation': 'Analysis error'
        }

@safe_calculation_wrapper
def compare_market_volume(symbols: List[str] = None, period: str = '3mo', show_debug: bool = False) -> Dict[str, Any]:
    """
    Calculate market-wide volume comparison across major symbols
    Uses local imports to avoid circular dependencies
    """
    try:
        # Local imports to avoid circular dependencies
        import streamlit as st
        import yfinance as yf
        
        if symbols is None:
            symbols = ['SPY', 'QQQ', 'IWM']
        
        if show_debug:
            st.write(f"📊 Analyzing market volume for: {', '.join(symbols)}")
        
        # Get market data with local caching
        market_data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if len(data) > 30:
                    market_data[symbol] = data
                    
            except Exception as e:
                logger.error(f"Error fetching volume data for {symbol}: {e}")
                continue
        
        if not market_data:
            return {'error': 'No market data available'}
        
        market_volume_results = {}
        
        for symbol, data in market_data.items():
            try:
                volume_analysis = analyze_volume_profile(data)
                if 'error' not in volume_analysis:
                    market_volume_results[symbol] = {
                        'volume_regime': volume_analysis.get('volume_regime'),
                        'volume_ratio_5d_30d': volume_analysis.get('volume_ratio_5d_30d'),
                        'regime_score': volume_analysis.get('regime_score'),
                        'volume_5d_trend_pct': volume_analysis.get('volume_5d_trend_pct'),
                        'volume_strength_factor': volume_analysis.get('volume_strength_factor')
                    }
                
            except Exception as e:
                if show_debug:
                    st.write(f"  • {symbol}: Error - {str(e)}")
                continue
        
        if not market_volume_results:
            return {'error': 'No valid volume analysis results'}
        
        # Calculate market volume environment
        environment_analysis = analyze_market_volume_environment(market_volume_results)
        
        return {
            'individual_results': market_volume_results,
            'market_environment': environment_analysis,
            'analysis_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbols_analyzed': list(market_volume_results.keys())
        }
        
    except Exception as e:
        logger.error(f"Market volume comparison error: {e}")
        return {'error': f'Market volume analysis failed: {str(e)}'}

def analyze_market_volume_environment(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze overall market volume environment"""
    try:
        if not results:
            return {'environment': 'Unknown', 'description': 'No data available'}
        
        # Calculate average metrics
        avg_ratio = np.mean([data['volume_ratio_5d_30d'] for data in results.values()])
        avg_score = np.mean([data['regime_score'] for data in results.values()])
        avg_trend = np.mean([data['volume_5d_trend_pct'] for data in results.values()])
        
        # Count regime classifications
        regime_counts = {}
        for data in results.values():
            regime = data['volume_regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Determine dominant regime
        dominant_regime = max(regime_counts, key=regime_counts.get)
        
        # Market environment classification
        if avg_score >= 80:
            environment = "High Volume"
            description = f"🔥 High volume environment - {dominant_regime} activity dominates ({avg_ratio:.1f}x average)"
        elif avg_score >= 65:
            environment = "Above Normal Volume"  
            description = f"📈 Above normal volume environment - {dominant_regime} activity ({avg_ratio:.1f}x average)"
        elif avg_score >= 35:
            environment = "Normal Volume"
            description = f"⚖️ Normal volume environment - {dominant_regime} activity ({avg_ratio:.1f}x average)"
        else:
            environment = "Low Volume"
            description = f"📉 Low volume environment - {dominant_regime} activity ({avg_ratio:.1f}x average)"
        
        return {
            'environment': environment,
            'description': description,
            'avg_volume_ratio': round(avg_ratio, 2),
            'avg_regime_score': round(avg_score, 1),
            'avg_trend_pct': round(avg_trend, 2),
            'dominant_regime': dominant_regime,
            'regime_distribution': regime_counts
        }
        
    except Exception as e:
        logger.error(f"Market volume environment analysis error: {e}")
        return {
            'environment': 'Analysis Error',
            'description': 'Unable to determine market volume environment'
        }

# Aliases for the expected function names to maintain compatibility
calculate_volume_analysis = analyze_volume_profile
get_volume_interpretation = interpret_volume_data
calculate_market_volume_comparison = compare_market_volume
