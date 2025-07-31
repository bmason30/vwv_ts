"""
Volume analysis for VWV Trading System
Comprehensive volume trend analysis with 5-day and 30-day rolling metrics
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def calculate_volume_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive volume analysis with 5-day and 30-day rolling metrics
    
    Returns:
        Dict containing volume metrics, trends, and classification
    """
    try:
        if len(data) < 30:
            return {
                'error': 'Insufficient data for volume analysis (minimum 30 periods required)',
                'data_points': len(data)
            }

        volume = data['Volume'].copy()
        current_volume = volume.iloc[-1]
        
        # 5-Day Rolling Volume Analysis
        volume_5d_rolling = volume.rolling(window=5).mean()
        current_5d_avg = volume_5d_rolling.iloc[-1]
        prev_5d_avg = volume_5d_rolling.iloc[-2] if len(volume_5d_rolling) > 1 else current_5d_avg
        
        # 5-Day Volume Trend Analysis
        volume_5d_trend_direction = "Increasing" if current_5d_avg > prev_5d_avg else "Decreasing"
        volume_5d_trend_strength = abs((current_5d_avg - prev_5d_avg) / prev_5d_avg * 100) if prev_5d_avg > 0 else 0
        
        # Volume trend momentum (rate of change over 5 days)
        volume_5d_values = volume_5d_rolling.tail(5).dropna()
        if len(volume_5d_values) >= 2:
            volume_momentum = (volume_5d_values.iloc[-1] - volume_5d_values.iloc[0]) / volume_5d_values.iloc[0] * 100
        else:
            volume_momentum = 0
        
        # 30-Day Rolling Volume Analysis
        volume_30d_rolling = volume.rolling(window=30).mean()
        current_30d_avg = volume_30d_rolling.iloc[-1]
        
        # 5-Day vs 30-Day Comparison
        volume_ratio_5d_30d = current_5d_avg / current_30d_avg if current_30d_avg > 0 else 1
        volume_deviation_pct = ((current_5d_avg - current_30d_avg) / current_30d_avg * 100) if current_30d_avg > 0 else 0
        
        # Volume Regime Classification
        if volume_ratio_5d_30d >= 2.0:
            volume_regime = "Extreme High"
            regime_score = 95
        elif volume_ratio_5d_30d >= 1.5:
            volume_regime = "High"
            regime_score = 80
        elif volume_ratio_5d_30d >= 1.2:
            volume_regime = "Above Normal"
            regime_score = 65
        elif volume_ratio_5d_30d >= 0.8:
            volume_regime = "Normal"
            regime_score = 50
        elif volume_ratio_5d_30d >= 0.5:
            volume_regime = "Below Normal"
            regime_score = 35
        else:
            volume_regime = "Low"
            regime_score = 20
        
        # Volume Breakout Detection
        volume_std_30d = volume.rolling(window=30).std().iloc[-1]
        volume_z_score = (current_5d_avg - current_30d_avg) / volume_std_30d if volume_std_30d > 0 else 0
        
        if abs(volume_z_score) >= 2.0:
            volume_breakout = "Significant Breakout" if volume_z_score > 0 else "Significant Breakdown"
        elif abs(volume_z_score) >= 1.5:
            volume_breakout = "Moderate Breakout" if volume_z_score > 0 else "Moderate Breakdown"
        else:
            volume_breakout = "Normal Range"
        
        # Volume Strength Factor for composite scoring
        if volume_ratio_5d_30d >= 2.0:
            volume_strength_factor = 1.3  # Very high volume boosts other indicators
        elif volume_ratio_5d_30d >= 1.5:
            volume_strength_factor = 1.15  # High volume
        elif volume_ratio_5d_30d >= 1.2:
            volume_strength_factor = 1.05  # Above normal
        elif volume_ratio_5d_30d >= 0.8:
            volume_strength_factor = 1.0   # Normal
        elif volume_ratio_5d_30d >= 0.5:
            volume_strength_factor = 0.95  # Below normal
        else:
            volume_strength_factor = 0.85  # Low volume reduces reliability
        
        # Advanced Volume Metrics
        # Volume acceleration (change in trend strength)
        volume_5d_values_extended = volume_5d_rolling.tail(10).dropna()
        if len(volume_5d_values_extended) >= 5:
            recent_trend = volume_5d_values_extended.tail(3).mean()
            earlier_trend = volume_5d_values_extended.head(3).mean()
            volume_acceleration = (recent_trend - earlier_trend) / earlier_trend * 100 if earlier_trend > 0 else 0
        else:
            volume_acceleration = 0
        
        # Volume consistency (how consistent the trend is)
        volume_5d_last_week = volume_5d_rolling.tail(7).dropna()
        if len(volume_5d_last_week) >= 2:
            volume_consistency = 100 - (volume_5d_last_week.std() / volume_5d_last_week.mean() * 100)
            volume_consistency = max(0, min(100, volume_consistency))
        else:
            volume_consistency = 50
        
        return {
            # 5-Day Rolling Metrics
            'volume_5d_avg': round(float(current_5d_avg), 0),
            'volume_5d_trend_direction': volume_5d_trend_direction,
            'volume_5d_trend_strength': round(float(volume_5d_trend_strength), 2),
            'volume_momentum': round(float(volume_momentum), 2),
            
            # 30-Day Comparison Metrics
            'volume_30d_avg': round(float(current_30d_avg), 0),
            'volume_ratio_5d_30d': round(float(volume_ratio_5d_30d), 2),
            'volume_deviation_pct': round(float(volume_deviation_pct), 2),
            
            # Volume Regime Analysis
            'volume_regime': volume_regime,
            'regime_score': regime_score,
            'volume_breakout': volume_breakout,
            'volume_z_score': round(float(volume_z_score), 2),
            
            # Composite Integration
            'volume_strength_factor': round(float(volume_strength_factor), 3),
            'volume_composite_score': regime_score,  # For composite technical scoring
            
            # Advanced Metrics
            'current_volume': round(float(current_volume), 0),
            'volume_acceleration': round(float(volume_acceleration), 2),
            'volume_consistency': round(float(volume_consistency), 1),
            
            # Metadata
            'data_points': len(data),
            'calculation_success': True
        }
        
    except Exception as e:
        logger.error(f"Volume analysis calculation error: {e}")
        return {
            'error': f'Volume analysis failed: {str(e)}',
            'data_points': len(data) if data is not None else 0,
            'calculation_success': False
        }

@safe_calculation_wrapper
def get_volume_interpretation(volume_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Provide human-readable interpretation of volume analysis results
    """
    try:
        if 'error' in volume_data or not volume_data.get('calculation_success', False):
            return {
                'overall': 'Volume analysis unavailable',
                'trend': 'Cannot determine trend',
                'strength': 'Cannot assess strength',
                'recommendation': 'Insufficient volume data'
            }
        
        regime = volume_data.get('volume_regime', 'Unknown')
        trend_direction = volume_data.get('volume_5d_trend_direction', 'Unknown')
        momentum = volume_data.get('volume_momentum', 0)
        
        # Overall interpretation
        if regime in ['Extreme High', 'High']:
            overall = f"Very active trading with {regime.lower()} volume levels"
        elif regime == 'Above Normal':
            overall = "Increased trading activity above normal levels"
        elif regime == 'Normal':
            overall = "Normal trading volume patterns"
        else:
            overall = f"Reduced trading activity with {regime.lower()} volume"
        
        # Trend interpretation
        if trend_direction == "Increasing" and momentum > 10:
            trend = "Strong increasing volume trend - growing interest"
        elif trend_direction == "Increasing":
            trend = "Moderate increasing volume trend"
        elif trend_direction == "Decreasing" and momentum < -10:
            trend = "Strong decreasing volume trend - waning interest"
        else:
            trend = "Moderate decreasing volume trend"
        
        # Strength assessment
        strength_factor = volume_data.get('volume_strength_factor', 1.0)
        if strength_factor >= 1.2:
            strength = "High volume strength - enhances other technical signals"
        elif strength_factor >= 1.05:
            strength = "Above average volume strength"
        elif strength_factor >= 0.95:
            strength = "Normal volume strength"
        else:
            strength = "Below normal volume strength - reduces signal reliability"
        
        # Trading recommendation
        breakout = volume_data.get('volume_breakout', 'Normal Range')
        if 'Significant' in breakout:
            recommendation = f"{breakout} detected - monitor for continuation or reversal"
        elif 'Moderate' in breakout:
            recommendation = f"{breakout} detected - watch for follow-through"
        else:
            recommendation = "Normal volume patterns - rely on other technical indicators"
        
        return {
            'overall': overall,
            'trend': trend,
            'strength': strength,
            'recommendation': recommendation
        }
        
    except Exception as e:
        logger.error(f"Volume interpretation error: {e}")
        return {
            'overall': 'Interpretation error',
            'trend': 'Cannot interpret trend',
            'strength': 'Cannot assess strength', 
            'recommendation': 'Review volume data manually'
        }

@safe_calculation_wrapper
def calculate_market_volume_comparison(symbols: list = ['SPY', 'QQQ', 'IWM']) -> Dict[str, Any]:
    """
    Calculate market-wide volume metrics for comparison
    """
    try:
        import yfinance as yf
        
        market_volume_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='2mo')  # Get enough data for 30-day rolling
                
                if len(data) >= 30:
                    volume_analysis = calculate_volume_analysis(data)
                    if volume_analysis.get('calculation_success', False):
                        market_volume_data[symbol] = {
                            'volume_regime': volume_analysis.get('volume_regime', 'Unknown'),
                            'volume_ratio_5d_30d': volume_analysis.get('volume_ratio_5d_30d', 1.0),
                            'regime_score': volume_analysis.get('regime_score', 50),
                            'volume_trend': volume_analysis.get('volume_5d_trend_direction', 'Unknown')
                        }
                        
            except Exception as e:
                logger.error(f"Error fetching market data for {symbol}: {e}")
                continue
        
        if market_volume_data:
            # Calculate overall market volume environment
            avg_regime_score = sum([data['regime_score'] for data in market_volume_data.values()]) / len(market_volume_data)
            avg_volume_ratio = sum([data['volume_ratio_5d_30d'] for data in market_volume_data.values()]) / len(market_volume_data)
            
            # Market volume regime classification
            if avg_regime_score >= 80:
                market_regime = "High Volume Market Environment"
            elif avg_regime_score >= 65:  
                market_regime = "Above Normal Volume Environment"
            elif avg_regime_score >= 35:
                market_regime = "Normal Volume Environment"
            else:
                market_regime = "Low Volume Market Environment"
            
            return {
                'individual_metrics': market_volume_data,
                'market_regime': market_regime,
                'avg_regime_score': round(float(avg_regime_score), 1),
                'avg_volume_ratio': round(float(avg_volume_ratio), 2),
                'sample_symbols': symbols,
                'calculation_success': True
            }
        else:
            return {
                'error': 'Could not retrieve market volume data',
                'calculation_success': False
            }
            
    except Exception as e:
        logger.error(f"Market volume comparison error: {e}")
        return {
            'error': f'Market volume analysis failed: {str(e)}',
            'calculation_success': False
        }
