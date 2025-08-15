"""
VWV Trading System - Enhanced Technical Composite Score v5.0.0
Major functionality change: Complete composite scoring algorithm implementation
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def calculate_composite_technical_score(analysis_results: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate a comprehensive technical composite score from multiple indicators.
    
    Score Range: 0-100
    - 0-20: Very Bearish
    - 21-40: Bearish  
    - 41-60: Neutral
    - 61-80: Bullish
    - 81-100: Very Bullish
    
    Returns:
        Tuple of (composite_score, component_details)
    """
    try:
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
        
        # Initialize component scores and weights
        component_scores = {}
        total_weight = 0
        weighted_sum = 0
        
        # === 1. MOMENTUM INDICATORS (40% total weight) ===
        
        # RSI (14) - 12% weight
        rsi_14 = comprehensive_technicals.get('rsi_14', 50)
        if rsi_14 is not None:
            # RSI scoring: optimal around 45-55, penalize extremes
            if rsi_14 < 30:
                rsi_score = 20 + (rsi_14 / 30) * 30  # 20-50 for oversold recovery
            elif rsi_14 > 70:
                rsi_score = 100 - ((rsi_14 - 70) / 30) * 30  # 70-100 for overbought
            else:
                rsi_score = 30 + ((rsi_14 - 30) / 40) * 40  # 30-70 linear
            
            rsi_weight = 12.0
            component_scores['rsi'] = {
                'value': rsi_14,
                'score': rsi_score,
                'weight': rsi_weight,
                'description': f"RSI({rsi_14:.1f})"
            }
            weighted_sum += rsi_score * rsi_weight
            total_weight += rsi_weight
        
        # MFI (14) - 10% weight 
        mfi_14 = comprehensive_technicals.get('mfi_14', 50)
        if mfi_14 is not None:
            # MFI scoring: similar to RSI but slightly different thresholds
            if mfi_14 < 20:
                mfi_score = 15 + (mfi_14 / 20) * 35
            elif mfi_14 > 80:
                mfi_score = 95 - ((mfi_14 - 80) / 20) * 25
            else:
                mfi_score = 25 + ((mfi_14 - 20) / 60) * 50
            
            mfi_weight = 10.0
            component_scores['mfi'] = {
                'value': mfi_14,
                'score': mfi_score,
                'weight': mfi_weight,
                'description': f"MFI({mfi_14:.1f})"
            }
            weighted_sum += mfi_score * mfi_weight
            total_weight += mfi_weight
        
        # Stochastic %K - 10% weight
        stochastic = comprehensive_technicals.get('stochastic', {})
        stoch_k = stochastic.get('k', 50) if stochastic else 50
        if stoch_k is not None:
            # Stochastic: penalize extreme overbought more heavily
            if stoch_k < 20:
                stoch_score = 25 + (stoch_k / 20) * 25  # Oversold can be bullish
            elif stoch_k > 80:
                stoch_score = 80 - ((stoch_k - 80) / 20) * 40  # Heavily penalize extreme overbought
            else:
                stoch_score = 40 + ((stoch_k - 20) / 60) * 35
            
            stoch_weight = 10.0
            component_scores['stochastic'] = {
                'value': stoch_k,
                'score': stoch_score,
                'weight': stoch_weight,
                'description': f"Stoch({stoch_k:.1f})"
            }
            weighted_sum += stoch_score * stoch_weight
            total_weight += stoch_weight
        
        # Williams %R - 8% weight
        williams_r = comprehensive_technicals.get('williams_r', -50)
        if williams_r is not None:
            # Convert Williams %R (-100 to 0) to 0-100 scale
            williams_normalized = (williams_r + 100)  # Now 0-100
            
            if williams_normalized < 20:
                williams_score = 20 + (williams_normalized / 20) * 30
            elif williams_normalized > 80:
                williams_score = 90 - ((williams_normalized - 80) / 20) * 30
            else:
                williams_score = 35 + ((williams_normalized - 20) / 60) * 40
            
            williams_weight = 8.0
            component_scores['williams_r'] = {
                'value': williams_r,
                'score': williams_score,
                'weight': williams_weight,
                'description': f"Williams({williams_r:.1f})"
            }
            weighted_sum += williams_score * williams_weight
            total_weight += williams_weight
        
        # === 2. TREND INDICATORS (30% total weight) ===
        
        # MACD Histogram - 15% weight
        macd_data = comprehensive_technicals.get('macd', {})
        macd_hist = macd_data.get('histogram', 0) if macd_data else 0
        if macd_hist is not None:
            # MACD Histogram: positive = bullish, negative = bearish
            # Normalize around typical MACD values
            macd_normalized = max(-1, min(1, macd_hist * 100))  # Scale and clamp
            macd_score = 50 + (macd_normalized * 40)  # -1 to 1 becomes 10 to 90
            
            macd_weight = 15.0
            component_scores['macd'] = {
                'value': macd_hist,
                'score': macd_score,
                'weight': macd_weight,
                'description': f"MACD({macd_hist:.4f})"
            }
            weighted_sum += macd_score * macd_weight
            total_weight += macd_weight
        
        # Bollinger Band Position - 10% weight
        bb_data = comprehensive_technicals.get('bollinger_bands', {})
        bb_position = bb_data.get('position', 50) if bb_data else 50
        if bb_position is not None:
            # BB Position: 0-100, but extreme positions can be reversal signals
            if bb_position < 10:
                bb_score = 35 + (bb_position / 10) * 15  # Oversold bounce potential
            elif bb_position > 90:
                bb_score = 75 - ((bb_position - 90) / 10) * 25  # Overbought risk
            else:
                bb_score = 25 + (bb_position / 100) * 50  # Linear for middle range
            
            bb_weight = 10.0
            component_scores['bollinger'] = {
                'value': bb_position,
                'score': bb_score,
                'weight': bb_weight,
                'description': f"BB Position({bb_position:.1f}%)"
            }
            weighted_sum += bb_score * bb_weight
            total_weight += bb_weight
        
        # EMA Trend Strength - 5% weight
        current_price = analysis_results.get('current_price', 0)
        if fibonacci_emas and current_price > 0:
            ema_21 = fibonacci_emas.get('ema_21', current_price)
            ema_55 = fibonacci_emas.get('ema_55', current_price)
            ema_89 = fibonacci_emas.get('ema_89', current_price)
            
            # Count how many EMAs price is above
            emas_above = sum([
                current_price > ema_21,
                current_price > ema_55, 
                current_price > ema_89
            ])
            
            ema_score = 20 + (emas_above / 3) * 60  # 0-3 EMAs above = 20-80 score
            ema_weight = 5.0
            
            component_scores['ema_trend'] = {
                'value': emas_above,
                'score': ema_score,
                'weight': ema_weight,
                'description': f"EMA Trend({emas_above}/3)"
            }
            weighted_sum += ema_score * ema_weight
            total_weight += ema_weight
        
        # === 3. VOLUME CONFIRMATION (20% total weight) ===
        
        # Volume Ratio - 20% weight
        volume_ratio = comprehensive_technicals.get('volume_ratio', 1.0)
        if volume_ratio is not None:
            # Volume ratio scoring: >1.5 very bullish, <0.7 bearish
            if volume_ratio >= 2.0:
                vol_score = 90
            elif volume_ratio >= 1.5:
                vol_score = 70 + ((volume_ratio - 1.5) / 0.5) * 20
            elif volume_ratio >= 1.0:
                vol_score = 50 + ((volume_ratio - 1.0) / 0.5) * 20
            elif volume_ratio >= 0.7:
                vol_score = 30 + ((volume_ratio - 0.7) / 0.3) * 20
            else:
                vol_score = 30 * (volume_ratio / 0.7)
            
            vol_weight = 20.0
            component_scores['volume'] = {
                'value': volume_ratio,
                'score': vol_score,
                'weight': vol_weight,
                'description': f"Volume({volume_ratio:.2f}x)"
            }
            weighted_sum += vol_score * vol_weight
            total_weight += vol_weight
        
        # === 4. VOLATILITY ASSESSMENT (10% total weight) ===
        
        # 20-day Volatility - 10% weight
        volatility_20d = comprehensive_technicals.get('volatility_20d', 20)
        if volatility_20d is not None and volatility_20d > 0:
            # Volatility scoring: moderate vol (15-25%) optimal, extreme vol penalized
            if volatility_20d < 10:
                vol_score = 40  # Very low vol - stagnant
            elif volatility_20d < 15:
                vol_score = 60 + ((15 - volatility_20d) / 5) * 10
            elif volatility_20d <= 25:
                vol_score = 75  # Optimal range
            elif volatility_20d <= 40:
                vol_score = 75 - ((volatility_20d - 25) / 15) * 25
            else:
                vol_score = 30  # Extreme volatility
            
            vol_weight = 10.0
            component_scores['volatility'] = {
                'value': volatility_20d,
                'score': vol_score,
                'weight': vol_weight,
                'description': f"Volatility({volatility_20d:.1f}%)"
            }
            weighted_sum += vol_score * vol_weight
            total_weight += vol_weight
        
        # === FINAL CALCULATION ===
        
        if total_weight > 0:
            composite_score = weighted_sum / total_weight
            composite_score = max(0, min(100, composite_score))  # Clamp to 0-100
        else:
            composite_score = 50.0  # Default if no data
        
        # Create detailed breakdown
        score_details = {
            'composite_score': round(composite_score, 1),
            'total_weight': total_weight,
            'components': component_scores,
            'interpretation': get_score_interpretation(composite_score),
            'signal_strength': get_signal_strength(composite_score),
            'component_count': len(component_scores)
        }
        
        return round(composite_score, 1), score_details
        
    except Exception as e:
        # Fallback on error
        return 50.0, {'error': f'Calculation failed: {str(e)}'}

def get_score_interpretation(score: float) -> str:
    """Get textual interpretation of the composite score."""
    if score >= 80:
        return "Very Bullish"
    elif score >= 65:
        return "Bullish"
    elif score >= 55:
        return "Slightly Bullish"
    elif score >= 45:
        return "Neutral"
    elif score >= 35:
        return "Slightly Bearish"
    elif score >= 20:
        return "Bearish"
    else:
        return "Very Bearish"

def get_signal_strength(score: float) -> str:
    """Get signal strength classification."""
    if score >= 75:
        return "STRONG"
    elif score >= 60:
        return "MODERATE"
    elif score >= 40:
        return "WEAK"
    else:
        return "VERY_WEAK"

def get_color_for_score(score: float) -> str:
    """Get color code for score visualization."""
    if score >= 70:
        return "#22C55E"  # Green
    elif score >= 55:
        return "#84CC16"  # Light green
    elif score >= 45:
        return "#EAB308"  # Yellow
    elif score >= 30:
        return "#F97316"  # Orange
    else:
        return "#EF4444"  # Red
