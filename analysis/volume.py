"""
Volume analysis module for VWV Trading System v4.2.1
Complete Volume Analysis with 5D/30D rolling metrics and regime detection
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def calculate_complete_volume_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate complete volume analysis with 5D/30D rolling metrics and regime detection"""
    try:
        if len(data) < 30:
            return {
                'error': 'Insufficient data for volume analysis',
                'volume_regime': 'Unknown',
                'volume_score': 50
            }

        volume = data['Volume']
        current_volume = float(volume.iloc[-1])
        
        # 5-Day Rolling Volume Analysis
        volume_5d = volume.rolling(5).mean()
        current_5d_avg = float(volume_5d.iloc[-1]) if not pd.isna(volume_5d.iloc[-1]) else current_volume
        
        # Volume trend over last 5 days
        if len(volume_5d) >= 5:
            volume_5d_trend = (current_5d_avg - float(volume_5d.iloc[-5])) / float(volume_5d.iloc[-5]) * 100
        else:
            volume_5d_trend = 0.0
            
        # 30-Day Volume Comparison
        volume_30d = volume.rolling(30).mean()
        volume_30d_avg = float(volume_30d.iloc[-1]) if not pd.isna(volume_30d.iloc[-1]) else current_volume
        
        # Volume ratio (current vs 30-day average)
        volume_ratio = current_volume / volume_30d_avg if volume_30d_avg > 0 else 1.0
        
        # Volume Z-Score for breakout detection
        volume_std = volume.rolling(30).std().iloc[-1]
        if not pd.isna(volume_std) and volume_std > 0:
            volume_zscore = (current_volume - volume_30d_avg) / volume_std
        else:
            volume_zscore = 0.0
            
        # Volume Regime Classification
        if volume_ratio >= 2.0:
            volume_regime = "Extreme High"
            volume_score = 95
        elif volume_ratio >= 1.5:
            volume_regime = "High"
            volume_score = 80
        elif volume_ratio >= 1.2:
            volume_regime = "Above Average"
            volume_score = 65
        elif volume_ratio >= 0.8:
            volume_regime = "Normal"
            volume_score = 50
        elif volume_ratio >= 0.5:
            volume_regime = "Below Average"
            volume_score = 35
        else:
            volume_regime = "Very Low"
            volume_score = 20
            
        # Volume-based options strategy
        if volume_regime in ["Extreme High", "High"]:
            options_strategy = "Sell premium - high activity suggests movement completion"
        elif volume_regime == "Above Average":
            options_strategy = "Neutral strategies - confirm direction first"
        elif volume_regime == "Normal":
            options_strategy = "Standard strategies - normal market conditions"
        else:
            options_strategy = "Buy premium - low volume suggests potential breakout"
            
        # Trading implications based on volume analysis
        if volume_regime in ["Extreme High", "High"]:
            trading_implications = "High conviction moves, watch for exhaustion signals"
        elif volume_regime == "Above Average":
            trading_implications = "Increasing interest, monitor for continuation"
        elif volume_regime == "Normal":
            trading_implications = "Standard market participation, no volume edge"
        else:
            trading_implications = "Low participation, be cautious of false moves"
            
        # Volume strength factor for technical scoring (0.85 to 1.3)
        if volume_score >= 80:
            volume_strength_factor = 1.3
        elif volume_score >= 65:
            volume_strength_factor = 1.15  
        elif volume_score >= 35:
            volume_strength_factor = 1.0
        else:
            volume_strength_factor = 0.85
            
        return {
            'current_volume': int(current_volume),
            'volume_5d_avg': int(current_5d_avg),
            'volume_30d_avg': int(volume_30d_avg),
            'volume_ratio': round(volume_ratio, 2),
            'volume_trend_5d': round(volume_5d_trend, 2),
            'volume_zscore': round(volume_zscore, 2),
            'volume_regime': volume_regime,
            'volume_score': volume_score,
            'options_strategy': options_strategy,
            'trading_implications': trading_implications,
            'volume_strength_factor': volume_strength_factor
        }
        
    except Exception as e:
        logger.error(f"Volume analysis calculation error: {e}")
        return {
            'error': f'Volume analysis failed: {str(e)}',
            'volume_regime': 'Unknown',
            'volume_score': 50,
            'volume_strength_factor': 1.0
        }

def get_volume_trading_implications(volume_regime: str, volume_score: int) -> str:
    """Get detailed trading implications based on volume analysis"""
    if volume_regime == "Extreme High":
        return """
        Extreme volume suggests climactic action:
        • Watch for reversal signals - exhaustion possible
        • High conviction breakouts/breakdowns likely valid
        • Consider profit-taking on existing positions
        • New entries require extra confirmation
        """
    elif volume_regime == "High":
        return """
        High volume indicates strong conviction:
        • Breakouts more likely to sustain
        • Trend continuation probable
        • Good environment for momentum strategies
        • Stop losses can be tighter due to conviction
        """
    elif volume_regime == "Above Average":
        return """
        Increasing participation suggests building interest:
        • Monitor for acceleration of current trends
        • Good setup for continuation patterns
        • Volume confirms price action validity
        • Standard risk management applies
        """
    elif volume_regime == "Normal":
        return """
        Normal volume conditions:
        • No significant volume edge present
        • Rely on other technical factors
        • Standard position sizing appropriate
        • Normal stop loss distances recommended
        """
    else:  # Below Average or Very Low
        return """
        Low volume suggests lack of conviction:
        • Be skeptical of breakouts/breakdowns
        • Prefer range-bound strategies
        • Reduce position sizes due to uncertainty
        • Watch for volume expansion for confirmation
        """

@safe_calculation_wrapper        
def calculate_market_wide_volume_analysis(show_debug=False) -> Dict[str, Any]:
    """Calculate market-wide volume environment - SAFE MINIMAL VERSION"""
    try:
        # Minimal safe implementation to prevent 503 errors
        return {
            'market_volume_score': 50.0,
            'market_volume_regime': 'Normal Volume',
            'individual_analysis': {
                'SPY': {'volume_score': 50, 'volume_regime': 'Normal'},
                'QQQ': {'volume_score': 50, 'volume_regime': 'Normal'}, 
                'IWM': {'volume_score': 50, 'volume_regime': 'Normal'}
            },
            'analysis_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'note': 'Safe minimal version - full implementation needs debugging'
        }
        
    except Exception as e:
        logger.error(f"Market-wide volume analysis error: {e}")
        return {
            'error': f'Market-wide volume analysis failed: {str(e)}',
            'market_volume_regime': 'Unknown',
            'market_volume_score': 50
        }
