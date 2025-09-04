"""
Filename: analysis/volume.py
VWV Trading System v4.2.1
Created/Updated: 2025-09-04 11:03:05 EDT
Version: 1.0.0 - Initial integration of the module
Purpose: Provides detailed volume analysis for a given security.
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
        if 'Volume' not in data.columns or data['Volume'].isnull().all():
            return {'error': 'Volume data not available'}
        if len(data) < 30:
            return {'error': 'Insufficient data for volume analysis (requires 30 days)'}

        volume = data['Volume']
        current_volume = float(volume.iloc[-1])
        
        # 5-Day Rolling Volume Analysis
        volume_5d_avg = volume.rolling(5).mean().iloc[-1]
        
        # Volume trend over last 5 days
        volume_5d_trend = ((volume_5d_avg / volume.rolling(5).mean().iloc[-5]) - 1) * 100 if len(volume) >= 10 else 0.0
            
        # 30-Day Volume Comparison
        volume_30d_avg = volume.rolling(30).mean().iloc[-1]
        
        # Volume ratio (current vs 30-day average)
        volume_ratio = current_volume / volume_30d_avg if volume_30d_avg > 0 else 1.0
        
        # Volume Z-Score for breakout detection
        volume_std = volume.rolling(30).std().iloc[-1]
        volume_zscore = (current_volume - volume_30d_avg) / volume_std if not pd.isna(volume_std) and volume_std > 0 else 0.0
            
        # Volume Regime Classification
        if volume_ratio >= 2.0:
            volume_regime, volume_score = "Climactic", 95
        elif volume_ratio >= 1.5:
            volume_regime, volume_score = "High", 80
        elif volume_ratio >= 1.2:
            volume_regime, volume_score = "Above Average", 65
        elif volume_ratio >= 0.8:
            volume_regime, volume_score = "Normal", 50
        elif volume_ratio >= 0.5:
            volume_regime, volume_score = "Below Average", 35
        else:
            volume_regime, volume_score = "Very Low", 20
            
        # Trading implications based on volume analysis
        if volume_regime == "Climactic":
            trading_implications = "High conviction, but watch for exhaustion signals."
        elif volume_regime == "High":
            trading_implications = "Strong participation, trend continuation likely."
        elif volume_regime == "Above Average":
            trading_implications = "Increasing interest, monitor for continuation."
        elif volume_regime == "Normal":
            trading_implications = "Standard market participation, no volume edge."
        else:
            trading_implications = "Low participation, be cautious of false moves."
            
        return {
            'current_volume': int(current_volume),
            'volume_5d_avg': int(volume_5d_avg),
            'volume_30d_avg': int(volume_30d_avg),
            'volume_ratio': round(volume_ratio, 2),
            'volume_trend_5d': round(volume_5d_trend, 2),
            'volume_zscore': round(volume_zscore, 2),
            'volume_regime': volume_regime,
            'volume_score': volume_score,
            'trading_implications': trading_implications
        }
        
    except Exception as e:
        logger.error(f"Volume analysis calculation error: {e}")
        return {'error': f'Volume analysis failed: {str(e)}'}

@safe_calculation_wrapper
def calculate_market_wide_volume_analysis(show_debug=False) -> Dict[str, Any]:
    """Calculate market-wide volume environment - SAFE MINIMAL VERSION"""
    return {
        'market_volume_score': 50.0,
        'market_volume_regime': 'Normal Volume',
        'note': 'Safe minimal version - full implementation pending'
    }
