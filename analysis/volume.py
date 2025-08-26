"""
Simple Volatility Analysis Module for VWV Trading System
Working baseline implementation without complex features
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def calculate_simple_volatility_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate simple volatility analysis that actually works"""
    try:
        if len(data) < 20:
            return {
                'error': 'Insufficient data for volatility analysis',
                'volatility_regime': 'Unknown',
                'volatility_10d': 20.0,
                'volatility_20d': 20.0,
                'volatility_percentile': 50.0
            }

        # Calculate returns
        returns = data['Close'].pct_change().dropna()
        
        if len(returns) < 20:
            return {
                'error': 'Insufficient return data',
                'volatility_regime': 'Unknown',
                'volatility_10d': 20.0,
                'volatility_20d': 20.0,
                'volatility_percentile': 50.0
            }
        
        # 10-day volatility (annualized)
        volatility_10d = float(returns.tail(10).std() * np.sqrt(252) * 100)
        
        # 20-day volatility (annualized)  
        volatility_20d = float(returns.tail(20).std() * np.sqrt(252) * 100)
        
        # Calculate volatility percentile (current vs historical)
        if len(returns) >= 60:
            # Use 60-day rolling volatility for percentile calculation
            rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100
            rolling_vol = rolling_vol.dropna()
            
            if len(rolling_vol) > 1:
                current_vol = rolling_vol.iloc[-1]
                volatility_percentile = float((rolling_vol <= current_vol).sum() / len(rolling_vol) * 100)
            else:
                volatility_percentile = 50.0
        else:
            volatility_percentile = 50.0
        
        # Determine volatility regime based on percentile and absolute levels
        if volatility_percentile >= 80 or volatility_20d >= 40:
            volatility_regime = "Very High"
        elif volatility_percentile >= 65 or volatility_20d >= 30:
            volatility_regime = "High"
        elif volatility_percentile >= 35 and volatility_20d >= 15:
            volatility_regime = "Normal"
        elif volatility_percentile >= 20 or volatility_20d >= 10:
            volatility_regime = "Low"
        else:
            volatility_regime = "Very Low"
        
        return {
            'volatility_10d': round(volatility_10d, 1),
            'volatility_20d': round(volatility_20d, 1),
            'volatility_percentile': round(volatility_percentile, 0),
            'volatility_regime': volatility_regime,
            'analysis_success': True
        }
        
    except Exception as e:
        logger.error(f"Simple volatility analysis error: {e}")
        return {
            'error': f'Volatility analysis failed: {str(e)}',
            'volatility_regime': 'Unknown',
            'volatility_10d': 20.0,
            'volatility_20d': 20.0,
            'volatility_percentile': 50.0
        }
