"""
Enhanced Volume Analysis Module for VWV Trading System v4.2.1
Comprehensive volume indicators with composite scoring system (0-100 scale)
Date: August 21, 2025 - 3:15 PM EST
"""
import pandas as pd
import numpy as np
import streamlit as st
import logging
from typing import Dict, Any, Optional, Tuple
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

# Volume Analysis Configuration and Weights
VOLUME_WEIGHTS = {
    'obv_signal': 0.15,           # On-Balance Volume
    'mfi_signal': 0.15,           # Money Flow Index  
    'ad_line_signal': 0.12,       # Accumulation/Distribution Line
    'cmf_signal': 0.10,           # Chaikin Money Flow
    'vroc_signal': 0.08,          # Volume Rate of Change
    'relative_volume': 0.08,      # Current vs Historical Volume
    'force_index_signal': 0.07,   # Force Index
    'volume_oscillator': 0.06,    # Volume Oscillator
    'ease_of_movement': 0.06,     # Ease of Movement
    'vpt_signal': 0.04,           # Volume Price Trend
    'volume_momentum': 0.04,      # Volume Momentum
    'volume_breakout': 0.03,      # Volume Breakout Analysis
    'volume_trend': 0.02,         # Volume Trend Analysis
    'volume_divergence': 0.02     # Price-Volume Divergence
}

@safe_calculation_wrapper
def calculate_obv_signal(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate On-Balance Volume signal (0-100 scale)"""
    try:
        if len(data) < 20:
            return {'score': 50, 'value': 0, 'trend': 'Neutral', 'details': 'Insufficient data'}
        
        close = data['Close']
        volume = data['Volume']
        
        # Calculate OBV
        obv = [0]
        for i in range(1, len(data)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        
        obv_series = pd.Series(obv, index=data.index)
        
        # Calculate OBV moving averages for trend
        obv_ma_10 = obv_series.rolling(10).mean()
        obv_ma_20 = obv_series.rolling(20).mean()
        
        current_obv = obv_series.iloc[-1]
        current_ma_10 = obv_ma_10.iloc[-1]
        current_ma_20 = obv_ma_20.iloc[-1]
        
        # Score based on OBV trend and position
        if current_obv > current_ma_10 > current_ma_20:
            score = 85  # Strong bullish
            trend = 'Strong Bullish'
        elif current_obv > current_ma_10:
            score = 70  # Bullish
            trend = 'Bullish'
        elif current_obv < current_ma_10 < current_ma_20:
            score = 15  # Strong bearish
            trend = 'Strong Bearish'
        elif current_obv < current_ma_10:
            score = 30  # Bearish
            trend = 'Bearish'
        else:
            score = 50  # Neutral
            trend = 'Neutral'
        
        return {
            'score': score,
            'value': round(float(current_obv), 0),
            'trend': trend,
            'details': f"OBV: {current_obv:.0f}, MA10: {current_ma_10:.0f}"
        }
        
    except Exception as e:
        logger.error(f"OBV calculation error: {e}")
        return {'score': 50, 'value': 0, 'trend': 'Error', 'details': str(e)}

@safe_calculation_wrapper
def calculate_mfi_signal(data: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
    """Calculate Money Flow Index signal (0-100 scale)"""
    try:
        if len(data) < period + 5:
            return {'score': 50, 'value': 50, 'trend': 'Neutral', 'details': 'Insufficient data'}
        
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']
        
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate money flow
        money_flow = typical_price * volume
        
        # Calculate positive and negative money flow
        pos_flow = []
        neg_flow = []
        
        for i in range(1, len(data)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                pos_flow.append(money_flow.iloc[i])
                neg_flow.append(0)
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                pos_flow.append(0)
                neg_flow.append(money_flow.iloc[i])
            else:
                pos_flow.append(0)
                neg_flow.append(0)
        
        pos_flow_series = pd.Series([0] + pos_flow, index=data.index)
        neg_flow_series = pd.Series([0] + neg_flow, index=data.index)
        
        # Calculate MFI
        pos_mf = pos_flow_series.rolling(period).sum()
        neg_mf = neg_flow_series.rolling(period).sum()
        
        money_ratio = pos_mf / (neg_mf + 1e-10)  # Avoid division by zero
        mfi = 100 - (100 / (1 + money_ratio))
        
        current_mfi = float(mfi.iloc[-1])
        
        # Score based on MFI levels
        if current_mfi >= 80:
            score = 20  # Overbought (bearish)
            trend = 'Overbought'
        elif current_mfi >= 60:
            score = 40  # Moderately overbought
            trend = 'Moderately Overbought'
        elif current_mfi >= 40:
            score = 60  # Neutral
            trend = 'Neutral'
        elif current_mfi >= 20:
            score = 80  # Moderately oversold (bullish)
            trend = 'Moderately Oversold'
        else:
            score = 90  # Oversold (very bullish)
            trend = 'Oversold'
        
        return {
            'score': score,
            'value': round(current_mfi, 1),
            'trend': trend,
            'details': f"MFI: {current_mfi:.1f} (Oversold<20, Overbought>80)"
        }
        
    except Exception as e:
        logger.error(f"MFI calculation error: {e}")
        return {'score': 50, 'value': 50, 'trend': 'Error', 'details': str(e)}

@safe_calculation_wrapper
def calculate_ad_line_signal(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Accumulation/Distribution Line signal (0-100 scale)"""
    try:
        if len(data) < 20:
            return {'score': 50, 'value': 0, 'trend': 'Neutral', 'details': 'Insufficient data'}
        
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']
        
        # Calculate Money Flow Multiplier
        clv = ((close - low) - (high - close)) / (high - low + 1e-10)
        
        # Calculate A/D Line
        ad_line = (clv * volume).cumsum()
        
        # Calculate trend using moving averages
        ad_ma_10 = ad_line.rolling(10).mean()
        ad_ma_20 = ad_line.rolling(20).mean()
        
        current_ad = ad_line.iloc[-1]
        current_ma_10 = ad_ma_10.iloc[-1]
        current_ma_20 = ad_ma_20.iloc[-1]
        
        # Score based on A/D trend
        if current_ad > current_ma_10 > current_ma_20:
            score = 85
            trend = 'Strong Accumulation'
        elif current_ad > current_ma_10:
            score = 70
            trend = 'Accumulation'
        elif current_ad < current_ma_10 < current_ma_20:
            score = 15
            trend = 'Strong Distribution'
        elif current_ad < current_ma_10:
            score = 30
            trend = 'Distribution'
        else:
            score = 50
            trend = 'Neutral'
        
        return {
            'score': score,
            'value': round(float(current_ad), 0),
            'trend': trend,
            'details': f"A/D Line: {current_ad:.0f}"
        }
        
    except Exception as e:
        logger.error(f"A/D Line calculation error: {e}")
        return {'score': 50, 'value': 0, 'trend': 'Error', 'details': str(e)}

@safe_calculation_wrapper
def calculate_cmf_signal(data: pd.DataFrame, period: int = 20) -> Dict[str, Any]:
    """Calculate Chaikin Money Flow signal (0-100 scale)"""
    try:
        if len(data) < period + 5:
            return {'score': 50, 'value': 0, 'trend': 'Neutral', 'details': 'Insufficient data'}
        
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']
        
        # Calculate Money Flow Multiplier
        clv = ((close - low) - (high - close)) / (high - low + 1e-10)
        
        # Calculate Money Flow Volume
        mfv = clv * volume
        
        # Calculate CMF
        cmf = mfv.rolling(period).sum() / volume.rolling(period).sum()
        current_cmf = float(cmf.iloc[-1])
        
        # Score based on CMF levels
        if current_cmf >= 0.25:
            score = 90
            trend = 'Very Strong Buying'
        elif current_cmf >= 0.10:
            score = 75
            trend = 'Strong Buying'
        elif current_cmf >= 0.05:
            score = 65
            trend = 'Moderate Buying'
        elif current_cmf >= -0.05:
            score = 50
            trend = 'Neutral'
        elif current_cmf >= -0.10:
            score = 35
            trend = 'Moderate Selling'
        elif current_cmf >= -0.25:
            score = 25
            trend = 'Strong Selling'
        else:
            score = 10
            trend = 'Very Strong Selling'
        
        return {
            'score': score,
            'value': round(current_cmf, 3),
            'trend': trend,
            'details': f"CMF: {current_cmf:.3f} (-0.25 to +0.25 range)"
        }
        
    except Exception as e:
        logger.error(f"CMF calculation error: {e}")
        return {'score': 50, 'value': 0, 'trend': 'Error', 'details': str(e)}

@safe_calculation_wrapper
def calculate_vroc_signal(data: pd.DataFrame, period: int = 12) -> Dict[str, Any]:
    """Calculate Volume Rate of Change signal (0-100 scale)"""
    try:
        if len(data) < period + 5:
            return {'score': 50, 'value': 0, 'trend': 'Neutral', 'details': 'Insufficient data'}
        
        volume = data['Volume']
        vroc = ((volume - volume.shift(period)) / volume.shift(period)) * 100
        current_vroc = float(vroc.iloc[-1])
        
        # Score based on VROC levels
        if current_vroc >= 100:
            score = 95
            trend = 'Extreme Volume Surge'
        elif current_vroc >= 50:
            score = 85
            trend = 'High Volume Growth'
        elif current_vroc >= 20:
            score = 70
            trend = 'Moderate Volume Growth'
        elif current_vroc >= -20:
            score = 50
            trend = 'Normal Volume'
        elif current_vroc >= -50:
            score = 30
            trend = 'Volume Decline'
        else:
            score = 15
            trend = 'Volume Collapse'
        
        return {
            'score': score,
            'value': round(current_vroc, 1),
            'trend': trend,
            'details': f"VROC: {current_vroc:.1f}% ({period}-period)"
        }
        
    except Exception as e:
        logger.error(f"VROC calculation error: {e}")
        return {'score': 50, 'value': 0, 'trend': 'Error', 'details': str(e)}

@safe_calculation_wrapper
def calculate_relative_volume(data: pd.DataFrame, period: int = 30) -> Dict[str, Any]:
    """Calculate Relative Volume signal (0-100 scale)"""
    try:
        if len(data) < period + 5:
            return {'score': 50, 'value': 1.0, 'trend': 'Normal', 'details': 'Insufficient data'}
        
        volume = data['Volume']
        current_volume = volume.iloc[-1]
        avg_volume = volume.rolling(period).mean().iloc[-1]
        
        rel_vol = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Score based on relative volume
        if rel_vol >= 3.0:
            score = 95
            trend = 'Extreme High'
        elif rel_vol >= 2.0:
            score = 85
            trend = 'Very High'
        elif rel_vol >= 1.5:
            score = 75
            trend = 'High'
        elif rel_vol >= 1.2:
            score = 65
            trend = 'Above Normal'
        elif rel_vol >= 0.8:
            score = 50
            trend = 'Normal'
        elif rel_vol >= 0.5:
            score = 35
            trend = 'Below Normal'
        else:
            score = 20
            trend = 'Low'
        
        return {
            'score': score,
            'value': round(rel_vol, 2),
            'trend': trend,
            'details': f"Current: {current_volume:.0f}, Avg: {avg_volume:.0f}"
        }
        
    except Exception as e:
        logger.error(f"Relative volume calculation error: {e}")
        return {'score': 50, 'value': 1.0, 'trend': 'Error', 'details': str(e)}

@safe_calculation_wrapper
def calculate_force_index_signal(data: pd.DataFrame, period: int = 13) -> Dict[str, Any]:
    """Calculate Force Index signal (0-100 scale)"""
    try:
        if len(data) < period + 5:
            return {'score': 50, 'value': 0, 'trend': 'Neutral', 'details': 'Insufficient data'}
        
        close = data['Close']
        volume = data['Volume']
        
        # Calculate Force Index
        force_index = (close - close.shift(1)) * volume
        force_ma = force_index.rolling(period).mean()
        
        current_force = float(force_ma.iloc[-1])
        
        # Normalize by average volume for scoring
        avg_volume = volume.rolling(period).mean().iloc[-1]
        normalized_force = current_force / avg_volume if avg_volume > 0 else 0
        
        # Score based on normalized force index
        if normalized_force >= 2.0:
            score = 90
            trend = 'Very Strong Bullish'
        elif normalized_force >= 1.0:
            score = 75
            trend = 'Strong Bullish'
        elif normalized_force >= 0.5:
            score = 65
            trend = 'Moderate Bullish'
        elif normalized_force >= -0.5:
            score = 50
            trend = 'Neutral'
        elif normalized_force >= -1.0:
            score = 35
            trend = 'Moderate Bearish'
        elif normalized_force >= -2.0:
            score = 25
            trend = 'Strong Bearish'
        else:
            score = 10
            trend = 'Very Strong Bearish'
        
        return {
            'score': score,
            'value': round(current_force, 0),
            'trend': trend,
            'details': f"Force Index: {current_force:.0f}"
        }
        
    except Exception as e:
        logger.error(f"Force Index calculation error: {e}")
        return {'score': 50, 'value': 0, 'trend': 'Error', 'details': str(e)}

@safe_calculation_wrapper
def calculate_volume_oscillator(data: pd.DataFrame, fast: int = 5, slow: int = 20) -> Dict[str, Any]:
    """Calculate Volume Oscillator signal (0-100 scale)"""
    try:
        if len(data) < slow + 5:
            return {'score': 50, 'value': 0, 'trend': 'Neutral', 'details': 'Insufficient data'}
        
        volume = data['Volume']
        fast_ma = volume.rolling(fast).mean()
        slow_ma = volume.rolling(slow).mean()
        
        vol_osc = ((fast_ma - slow_ma) / slow_ma) * 100
        current_osc = float(vol_osc.iloc[-1])
        
        # Score based on volume oscillator
        if current_osc >= 20:
            score = 85
            trend = 'Strong Volume Momentum'
        elif current_osc >= 10:
            score = 70
            trend = 'Positive Volume Momentum'
        elif current_osc >= 5:
            score = 60
            trend = 'Mild Positive Momentum'
        elif current_osc >= -5:
            score = 50
            trend = 'Neutral'
        elif current_osc >= -10:
            score = 40
            trend = 'Mild Negative Momentum'
        elif current_osc >= -20:
            score = 30
            trend = 'Negative Volume Momentum'
        else:
            score = 15
            trend = 'Weak Volume Momentum'
        
        return {
            'score': score,
            'value': round(current_osc, 1),
            'trend': trend,
            'details': f"Vol Oscillator: {current_osc:.1f}%"
        }
        
    except Exception as e:
        logger.error(f"Volume Oscillator calculation error: {e}")
        return {'score': 50, 'value': 0, 'trend': 'Error', 'details': str(e)}

@safe_calculation_wrapper
def calculate_ease_of_movement(data: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
    """Calculate Ease of Movement signal (0-100 scale)"""
    try:
        if len(data) < period + 5:
            return {'score': 50, 'value': 0, 'trend': 'Neutral', 'details': 'Insufficient data'}
        
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        # Calculate Distance Moved
        dm = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
        
        # Calculate Box Height
        bh = volume / (high - low + 1e-10)
        
        # Calculate EMV
        emv = dm / bh
        emv_ma = emv.rolling(period).mean()
        
        current_emv = float(emv_ma.iloc[-1])
        
        # Score based on EMV
        if current_emv >= 1000:
            score = 90
            trend = 'Very Easy Upward'
        elif current_emv >= 500:
            score = 75
            trend = 'Easy Upward'
        elif current_emv >= 100:
            score = 65
            trend = 'Moderate Upward'
        elif current_emv >= -100:
            score = 50
            trend = 'Neutral'
        elif current_emv >= -500:
            score = 35
            trend = 'Moderate Downward'
        elif current_emv >= -1000:
            score = 25
            trend = 'Easy Downward'
        else:
            score = 10
            trend = 'Very Easy Downward'
        
        return {
            'score': score,
            'value': round(current_emv, 0),
            'trend': trend,
            'details': f"EMV: {current_emv:.0f}"
        }
        
    except Exception as e:
        logger.error(f"EMV calculation error: {e}")
        return {'score': 50, 'value': 0, 'trend': 'Error', 'details': str(e)}

@safe_calculation_wrapper
def calculate_composite_volume_score(data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    """Calculate composite volume score (0-100) with component breakdown"""
    try:
        if len(data) < 30:
            return 50.0, {'error': 'Insufficient data for volume analysis'}
        
        # Calculate all component signals
        components = {
            'obv_signal': calculate_obv_signal(data),
            'mfi_signal': calculate_mfi_signal(data),
            'ad_line_signal': calculate_ad_line_signal(data),
            'cmf_signal': calculate_cmf_signal(data),
            'vroc_signal': calculate_vroc_signal(data),
            'relative_volume': calculate_relative_volume(data),
            'force_index_signal': calculate_force_index_signal(data),
            'volume_oscillator': calculate_volume_oscillator(data),
            'ease_of_movement': calculate_ease_of_movement(data)
        }
        
        # Calculate additional components
        vpt_signal = calculate_vpt_signal(data)
        volume_momentum = calculate_volume_momentum(data)
        volume_breakout = calculate_volume_breakout_analysis(data)
        volume_trend = calculate_volume_trend_analysis(data)
        volume_divergence = calculate_volume_divergence_analysis(data)
        
        components.update({
            'vpt_signal': vpt_signal,
            'volume_momentum': volume_momentum,
            'volume_breakout': volume_breakout,
            'volume_trend': volume_trend,
            'volume_divergence': volume_divergence
        })
        
        # Calculate weighted composite score
        total_score = 0.0
        total_weight = 0.0
        component_details = {}
        
        for component_name, component_data in components.items():
            if component_name in VOLUME_WEIGHTS and 'score' in component_data:
                weight = VOLUME_WEIGHTS[component_name]
                score = component_data['score']
                total_score += score * weight
                total_weight += weight
                
                component_details[component_name] = {
                    'score': score,
                    'weight': weight,
                    'contribution': score * weight,
                    'trend': component_data.get('trend', 'Unknown'),
                    'value': component_data.get('value', 0),
                    'details': component_data.get('details', '')
                }
        
        # Normalize score to 0-100
        if total_weight > 0:
            composite_score = total_score / total_weight
        else:
            composite_score = 50.0
        
        # Ensure score is within bounds
        composite_score = max(0.0, min(100.0, composite_score))
        
        return round(composite_score, 1), component_details
        
    except Exception as e:
        logger.error(f"Composite volume score calculation error: {e}")
        return 50.0, {'error': str(e)}

@safe_calculation_wrapper
def calculate_vpt_signal(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Volume Price Trend signal"""
    try:
        if len(data) < 20:
            return {'score': 50, 'value': 0, 'trend': 'Neutral', 'details': 'Insufficient data'}
        
        close = data['Close']
        volume = data['Volume']
        
        price_change = close.pct_change()
        vpt = (price_change * volume).cumsum()
        
        # Calculate trend
        vpt_ma = vpt.rolling(10).mean()
        current_vpt = vpt.iloc[-1]
        current_ma = vpt_ma.iloc[-1]
        
        if current_vpt > current_ma:
            score = 70
            trend = 'Bullish'
        elif current_vpt < current_ma:
            score = 30
            trend = 'Bearish'
        else:
            score = 50
            trend = 'Neutral'
        
        return {
            'score': score,
            'value': round(float(current_vpt), 0),
            'trend': trend,
            'details': f"VPT: {current_vpt:.0f}"
        }
        
    except Exception as e:
        logger.error(f"VPT calculation error: {e}")
        return {'score': 50, 'value': 0, 'trend': 'Error', 'details': str(e)}

@safe_calculation_wrapper
def calculate_volume_momentum(data: pd.DataFrame, period: int = 10) -> Dict[str, Any]:
    """Calculate Volume Momentum signal"""
    try:
        if len(data) < period * 2:
            return {'score': 50, 'value': 0, 'trend': 'Neutral', 'details': 'Insufficient data'}
        
        volume = data['Volume']
        vol_roc = volume.pct_change(period)
        vol_momentum = vol_roc.rolling(period).mean()
        
        current_momentum = float(vol_momentum.iloc[-1])
        
        if current_momentum >= 0.2:
            score = 85
            trend = 'Strong Acceleration'
        elif current_momentum >= 0.1:
            score = 70
            trend = 'Moderate Acceleration'
        elif current_momentum >= -0.1:
            score = 50
            trend = 'Stable'
        elif current_momentum >= -0.2:
            score = 30
            trend = 'Moderate Deceleration'
        else:
            score = 15
            trend = 'Strong Deceleration'
        
        return {
            'score': score,
            'value': round(current_momentum * 100, 1),
            'trend': trend,
            'details': f"Momentum: {current_momentum * 100:.1f}%"
        }
        
    except Exception as e:
        logger.error(f"Volume momentum calculation error: {e}")
        return {'score': 50, 'value': 0, 'trend': 'Error', 'details': str(e)}

@safe_calculation_wrapper
def calculate_volume_breakout_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Volume Breakout signal"""
    try:
        if len(data) < 30:
            return {'score': 50, 'value': 0, 'trend': 'Normal', 'details': 'Insufficient data'}
        
        volume = data['Volume']
        vol_mean = volume.rolling(30).mean()
        vol_std = volume.rolling(30).std()
        
        current_vol = volume.iloc[-1]
        threshold = vol_mean.iloc[-1] + 2 * vol_std.iloc[-1]
        
        if current_vol > threshold:
            score = 90
            trend = 'Volume Breakout'
        elif current_vol > vol_mean.iloc[-1] + vol_std.iloc[-1]:
            score = 70
            trend = 'High Volume'
        else:
            score = 50
            trend = 'Normal Volume'
        
        return {
            'score': score,
            'value': round(float(current_vol / vol_mean.iloc[-1]), 2),
            'trend': trend,
            'details': f"Current vs Mean: {current_vol / vol_mean.iloc[-1]:.2f}x"
        }
        
    except Exception as e:
        logger.error(f"Volume breakout calculation error: {e}")
        return {'score': 50, 'value': 0, 'trend': 'Error', 'details': str(e)}

@safe_calculation_wrapper
def calculate_volume_trend_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Volume Trend signal"""
    try:
        if len(data) < 20:
            return {'score': 50, 'value': 0, 'trend': 'Neutral', 'details': 'Insufficient data'}
        
        volume = data['Volume']
        vol_sma_5 = volume.rolling(5).mean()
        vol_sma_20 = volume.rolling(20).mean()
        
        current_5 = vol_sma_5.iloc[-1]
        current_20 = vol_sma_20.iloc[-1]
        
        if current_5 > current_20 * 1.1:
            score = 75
            trend = 'Rising Volume'
        elif current_5 < current_20 * 0.9:
            score = 25
            trend = 'Falling Volume'
        else:
            score = 50
            trend = 'Stable Volume'
        
        return {
            'score': score,
            'value': round(float(current_5 / current_20), 2),
            'trend': trend,
            'details': f"5-day vs 20-day: {current_5 / current_20:.2f}"
        }
        
    except Exception as e:
        logger.error(f"Volume trend calculation error: {e}")
        return {'score': 50, 'value': 0, 'trend': 'Error', 'details': str(e)}

@safe_calculation_wrapper
def calculate_volume_divergence_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Volume-Price Divergence signal"""
    try:
        if len(data) < 20:
            return {'score': 50, 'value': 0, 'trend': 'No Divergence', 'details': 'Insufficient data'}
        
        close = data['Close']
        volume = data['Volume']
        
        # Calculate recent trends
        price_trend = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]
        vol_trend = (volume.rolling(5).mean().iloc[-1] - volume.rolling(5).mean().iloc[-10]) / volume.rolling(5).mean().iloc[-10]
        
        # Check for divergence
        if price_trend > 0.02 and vol_trend < -0.1:
            score = 25  # Bearish divergence
            trend = 'Bearish Divergence'
        elif price_trend < -0.02 and vol_trend > 0.1:
            score = 75  # Bullish divergence
            trend = 'Bullish Divergence'
        else:
            score = 50
            trend = 'No Divergence'
        
        return {
            'score': score,
            'value': round(abs(price_trend - vol_trend) * 100, 1),
            'trend': trend,
            'details': f"Price: {price_trend*100:.1f}%, Vol: {vol_trend*100:.1f}%"
        }
        
    except Exception as e:
        logger.error(f"Volume divergence calculation error: {e}")
        return {'score': 50, 'value': 0, 'trend': 'Error', 'details': str(e)}

@safe_calculation_wrapper
def calculate_complete_volume_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate complete volume analysis with composite score"""
    try:
        if len(data) < 30:
            return {
                'error': 'Insufficient data for volume analysis',
                'composite_score': 50,
                'components': {}
            }
        
        # Calculate composite score and components
        composite_score, components = calculate_composite_volume_score(data)
        
        # Basic volume metrics (maintain compatibility)
        volume = data['Volume']
        current_volume = int(volume.iloc[-1])
        volume_5d_avg = int(volume.rolling(5).mean().iloc[-1])
        volume_30d_avg = int(volume.rolling(30).mean().iloc[-1])
        volume_ratio = current_volume / volume_30d_avg if volume_30d_avg > 0 else 1.0
        
        # Volume regime based on composite score
        if composite_score >= 80:
            volume_regime = "Very Strong"
        elif composite_score >= 65:
            volume_regime = "Strong" 
        elif composite_score >= 55:
            volume_regime = "Above Normal"
        elif composite_score >= 45:
            volume_regime = "Normal"
        elif composite_score >= 35:
            volume_regime = "Below Normal"
        elif composite_score >= 20:
            volume_regime = "Weak"
        else:
            volume_regime = "Very Weak"
        
        # Trading implications based on score
        if composite_score >= 75:
            trading_implications = "Strong buying interest. Consider position accumulation. Watch for continuation."
        elif composite_score >= 60:
            trading_implications = "Moderate buying interest. Good for trend following strategies."
        elif composite_score >= 40:
            trading_implications = "Mixed signals. Wait for clearer direction before major positions."
        elif composite_score >= 25:
            trading_implications = "Weak volume activity. Consider reduced position sizes."
        else:
            trading_implications = "Very weak volume. Avoid new positions, consider exit strategies."
        
        return {
            'composite_score': composite_score,
            'components': components,
            'current_volume': current_volume,
            'volume_5d_avg': volume_5d_avg,
            'volume_30d_avg': volume_30d_avg,
            'volume_ratio': round(volume_ratio, 2),
            'volume_regime': volume_regime,
            'volume_score': int(composite_score),  # For compatibility
            'trading_implications': trading_implications,
            'system_status': 'Enhanced Volume Analysis v4.2.1'
        }
        
    except Exception as e:
        logger.error(f"Complete volume analysis error: {e}")
        return {
            'error': str(e),
            'composite_score': 50,
            'components': {},
            'volume_regime': 'Error',
            'volume_score': 50,
            'trading_implications': 'Analysis unavailable due to error.'
        }

def create_volume_score_bar(score: float) -> str:
    """Create gradient bar HTML for volume composite score (matches technical score bar style)"""
    
    score = round(float(score), 1)
    
    # Determine interpretation and color based on the score  
    if score >= 80:
        interpretation = "Very Strong Volume"
        primary_color = "#00A86B"  # Jade green
    elif score >= 65:
        interpretation = "Strong Volume" 
        primary_color = "#32CD32"  # Lime green
    elif score >= 55:
        interpretation = "Above Normal Volume"
        primary_color = "#9ACD32"  # Yellow green
    elif score >= 45:
        interpretation = "Normal Volume"
        primary_color = "#FFD700"  # Gold
    elif score >= 35:
        interpretation = "Below Normal Volume"
        primary_color = "#FF8C00"  # Dark orange
    elif score >= 20:
        interpretation = "Weak Volume"
        primary_color = "#FF4500"  # Orange red
    else:
        interpretation = "Very Weak Volume"
        primary_color = "#DC143C"  # Crimson
    
    # Create professional gradient bar HTML (same style as technical score bar)
    html = f"""
    <div style="margin-bottom: 1rem;">
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div>
                <span style="font-weight: 700; color: #ffffff; font-size: 1.3em;">
                    Volume Composite Score
                </span>
                <div style="font-size: 0.95em; color: #b0b0b0; margin-top: 0.3rem;">
                    Aggregated signal from 14 volume indicators
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-weight: 700; color: {primary_color}; font-size: 2.2em; line-height: 1;">
                    {score}
                </div>
                <div style="font-size: 0.95em; color: {primary_color}; font-weight: 600;">
                    {interpretation}
                </div>
            </div>
        </div>
        
        <div style="position: relative; width: 100%; height: 28px; 
                    background: linear-gradient(to right, 
                        #DC143C 0%, #FF4500 15%, #FF8C00 30%, #FFD700 50%, 
                        #9ACD32 70%, #32CD32 85%, #00A86B 100%); 
                    border-radius: 14px; 
                    border: 1px solid #404040; 
                    overflow: hidden;
                    box-shadow: inset 0 2px 4px rgba(0,0,0,0.4);">
            
            <div style="position: absolute; left: {score}%; top: 50%; transform: translate(-50%, -50%); 
                        width: 4px; height: 32px; 
                        background: #ffffff; 
                        border: 1px solid #1a1a1a; 
                        border-radius: 4px; 
                        box-shadow: 0 2px 5px rgba(0,0,0,0.5); 
                        z-index: 10;">
            </div>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.8em; color: #a0a0a0;">
            <span>0 (Very Weak)</span>
            <span>50 (Normal)</span>
            <span>100 (Very Strong)</span>
        </div>
    </div>
    """
    
    return html

# Legacy function for compatibility
calculate_market_wide_volume_analysis = calculate_complete_volume_analysis
