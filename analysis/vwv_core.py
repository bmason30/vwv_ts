"""
Williams VIX Fix Core System - The Heart of VWV Trading
32+ years of proven logic (1993-2025)
6-component confluence system with risk management
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

# Default VWV Core Configuration
DEFAULT_VWV_CONFIG = {
    'lookback_period': 22,
    'wvf_multiplier': 1.2,
    'ma_fast': 20,
    'ma_medium': 50,
    'ma_slow': 200,
    'volume_fast': 20,
    'volume_slow': 50,
    'vwap_deviation_multiplier': 1.5,
    'rsi_period': 14,
    'volatility_period': 20,
    'final_scaling_multiplier': 1.5,
    'component_weights': {
        'wvf': 1.5,
        'ma_confluence': 1.2,
        'volume_confluence': 0.8,
        'vwap_analysis': 0.6,
        'momentum': 0.7,
        'volatility_filter': 0.4
    }
}

@safe_calculation_wrapper
def calculate_williams_vix_fix(data: pd.DataFrame, config: Dict = None) -> pd.Series:
    """
    Calculate Williams VIX Fix - The core fear gauge
    Formula: ((Highest(Close, lookback) - Low) / Highest(Close, lookback)) * 100
    """
    if config is None:
        config = DEFAULT_VWV_CONFIG
    
    try:
        lookback = config['lookback_period']
        
        if len(data) < lookback + 5:
            return pd.Series([0] * len(data), index=data.index)
        
        # Calculate highest close over lookback period
        highest_close = data['Close'].rolling(window=lookback).max()
        
        # Williams VIX Fix formula
        wvf = ((highest_close - data['Low']) / highest_close) * 100
        wvf = wvf.fillna(0)
        
        return wvf
        
    except Exception as e:
        logger.error(f"Williams VIX Fix calculation error: {e}")
        return pd.Series([0] * len(data), index=data.index)

@safe_calculation_wrapper
def calculate_ma_confluence(data: pd.DataFrame, config: Dict = None) -> float:
    """Calculate Moving Average Confluence Score"""
    if config is None:
        config = DEFAULT_VWV_CONFIG
    
    try:
        current_price = data['Close'].iloc[-1]
        ma_fast = data['Close'].rolling(config['ma_fast']).mean().iloc[-1]
        ma_medium = data['Close'].rolling(config['ma_medium']).mean().iloc[-1]
        ma_slow = data['Close'].rolling(config['ma_slow']).mean().iloc[-1]
        
        confluence_score = 0
        
        # Price position relative to MAs
        if current_price > ma_fast:
            confluence_score += 1
        if current_price > ma_medium:
            confluence_score += 1
        if current_price > ma_slow:
            confluence_score += 1
            
        # MA alignment bonus
        if ma_fast > ma_medium > ma_slow:
            confluence_score += 1
        elif ma_fast < ma_medium < ma_slow:
            confluence_score -= 1
            
        return float(confluence_score)
        
    except Exception as e:
        logger.error(f"MA confluence calculation error: {e}")
        return 0.0

@safe_calculation_wrapper
def calculate_volume_confluence(data: pd.DataFrame, config: Dict = None) -> float:
    """Calculate Volume Confluence Score"""
    if config is None:
        config = DEFAULT_VWV_CONFIG
    
    try:
        current_volume = data['Volume'].iloc[-1]
        volume_fast = data['Volume'].rolling(config['volume_fast']).mean().iloc[-1]
        
        volume_score = 0
        
        # Volume analysis
        if current_volume > volume_fast * 1.5:
            volume_score += 2  # High volume
        elif current_volume > volume_fast:
            volume_score += 1  # Above average
        elif current_volume < volume_fast * 0.5:
            volume_score -= 1  # Low volume
            
        return float(volume_score)
        
    except Exception as e:
        logger.error(f"Volume confluence calculation error: {e}")
        return 0.0

@safe_calculation_wrapper
def calculate_vwap_analysis(data: pd.DataFrame, config: Dict = None) -> float:
    """Enhanced VWAP Analysis"""
    if config is None:
        config = DEFAULT_VWV_CONFIG
    
    try:
        # Calculate VWAP
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        current_vwap = vwap.iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        vwap_score = 0
        
        # Price vs VWAP
        if current_price > current_vwap * 1.01:
            vwap_score += 1
        elif current_price < current_vwap * 0.99:
            vwap_score -= 1
            
        return float(vwap_score)
        
    except Exception as e:
        logger.error(f"VWAP analysis calculation error: {e}")
        return 0.0

@safe_calculation_wrapper
def calculate_momentum_component(data: pd.DataFrame, config: Dict = None) -> float:
    """Calculate RSI-like Momentum Component"""
    if config is None:
        config = DEFAULT_VWV_CONFIG
    
    try:
        # RSI calculation
        close = data['Close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=config['rsi_period']).mean()
        
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        momentum_score = 0
        
        # RSI scoring (oversold favored)
        if current_rsi < 20:
            momentum_score += 3  # Extremely oversold
        elif current_rsi < 30:
            momentum_score += 2  # Oversold
        elif current_rsi < 40:
            momentum_score += 1  # Approaching oversold
        elif current_rsi > 80:
            momentum_score -= 2  # Overbought
            
        return float(momentum_score)
        
    except Exception as e:
        logger.error(f"Momentum calculation error: {e}")
        return 0.0

@safe_calculation_wrapper
def calculate_volatility_filter(data: pd.DataFrame, config: Dict = None) -> float:
    """Calculate Volatility Filter Component"""
    if config is None:
        config = DEFAULT_VWV_CONFIG
    
    try:
        # Realized volatility
        returns = data['Close'].pct_change().dropna()
        volatility = returns.rolling(config['volatility_period']).std() * np.sqrt(252) * 100
        current_vol = volatility.iloc[-1]
        
        volatility_score = 0
        
        # High volatility amplification
        if current_vol > 30:  # High volatility
            volatility_score += 2
        elif current_vol > 20:
            volatility_score += 1
            
        return float(volatility_score)
        
    except Exception as e:
        logger.error(f"Volatility filter calculation error: {e}")
        return 0.0

@safe_calculation_wrapper
def calculate_vwv_confluence_score(data: pd.DataFrame, config: Dict = None) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate complete VWV Confluence Score
    Main function combining all 6 components
    """
    if config is None:
        config = DEFAULT_VWV_CONFIG
    
    try:
        # Calculate Williams VIX Fix
        wvf_series = calculate_williams_vix_fix(data, config)
        current_wvf = wvf_series.iloc[-1]
        wvf_normalized = min(current_wvf / 20, 5.0)  # Normalize to 0-5
        
        # Calculate all components
        ma_confluence = calculate_ma_confluence(data, config)
        volume_confluence = calculate_volume_confluence(data, config)
        vwap_analysis = calculate_vwap_analysis(data, config)
        momentum_component = calculate_momentum_component(data, config)
        volatility_filter = calculate_volatility_filter(data, config)
        
        # Apply weights
        weights = config['component_weights']
        
        weighted_score = (
            wvf_normalized * weights['wvf'] +
            ma_confluence * weights['ma_confluence'] +
            volume_confluence * weights['volume_confluence'] +
            vwap_analysis * weights['vwap_analysis'] +
            momentum_component * weights['momentum'] +
            volatility_filter * weights['volatility_filter']
        )
        
        # Apply final scaling
        final_score = weighted_score * config['final_scaling_multiplier']
        
        # Classify signal
        signal_classification = classify_vwv_signal(final_score)
        
        # Component details
        component_details = {
            'wvf_raw': round(current_wvf, 2),
            'wvf_normalized': round(wvf_normalized, 2),
            'ma_confluence': round(ma_confluence, 2),
            'volume_confluence': round(volume_confluence, 2),
            'vwap_analysis': round(vwap_analysis, 2),
            'momentum_component': round(momentum_component, 2),
            'volatility_filter': round(volatility_filter, 2),
            'weighted_score': round(weighted_score, 2),
            'final_score': round(final_score, 2),
            'signal_classification': signal_classification,
            'component_weights': weights
        }
        
        return final_score, component_details
        
    except Exception as e:
        logger.error(f"VWV confluence score calculation error: {e}")
        return 0.0, {'error': str(e)}

def classify_vwv_signal(score: float) -> str:
    """Classify VWV signal strength"""
    if score >= 5.5:
        return "VERY_STRONG"
    elif score >= 4.5:
        return "STRONG"
    elif score >= 3.5:
        return "GOOD"
    else:
        return "WEAK"

@safe_calculation_wrapper
def calculate_vwv_risk_management(data: pd.DataFrame, signal_strength: str, current_price: float) -> Dict[str, Any]:
    """Calculate VWV Risk Management Levels"""
    try:
        # Calculate ATR for dynamic stops
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift(1)).abs()
        low_close = (data['Low'] - data['Close'].shift(1)).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_14 = true_range.rolling(14).mean().iloc[-1]
        
        # Risk multipliers by signal strength
        multipliers = {
            'VERY_STRONG': {'stop': 1.5, 'target1': 2.5, 'target2': 4.0},
            'STRONG': {'stop': 1.8, 'target1': 2.2, 'target2': 3.5},
            'GOOD': {'stop': 2.0, 'target1': 2.0, 'target2': 3.0},
            'WEAK': {'stop': 1.2, 'target1': 1.5, 'target2': 2.0}
        }
        
        multiplier = multipliers.get(signal_strength, multipliers['GOOD'])
        
        # Calculate levels
        stop_loss = current_price - (atr_14 * multiplier['stop'])
        target_1 = current_price + (atr_14 * multiplier['target1'])
        target_2 = current_price + (atr_14 * multiplier['target2'])
        
        # Risk-reward
        risk = current_price - stop_loss
        reward = target_1 - current_price
        rr_ratio = reward / risk if risk > 0 else 0
        
        return {
            'stop_loss': round(stop_loss, 2),
            'target_1': round(target_1, 2),
            'target_2': round(target_2, 2),
            'atr_14': round(atr_14, 2),
            'risk_reward_ratio': round(rr_ratio, 2),
            'signal_strength': signal_strength
        }
        
    except Exception as e:
        logger.error(f"Risk management calculation error: {e}")
        return {}
