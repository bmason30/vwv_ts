"""
Williams VIX Fix (VWV) Core System Implementation
Professional-grade market timing with 6-component confluence analysis
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from utils.decorators import safe_calculation_wrapper
from config.settings import DEFAULT_VWV_CONFIG

logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def calculate_williams_vix_fix(data: pd.DataFrame, period: int = 22, multiplier: float = 2.0) -> pd.Series:
    """
    Calculate Williams VIX Fix (WVF) - Fear Gauge
    
    The Williams VIX Fix is designed to replicate the VIX on any instrument.
    It measures the relative position of the close to the highest high over a lookback period.
    """
    try:
        if len(data) < period + 1:
            return pd.Series([0] * len(data), index=data.index)
        
        close = data['Close']
        
        # Calculate highest high over the period
        highest_high = close.rolling(window=period).max()
        
        # Williams VIX Fix formula
        wvf = ((highest_high - close) / highest_high) * 100
        
        # Apply multiplier for sensitivity
        wvf_filtered = wvf * multiplier
        
        # Fill NaN values
        wvf_filtered = wvf_filtered.fillna(0)
        
        return wvf_filtered
        
    except Exception as e:
        logger.error(f"Williams VIX Fix calculation error: {e}")
        return pd.Series([0] * len(data), index=data.index)

@safe_calculation_wrapper
def calculate_ma_confluence(data: pd.DataFrame, periods: list = [20, 50, 200]) -> Dict[str, Any]:
    """
    Calculate Moving Average Confluence Analysis
    Determines if price is above/below key MAs and confluence strength
    """
    try:
        if len(data) < max(periods):
            return {'score': 50, 'confluence_count': 0, 'details': {}}
        
        close = data['Close']
        current_price = close.iloc[-1]
        confluence_details = {}
        above_count = 0
        
        for period in periods:
            if len(close) >= period:
                ma_value = close.rolling(period).mean().iloc[-1]
                is_above = current_price > ma_value
                
                confluence_details[f'MA_{period}'] = {
                    'value': round(float(ma_value), 2),
                    'above': is_above,
                    'distance_pct': round(((current_price - ma_value) / ma_value) * 100, 2)
                }
                
                if is_above:
                    above_count += 1
        
        # Calculate confluence score (0-100)
        confluence_score = (above_count / len(periods)) * 100
        
        return {
            'score': round(confluence_score, 1),
            'confluence_count': above_count,
            'total_mas': len(periods),
            'details': confluence_details
        }
        
    except Exception as e:
        logger.error(f"MA Confluence calculation error: {e}")
        return {'score': 50, 'confluence_count': 0, 'details': {}}

@safe_calculation_wrapper
def calculate_volume_confluence(data: pd.DataFrame, periods: list = [20, 50]) -> Dict[str, Any]:
    """
    Calculate Volume Confluence Analysis
    Analyzes volume relative to moving averages
    """
    try:
        if len(data) < max(periods):
            return {'score': 50, 'strength': 'Neutral', 'details': {}}
        
        volume = data['Volume']
        current_volume = volume.iloc[-1]
        volume_details = {}
        above_average_count = 0
        
        for period in periods:
            if len(volume) >= period:
                vol_ma = volume.rolling(period).mean().iloc[-1]
                ratio = current_volume / vol_ma if vol_ma > 0 else 1
                
                volume_details[f'Vol_MA_{period}'] = {
                    'ma_value': round(float(vol_ma), 0),
                    'ratio': round(float(ratio), 2),
                    'above_average': ratio > 1.0
                }
                
                if ratio > 1.0:
                    above_average_count += 1
        
        # Calculate volume strength score
        if above_average_count == len(periods):
            volume_score = 85  # Strong volume
            strength = "Strong"
        elif above_average_count > 0:
            volume_score = 65  # Moderate volume
            strength = "Moderate"
        else:
            volume_score = 35  # Weak volume
            strength = "Weak"
        
        return {
            'score': volume_score,
            'strength': strength,
            'current_volume': round(float(current_volume), 0),
            'details': volume_details
        }
        
    except Exception as e:
        logger.error(f"Volume Confluence calculation error: {e}")
        return {'score': 50, 'strength': 'Neutral', 'details': {}}

@safe_calculation_wrapper
def calculate_vwap_analysis_enhanced(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Enhanced VWAP Analysis for VWV System
    """
    try:
        if len(data) < 20:
            return {'score': 50, 'position': 'Neutral', 'vwap_value': 0}
        
        # Calculate VWAP
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        volume = data['Volume']
        
        # Use recent 20-day period for daily VWAP
        recent_data = data.tail(20)
        recent_tp = typical_price.tail(20)
        recent_vol = volume.tail(20)
        
        vwap = (recent_tp * recent_vol).sum() / recent_vol.sum()
        current_price = data['Close'].iloc[-1]
        
        # Calculate VWAP score
        if current_price > vwap * 1.02:  # 2% above VWAP
            vwap_score = 85
            position = "Strong Above"
        elif current_price > vwap:
            vwap_score = 65
            position = "Above"
        elif current_price < vwap * 0.98:  # 2% below VWAP
            vwap_score = 15
            position = "Strong Below"
        else:
            vwap_score = 35
            position = "Below"
        
        distance_pct = ((current_price - vwap) / vwap) * 100
        
        return {
            'score': vwap_score,
            'position': position,
            'vwap_value': round(float(vwap), 2),
            'distance_pct': round(float(distance_pct), 2)
        }
        
    except Exception as e:
        logger.error(f"Enhanced VWAP analysis error: {e}")
        return {'score': 50, 'position': 'Neutral', 'vwap_value': 0}

@safe_calculation_wrapper
def calculate_rsi_momentum_detection(data: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
    """
    RSI-like Momentum Detection for VWV System
    """
    try:
        if len(data) < period + 1:
            return {'score': 50, 'level': 'Neutral', 'rsi_value': 50}
        
        close = data['Close']
        delta = close.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # VWV-style RSI scoring (contrarian bias for bottom-picking)
        if current_rsi < 25:
            rsi_score = 90  # Very oversold - very bullish
            level = "Very Oversold"
        elif current_rsi < 35:
            rsi_score = 75  # Oversold - bullish
            level = "Oversold"
        elif current_rsi > 75:
            rsi_score = 10  # Very overbought - very bearish
            level = "Very Overbought"
        elif current_rsi > 65:
            rsi_score = 25  # Overbought - bearish
            level = "Overbought"
        else:
            # Neutral zone with slight contrarian bias
            rsi_score = 50 + (50 - current_rsi) * 0.2
            level = "Neutral"
        
        return {
            'score': round(float(rsi_score), 1),
            'level': level,
            'rsi_value': round(float(current_rsi), 2)
        }
        
    except Exception as e:
        logger.error(f"RSI momentum detection error: {e}")
        return {'score': 50, 'level': 'Neutral', 'rsi_value': 50}

@safe_calculation_wrapper
def calculate_volatility_filter(data: pd.DataFrame, period: int = 20) -> Dict[str, Any]:
    """
    Volatility Filter with Amplification for VWV System
    """
    try:
        if len(data) < period:
            return {'score': 50, 'regime': 'Normal', 'volatility': 20}
        
        close = data['Close']
        returns = close.pct_change().dropna()
        
        if len(returns) < period:
            return {'score': 50, 'regime': 'Normal', 'volatility': 20}
        
        # Calculate rolling volatility (annualized)
        volatility = returns.rolling(period).std().iloc[-1] * (252 ** 0.5) * 100
        
        # Volatility-based scoring for VWV system
        if volatility > 35:
            vol_score = 85  # High volatility increases signal strength
            regime = "High Volatility"
        elif volatility > 25:
            vol_score = 70  # Moderate-high volatility
            regime = "Elevated Volatility"
        elif volatility < 12:
            vol_score = 30  # Low volatility reduces signal strength
            regime = "Low Volatility"
        else:
            vol_score = 50  # Normal volatility
            regime = "Normal Volatility"
        
        return {
            'score': vol_score,
            'regime': regime,
            'volatility': round(float(volatility), 2)
        }
        
    except Exception as e:
        logger.error(f"Volatility filter calculation error: {e}")
        return {'score': 50, 'regime': 'Normal', 'volatility': 20}

@safe_calculation_wrapper
def calculate_vwv_composite_score(data: pd.DataFrame, config: Dict[str, Any] = None) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate VWV Composite Score with 6-Component Confluence System
    
    Components:
    1. Williams VIX Fix (WVF) - Fear gauge
    2. MA Confluence - Trend alignment
    3. Volume Confluence - Volume confirmation
    4. Enhanced VWAP - Price/volume relationship
    5. RSI Momentum - Oversold detection
    6. Volatility Filter - Market regime amplification
    """
    try:
        if config is None:
            config = DEFAULT_VWV_CONFIG
        
        # Calculate all 6 components
        wvf_data = calculate_williams_vix_fix(
            data, 
            config.get('wvf_period', 22), 
            config.get('wvf_multiplier', 2.0)
        )
        
        ma_confluence = calculate_ma_confluence(data, config.get('ma_periods', [20, 50, 200]))
        volume_confluence = calculate_volume_confluence(data, config.get('volume_periods', [20, 50]))
        vwap_analysis = calculate_vwap_analysis_enhanced(data)
        rsi_momentum = calculate_rsi_momentum_detection(data, config.get('rsi_period', 14))
        volatility_filter = calculate_volatility_filter(data, config.get('volatility_period', 20))
        
        # Get current WVF value
        current_wvf = wvf_data.iloc[-1] if len(wvf_data) > 0 else 0
        
        # Component weights from config
        weights = config.get('weights', {
            'wvf': 0.8, 
            'ma': 1.2, 
            'volume': 0.6,
            'vwap': 0.4, 
            'momentum': 0.5, 
            'volatility': 0.3
        })
        
        # Calculate weighted composite score
        component_scores = {
            'wvf_score': min(100, current_wvf * 10),  # Scale WVF to 0-100
            'ma_confluence_score': ma_confluence['score'],
            'volume_confluence_score': volume_confluence['score'],
            'vwap_score': vwap_analysis['score'],
            'rsi_momentum_score': rsi_momentum['score'],
            'volatility_score': volatility_filter['score']
        }
        
        # Apply weights and calculate composite
        weighted_score = (
            component_scores['wvf_score'] * weights['wvf'] +
            component_scores['ma_confluence_score'] * weights['ma'] +
            component_scores['volume_confluence_score'] * weights['volume'] +
            component_scores['vwap_score'] * weights['vwap'] +
            component_scores['rsi_momentum_score'] * weights['momentum'] +
            component_scores['volatility_score'] * weights['volatility']
        )
        
        # Normalize by total weights
        total_weight = sum(weights.values())
        normalized_score = weighted_score / total_weight
        
        # Apply scaling multiplier
        scaling_multiplier = config.get('scaling_multiplier', 1.5)
        final_score = min(100, normalized_score * scaling_multiplier)
        
        # Determine signal strength
        signal_thresholds = config.get('signal_thresholds', {
            'good': 3.5, 
            'strong': 4.5, 
            'very_strong': 5.5
        })
        
        # Convert to 0-10 scale for thresholds
        score_0_10 = final_score / 10
        
        if score_0_10 >= signal_thresholds['very_strong']:
            signal_strength = "VERY_STRONG"
            signal_color = "🔴"
        elif score_0_10 >= signal_thresholds['strong']:
            signal_strength = "STRONG"
            signal_color = "🟡"
        elif score_0_10 >= signal_thresholds['good']:
            signal_strength = "GOOD"
            signal_color = "🟢"
        else:
            signal_strength = "WEAK"
            signal_color = "⚪"
        
        # Build detailed results
        vwv_details = {
            'components': {
                'williams_vix_fix': {
                    'current_value': round(float(current_wvf), 2),
                    'score': round(component_scores['wvf_score'], 1),
                    'weight': weights['wvf']
                },
                'ma_confluence': {
                    'score': ma_confluence['score'],
                    'confluence_count': ma_confluence['confluence_count'],
                    'weight': weights['ma'],
                    'details': ma_confluence['details']
                },
                'volume_confluence': {
                    'score': volume_confluence['score'],
                    'strength': volume_confluence['strength'],
                    'weight': weights['volume'],
                    'details': volume_confluence['details']
                },
                'vwap_analysis': {
                    'score': vwap_analysis['score'],
                    'position': vwap_analysis['position'],
                    'vwap_value': vwap_analysis['vwap_value'],
                    'weight': weights['vwap']
                },
                'rsi_momentum': {
                    'score': rsi_momentum['score'],
                    'level': rsi_momentum['level'],
                    'rsi_value': rsi_momentum['rsi_value'],
                    'weight': weights['momentum']
                },
                'volatility_filter': {
                    'score': volatility_filter['score'],
                    'regime': volatility_filter['regime'],
                    'volatility': volatility_filter['volatility'],
                    'weight': weights['volatility']
                }
            },
            'signal_classification': {
                'strength': signal_strength,
                'color': signal_color,
                'score_0_10': round(score_0_10, 2),
                'thresholds': signal_thresholds
            },
            'risk_management': {
                'stop_loss_pct': config.get('stop_loss_pct', 0.022),
                'take_profit_pct': config.get('take_profit_pct', 0.055),
                'risk_reward_ratio': round(config.get('take_profit_pct', 0.055) / config.get('stop_loss_pct', 0.022), 1)
            }
        }
        
        return round(final_score, 1), vwv_details
        
    except Exception as e:
        logger.error(f"VWV composite score calculation error: {e}")
        return 50.0, {'error': str(e)}

def get_vwv_signal_interpretation(score: float, signal_strength: str) -> str:
    """Get interpretation of VWV signal"""
    interpretations = {
        "VERY_STRONG": "Extreme oversold condition with full confluence alignment. High probability setup for reversal/bounce.",
        "STRONG": "Strong oversold condition with good confluence. Above-average probability setup.",
        "GOOD": "Basic oversold condition with some confluence. Standard probability setup.",
        "WEAK": "Insufficient oversold condition or poor confluence. Low probability setup."
    }
    
    return interpretations.get(signal_strength, "Signal strength not classified")

@safe_calculation_wrapper
def calculate_vwv_system_complete(data: pd.DataFrame, symbol: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Complete VWV System Analysis
    Returns comprehensive VWV analysis results
    """
    try:
        # Calculate VWV composite score
        vwv_score, vwv_details = calculate_vwv_composite_score(data, config)
        
        # Get signal details
        signal_classification = vwv_details.get('signal_classification', {})
        signal_strength = signal_classification.get('strength', 'WEAK')
        signal_interpretation = get_vwv_signal_interpretation(vwv_score, signal_strength)
        
        # Current market data
        current_price = data['Close'].iloc[-1]
        current_date = data.index[-1].strftime('%Y-%m-%d')
        
        # Risk management levels
        risk_mgmt = vwv_details.get('risk_management', {})
        stop_loss_price = current_price * (1 - risk_mgmt.get('stop_loss_pct', 0.022))
        take_profit_price = current_price * (1 + risk_mgmt.get('take_profit_pct', 0.055))
        
        return {
            'symbol': symbol,
            'timestamp': current_date,
            'current_price': round(float(current_price), 2),
            'vwv_score': vwv_score,
            'signal_strength': signal_strength,
            'signal_color': signal_classification.get('color', '⚪'),
            'signal_interpretation': signal_interpretation,
            'components': vwv_details.get('components', {}),
            'risk_management': {
                'stop_loss_price': round(stop_loss_price, 2),
                'take_profit_price': round(take_profit_price, 2),
                'stop_loss_pct': risk_mgmt.get('stop_loss_pct', 0.022) * 100,
                'take_profit_pct': risk_mgmt.get('take_profit_pct', 0.055) * 100,
                'risk_reward_ratio': risk_mgmt.get('risk_reward_ratio', 2.5)
            },
            'system_status': 'OPERATIONAL'
        }
        
    except Exception as e:
        logger.error(f"Complete VWV system analysis error: {e}")
        return {
            'symbol': symbol,
            'error': str(e),
            'system_status': 'ERROR'
        }
