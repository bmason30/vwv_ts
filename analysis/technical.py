"""
VWV Trading System - Enhanced Technical Analysis Module v5.0.0
Major functionality change: Complete composite scoring algorithm implementation

This file replaces the placeholder calculate_composite_technical_score function
with a comprehensive weighted scoring system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from utils.decorators import safe_calculation_wrapper

@safe_calculation_wrapper
def safe_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI with safe handling."""
    if len(close) < period + 1:
        return pd.Series([50] * len(close), index=close.index)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

@safe_calculation_wrapper
def calculate_daily_vwap(data: pd.DataFrame) -> float:
    """Calculate Volume Weighted Average Price for the current day."""
    if len(data) < 1 or data['Volume'].sum() == 0:
        return float(data['Close'].iloc[-1]) if not data.empty else None
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (typical_price * data['Volume']).sum() / data['Volume'].sum()
    return round(float(vwap), 2)

@safe_calculation_wrapper
def calculate_fibonacci_emas(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate Fibonacci sequence EMAs."""
    if len(data) < 21: return {}
    close = data['Close']
    fibonacci_periods = [21, 55, 89, 144, 233]
    emas = {}
    for period in fibonacci_periods:
        if len(close) >= period:
            ema = close.ewm(span=period, adjust=False).mean().iloc[-1]
            emas[f'ema_{period}'] = round(float(ema), 2)
    return emas

@safe_calculation_wrapper
def calculate_point_of_control_enhanced(data: pd.DataFrame) -> float:
    """Calculate enhanced Point of Control."""
    if len(data) < 1: return float(data['Close'].iloc[-1]) if not data.empty else None
    price_range = data['High'].max() - data['Low'].min()
    if price_range == 0: return float(data['Close'].iloc[-1])
    volume_by_price = data.groupby(pd.cut(data['Close'], bins=100))['Volume'].sum()
    if not volume_by_price.empty:
        return float(volume_by_price.idxmax().mid)
    return float(data['Close'].iloc[-1])

@safe_calculation_wrapper
def calculate_mfi(data: pd.DataFrame, period: int = 14) -> float:
    """Calculate Money Flow Index."""
    if len(data) < period + 1: return 50.0
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
    mfi_ratio = positive_flow / negative_flow.replace(0, np.inf)
    mfi = 100 - (100 / (1 + mfi_ratio))
    return float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 50.0

@safe_calculation_wrapper
def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
    """Calculate MACD."""
    if len(close) < slow: return {'macd': 0, 'signal': 0, 'histogram': 0}
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {'macd': round(float(macd_line.iloc[-1]), 4), 'signal': round(float(signal_line.iloc[-1]), 4), 'histogram': round(float(histogram.iloc[-1]), 4)}

@safe_calculation_wrapper
def calculate_atr(data: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range."""
    if len(data) < period + 1: return 0.0
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift(1)).abs()
    low_close = (data['Low'] - data['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()
    return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0

@safe_calculation_wrapper
def calculate_bollinger_bands(close: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, float]:
    """Calculate Bollinger Bands."""
    if len(close) < period:
        current_close = float(close.iloc[-1])
        return {'upper': current_close * 1.02, 'middle': current_close, 'lower': current_close * 0.98, 'position': 50}
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    current_close, upper_val, lower_val = close.iloc[-1], upper_band.iloc[-1], lower_band.iloc[-1]
    bb_position = ((current_close - lower_val) / (upper_val - lower_val)) * 100 if upper_val != lower_val else 50
    return {'upper': round(float(upper_val), 2), 'middle': round(float(sma.iloc[-1]), 2), 'lower': round(float(lower_val), 2), 'position': round(float(bb_position), 1)}

@safe_calculation_wrapper
def calculate_stochastic(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
    """Calculate Stochastic Oscillator."""
    if len(data) < k_period: return {'k': 50, 'd': 50}
    lowest_low = data['Low'].rolling(k_period).min()
    highest_high = data['High'].rolling(k_period).max()
    k_percent = ((data['Close'] - lowest_low) / (highest_high - lowest_low).replace(0, np.inf)) * 100
    d_percent = k_percent.rolling(d_period).mean()
    return {'k': round(float(k_percent.iloc[-1]), 2), 'd': round(float(d_percent.iloc[-1]), 2)}

@safe_calculation_wrapper
def calculate_williams_r(data: pd.DataFrame, period: int = 14) -> float:
    """Calculate Williams %R."""
    if len(data) < period: return -50.0
    highest_high = data['High'].rolling(period).max()
    lowest_low = data['Low'].rolling(period).min()
    williams_r = ((highest_high - data['Close']) / (highest_high - lowest_low).replace(0, np.inf)) * -100
    return float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else -50.0

@safe_calculation_wrapper
def calculate_weekly_deviations(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate weekly standard deviation levels."""
    if len(data) < 50: return {}
    weekly_data = data.resample('W-FRI').agg({'Close': 'last'}).dropna()
    if len(weekly_data) < 10: return {}
    recent_weekly = weekly_data['Close'].tail(20)
    mean_price, std_price = recent_weekly.mean(), recent_weekly.std()
    if pd.isna(std_price) or std_price == 0: return {}
    deviations = {'mean_price': round(float(mean_price), 2), 'std_price': round(float(std_price), 2)}
    for std_level in [1, 2, 3]:
        upper, lower = mean_price + (std_level * std_price), mean_price - (std_level * std_price)
        deviations[f'{std_level}_std'] = {'upper': round(float(upper), 2), 'lower': round(float(lower), 2)}
    return deviations

@safe_calculation_wrapper
def calculate_comprehensive_technicals(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate a comprehensive set of technical indicators."""
    if len(data) < 50: return {}
    close, volume = data['Close'], data['Volume']
    volume_sma_20 = volume.rolling(20).mean().iloc[-1]
    returns = close.pct_change().dropna()
    volatility_20d = returns.rolling(20).std().iloc[-1] * (252 ** 0.5) * 100 if len(returns) >= 20 else 0
    return {
        'rsi_14': round(float(safe_rsi(close, 14).iloc[-1]), 2),
        'mfi_14': round(float(calculate_mfi(data, 14)), 2),
        'macd': calculate_macd(close),
        'bollinger_bands': calculate_bollinger_bands(close),
        'stochastic': calculate_stochastic(data),
        'williams_r': calculate_williams_r(data),
        'volume_ratio': round(float(volume.iloc[-1] / volume_sma_20), 2) if volume_sma_20 > 0 else 1,
        'volatility_20d': round(float(volatility_20d), 2)
    }

def calculate_composite_technical_score(analysis_results: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate a comprehensive technical composite score from multiple indicators.
    
    MAJOR CHANGE v5.0.0: Replaced placeholder with full weighted scoring algorithm
    
    Score Range: 0-100
    - 0-20: Very Bearish
    - 21-40: Bearish  
    - 41-60: Neutral
    - 61-80: Bullish
    - 81-100: Very Bullish
    
    Component Weights:
    - Momentum Indicators: 40% (RSI 12%, MFI 10%, Stochastic 10%, Williams 8%)
    - Trend Indicators: 30% (MACD 15%, Bollinger 10%, EMA Trend 5%)
    - Volume Confirmation: 20% (Volume Ratio 20%)
    - Volatility Assessment: 10% (20-day Volatility 10%)
    
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

@safe_calculation_wrapper
def calculate_enhanced_technical_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate all technical, volume, and volatility analyses."""
    if len(data) < 50: return {'error': 'Insufficient data'}
    enhanced_indicators = {
        'daily_vwap': calculate_daily_vwap(data),
        'fibonacci_emas': calculate_fibonacci_emas(data),
        'point_of_control': calculate_point_of_control_enhanced(data),
        'weekly_deviations': calculate_weekly_deviations(data),
        'comprehensive_technicals': calculate_comprehensive_technicals(data),
    }
    # Safe imports for optional modules
    try:
        from .volume import calculate_complete_volume_analysis
        enhanced_indicators['volume_analysis'] = calculate_complete_volume_analysis(data)
    except ImportError:
        pass
    
    try:
        from .volatility import calculate_complete_volatility_analysis
        enhanced_indicators['volatility_analysis'] = calculate_complete_volatility_analysis(data)
    except ImportError:
        pass
    
    return enhanced_indicators

def generate_technical_signals(analysis_results: Dict[str, Any]) -> str:
    """Generates a discrete trading signal based on the composite score and other indicators."""
    if not analysis_results or 'enhanced_indicators' not in analysis_results: return 'HOLD'
    score, _ = calculate_composite_technical_score(analysis_results)
    technicals = analysis_results.get('enhanced_indicators', {}).get('comprehensive_technicals', {})
    macd_hist = technicals.get('macd', {}).get('histogram', 0)
    if score >= 80 and macd_hist > 0: return "STRONG_BUY"
    elif score >= 60 and macd_hist > 0: return "BUY"
    elif score <= 20 and macd_hist < 0: return "STRONG_SELL"
    elif score <= 40 and macd_hist < 0: return "SELL"
    else: return "HOLD"
