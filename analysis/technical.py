"""
Filename: analysis/technical.py
VWV Trading System v4.2.1
Created/Updated: 2025-09-04 13:50:18 EDT
Version: 7.0.1 - Restored correct composite score calculation logic
Purpose: Provides comprehensive technical analysis and a weighted composite score.
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
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

@safe_calculation_wrapper
def calculate_daily_vwap(data: pd.DataFrame) -> Optional[float]:
    """Calculate Volume Weighted Average Price for the current day."""
    if len(data) < 1 or 'Volume' not in data.columns or data['Volume'].sum() == 0:
        return float(data['Close'].iloc[-1]) if not data.empty else None
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (typical_price * data['Volume']).sum() / data['Volume'].sum()
    return round(float(vwap), 2)

@safe_calculation_wrapper
def calculate_fibonacci_emas(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate Fibonacci sequence EMAs."""
    if len(data) < 233: return {}
    close = data['Close']
    fib_periods = [21, 55, 89, 144, 233]
    emas = {f'ema_{p}': round(float(close.ewm(span=p, adjust=False).mean().iloc[-1]), 2) for p in fib_periods}
    return emas

@safe_calculation_wrapper
def calculate_point_of_control_enhanced(data: pd.DataFrame) -> Optional[float]:
    """Calculate enhanced Point of Control."""
    if len(data) < 2: return float(data['Close'].iloc[-1]) if not data.empty else None
    volume_by_price = data.groupby(pd.cut(data['Close'], bins=max(1, min(100, len(data)//2))))['Volume'].sum()
    if not volume_by_price.empty:
        return float(volume_by_price.idxmax().mid)
    return float(data['Close'].iloc[-1])

@safe_calculation_wrapper
def calculate_mfi(data: pd.DataFrame, period: int = 14) -> float:
    if len(data) < period + 1: return 50.0
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
    mfi_ratio = positive_flow / negative_flow.replace(0, 1e-10)
    mfi = 100 - (100 / (1 + mfi_ratio))
    return float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 50.0

@safe_calculation_wrapper
def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
    if len(close) < slow: return {'macd': 0, 'signal': 0, 'histogram': 0}
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {'macd': round(float(macd_line.iloc[-1]), 4), 'signal': round(float(signal_line.iloc[-1]), 4), 'histogram': round(float(histogram.iloc[-1]), 4)}

@safe_calculation_wrapper
def calculate_bollinger_bands(close: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, float]:
    if len(close) < period: return {'upper': 0, 'middle': 0, 'lower': 0, 'position': 50}
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper_band, lower_band = sma + (std * std_dev), sma - (std * std_dev)
    current_close, upper_val, lower_val = close.iloc[-1], upper_band.iloc[-1], lower_band.iloc[-1]
    bb_position = ((current_close - lower_val) / (upper_val - lower_val)) * 100 if (upper_val - lower_val) != 0 else 50
    return {'upper': round(float(upper_val), 2), 'middle': round(float(sma.iloc[-1]), 2), 'lower': round(float(lower_val), 2), 'position': round(float(bb_position), 1)}

@safe_calculation_wrapper
def calculate_stochastic(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
    if len(data) < k_period: return {'k': 50, 'd': 50}
    lowest_low, highest_high = data['Low'].rolling(k_period).min(), data['High'].rolling(k_period).max()
    k_percent = ((data['Close'] - lowest_low) / (highest_high - lowest_low).replace(0, 1e-10)) * 100
    d_percent = k_percent.rolling(d_period).mean()
    return {'k': round(float(k_percent.iloc[-1]), 2), 'd': round(float(d_percent.iloc[-1]), 2)}

@safe_calculation_wrapper
def calculate_williams_r(data: pd.DataFrame, period: int = 14) -> float:
    if len(data) < period: return -50.0
    highest_high, lowest_low = data['High'].rolling(period).max(), data['Low'].rolling(period).min()
    williams_r = ((highest_high - data['Close']) / (highest_high - lowest_low).replace(0, 1e-10)) * -100
    return float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else -50.0

@safe_calculation_wrapper
def calculate_momentum_divergence(data: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
    if len(data) < period * 2: return {'divergence_score': 50, 'signals': [], 'strength': 'Neutral'}
    close = data['Close']
    rsi = safe_rsi(close, period)
    macd_hist = calculate_macd(close)['histogram']
    lookback = 20
    recent_close, recent_rsi = close.tail(lookback), rsi.tail(lookback)
    divergence_score, signals, strength = 50, [], 'Neutral'
    price_trend = np.polyfit(range(lookback), recent_close, 1)[0]
    rsi_trend = np.polyfit(range(lookback), recent_rsi, 1)[0]
    if price_trend < 0 and rsi_trend > 0.1: signals.append('Bullish RSI Divergence'); divergence_score += 20
    if price_trend > 0 and rsi_trend < -0.1: signals.append('Bearish RSI Divergence'); divergence_score -= 20
    divergence_score = np.clip(divergence_score, 0, 100)
    if divergence_score >= 70: strength = 'Strong Bullish'
    elif divergence_score >= 60: strength = 'Bullish'
    elif divergence_score <= 30: strength = 'Strong Bearish'
    elif divergence_score <= 40: strength = 'Bearish'
    return {'divergence_score': round(divergence_score, 1), 'signals': signals, 'strength': strength}

@safe_calculation_wrapper
def calculate_comprehensive_technicals(data: pd.DataFrame) -> Dict[str, Any]:
    if len(data) < 50: return {'error': 'Insufficient data'}
    close, volume = data['Close'], data['Volume']
    volume_sma_20 = volume.rolling(20).mean().iloc[-1]
    returns = close.pct_change().dropna()
    volatility_20d = returns.rolling(20).std().iloc[-1] * (252 ** 0.5) * 100 if len(returns) >= 20 else 0
    return {
        'rsi_14': round(float(safe_rsi(close, 14).iloc[-1]), 2),
        'mfi_14': calculate_mfi(data, 14),
        'macd': calculate_macd(close),
        'bollinger_bands': calculate_bollinger_bands(close),
        'stochastic': calculate_stochastic(data),
        'williams_r': calculate_williams_r(data),
        'volume_ratio': round(float(volume.iloc[-1] / volume_sma_20), 2) if volume_sma_20 > 0 else 1,
        'volatility_20d': round(float(volatility_20d), 2),
        'momentum_divergence': calculate_momentum_divergence(data)
    }

def calculate_composite_technical_score(analysis_results: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """ CORRECTED: Calculate comprehensive technical composite score with momentum divergence """
    try:
        technicals = analysis_results.get('enhanced_indicators', {}).get('comprehensive_technicals', {})
        if not technicals or 'error' in technicals: return 50.0, {'error': 'Comprehensive technicals not available'}

        component_scores, weighted_sum, total_weight = {}, 0, 0
        
        # RSI
        rsi_14 = technicals.get('rsi_14', 50)
        if rsi_14 < 30: rsi_score = (rsi_14 / 30) * 50
        elif rsi_14 > 70: rsi_score = 100 - ((rsi_14 - 70) / 30) * 50
        else: rsi_score = 50 + ((rsi_14 - 50) / 20) * 20
        weighted_sum += rsi_score * 10; total_weight += 10
        
        # MFI
        mfi_14 = technicals.get('mfi_14', 50)
        if mfi_14 < 20: mfi_score = (mfi_14 / 20) * 50
        elif mfi_14 > 80: mfi_score = 100 - ((mfi_14 - 80) / 20) * 50
        else: mfi_score = 50 + ((mfi_14 - 50) / 30) * 20
        weighted_sum += mfi_score * 9; total_weight += 9
        
        # Stochastic
        stoch_k = technicals.get('stochastic', {}).get('k', 50)
        weighted_sum += stoch_k * 8; total_weight += 8
        
        # Williams %R
        williams_r = technicals.get('williams_r', -50)
        williams_score = 100 + williams_r
        weighted_sum += williams_score * 8; total_weight += 8
        
        # MACD
        macd_hist = technicals.get('macd', {}).get('histogram', 0)
        macd_score = np.clip(50 + (macd_hist * 100), 0, 100)
        weighted_sum += macd_score * 15; total_weight += 15
        
        # Bollinger Bands
        bb_position = technicals.get('bollinger_bands', {}).get('position', 50)
        weighted_sum += bb_position * 10; total_weight += 10
        
        # Volume
        volume_ratio = technicals.get('volume_ratio', 1.0)
        vol_score = np.clip(volume_ratio * 50, 0, 100)
        weighted_sum += vol_score * 20; total_weight += 20
        
        # Volatility
        volatility_20d = technicals.get('volatility_20d', 20)
        volatility_score = np.clip(100 - (volatility_20d * 1.5), 0, 100)
        weighted_sum += volatility_score * 5; total_weight += 5
        
        # Divergence
        divergence_score = technicals.get('momentum_divergence', {}).get('divergence_score', 50)
        weighted_sum += divergence_score * 10; total_weight += 10
        
        composite_score = weighted_sum / total_weight if total_weight > 0 else 50.0
        return round(composite_score, 1), technicals
    except Exception as e:
        return 50.0, {'error': f'Score calculation failed: {str(e)}'}

def calculate_weekly_deviations(data: pd.DataFrame) -> Dict[str, Any]: return {}
