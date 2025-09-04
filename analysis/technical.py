"""
Filename: analysis/technical.py
VWV Trading System v4.2.1
Created/Updated: 2025-09-04 12:11:17 EDT
Version: 7.0.0 - Integrated Momentum Divergence into composite score
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
    rs = gain / loss.replace(0, 1e-10) # Avoid division by zero
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
    if len(data) < 233: return {} # Ensure enough data for the largest EMA
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
    if len(data) < period * 2: return {'divergence_score': 50, 'signals': []}
    close = data['Close']
    rsi = safe_rsi(close, period)
    macd_hist = calculate_macd(close)['histogram']
    lookback = 20
    recent_close, recent_rsi = close.tail(lookback), rsi.tail(lookback)
    divergence_score, signals = 50, []
    price_trend = np.polyfit(range(lookback), recent_close, 1)[0]
    rsi_trend = np.polyfit(range(lookback), recent_rsi, 1)[0]
    if price_trend < 0 and rsi_trend > 0:
        signals.append('Bullish RSI Divergence'); divergence_score += 25
    if price_trend > 0 and rsi_trend < 0:
        signals.append('Bearish RSI Divergence'); divergence_score -= 25
    return {'divergence_score': np.clip(divergence_score, 0, 100), 'signals': signals, 'strength': 'Neutral'}

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
    try:
        technicals = analysis_results.get('enhanced_indicators', {}).get('comprehensive_technicals', {})
        if not technicals or 'error' in technicals: return 50.0, {}
        rsi_score = 100 - technicals.get('rsi_14', 50) if technicals.get('rsi_14', 50) > 70 else technicals.get('rsi_14', 50)
        mfi_score = 100 - technicals.get('mfi_14', 50) if technicals.get('mfi_14', 50) > 80 else technicals.get('mfi_14', 50)
        stoch_score = technicals.get('stochastic', {}).get('k', 50)
        williams_score = 100 + technicals.get('williams_r', -50)
        macd_score = 50 + (technicals.get('macd', {}).get('histogram', 0) * 100)
        bb_score = technicals.get('bollinger_bands', {}).get('position', 50)
        divergence_score = technicals.get('momentum_divergence', {}).get('divergence_score', 50)
        weighted_score = (
            (rsi_score * 0.10) + (mfi_score * 0.09) + (stoch_score * 0.08) + (williams_score * 0.08) +
            (macd_score * 0.15) + (bb_score * 0.10) +
            (min(technicals.get('volume_ratio', 1.0), 2.5) * 40 * 0.20) +
            ((100 - min(technicals.get('volatility_20d', 20), 100)) * 0.05) +
            (divergence_score * 0.10)
        )
        return np.clip(weighted_score, 0, 100), technicals
    except Exception as e:
        return 50.0, {'error': f'Score calculation failed: {e}'}

def calculate_weekly_deviations(data: pd.DataFrame) -> Dict[str, Any]: return {}
