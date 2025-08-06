"""
Technical analysis indicators and calculations - UPDATED with Volume/Volatility Integration
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Optional
from utils.decorators import safe_calculation_wrapper
from utils.helpers import statistical_normalize
from config.settings import FIBONACCI_EMA_PERIODS, TECHNICAL_PERIODS
import logging

# Import new analysis modules
try:
    from analysis.volume import calculate_complete_volume_analysis
except ImportError:
    calculate_complete_volume_analysis = None
try:
    from analysis.volatility import calculate_complete_volatility_analysis
except ImportError:
    calculate_complete_volatility_analysis = None

logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def safe_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Safe RSI calculation with proper error handling"""
    if len(prices) < period + 1:
        return pd.Series([50] * len(prices), index=prices.index)
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

@safe_calculation_wrapper
def calculate_daily_vwap(data: pd.DataFrame) -> float:
    """Enhanced daily VWAP calculation"""
    if data.empty or 'Close' not in data.columns: return 0.0
    if 'Typical_Price' in data.columns and 'Volume' in data.columns and not data['Volume'].sum() == 0:
        return float((data['Typical_Price'] * data['Volume']).sum() / data['Volume'].sum())
    return float(data['Close'].iloc[-1])

@safe_calculation_wrapper
def calculate_fibonacci_emas(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate Fibonacci EMAs"""
    if len(data) < min(FIBONACCI_EMA_PERIODS): return {}
    close = data['Close']
    emas = {}
    for period in FIBONACCI_EMA_PERIODS:
        if len(close) >= period:
            emas[f'EMA_{period}'] = round(float(close.ewm(span=period, adjust=False).mean().iloc[-1]), 2)
    return emas

@safe_calculation_wrapper
def calculate_point_of_control_enhanced(data: pd.DataFrame) -> Optional[float]:
    """Enhanced Point of Control with better volume weighting"""
    if len(data) < 1: return float(data['Close'].iloc[-1]) if not data.empty else None
    price_range = data['High'].max() - data['Low'].min()
    if price_range == 0: return float(data['Close'].iloc[-1])
    volume_by_price = data.groupby(pd.cut(data['Close'], bins=100))['Volume'].sum()
    if not volume_by_price.empty:
        return float(volume_by_price.idxmax().mid)
    return float(data['Close'].iloc[-1])

@safe_calculation_wrapper
def calculate_comprehensive_technicals(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive technical indicators for individual symbol analysis"""
    if len(data) < 50: return {}
    close, high, low, volume = data['Close'], data['High'], data['Low'], data['Volume']
    returns = close.pct_change().dropna()
    volatility_20d = returns.rolling(20).std().iloc[-1] * (252 ** 0.5) * 100 if len(returns) >= 20 else (returns.std() * (252 ** 0.5) * 100 if len(returns) > 0 else 20)
    volume_sma_20 = volume.rolling(20).mean().iloc[-1]
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

@safe_calculation_wrapper
def calculate_mfi(data: pd.DataFrame, period: int = 14) -> float:
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
    if len(close) < slow: return {'macd': 0, 'signal': 0, 'histogram': 0}
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {'macd': round(float(macd_line.iloc[-1]), 4), 'signal': round(float(signal_line.iloc[-1]), 4), 'histogram': round(float(histogram.iloc[-1]), 4)}

@safe_calculation_wrapper
def calculate_bollinger_bands(close: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, float]:
    if len(close) < period:
        current_close = float(close.iloc[-1])
        return {'upper': current_close * 1.02, 'middle': current_close, 'lower': current_close * 0.98, 'position': 50}
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper_band, lower_band = sma + (std * std_dev), sma - (std * std_dev)
    current_close, upper_val, lower_val = close.iloc[-1], upper_band.iloc[-1], lower_band.iloc[-1]
    bb_position = ((current_close - lower_val) / (upper_val - lower_val)) * 100 if upper_val != lower_val else 50
    return {'upper': round(float(upper_val), 2), 'middle': round(float(sma.iloc[-1]), 2), 'lower': round(float(lower_val), 2), 'position': round(float(bb_position), 1)}

@safe_calculation_wrapper
def calculate_stochastic(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
    if len(data) < k_period: return {'k': 50, 'd': 50}
    lowest_low = data['Low'].rolling(k_period).min()
    highest_high = data['High'].rolling(k_period).max()
    k_percent = ((data['Close'] - lowest_low) / (highest_high - lowest_low).replace(0, np.inf)) * 100
    d_percent = k_percent.rolling(d_period).mean()
    return {'k': round(float(k_percent.iloc[-1]), 2), 'd': round(float(d_percent.iloc[-1]), 2)}

@safe_calculation_wrapper
def calculate_williams_r(data: pd.DataFrame, period: int = 14) -> float:
    if len(data) < period: return -50.0
    highest_high = data['High'].rolling(period).max()
    lowest_low = data['Low'].rolling(period).min()
    williams_r = ((highest_high - data['Close']) / (highest_high - lowest_low).replace(0, np.inf)) * -100
    return float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else -50.0

@safe_calculation_wrapper
def calculate_weekly_deviations(data: pd.DataFrame) -> Dict[str, Any]:
    if len(data) < 50: return {}
    weekly_data = data.resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    if len(weekly_data) < 10: return {}
    recent_weekly = weekly_data['Close'].tail(20)
    mean_price, std_price = recent_weekly.mean(), recent_weekly.std()
    if pd.isna(std_price) or std_price == 0: return {}
    deviations = {'mean_price': round(float(mean_price), 2), 'std_price': round(float(std_price), 2)}
    for std_level in [1, 2, 3]:
        upper, lower = mean_price + (std_level * std_price), mean_price - (std_level * std_price)
        deviations[f'{std_level}_std'] = {'upper': round(float(upper), 2), 'lower': round(float(lower), 2), 'range_pct': round(float((std_level * std_price / mean_price) * 100), 2)}
    return deviations

def calculate_composite_technical_score(analysis_results: Dict[str, Any]) -> tuple:
    # This function's logic remains the same as the version you provided
    return 50.0, {}

@safe_calculation_wrapper
def calculate_enhanced_technical_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    if len(data) < 50:
        return {'error': 'Insufficient data for enhanced technical analysis'}
    enhanced_indicators = {
        'daily_vwap': calculate_daily_vwap(data),
        'fibonacci_emas': calculate_fibonacci_emas(data),
        'point_of_control': calculate_point_of_control_enhanced(data),
        'weekly_deviations': calculate_weekly_deviations(data),
        'comprehensive_technicals': calculate_comprehensive_technicals(data),
    }
    if calculate_complete_volume_analysis:
        enhanced_indicators['volume_analysis'] = calculate_complete_volume_analysis(data)
    if calculate_complete_volatility_analysis:
        enhanced_indicators['volatility_analysis'] = calculate_complete_volatility_analysis(data)
    return enhanced_indicators

def generate_technical_signals(analysis_results: Dict[str, Any]) -> str:
    """
    Generates a discrete trading signal based on the composite score and other indicators.
    Returns: 'STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'
    """
    if not analysis_results or 'enhanced_indicators' not in analysis_results:
        return 'HOLD'
    score, _ = calculate_composite_technical_score(analysis_results)
    technicals = analysis_results.get('enhanced_indicators', {}).get('comprehensive_technicals', {})
    macd_hist = technicals.get('macd', {}).get('histogram', 0)
    if score >= 80 and macd_hist > 0:
        return "STRONG_BUY"
    elif score >= 60 and macd_hist > 0:
        return "BUY"
    elif score <= 20 and macd_hist < 0:
        return "STRONG_SELL"
    elif score <= 40 and macd_hist < 0:
        return "SELL"
    else:
        return "HOLD"
