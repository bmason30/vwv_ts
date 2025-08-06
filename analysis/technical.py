"""
Technical analysis indicators and calculations - UPDATED with Volume/Volatility Integration
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Optional
from utils.decorators import safe_calculation_wrapper
from utils.helpers import statistical_normalize
from config.settings import FIBONACCI_EMA_PERIODS
import logging

# Safe imports for optional modules
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
    """Enhanced Point of Control using volume profiling"""
    if len(data) < 1: return float(data['Close'].iloc[-1]) if not data.empty else None
    price_range = data['High'].max() - data['Low'].min()
    if price_range == 0: return float(data['Close'].iloc[-1])
    # Create price bins for volume aggregation
    volume_by_price = data.groupby(pd.cut(data['Close'], bins=100))['Volume'].sum()
    if not volume_by_price.empty:
        # Find the price bin with the highest volume and return its midpoint
        return float(volume_by_price.idxmax().mid)
    return float(data['Close'].iloc[-1])

@safe_calculation_wrapper
def calculate_mfi(data: pd.DataFrame, period: int = 14) -> float:
    """Calculate Money Flow Index"""
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
    """Calculate MACD"""
    if len(close) < slow: return {'macd': 0, 'signal': 0, 'histogram': 0}
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {'macd': round(float(macd_line.iloc[-1]), 4), 'signal': round(float(signal_line.iloc[-1]), 4), 'histogram': round(float(histogram.iloc[-1]), 4)}

@safe_calculation_wrapper
def calculate_atr(data: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range"""
    if len(data) < period + 1: return 0.0
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift(1)).abs()
    low_close = (data['Low'] - data['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1/period, adjust=False).mean()
    return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0

@safe_calculation_wrapper
def calculate_bollinger_bands(close: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, float]:
    """Calculate Bollinger Bands
