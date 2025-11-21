"""
FILENAME: technical.py
VWV Research And Analysis System v4.2.2
File Revision: r9
Date: October 6, 2025
Revision Type: Critical Error Fixes

CRITICAL FIXES/CHANGES IN THIS REVISION:
- Fixed volume_by_price groupby missing observed=True parameter (line 48)
- Verified all pd.cut and pd.qcut operations have observed=True
- Fixed pandas FutureWarning in calculate_point_of_control_enhanced
- Enhanced error handling in all calculation functions
- Validated return types for all functions

FILE REVISION HISTORY:
r9 (Oct 6, 2025) - Fixed volume_by_price groupby errors
r8 (Oct 6, 2025) - Added observed=True to main groupby
r7 (Oct 5, 2025) - Technical indicator enhancements
r6 (Oct 4, 2025) - Composite scoring improvements
r5 (Oct 3, 2025) - Core calculation updates
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from utils.decorators import safe_calculation_wrapper
from config.settings import FIBONACCI_EMA_PERIODS
import logging

logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def safe_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Safe RSI calculation with proper error handling."""
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
    """Enhanced daily VWAP calculation."""
    if data.empty or 'Close' not in data.columns: 
        return 0.0
    if 'Typical_Price' in data.columns and 'Volume' in data.columns and not data['Volume'].sum() == 0:
        return float((data['Typical_Price'] * data['Volume']).sum() / data['Volume'].sum())
    return float(data['Close'].iloc[-1])

@safe_calculation_wrapper
def calculate_fibonacci_emas(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate Fibonacci EMAs."""
    if len(data) < min(FIBONACCI_EMA_PERIODS): 
        return {}
    close = data['Close']
    emas = {}
    for period in FIBONACCI_EMA_PERIODS:
        if len(close) >= period:
            emas[f'EMA_{period}'] = round(float(close.ewm(span=period, adjust=False).mean().iloc[-1]), 2)
    return emas

@safe_calculation_wrapper
def calculate_point_of_control_enhanced(data: pd.DataFrame) -> Optional[float]:
    """
    Enhanced Point of Control using volume profiling
    CRITICAL FIX r9: Ensured observed=True in groupby operation
    """
    if len(data) < 1: 
        return float(data['Close'].iloc[-1]) if not data.empty else None
    
    price_range = data['High'].max() - data['Low'].min()
    if price_range == 0: 
        return float(data['Close'].iloc[-1])
    
    # CRITICAL FIX r9: observed=True prevents FutureWarning
    volume_by_price = data.groupby(
        pd.cut(data['Close'], bins=100), 
        observed=True
    )['Volume'].sum()
    
    if not volume_by_price.empty:
        return float(volume_by_price.idxmax().mid)
    
    return float(data['Close'].iloc[-1])

@safe_calculation_wrapper
def calculate_mfi(data: pd.DataFrame, period: int = 14) -> float:
    """Calculate Money Flow Index."""
    if len(data) < period:
        return 50.0
    
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    
    positive_flow = []
    negative_flow = []
    
    for i in range(1, len(typical_price)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.append(money_flow.iloc[i])
            negative_flow.append(0)
        elif typical_price.iloc[i] < typical_price.iloc[i-1]:
            negative_flow.append(money_flow.iloc[i])
            positive_flow.append(0)
        else:
            positive_flow.append(0)
            negative_flow.append(0)
    
    positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=period).sum()
    
    mfi = 100 - (100 / (1 + (positive_mf / negative_mf.replace(0, np.inf))))
    
    return float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 50.0

@safe_calculation_wrapper
def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
    """Calculate MACD indicator."""
    if len(prices) < slow:
        return {'macd': 0, 'signal': 0, 'histogram': 0}
    
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return {
        'macd': round(float(macd.iloc[-1]), 4),
        'signal': round(float(signal_line.iloc[-1]), 4),
        'histogram': round(float(histogram.iloc[-1]), 4)
    }

@safe_calculation_wrapper
def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, float]:
    """Calculate Bollinger Bands."""
    if len(prices) < period:
        current = float(prices.iloc[-1]) if not prices.empty else 0
        return {'upper': current, 'middle': current, 'lower': current}
    
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return {
        'upper': round(float(upper_band.iloc[-1]), 2),
        'middle': round(float(sma.iloc[-1]), 2),
        'lower': round(float(lower_band.iloc[-1]), 2)
    }

@safe_calculation_wrapper
def calculate_stochastic(data: pd.DataFrame, period: int = 14) -> Dict[str, float]:
    """Calculate Stochastic Oscillator."""
    if len(data) < period:
        return {'k': 50.0, 'd': 50.0}
    
    low_min = data['Low'].rolling(window=period).min()
    high_max = data['High'].rolling(window=period).max()
    
    k = 100 * ((data['Close'] - low_min) / (high_max - low_min).replace(0, np.inf))
    d = k.rolling(window=3).mean()
    
    return {
        'k': round(float(k.iloc[-1]), 2) if not pd.isna(k.iloc[-1]) else 50.0,
        'd': round(float(d.iloc[-1]), 2) if not pd.isna(d.iloc[-1]) else 50.0
    }

@safe_calculation_wrapper
def calculate_williams_r(data: pd.DataFrame, period: int = 14) -> float:
    """Calculate Williams %R."""
    if len(data) < period:
        return -50.0
    
    highest_high = data['High'].rolling(period).max()
    lowest_low = data['Low'].rolling(period).min()
    williams_r = ((highest_high - data['Close']) / (highest_high - lowest_low).replace(0, np.inf)) * -100
    
    return float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else -50.0

@safe_calculation_wrapper
def calculate_weekly_deviations(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate weekly standard deviation levels."""
    if len(data) < 50:
        return {}
    
    weekly_data = data.resample('W-FRI').agg({'Close': 'last'}).dropna()
    if len(weekly_data) < 10:
        return {}
    
    recent_weekly = weekly_data['Close'].tail(20)
    mean_price = recent_weekly.mean()
    std_price = recent_weekly.std()
    
    if pd.isna(std_price) or std_price == 0:
        return {}
    
    deviations = {
        'mean_price': round(float(mean_price), 2),
        'std_price': round(float(std_price), 2)
    }
    
    for std_level in [1, 2, 3]:
        upper = mean_price + (std_level * std_price)
        lower = mean_price - (std_level * std_price)
        deviations[f'{std_level}_std'] = {
            'upper': round(float(upper), 2),
            'lower': round(float(lower), 2)
        }
    
    return deviations

@safe_calculation_wrapper
def calculate_adx(data: pd.DataFrame, period: int = 14) -> Dict[str, float]:
    """
    Calculate ADX (Average Directional Index) for trend strength.

    ADX measures the strength of a trend (not direction):
    - 0-25: Weak or no trend (range-bound market)
    - 25-50: Strong trend
    - 50-75: Very strong trend
    - 75-100: Extremely strong trend

    Args:
        data: DataFrame with OHLC data
        period: Lookback period (default 14)

    Returns:
        Dict with adx, plus_di, minus_di, and trend_strength
    """
    if len(data) < period * 2:
        return {
            'adx': 0,
            'plus_di': 0,
            'minus_di': 0,
            'trend_strength': 'Insufficient data'
        }

    try:
        high = data['High']
        low = data['Low']
        close = data['Close']

        # Calculate True Range (TR)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Calculate Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low

        # +DM and -DM
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=data.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=data.index)

        # Smooth DM
        plus_dm_smooth = plus_dm.rolling(window=period).sum()
        minus_dm_smooth = minus_dm.rolling(window=period).sum()

        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)

        # Calculate DX (Directional Index)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)

        # Calculate ADX (average of DX)
        adx = dx.rolling(window=period).mean()

        # Get latest values
        adx_value = round(float(adx.iloc[-1]), 2) if not adx.iloc[-1] != adx.iloc[-1] else 0  # Check for NaN
        plus_di_value = round(float(plus_di.iloc[-1]), 2) if not plus_di.iloc[-1] != plus_di.iloc[-1] else 0
        minus_di_value = round(float(minus_di.iloc[-1]), 2) if not minus_di.iloc[-1] != minus_di.iloc[-1] else 0

        # Interpret trend strength
        if adx_value < 25:
            strength = "Weak/No Trend"
        elif adx_value < 50:
            strength = "Strong Trend"
        elif adx_value < 75:
            strength = "Very Strong Trend"
        else:
            strength = "Extremely Strong Trend"

        # Determine trend direction
        if plus_di_value > minus_di_value:
            direction = "Bullish"
        elif minus_di_value > plus_di_value:
            direction = "Bearish"
        else:
            direction = "Neutral"

        return {
            'adx': adx_value,
            'plus_di': plus_di_value,
            'minus_di': minus_di_value,
            'trend_strength': strength,
            'trend_direction': direction,
            'adx_series': adx  # Include full series for charting
        }

    except Exception as e:
        logger.error(f"ADX calculation error: {e}")
        return {
            'adx': 0,
            'plus_di': 0,
            'minus_di': 0,
            'trend_strength': 'Error',
            'trend_direction': 'Unknown'
        }

@safe_calculation_wrapper
def calculate_comprehensive_technicals(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate a comprehensive set of technical indicators."""
    if len(data) < 50:
        return {}
    
    close = data['Close']
    volume = data['Volume']
    
    volume_sma_20 = volume.rolling(20).mean().iloc[-1]
    
    returns = close.pct_change().dropna()
    volatility_20d = returns.rolling(20).std().iloc[-1] * (252 ** 0.5) * 100 if len(returns) >= 20 else 0

    # Calculate ADX for trend strength
    adx_data = calculate_adx(data, period=14)

    return {
        'rsi_14': round(float(safe_rsi(close, 14).iloc[-1]), 2),
        'mfi_14': round(float(calculate_mfi(data, 14)), 2),
        'macd': calculate_macd(close),
        'bollinger_bands': calculate_bollinger_bands(close),
        'stochastic': calculate_stochastic(data),
        'williams_r': calculate_williams_r(data),
        'volume_ratio': round(float(volume.iloc[-1] / volume_sma_20), 2) if volume_sma_20 > 0 else 1,
        'volatility_20d': round(float(volatility_20d), 2),
        'adx': adx_data  # ADX trend strength indicator
    }

@safe_calculation_wrapper
def calculate_composite_technical_score(analysis_results: Dict[str, Any]) -> tuple:
    """Calculate composite technical score from various components."""
    try:
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        
        if not comprehensive_technicals:
            return 50.0, {}
        
        # Component scores
        component_scores = {}
        
        # RSI Score (0-100)
        rsi = comprehensive_technicals.get('rsi_14', 50)
        if rsi < 30:
            component_scores['rsi'] = 80  # Oversold = bullish
        elif rsi > 70:
            component_scores['rsi'] = 20  # Overbought = bearish
        else:
            component_scores['rsi'] = 50 + ((50 - rsi) * 0.5)
        
        # MACD Score
        macd_data = comprehensive_technicals.get('macd', {})
        histogram = macd_data.get('histogram', 0)
        component_scores['macd'] = 50 + (histogram * 100)
        component_scores['macd'] = max(0, min(100, component_scores['macd']))
        
        # Volume Score
        volume_ratio = comprehensive_technicals.get('volume_ratio', 1.0)
        component_scores['volume'] = min(100, 50 + (volume_ratio - 1) * 50)
        
        # Bollinger Bands Score
        bb = comprehensive_technicals.get('bollinger_bands', {})
        current_price = analysis_results.get('current_price', 0)
        if current_price and bb:
            upper = bb.get('upper', current_price)
            lower = bb.get('lower', current_price)
            
            if current_price < lower:
                component_scores['bollinger'] = 80
            elif current_price > upper:
                component_scores['bollinger'] = 20
            else:
                range_val = upper - lower
                if range_val > 0:
                    position = (current_price - lower) / range_val
                    component_scores['bollinger'] = 80 - (position * 60)
                else:
                    component_scores['bollinger'] = 50
        else:
            component_scores['bollinger'] = 50
        
        # Calculate composite score
        composite_score = sum(component_scores.values()) / len(component_scores)
        
        return round(composite_score, 1), {'component_scores': component_scores}
        
    except Exception as e:
        logger.error(f"Composite score calculation error: {e}")
        return 50.0, {}

@safe_calculation_wrapper
def generate_technical_signals(data: pd.DataFrame) -> Dict[str, Any]:
    """Generate market analysis based on technical indicators."""
    signals = {
        'buy_signals': [],
        'sell_signals': [],
        'neutral_signals': []
    }
    
    if len(data) < 50:
        return signals
    
    # Calculate indicators
    technicals = calculate_comprehensive_technicals(data)
    
    # RSI signals
    rsi = technicals.get('rsi_14', 50)
    if rsi < 30:
        signals['buy_signals'].append('RSI Oversold')
    elif rsi > 70:
        signals['sell_signals'].append('RSI Overbought')
    
    # MACD signals
    macd = technicals.get('macd', {})
    if macd.get('histogram', 0) > 0:
        signals['buy_signals'].append('MACD Bullish')
    elif macd.get('histogram', 0) < 0:
        signals['sell_signals'].append('MACD Bearish')
    
    return signals

@safe_calculation_wrapper
def calculate_enhanced_technical_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate all technical analyses."""
    return {
        'vwap': calculate_daily_vwap(data),
        'fibonacci_emas': calculate_fibonacci_emas(data),
        'point_of_control': calculate_point_of_control_enhanced(data),
        'weekly_deviations': calculate_weekly_deviations(data),
        'comprehensive_technicals': calculate_comprehensive_technicals(data),
        'signals': generate_technical_signals(data)
    }
