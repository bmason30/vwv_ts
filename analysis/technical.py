"""
Technical analysis indicators and calculations - v8.0.0 ENHANCED
Added: ADX, CCI, OBV, Aroon, Parabolic SAR, CMF, Ultimate Oscillator
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from utils.decorators import safe_calculation_wrapper
from config.settings import FIBONACCI_EMA_PERIODS
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# EXISTING INDICATORS (Preserved)
# ============================================================================

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
    """Enhanced Point of Control using volume profiling."""
    if len(data) < 1:
        return float(data['Close'].iloc[-1]) if not data.empty else None
    price_range = data['High'].max() - data['Low'].min()
    if price_range == 0:
        return float(data['Close'].iloc[-1])
    volume_by_price = data.groupby(pd.cut(data['Close'], bins=100))['Volume'].sum()
    if not volume_by_price.empty:
        return float(volume_by_price.idxmax().mid)
    return float(data['Close'].iloc[-1])

@safe_calculation_wrapper
def calculate_mfi(data: pd.DataFrame, period: int = 14) -> float:
    """Calculate Money Flow Index."""
    if len(data) < period + 1:
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
            positive_flow.append(0)
            negative_flow.append(money_flow.iloc[i])
        else:
            positive_flow.append(0)
            negative_flow.append(0)
    
    positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
    negative_mf = pd.Series(negative_flow).rolling(window=period).sum()
    
    mfi = 100 - (100 / (1 + (positive_mf / negative_mf.replace(0, np.inf))))
    return float(mfi.iloc[-1]) if not mfi.empty else 50.0

# ============================================================================
# NEW INDICATORS
# ============================================================================

@safe_calculation_wrapper
def calculate_adx(data: pd.DataFrame, period: int = 14) -> Dict[str, float]:
    """
    Calculate Average Directional Index (ADX) for trend strength
    Returns ADX, +DI, -DI
    ADX > 25 indicates strong trend, < 20 indicates weak/no trend
    """
    if len(data) < period * 2:
        return {'adx': 25.0, 'plus_di': 25.0, 'minus_di': 25.0}
    
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Calculate Directional Movement
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    plus_dm = pd.Series(0.0, index=data.index)
    minus_dm = pd.Series(0.0, index=data.index)
    
    plus_dm[up_move > down_move] = up_move[up_move > down_move].clip(lower=0)
    minus_dm[down_move > up_move] = down_move[down_move > up_move].clip(lower=0)
    
    # Calculate Directional Indicators
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    # Calculate ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
    adx = dx.rolling(window=period).mean()
    
    return {
        'adx': round(float(adx.iloc[-1]), 2),
        'plus_di': round(float(plus_di.iloc[-1]), 2),
        'minus_di': round(float(minus_di.iloc[-1]), 2)
    }

@safe_calculation_wrapper
def calculate_cci(data: pd.DataFrame, period: int = 20) -> float:
    """
    Calculate Commodity Channel Index (CCI)
    CCI > 100: overbought, CCI < -100: oversold
    """
    if len(data) < period:
        return 0.0
    
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    sma = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean())
    
    cci = (typical_price - sma) / (0.015 * mean_deviation)
    return float(cci.iloc[-1])

@safe_calculation_wrapper
def calculate_obv(data: pd.DataFrame) -> float:
    """
    Calculate On-Balance Volume (OBV)
    Returns current OBV value
    """
    if len(data) < 2:
        return 0.0
    
    obv = pd.Series(0.0, index=data.index)
    obv.iloc[0] = data['Volume'].iloc[0]
    
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return float(obv.iloc[-1])

@safe_calculation_wrapper
def calculate_aroon(data: pd.DataFrame, period: int = 25) -> Dict[str, float]:
    """
    Calculate Aroon Up and Aroon Down
    Both range from 0-100
    Aroon Up > 70 and Aroon Down < 30: strong uptrend
    Aroon Down > 70 and Aroon Up < 30: strong downtrend
    """
    if len(data) < period:
        return {'aroon_up': 50.0, 'aroon_down': 50.0}
    
    aroon_up = []
    aroon_down = []
    
    for i in range(period, len(data)):
        window = data.iloc[i-period:i+1]
        
        # Days since highest high
        days_since_high = period - window['High'].values.argmax()
        aroon_up.append(((period - days_since_high) / period) * 100)
        
        # Days since lowest low
        days_since_low = period - window['Low'].values.argmin()
        aroon_down.append(((period - days_since_low) / period) * 100)
    
    return {
        'aroon_up': round(float(aroon_up[-1]), 2) if aroon_up else 50.0,
        'aroon_down': round(float(aroon_down[-1]), 2) if aroon_down else 50.0
    }

@safe_calculation_wrapper
def calculate_cmf(data: pd.DataFrame, period: int = 20) -> float:
    """
    Calculate Chaikin Money Flow (CMF)
    Ranges from -1 to +1
    > 0: buying pressure, < 0: selling pressure
    """
    if len(data) < period:
        return 0.0
    
    # Money Flow Multiplier
    mf_multiplier = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
    mf_multiplier = mf_multiplier.fillna(0)
    
    # Money Flow Volume
    mf_volume = mf_multiplier * data['Volume']
    
    # CMF
    cmf = mf_volume.rolling(window=period).sum() / data['Volume'].rolling(window=period).sum()
    return float(cmf.iloc[-1])

@safe_calculation_wrapper
def calculate_ultimate_oscillator(data: pd.DataFrame) -> float:
    """
    Calculate Ultimate Oscillator (7, 14, 28 periods)
    Ranges from 0-100
    > 70: overbought, < 30: oversold
    """
    if len(data) < 28:
        return 50.0
    
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    # True Range and Buying Pressure
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    
    bp = close - pd.concat([low, close.shift()], axis=1).min(axis=1)
    
    # Calculate averages for 3 periods
    avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
    avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
    avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
    
    # Ultimate Oscillator
    uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / (4 + 2 + 1)
    
    return float(uo.iloc[-1])

@safe_calculation_wrapper
def calculate_parabolic_sar(data: pd.DataFrame, af_start: float = 0.02, af_max: float = 0.2) -> Dict[str, Any]:
    """
    Calculate Parabolic SAR
    Returns current SAR value and trend direction
    """
    if len(data) < 5:
        return {'sar': float(data['Close'].iloc[-1]), 'trend': 'neutral'}
    
    high = data['High'].values
    low = data['Low'].values
    close = data['Close'].values
    
    # Initialize
    sar = np.zeros(len(data))
    trend = np.zeros(len(data))
    ep = 0
    af = af_start
    
    # Start with downtrend assumption
    sar[0] = high[0]
    trend[0] = -1
    ep = low[0]
    
    for i in range(1, len(data)):
        sar[i] = sar[i-1] + af * (ep - sar[i-1])
        
        if trend[i-1] == 1:  # Uptrend
            if low[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep
                ep = low[i]
                af = af_start
            else:
                trend[i] = 1
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_start, af_max)
        else:  # Downtrend
            if high[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                trend[i] = -1
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_start, af_max)
    
    current_trend = 'bullish' if trend[-1] == 1 else 'bearish'
    
    return {
        'sar': round(float(sar[-1]), 2),
        'trend': current_trend
    }

# ============================================================================
# COMPREHENSIVE TECHNICALS (Enhanced)
# ============================================================================

@safe_calculation_wrapper
def calculate_comprehensive_technicals(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate all technical indicators - ENHANCED VERSION
    """
    if len(data) < 20:
        return {}
    
    try:
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        # === EXISTING INDICATORS ===
        rsi_14 = safe_rsi(close, 14)
        mfi_14 = calculate_mfi(data, 14)
        
        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        
        # Stochastic
        low_14 = low.rolling(window=14).min()
        high_14 = high.rolling(window=14).max()
        stoch_k = 100 * ((close - low_14) / (high_14 - low_14))
        stoch_d = stoch_k.rolling(window=3).mean()
        
        # Williams %R
        williams_r = -100 * ((high_14 - close) / (high_14 - low_14))
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(window=14).mean()
        
        # Bollinger Bands
        bb_middle = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        bb_width = (bb_upper - bb_lower) / bb_middle
        
        # Volume
        volume_sma_20 = volume.rolling(window=20).mean()
        
        # Volatility
        returns = close.pct_change()
        volatility_20d = returns.rolling(window=20).std() * np.sqrt(252) * 100
        
        # === NEW INDICATORS ===
        adx_data = calculate_adx(data, 14)
        cci_value = calculate_cci(data, 20)
        obv_value = calculate_obv(data)
        aroon_data = calculate_aroon(data, 25)
        cmf_value = calculate_cmf(data, 20)
        uo_value = calculate_ultimate_oscillator(data)
        psar_data = calculate_parabolic_sar(data)
        
        return {
            # Momentum Oscillators
            'rsi_14': round(float(rsi_14.iloc[-1]), 2),
            'mfi_14': round(float(mfi_14), 2),
            'stochastic': {
                'k': round(float(stoch_k.iloc[-1]), 2),
                'd': round(float(stoch_d.iloc[-1]), 2)
            },
            'williams_r': round(float(williams_r.iloc[-1]), 2),
            'cci': round(float(cci_value), 2),
            'ultimate_oscillator': round(float(uo_value), 2),
            
            # Trend Indicators
            'macd': {
                'macd': round(float(macd_line.iloc[-1]), 4),
                'signal': round(float(signal_line.iloc[-1]), 4),
                'histogram': round(float(histogram.iloc[-1]), 4)
            },
            'adx': adx_data,
            'aroon': aroon_data,
            'parabolic_sar': psar_data,
            
            # Volatility
            'atr_14': round(float(atr_14.iloc[-1]), 2),
            'bollinger_bands': {
                'upper': round(float(bb_upper.iloc[-1]), 2),
                'middle': round(float(bb_middle.iloc[-1]), 2),
                'lower': round(float(bb_lower.iloc[-1]), 2),
                'width': round(float(bb_width.iloc[-1]), 4)
            },
            'volatility_20d': round(float(volatility_20d.iloc[-1]), 2),
            
            # Volume
            'current_volume': float(volume.iloc[-1]),
            'volume_sma_20': float(volume_sma_20.iloc[-1]),
            'obv': round(float(obv_value), 0),
            'cmf': round(float(cmf_value), 3),
        }
        
    except Exception as e:
        logger.error(f"Comprehensive technicals calculation error: {e}")
        return {}

@safe_calculation_wrapper
def calculate_weekly_deviations(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate weekly price deviations."""
    if len(data) < 5:
        return {}
    
    try:
        recent_5d = data.tail(5)
        week_high = recent_5d['High'].max()
        week_low = recent_5d['Low'].min()
        current_close = data['Close'].iloc[-1]
        
        return {
            'week_high': round(float(week_high), 2),
            'week_low': round(float(week_low), 2),
            'from_high_pct': round(((current_close - week_high) / week_high) * 100, 2),
            'from_low_pct': round(((current_close - week_low) / week_low) * 100, 2)
        }
    except Exception as e:
        logger.error(f"Weekly deviations error: {e}")
        return {}

@safe_calculation_wrapper
def calculate_composite_technical_score(analysis_results: Dict[str, Any]) -> Tuple[float, Dict]:
    """
    Calculate composite technical score (0-100) - ENHANCED
    Now includes new indicators in scoring
    """
    try:
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        technicals = enhanced_indicators.get('comprehensive_technicals', {})
        
        if not technicals:
            return 50.0, {}
        
        scores = []
        details = {}
        
        # RSI (Weight: 10%)
        rsi = technicals.get('rsi_14', 50)
        if rsi < 30:
            rsi_score = 100 - (rsi / 30 * 30)  # Oversold = bullish
        elif rsi > 70:
            rsi_score = (100 - rsi) / 30 * 30  # Overbought = bearish
        else:
            rsi_score = 50 + ((50 - abs(50 - rsi)) / 20 * 20)
        scores.append(rsi_score * 0.10)
        details['rsi'] = {'value': rsi, 'score': round(rsi_score, 2), 'weight': '10%'}
        
        # MFI (Weight: 8%)
        mfi = technicals.get('mfi_14', 50)
        if mfi < 20:
            mfi_score = 100 - (mfi / 20 * 30)
        elif mfi > 80:
            mfi_score = (100 - mfi) / 20 * 30
        else:
            mfi_score = 50 + ((50 - abs(50 - mfi)) / 30 * 20)
        scores.append(mfi_score * 0.08)
        details['mfi'] = {'value': mfi, 'score': round(mfi_score, 2), 'weight': '8%'}
        
        # MACD (Weight: 12%)
        macd = technicals.get('macd', {})
        histogram = macd.get('histogram', 0)
        macd_score = 50 + (histogram * 1000)
        macd_score = max(0, min(100, macd_score))
        scores.append(macd_score * 0.12)
        details['macd'] = {'histogram': histogram, 'score': round(macd_score, 2), 'weight': '12%'}
        
        # ADX with Directional Movement (Weight: 15%)
        adx_data = technicals.get('adx', {})
        adx = adx_data.get('adx', 25)
        plus_di = adx_data.get('plus_di', 25)
        minus_di = adx_data.get('minus_di', 25)
        
        if adx > 25:  # Strong trend
            if plus_di > minus_di:
                adx_score = 50 + (adx - 25) * 0.8
            else:
                adx_score = 50 - (adx - 25) * 0.8
        else:
            adx_score = 50
        adx_score = max(0, min(100, adx_score))
        scores.append(adx_score * 0.15)
        details['adx'] = {'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di, 
                         'score': round(adx_score, 2), 'weight': '15%'}
        
        # CCI (Weight: 8%)
        cci = technicals.get('cci', 0)
        if cci < -100:
            cci_score = 100 - (abs(cci) - 100) / 2
        elif cci > 100:
            cci_score = 100 - (cci - 100) / 2
        else:
            cci_score = 50 + (cci / 2)
        cci_score = max(0, min(100, cci_score))
        scores.append(cci_score * 0.08)
        details['cci'] = {'value': cci, 'score': round(cci_score, 2), 'weight': '8%'}
        
        # Aroon (Weight: 10%)
        aroon = technicals.get('aroon', {})
        aroon_up = aroon.get('aroon_up', 50)
        aroon_down = aroon.get('aroon_down', 50)
        aroon_score = aroon_up - aroon_down + 50
        aroon_score = max(0, min(100, aroon_score))
        scores.append(aroon_score * 0.10)
        details['aroon'] = {'up': aroon_up, 'down': aroon_down, 
                           'score': round(aroon_score, 2), 'weight': '10%'}
        
        # CMF (Weight: 8%)
        cmf = technicals.get('cmf', 0)
        cmf_score = 50 + (cmf * 50)
        cmf_score = max(0, min(100, cmf_score))
        scores.append(cmf_score * 0.08)
        details['cmf'] = {'value': cmf, 'score': round(cmf_score, 2), 'weight': '8%'}
        
        # Stochastic (Weight: 8%)
        stoch = technicals.get('stochastic', {})
        stoch_k = stoch.get('k', 50)
        if stoch_k < 20:
            stoch_score = 100 - (stoch_k / 20 * 30)
        elif stoch_k > 80:
            stoch_score = (100 - stoch_k) / 20 * 30
        else:
            stoch_score = 50 + ((50 - abs(50 - stoch_k)) / 30 * 20)
        scores.append(stoch_score * 0.08)
        details['stochastic'] = {'k': stoch_k, 'score': round(stoch_score, 2), 'weight': '8%'}
        
        # Ultimate Oscillator (Weight: 8%)
        uo = technicals.get('ultimate_oscillator', 50)
        if uo < 30:
            uo_score = 100 - (uo / 30 * 30)
        elif uo > 70:
            uo_score = (100 - uo) / 30 * 30
        else:
            uo_score = 50 + ((50 - abs(50 - uo)) / 20 * 20)
        scores.append(uo_score * 0.08)
        details['ultimate_oscillator'] = {'value': uo, 'score': round(uo_score, 2), 'weight': '8%'}
        
        # Parabolic SAR (Weight: 8%)
        psar = technicals.get('parabolic_sar', {})
        psar_trend = psar.get('trend', 'neutral')
        psar_score = 70 if psar_trend == 'bullish' else 30 if psar_trend == 'bearish' else 50
        scores.append(psar_score * 0.08)
        details['parabolic_sar'] = {'trend': psar_trend, 'score': psar_score, 'weight': '8%'}
        
        # Williams %R (Weight: 5%)
        williams = technicals.get('williams_r', -50)
        if williams < -80:
            williams_score = 100 + williams + 80
        elif williams > -20:
            williams_score = 100 - (williams + 20)
        else:
            williams_score = 50 + ((50 + williams) / 30 * 20)
        williams_score = max(0, min(100, williams_score))
        scores.append(williams_score * 0.05)
        details['williams_r'] = {'value': williams, 'score': round(williams_score, 2), 'weight': '5%'}
        
        # Calculate final composite score
        final_score = sum(scores)
        final_score = max(0, min(100, final_score))
        
        return round(final_score, 1), details
        
    except Exception as e:
        logger.error(f"Composite score calculation error: {e}")
        return 50.0, {}

def generate_technical_signals(analysis_results: Dict[str, Any]) -> str:
    """Generate trading signal based on composite score."""
    score, _ = calculate_composite_technical_score(analysis_results)
    
    if score >= 75:
        return "STRONG_BUY"
    elif score >= 60:
        return "BUY"
    elif score >= 55:
        return "WEAK_BUY"
    elif score >= 45:
        return "HOLD"
    elif score >= 40:
        return "WEAK_SELL"
    elif score >= 25:
        return "SELL"
    else:
        return "STRONG_SELL"
