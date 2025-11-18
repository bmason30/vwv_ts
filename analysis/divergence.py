"""
Momentum Divergence Detection Module
Phase 1a: Simplified slope-based detection
Phase 1b: Advanced peak matching with hidden divergence
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, argrelextrema
from config.settings import get_momentum_divergence_config


# Helper functions to calculate full oscillator series
def calculate_rsi_series(close, period=14):
    """Calculate RSI series for full dataframe."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_mfi_series(data, period=14):
    """Calculate Money Flow Index series."""
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']

    positive_flow = pd.Series(np.where(typical_price > typical_price.shift(1), money_flow, 0), index=data.index)
    negative_flow = pd.Series(np.where(typical_price < typical_price.shift(1), money_flow, 0), index=data.index)

    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()

    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    return mfi


def calculate_stochastic_series(data, period=14):
    """Calculate Stochastic %K series."""
    low_min = data['Low'].rolling(window=period).min()
    high_max = data['High'].rolling(window=period).max()

    stoch_k = 100 * (data['Close'] - low_min) / (high_max - low_min)
    return stoch_k


def calculate_williams_r_series(data, period=14):
    """Calculate Williams %R series."""
    high_max = data['High'].rolling(window=period).max()
    low_min = data['Low'].rolling(window=period).min()

    williams = -100 * (high_max - data['Close']) / (high_max - low_min)
    return williams


def calculate_divergence_score(data, technicals, use_advanced=True):
    """
    Calculate divergence score by detecting price/oscillator divergence.
    Phase 1b: Uses advanced peak matching by default.

    Args:
        data: DataFrame with OHLCV data
        technicals: Dict with technical indicators (not used in advanced mode)
        use_advanced: If True, use Phase 1b peak matching; if False, use Phase 1a slope method

    Returns:
        Dict with divergence score and detected divergences
    """
    config = get_momentum_divergence_config()

    if len(data) < config['lookback_period']:
        return {
            'score': 0,
            'divergences': [],
            'status': 'Insufficient data',
            'error': f"Need {config['lookback_period']} bars minimum"
        }

    try:
        lookback = config['lookback_period']
        recent_data = data.tail(lookback).copy()

        if use_advanced:
            # Phase 1b: Advanced peak matching
            divergences = detect_advanced_divergences(recent_data, config)
        else:
            # Phase 1a: Simple slope comparison (legacy)
            divergences = []
            oscillators_to_check = {
                'rsi': technicals.get('rsi_14'),
                'mfi': technicals.get('mfi_14'),
                'stochastic': technicals.get('stochastic_k'),
                'williams_r': technicals.get('williams_r')
            }

            for osc_name, osc_value in oscillators_to_check.items():
                if osc_value is None:
                    continue

                div_result = detect_simple_divergence(
                    recent_data['Close'],
                    osc_name,
                    osc_value,
                    config
                )

                if div_result:
                    divergences.append(div_result)

        # Calculate total score
        total_score = sum(d['score'] for d in divergences)

        # Interpret score
        status = interpret_divergence_score(total_score)

        return {
            'score': round(total_score, 1),
            'divergences': divergences,
            'status': status,
            'total_divergences': len(divergences),
            'bullish_count': sum(1 for d in divergences if d['type'] in ['bullish', 'hidden_bullish']),
            'bearish_count': sum(1 for d in divergences if d['type'] in ['bearish', 'hidden_bearish']),
            'regular_count': sum(1 for d in divergences if d['type'] in ['bullish', 'bearish']),
            'hidden_count': sum(1 for d in divergences if d['type'] in ['hidden_bullish', 'hidden_bearish'])
        }

    except Exception as e:
        return {
            'score': 0,
            'divergences': [],
            'status': 'Error',
            'error': str(e)
        }


def detect_simple_divergence(price_series, oscillator_name, oscillator_value, config):
    """
    Simplified divergence detection using slope comparison.

    Args:
        price_series: Series of closing prices
        oscillator_name: Name of oscillator ('rsi', 'mfi', etc.)
        oscillator_value: Current oscillator value
        config: Configuration dict

    Returns:
        Dict with divergence details or None if no divergence
    """
    try:
        # Get oscillator data from technical analysis if available
        # For now, use simplified slope detection
        lookback = min(10, len(price_series) - 1)

        if lookback < 5:
            return None

        # Calculate price slope (recent 10 bars)
        price_recent = price_series.iloc[-lookback:]
        price_slope = (price_recent.iloc[-1] - price_recent.iloc[0]) / lookback

        # Estimate oscillator slope based on thresholds
        # This is simplified - in reality we'd need full oscillator history
        osc_thresh = config['thresholds']

        # Bullish divergence: Price declining but oscillator rising (or oversold)
        # Bearish divergence: Price rising but oscillator declining (or overbought)

        if oscillator_name == 'rsi':
            oversold = osc_thresh['rsi_oversold']
            overbought = osc_thresh['rsi_overbought']

            if price_slope < 0 and oscillator_value < oversold + 10:
                # Potential bullish divergence
                return {
                    'type': 'bullish',
                    'oscillator': oscillator_name,
                    'score': config['score_weights']['bullish_divergence'],
                    'strength': 'moderate',
                    'description': f'Bullish divergence on {oscillator_name.upper()} ({oscillator_value:.1f})'
                }
            elif price_slope > 0 and oscillator_value > overbought - 10:
                # Potential bearish divergence
                return {
                    'type': 'bearish',
                    'oscillator': oscillator_name,
                    'score': config['score_weights']['bearish_divergence'],
                    'strength': 'moderate',
                    'description': f'Bearish divergence on {oscillator_name.upper()} ({oscillator_value:.1f})'
                }

        elif oscillator_name == 'mfi':
            oversold = osc_thresh['mfi_oversold']
            overbought = osc_thresh['mfi_overbought']

            if price_slope < 0 and oscillator_value < oversold + 10:
                return {
                    'type': 'bullish',
                    'oscillator': oscillator_name,
                    'score': config['score_weights']['bullish_divergence'],
                    'strength': 'moderate',
                    'description': f'Bullish divergence on {oscillator_name.upper()} ({oscillator_value:.1f})'
                }
            elif price_slope > 0 and oscillator_value > overbought - 10:
                return {
                    'type': 'bearish',
                    'oscillator': oscillator_name,
                    'score': config['score_weights']['bearish_divergence'],
                    'strength': 'moderate',
                    'description': f'Bearish divergence on {oscillator_name.upper()} ({oscillator_value:.1f})'
                }

        elif oscillator_name == 'stochastic':
            oversold = osc_thresh['stochastic_oversold']
            overbought = osc_thresh['stochastic_overbought']

            if price_slope < 0 and oscillator_value < oversold + 10:
                return {
                    'type': 'bullish',
                    'oscillator': oscillator_name,
                    'score': config['score_weights']['bullish_divergence'],
                    'strength': 'moderate',
                    'description': f'Bullish divergence on Stochastic ({oscillator_value:.1f})'
                }
            elif price_slope > 0 and oscillator_value > overbought - 10:
                return {
                    'type': 'bearish',
                    'oscillator': oscillator_name,
                    'score': config['score_weights']['bearish_divergence'],
                    'strength': 'moderate',
                    'description': f'Bearish divergence on Stochastic ({oscillator_value:.1f})'
                }

        elif oscillator_name == 'williams_r':
            oversold = osc_thresh['williams_oversold']
            overbought = osc_thresh['williams_overbought']

            if price_slope < 0 and oscillator_value < oversold + 10:
                return {
                    'type': 'bullish',
                    'oscillator': oscillator_name,
                    'score': config['score_weights']['bullish_divergence'],
                    'strength': 'moderate',
                    'description': f'Bullish divergence on Williams %R ({oscillator_value:.1f})'
                }
            elif price_slope > 0 and oscillator_value > overbought - 10:
                return {
                    'type': 'bearish',
                    'oscillator': oscillator_name,
                    'score': config['score_weights']['bearish_divergence'],
                    'strength': 'moderate',
                    'description': f'Bearish divergence on Williams %R ({oscillator_value:.1f})'
                }

        return None

    except Exception as e:
        return None


def detect_advanced_divergences(data, config):
    """
    Advanced divergence detection using peak/trough matching.
    Phase 1b implementation - scans all oscillators for divergences.

    Args:
        data: DataFrame with OHLCV data
        config: Configuration dict

    Returns:
        List of divergence dicts
    """
    divergences = []

    # Calculate full oscillator series
    rsi_series = calculate_rsi_series(data['Close'], period=14)
    mfi_series = calculate_mfi_series(data, period=14)
    stoch_series = calculate_stochastic_series(data, period=14)
    williams_series = calculate_williams_r_series(data, period=14)

    oscillator_data = {
        'rsi': rsi_series,
        'mfi': mfi_series,
        'stochastic': stoch_series,
        'williams_r': williams_series
    }

    price_series = data['Close']

    # Check each oscillator for divergence
    for osc_name, osc_series in oscillator_data.items():
        if osc_series is None or osc_series.isna().all():
            continue

        # Detect regular divergence
        regular_div = detect_peak_divergence(
            price_series,
            osc_series,
            osc_name,
            config,
            hidden=False
        )
        if regular_div:
            divergences.extend(regular_div)

        # Detect hidden divergence
        hidden_div = detect_peak_divergence(
            price_series,
            osc_series,
            osc_name,
            config,
            hidden=True
        )
        if hidden_div:
            divergences.extend(hidden_div)

    return divergences


def detect_peak_divergence(price_series, osc_series, osc_name, config, hidden=False):
    """
    Detect divergence by matching price and oscillator peaks/troughs.

    Args:
        price_series: Series of price data
        osc_series: Series of oscillator data
        osc_name: Name of oscillator
        config: Configuration dict
        hidden: If True, detect hidden divergence; if False, detect regular divergence

    Returns:
        List of divergence dicts or empty list
    """
    divergences = []

    # Remove NaN values
    valid_idx = ~(price_series.isna() | osc_series.isna())
    price_clean = price_series[valid_idx]
    osc_clean = osc_series[valid_idx]

    if len(price_clean) < config['min_swing_distance'] * 2:
        return []

    # Find peaks and troughs using scipy.signal.find_peaks
    prominence = config['peak_prominence']

    # Find price peaks (highs)
    price_peaks, _ = find_peaks(
        price_clean.values,
        distance=config['min_swing_distance'],
        prominence=prominence * price_clean.mean()
    )

    # Find price troughs (lows)
    price_troughs, _ = find_peaks(
        -price_clean.values,
        distance=config['min_swing_distance'],
        prominence=prominence * price_clean.mean()
    )

    # Find oscillator peaks
    osc_peaks, _ = find_peaks(
        osc_clean.values,
        distance=config['min_swing_distance']
    )

    # Find oscillator troughs
    osc_troughs, _ = find_peaks(
        -osc_clean.values,
        distance=config['min_swing_distance']
    )

    if not hidden:
        # Regular bullish divergence: Price makes lower low, oscillator makes higher low
        div = detect_bullish_divergence(
            price_clean, osc_clean,
            price_troughs, osc_troughs,
            osc_name, config
        )
        if div:
            divergences.append(div)

        # Regular bearish divergence: Price makes higher high, oscillator makes lower high
        div = detect_bearish_divergence(
            price_clean, osc_clean,
            price_peaks, osc_peaks,
            osc_name, config
        )
        if div:
            divergences.append(div)
    else:
        # Hidden bullish divergence: Price makes higher low, oscillator makes lower low
        div = detect_hidden_bullish_divergence(
            price_clean, osc_clean,
            price_troughs, osc_troughs,
            osc_name, config
        )
        if div:
            divergences.append(div)

        # Hidden bearish divergence: Price makes lower high, oscillator makes higher high
        div = detect_hidden_bearish_divergence(
            price_clean, osc_clean,
            price_peaks, osc_peaks,
            osc_name, config
        )
        if div:
            divergences.append(div)

    return divergences


def detect_bullish_divergence(price_series, osc_series, price_troughs, osc_troughs, osc_name, config):
    """Regular bullish divergence: Price LL, Oscillator HL"""
    if len(price_troughs) < 2 or len(osc_troughs) < 2:
        return None

    # Get last two troughs
    p_idx1, p_idx2 = price_troughs[-2], price_troughs[-1]
    price_vals = price_series.iloc[[p_idx1, p_idx2]].values

    # Find closest oscillator troughs to price troughs
    osc_idx1 = find_closest_peak(p_idx1, osc_troughs)
    osc_idx2 = find_closest_peak(p_idx2, osc_troughs)

    if osc_idx1 is None or osc_idx2 is None:
        return None

    osc_vals = osc_series.iloc[[osc_idx1, osc_idx2]].values

    # Check for bullish divergence: price makes lower low, oscillator makes higher low
    if price_vals[1] < price_vals[0] and osc_vals[1] > osc_vals[0]:
        strength = calculate_divergence_strength(price_vals, osc_vals, 'bullish')
        return {
            'type': 'bullish',
            'oscillator': osc_name,
            'score': config['score_weights']['bullish_divergence'],
            'strength': strength,
            'description': f'Regular bullish divergence on {osc_name.upper()}: Price LL, {osc_name.upper()} HL',
            'price_vals': price_vals.tolist(),
            'osc_vals': osc_vals.tolist()
        }

    return None


def detect_bearish_divergence(price_series, osc_series, price_peaks, osc_peaks, osc_name, config):
    """Regular bearish divergence: Price HH, Oscillator LH"""
    if len(price_peaks) < 2 or len(osc_peaks) < 2:
        return None

    # Get last two peaks
    p_idx1, p_idx2 = price_peaks[-2], price_peaks[-1]
    price_vals = price_series.iloc[[p_idx1, p_idx2]].values

    # Find closest oscillator peaks to price peaks
    osc_idx1 = find_closest_peak(p_idx1, osc_peaks)
    osc_idx2 = find_closest_peak(p_idx2, osc_peaks)

    if osc_idx1 is None or osc_idx2 is None:
        return None

    osc_vals = osc_series.iloc[[osc_idx1, osc_idx2]].values

    # Check for bearish divergence: price makes higher high, oscillator makes lower high
    if price_vals[1] > price_vals[0] and osc_vals[1] < osc_vals[0]:
        strength = calculate_divergence_strength(price_vals, osc_vals, 'bearish')
        return {
            'type': 'bearish',
            'oscillator': osc_name,
            'score': config['score_weights']['bearish_divergence'],
            'strength': strength,
            'description': f'Regular bearish divergence on {osc_name.upper()}: Price HH, {osc_name.upper()} LH',
            'price_vals': price_vals.tolist(),
            'osc_vals': osc_vals.tolist()
        }

    return None


def detect_hidden_bullish_divergence(price_series, osc_series, price_troughs, osc_troughs, osc_name, config):
    """Hidden bullish divergence: Price HL, Oscillator LL (continuation signal)"""
    if len(price_troughs) < 2 or len(osc_troughs) < 2:
        return None

    p_idx1, p_idx2 = price_troughs[-2], price_troughs[-1]
    price_vals = price_series.iloc[[p_idx1, p_idx2]].values

    osc_idx1 = find_closest_peak(p_idx1, osc_troughs)
    osc_idx2 = find_closest_peak(p_idx2, osc_troughs)

    if osc_idx1 is None or osc_idx2 is None:
        return None

    osc_vals = osc_series.iloc[[osc_idx1, osc_idx2]].values

    # Check for hidden bullish: price makes higher low, oscillator makes lower low
    if price_vals[1] > price_vals[0] and osc_vals[1] < osc_vals[0]:
        strength = calculate_divergence_strength(price_vals, osc_vals, 'hidden_bullish')
        return {
            'type': 'hidden_bullish',
            'oscillator': osc_name,
            'score': config['score_weights']['hidden_bullish'],
            'strength': strength,
            'description': f'Hidden bullish divergence on {osc_name.upper()}: Price HL, {osc_name.upper()} LL (trend continuation)',
            'price_vals': price_vals.tolist(),
            'osc_vals': osc_vals.tolist()
        }

    return None


def detect_hidden_bearish_divergence(price_series, osc_series, price_peaks, osc_peaks, osc_name, config):
    """Hidden bearish divergence: Price LH, Oscillator HH (continuation signal)"""
    if len(price_peaks) < 2 or len(osc_peaks) < 2:
        return None

    p_idx1, p_idx2 = price_peaks[-2], price_peaks[-1]
    price_vals = price_series.iloc[[p_idx1, p_idx2]].values

    osc_idx1 = find_closest_peak(p_idx1, osc_peaks)
    osc_idx2 = find_closest_peak(p_idx2, osc_peaks)

    if osc_idx1 is None or osc_idx2 is None:
        return None

    osc_vals = osc_series.iloc[[osc_idx1, osc_idx2]].values

    # Check for hidden bearish: price makes lower high, oscillator makes higher high
    if price_vals[1] < price_vals[0] and osc_vals[1] > osc_vals[0]:
        strength = calculate_divergence_strength(price_vals, osc_vals, 'hidden_bearish')
        return {
            'type': 'hidden_bearish',
            'oscillator': osc_name,
            'score': config['score_weights']['hidden_bearish'],
            'strength': strength,
            'description': f'Hidden bearish divergence on {osc_name.upper()}: Price LH, {osc_name.upper()} HH (trend continuation)',
            'price_vals': price_vals.tolist(),
            'osc_vals': osc_vals.tolist()
        }

    return None


def find_closest_peak(target_idx, peak_indices, max_distance=5):
    """Find the closest peak index to a target index."""
    if len(peak_indices) == 0:
        return None

    distances = np.abs(peak_indices - target_idx)
    closest_idx = np.argmin(distances)

    if distances[closest_idx] <= max_distance:
        return peak_indices[closest_idx]

    return None


def calculate_divergence_strength(price_vals, osc_vals, div_type):
    """Calculate divergence strength based on magnitude of difference."""
    price_change_pct = abs((price_vals[1] - price_vals[0]) / price_vals[0]) * 100
    osc_change_pct = abs((osc_vals[1] - osc_vals[0]) / max(abs(osc_vals[0]), 1)) * 100

    # Combined magnitude
    magnitude = (price_change_pct + osc_change_pct) / 2

    if magnitude > 10:
        return 'strong'
    elif magnitude > 5:
        return 'moderate'
    else:
        return 'weak'


def interpret_divergence_score(score):
    """
    Interpret divergence score and return status string.

    Args:
        score: Divergence score (can be negative)

    Returns:
        String interpretation
    """
    if score >= 20:
        return "Strong Bullish Divergence"
    elif score >= 10:
        return "Moderate Bullish Divergence"
    elif score > 0:
        return "Weak Bullish Divergence"
    elif score == 0:
        return "No Divergence Detected"
    elif score > -10:
        return "Weak Bearish Divergence"
    elif score > -20:
        return "Moderate Bearish Divergence"
    else:
        return "Strong Bearish Divergence"


def calculate_comprehensive_divergence(data, technicals):
    """
    Comprehensive divergence analysis including multiple timeframes.
    TO BE IMPLEMENTED in Phase 1b.

    Args:
        data: DataFrame with OHLCV data
        technicals: Dict with technical indicators

    Returns:
        Dict with comprehensive divergence analysis
    """
    # Placeholder for future enhancement
    # Will analyze divergence across multiple timeframes
    # and provide confluence scoring
    pass


# Utility functions for future enhancements

def find_price_peaks(price_series, prominence=0.02):
    """Find peaks in price data using scipy."""
    peaks, properties = find_peaks(
        price_series.values,
        prominence=prominence * price_series.mean()
    )
    return peaks, properties


def find_price_troughs(price_series, prominence=0.02):
    """Find troughs in price data using scipy."""
    troughs, properties = find_peaks(
        -price_series.values,
        prominence=prominence * price_series.mean()
    )
    return troughs, properties


def match_peaks_troughs(price_peaks, osc_peaks, max_distance=3):
    """
    Match price peaks with oscillator peaks.
    TO BE IMPLEMENTED in Phase 1b.
    """
    # Will implement peak matching algorithm
    pass
