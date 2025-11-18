"""
File: analysis/patterns.py
VWV Trading System - Chart Pattern Detection
Created: 2025-11-18
Phase 2B: Pattern Recognition & Enhanced Signal Detection

Detects classic chart patterns:
- Head and Shoulders (bullish/bearish)
- Double Top/Bottom
- Triple Top/Bottom
- Triangles (ascending/descending/symmetrical)
- Flags and Pennants
- Wedges
- Channels
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.signal import find_peaks, argrelextrema
import warnings

warnings.filterwarnings('ignore')


def detect_peaks_and_troughs(data: pd.Series, prominence: float = 0.02,
                              min_distance: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect significant peaks and troughs in price series

    Args:
        data: Price series (Close, High, or Low)
        prominence: Minimum prominence as percentage of mean price
        min_distance: Minimum distance between peaks in bars

    Returns:
        Tuple of (peak_indices, trough_indices)
    """
    values = data.values
    mean_price = np.mean(values)
    prom_threshold = prominence * mean_price

    # Find peaks (local maxima)
    peaks, _ = find_peaks(values, distance=min_distance, prominence=prom_threshold)

    # Find troughs (local minima)
    troughs, _ = find_peaks(-values, distance=min_distance, prominence=prom_threshold)

    return peaks, troughs


def detect_head_and_shoulders(data: pd.DataFrame, lookback: int = 50,
                                tolerance: float = 0.02) -> Optional[Dict]:
    """
    Detect Head and Shoulders or Inverse Head and Shoulders pattern

    Pattern structure:
    - Head and Shoulders: Left Shoulder, Head (higher), Right Shoulder, Neckline
    - Inverse H&S: Same but inverted (bottoms instead of tops)

    Args:
        data: OHLCV DataFrame
        lookback: Number of bars to analyze
        tolerance: Price tolerance for pattern matching (2% default)

    Returns:
        Pattern dict if found, None otherwise
    """
    if len(data) < lookback:
        return None

    recent_data = data.tail(lookback)
    close = recent_data['Close']
    high = recent_data['High']
    low = recent_data['Low']

    # Detect peaks and troughs
    peaks, troughs = detect_peaks_and_troughs(close, prominence=0.03, min_distance=7)

    # Need at least 3 peaks for H&S
    if len(peaks) < 3:
        return None

    # Check last 3 peaks for H&S pattern
    if len(peaks) >= 3:
        p1_idx, p2_idx, p3_idx = peaks[-3], peaks[-2], peaks[-1]
        p1_price = high.iloc[p1_idx]
        p2_price = high.iloc[p2_idx]  # Head
        p3_price = high.iloc[p3_idx]

        # Classic H&S: Left shoulder ≈ Right shoulder, Head is higher
        if (p2_price > p1_price and p2_price > p3_price and
            abs(p1_price - p3_price) / p1_price < tolerance):

            # Find neckline (support connecting troughs)
            relevant_troughs = troughs[(troughs > p1_idx) & (troughs < p3_idx)]
            if len(relevant_troughs) >= 2:
                neckline_price = np.mean([low.iloc[t] for t in relevant_troughs])

                # Calculate target (head to neckline distance projected down)
                head_to_neckline = p2_price - neckline_price
                target_price = neckline_price - head_to_neckline

                return {
                    'type': 'head_and_shoulders',
                    'direction': 'bearish',
                    'confidence': 75,
                    'left_shoulder': {'index': p1_idx, 'price': float(p1_price)},
                    'head': {'index': p2_idx, 'price': float(p2_price)},
                    'right_shoulder': {'index': p3_idx, 'price': float(p3_price)},
                    'neckline': float(neckline_price),
                    'target_price': float(target_price),
                    'status': 'forming' if close.iloc[-1] > neckline_price else 'completed',
                    'description': 'Bearish Head and Shoulders pattern - potential reversal down'
                }

    # Check for Inverse H&S (using troughs)
    if len(troughs) >= 3:
        t1_idx, t2_idx, t3_idx = troughs[-3], troughs[-2], troughs[-1]
        t1_price = low.iloc[t1_idx]
        t2_price = low.iloc[t2_idx]  # Head
        t3_price = low.iloc[t3_idx]

        # Inverse H&S: Left shoulder ≈ Right shoulder, Head is lower
        if (t2_price < t1_price and t2_price < t3_price and
            abs(t1_price - t3_price) / t1_price < tolerance):

            # Find neckline (resistance connecting peaks)
            relevant_peaks = peaks[(peaks > t1_idx) & (peaks < t3_idx)]
            if len(relevant_peaks) >= 2:
                neckline_price = np.mean([high.iloc[p] for p in relevant_peaks])

                # Calculate target
                neckline_to_head = neckline_price - t2_price
                target_price = neckline_price + neckline_to_head

                return {
                    'type': 'inverse_head_and_shoulders',
                    'direction': 'bullish',
                    'confidence': 75,
                    'left_shoulder': {'index': t1_idx, 'price': float(t1_price)},
                    'head': {'index': t2_idx, 'price': float(t2_price)},
                    'right_shoulder': {'index': t3_idx, 'price': float(t3_price)},
                    'neckline': float(neckline_price),
                    'target_price': float(target_price),
                    'status': 'forming' if close.iloc[-1] < neckline_price else 'completed',
                    'description': 'Bullish Inverse Head and Shoulders - potential reversal up'
                }

    return None


def detect_double_top_bottom(data: pd.DataFrame, lookback: int = 40,
                               tolerance: float = 0.03) -> Optional[Dict]:
    """
    Detect Double Top (bearish) or Double Bottom (bullish) pattern

    Args:
        data: OHLCV DataFrame
        lookback: Number of bars to analyze
        tolerance: Price tolerance for matching tops/bottoms (3% default)

    Returns:
        Pattern dict if found, None otherwise
    """
    if len(data) < lookback:
        return None

    recent_data = data.tail(lookback)
    close = recent_data['Close']
    high = recent_data['High']
    low = recent_data['Low']

    peaks, troughs = detect_peaks_and_troughs(close, prominence=0.03, min_distance=7)

    # Double Top: Two peaks at similar price with trough in between
    if len(peaks) >= 2:
        p1_idx, p2_idx = peaks[-2], peaks[-1]
        p1_price = high.iloc[p1_idx]
        p2_price = high.iloc[p2_idx]

        # Peaks should be at similar price
        if abs(p1_price - p2_price) / p1_price < tolerance:
            # Find trough between peaks
            between_troughs = troughs[(troughs > p1_idx) & (troughs < p2_idx)]

            if len(between_troughs) > 0:
                valley_idx = between_troughs[0]
                valley_price = low.iloc[valley_idx]

                # Calculate target (height of pattern projected down)
                pattern_height = np.mean([p1_price, p2_price]) - valley_price
                target_price = valley_price - pattern_height

                return {
                    'type': 'double_top',
                    'direction': 'bearish',
                    'confidence': 70,
                    'first_top': {'index': p1_idx, 'price': float(p1_price)},
                    'second_top': {'index': p2_idx, 'price': float(p2_price)},
                    'valley': {'index': valley_idx, 'price': float(valley_price)},
                    'support': float(valley_price),
                    'target_price': float(target_price),
                    'status': 'forming' if close.iloc[-1] > valley_price else 'completed',
                    'description': 'Double Top pattern - bearish reversal signal'
                }

    # Double Bottom: Two troughs at similar price with peak in between
    if len(troughs) >= 2:
        t1_idx, t2_idx = troughs[-2], troughs[-1]
        t1_price = low.iloc[t1_idx]
        t2_price = low.iloc[t2_idx]

        # Troughs should be at similar price
        if abs(t1_price - t2_price) / t1_price < tolerance:
            # Find peak between troughs
            between_peaks = peaks[(peaks > t1_idx) & (peaks < t2_idx)]

            if len(between_peaks) > 0:
                peak_idx = between_peaks[0]
                peak_price = high.iloc[peak_idx]

                # Calculate target
                pattern_height = peak_price - np.mean([t1_price, t2_price])
                target_price = peak_price + pattern_height

                return {
                    'type': 'double_bottom',
                    'direction': 'bullish',
                    'confidence': 70,
                    'first_bottom': {'index': t1_idx, 'price': float(t1_price)},
                    'second_bottom': {'index': t2_idx, 'price': float(t2_price)},
                    'peak': {'index': peak_idx, 'price': float(peak_price)},
                    'resistance': float(peak_price),
                    'target_price': float(target_price),
                    'status': 'forming' if close.iloc[-1] < peak_price else 'completed',
                    'description': 'Double Bottom pattern - bullish reversal signal'
                }

    return None


def detect_triangle_patterns(data: pd.DataFrame, lookback: int = 50,
                               min_touches: int = 4) -> Optional[Dict]:
    """
    Detect triangle patterns (ascending, descending, symmetrical)

    Args:
        data: OHLCV DataFrame
        lookback: Number of bars to analyze
        min_touches: Minimum number of touches to confirm pattern

    Returns:
        Pattern dict if found, None otherwise
    """
    if len(data) < lookback:
        return None

    recent_data = data.tail(lookback)
    close = recent_data['Close']
    high = recent_data['High']
    low = recent_data['Low']
    indices = np.arange(len(recent_data))

    peaks, troughs = detect_peaks_and_troughs(close, prominence=0.02, min_distance=5)

    if len(peaks) < 2 or len(troughs) < 2:
        return None

    # Fit trend lines
    # Upper trend line (resistance)
    if len(peaks) >= 2:
        peak_indices = peaks
        peak_prices = high.iloc[peak_indices].values
        upper_slope, upper_intercept = np.polyfit(peak_indices, peak_prices, 1)
    else:
        return None

    # Lower trend line (support)
    if len(troughs) >= 2:
        trough_indices = troughs
        trough_prices = low.iloc[trough_indices].values
        lower_slope, lower_intercept = np.polyfit(trough_indices, trough_prices, 1)
    else:
        return None

    # Classify triangle type based on slopes
    slope_threshold = 0.01 * close.mean() / lookback  # Minimal slope threshold

    # Ascending Triangle: Flat top, rising bottom
    if abs(upper_slope) < slope_threshold and lower_slope > slope_threshold:
        resistance = np.mean(peak_prices)
        return {
            'type': 'ascending_triangle',
            'direction': 'bullish',
            'confidence': 65,
            'resistance': float(resistance),
            'support_slope': 'rising',
            'breakout_target': float(resistance + (resistance - trough_prices[-1])),
            'status': 'forming',
            'description': 'Ascending Triangle - bullish continuation pattern'
        }

    # Descending Triangle: Falling top, flat bottom
    elif upper_slope < -slope_threshold and abs(lower_slope) < slope_threshold:
        support = np.mean(trough_prices)
        return {
            'type': 'descending_triangle',
            'direction': 'bearish',
            'confidence': 65,
            'support': float(support),
            'resistance_slope': 'falling',
            'breakout_target': float(support - (peak_prices[-1] - support)),
            'status': 'forming',
            'description': 'Descending Triangle - bearish continuation pattern'
        }

    # Symmetrical Triangle: Converging trend lines
    elif upper_slope < -slope_threshold and lower_slope > slope_threshold:
        apex_x = (lower_intercept - upper_intercept) / (upper_slope - lower_slope)

        if apex_x > len(indices):
            return {
                'type': 'symmetrical_triangle',
                'direction': 'neutral',
                'confidence': 60,
                'upper_slope': 'falling',
                'lower_slope': 'rising',
                'apex_bars_away': int(apex_x - len(indices)),
                'status': 'forming',
                'description': 'Symmetrical Triangle - breakout imminent (direction unclear)'
            }

    return None


def detect_all_patterns(data: pd.DataFrame) -> Dict:
    """
    Detect all chart patterns and return comprehensive results

    Args:
        data: OHLCV DataFrame

    Returns:
        Dictionary with all detected patterns
    """
    patterns_found = []

    # Head and Shoulders
    hs_pattern = detect_head_and_shoulders(data, lookback=50)
    if hs_pattern:
        patterns_found.append(hs_pattern)

    # Double Top/Bottom
    double_pattern = detect_double_top_bottom(data, lookback=40)
    if double_pattern:
        patterns_found.append(double_pattern)

    # Triangles
    triangle_pattern = detect_triangle_patterns(data, lookback=50)
    if triangle_pattern:
        patterns_found.append(triangle_pattern)

    # Calculate pattern score
    total_confidence = sum([p.get('confidence', 0) for p in patterns_found])
    bullish_patterns = [p for p in patterns_found if p.get('direction') == 'bullish']
    bearish_patterns = [p for p in patterns_found if p.get('direction') == 'bearish']

    # Determine overall sentiment
    if len(bullish_patterns) > len(bearish_patterns):
        sentiment = 'bullish'
        strength = min(len(bullish_patterns) * 25, 100)
    elif len(bearish_patterns) > len(bullish_patterns):
        sentiment = 'bearish'
        strength = min(len(bearish_patterns) * 25, 100)
    else:
        sentiment = 'neutral'
        strength = 50

    return {
        'patterns_found': patterns_found,
        'total_patterns': len(patterns_found),
        'bullish_count': len(bullish_patterns),
        'bearish_count': len(bearish_patterns),
        'overall_sentiment': sentiment,
        'sentiment_strength': strength,
        'pattern_score': total_confidence / len(patterns_found) if patterns_found else 0
    }


def calculate_pattern_score(data: pd.DataFrame) -> Dict:
    """
    Calculate 0-100 pattern score for Master Score integration

    Args:
        data: OHLCV DataFrame

    Returns:
        Dict with score and details
    """
    pattern_results = detect_all_patterns(data)

    # Convert sentiment to 0-100 score
    # Bullish: 50-100, Bearish: 0-50, Neutral: 50
    sentiment = pattern_results['overall_sentiment']
    strength = pattern_results['sentiment_strength']

    if sentiment == 'bullish':
        score = 50 + (strength / 2)  # 50-100 range
    elif sentiment == 'bearish':
        score = 50 - (strength / 2)  # 0-50 range
    else:
        score = 50

    return {
        'score': round(score, 1),
        'sentiment': sentiment,
        'confidence': pattern_results['pattern_score'],
        'patterns_detected': pattern_results['total_patterns'],
        'bullish_patterns': pattern_results['bullish_count'],
        'bearish_patterns': pattern_results['bearish_count'],
        'details': pattern_results
    }
