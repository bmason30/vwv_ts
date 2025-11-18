"""
File: analysis/candlestick.py
VWV Trading System - Candlestick Pattern Recognition
Created: 2025-11-18
Phase 2B: Pattern Recognition & Enhanced Signal Detection

Detects classic candlestick patterns:
- Bullish: Hammer, Bullish Engulfing, Piercing Line, Morning Star, Three White Soldiers
- Bearish: Shooting Star, Bearish Engulfing, Dark Cloud Cover, Evening Star, Three Black Crows
- Reversal: Doji variations, Spinning Top, Harami
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


def calculate_body_size(open_price, close_price):
    """Calculate candle body size"""
    return abs(close_price - open_price)


def calculate_upper_shadow(open_price, close_price, high):
    """Calculate upper shadow (wick) size"""
    return high - max(open_price, close_price)


def calculate_lower_shadow(open_price, close_price, low):
    """Calculate lower shadow (tail) size"""
    return min(open_price, close_price) - low


def calculate_candle_range(high, low):
    """Calculate total candle range"""
    return high - low


def is_bullish_candle(open_price, close_price):
    """Check if candle is bullish (close > open)"""
    return close_price > open_price


def is_bearish_candle(open_price, close_price):
    """Check if candle is bearish (close < open)"""
    return close_price < open_price


def detect_hammer(row: pd.Series, prev_trend: str = 'down') -> Optional[Dict]:
    """
    Detect Hammer pattern (bullish reversal)

    Characteristics:
    - Small body at top of candle
    - Long lower shadow (2-3x body size)
    - Little to no upper shadow
    - Appears after downtrend

    Args:
        row: Single candle data (O, H, L, C)
        prev_trend: Previous trend direction

    Returns:
        Pattern dict if detected, None otherwise
    """
    o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']

    body = calculate_body_size(o, c)
    lower_shadow = calculate_lower_shadow(o, c, l)
    upper_shadow = calculate_upper_shadow(o, c, h)
    total_range = calculate_candle_range(h, l)

    # Hammer criteria
    if (lower_shadow >= 2 * body and  # Long lower shadow
        upper_shadow <= body * 0.5 and  # Small upper shadow
        body >= total_range * 0.1 and  # Body not too small
        prev_trend == 'down'):  # After downtrend

        strength = 'strong' if lower_shadow >= 3 * body else 'moderate'

        return {
            'name': 'hammer',
            'type': 'reversal',
            'direction': 'bullish',
            'strength': strength,
            'reliability': 65,  # Historical win rate %
            'description': 'Hammer - bullish reversal after downtrend',
            'confirmation': 'Wait for next candle to close above hammer high'
        }

    return None


def detect_shooting_star(row: pd.Series, prev_trend: str = 'up') -> Optional[Dict]:
    """
    Detect Shooting Star pattern (bearish reversal)

    Characteristics:
    - Small body at bottom of candle
    - Long upper shadow (2-3x body size)
    - Little to no lower shadow
    - Appears after uptrend

    Args:
        row: Single candle data
        prev_trend: Previous trend direction

    Returns:
        Pattern dict if detected
    """
    o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']

    body = calculate_body_size(o, c)
    lower_shadow = calculate_lower_shadow(o, c, l)
    upper_shadow = calculate_upper_shadow(o, c, h)
    total_range = calculate_candle_range(h, l)

    # Shooting Star criteria
    if (upper_shadow >= 2 * body and
        lower_shadow <= body * 0.5 and
        body >= total_range * 0.1 and
        prev_trend == 'up'):

        strength = 'strong' if upper_shadow >= 3 * body else 'moderate'

        return {
            'name': 'shooting_star',
            'type': 'reversal',
            'direction': 'bearish',
            'strength': strength,
            'reliability': 63,
            'description': 'Shooting Star - bearish reversal after uptrend',
            'confirmation': 'Wait for next candle to close below shooting star low'
        }

    return None


def detect_engulfing(data: pd.DataFrame, index: int) -> Optional[Dict]:
    """
    Detect Bullish or Bearish Engulfing pattern

    Bullish Engulfing:
    - Previous candle is bearish (red)
    - Current candle is bullish (green) and larger
    - Current body completely engulfs previous body

    Bearish Engulfing:
    - Previous candle is bullish (green)
    - Current candle is bearish (red) and larger
    - Current body completely engulfs previous body

    Args:
        data: OHLCV DataFrame
        index: Current candle index

    Returns:
        Pattern dict if detected
    """
    if index < 1:
        return None

    prev = data.iloc[index - 1]
    curr = data.iloc[index]

    prev_body = calculate_body_size(prev['Open'], prev['Close'])
    curr_body = calculate_body_size(curr['Open'], curr['Close'])

    # Bullish Engulfing
    if (is_bearish_candle(prev['Open'], prev['Close']) and
        is_bullish_candle(curr['Open'], curr['Close']) and
        curr['Open'] < prev['Close'] and
        curr['Close'] > prev['Open'] and
        curr_body > prev_body):

        return {
            'name': 'bullish_engulfing',
            'type': 'reversal',
            'direction': 'bullish',
            'strength': 'strong',
            'reliability': 70,
            'description': 'Bullish Engulfing - strong bullish reversal',
            'confirmation': 'Pattern complete - high reliability'
        }

    # Bearish Engulfing
    if (is_bullish_candle(prev['Open'], prev['Close']) and
        is_bearish_candle(curr['Open'], curr['Close']) and
        curr['Open'] > prev['Close'] and
        curr['Close'] < prev['Open'] and
        curr_body > prev_body):

        return {
            'name': 'bearish_engulfing',
            'type': 'reversal',
            'direction': 'bearish',
            'strength': 'strong',
            'reliability': 68,
            'description': 'Bearish Engulfing - strong bearish reversal',
            'confirmation': 'Pattern complete - high reliability'
        }

    return None


def detect_doji(row: pd.Series) -> Optional[Dict]:
    """
    Detect Doji patterns (indecision)

    Types:
    - Standard Doji: Open ≈ Close, small body
    - Dragonfly Doji: Long lower shadow, no upper shadow (bullish)
    - Gravestone Doji: Long upper shadow, no lower shadow (bearish)
    - Long-legged Doji: Long shadows on both sides (high volatility)

    Args:
        row: Single candle data

    Returns:
        Pattern dict if detected
    """
    o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']

    body = calculate_body_size(o, c)
    lower_shadow = calculate_lower_shadow(o, c, l)
    upper_shadow = calculate_upper_shadow(o, c, h)
    total_range = calculate_candle_range(h, l)

    # Doji: Very small body relative to total range
    if body <= total_range * 0.1:

        # Dragonfly Doji: Long lower shadow, almost no upper shadow
        if lower_shadow >= total_range * 0.6 and upper_shadow <= total_range * 0.1:
            return {
                'name': 'dragonfly_doji',
                'type': 'reversal',
                'direction': 'bullish',
                'strength': 'moderate',
                'reliability': 62,
                'description': 'Dragonfly Doji - potential bullish reversal',
                'confirmation': 'More reliable at support levels'
            }

        # Gravestone Doji: Long upper shadow, almost no lower shadow
        if upper_shadow >= total_range * 0.6 and lower_shadow <= total_range * 0.1:
            return {
                'name': 'gravestone_doji',
                'type': 'reversal',
                'direction': 'bearish',
                'strength': 'moderate',
                'reliability': 60,
                'description': 'Gravestone Doji - potential bearish reversal',
                'confirmation': 'More reliable at resistance levels'
            }

        # Long-legged Doji: Both shadows are long
        if (lower_shadow >= total_range * 0.3 and upper_shadow >= total_range * 0.3):
            return {
                'name': 'long_legged_doji',
                'type': 'indecision',
                'direction': 'neutral',
                'strength': 'moderate',
                'reliability': 55,
                'description': 'Long-legged Doji - high indecision and volatility',
                'confirmation': 'Wait for direction confirmation'
            }

        # Standard Doji
        return {
            'name': 'doji',
            'type': 'indecision',
            'direction': 'neutral',
            'strength': 'weak',
            'reliability': 50,
            'description': 'Doji - market indecision',
            'confirmation': 'Wait for next candle direction'
        }

    return None


def detect_morning_star(data: pd.DataFrame, index: int) -> Optional[Dict]:
    """
    Detect Morning Star pattern (bullish reversal)

    Three-candle pattern:
    1. Large bearish candle
    2. Small body candle (gap down) - star
    3. Large bullish candle closing above midpoint of first candle

    Args:
        data: OHLCV DataFrame
        index: Current candle index

    Returns:
        Pattern dict if detected
    """
    if index < 2:
        return None

    c1 = data.iloc[index - 2]  # First candle
    c2 = data.iloc[index - 1]  # Star
    c3 = data.iloc[index]       # Third candle

    c1_body = calculate_body_size(c1['Open'], c1['Close'])
    c2_body = calculate_body_size(c2['Open'], c2['Close'])
    c3_body = calculate_body_size(c3['Open'], c3['Close'])

    # Morning Star criteria
    if (is_bearish_candle(c1['Open'], c1['Close']) and  # First candle bearish
        c1_body > c2_body * 2 and  # Star is small
        is_bullish_candle(c3['Open'], c3['Close']) and  # Third candle bullish
        c3['Close'] > (c1['Open'] + c1['Close']) / 2):  # Closes above midpoint

        return {
            'name': 'morning_star',
            'type': 'reversal',
            'direction': 'bullish',
            'strength': 'strong',
            'reliability': 75,
            'description': 'Morning Star - strong bullish reversal',
            'confirmation': 'Three-candle pattern complete'
        }

    return None


def detect_evening_star(data: pd.DataFrame, index: int) -> Optional[Dict]:
    """
    Detect Evening Star pattern (bearish reversal)

    Three-candle pattern:
    1. Large bullish candle
    2. Small body candle (gap up) - star
    3. Large bearish candle closing below midpoint of first candle

    Args:
        data: OHLCV DataFrame
        index: Current candle index

    Returns:
        Pattern dict if detected
    """
    if index < 2:
        return None

    c1 = data.iloc[index - 2]
    c2 = data.iloc[index - 1]
    c3 = data.iloc[index]

    c1_body = calculate_body_size(c1['Open'], c1['Close'])
    c2_body = calculate_body_size(c2['Open'], c2['Close'])
    c3_body = calculate_body_size(c3['Open'], c3['Close'])

    # Evening Star criteria
    if (is_bullish_candle(c1['Open'], c1['Close']) and
        c1_body > c2_body * 2 and
        is_bearish_candle(c3['Open'], c3['Close']) and
        c3['Close'] < (c1['Open'] + c1['Close']) / 2):

        return {
            'name': 'evening_star',
            'type': 'reversal',
            'direction': 'bearish',
            'strength': 'strong',
            'reliability': 73,
            'description': 'Evening Star - strong bearish reversal',
            'confirmation': 'Three-candle pattern complete'
        }

    return None


def detect_three_white_soldiers(data: pd.DataFrame, index: int) -> Optional[Dict]:
    """
    Detect Three White Soldiers (bullish continuation)

    Three consecutive bullish candles, each closing higher than previous

    Args:
        data: OHLCV DataFrame
        index: Current candle index

    Returns:
        Pattern dict if detected
    """
    if index < 2:
        return None

    c1 = data.iloc[index - 2]
    c2 = data.iloc[index - 1]
    c3 = data.iloc[index]

    # All three must be bullish with higher closes
    if (is_bullish_candle(c1['Open'], c1['Close']) and
        is_bullish_candle(c2['Open'], c2['Close']) and
        is_bullish_candle(c3['Open'], c3['Close']) and
        c2['Close'] > c1['Close'] and
        c3['Close'] > c2['Close']):

        # Each candle should open within previous body
        if (c2['Open'] > c1['Open'] and c2['Open'] < c1['Close'] and
            c3['Open'] > c2['Open'] and c3['Open'] < c2['Close']):

            return {
                'name': 'three_white_soldiers',
                'type': 'continuation',
                'direction': 'bullish',
                'strength': 'strong',
                'reliability': 72,
                'description': 'Three White Soldiers - strong bullish continuation',
                'confirmation': 'Pattern complete - uptrend likely continues'
            }

    return None


def detect_three_black_crows(data: pd.DataFrame, index: int) -> Optional[Dict]:
    """
    Detect Three Black Crows (bearish continuation)

    Three consecutive bearish candles, each closing lower than previous

    Args:
        data: OHLCV DataFrame
        index: Current candle index

    Returns:
        Pattern dict if detected
    """
    if index < 2:
        return None

    c1 = data.iloc[index - 2]
    c2 = data.iloc[index - 1]
    c3 = data.iloc[index]

    # All three must be bearish with lower closes
    if (is_bearish_candle(c1['Open'], c1['Close']) and
        is_bearish_candle(c2['Open'], c2['Close']) and
        is_bearish_candle(c3['Open'], c3['Close']) and
        c2['Close'] < c1['Close'] and
        c3['Close'] < c2['Close']):

        # Each candle should open within previous body
        if (c2['Open'] < c1['Open'] and c2['Open'] > c1['Close'] and
            c3['Open'] < c2['Open'] and c3['Open'] > c2['Close']):

            return {
                'name': 'three_black_crows',
                'type': 'continuation',
                'direction': 'bearish',
                'strength': 'strong',
                'reliability': 70,
                'description': 'Three Black Crows - strong bearish continuation',
                'confirmation': 'Pattern complete - downtrend likely continues'
            }

    return None


def determine_trend(data: pd.DataFrame, lookback: int = 10) -> str:
    """
    Determine current trend direction for pattern context

    Args:
        data: OHLCV DataFrame
        lookback: Number of bars to analyze

    Returns:
        'up', 'down', or 'sideways'
    """
    if len(data) < lookback:
        return 'sideways'

    recent = data.tail(lookback)
    close_prices = recent['Close'].values

    # Simple trend: compare recent price to lookback price
    first_price = close_prices[0]
    last_price = close_prices[-1]
    price_change_pct = ((last_price - first_price) / first_price) * 100

    if price_change_pct > 3:
        return 'up'
    elif price_change_pct < -3:
        return 'down'
    else:
        return 'sideways'


def scan_all_candlestick_patterns(data: pd.DataFrame, lookback: int = 5) -> Dict:
    """
    Scan for all candlestick patterns in recent candles

    Args:
        data: OHLCV DataFrame
        lookback: Number of recent candles to analyze

    Returns:
        Dictionary with all detected patterns
    """
    if len(data) < lookback:
        return {
            'patterns_found': [],
            'total_patterns': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'overall_sentiment': 'neutral',
            'confidence': 0
        }

    patterns_found = []
    recent_data = data.tail(lookback)

    # Determine prevailing trend
    prev_trend = determine_trend(data, lookback=20)

    # Scan each candle
    for i in range(len(recent_data)):
        global_index = len(data) - len(recent_data) + i
        row = recent_data.iloc[i]

        # Single-candle patterns
        hammer = detect_hammer(row, prev_trend)
        if hammer:
            patterns_found.append({**hammer, 'date': row.name})

        shooting_star = detect_shooting_star(row, prev_trend)
        if shooting_star:
            patterns_found.append({**shooting_star, 'date': row.name})

        doji = detect_doji(row)
        if doji:
            patterns_found.append({**doji, 'date': row.name})

        # Multi-candle patterns (need enough history)
        if i >= 1:
            engulfing = detect_engulfing(recent_data, i)
            if engulfing:
                patterns_found.append({**engulfing, 'date': row.name})

        if i >= 2:
            morning_star = detect_morning_star(recent_data, i)
            if morning_star:
                patterns_found.append({**morning_star, 'date': row.name})

            evening_star = detect_evening_star(recent_data, i)
            if evening_star:
                patterns_found.append({**evening_star, 'date': row.name})

            three_white = detect_three_white_soldiers(recent_data, i)
            if three_white:
                patterns_found.append({**three_white, 'date': row.name})

            three_black = detect_three_black_crows(recent_data, i)
            if three_black:
                patterns_found.append({**three_black, 'date': row.name})

    # Calculate summary
    bullish_patterns = [p for p in patterns_found if p.get('direction') == 'bullish']
    bearish_patterns = [p for p in patterns_found if p.get('direction') == 'bearish']

    if len(bullish_patterns) > len(bearish_patterns):
        sentiment = 'bullish'
    elif len(bearish_patterns) > len(bullish_patterns):
        sentiment = 'bearish'
    else:
        sentiment = 'neutral'

    # Average reliability as confidence
    avg_reliability = np.mean([p.get('reliability', 50) for p in patterns_found]) if patterns_found else 0

    return {
        'patterns_found': patterns_found,
        'total_patterns': len(patterns_found),
        'bullish_count': len(bullish_patterns),
        'bearish_count': len(bearish_patterns),
        'overall_sentiment': sentiment,
        'confidence': round(avg_reliability, 1)
    }


def calculate_candlestick_score(data: pd.DataFrame) -> Dict:
    """
    Calculate 0-100 candlestick score for Master Score integration

    Args:
        data: OHLCV DataFrame

    Returns:
        Dict with score and details
    """
    results = scan_all_candlestick_patterns(data, lookback=5)

    # Convert sentiment to 0-100 score
    sentiment = results['overall_sentiment']
    bullish_count = results['bullish_count']
    bearish_count = results['bearish_count']

    if sentiment == 'bullish':
        score = 50 + min(bullish_count * 10, 50)  # 50-100 range
    elif sentiment == 'bearish':
        score = 50 - min(bearish_count * 10, 50)  # 0-50 range
    else:
        score = 50

    return {
        'score': round(score, 1),
        'sentiment': sentiment,
        'confidence': results['confidence'],
        'patterns_detected': results['total_patterns'],
        'bullish_patterns': results['bullish_count'],
        'bearish_patterns': results['bearish_count'],
        'details': results
    }
