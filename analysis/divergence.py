"""
Momentum Divergence Detection Module
Simplified implementation for Phase 1a - detects price/oscillator divergence
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, argrelextrema
from config.settings import get_momentum_divergence_config


def calculate_divergence_score(data, technicals):
    """
    Calculate divergence score by detecting price/oscillator divergence.

    Args:
        data: DataFrame with OHLCV data
        technicals: Dict with technical indicators (RSI, MFI, Stochastic, Williams %R)

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
        # Prepare data
        lookback = config['lookback_period']
        recent_data = data.tail(lookback).copy()

        # Calculate slopes for price and oscillators
        divergences = []
        total_score = 0

        # Check each oscillator for divergence
        oscillators_to_check = {
            'rsi': technicals.get('rsi_14'),
            'mfi': technicals.get('mfi_14'),
            'stochastic': technicals.get('stochastic_k'),
            'williams_r': technicals.get('williams_r')
        }

        for osc_name, osc_value in oscillators_to_check.items():
            if osc_value is None:
                continue

            # Detect divergence using simplified slope comparison
            div_result = detect_simple_divergence(
                recent_data['Close'],
                osc_name,
                osc_value,
                config
            )

            if div_result:
                divergences.append(div_result)
                total_score += div_result['score']

        # Interpret score
        status = interpret_divergence_score(total_score)

        return {
            'score': round(total_score, 1),
            'divergences': divergences,
            'status': status,
            'total_divergences': len(divergences),
            'bullish_count': sum(1 for d in divergences if d['type'] in ['bullish', 'hidden_bullish']),
            'bearish_count': sum(1 for d in divergences if d['type'] in ['bearish', 'hidden_bearish'])
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


def detect_advanced_divergence(price_data, oscillator_data, config):
    """
    Advanced divergence detection using peak/trough matching.
    TO BE IMPLEMENTED in Phase 1b - uses scipy.signal.find_peaks

    Args:
        price_data: Series of price data
        oscillator_data: Series of oscillator data
        config: Configuration dict

    Returns:
        Dict with divergence details or None
    """
    # Placeholder for Phase 1b implementation
    # Will use scipy.signal.find_peaks for proper peak detection
    # and compare price peaks with oscillator peaks
    pass


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
