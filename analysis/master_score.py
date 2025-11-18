"""
Master Score System - Unified Scoring Across All Analysis Modules
Phase 1a implementation - aggregates scores from all modules into 0-100 scale
"""

import numpy as np
from config.settings import get_master_score_config


def calculate_master_score(analysis_results):
    """
    Calculate unified master score from all analysis modules.

    Args:
        analysis_results: Dict containing results from all analysis modules:
            - technical_score: Technical analysis score (0-100)
            - fundamental_score: Fundamental analysis score (0-100)
            - vwv_signal: VWV signal strength (0-10)
            - momentum_score: Momentum indicators score (0-100)
            - divergence_score: Divergence detection score (-30 to +30)
            - volume_score: Volume analysis score (0-5)
            - volatility_score: Volatility analysis score (0-5)

    Returns:
        Dict with master score and breakdown
    """
    config = get_master_score_config()
    weights = config['weights']
    normalization = config['normalization']

    # Normalize each component to 0-100 scale
    normalized_scores = {}
    component_details = {}

    # Technical Score (already 0-100)
    technical_raw = analysis_results.get('technical_score', 50)
    normalized_scores['technical'] = normalize_score(
        technical_raw,
        0,
        normalization['technical_max']
    )
    component_details['technical'] = {
        'raw': technical_raw,
        'normalized': normalized_scores['technical'],
        'weight': weights['technical']
    }

    # Fundamental Score (already 0-100)
    fundamental_raw = analysis_results.get('fundamental_score', 50)
    normalized_scores['fundamental'] = normalize_score(
        fundamental_raw,
        0,
        normalization['fundamental_max']
    )
    component_details['fundamental'] = {
        'raw': fundamental_raw,
        'normalized': normalized_scores['fundamental'],
        'weight': weights['fundamental']
    }

    # VWV Signal (0-10 scale)
    vwv_raw = analysis_results.get('vwv_signal', 0)
    normalized_scores['vwv_signal'] = normalize_score(
        vwv_raw,
        0,
        normalization['vwv_max']
    )
    component_details['vwv_signal'] = {
        'raw': vwv_raw,
        'normalized': normalized_scores['vwv_signal'],
        'weight': weights['vwv_signal']
    }

    # Momentum Score (0-100 scale from RSI/MFI/etc.)
    momentum_raw = analysis_results.get('momentum_score', 50)
    normalized_scores['momentum'] = normalize_score(
        momentum_raw,
        0,
        normalization['momentum_max']
    )
    component_details['momentum'] = {
        'raw': momentum_raw,
        'normalized': normalized_scores['momentum'],
        'weight': weights['momentum']
    }

    # Divergence Score (-30 to +30 scale)
    divergence_raw = analysis_results.get('divergence_score', 0)
    # Shift to 0-60 range, then normalize to 0-100
    divergence_shifted = divergence_raw + 30
    normalized_scores['divergence'] = normalize_score(
        divergence_shifted,
        0,
        60  # -30 to +30 becomes 0 to 60
    )
    component_details['divergence'] = {
        'raw': divergence_raw,
        'normalized': normalized_scores['divergence'],
        'weight': weights['divergence']
    }

    # Volume Score (0-5 scale)
    volume_raw = analysis_results.get('volume_score', 0)
    normalized_scores['volume'] = normalize_score(
        volume_raw,
        0,
        normalization['volume_max']
    )
    component_details['volume'] = {
        'raw': volume_raw,
        'normalized': normalized_scores['volume'],
        'weight': weights['volume']
    }

    # Volatility Score (0-5 scale)
    volatility_raw = analysis_results.get('volatility_score', 0)
    normalized_scores['volatility'] = normalize_score(
        volatility_raw,
        0,
        normalization['volatility_max']
    )
    component_details['volatility'] = {
        'raw': volatility_raw,
        'normalized': normalized_scores['volatility'],
        'weight': weights['volatility']
    }

    # Calculate weighted master score
    master_score = 0
    for component, normalized_value in normalized_scores.items():
        weight = weights[component]
        master_score += normalized_value * weight

    # Ensure score is within 0-100
    master_score = np.clip(master_score, 0, 100)

    # Interpret score
    interpretation = interpret_master_score(master_score, config)

    # Calculate signal strength
    signal_strength = calculate_signal_strength(master_score, config)

    return {
        'master_score': round(master_score, 1),
        'interpretation': interpretation,
        'signal_strength': signal_strength,
        'components': component_details,
        'normalized_scores': {k: round(v, 1) for k, v in normalized_scores.items()},
        'weights': weights
    }


def normalize_score(value, min_val, max_val):
    """
    Normalize a value to 0-100 scale.

    Args:
        value: Raw value to normalize
        min_val: Minimum value of the scale
        max_val: Maximum value of the scale

    Returns:
        Normalized value (0-100)
    """
    if max_val == min_val:
        return 50.0  # Default to neutral if no range

    normalized = ((value - min_val) / (max_val - min_val)) * 100
    return np.clip(normalized, 0, 100)


def interpret_master_score(score, config):
    """
    Interpret master score and return sentiment.

    Args:
        score: Master score (0-100)
        config: Configuration dict

    Returns:
        String interpretation
    """
    thresholds = config['score_thresholds']

    if score >= thresholds['extreme_bullish']:
        return "🟢 Extreme Bullish"
    elif score >= thresholds['strong_bullish']:
        return "🟢 Strong Bullish"
    elif score >= thresholds['moderate_bullish']:
        return "🟢 Moderate Bullish"
    elif score >= thresholds['neutral_high']:
        return "⚪ Neutral (Bullish Lean)"
    elif score >= thresholds['neutral_low']:
        return "⚪ Neutral"
    elif score >= thresholds['moderate_bearish']:
        return "⚪ Neutral (Bearish Lean)"
    elif score >= thresholds['strong_bearish']:
        return "🔴 Moderate Bearish"
    elif score >= thresholds['extreme_bearish']:
        return "🔴 Strong Bearish"
    else:
        return "🔴 Extreme Bearish"


def calculate_signal_strength(score, config):
    """
    Calculate signal strength based on master score.

    Args:
        score: Master score (0-100)
        config: Configuration dict

    Returns:
        String signal strength
    """
    strength = config['signal_strength']

    # Calculate distance from neutral (50)
    distance_from_neutral = abs(score - 50)

    if distance_from_neutral >= 35:  # Score <= 15 or >= 85
        return "Very Strong"
    elif distance_from_neutral >= 20:  # Score <= 30 or >= 70
        return "Strong"
    elif distance_from_neutral >= 5:  # Score between 45-55
        return "Moderate"
    else:
        return "Weak"


def calculate_component_agreement(analysis_results):
    """
    Calculate agreement/disagreement between different analysis components.

    Args:
        analysis_results: Dict with all analysis results

    Returns:
        Dict with agreement metrics
    """
    # Extract normalized scores (all on 0-100 scale)
    scores = []
    labels = []

    if 'technical_score' in analysis_results:
        scores.append(analysis_results['technical_score'])
        labels.append('Technical')

    if 'fundamental_score' in analysis_results:
        scores.append(analysis_results['fundamental_score'])
        labels.append('Fundamental')

    if 'momentum_score' in analysis_results:
        scores.append(analysis_results['momentum_score'])
        labels.append('Momentum')

    if len(scores) < 2:
        return {
            'agreement_level': 'Insufficient data',
            'std_dev': 0,
            'consensus': 'N/A'
        }

    # Calculate standard deviation (lower = more agreement)
    std_dev = np.std(scores)
    mean_score = np.mean(scores)

    # Interpret agreement
    if std_dev < 10:
        agreement_level = "Strong Agreement"
        consensus = "High Confidence"
    elif std_dev < 20:
        agreement_level = "Moderate Agreement"
        consensus = "Medium Confidence"
    else:
        agreement_level = "Low Agreement"
        consensus = "Low Confidence - Mixed Signals"

    return {
        'agreement_level': agreement_level,
        'consensus': consensus,
        'std_dev': round(std_dev, 1),
        'mean_score': round(mean_score, 1),
        'component_count': len(scores)
    }


def calculate_master_score_with_agreement(analysis_results):
    """
    Calculate master score with component agreement analysis.

    Args:
        analysis_results: Dict containing all analysis results

    Returns:
        Dict with master score and agreement metrics
    """
    master_result = calculate_master_score(analysis_results)
    agreement_result = calculate_component_agreement(analysis_results)

    return {
        **master_result,
        'agreement': agreement_result
    }


def get_score_color(score):
    """
    Get color for score visualization.

    Args:
        score: Score value (0-100)

    Returns:
        Color string for UI
    """
    if score >= 70:
        return "green"
    elif score >= 60:
        return "lightgreen"
    elif score >= 40:
        return "orange"
    elif score >= 30:
        return "darkorange"
    else:
        return "red"


def create_master_score_summary(master_result):
    """
    Create a human-readable summary of the master score.

    Args:
        master_result: Dict from calculate_master_score()

    Returns:
        String summary
    """
    score = master_result['master_score']
    interpretation = master_result['interpretation']
    strength = master_result['signal_strength']

    summary = f"Master Score: {score:.1f}/100 - {interpretation}\n"
    summary += f"Signal Strength: {strength}\n\n"
    summary += "Component Contributions:\n"

    components = master_result['components']
    for name, details in components.items():
        contribution = details['normalized'] * details['weight']
        summary += f"  • {name.title()}: {details['raw']} → {details['normalized']:.1f} "
        summary += f"(weight: {details['weight']:.0%}) = +{contribution:.1f}\n"

    return summary
