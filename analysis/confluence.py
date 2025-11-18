"""
Signal Confluence Dashboard Module
Phase 1b implementation - tracks agreement across all analysis modules
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def calculate_signal_confluence(analysis_results):
    """
    Calculate signal confluence across all analysis modules.

    Args:
        analysis_results: Dict containing all analysis results

    Returns:
        Dict with confluence metrics and agreement matrix
    """
    enhanced_indicators = analysis_results.get('enhanced_indicators', {})

    # Extract signals from all modules
    signals = extract_all_signals(enhanced_indicators, analysis_results)

    # Calculate module agreement
    agreement_matrix = calculate_agreement_matrix(signals)

    # Calculate confluence score
    confluence_score = calculate_confluence_score(signals)

    # Identify conflicting signals
    conflicts = identify_conflicts(signals)

    # Calculate confidence level
    confidence = calculate_confidence_level(signals, agreement_matrix)

    return {
        'signals': signals,
        'agreement_matrix': agreement_matrix,
        'confluence_score': confluence_score,
        'conflicts': conflicts,
        'confidence': confidence,
        'total_modules': len(signals),
        'bullish_modules': sum(1 for s in signals.values() if s['direction'] == 'bullish'),
        'bearish_modules': sum(1 for s in signals.values() if s['direction'] == 'bearish'),
        'neutral_modules': sum(1 for s in signals.values() if s['direction'] == 'neutral')
    }


def extract_all_signals(enhanced_indicators, analysis_results):
    """
    Extract directional signals from all analysis modules.

    Returns:
        Dict with module name as key and signal details as value
    """
    signals = {}

    # 1. Technical Analysis Signal
    comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
    if comprehensive_technicals:
        technical_signal = extract_technical_signal(comprehensive_technicals)
        signals['technical'] = technical_signal

    # 2. Fundamental Analysis Signal
    fundamental_signal = extract_fundamental_signal(enhanced_indicators)
    if fundamental_signal:
        signals['fundamental'] = fundamental_signal

    # 3. Momentum Signal
    momentum_signal = extract_momentum_signal(comprehensive_technicals)
    if momentum_signal:
        signals['momentum'] = momentum_signal

    # 4. Divergence Signal
    divergence = enhanced_indicators.get('divergence', {})
    if divergence and divergence.get('score') is not None:
        div_score = divergence.get('score', 0)
        signals['divergence'] = {
            'direction': 'bullish' if div_score > 5 else ('bearish' if div_score < -5 else 'neutral'),
            'strength': abs(div_score) / 30 * 100,  # Normalize to 0-100
            'score': div_score,
            'description': divergence.get('status', 'Unknown')
        }

    # 5. ADX Trend Strength
    adx_data = comprehensive_technicals.get('adx', {})
    if isinstance(adx_data, dict) and adx_data.get('adx'):
        signals['adx_trend'] = {
            'direction': adx_data.get('trend_direction', 'neutral').lower(),
            'strength': adx_data.get('adx', 0),
            'score': adx_data.get('adx', 0),
            'description': adx_data.get('trend_strength', 'Unknown')
        }

    # 6. Volume Analysis
    volume_analysis = enhanced_indicators.get('volume_analysis', {})
    if volume_analysis and not isinstance(volume_analysis, dict) or 'error' not in volume_analysis:
        volume_signal = extract_volume_signal(volume_analysis)
        if volume_signal:
            signals['volume'] = volume_signal

    # 7. Volatility Analysis
    volatility_analysis = enhanced_indicators.get('volatility_analysis', {})
    if volatility_analysis and not isinstance(volatility_analysis, dict) or 'error' not in volatility_analysis:
        volatility_signal = extract_volatility_signal(volatility_analysis)
        if volatility_signal:
            signals['volatility'] = volatility_signal

    # 8. Master Score Signal
    master_score = enhanced_indicators.get('master_score', {})
    if master_score and master_score.get('master_score') is not None:
        score = master_score.get('master_score', 50)
        signals['master_score'] = {
            'direction': 'bullish' if score > 55 else ('bearish' if score < 45 else 'neutral'),
            'strength': abs(score - 50) * 2,  # Normalize distance from 50
            'score': score,
            'description': master_score.get('interpretation', 'Unknown')
        }

    return signals


def extract_technical_signal(comprehensive_technicals):
    """Extract directional signal from technical indicators."""
    if not comprehensive_technicals:
        return None

    # Use RSI, MACD, and moving average position
    rsi = comprehensive_technicals.get('rsi_14', 50)
    macd_data = comprehensive_technicals.get('macd', {})
    macd_hist = macd_data.get('histogram', 0) if isinstance(macd_data, dict) else 0

    # Simple scoring
    score = 0
    if rsi > 50:
        score += (rsi - 50) / 50 * 50  # 0-50 points
    else:
        score += (rsi - 50) / 50 * 50  # -50 to 0 points

    if macd_hist > 0:
        score += 25
    else:
        score -= 25

    direction = 'bullish' if score > 10 else ('bearish' if score < -10 else 'neutral')

    return {
        'direction': direction,
        'strength': abs(score),
        'score': score,
        'description': f"RSI: {rsi:.1f}, MACD: {'Bullish' if macd_hist > 0 else 'Bearish'}"
    }


def extract_fundamental_signal(enhanced_indicators):
    """Extract directional signal from fundamental analysis."""
    graham = enhanced_indicators.get('graham_score', {})
    piotroski = enhanced_indicators.get('piotroski_score', {})

    if 'error' in graham or 'error' in piotroski:
        return None

    graham_score = graham.get('score', 0)
    piotroski_score = piotroski.get('score', 0)

    # Normalize to 0-100
    graham_pct = (graham_score / 10) * 100 if graham.get('total_possible', 10) > 0 else 0
    piotroski_pct = (piotroski_score / 9) * 100 if piotroski.get('total_possible', 9) > 0 else 0

    avg_score = (graham_pct + piotroski_pct) / 2

    direction = 'bullish' if avg_score > 60 else ('bearish' if avg_score < 40 else 'neutral')

    return {
        'direction': direction,
        'strength': avg_score,
        'score': avg_score,
        'description': f"Graham: {graham_score}/10, Piotroski: {piotroski_score}/9"
    }


def extract_momentum_signal(comprehensive_technicals):
    """Extract signal from momentum oscillators."""
    if not comprehensive_technicals:
        return None

    rsi = comprehensive_technicals.get('rsi_14', 50)
    mfi = comprehensive_technicals.get('mfi_14', 50)
    stoch = comprehensive_technicals.get('stochastic', {})
    stoch_k = stoch.get('k', 50) if isinstance(stoch, dict) else 50

    # Average momentum
    avg_momentum = (rsi + mfi + stoch_k) / 3

    direction = 'bullish' if avg_momentum > 55 else ('bearish' if avg_momentum < 45 else 'neutral')

    return {
        'direction': direction,
        'strength': abs(avg_momentum - 50) * 2,
        'score': avg_momentum,
        'description': f"Avg: {avg_momentum:.1f} (RSI/MFI/Stoch)"
    }


def extract_volume_signal(volume_analysis):
    """Extract signal from volume analysis (placeholder)."""
    # This is a placeholder - actual implementation depends on volume_analysis structure
    return None


def extract_volatility_signal(volatility_analysis):
    """Extract signal from volatility analysis (placeholder)."""
    # This is a placeholder - actual implementation depends on volatility_analysis structure
    return None


def calculate_agreement_matrix(signals):
    """
    Calculate agreement matrix showing which modules agree/disagree.

    Returns:
        Dict with pairwise agreement percentages
    """
    if len(signals) < 2:
        return {}

    agreement = {}
    module_names = list(signals.keys())

    for i, mod1 in enumerate(module_names):
        for mod2 in module_names[i+1:]:
            sig1 = signals[mod1]
            sig2 = signals[mod2]

            # Check agreement
            if sig1['direction'] == sig2['direction']:
                agreement_pct = 100
            elif sig1['direction'] == 'neutral' or sig2['direction'] == 'neutral':
                agreement_pct = 50
            else:
                agreement_pct = 0

            key = f"{mod1}_vs_{mod2}"
            agreement[key] = {
                'module1': mod1,
                'module2': mod2,
                'agreement': agreement_pct,
                'status': 'Agree' if agreement_pct == 100 else ('Partial' if agreement_pct == 50 else 'Conflict')
            }

    return agreement


def calculate_confluence_score(signals):
    """
    Calculate overall confluence score (0-100).

    Higher score = more modules agree on direction
    """
    if not signals:
        return 50

    bullish_count = sum(1 for s in signals.values() if s['direction'] == 'bullish')
    bearish_count = sum(1 for s in signals.values() if s['direction'] == 'bearish')
    neutral_count = sum(1 for s in signals.values() if s['direction'] == 'neutral')

    total = len(signals)

    # Calculate consensus
    if bullish_count > bearish_count:
        # Bullish consensus
        consensus_strength = bullish_count / total
        score = 50 + (consensus_strength * 50)
    elif bearish_count > bullish_count:
        # Bearish consensus
        consensus_strength = bearish_count / total
        score = 50 - (consensus_strength * 50)
    else:
        # No clear consensus
        score = 50

    return round(score, 1)


def identify_conflicts(signals):
    """
    Identify conflicting signals between modules.

    Returns:
        List of conflict descriptions
    """
    conflicts = []

    # Check for strong disagreements
    bullish_modules = [name for name, sig in signals.items() if sig['direction'] == 'bullish']
    bearish_modules = [name for name, sig in signals.items() if sig['direction'] == 'bearish']

    if bullish_modules and bearish_modules:
        conflicts.append({
            'type': 'directional_conflict',
            'bullish': bullish_modules,
            'bearish': bearish_modules,
            'description': f"{len(bullish_modules)} bullish vs {len(bearish_modules)} bearish"
        })

    # Check for strength mismatches
    strong_signals = [(name, sig) for name, sig in signals.items() if sig['strength'] > 70]
    weak_signals = [(name, sig) for name, sig in signals.items() if sig['strength'] < 30]

    if strong_signals and weak_signals:
        conflicts.append({
            'type': 'strength_mismatch',
            'strong': [name for name, _ in strong_signals],
            'weak': [name for name, _ in weak_signals],
            'description': f"{len(strong_signals)} strong signals vs {len(weak_signals)} weak"
        })

    return conflicts


def calculate_confidence_level(signals, agreement_matrix):
    """
    Calculate overall confidence level in the signals.

    Returns:
        Dict with confidence score and level
    """
    if not signals:
        return {'score': 0, 'level': 'No Data'}

    # Calculate average agreement
    if agreement_matrix:
        avg_agreement = sum(a['agreement'] for a in agreement_matrix.values()) / len(agreement_matrix)
    else:
        avg_agreement = 50

    # Factor in number of signals
    signal_count_factor = min(len(signals) / 8, 1.0)  # Max at 8 modules

    # Factor in signal strength
    avg_strength = sum(s['strength'] for s in signals.values()) / len(signals)
    strength_factor = avg_strength / 100

    # Combined confidence
    confidence_score = (avg_agreement * 0.5 + signal_count_factor * 100 * 0.3 + avg_strength * 0.2)

    # Interpret level
    if confidence_score >= 80:
        level = 'Very High'
    elif confidence_score >= 65:
        level = 'High'
    elif confidence_score >= 50:
        level = 'Medium'
    elif confidence_score >= 35:
        level = 'Low'
    else:
        level = 'Very Low'

    return {
        'score': round(confidence_score, 1),
        'level': level,
        'avg_agreement': round(avg_agreement, 1),
        'signal_count': len(signals),
        'avg_strength': round(avg_strength, 1)
    }


def create_confluence_summary(confluence_result):
    """
    Create human-readable summary of confluence analysis.

    Args:
        confluence_result: Dict from calculate_signal_confluence()

    Returns:
        String summary
    """
    bullish = confluence_result['bullish_modules']
    bearish = confluence_result['bearish_modules']
    neutral = confluence_result['neutral_modules']
    total = confluence_result['total_modules']

    confluence_score = confluence_result['confluence_score']
    confidence = confluence_result['confidence']

    summary = f"**Signal Confluence Analysis**\n\n"
    summary += f"- **Confluence Score**: {confluence_score}/100\n"
    summary += f"- **Confidence Level**: {confidence['level']} ({confidence['score']:.1f})\n"
    summary += f"- **Module Breakdown**: {bullish} Bullish, {bearish} Bearish, {neutral} Neutral (of {total})\n\n"

    if confluence_score > 60:
        summary += "✅ **Strong Bullish Confluence** - Most modules agree on upward direction\n"
    elif confluence_score < 40:
        summary += "🔴 **Strong Bearish Confluence** - Most modules agree on downward direction\n"
    else:
        summary += "⚪ **Mixed Signals** - No clear consensus across modules\n"

    conflicts = confluence_result['conflicts']
    if conflicts:
        summary += f"\n⚠️ **{len(conflicts)} Conflict(s) Detected**:\n"
        for conflict in conflicts:
            summary += f"  - {conflict['description']}\n"

    return summary
