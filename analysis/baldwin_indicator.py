"""
Filename: analysis/baldwin_indicator.py
VWV Trading System v4.2.1
Created/Updated: 2025-08-29 09:19:30 EDT
Version: 4.0.0 - Complete refactor to multi-factor composite scoring for all pillars
Purpose: Baldwin Market Regime Indicator - Multi-factor traffic light system (GREEN/YELLOW/RED)
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from typing import Dict, Any, List
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

# Baldwin Indicator Configuration V4.0.0
BALDWIN_CONFIG = {
    'weights': { 'momentum': 0.50, 'liquidity': 0.30, 'sentiment': 0.20 },
    'symbols': {
        'spy': 'SPY', 'qqq': 'QQQ', 'iwm': 'IWM', 'fngd': 'FNGD', 'vix': '^VIX', 'uup': 'UUP', 'tlt': 'TLT',
        'hyg': 'HYG', 'lqd': 'LQD', 'sure': 'SURE', 'copy': 'COPY', 'nanc': 'NANC', 'gop': 'GOP'
    },
    'watchlist_symbols': ["EPD", "DPZ", "STEW", "TFSL"],
    'thresholds': { 'green': 70, 'yellow': 40, 'red': 40 },
    'vix_warning_level': 21, 'cache_ttl': 300,
    'momentum_sub_weights': {'trend': 0.5, 'breakout': 0.3, 'roc': 0.2}
}

_baldwin_cache, _cache_timestamps = {}, {}

def fetch_baldwin_data(symbols: List[str], period: str = '1y'):
    """Fetch and clean data with caching and backfilling."""
    import time
    cache_key = f"{'-'.join(sorted(symbols))}_{period}"
    current_time = time.time()
    if (cache_key in _baldwin_cache and current_time - _cache_timestamps.get(cache_key, 0) < BALDWIN_CONFIG['cache_ttl']):
        return _baldwin_cache[cache_key]
    try:
        data = yf.download(symbols, period=period, interval="1d", progress=False)
        if data.empty: return None
        data.bfill(inplace=True).ffill(inplace=True)
        _baldwin_cache[cache_key], _cache_timestamps[cache_key] = data, current_time
        return data
    except Exception: return None

@safe_calculation_wrapper
def calculate_normalized_strength_score(price_series: pd.Series) -> Dict[str, Any]:
    """Calculates a 0-100 score based on price position relative to 20, 50, 200 EMAs."""
    if price_series.isnull().all() or len(price_series) < 200:
        return {'score': 50, 'price': 0, 'ema_20': 0, 'ema_50': 0, 'ema_200': 0}

    ema_20 = price_series.ewm(span=20, adjust=False).mean().iloc[-1]
    ema_50 = price_series.ewm(span=50, adjust=False).mean().iloc[-1]
    ema_200 = price_series.ewm(span=200, adjust=False).mean().iloc[-1]
    current_price = price_series.iloc[-1]
    price_points = sorted([ema_200, ema_50, ema_20])
    score_points = sorted([15, 50, 85])
    score = np.clip(np.interp(current_price, price_points, score_points), 0, 100)

    return {'score': score, 'price': current_price, 'ema_20': ema_20, 'ema_50': ema_50, 'ema_200': ema_200}

@safe_calculation_wrapper
def calculate_breakout_score(price_series: pd.Series, lookback: int = 50) -> Dict[str, Any]:
    """Calculates a score based on proximity to recent highs/lows."""
    if len(price_series) < lookback: return {'score': 50, 'status': 'Insufficient Data'}
    
    high_50d, low_50d = price_series.tail(lookback).max(), price_series.tail(lookback).min()
    current_price = price_series.iloc[-1]
    
    if current_price >= high_50d: return {'score': 100, 'status': f'Breakout of {lookback}D High'}
    if current_price <= low_50d: return {'score': 0, 'status': f'Breakdown of {lookback}D Low'}
    
    score = np.interp(current_price, [low_50d, high_50d], [0, 100])
    return {'score': score, 'status': 'In Range'}

@safe_calculation_wrapper
def calculate_roc_score(price_series: pd.Series, period: int = 20) -> Dict[str, Any]:
    """Calculates a normalized score from Rate of Change (ROC)."""
    roc = price_series.pct_change(period).iloc[-1] * 100
    score = np.clip(np.interp(roc, [-10, 10], [0, 100]), 0, 100)
    return {'score': score, 'roc_pct': roc}

@safe_calculation_wrapper
def calculate_momentum_component(market_data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Momentum Component as a composite of Trend, Breakout, and ROC."""
    spy_trend = calculate_normalized_strength_score(market_data['Close']['SPY'])
    spy_breakout = calculate_breakout_score(market_data['Close']['SPY'])
    spy_roc = calculate_roc_score(market_data['Close']['SPY'])
    
    spy_composite_score = (spy_trend['score'] * BALDWIN_CONFIG['momentum_sub_weights']['trend'] +
                           spy_breakout['score'] * BALDWIN_CONFIG['momentum_sub_weights']['breakout'] +
                           spy_roc['score'] * BALDWIN_CONFIG['momentum_sub_weights']['roc'])

    iwm_trend = calculate_normalized_strength_score(market_data['Close']['IWM'])
    underperformance_penalty = 20 if iwm_trend['score'] < spy_trend['score'] else 0
    market_internals_score = max(0, iwm_trend['score'] - underperformance_penalty)
    
    fngd_price, fngd_ema20 = market_data['Close']['FNGD'].iloc[-1], market_data['Close']['FNGD'].ewm(span=20, adjust=False).mean().iloc[-1]
    vix_level = market_data['Close']['^VIX'].iloc[-1]
    leverage_fear_score = max(0, 100 - (40 if fngd_price > fngd_ema20 else 0) - (30 if vix_level > BALDWIN_CONFIG['vix_warning_level'] else 0))
    
    is_downturn = spy_trend['price'] < spy_trend['ema_50']
    recovery_bonus = 15 if is_downturn and (market_data['Close']['IWM'].pct_change(5).iloc[-1] > market_data['Close']['SPY'].pct_change(5).iloc[-1] or fngd_price < fngd_ema20) else 0
    
    final_score = min(100, ((spy_composite_score * 0.5) + (market_internals_score * 0.3) + (leverage_fear_score * 0.2)) + recovery_bonus)

    return {
        'component_score': round(final_score, 1),
        'details': {
            'Broad Market (SPY)': {'score': round(spy_composite_score, 1), 'trend': spy_trend, 'breakout': spy_breakout, 'roc': spy_roc},
            'Market Internals (IWM)': {'score': round(market_internals_score, 1), 'trend': iwm_trend, 'penalty': underperformance_penalty},
            'Leverage & Fear': {'score': round(leverage_fear_score, 1), 'vix': vix_level, 'fngd_above_ema': fngd_price > fngd_ema20},
            'Recovery Bonus': {'score': recovery_bonus, 'active': recovery_bonus > 0}
        }
    }

@safe_calculation_wrapper
def calculate_liquidity_credit_component(market_data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Liquidity & Credit Component with detailed breakdown."""
    uup_strength = calculate_normalized_strength_score(market_data['Close']['UUP'])
    tlt_strength = calculate_normalized_strength_score(market_data['Close']['TLT'])
    flight_to_safety_score = ((100 - uup_strength['score']) * 0.6) + ((100 - tlt_strength['score']) * 0.4)
    
    ratio = market_data['Close']['HYG'] / market_data['Close']['LQD']
    ratio_ema20 = ratio.ewm(span=20, adjust=False).mean().iloc[-1]
    credit_spread_score = 90 if ratio.iloc[-1] > ratio_ema20 else 10
    
    return {
        'component_score': round((flight_to_safety_score * 0.5) + (credit_spread_score * 0.5), 1),
        'details': {
            'Flight-to-Safety': {'score': round(flight_to_safety_score, 1), 'uup_strength': uup_strength, 'tlt_strength': tlt_strength},
            'Credit Spreads': {'score': credit_spread_score, 'ratio': round(ratio.iloc[-1], 2), 'ema': round(ratio_ema20, 2)}
        }
    }

@safe_calculation_wrapper
def calculate_sentiment_entry_component(main_data: pd.DataFrame, watchlist_data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Sentiment & Entry using ETF proxies with detailed breakdown."""
    insider_etfs = {etf: calculate_normalized_strength_score(main_data['Close'][etf]) for etf in ['SURE', 'COPY']}
    political_etfs = {etf: calculate_normalized_strength_score(main_data['Close'][etf]) for etf in ['NANC', 'GOP']}
    
    insider_score = np.mean([s['score'] for s in insider_etfs.values()])
    political_score = np.mean([s['score'] for s in political_etfs.values()])
    sentiment_base_score = (insider_score * 0.70) + (political_score * 0.30)
    sentiment_signal_active = sentiment_base_score > 60
    
    confirmed, ticker_name = False, "None"
    if sentiment_signal_active:
        for ticker in BALDWIN_CONFIG['watchlist_symbols']:
            if ticker in watchlist_data['Close'].columns and not watchlist_data['Close'][ticker].isnull().all() and (watchlist_data['Close'][ticker].iloc[-1] > watchlist_data['Close'][ticker].ewm(span=20, adjust=False).mean().iloc[-1]):
                confirmed, ticker_name = True, ticker
                break
    
    return {
        'component_score': round(95 if confirmed else sentiment_base_score, 1),
        'details': {
            'Sentiment ETFs': {'score': round(sentiment_base_score, 1), 'insider_avg': insider_score, 'political_avg': political_score, 'details': {**insider_etfs, **political_etfs}},
            'Entry Confirmation': {'active': sentiment_signal_active, 'confirmed': confirmed, 'ticker': ticker_name}
        }
    }

@safe_calculation_wrapper
def calculate_baldwin_indicator_complete(show_debug: bool = False) -> Dict[str, Any]:
    """Calculate complete Baldwin Market Regime Indicator V4.0.0"""
    main_data, watchlist_data = fetch_baldwin_data(list(BALDWIN_CONFIG['symbols'].values())), fetch_baldwin_data(BALDWIN_CONFIG['watchlist_symbols'])
    if main_data is None or watchlist_data is None: return {'error': 'Failed to fetch market data', 'status': 'DATA_ERROR'}

    momentum, liquidity, sentiment = calculate_momentum_component(main_data), calculate_liquidity_credit_component(main_data), calculate_sentiment_entry_component(main_data, watchlist_data)
    final_score = (momentum['component_score'] * 0.5) + (liquidity['component_score'] * 0.3) + (sentiment['component_score'] * 0.2)
    
    if final_score >= 70: regime, strat = "GREEN", "Risk-on: Favorable conditions."
    elif final_score >= 40: regime, strat = "YELLOW", "Caution: Neutral or transitioning conditions."
    else: regime, strat = "RED", "Risk-off: Unfavorable conditions."
    
    return {
        'baldwin_score': round(final_score, 1), 'market_regime': regime, 'strategy': strat,
        'components': {'Momentum': momentum, 'Liquidity_Credit': liquidity, 'Sentiment_Entry': sentiment},
        'status': 'OPERATIONAL', 'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def format_baldwin_for_display(baldwin_results: Dict[str, Any]) -> Dict[str, Any]:
    """Format V4.0.0 Baldwin results for UI display"""
    if 'error' in baldwin_results: return baldwin_results
    components = baldwin_results.get('components', {})
    summary = [{'Component': name.replace('_', ' & '), 'Score': f"{data.get('component_score', 0):.1f}/100", 'Weight': f"{BALDWIN_CONFIG['weights'][name.lower().split('_')[0]]*100:.0f}%"} for name, data in components.items()]
    return {
        'component_summary': summary, 'detailed_breakdown': components,
        'overall_score': baldwin_results.get('baldwin_score', 0), 'regime': baldwin_results.get('market_regime', 'UNKNOWN'),
        'strategy': baldwin_results.get('strategy', 'N/A'), 'timestamp': baldwin_results.get('timestamp', '')
    }
