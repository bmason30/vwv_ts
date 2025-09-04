"""
Filename: analysis/baldwin_indicator.py
VWV Trading System v4.2.1
Created/Updated: 2025-09-04 09:40:11 EDT
Version: 4.2.0 - Definitive fix for data combination logic using standard methods
Purpose: Baldwin Market Regime Indicator - Multi-factor traffic light system (GREEN/YELLOW/RED)
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from typing import Dict, Any, List
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

# Baldwin Indicator Configuration V4.2.0
BALDWIN_CONFIG = {
    'weights': { 'momentum': 0.50, 'liquidity': 0.30, 'sentiment': 0.20 },
    'symbols': {
        'spy': 'SPY', 'qqq': 'QQQ', 'iwm': 'IWM', 'fngd': 'FNGD', 'vix': '^VIX', 'uup': 'UUP', 'tlt': 'TLT',
        'hyg': 'HYG', 'lqd': 'LQD', 'sure': 'SURE', 'copy': 'COPY', 'nanc': 'NANC', 'gop': 'GOP'
    },
    'watchlist_symbols': ["EPD", "DPZ", "COST", "TFSL"],
    'thresholds': { 'green': 70, 'yellow': 40, 'red': 40 },
    'vix_warning_level': 21, 'cache_ttl': 300,
    'momentum_sub_weights': {'trend': 0.5, 'breakout': 0.3, 'roc': 0.2}
}

_baldwin_cache, _cache_timestamps = {}, {}

def fetch_baldwin_data(symbols: List[str], period: str = '1y'):
    """Robustly fetches data using a two-pass approach for maximum reliability."""
    import time
    cache_key = f"{'-'.join(sorted(symbols))}_{period}"
    current_time = time.time()
    if (cache_key in _baldwin_cache and current_time - _cache_timestamps.get(cache_key, 0) < BALDWIN_CONFIG['cache_ttl']):
        return _baldwin_cache[cache_key]

    # Pass 1: Try to fetch all symbols at once for speed.
    try:
        data = yf.download(symbols, period=period, interval="1d", progress=False, auto_adjust=True)
        if not data.empty and isinstance(data.columns, pd.MultiIndex):
            data.bfill(inplace=True); data.ffill(inplace=True)
            _baldwin_cache[cache_key], _cache_timestamps[cache_key] = data, current_time
            return data
    except Exception as e:
        logger.warning(f"Group yfinance download failed: {e}. Falling back to individual fetching.")

    # Pass 2: If group fetch fails, fetch one by one (more robust).
    all_data = {}
    for symbol in symbols:
        try:
            ticker_data = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
            if not ticker_data.empty: all_data[symbol] = ticker_data
        except Exception as e:
            logger.error(f"Could not fetch data for single symbol {symbol}: {e}")
    
    if not all_data:
        logger.error(f"Failed to fetch any valid ticker data for list: {symbols}")
        return None

    combined_df = pd.concat(all_data, axis=1)
    combined_df.columns = combined_df.columns.swaplevel(0, 1)
    combined_df.sort_index(axis=1, level=0, inplace=True)
    combined_df.bfill(inplace=True); combined_df.ffill(inplace=True)
    
    _baldwin_cache[cache_key], _cache_timestamps[cache_key] = combined_df, current_time
    return combined_df

# ... The rest of the calculation functions are unchanged ...
@safe_calculation_wrapper
def calculate_normalized_strength_score(price_series: pd.Series) -> Dict[str, Any]:
    if price_series.isnull().all() or len(price_series) < 200: return {'score': 50}
    ema_20, ema_50, ema_200 = price_series.ewm(span=20, adjust=False).mean().iloc[-1], price_series.ewm(span=50, adjust=False).mean().iloc[-1], price_series.ewm(span=200, adjust=False).mean().iloc[-1]
    current_price = price_series.iloc[-1]
    price_points, score_points = sorted([ema_200, ema_50, ema_20]), sorted([15, 50, 85])
    score = np.clip(np.interp(current_price, price_points, score_points), 0, 100)
    return {'score': score, 'price': current_price, 'ema_20': ema_20, 'ema_50': ema_50, 'ema_200': ema_200}

@safe_calculation_wrapper
def calculate_breakout_score(price_series: pd.Series, lookback: int = 50) -> Dict[str, Any]:
    if len(price_series) < lookback: return {'score': 50, 'status': 'Insufficient Data'}
    high_50d, low_50d = price_series.tail(lookback).max(), price_series.tail(lookback).min()
    current_price = price_series.iloc[-1]
    if current_price >= high_50d: return {'score': 100, 'status': f'Breakout'}
    if current_price <= low_50d: return {'score': 0, 'status': f'Breakdown'}
    return {'score': np.interp(current_price, [low_50d, high_50d], [0, 100]), 'status': 'In Range'}

@safe_calculation_wrapper
def calculate_roc_score(price_series: pd.Series, period: int = 20) -> Dict[str, Any]:
    roc = price_series.pct_change(period).iloc[-1] * 100
    return {'score': np.clip(np.interp(roc, [-10, 10], [0, 100]), 0, 100), 'roc_pct': roc}

@safe_calculation_wrapper
def calculate_momentum_component(market_data: pd.DataFrame) -> Dict[str, Any]:
    spy_trend, spy_breakout, spy_roc = calculate_normalized_strength_score(market_data['Close']['SPY']), calculate_breakout_score(market_data['Close']['SPY']), calculate_roc_score(market_data['Close']['SPY'])
    spy_composite_score = (spy_trend['score'] * 0.5 + spy_breakout['score'] * 0.3 + spy_roc['score'] * 0.2)
    iwm_trend = calculate_normalized_strength_score(market_data['Close']['IWM'])
    market_internals_score = max(0, iwm_trend['score'] - (20 if iwm_trend['score'] < spy_trend['score'] else 0))
    fngd_price, fngd_ema20, vix_level = market_data['Close']['FNGD'].iloc[-1], market_data['Close']['FNGD'].ewm(span=20, adjust=False).mean().iloc[-1], market_data['Close']['^VIX'].iloc[-1]
    leverage_fear_score = max(0, 100 - (40 if fngd_price > fngd_ema20 else 0) - (30 if vix_level > BALDWIN_CONFIG['vix_warning_level'] else 0))
    is_downturn = spy_trend['price'] < spy_trend['ema_50']
    recovery_bonus = 15 if is_downturn and (market_data['Close']['IWM'].pct_change(5).iloc[-1] > market_data['Close']['SPY'].pct_change(5).iloc[-1] or fngd_price < fngd_ema20) else 0
    final_score = min(100, ((spy_composite_score * 0.5) + (market_internals_score * 0.3) + (leverage_fear_score * 0.2)) + recovery_bonus)
    return {'component_score': round(final_score, 1), 'details': {'Broad Market (SPY)': {'score': round(spy_composite_score, 1), 'trend': spy_trend, 'breakout': spy_breakout, 'roc': spy_roc}, 'Market Internals (IWM)': {'score': round(market_internals_score, 1), 'trend': iwm_trend, 'penalty': (20 if iwm_trend['score'] < spy_trend['score'] else 0)}, 'Leverage & Fear': {'score': round(leverage_fear_score, 1), 'vix': vix_level, 'fngd_above_ema': fngd_price > fngd_ema20}, 'Recovery Bonus': {'score': recovery_bonus, 'active': recovery_bonus > 0}}}

@safe_calculation_wrapper
def calculate_liquidity_credit_component(market_data: pd.DataFrame) -> Dict[str, Any]:
    uup_strength, tlt_strength = calculate_normalized_strength_score(market_data['Close']['UUP']), calculate_normalized_strength_score(market_data['Close']['TLT'])
    flight_to_safety_score = ((100 - uup_strength['score']) * 0.6) + ((100 - tlt_strength['score']) * 0.4)
    ratio, ratio_ema20 = (market_data['Close']['HYG'] / market_data['Close']['LQD']), (market_data['Close']['HYG'] / market_data['Close']['LQD']).ewm(span=20, adjust=False).mean().iloc[-1]
    credit_spread_score = 90 if ratio.iloc[-1] > ratio_ema20 else 10
    return {'component_score': round((flight_to_safety_score * 0.5) + (credit_spread_score * 0.5), 1), 'details': {'Flight-to-Safety': {'score': round(flight_to_safety_score, 1), 'uup_strength': uup_strength, 'tlt_strength': tlt_strength}, 'Credit Spreads': {'score': credit_spread_score, 'ratio': round(ratio.iloc[-1], 2), 'ema': round(ratio_ema20, 2)}}}

@safe_calculation_wrapper
def calculate_sentiment_entry_component(main_data: pd.DataFrame, watchlist_data: pd.DataFrame) -> Dict[str, Any]:
    insider_etfs = {etf: calculate_normalized_strength_score(main_data['Close'][etf]) for etf in ['SURE', 'COPY']}
    political_etfs = {etf: calculate_normalized_strength_score(main_data['Close'][etf]) for etf in ['NANC', 'GOP']}
    insider_score, political_score = np.mean([s['score'] for s in insider_etfs.values()]), np.mean([s['score'] for s in political_etfs.values()])
    sentiment_base_score = (insider_score * 0.70) + (political_score * 0.30)
    sentiment_signal_active = sentiment_base_score > 60
    confirmed, ticker_name = False, "None"
    if sentiment_signal_active:
        for ticker in BALDWIN_CONFIG['watchlist_symbols']:
            if ticker in watchlist_data['Close'].columns and not watchlist_data['Close'][ticker].isnull().all() and (watchlist_data['Close'][ticker].iloc[-1] > watchlist_data['Close'][ticker].ewm(span=20, adjust=False).mean().iloc[-1]):
                confirmed, ticker_name = True, ticker
                break
    return {'component_score': round(95 if confirmed else sentiment_base_score, 1), 'details': {'Sentiment ETFs': {'score': round(sentiment_base_score, 1), 'insider_avg': insider_score, 'political_avg': political_score, 'details': {**insider_etfs, **political_etfs}}, 'Entry Confirmation': {'active': sentiment_signal_active, 'confirmed': confirmed, 'ticker': ticker_name}}}

@safe_calculation_wrapper
def calculate_baldwin_indicator_complete(show_debug: bool = False) -> Dict[str, Any]:
    main_data, watchlist_data = fetch_baldwin_data(list(BALDWIN_CONFIG['symbols'].values())), fetch_baldwin_data(BALDWIN_CONFIG['watchlist_symbols'])
    if main_data is None or watchlist_data is None: return {'error': 'Failed to fetch market data', 'status': 'DATA_ERROR'}
    momentum, liquidity, sentiment = calculate_momentum_component(main_data), calculate_liquidity_credit_component(main_data), calculate_sentiment_entry_component(main_data, watchlist_data)
    final_score = (momentum['component_score'] * 0.5) + (liquidity['component_score'] * 0.3) + (sentiment['component_score'] * 0.2)
    if final_score >= 70: regime, strat = "GREEN", "Risk-on: Favorable conditions."
    elif final_score >= 40: regime, strat = "YELLOW", "Caution: Neutral or transitioning conditions."
    else: regime, strat = "RED", "Risk-off: Unfavorable conditions."
    return {'baldwin_score': round(final_score, 1), 'market_regime': regime, 'strategy': strat, 'components': {'Momentum': momentum, 'Liquidity_Credit': liquidity, 'Sentiment_Entry': sentiment}, 'status': 'OPERATIONAL', 'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

def format_baldwin_for_display(baldwin_results: Dict[str, Any]) -> Dict[str, Any]:
    if 'error' in baldwin_results: return baldwin_results
    components = baldwin_results.get('components', {})
    summary = [{'Component': name.replace('_', ' & '), 'Score': f"{data.get('component_score', 0):.1f}/100", 'Weight': f"{BALDWIN_CONFIG['weights'][name.lower().split('_')[0]]*100:.0f}%"} for name, data in components.items()]
    return {'component_summary': summary, 'detailed_breakdown': components, 'overall_score': baldwin_results.get('overall_score', 0), 'regime': baldwin_results.get('market_regime', 'UNKNOWN'), 'strategy': baldwin_results.get('strategy', 'N/A'), 'timestamp': baldwin_results.get('timestamp', '')}
