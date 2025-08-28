"""
Filename: analysis/baldwin_indicator.py
VWV Trading System v4.2.1
Created/Updated: 2025-08-28 18:03:52 EST
Version: 2.0.1 - Replaced insider data placeholder with ETF-based sentiment calculation
Purpose: Baldwin Market Regime Indicator - Multi-factor traffic light system (GREEN/YELLOW/RED)
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
from typing import Dict, Any, List
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

# Baldwin Indicator Configuration V2.0.1
BALDWIN_CONFIG = {
    'weights': {
        'momentum': 0.50,    # 50%
        'liquidity': 0.30,   # 30%
        'sentiment': 0.20    # 20%
    },
    'ema_periods': [20, 50, 200],
    'symbols': {
        'spy': 'SPY',       # S&P 500
        'qqq': 'QQQ',       # Nasdaq
        'iwm': 'IWM',       # Russell 2000
        'fngd': 'FNGD',     # Inverse FANG ETN
        'vix': '^VIX',      # Volatility Index
        'uup': 'UUP',       # US Dollar Bull ETF
        'tlt': 'TLT',       # 20+ Year Treasury ETF
        'hyg': 'HYG',       # High-yield bond ETF
        'lqd': 'LQD',       # Investment-grade bond ETF
        'sure': 'SURE',     # Insider sentiment ETFs
        'copy': 'COPY',
        'nanc': 'NANC',     # Political insider ETFs
        'gop': 'GOP'
    },
    'watchlist_symbols': ["EPD", "DPZ", "STEW", "TFSL"], # For entry confirmation
    'thresholds': {
        'green': 70,        # >= 70 = GREEN
        'yellow': 40,       # 40-69 = YELLOW
        'red': 40           # < 40 = RED
    },
    'vix_warning_level': 21,
    'cache_ttl': 300  # 5 minutes
}

# Simple dictionary cache to avoid streamlit dependency issues
_baldwin_cache = {}
_cache_timestamps = {}

def fetch_baldwin_data(symbols: List[str], period: str = '1y'):
    """Fetch all required data for Baldwin Indicator with simple caching"""
    import time
    cache_key = f"{'-'.join(sorted(symbols))}_{period}"
    current_time = time.time()
    
    if (cache_key in _baldwin_cache and 
        cache_key in _cache_timestamps and 
        current_time - _cache_timestamps[cache_key] < BALDWIN_CONFIG['cache_ttl']):
        return _baldwin_cache[cache_key]
    
    try:
        data = yf.download(symbols, period=period, interval="1d", progress=False)
        if data.empty:
            logger.warning(f"No data fetched for symbols: {symbols}")
            return None
        _baldwin_cache[cache_key] = data
        _cache_timestamps[cache_key] = current_time
        return data
    except Exception as e:
        logger.error(f"Baldwin data fetch error for {symbols}: {e}")
        return None

@safe_calculation_wrapper
def calculate_ema_position_score(data: pd.Series, current_price: float, ema_periods: list = [20, 50, 200]) -> Dict[str, Any]:
    """Calculate position relative to EMAs and assign score"""
    ema_scores = {}
    for period in ema_periods:
        if len(data) >= period:
            ema = data.ewm(span=period, adjust=False).mean().iloc[-1]
            ema_scores[f'ema_{period}'] = {
                'value': round(float(ema), 2),
                'above': current_price > ema
            }
    
    if current_price > ema_scores.get('ema_20', {}).get('value', float('inf')):
        return {'score': 100, 'description': "Above 20 EMA"}
    elif current_price > ema_scores.get('ema_50', {}).get('value', float('inf')):
        return {'score': 70, 'description': "Above 50 EMA"}
    elif current_price > ema_scores.get('ema_200', {}).get('value', float('inf')):
        return {'score': 30, 'description': "Above 200 EMA"}
    else:
        return {'score': 0, 'description': "Below All EMAs"}

@safe_calculation_wrapper
def calculate_momentum_component(market_data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Momentum Component (50% weight) with V2 Recovery Scan"""
    spy_score = calculate_ema_position_score(market_data['Close']['SPY'], market_data['Close']['SPY'].iloc[-1])
    qqq_score = calculate_ema_position_score(market_data['Close']['QQQ'], market_data['Close']['QQQ'].iloc[-1])
    broad_market_score = (spy_score['score'] * 0.6) + (qqq_score['score'] * 0.4)
    
    iwm_score = calculate_ema_position_score(market_data['Close']['IWM'], market_data['Close']['IWM'].iloc[-1])
    underperformance_penalty = 20 if iwm_score['score'] < spy_score['score'] else 0
    market_internals_score = max(0, iwm_score['score'] - underperformance_penalty)
    
    fngd_price = market_data['Close']['FNGD'].iloc[-1]
    fngd_ema20 = market_data['Close']['FNGD'].ewm(span=20, adjust=False).mean().iloc[-1]
    fngd_penalty = 40 if fngd_price > fngd_ema20 else 0
    
    vix_level = market_data['Close']['^VIX'].iloc[-1]
    vix_penalty = 30 if vix_level > BALDWIN_CONFIG['vix_warning_level'] else 0
    leverage_fear_score = max(0, 100 - fngd_penalty - vix_penalty)
    
    recovery_bonus = 0
    spy_ema50 = market_data['Close']['SPY'].ewm(span=50, adjust=False).mean().iloc[-1]
    is_downturn = market_data['Close']['SPY'].iloc[-1] < spy_ema50
    
    if is_downturn:
        iwm_stronger = market_data['Close']['IWM'].pct_change(5).iloc[-1] > market_data['Close']['SPY'].pct_change(5).iloc[-1]
        fngd_recovering = fngd_price < fngd_ema20
        if iwm_stronger or fngd_recovering:
            recovery_bonus = 15

    momentum_total = (broad_market_score * 0.4) + (market_internals_score * 0.3) + (leverage_fear_score * 0.3)
    final_score = min(100, momentum_total + recovery_bonus)

    return {
        'component_score': round(final_score, 1),
        'sub_components': {
            'Broad Market': broad_market_score, 'Market Internals': market_internals_score,
            'Leverage & Fear': leverage_fear_score, 'Recovery Bonus': recovery_bonus
        }
    }

@safe_calculation_wrapper
def calculate_liquidity_credit_component(market_data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Liquidity & Credit Component (30% weight) with V2 Credit Spreads"""
    uup_price = market_data['Close']['UUP'].iloc[-1]
    uup_ema20 = market_data['Close']['UUP'].ewm(span=20, adjust=False).mean().iloc[-1]
    dollar_score = 20 if uup_price > uup_ema20 else 80

    tlt_price = market_data['Close']['TLT'].iloc[-1]
    tlt_ema20 = market_data['Close']['TLT'].ewm(span=20, adjust=False).mean().iloc[-1]
    bond_score = 20 if tlt_price > tlt_ema20 else 80
    
    flight_to_safety_score = (dollar_score * 0.6) + (bond_score * 0.4)

    hyg_lqd_ratio = market_data['Close']['HYG'] / market_data['Close']['LQD']
    ratio_ema20 = hyg_lqd_ratio.ewm(span=20, adjust=False).mean().iloc[-1]
    credit_spread_score = 90 if hyg_lqd_ratio.iloc[-1] > ratio_ema20 else 10
    
    final_score = (flight_to_safety_score * 0.5) + (credit_spread_score * 0.5)
    
    return {
        'component_score': round(final_score, 1),
        'sub_components': { 'Flight-to-Safety': flight_to_safety_score, 'Credit Spreads': credit_spread_score }
    }

@safe_calculation_wrapper
def calculate_sentiment_entry_component(main_data: pd.DataFrame, watchlist_data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Sentiment & Entry using ETF proxies (SURE/COPY 70%, NANC/GOP 30%)"""
    # Stage 1: Calculate sentiment score from ETFs
    insider_etfs = {'SURE': 0, 'COPY': 0}
    political_etfs = {'NANC': 0, 'GOP': 0}
    
    for etf in insider_etfs:
        price = main_data['Close'][etf].iloc[-1]
        ema20 = main_data['Close'][etf].ewm(span=20, adjust=False).mean().iloc[-1]
        insider_etfs[etf] = 100 if price > ema20 else 0

    for etf in political_etfs:
        price = main_data['Close'][etf].iloc[-1]
        ema20 = main_data['Close'][etf].ewm(span=20, adjust=False).mean().iloc[-1]
        political_etfs[etf] = 100 if price > ema20 else 0
        
    insider_score = np.mean(list(insider_etfs.values()))
    political_score = np.mean(list(political_etfs.values()))
    
    sentiment_base_score = (insider_score * 0.70) + (political_score * 0.30)
    sentiment_signal_active = sentiment_base_score > 50

    # Stage 2: Entry Confirmation on Watchlist
    confirmation_found = False
    confirmed_ticker = "None"
    if sentiment_signal_active:
        for ticker in BALDWIN_CONFIG['watchlist_symbols']:
            if ticker in watchlist_data['Close'].columns and not watchlist_data['Close'][ticker].isnull().all():
                price = watchlist_data['Close'][ticker].iloc[-1]
                ema20 = watchlist_data['Close'][ticker].ewm(span=20, adjust=False).mean().iloc[-1]
                if price > ema20:
                    confirmation_found = True
                    confirmed_ticker = ticker
                    break
    
    final_score = 95 if confirmation_found else sentiment_base_score
    
    return {
        'component_score': round(final_score, 1),
        'sub_components': {
            'Sentiment Signal Active': sentiment_signal_active,
            'Confirmation Found': confirmation_found,
            'Confirmed Ticker': confirmed_ticker,
            'ETF Score': sentiment_base_score
        }
    }

@safe_calculation_wrapper
def calculate_baldwin_indicator_complete(show_debug: bool = False) -> Dict[str, Any]:
    """Calculate complete Baldwin Market Regime Indicator V2.0.1"""
    main_data = fetch_baldwin_data(list(BALDWIN_CONFIG['symbols'].values()))
    watchlist_data = fetch_baldwin_data(BALDWIN_CONFIG['watchlist_symbols'])
    
    if main_data is None or watchlist_data is None or main_data.isnull().values.any():
        return {'error': 'Insufficient market data for Baldwin Indicator calculation', 'status': 'DATA_ERROR'}

    momentum_result = calculate_momentum_component(main_data)
    liquidity_result = calculate_liquidity_credit_component(main_data)
    sentiment_result = calculate_sentiment_entry_component(main_data, watchlist_data)
    
    final_score = (
        momentum_result['component_score'] * BALDWIN_CONFIG['weights']['momentum'] +
        liquidity_result['component_score'] * BALDWIN_CONFIG['weights']['liquidity'] +
        sentiment_result['component_score'] * BALDWIN_CONFIG['weights']['sentiment']
    )
    
    if final_score >= BALDWIN_CONFIG['thresholds']['green']:
        market_regime, strategy = "GREEN", "Risk-on: Favorable conditions for long positions."
    elif final_score >= BALDWIN_CONFIG['thresholds']['yellow']:
        market_regime, strategy = "YELLOW", "Caution: Neutral or transitioning conditions. Hedge appropriately."
    else:
        market_regime, strategy = "RED", "Risk-off: Unfavorable conditions. Raise cash or hedge aggressively."
    
    return {
        'baldwin_score': round(final_score, 1), 'market_regime': market_regime, 'strategy': strategy,
        'components': { 'Momentum': momentum_result, 'Liquidity_Credit': liquidity_result, 'Sentiment_Entry': sentiment_result },
        'status': 'OPERATIONAL', 'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def format_baldwin_for_display(baldwin_results: Dict[str, Any]) -> Dict[str, Any]:
    """Format V2.0.1 Baldwin results for UI display"""
    if 'error' in baldwin_results:
        return baldwin_results
    
    components = baldwin_results.get('components', {})
    component_summary = []
    for comp_name, comp_data in components.items():
        comp_name_formatted = comp_name.replace('_', ' & ')
        weight_key = comp_name.lower().split('_')[0]
        component_summary.append({
            'Component': comp_name_formatted,
            'Score': f"{comp_data.get('component_score', 0):.1f}/100",
            'Weight': f"{BALDWIN_CONFIG['weights'].get(weight_key, 0)*100:.0f}%",
        })
    
    return {
        'component_summary': component_summary, 'detailed_breakdown': components,
        'overall_score': baldwin_results.get('baldwin_score', 0), 'regime': baldwin_results.get('market_regime', 'UNKNOWN'),
        'strategy': baldwin_results.get('strategy', 'No strategy available'), 'timestamp': baldwin_results.get('timestamp', '')
    }
