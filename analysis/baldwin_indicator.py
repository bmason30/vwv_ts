"""
Baldwin Market Regime Indicator v1.0.0
Multi-factor model distilling Momentum, Liquidity, and Sentiment into traffic-light system
🟢 GREEN: Risk-on, press longs, buy dips
🟡 YELLOW: Caution, hedge, wait for clarity  
🔴 RED: Risk-off, hedge aggressively, raise cash
"""
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import logging
from typing import Dict, Any, Optional, Tuple
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

# Baldwin Indicator Configuration
BALDWIN_CONFIG = {
    'weights': {
        'momentum': 0.60,    # 60% - Most heavily weighted
        'liquidity': 0.25,   # 25% - Funding conditions
        'sentiment': 0.15    # 15% - Smart money
    },
    'ema_periods': [20, 50, 200],
    'symbols': {
        'spy': 'SPY',       # S&P 500
        'qqq': 'QQQ',       # Nasdaq
        'iwm': 'IWM',       # Russell 2000
        'fngd': 'FNGD',     # Inverse FANG ETN
        'vix': '^VIX',      # Volatility Index
        'uup': 'UUP',       # US Dollar Bull ETF
        'tlt': 'TLT'        # 20+ Year Treasury ETF
    },
    'thresholds': {
        'green': 70,        # >= 70 = GREEN
        'yellow': 40,       # 40-69 = YELLOW
        'red': 40           # < 40 = RED
    },
    'vix_warning_level': 21,
    'cache_ttl': 300  # 5 minutes
}

@st.cache_data(ttl=BALDWIN_CONFIG['cache_ttl'])
def fetch_baldwin_data(symbols: list, period: str = '6mo'):
    """Cached function to fetch all required data for Baldwin Indicator"""
    data = {}
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if len(hist) > 50:  # Ensure sufficient data
                data[symbol] = hist
            else:
                logger.warning(f"Insufficient data for {symbol}")
                
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            continue
    
    return data

@safe_calculation_wrapper
def calculate_ema_position_score(data: pd.DataFrame, current_price: float, ema_periods: list = [20, 50, 200]) -> Dict[str, Any]:
    """Calculate position relative to EMAs and assign score"""
    try:
        close = data['Close']
        ema_scores = {}
        
        for period in ema_periods:
            if len(close) >= period:
                ema = close.ewm(span=period).mean().iloc[-1]
                ema_scores[f'ema_{period}'] = {
                    'value': round(float(ema), 2),
                    'above': current_price > ema,
                    'distance_pct': round(((current_price - ema) / ema) * 100, 2)
                }
        
        # Scoring logic based on EMA positioning
        if current_price > ema_scores.get('ema_20', {}).get('value', 0):
            position_score = 100  # Strongly bullish
            position_desc = "Above 20 EMA - Strongly Bullish"
        elif current_price > ema_scores.get('ema_50', {}).get('value', 0):
            position_score = 70   # Neutral-bullish
            position_desc = "Above 50 EMA - Neutral Bullish"
        elif current_price > ema_scores.get('ema_200', {}).get('value', 0):
            position_score = 30   # Warning
            position_desc = "Above 200 EMA Only - Warning"
        else:
            position_score = 0    # Strongly bearish
            position_desc = "Below All EMAs - Strongly Bearish"
        
        return {
            'score': position_score,
            'description': position_desc,
            'ema_details': ema_scores,
            'current_price': current_price
        }
        
    except Exception as e:
        logger.error(f"EMA position calculation error: {e}")
        return {'score': 50, 'description': 'Error calculating EMA position', 'ema_details': {}}

@safe_calculation_wrapper
def calculate_momentum_component(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Calculate Momentum Component (60% weight)"""
    try:
        momentum_scores = {}
        
        # 1. Broad Market Trend (SPY + QQQ)
        spy_data = market_data.get('SPY')
        qqq_data = market_data.get('QQQ')
        
        broad_market_score = 0
        broad_market_details = {}
        
        if spy_data is not None and len(spy_data) > 0:
            spy_price = spy_data['Close'].iloc[-1]
            spy_analysis = calculate_ema_position_score(spy_data, spy_price)
            broad_market_details['SPY'] = spy_analysis
            broad_market_score += spy_analysis['score'] * 0.6  # SPY weight
        
        if qqq_data is not None and len(qqq_data) > 0:
            qqq_price = qqq_data['Close'].iloc[-1]
            qqq_analysis = calculate_ema_position_score(qqq_data, qqq_price)
            broad_market_details['QQQ'] = qqq_analysis
            broad_market_score += qqq_analysis['score'] * 0.4  # QQQ weight
        
        momentum_scores['broad_market'] = {
            'score': round(broad_market_score, 1),
            'weight': 0.4,  # 40% of momentum component
            'details': broad_market_details
        }
        
        # 2. Market Internals - Russell 2000 "Canary"
        iwm_data = market_data.get('IWM')
        if iwm_data is not None and len(iwm_data) > 0:
            iwm_price = iwm_data['Close'].iloc[-1]
            iwm_analysis = calculate_ema_position_score(iwm_data, iwm_price)
            
            # Check for relative underperformance
            underperformance_penalty = 0
            if spy_data is not None and iwm_analysis['score'] < broad_market_details.get('SPY', {}).get('score', 50):
                underperformance_penalty = 20  # Penalty for Russell underperformance
            
            iwm_final_score = max(0, iwm_analysis['score'] - underperformance_penalty)
            
            momentum_scores['market_internals'] = {
                'score': round(iwm_final_score, 1),
                'weight': 0.3,  # 30% of momentum component
                'underperformance_penalty': underperformance_penalty,
                'details': iwm_analysis
            }
        else:
            momentum_scores['market_internals'] = {'score': 50, 'weight': 0.3, 'details': {}}
        
        # 3. Leverage & Fear Gauge (FNGD + VIX)
        fear_score = 50  # Default neutral
        fear_details = {}
        
        # FNGD Analysis
        fngd_data = market_data.get('FNGD')
        if fngd_data is not None and len(fngd_data) > 0:
            fngd_price = fngd_data['Close'].iloc[-1]
            fngd_ema20 = fngd_data['Close'].ewm(span=20).mean().iloc[-1]
            
            if fngd_price > fngd_ema20:
                fngd_penalty = 40  # Heavy penalty for FNGD spike
                fear_details['FNGD'] = {
                    'price': round(float(fngd_price), 2),
                    'ema20': round(float(fngd_ema20), 2),
                    'above_ema': True,
                    'penalty': fngd_penalty,
                    'signal': 'Leveraged unwinding detected'
                }
            else:
                fngd_penalty = 0
                fear_details['FNGD'] = {
                    'price': round(float(fngd_price), 2),
                    'ema20': round(float(fngd_ema20), 2),
                    'above_ema': False,
                    'penalty': fngd_penalty,
                    'signal': 'Leveraged trades stable'
                }
        else:
            fngd_penalty = 0
            fear_details['FNGD'] = {'signal': 'Data not available'}
        
        # VIX Analysis
        vix_data = market_data.get('^VIX')
        vix_penalty = 0
        if vix_data is not None and len(vix_data) > 0:
            vix_level = vix_data['Close'].iloc[-1]
            
            if vix_level > BALDWIN_CONFIG['vix_warning_level']:
                vix_penalty = 30  # Penalty for elevated VIX
                fear_details['VIX'] = {
                    'level': round(float(vix_level), 2),
                    'warning_level': BALDWIN_CONFIG['vix_warning_level'],
                    'elevated': True,
                    'penalty': vix_penalty,
                    'signal': 'Elevated volatility warning'
                }
            else:
                fear_details['VIX'] = {
                    'level': round(float(vix_level), 2),
                    'warning_level': BALDWIN_CONFIG['vix_warning_level'],
                    'elevated': False,
                    'penalty': vix_penalty,
                    'signal': 'Volatility contained'
                }
        else:
            fear_details['VIX'] = {'signal': 'Data not available'}
        
        # Calculate final fear score
        fear_score = max(0, 100 - fngd_penalty - vix_penalty)
        
        momentum_scores['leverage_fear'] = {
            'score': fear_score,
            'weight': 0.3,  # 30% of momentum component
            'details': fear_details
        }
        
        # Calculate weighted momentum component score
        momentum_total = (
            momentum_scores['broad_market']['score'] * momentum_scores['broad_market']['weight'] +
            momentum_scores['market_internals']['score'] * momentum_scores['market_internals']['weight'] +
            momentum_scores['leverage_fear']['score'] * momentum_scores['leverage_fear']['weight']
        )
        
        return {
            'component_score': round(momentum_total, 1),
            'weight': BALDWIN_CONFIG['weights']['momentum'],
            'sub_components': momentum_scores
        }
        
    except Exception as e:
        logger.error(f"Momentum component calculation error: {e}")
        return {'component_score': 50, 'weight': 0.60, 'sub_components': {}}

@safe_calculation_wrapper
def calculate_liquidity_component(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Calculate Liquidity Component (25% weight)"""
    try:
        liquidity_scores = {}
        
        # 1. U.S. Dollar Trend (UUP)
        uup_data = market_data.get('UUP')
        if uup_data is not None and len(uup_data) > 0:
            uup_price = uup_data['Close'].iloc[-1]
            uup_ema20 = uup_data['Close'].ewm(span=20).mean().iloc[-1]
            uup_ema50 = uup_data['Close'].ewm(span=50).mean().iloc[-1]
            
            # Strong dollar is negative for equities
            if uup_price > uup_ema20 and uup_ema20 > uup_ema50:
                dollar_score = 20  # Strong uptrend = negative
                dollar_signal = "Strong dollar uptrend - negative for equities"
            elif uup_price > uup_ema20:
                dollar_score = 40  # Moderate uptrend = caution
                dollar_signal = "Moderate dollar strength"
            else:
                dollar_score = 80  # Stable/weak dollar = positive
                dollar_signal = "Dollar stable/weak - positive for equities"
            
            liquidity_scores['dollar_trend'] = {
                'score': dollar_score,
                'weight': 0.6,  # 60% of liquidity component
                'details': {
                    'price': round(float(uup_price), 2),
                    'ema20': round(float(uup_ema20), 2),
                    'ema50': round(float(uup_ema50), 2),
                    'signal': dollar_signal
                }
            }
        else:
            liquidity_scores['dollar_trend'] = {'score': 50, 'weight': 0.6, 'details': {}}
        
        # 2. Treasury Bond Trend (TLT)
        tlt_data = market_data.get('TLT')
        if tlt_data is not None and len(tlt_data) > 0:
            tlt_price = tlt_data['Close'].iloc[-1]
            tlt_ema20 = tlt_data['Close'].ewm(span=20).mean().iloc[-1]
            
            # Check for flight-to-safety in bonds during equity stress
            # This is context-dependent - need to check if SPY is also declining
            spy_data = market_data.get('SPY')
            spy_declining = False
            if spy_data is not None and len(spy_data) >= 5:
                spy_recent = spy_data['Close'].tail(5)
                spy_declining = spy_recent.iloc[-1] < spy_recent.iloc[0]  # 5-day decline
            
            if tlt_price > tlt_ema20 and spy_declining:
                bond_score = 20  # Flight to safety = negative for equities
                bond_signal = "Flight to safety in bonds - risk-off"
            elif tlt_price > tlt_ema20:
                bond_score = 60  # Bond rally without equity stress = neutral
                bond_signal = "Bond rally - falling yields"
            else:
                bond_score = 80  # Stable bonds = positive for equities
                bond_signal = "Bond stability - normal conditions"
            
            liquidity_scores['bond_trend'] = {
                'score': bond_score,
                'weight': 0.4,  # 40% of liquidity component
                'details': {
                    'price': round(float(tlt_price), 2),
                    'ema20': round(float(tlt_ema20), 2),
                    'spy_declining': spy_declining,
                    'signal': bond_signal
                }
            }
        else:
            liquidity_scores['bond_trend'] = {'score': 50, 'weight': 0.4, 'details': {}}
        
        # Calculate weighted liquidity component score
        liquidity_total = (
            liquidity_scores['dollar_trend']['score'] * liquidity_scores['dollar_trend']['weight'] +
            liquidity_scores['bond_trend']['score'] * liquidity_scores['bond_trend']['weight']
        )
        
        return {
            'component_score': round(liquidity_total, 1),
            'weight': BALDWIN_CONFIG['weights']['liquidity'],
            'sub_components': liquidity_scores
        }
        
    except Exception as e:
        logger.error(f"Liquidity component calculation error: {e}")
        return {'component_score': 50, 'weight': 0.25, 'sub_components': {}}

@safe_calculation_wrapper
def calculate_sentiment_component() -> Dict[str, Any]:
    """Calculate Sentiment Component (15% weight) - Placeholder for insider data"""
    try:
        # Placeholder for insider buy-to-sell ratio
        # This would require insider trading data which is typically premium
        # For now, return neutral score with note
        
        sentiment_score = 50  # Neutral default
        
        return {
            'component_score': sentiment_score,
            'weight': BALDWIN_CONFIG['weights']['sentiment'],
            'sub_components': {
                'insider_ratio': {
                    'score': sentiment_score,
                    'weight': 1.0,
                    'details': {
                        'note': 'Insider data integration pending - premium data required',
                        'signal': 'Neutral (no data)',
                        'future_enhancement': 'Track aggregate insider buy-to-sell ratio'
                    }
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Sentiment component calculation error: {e}")
        return {'component_score': 50, 'weight': 0.15, 'sub_components': {}}

@safe_calculation_wrapper
def calculate_baldwin_indicator_complete(show_debug: bool = False) -> Dict[str, Any]:
    """Calculate complete Baldwin Market Regime Indicator"""
    try:
        if show_debug:
            st.write("📊 Calculating Baldwin Market Regime Indicator...")
        
        # Fetch all required market data
        symbols = list(BALDWIN_CONFIG['symbols'].values())
        market_data = fetch_baldwin_data(symbols)
        
        if len(market_data) < 3:  # Need minimum data
            return {
                'error': 'Insufficient market data for Baldwin Indicator',
                'status': 'DATA_ERROR'
            }
        
        # Calculate all three components
        momentum_result = calculate_momentum_component(market_data)
        liquidity_result = calculate_liquidity_component(market_data)
        sentiment_result = calculate_sentiment_component()
        
        # Calculate weighted final score
        final_score = (
            momentum_result['component_score'] * momentum_result['weight'] +
            liquidity_result['component_score'] * liquidity_result['weight'] +
            sentiment_result['component_score'] * sentiment_result['weight']
        )
        
        # Determine market regime based on thresholds
        if final_score >= BALDWIN_CONFIG['thresholds']['green']:
            market_regime = "GREEN"
            regime_color = "🟢"
            strategy = "Risk-on: Press longs, buy dips"
            regime_description = "Favorable conditions - positive momentum and sufficient liquidity"
        elif final_score >= BALDWIN_CONFIG['thresholds']['yellow']:
            market_regime = "YELLOW"
            regime_color = "🟡"
            strategy = "Caution: Exercise hedging, wait for clarity"
            regime_description = "Neutral/deteriorating conditions - potential transition period"
        else:
            market_regime = "RED"
            regime_color = "🔴"
            strategy = "Risk-off: Hedge aggressively, raise cash"
            regime_description = "Unfavorable conditions - negative momentum dominates"
        
        # Build comprehensive results
        return {
            'baldwin_score': round(final_score, 1),
            'market_regime': market_regime,
            'regime_color': regime_color,
            'strategy': strategy,
            'regime_description': regime_description,
            'components': {
                'momentum': momentum_result,
                'liquidity': liquidity_result,
                'sentiment': sentiment_result
            },
            'configuration': BALDWIN_CONFIG,
            'status': 'OPERATIONAL',
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_quality': {
                'symbols_fetched': len(market_data),
                'symbols_required': len(symbols),
                'data_coverage': f"{len(market_data)}/{len(symbols)}"
            }
        }
        
    except Exception as e:
        logger.error(f"Baldwin Indicator calculation error: {e}")
        return {
            'error': str(e),
            'status': 'CALCULATION_ERROR',
            'baldwin_score': 50,
            'market_regime': 'UNKNOWN'
        }

def format_baldwin_for_display(baldwin_results: Dict[str, Any]) -> Dict[str, Any]:
    """Format Baldwin results for UI display"""
    try:
        if 'error' in baldwin_results:
            return baldwin_results
        
        # Create component summary table
        components = baldwin_results.get('components', {})
        component_summary = []
        
        for comp_name, comp_data in components.items():
            component_summary.append({
                'Component': comp_name.title(),
                'Score': f"{comp_data['component_score']:.1f}/100",
                'Weight': f"{comp_data['weight']*100:.0f}%",
                'Contribution': f"{comp_data['component_score'] * comp_data['weight']:.1f}"
            })
        
        # Create detailed breakdown for each component
        detailed_breakdown = {}
        
        # Momentum breakdown
        momentum = components.get('momentum', {})
        momentum_subs = momentum.get('sub_components', {})
        detailed_breakdown['momentum'] = []
        
        for sub_name, sub_data in momentum_subs.items():
            detailed_breakdown['momentum'].append({
                'Sub-Component': sub_name.replace('_', ' ').title(),
                'Score': f"{sub_data['score']:.1f}",
                'Weight': f"{sub_data['weight']*100:.0f}%",
                'Details': str(sub_data.get('details', {}))
            })
        
        # Liquidity breakdown
        liquidity = components.get('liquidity', {})
        liquidity_subs = liquidity.get('sub_components', {})
        detailed_breakdown['liquidity'] = []
        
        for sub_name, sub_data in liquidity_subs.items():
            detailed_breakdown['liquidity'].append({
                'Sub-Component': sub_name.replace('_', ' ').title(),
                'Score': f"{sub_data['score']:.1f}",
                'Weight': f"{sub_data['weight']*100:.0f}%",
                'Details': str(sub_data.get('details', {}))
            })
        
        # Sentiment breakdown
        sentiment = components.get('sentiment', {})
        sentiment_subs = sentiment.get('sub_components', {})
        detailed_breakdown['sentiment'] = []
        
        for sub_name, sub_data in sentiment_subs.items():
            detailed_breakdown['sentiment'].append({
                'Sub-Component': sub_name.replace('_', ' ').title(),
                'Score': f"{sub_data['score']:.1f}",
                'Weight': f"{sub_data['weight']*100:.0f}%",
                'Details': str(sub_data.get('details', {}))
            })
        
        return {
            'component_summary': component_summary,
            'detailed_breakdown': detailed_breakdown,
            'overall_score': baldwin_results.get('baldwin_score', 0),
            'regime': baldwin_results.get('market_regime', 'UNKNOWN'),
            'regime_color': baldwin_results.get('regime_color', '⚪'),
            'strategy': baldwin_results.get('strategy', 'No strategy available'),
            'description': baldwin_results.get('regime_description', ''),
            'timestamp': baldwin_results.get('timestamp', ''),
            'status': baldwin_results.get('status', 'UNKNOWN')
        }
        
    except Exception as e:
        logger.error(f"Baldwin display formatting error: {e}")
        return {'error': str(e), 'status': 'FORMATTING_ERROR'}
