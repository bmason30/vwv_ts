"""
VWV Trading System - Comprehensive Market Analysis v7.0.0
Major functionality enhancement: All 4 available improvements implemented

NEW FEATURES:
1. ✅ Momentum Divergence Analysis
2. ✅ Intermarket Analysis (Dollar, Yields, Commodities)  
3. ✅ Volatility Regime Analysis (VIX-based)
4. ✅ Seasonal Pattern Analysis

PLUS: Enhanced breakout/breakdown analysis v6.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

# === CONFIGURATION ===

# Intermarket symbols available through yfinance
INTERMARKET_SYMBOLS = {
    'dollar': 'UUP',      # US Dollar Bull ETF
    'dollar_index': '^DX-Y.NYB',  # DXY Alternative
    'yield_10yr': '^TNX',  # 10-Year Treasury Yield
    'yield_2yr': '^IRX',   # 3-Month Treasury (proxy for 2yr)
    'gold': 'GLD',        # Gold ETF
    'oil': 'USO',         # Oil ETF
    'commodities': 'DJP', # Commodities ETF
    'vix': '^VIX',        # Volatility Index
    'bonds': 'TLT'        # 20+ Year Treasury ETF
}

# Seasonal analysis configuration
SEASONAL_CONFIG = {
    'months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
    'lookback_years': 5,
    'min_observations': 20
}

# Enhanced symbol universe for breakout analysis
ENHANCED_BREAKOUT_SYMBOLS = [
    # Core Indices
    'SPY', 'QQQ', 'IWM', 'VTI', 'DIA',
    # Sectors
    'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLRE', 'XLB', 'XLC',
    # International
    'EFA', 'EEM', 'FXI',
    # Commodities & Currencies
    'GLD', 'SLV', 'USO', 'UUP'
]

# === CACHING FUNCTIONS ===

@st.cache_data(ttl=600)  # 10-minute cache
def get_intermarket_data(symbols: Dict[str, str], period: str = '1y') -> Dict[str, pd.DataFrame]:
    """Cached function to fetch intermarket data"""
    data = {}
    
    for name, symbol in symbols.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if len(hist) > 50:
                data[name] = hist
            else:
                logger.warning(f"Insufficient data for {name} ({symbol})")
                
        except Exception as e:
            logger.error(f"Error fetching {name} ({symbol}): {e}")
            continue
    
    return data

@st.cache_data(ttl=600)
def get_correlation_etf_data(etf_symbols, period='1y'):
    """Cached function to fetch correlation ETF data"""
    etf_data = {}
    
    for etf in etf_symbols:
        try:
            etf_ticker = yf.Ticker(etf)
            etf_history = etf_ticker.history(period=period)
            
            if len(etf_history) > 50:
                etf_returns = etf_history['Close'].pct_change().dropna()
                etf_data[etf] = etf_returns
                
        except Exception as e:
            logger.error(f"Error fetching {etf} data: {e}")
            continue
    
    return etf_data

# === 1. MOMENTUM DIVERGENCE ANALYSIS ===

@safe_calculation_wrapper
def calculate_momentum_divergence(data: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
    """
    NEW FEATURE: Detect momentum divergences between price and indicators
    
    Divergences often signal trend reversals before they happen in price.
    """
    try:
        if len(data) < period * 2:
            return {'divergence_score': 50, 'signals': [], 'strength': 'Neutral'}
        
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # Calculate indicators
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        
        # Look for divergences in last 20 periods
        lookback = min(20, len(data) - period)
        recent_data = data.tail(lookback)
        recent_rsi = rsi.tail(lookback)
        recent_macd = macd.tail(lookback)
        
        divergence_signals = []
        divergence_score = 50  # Neutral starting point
        
        if len(recent_data) >= 10:
            # Check for bullish divergences (price makes lower lows, indicator makes higher lows)
            price_lows = recent_data['Low'].rolling(5).min()
            rsi_lows = recent_rsi.rolling(5).min()
            macd_lows = recent_macd.rolling(5).min()
            
            # Price trend vs RSI trend over last 10 periods
            price_trend = np.polyfit(range(10), close.tail(10).values, 1)[0]
            rsi_trend = np.polyfit(range(10), recent_rsi.tail(10).values, 1)[0]
            macd_trend = np.polyfit(range(10), recent_macd.tail(10).values, 1)[0]
            
            # Bullish divergence: price declining, momentum rising
            if price_trend < -0.001 and rsi_trend > 0.1:
                divergence_signals.append('Bullish RSI Divergence')
                divergence_score += 15
                
            if price_trend < -0.001 and macd_trend > 0:
                divergence_signals.append('Bullish MACD Divergence')
                divergence_score += 15
            
            # Bearish divergence: price rising, momentum falling
            if price_trend > 0.001 and rsi_trend < -0.1:
                divergence_signals.append('Bearish RSI Divergence')
                divergence_score -= 15
                
            if price_trend > 0.001 and macd_trend < 0:
                divergence_signals.append('Bearish MACD Divergence')
                divergence_score -= 15
            
            # Hidden divergences (trend continuation signals)
            current_price = close.iloc[-1]
            prev_price = close.iloc[-10]
            current_rsi = rsi.iloc[-1]
            prev_rsi = rsi.iloc[-10]
            
            # Hidden bullish: higher lows in price, lower lows in RSI (uptrend continuation)
            if current_price > prev_price and current_rsi < prev_rsi and price_trend > 0:
                divergence_signals.append('Hidden Bullish Divergence')
                divergence_score += 10
            
            # Hidden bearish: lower highs in price, higher highs in RSI (downtrend continuation)
            if current_price < prev_price and current_rsi > prev_rsi and price_trend < 0:
                divergence_signals.append('Hidden Bearish Divergence')
                divergence_score -= 10
        
        # Clamp score
        divergence_score = max(0, min(100, divergence_score))
        
        # Strength classification
        if divergence_score >= 70:
            strength = 'Strong Bullish'
        elif divergence_score >= 60:
            strength = 'Bullish'
        elif divergence_score >= 40:
            strength = 'Neutral'
        elif divergence_score >= 30:
            strength = 'Bearish'
        else:
            strength = 'Strong Bearish'
        
        return {
            'divergence_score': round(divergence_score, 1),
            'signals': divergence_signals,
            'strength': strength,
            'price_trend': round(price_trend * 1000, 2),  # Scale for readability
            'rsi_trend': round(rsi_trend, 2),
            'macd_trend': round(macd_trend * 1000, 2),
            'signal_count': len(divergence_signals)
        }
        
    except Exception as e:
        logger.error(f"Momentum divergence calculation error: {e}")
        return {'divergence_score': 50, 'signals': [], 'strength': 'Neutral', 'error': str(e)}

# === 2. INTERMARKET ANALYSIS ===

@safe_calculation_wrapper
def calculate_intermarket_analysis(show_debug: bool = False) -> Dict[str, Any]:
    """
    NEW FEATURE: Comprehensive intermarket analysis
    
    Analyzes relationships between dollars, yields, commodities, and equities
    """
    try:
        if show_debug:
            st.write("🌍 Fetching intermarket data...")
        
        # Get intermarket data
        intermarket_data = get_intermarket_data(INTERMARKET_SYMBOLS, period='6mo')
        
        if len(intermarket_data) < 3:
            return {'error': 'Insufficient intermarket data', 'regime': 'Unknown'}
        
        results = {}
        current_values = {}
        trends = {}
        
        # Analyze each market
        for market, data in intermarket_data.items():
            if len(data) < 20:
                continue
                
            current_price = data['Close'].iloc[-1]
            ma_20 = data['Close'].rolling(20).mean().iloc[-1]
            ma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else ma_20
            
            # Calculate 20-day trend
            returns_20d = (current_price / data['Close'].iloc[-21] - 1) * 100 if len(data) >= 21 else 0
            
            # Volatility
            vol_20d = data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
            
            trend_strength = 'Neutral'
            if returns_20d > 5:
                trend_strength = 'Strong Up'
            elif returns_20d > 2:
                trend_strength = 'Up'
            elif returns_20d < -5:
                trend_strength = 'Strong Down'
            elif returns_20d < -2:
                trend_strength = 'Down'
            
            results[market] = {
                'current_price': round(current_price, 2),
                'ma_20': round(ma_20, 2),
                'ma_50': round(ma_50, 2),
                'returns_20d': round(returns_20d, 2),
                'volatility_20d': round(vol_20d, 2),
                'trend_strength': trend_strength,
                'above_ma20': current_price > ma_20,
                'above_ma50': current_price > ma_50
            }
            
            current_values[market] = current_price
            trends[market] = returns_20d
        
        # === REGIME ANALYSIS ===
        regime_score = 50  # Start neutral
        regime_factors = []
        
        # Dollar strength analysis
        if 'dollar' in results:
            dollar_trend = results['dollar']['returns_20d']
            if dollar_trend > 3:
                regime_score -= 10  # Strong dollar = headwind for equities
                regime_factors.append('Strong Dollar (Equity Headwind)')
            elif dollar_trend < -3:
                regime_score += 10  # Weak dollar = tailwind for equities
                regime_factors.append('Weak Dollar (Equity Tailwind)')
        
        # Yield curve analysis
        if 'yield_10yr' in results:
            yield_trend = results['yield_10yr']['returns_20d']
            if yield_trend > 10:  # Rising yields
                regime_score -= 8
                regime_factors.append('Rising Yields (Growth Headwind)')
            elif yield_trend < -10:  # Falling yields
                regime_score += 8
                regime_factors.append('Falling Yields (Growth Tailwind)')
        
        # Gold analysis (risk-off indicator)
        if 'gold' in results:
            gold_trend = results['gold']['returns_20d']
            if gold_trend > 5:
                regime_score -= 5  # Flight to safety
                regime_factors.append('Gold Strength (Risk-Off)')
            elif gold_trend < -3:
                regime_score += 5  # Risk-on environment
                regime_factors.append('Gold Weakness (Risk-On)')
        
        # VIX analysis
        if 'vix' in results:
            vix_level = results['vix']['current_price']
            if vix_level > 25:
                regime_score -= 15  # High fear
                regime_factors.append(f'High VIX ({vix_level:.1f}) - Fear')
            elif vix_level < 15:
                regime_score += 10  # Complacency
                regime_factors.append(f'Low VIX ({vix_level:.1f}) - Complacency')
            elif vix_level > 20:
                regime_score -= 5
                regime_factors.append(f'Elevated VIX ({vix_level:.1f}) - Caution')
        
        # Commodities analysis
        if 'oil' in results:
            oil_trend = results['oil']['returns_20d']
            if oil_trend > 10:
                regime_score += 5  # Economic growth
                regime_factors.append('Oil Strength (Growth)')
            elif oil_trend < -15:
                regime_score -= 5  # Economic weakness
                regime_factors.append('Oil Weakness (Economic Concern)')
        
        # Clamp regime score
        regime_score = max(0, min(100, regime_score))
        
        # Regime classification
        if regime_score >= 70:
            market_regime = '🟢 Risk-On Environment'
        elif regime_score >= 55:
            market_regime = '🟡 Cautiously Bullish'
        elif regime_score >= 45:
            market_regime = '⚪ Neutral/Mixed'
        elif regime_score >= 30:
            market_regime = '🟠 Cautiously Bearish'
        else:
            market_regime = '🔴 Risk-Off Environment'
        
        return {
            'regime_score': round(regime_score, 1),
            'market_regime': market_regime,
            'regime_factors': regime_factors,
            'markets': results,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'data_quality': len(intermarket_data)
        }
        
    except Exception as e:
        logger.error(f"Intermarket analysis error: {e}")
        return {'error': str(e), 'regime': 'Unknown'}

# === 3. VOLATILITY REGIME ANALYSIS ===

@safe_calculation_wrapper
def calculate_volatility_regime(show_debug: bool = False) -> Dict[str, Any]:
    """
    NEW FEATURE: VIX-based volatility regime analysis
    
    Provides context for risk management and position sizing
    """
    try:
        if show_debug:
            st.write("📊 Analyzing volatility regime...")
        
        # Get VIX data
        vix_ticker = yf.Ticker('^VIX')
        vix_data = vix_ticker.history(period='2y')  # 2 years for percentiles
        
        if len(vix_data) < 100:
            return {'error': 'Insufficient VIX data', 'regime': 'Unknown'}
        
        current_vix = vix_data['Close'].iloc[-1]
        
        # Calculate percentiles over 2-year period
        vix_values = vix_data['Close'].values
        percentile_10 = np.percentile(vix_values, 10)
        percentile_25 = np.percentile(vix_values, 25)
        percentile_50 = np.percentile(vix_values, 50)
        percentile_75 = np.percentile(vix_values, 75)
        percentile_90 = np.percentile(vix_values, 90)
        
        # Current percentile
        current_percentile = (vix_values < current_vix).sum() / len(vix_values) * 100
        
        # VIX trend analysis
        vix_5d = vix_data['Close'].rolling(5).mean().iloc[-1]
        vix_20d = vix_data['Close'].rolling(20).mean().iloc[-1]
        
        # Volatility regime classification
        if current_vix >= percentile_90:
            vol_regime = '🔥 Extreme Fear'
            risk_level = 'Maximum'
            position_sizing = 'Minimal positions'
        elif current_vix >= percentile_75:
            vol_regime = '🚨 High Volatility'
            risk_level = 'High'
            position_sizing = 'Reduced positions'
        elif current_vix >= percentile_50:
            vol_regime = '⚠️ Elevated Volatility'
            risk_level = 'Medium'
            position_sizing = 'Normal positions'
        elif current_vix >= percentile_25:
            vol_regime = '😐 Normal Volatility'
            risk_level = 'Low'
            position_sizing = 'Normal positions'
        else:
            vol_regime = '😴 Low Volatility'
            risk_level = 'Very Low'
            position_sizing = 'Consider larger positions'
        
        # VIX structure analysis
        vix_trend = 'Neutral'
        if vix_5d > vix_20d * 1.1:
            vix_trend = 'Rising'
        elif vix_5d < vix_20d * 0.9:
            vix_trend = 'Falling'
        
        # Term structure (if available)
        term_structure = 'Normal'
        try:
            vix9d = yf.Ticker('^VIX9D').history(period='1mo')
            if len(vix9d) > 0:
                vix9d_current = vix9d['Close'].iloc[-1]
                if vix9d_current > current_vix * 1.05:
                    term_structure = 'Backwardation (Bullish)'
                elif vix9d_current < current_vix * 0.95:
                    term_structure = 'Contango (Bearish)'
        except:
            pass  # VIX9D might not be available
        
        return {
            'current_vix': round(current_vix, 2),
            'vix_percentile': round(current_percentile, 1),
            'vol_regime': vol_regime,
            'risk_level': risk_level,
            'position_sizing': position_sizing,
            'vix_trend': vix_trend,
            'term_structure': term_structure,
            'percentiles': {
                '10th': round(percentile_10, 2),
                '25th': round(percentile_25, 2),
                '50th': round(percentile_50, 2),
                '75th': round(percentile_75, 2),
                '90th': round(percentile_90, 2)
            },
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M')
        }
        
    except Exception as e:
        logger.error(f"Volatility regime analysis error: {e}")
        return {'error': str(e), 'regime': 'Unknown'}

# === 4. SEASONAL PATTERN ANALYSIS ===

@safe_calculation_wrapper
def calculate_seasonal_patterns(symbol: str, show_debug: bool = False) -> Dict[str, Any]:
    """
    NEW FEATURE: Statistical seasonal pattern analysis
    
    Analyzes historical performance by month and day of week
    """
    try:
        if show_debug:
            st.write(f"📅 Analyzing seasonal patterns for {symbol}...")
        
        # Get 5+ years of data for seasonal analysis
        ticker = yf.Ticker(symbol)
        data = ticker.history(period='5y')
        
        if len(data) < 500:  # Need substantial history
            return {'error': 'Insufficient historical data for seasonal analysis'}
        
        # Calculate daily returns
        data['Returns'] = data['Close'].pct_change()
        data['Month'] = data.index.month
        data['DayOfWeek'] = data.index.dayofweek  # 0=Monday
        data['Year'] = data.index.year
        
        # === MONTHLY ANALYSIS ===
        monthly_stats = {}
        for month in range(1, 13):
            month_data = data[data['Month'] == month]['Returns'].dropna()
            if len(month_data) >= SEASONAL_CONFIG['min_observations']:
                monthly_stats[month] = {
                    'avg_return': month_data.mean() * 100,
                    'win_rate': (month_data > 0).sum() / len(month_data) * 100,
                    'observations': len(month_data),
                    'volatility': month_data.std() * 100,
                    'best_month': month_data.max() * 100,
                    'worst_month': month_data.min() * 100
                }
        
        # === DAY OF WEEK ANALYSIS ===
        daily_stats = {}
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        for day in range(5):  # Trading days only
            day_data = data[data['DayOfWeek'] == day]['Returns'].dropna()
            if len(day_data) >= SEASONAL_CONFIG['min_observations']:
                daily_stats[day] = {
                    'day_name': day_names[day],
                    'avg_return': day_data.mean() * 100,
                    'win_rate': (day_data > 0).sum() / len(day_data) * 100,
                    'observations': len(day_data),
                    'volatility': day_data.std() * 100
                }
        
        # === CURRENT MONTH/DAY ANALYSIS ===
        current_month = datetime.now().month
        current_day = datetime.now().weekday()
        
        current_month_stats = monthly_stats.get(current_month, {})
        current_day_stats = daily_stats.get(current_day, {})
        
        # === SEASONALITY SCORE ===
        seasonality_score = 50  # Neutral
        seasonal_factors = []
        
        # Month bias
        if current_month_stats:
            month_return = current_month_stats.get('avg_return', 0)
            month_win_rate = current_month_stats.get('win_rate', 50)
            
            if month_return > 1.0 and month_win_rate > 60:
                seasonality_score += 15
                seasonal_factors.append(f"Strong {SEASONAL_CONFIG['months'][current_month-1]} (Historically +{month_return:.1f}%)")
            elif month_return < -0.5 and month_win_rate < 45:
                seasonality_score -= 15
                seasonal_factors.append(f"Weak {SEASONAL_CONFIG['months'][current_month-1]} (Historically {month_return:.1f}%)")
        
        # Day bias
        if current_day_stats:
            day_return = current_day_stats.get('avg_return', 0)
            day_win_rate = current_day_stats.get('win_rate', 50)
            day_name = current_day_stats.get('day_name', 'Unknown')
            
            if day_return > 0.1 and day_win_rate > 55:
                seasonality_score += 5
                seasonal_factors.append(f"Strong {day_name} (Historically +{day_return:.2f}%)")
            elif day_return < -0.1 and day_win_rate < 45:
                seasonality_score -= 5
                seasonal_factors.append(f"Weak {day_name} (Historically {day_return:.2f}%)")
        
        # Find best and worst months
        best_months = []
        worst_months = []
        
        for month, stats in monthly_stats.items():
            if stats['avg_return'] > 1.5 and stats['win_rate'] > 65:
                best_months.append((SEASONAL_CONFIG['months'][month-1], stats['avg_return']))
            elif stats['avg_return'] < -0.5 and stats['win_rate'] < 40:
                worst_months.append((SEASONAL_CONFIG['months'][month-1], stats['avg_return']))
        
        # Sort by performance
        best_months.sort(key=lambda x: x[1], reverse=True)
        worst_months.sort(key=lambda x: x[1])
        
        return {
            'seasonality_score': round(max(0, min(100, seasonality_score)), 1),
            'seasonal_factors': seasonal_factors,
            'current_month': SEASONAL_CONFIG['months'][current_month-1],
            'current_day': day_names[current_day] if current_day < 5 else 'Weekend',
            'current_month_stats': current_month_stats,
            'current_day_stats': current_day_stats,
            'monthly_stats': monthly_stats,
            'daily_stats': daily_stats,
            'best_months': best_months[:3],  # Top 3
            'worst_months': worst_months[:3],  # Bottom 3
            'data_period': f"{data.index[0].strftime('%Y-%m')} to {data.index[-1].strftime('%Y-%m')}",
            'total_observations': len(data)
        }
        
    except Exception as e:
        logger.error(f"Seasonal analysis error: {e}")
        return {'error': str(e)}

# === ENHANCED BREAKOUT ANALYSIS (from v6.0.0) ===

@safe_calculation_wrapper
def calculate_atr_dynamic_threshold(data: pd.DataFrame, period: int = 14, multiplier: float = 1.5) -> float:
    """Calculate ATR-based dynamic threshold for breakout detection"""
    if len(data) < period + 1:
        return 0.005  # Default 0.5% if insufficient data
    
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift(1)).abs()
    low_close = (data['Low'] - data['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean().iloc[-1]
    
    current_price = data['Close'].iloc[-1]
    atr_percentage = (atr / current_price) * multiplier
    
    return float(atr_percentage)

@safe_calculation_wrapper
def calculate_enhanced_breakout_analysis(symbols: List[str] = None, show_debug: bool = False) -> Dict[str, Any]:
    """Enhanced breakout/breakdown analysis with all improvements"""
    if symbols is None:
        symbols = ENHANCED_BREAKOUT_SYMBOLS[:12]  # Use first 12 for performance
    
    try:
        results = {}
        
        for symbol in symbols:
            try:
                if show_debug:
                    logger.info(f"📊 Enhanced analysis for {symbol}")
                
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='6mo')
                
                if len(data) < 50:
                    continue
                
                current_price = data['Close'].iloc[-1]
                atr_threshold = calculate_atr_dynamic_threshold(data)
                
                # Multi-timeframe breakout analysis
                timeframes = {'5d': 5, '10d': 10, '20d': 20}
                timeframe_scores = []
                
                for period in timeframes.values():
                    if len(data) >= period + 1:
                        period_high = data['High'].tail(period + 1).iloc[:-1].max()
                        period_low = data['Low'].tail(period + 1).iloc[:-1].min()
                        
                        breakout_level = period_high * (1 + atr_threshold)
                        breakdown_level = period_low * (1 - atr_threshold)
                        
                        if current_price > breakout_level:
                            timeframe_scores.append(80)
                        elif current_price > period_high:
                            timeframe_scores.append(60)
                        elif current_price < breakdown_level:
                            timeframe_scores.append(20)
                        elif current_price < period_low:
                            timeframe_scores.append(40)
                        else:
                            timeframe_scores.append(50)
                
                # Volume analysis
                current_volume = data['Volume'].iloc[-1]
                avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                
                volume_score = 50
                if volume_ratio >= 2.0:
                    volume_score = 90
                elif volume_ratio >= 1.5:
                    volume_score = 75
                elif volume_ratio >= 1.2:
                    volume_score = 65
                elif volume_ratio >= 0.8:
                    volume_score = 50
                else:
                    volume_score = 30
                
                # Moving average analysis
                ma_20 = data['Close'].rolling(20).mean().iloc[-1]
                ma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else ma_20
                
                ma_score = 50
                if current_price > ma_20 * 1.02 and current_price > ma_50 * 1.02:
                    ma_score = 80
                elif current_price > ma_20 and current_price > ma_50:
                    ma_score = 65
                elif current_price < ma_20 * 0.98 and current_price < ma_50 * 0.98:
                    ma_score = 20
                elif current_price < ma_20 and current_price < ma_50:
                    ma_score = 35
                
                # Composite scoring
                if timeframe_scores:
                    avg_timeframe = sum(timeframe_scores) / len(timeframe_scores)
                else:
                    avg_timeframe = 50
                
                breakout_score = (
                    avg_timeframe * 0.5 +
                    volume_score * 0.3 +
                    ma_score * 0.2
                )
                
                breakdown_score = 100 - breakout_score
                
                breakout_ratio = max(0, min(100, breakout_score))
                breakdown_ratio = max(0, min(100, breakdown_score))
                net_ratio = breakout_ratio - breakdown_ratio
                
                # Signal classification
                if net_ratio > 40:
                    signal_strength = "Very Bullish"
                elif net_ratio > 15:
                    signal_strength = "Bullish"
                elif net_ratio > -15:
                    signal_strength = "Neutral"
                elif net_ratio > -40:
                    signal_strength = "Bearish"
                else:
                    signal_strength = "Very Bearish"
                
                results[symbol] = {
                    'current_price': round(current_price, 2),
                    'breakout_ratio': round(breakout_ratio, 1),
                    'breakdown_ratio': round(breakdown_ratio, 1),
                    'net_ratio': round(net_ratio, 1),
                    'signal_strength': signal_strength,
                    'volume_ratio': round(volume_ratio, 2),
                    'ma_20': round(ma_20, 2),
                    'ma_50': round(ma_50, 2),
                    'analysis_method': 'Enhanced Multi-Factor v6.0.0'
                }
                
                if show_debug:
                    logger.info(f"  • {symbol}: {breakout_ratio:.1f}% breakout, {breakdown_ratio:.1f}% breakdown")
                
            except Exception as e:
                if show_debug:
                    logger.error(f"❌ Error analyzing {symbol}: {e}")
                continue
        
        # Overall market sentiment calculation
        if results:
            overall_breakout = sum([results[idx]['breakout_ratio'] for idx in results]) / len(results)
            overall_breakdown = sum([results[idx]['breakdown_ratio'] for idx in results]) / len(results)
            overall_net = overall_breakout - overall_breakdown
            
            # Market regime classification
            if overall_net > 50:
                market_regime = "🚀 Explosive Breakout Environment"
            elif overall_net > 30:
                market_regime = "📈 Strong Bullish Breakout Bias"
            elif overall_net > 10:
                market_regime = "📊 Mild Bullish Bias"
            elif overall_net > -10:
                market_regime = "⚖️ Balanced/Neutral Market"
            elif overall_net > -30:
                market_regime = "📉 Mild Bearish Bias"
            elif overall_net > -50:
                market_regime = "🔻 Strong Bearish Breakdown Bias"
            else:
                market_regime = "💥 Severe Breakdown Environment"
            
            results['OVERALL'] = {
                'breakout_ratio': round(overall_breakout, 1),
                'breakdown_ratio': round(overall_breakdown, 1),
                'net_ratio': round(overall_net, 1),
                'market_regime': market_regime,
                'sample_size': len(results),
                'analysis_method': 'Enhanced Multi-Factor v6.0.0'
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Enhanced breakout analysis error: {e}")
        return {}

# === EXISTING CORRELATION FUNCTIONS ===

@safe_calculation_wrapper
def calculate_market_correlations_enhanced(symbol_data, symbol, period='1y', show_debug=False):
    """Enhanced market correlations with caching to avoid redundant API calls"""
    try:
        comparison_etfs = ['FNGD', 'FNGU', 'MAGS']
        correlations = {}

        if show_debug:
            st.write(f"📊 Calculating correlations for {symbol}...")

        # Get symbol returns
        symbol_returns = symbol_data['Close'].pct_change().dropna()
        
        # Get cached ETF data
        etf_data = get_correlation_etf_data(comparison_etfs, period)

        for etf in comparison_etfs:
            try:
                if etf not in etf_data:
                    correlations[etf] = {'correlation': 0, 'beta': 0, 'relationship': 'No data available'}
                    continue
                
                etf_returns = etf_data[etf]
                
                # Align dates
                aligned_data = pd.concat([symbol_returns, etf_returns], axis=1, join='inner')
                aligned_data.columns = [symbol, etf]

                if len(aligned_data) > 30:
                    correlation = aligned_data[symbol].corr(aligned_data[etf])

                    # Calculate beta
                    covariance = aligned_data[symbol].cov(aligned_data[etf])
                    etf_variance = aligned_data[etf].var()
                    beta = covariance / etf_variance if etf_variance != 0 else 0

                    correlations[etf] = {
                        'correlation': round(float(correlation), 3),
                        'beta': round(float(beta), 3),
                        'relationship': get_correlation_description(correlation)
                    }

                    if show_debug:
                        st.write(f"  • {etf}: {correlation:.3f} correlation")
                else:
                    correlations[etf] = {'correlation': 0, 'beta': 0, 'relationship': 'Insufficient data'}

            except Exception as e:
                correlations[etf] = {'correlation': 0, 'beta': 0, 'relationship': f'Error: {str(e)[:20]}...'}
                if show_debug:
                    st.write(f"  • {etf}: Error - {str(e)}")

        return correlations

    except Exception as e:
        if show_debug:
            st.write(f"❌ Correlation calculation error: {str(e)}")
        return {}

def get_correlation_description(corr):
    """Get description of correlation strength"""
    abs_corr = abs(corr)
    if abs_corr >= 0.8:
        return "Very Strong"
    elif abs_corr >= 0.6:
        return "Strong"
    elif abs_corr >= 0.4:
        return "Moderate"
    elif abs_corr >= 0.2:
        return "Weak"
    else:
        return "Very Weak"

# Maintain backward compatibility
def calculate_breakout_breakdown_analysis(show_debug=False):
    """Wrapper function for backward compatibility"""
    return calculate_enhanced_breakout_analysis(['SPY', 'QQQ', 'IWM'], show_debug)
