"""
Market correlation and comparison analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import logging
from typing import Dict, Any
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

@st.cache_data(ttl=600)  # 10-minute cache for correlation data
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

@safe_calculation_wrapper
def calculate_breakout_breakdown_analysis(show_debug=False):
    """Calculate breakout/breakdown ratios for major indices"""
    try:
        indices = ['SPY', 'QQQ', 'IWM']
        results = {}
        
        for index in indices:
            try:
                if show_debug:
                    st.write(f"📊 Analyzing breakouts/breakdowns for {index}...")
                
                # Get recent data (3 months for reliable signals)
                ticker = yf.Ticker(index)
                data = ticker.history(period='3mo')
                
                if len(data) < 50:
                    continue
                    
                current_price = data['Close'].iloc[-1]
                
                # Multi-timeframe resistance levels
                resistance_10 = data['High'].rolling(10).max().iloc[-2]   # 10-day high
                resistance_20 = data['High'].rolling(20).max().iloc[-2]   # 20-day high
                resistance_50 = data['High'].rolling(50).max().iloc[-2]   # 50-day high
                
                # Multi-timeframe support levels  
                support_10 = data['Low'].rolling(10).min().iloc[-2]       # 10-day low
                support_20 = data['Low'].rolling(20).min().iloc[-2]       # 20-day low
                support_50 = data['Low'].rolling(50).min().iloc[-2]       # 50-day low
                
                # Breakout signals (price above resistance)
                breakout_10 = 1 if current_price > resistance_10 else 0
                breakout_20 = 1 if current_price > resistance_20 else 0
                breakout_50 = 1 if current_price > resistance_50 else 0
                
                # Breakdown signals (price below support)
                breakdown_10 = 1 if current_price < support_10 else 0
                breakdown_20 = 1 if current_price < support_20 else 0
                breakdown_50 = 1 if current_price < support_50 else 0
                
                # Calculate ratios
                total_breakouts = breakout_10 + breakout_20 + breakout_50
                total_breakdowns = breakdown_10 + breakdown_20 + breakdown_50
                
                breakout_ratio = (total_breakouts / 3) * 100  # Percentage of timeframes showing breakout
                breakdown_ratio = (total_breakdowns / 3) * 100  # Percentage of timeframes showing breakdown
                net_ratio = breakout_ratio - breakdown_ratio  # Net bias
                
                # Determine overall signal strength
                if net_ratio > 66:
                    signal_strength = "Very Bullish"
                elif net_ratio > 33:
                    signal_strength = "Bullish" 
                elif net_ratio > -33:
                    signal_strength = "Neutral"
                elif net_ratio > -66:
                    signal_strength = "Bearish"
                else:
                    signal_strength = "Very Bearish"
                
                results[index] = {
                    'current_price': round(current_price, 2),
                    'breakout_ratio': round(breakout_ratio, 1),
                    'breakdown_ratio': round(breakdown_ratio, 1),
                    'net_ratio': round(net_ratio, 1),
                    'signal_strength': signal_strength,
                    'breakout_levels': {
                        '10d': round(resistance_10, 2),
                        '20d': round(resistance_20, 2), 
                        '50d': round(resistance_50, 2)
                    },
                    'breakdown_levels': {
                        '10d': round(support_10, 2),
                        '20d': round(support_20, 2),
                        '50d': round(support_50, 2)
                    },
                    'active_breakouts': [f"{days}d" for days, signal in 
                                       [('10', breakout_10), ('20', breakout_20), ('50', breakout_50)] if signal],
                    'active_breakdowns': [f"{days}d" for days, signal in 
                                        [('10', breakdown_10), ('20', breakdown_20), ('50', breakdown_50)] if signal]
                }
                
            except Exception as e:
                if show_debug:
                    st.write(f"❌ Error analyzing {index}: {e}")
                continue
        
        # Calculate overall market sentiment
        if results:
            overall_breakout = sum([results[idx]['breakout_ratio'] for idx in results]) / len(results)
            overall_breakdown = sum([results[idx]['breakdown_ratio'] for idx in results]) / len(results)
            overall_net = overall_breakout - overall_breakdown
            
            # Market regime classification
            if overall_net > 50:
                market_regime = "🚀 Strong Breakout Environment"
            elif overall_net > 20:
                market_regime = "📈 Bullish Breakout Bias"
            elif overall_net > -20:
                market_regime = "⚖️ Balanced Market"
            elif overall_net > -50:
                market_regime = "📉 Bearish Breakdown Bias"
            else:
                market_regime = "🔻 Strong Breakdown Environment"
            
            results['OVERALL'] = {
                'breakout_ratio': round(overall_breakout, 1),
                'breakdown_ratio': round(overall_breakdown, 1), 
                'net_ratio': round(overall_net, 1),
                'market_regime': market_regime,
                'sample_size': len(results)
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Breakout/breakdown analysis error: {e}")
        return {}
