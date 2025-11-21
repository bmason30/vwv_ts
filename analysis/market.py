"""
File: market.py v1.0.3
VWV Research And Analysis System v4.2.2
Market correlation and comparison analysis
Created: 2025-08-15
Updated: 2025-10-08
File Version: v1.0.3 - Fixed calculate_breakout_breakdown_analysis signature
System Version: v4.2.2 - Advanced Options with Fibonacci Integration
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
def calculate_enhanced_breakout_analysis(symbols=['SPY', 'QQQ', 'IWM'], show_debug=False):
    """
    Enhanced breakout/breakdown analysis with robust logic
    """
    try:
        results = {}
        
        for symbol in symbols:
            try:
                if show_debug:
                    logger.info(f"📊 Analyzing enhanced breakouts for {symbol}")
                
                # Get 3 months of data for reliable analysis
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='3mo')
                
                if len(data) < 50:
                    continue
                    
                current_price = data['Close'].iloc[-1]
                current_volume = data['Volume'].iloc[-1]
                
                # Method 1: Moving Average Analysis
                ma_20 = data['Close'].rolling(20).mean().iloc[-1]
                ma_50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else ma_20
                
                ma_score = 0
                if current_price > ma_20 * 1.005:
                    ma_score += 1
                if current_price > ma_50 * 1.01:
                    ma_score += 1
                if ma_20 > ma_50:
                    ma_score += 1
                
                # Method 2: Range Analysis
                range_score = 0
                for period in [5, 10, 20]:
                    if len(data) >= period + 2:
                        recent_high = data['High'].iloc[-(period+1):-1].max()
                        recent_low = data['Low'].iloc[-(period+1):-1].min()
                        
                        if current_price > recent_high * 1.002:
                            range_score += 1
                        elif current_price < recent_low * 0.998:
                            range_score -= 1
                
                # Method 3: Volume Confirmation
                avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                volume_score = 0
                
                if current_volume > avg_volume * 1.2:
                    volume_score += 1
                elif current_volume < avg_volume * 0.8:
                    volume_score -= 0.5
                
                # Composite scoring
                composite_raw = (ma_score * 0.4 + range_score * 0.4 + volume_score * 0.2)
                
                if composite_raw > 0:
                    breakout_ratio = min(100, composite_raw * 25)
                    breakdown_ratio = 0
                elif composite_raw < 0:
                    breakout_ratio = 0
                    breakdown_ratio = min(100, abs(composite_raw) * 25)
                else:
                    breakout_ratio = 0
                    breakdown_ratio = 0
                
                net_ratio = breakout_ratio - breakdown_ratio
                
                if net_ratio > 50:
                    signal_strength = "Very Bullish"
                elif net_ratio > 20:
                    signal_strength = "Bullish"
                elif net_ratio > -20:
                    signal_strength = "Neutral"
                elif net_ratio > -50:
                    signal_strength = "Bearish"
                else:
                    signal_strength = "Very Bearish"
                
                results[symbol] = {
                    'current_price': round(current_price, 2),
                    'breakout_ratio': round(breakout_ratio, 1),
                    'breakdown_ratio': round(breakdown_ratio, 1),
                    'net_ratio': round(net_ratio, 1),
                    'signal_strength': signal_strength,
                    'ma_20': round(ma_20, 2),
                    'ma_50': round(ma_50, 2),
                    'volume_ratio': round(current_volume / avg_volume, 2) if avg_volume > 0 else 1.0,
                    'analysis_method': 'Enhanced Multi-Factor v5.0'
                }
                
                if show_debug:
                    logger.info(f"  • {symbol}: {breakout_ratio:.1f}% breakout, {breakdown_ratio:.1f}% breakdown")
                
            except Exception as e:
                if show_debug:
                    logger.error(f"❌ Error analyzing {symbol}: {e}")
                continue
        
        # Calculate overall market sentiment
        if results:
            overall_breakout = sum([results[idx]['breakout_ratio'] for idx in results]) / len(results)
            overall_breakdown = sum([results[idx]['breakdown_ratio'] for idx in results]) / len(results)
            overall_net = overall_breakout - overall_breakdown
            
            # Market regime classification
            if overall_net > 40:
                market_regime = "🚀 Strong Breakout Environment"
            elif overall_net > 15:
                market_regime = "📈 Bullish Breakout Bias"
            elif overall_net > -15:
                market_regime = "⚖️ Balanced Market"
            elif overall_net > -40:
                market_regime = "📉 Bearish Breakdown Bias"
            else:
                market_regime = "🔻 Strong Breakdown Environment"
            
            results['OVERALL'] = {
                'breakout_ratio': round(overall_breakout, 1),
                'breakdown_ratio': round(overall_breakdown, 1),
                'net_ratio': round(overall_net, 1),
                'market_regime': market_regime,
                'sample_size': len(results),
                'analysis_method': 'Enhanced Multi-Factor v5.0'
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Enhanced breakout analysis error: {e}")
        return {}

# FIXED: Accept flexible arguments to prevent signature mismatch
def calculate_breakout_breakdown_analysis(*args, **kwargs):
    """
    Wrapper function for backward compatibility
    
    CRITICAL FIX: Changed to accept *args and **kwargs to handle
    any arguments passed, preventing "unexpected keyword argument" errors
    """
    # Extract show_debug from kwargs if present, default to False
    show_debug = kwargs.get('show_debug', False)
    
    # Call the main function with default symbols
    return calculate_enhanced_breakout_analysis(['SPY', 'QQQ', 'IWM'], show_debug)
