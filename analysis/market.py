"""
Market correlation and analysis module for VWV Trading System v4.2.2
Correlation analysis and breakout/breakdown detection
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def calculate_market_correlations_enhanced(data: pd.DataFrame, symbol: str, show_debug: bool = False) -> Dict[str, Any]:
    """
    Calculate enhanced market correlations with major indices
    
    Args:
        data: Price data for the symbol
        symbol: Stock symbol
        show_debug: Whether to show debug information
    
    Returns:
        Dictionary containing correlation analysis
    """
    try:
        import yfinance as yf
        
        # Major market indices to compare against
        market_indices = {
            'SPY': 'S&P 500',
            'QQQ': 'Nasdaq 100',
            'DIA': 'Dow Jones',
            'IWM': 'Russell 2000'
        }
        
        # Additional tech-focused ETFs (with error handling)
        tech_etfs = {
            'MAGS': 'Magnificent 7',
            'FNGU': 'Tech Bull 3X',
            'FNGD': 'Tech Bear 3X'
        }
        
        correlations = {}
        
        # Calculate correlations with major indices
        for index_symbol, index_name in market_indices.items():
            try:
                if index_symbol == symbol:
                    continue
                    
                # Fetch index data
                index_data = yf.download(
                    index_symbol, 
                    start=data.index[0], 
                    end=data.index[-1],
                    progress=False,
                    show_errors=False
                )
                
                # Validate data
                if index_data is None or len(index_data) == 0 or not isinstance(index_data, pd.DataFrame):
                    if show_debug:
                        logger.warning(f"No data returned for {index_symbol}")
                    continue
                
                if 'Close' not in index_data.columns:
                    if show_debug:
                        logger.warning(f"No Close column for {index_symbol}")
                    continue
                
                # Calculate returns
                symbol_returns = data['Close'].pct_change().dropna()
                index_returns = index_data['Close'].pct_change().dropna()
                
                # Align dates
                aligned_data = pd.DataFrame({
                    'symbol': symbol_returns,
                    'index': index_returns
                }).dropna()
                
                if len(aligned_data) < 20:
                    if show_debug:
                        logger.warning(f"Insufficient data for correlation with {index_symbol}")
                    continue
                
                # Calculate correlation
                correlation = aligned_data['symbol'].corr(aligned_data['index'])
                
                # Calculate beta (simplified)
                covariance = aligned_data['symbol'].cov(aligned_data['index'])
                index_variance = aligned_data['index'].var()
                beta = covariance / index_variance if index_variance != 0 else 1.0
                
                correlations[index_symbol] = {
                    'name': index_name,
                    'correlation': round(float(correlation), 3),
                    'beta': round(float(beta), 2)
                }
                
            except Exception as e:
                if show_debug:
                    logger.warning(f"Error fetching {index_symbol} data: {str(e)}")
                continue
        
        # Try to add tech ETFs (with graceful failure)
        for etf_symbol, etf_name in tech_etfs.items():
            try:
                if etf_symbol == symbol:
                    continue
                
                etf_data = yf.download(
                    etf_symbol,
                    start=data.index[0],
                    end=data.index[-1],
                    progress=False,
                    show_errors=False
                )
                
                # Validate data before processing
                if etf_data is None or len(etf_data) == 0 or not isinstance(etf_data, pd.DataFrame):
                    continue
                
                if 'Close' not in etf_data.columns:
                    continue
                
                # Calculate returns
                symbol_returns = data['Close'].pct_change().dropna()
                etf_returns = etf_data['Close'].pct_change().dropna()
                
                # Align dates
                aligned_data = pd.DataFrame({
                    'symbol': symbol_returns,
                    'etf': etf_returns
                }).dropna()
                
                if len(aligned_data) < 20:
                    continue
                
                # Calculate correlation
                correlation = aligned_data['symbol'].corr(aligned_data['etf'])
                
                correlations[etf_symbol] = {
                    'name': etf_name,
                    'correlation': round(float(correlation), 3),
                    'beta': 1.0
                }
                
            except Exception as e:
                # Silently skip tech ETFs that fail
                if show_debug:
                    logger.debug(f"Tech ETF {etf_symbol} not available: {str(e)}")
                continue
        
        return {
            'correlations': correlations,
            'breakout_breakdown': calculate_breakout_breakdown_analysis(data, symbol)
        }
        
    except Exception as e:
        logger.error(f"Market correlation calculation error: {e}")
        return {
            'error': str(e),
            'correlations': {},
            'breakout_breakdown': {}
        }

@safe_calculation_wrapper
def calculate_breakout_breakdown_analysis(data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """
    Analyze potential breakouts and breakdowns
    
    Args:
        data: Price data
        symbol: Stock symbol
    
    Returns:
        Dictionary with breakout/breakdown analysis
    """
    try:
        if len(data) < 50:
            return {}
        
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        # Calculate 20 and 50 day highs/lows
        high_20 = high.rolling(20).max()
        low_20 = low.rolling(20).min()
        high_50 = high.rolling(50).max()
        low_50 = low.rolling(50).min()
        
        current_price = close.iloc[-1]
        
        # Breakout detection (price near highs)
        near_high_20 = (current_price >= high_20.iloc[-1] * 0.98)
        near_high_50 = (current_price >= high_50.iloc[-1] * 0.98)
        
        # Breakdown detection (price near lows)
        near_low_20 = (current_price <= low_20.iloc[-1] * 1.02)
        near_low_50 = (current_price <= low_50.iloc[-1] * 1.02)
        
        # Determine status
        if near_high_50:
            status = "Breakout Candidate - 50D High"
        elif near_high_20:
            status = "Breakout Candidate - 20D High"
        elif near_low_50:
            status = "Breakdown Risk - 50D Low"
        elif near_low_20:
            status = "Breakdown Risk - 20D Low"
        else:
            status = "Neutral Range"
        
        return {
            'status': status,
            'current_price': round(float(current_price), 2),
            'high_20d': round(float(high_20.iloc[-1]), 2),
            'low_20d': round(float(low_20.iloc[-1]), 2),
            'high_50d': round(float(high_50.iloc[-1]), 2),
            'low_50d': round(float(low_50.iloc[-1]), 2),
            'breakouts': [symbol] if near_high_20 or near_high_50 else [],
            'breakdowns': [symbol] if near_low_20 or near_low_50 else []
        }
        
    except Exception as e:
        logger.error(f"Breakout/breakdown analysis error: {e}")
        return {}
