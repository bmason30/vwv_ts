"""
Simple Volatility Analysis Module for VWV Trading System
Working baseline implementation without complex features
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def calculate_simple_volatility_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate simple volatility analysis that actually works"""
    try:
        if len(data) < 20:
            return {
                'error': 'Insufficient data for volatility analysis',
                'volatility_regime': 'Unknown',
                'volatility_10d': 20.0,
                'volatility_20d': 20.0,
                'volatility_percentile': 50.0
            }

        # Calculate returns
        returns = data['Close'].pct_change().dropna()
        
        if len(returns) < 20:
            return {
                'error': 'Insufficient return data',
                'volatility_regime': 'Unknown',
                'volatility_10d': 20.0,
                'volatility_20d': 20.0,
                'volatility_percentile': 50.0
            }
        
        # 10-day volatility (annualized)
        volatility_10d = float(returns.tail(10).std() * np.sqrt(252) * 100)
        
        # 20-day volatility (annualized)  
        volatility_20d = float(returns.tail(20).std() * np.sqrt(252) * 100)
        
        # Calculate volatility percentile (current vs historical)
        if len(returns) >= 60:
            # Use 60-day rolling volatility for percentile calculation
            rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100
            rolling_vol = rolling_vol.dropna()
            
            if len(rolling_vol) > 1:
                current_vol = rolling_vol.iloc[-1]
                volatility_percentile = float((rolling_vol <= current_vol).sum() / len(rolling_vol) * 100)
            else:
                volatility_percentile = 50.0
        else:
            volatility_percentile = 50.0
        
        # Determine volatility regime based on percentile and absolute levels
        if volatility_percentile >= 80 or volatility_20d >= 40:
            volatility_regime = "Very High"
        elif volatility_percentile >= 65 or volatility_20d >= 30:
            volatility_regime = "High"
        elif volatility_percentile >= 35 and volatility_20d >= 15:
            volatility_regime = "Normal"
        elif volatility_percentile >= 20 or volatility_20d >= 10:
            volatility_regime = "Low"
        else:
            volatility_regime = "Very Low"
        
        return {
            'volatility_10d': round(volatility_10d, 1),
            'volatility_20d': round(volatility_20d, 1),
            'volatility_percentile': round(volatility_percentile, 0),
            'volatility_regime': volatility_regime,
            'analysis_success': True
        }
        
    except Exception as e:
        logger.error(f"Simple volatility analysis error: {e}")
        return {
            'error': f'Volatility analysis failed: {str(e)}',
            'volatility_regime': 'Unknown',
            'volatility_10d': 20.0,
            'volatility_20d': 20.0,
            'volatility_percentile': 50.0
        }

@safe_calculation_wrapper        
def calculate_market_wide_volume_analysis(show_debug=False) -> Dict[str, Any]:
    """Calculate market-wide volume environment across SPY, QQQ, IWM"""
    try:
        import yfinance as yf
        
        major_indices = ['SPY', 'QQQ', 'IWM']
        volume_data = {}
        
        for symbol in major_indices:
            try:
                # Fetch 3 months of data for volume analysis
                ticker_data = yf.download(symbol, period='3mo', progress=False)
                
                if not ticker_data.empty and len(ticker_data) > 30:
                    # Calculate volume analysis for each index
                    volume_analysis = calculate_complete_volume_analysis(ticker_data)
                    if 'error' not in volume_analysis:
                        volume_data[symbol] = {
                            'volume_score': volume_analysis.get('volume_score', 50),
                            'volume_regime': volume_analysis.get('volume_regime', 'Unknown'),
                            'volume_ratio': volume_analysis.get('volume_ratio', 1.0),
                            'current_volume': volume_analysis.get('current_volume', 0),
                            'volume_30d_avg': volume_analysis.get('volume_30d_avg', 0)
                        }
                    else:
                        volume_data[symbol] = {'error': volume_analysis['error']}
                        
            except Exception as e:
                if show_debug:
                    st.write(f"Error fetching {symbol} volume data: {e}")
                volume_data[symbol] = {'error': f'Data fetch failed: {str(e)}'}
        
        # Calculate market-wide volume environment
        valid_scores = [data['volume_score'] for data in volume_data.values() 
                       if 'volume_score' in data and 'error' not in data]
        
        if valid_scores:
            market_volume_score = sum(valid_scores) / len(valid_scores)
            
            # Determine market volume regime
            if market_volume_score >= 75:
                market_volume_regime = "High Volume Environment"
            elif market_volume_score >= 60:
                market_volume_regime = "Above Average Volume"
            elif market_volume_score >= 40:
                market_volume_regime = "Normal Volume"
            elif market_volume_score >= 25:
                market_volume_regime = "Below Average Volume"
            else:
                market_volume_regime = "Low Volume Environment"
        else:
            market_volume_score = 50
            market_volume_regime = "Unknown Volume Environment"
            
        return {
            'market_volume_score': round(market_volume_score, 1),
            'market_volume_regime': market_volume_regime,
            'individual_analysis': volume_data,
            'analysis_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        logger.error(f"Market-wide volume analysis error: {e}")
        return {
            'error': f'Market-wide volume analysis failed: {str(e)}',
            'market_volume_regime': 'Unknown',
            'market_volume_score': 50
        }
