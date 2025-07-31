"""
Volatility analysis module for VWV Trading System v4.2.1
Complete Volatility Analysis with 5D/30D rolling metrics and regime detection
"""
import pandas as pd
import numpy as np
import streamlit as st
import logging
from typing import Dict, Any, Optional
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def calculate_complete_volatility_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate complete volatility analysis with 5D/30D rolling metrics and regime detection"""
    try:
        if len(data) < 30:
            return {
                'error': 'Insufficient data for volatility analysis',
                'volatility_regime': 'Unknown',
                'volatility_score': 50
            }

        # Calculate returns
        returns = data['Close'].pct_change().dropna()
        
        if len(returns) < 30:
            return {
                'error': 'Insufficient return data for volatility analysis',
                'volatility_regime': 'Unknown', 
                'volatility_score': 50
            }
        
        # 5-Day Rolling Volatility (annualized)
        volatility_5d = returns.rolling(5).std() * np.sqrt(252) * 100  # Annualized percentage
        current_vol_5d = float(volatility_5d.iloc[-1]) if not pd.isna(volatility_5d.iloc[-1]) else 20.0
        
        # 30-Day Rolling Volatility (annualized)
        volatility_30d = returns.rolling(30).std() * np.sqrt(252) * 100
        current_vol_30d = float(volatility_30d.iloc[-1]) if not pd.isna(volatility_30d.iloc[-1]) else 20.0
        
        # Volatility ratio (5D vs 30D)
        vol_ratio = current_vol_5d / current_vol_30d if current_vol_30d > 0 else 1.0
        
        # Volatility trend over last 5 days
        if len(volatility_5d) >= 5:
            vol_5d_prev = float(volatility_5d.iloc[-5]) if not pd.isna(volatility_5d.iloc[-5]) else current_vol_5d
            vol_trend = (current_vol_5d - vol_5d_prev) / vol_5d_prev * 100 if vol_5d_prev > 0 else 0
        else:
            vol_trend = 0.0
            
        # Historical volatility percentile (using 60-day lookback)
        if len(volatility_30d) >= 60:
            vol_60d = volatility_30d.tail(60)
            vol_percentile = (vol_60d <= current_vol_30d).sum() / len(vol_60d) * 100
        else:
            vol_percentile = 50.0
            
        # Volatility Regime Classification
        if current_vol_30d >= 50:
            vol_regime = "Extreme High"
            vol_score = 95
            options_strategy = "Aggressive Premium Selling"
        elif current_vol_30d >= 35:
            vol_regime = "High"  
            vol_score = 80
            options_strategy = "Premium Selling"
        elif current_vol_30d >= 25:
            vol_regime = "Above Normal"
            vol_score = 65
            options_strategy = "Slight Selling Bias"
        elif current_vol_30d >= 15:
            vol_regime = "Normal"
            vol_score = 50
            options_strategy = "Neutral"
        elif current_vol_30d >= 10:
            vol_regime = "Below Normal"
            vol_score = 35
            options_strategy = "Slight Buying Bias"
        else:
            vol_regime = "Low"
            vol_score = 20
            options_strategy = "Premium Buying"
            
        # Risk-adjusted returns (Sharpe-like ratio)
        if current_vol_30d > 0:
            mean_return = returns.tail(30).mean() * 252 * 100  # Annualized return %
            risk_adjusted_return = mean_return / current_vol_30d
        else:
            risk_adjusted_return = 0.0
            
        # Volatility acceleration (rate of change in volatility)
        if len(volatility_30d) >= 60:
            prev_vol_30d = float(volatility_30d.iloc[-60]) if not pd.isna(volatility_30d.iloc[-60]) else current_vol_30d
            vol_acceleration = (current_vol_30d - prev_vol_30d) / prev_vol_30d * 100 if prev_vol_30d > 0 else 0
        else:
            vol_acceleration = 0.0
            
        # Volatility consistency (coefficient of variation of volatility)
        vol_cv = volatility_30d.tail(30).std() / volatility_30d.tail(30).mean() * 100 if volatility_30d.tail(30).mean() > 0 else 0
        
        # Volatility strength factor for technical scoring (0.85 to 1.3)
        if vol_score >= 80:
            vol_strength_factor = 1.3
        elif vol_score >= 65:
            vol_strength_factor = 1.15  
        elif vol_score >= 35:
            vol_strength_factor = 1.0
        else:
            vol_strength_factor = 0.85
            
        return {
            'volatility_5d': round(current_vol_5d, 2),
            'volatility_30d': round(current_vol_30d, 2),
            'volatility_ratio': round(vol_ratio, 2),
            'volatility_trend': round(vol_trend, 2),
            'volatility_percentile': round(vol_percentile, 1),
            'volatility_regime': vol_regime,
            'volatility_score': vol_score,
            'options_strategy': options_strategy,
            'risk_adjusted_return': round(risk_adjusted_return, 2),
            'volatility_acceleration': round(vol_acceleration, 2),
            'volatility_consistency': round(float(vol_cv), 2),
            'volatility_strength_factor': vol_strength_factor,
            'trading_implications': get_volatility_trading_implications(vol_regime, vol_percentile)
        }
        
    except Exception as e:
        logger.error(f"Volatility analysis calculation error: {e}")
        return {
            'error': f'Volatility analysis failed: {str(e)}',
            'volatility_regime': 'Unknown',
            'volatility_score': 50,
            'volatility_strength_factor': 1.0
        }

@safe_calculation_wrapper        
def calculate_market_wide_volatility_analysis(show_debug=False) -> Dict[str, Any]:
    """Calculate market-wide volatility environment across SPY, QQQ, IWM"""
    try:
        import yfinance as yf
        
        major_indices = ['SPY', 'QQQ', 'IWM']
        market_volatility_data = {}
        
        for symbol in major_indices:
            try:
                if show_debug:
                    st.write(f"📊 Fetching volatility data for {symbol}...")
                    
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='3mo')  # 3 months for better volatility analysis
                
                if len(data) >= 30:
                    volatility_analysis = calculate_complete_volatility_analysis(data)
                    if 'error' not in volatility_analysis:
                        market_volatility_data[symbol] = volatility_analysis
                        
            except Exception as e:
                if show_debug:
                    st.write(f"❌ Error fetching {symbol}: {e}")
                continue
                
        if len(market_volatility_data) >= 2:
            # Calculate overall market volatility environment
            avg_volatility_score = sum([data['volatility_score'] for data in market_volatility_data.values()]) / len(market_volatility_data)
            avg_volatility = sum([data['volatility_30d'] for data in market_volatility_data.values()]) / len(market_volatility_data)
            
            # Classify market volatility environment
            if avg_volatility >= 35:
                market_volatility_environment = "⚡ High Volatility Market"
            elif avg_volatility >= 25:
                market_volatility_environment = "🌊 Above Normal Volatility"
            elif avg_volatility >= 15:
                market_volatility_environment = "⚖️ Normal Volatility"
            else:
                market_volatility_environment = "😴 Low Volatility Market"
                
            return {
                'market_indices': market_volatility_data,
                'average_volatility_score': round(avg_volatility_score, 1),
                'average_volatility': round(avg_volatility, 2),
                'market_volatility_environment': market_volatility_environment,
                'sample_size': len(market_volatility_data)
            }
        else:
            return {'error': 'Insufficient market data for volatility environment analysis'}
            
    except Exception as e:
        logger.error(f"Market-wide volatility analysis error: {e}")
        return {'error': f'Market volatility analysis failed: {str(e)}'}

def get_volatility_trading_implications(volatility_regime: str, percentile: float) -> str:
    """Get trading implications based on volatility analysis"""
    if volatility_regime == "Extreme High":
        return "🚨 Extreme volatility - Aggressive premium selling, tight risk management"
    elif volatility_regime == "High":
        if percentile > 80:
            return "⚡ High volatility (80th+ percentile) - Excellent premium selling opportunities"
        else:
            return "📈 High volatility - Good for premium selling strategies"
    elif volatility_regime == "Above Normal":
        return "📊 Elevated volatility - Moderate premium selling bias"
    elif volatility_regime == "Below Normal":
        return "⚠️ Low volatility - Consider premium buying or wait for expansion"
    elif volatility_regime == "Low":
        return "😴 Very low volatility - Premium buying opportunities, volatility expansion likely"
    else:
        return "⚖️ Normal volatility environment - Standard options strategies"

def get_volatility_regime_color(volatility_regime: str) -> str:
    """Get color coding for volatility regime display"""
    color_map = {
        "Extreme High": "#DC143C",    # Crimson
        "High": "#FF4500",            # Orange Red
        "Above Normal": "#FF8C00",    # Dark Orange
        "Normal": "#32CD32",          # Lime Green
        "Below Normal": "#87CEEB",    # Sky Blue
        "Low": "#4169E1",             # Royal Blue
        "Unknown": "#808080"          # Gray
    }
    return color_map.get(volatility_regime, "#808080")
