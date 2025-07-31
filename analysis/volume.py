"""
Volume analysis module for VWV Trading System v4.2.1
Complete Volume Analysis with 5D/30D rolling metrics and regime detection
"""
import pandas as pd
import numpy as np
import streamlit as st
import logging
from typing import Dict, Any, Optional
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def calculate_complete_volume_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate complete volume analysis with 5D/30D rolling metrics and regime detection"""
    try:
        if len(data) < 30:
            return {
                'error': 'Insufficient data for volume analysis',
                'volume_regime': 'Unknown',
                'volume_score': 50
            }

        volume = data['Volume']
        current_volume = float(volume.iloc[-1])
        
        # 5-Day Rolling Volume Analysis
        volume_5d = volume.rolling(5).mean()
        current_5d_avg = float(volume_5d.iloc[-1]) if not pd.isna(volume_5d.iloc[-1]) else current_volume
        
        # Volume trend over last 5 days
        if len(volume_5d) >= 5:
            volume_5d_trend = (current_5d_avg - float(volume_5d.iloc[-5])) / float(volume_5d.iloc[-5]) * 100
        else:
            volume_5d_trend = 0.0
            
        # 30-Day Volume Comparison
        volume_30d = volume.rolling(30).mean()
        volume_30d_avg = float(volume_30d.iloc[-1]) if not pd.isna(volume_30d.iloc[-1]) else current_volume
        
        # Volume ratio (current vs 30-day average)
        volume_ratio = current_volume / volume_30d_avg if volume_30d_avg > 0 else 1.0
        
        # Volume Z-Score for breakout detection
        volume_std = volume.rolling(30).std().iloc[-1]
        if not pd.isna(volume_std) and volume_std > 0:
            volume_zscore = (current_volume - volume_30d_avg) / volume_std
        else:
            volume_zscore = 0.0
            
        # Volume Regime Classification
        if volume_ratio >= 2.0:
            volume_regime = "Extreme High"
            volume_score = 95
        elif volume_ratio >= 1.5:
            volume_regime = "High"
            volume_score = 80
        elif volume_ratio >= 1.2:
            volume_regime = "Above Normal"
            volume_score = 65
        elif volume_ratio >= 0.8:
            volume_regime = "Normal"
            volume_score = 50
        elif volume_ratio >= 0.5:
            volume_regime = "Below Normal"
            volume_score = 35
        else:
            volume_regime = "Low"
            volume_score = 20
            
        # Volume breakout detection
        volume_breakout = "None"
        if abs(volume_zscore) >= 2.0:
            volume_breakout = "Extreme" if volume_zscore > 0 else "Extreme Collapse"
        elif abs(volume_zscore) >= 1.5:
            volume_breakout = "Strong" if volume_zscore > 0 else "Strong Decline"
        elif abs(volume_zscore) >= 1.0:
            volume_breakout = "Moderate" if volume_zscore > 0 else "Moderate Decline"
            
        # Volume acceleration (rate of change)
        if len(volume_5d) >= 10:
            prev_5d = float(volume_5d.iloc[-10])
            volume_acceleration = (current_5d_avg - prev_5d) / prev_5d * 100 if prev_5d > 0 else 0
        else:
            volume_acceleration = 0.0
            
        # Volume consistency (coefficient of variation)
        volume_cv = (volume_std / volume_30d_avg) * 100 if volume_30d_avg > 0 and not pd.isna(volume_std) else 0
        
        # Volume strength factor for technical scoring (0.85 to 1.3)
        if volume_score >= 80:
            volume_strength_factor = 1.3
        elif volume_score >= 65:
            volume_strength_factor = 1.15
        elif volume_score >= 35:
            volume_strength_factor = 1.0
        else:
            volume_strength_factor = 0.85
            
        return {
            'current_volume': int(current_volume),
            'volume_5d_avg': int(current_5d_avg),
            'volume_30d_avg': int(volume_30d_avg),
            'volume_ratio': round(volume_ratio, 2),
            'volume_5d_trend': round(volume_5d_trend, 2),
            'volume_zscore': round(float(volume_zscore), 2),
            'volume_regime': volume_regime,
            'volume_score': volume_score,
            'volume_breakout': volume_breakout,
            'volume_acceleration': round(volume_acceleration, 2),
            'volume_consistency': round(volume_cv, 2),
            'volume_strength_factor': volume_strength_factor,
            'trading_implications': get_volume_trading_implications(volume_regime, volume_breakout)
        }
        
    except Exception as e:
        logger.error(f"Volume analysis calculation error: {e}")
        return {
            'error': f'Volume analysis failed: {str(e)}',
            'volume_regime': 'Unknown',
            'volume_score': 50,
            'volume_strength_factor': 1.0
        }

@safe_calculation_wrapper        
def calculate_market_wide_volume_analysis(show_debug=False) -> Dict[str, Any]:
    """Calculate market-wide volume environment across SPY, QQQ, IWM"""
    try:
        import yfinance as yf
        
        major_indices = ['SPY', 'QQQ', 'IWM']
        market_volume_data = {}
        
        for symbol in major_indices:
            try:
                if show_debug:
                    st.write(f"📊 Fetching volume data for {symbol}...")
                    
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='3mo')  # 3 months for better volume analysis
                
                if len(data) >= 30:
                    volume_analysis = calculate_complete_volume_analysis(data)
                    if 'error' not in volume_analysis:
                        market_volume_data[symbol] = volume_analysis
                        
            except Exception as e:
                if show_debug:
                    st.write(f"❌ Error fetching {symbol}: {e}")
                continue
                
        if len(market_volume_data) >= 2:
            # Calculate overall market volume environment
            avg_volume_score = sum([data['volume_score'] for data in market_volume_data.values()]) / len(market_volume_data)
            
            # Classify market volume environment
            if avg_volume_score >= 80:
                market_volume_environment = "🔥 High Activity Market"
            elif avg_volume_score >= 65:
                market_volume_environment = "📈 Above Normal Activity"
            elif avg_volume_score >= 35:
                market_volume_environment = "⚖️ Normal Activity"
            else:
                market_volume_environment = "😴 Low Activity Market"
                
            return {
                'market_indices': market_volume_data,
                'average_volume_score': round(avg_volume_score, 1),
                'market_volume_environment': market_volume_environment,
                'sample_size': len(market_volume_data)
            }
        else:
            return {'error': 'Insufficient market data for environment analysis'}
            
    except Exception as e:
        logger.error(f"Market-wide volume analysis error: {e}")
        return {'error': f'Market volume analysis failed: {str(e)}'}

def get_volume_trading_implications(volume_regime: str, volume_breakout: str) -> str:
    """Get trading implications based on volume analysis"""
    if volume_regime == "Extreme High":
        if "Extreme" in volume_breakout:
            return "🚀 Major breakout likely - High conviction trades, larger position sizes"
        else:
            return "⚡ High activity environment - Good for momentum trades"
    elif volume_regime == "High":
        return "📈 Increased activity - Above normal conviction, moderate position increases"
    elif volume_regime == "Above Normal":
        return "📊 Slightly elevated activity - Normal position sizing with slight bias"
    elif volume_regime == "Below Normal":
        return "⚠️ Reduced activity - Lower conviction, reduce position sizes"
    elif volume_regime == "Low":
        return "😴 Low activity environment - Minimal positions, wait for volume confirmation"
    else:
        return "⚖️ Normal volume environment - Standard position sizing and strategy"

def get_volume_regime_color(volume_regime: str) -> str:
    """Get color coding for volume regime display"""
    color_map = {
        "Extreme High": "#FF4500",    # Orange Red
        "High": "#FF8C00",            # Dark Orange  
        "Above Normal": "#FFD700",    # Gold
        "Normal": "#32CD32",          # Lime Green
        "Below Normal": "#87CEEB",    # Sky Blue
        "Low": "#4169E1",             # Royal Blue
        "Unknown": "#808080"          # Gray
    }
    return color_map.get(volume_regime, "#808080")
