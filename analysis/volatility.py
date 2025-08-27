"""
File: analysis/volatility.py
Advanced Volatility Analysis Module for VWV Trading System
Version: v4.2.1-COMPLETE-MISSING-MODULE-2025-08-27-19-00-00-EST
The actual advanced volatility analysis file that was missing from deployment
Last Updated: August 27, 2025 - 7:00 PM EST
"""
import pandas as pd
import numpy as np
import streamlit as st
import logging
from typing import Dict, Any, Optional
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

# Research-based weights for volatility indicators (sum = 1.0)
VOLATILITY_INDICATOR_WEIGHTS = {
    'historical_20d': 0.15,      # Primary volatility measure
    'historical_10d': 0.12,      # Short-term volatility
    'realized_vol': 0.13,        # Actual price movements
    'volatility_percentile': 0.11, # Relative positioning
    'volatility_rank': 0.09,     # Ranking system
    'garch_vol': 0.08,          # Advanced modeling
    'parkinson_vol': 0.07,      # High-low range based
    'garman_klass_vol': 0.06,   # OHLC-based estimator
    'rogers_satchell_vol': 0.05, # Drift-independent
    'yang_zhang_vol': 0.04,     # Combined estimators
    'volatility_of_volatility': 0.03, # Second-order volatility
    'volatility_momentum': 0.03, # Rate of change
    'volatility_mean_reversion': 0.02, # Mean reversion tendency
    'volatility_clustering': 0.02  # Persistence analysis
}

@safe_calculation_wrapper
def calculate_complete_volatility_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive volatility analysis with 14 indicators and weighted composite scoring"""
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
        
        # Calculate high-low and OHLC data for advanced volatility estimators
        high_low_ratio = np.log(data['High'] / data['Low'])
        close_open_ratio = np.log(data['Close'] / data['Open'])
        
        # 1. Historical Volatility (20D) - Primary measure
        volatility_20d = returns.rolling(20).std() * np.sqrt(252) * 100
        current_vol_20d = float(volatility_20d.iloc[-1]) if not pd.isna(volatility_20d.iloc[-1]) else 20.0
        
        # 2. Historical Volatility (10D) - Short-term
        volatility_10d = returns.rolling(10).std() * np.sqrt(252) * 100
        current_vol_10d = float(volatility_10d.iloc[-1]) if not pd.isna(volatility_10d.iloc[-1]) else 20.0
        
        # 3. Realized Volatility - Actual price movements
        realized_vol = np.sqrt(np.sum(returns.iloc[-20:] ** 2)) * np.sqrt(252) * 100 if len(returns) >= 20 else current_vol_20d
        
        # 4. Volatility Percentile - Position relative to historical range
        vol_percentile = (volatility_20d.iloc[-1] / volatility_20d.quantile(0.95)) * 100 if len(volatility_20d.dropna()) > 10 else 50.0
        vol_percentile = min(100, max(0, vol_percentile))
        
        # 5. Volatility Rank - Ranking system
        vol_rank = (volatility_20d.rank(pct=True).iloc[-1] * 100) if len(volatility_20d.dropna()) > 10 else 50.0
        
        # 6. GARCH Volatility (simplified implementation)
        try:
            # Simple GARCH(1,1) approximation
            alpha, beta = 0.1, 0.85  # Typical GARCH parameters
            garch_vol = current_vol_20d
            if len(returns) >= 5:
                for i in range(min(5, len(returns))):
                    garch_vol = np.sqrt(alpha * (returns.iloc[-(i+1)] * 100) ** 2 + beta * (garch_vol ** 2))
        except:
            garch_vol = current_vol_20d
            
        # 7. Parkinson Volatility - High-Low range based
        parkinson_vol = np.sqrt(np.mean(high_low_ratio.iloc[-20:] ** 2) * 252 / (4 * np.log(2))) * 100 if len(high_low_ratio) >= 20 else current_vol_20d
        
        # 8. Garman-Klass Volatility - OHLC-based estimator
        try:
            gk_component1 = 0.5 * (high_low_ratio ** 2)
            gk_component2 = (2 * np.log(2) - 1) * (close_open_ratio ** 2)
            garman_klass_vol = np.sqrt(np.mean((gk_component1 - gk_component2).iloc[-20:]) * 252) * 100 if len(data) >= 20 else current_vol_20d
        except:
            garman_klass_vol = current_vol_20d
            
        # 9. Rogers-Satchell Volatility - Drift-independent
        try:
            log_high_close = np.log(data['High'] / data['Close'])
            log_high_open = np.log(data['High'] / data['Open'])
            log_low_close = np.log(data['Low'] / data['Close'])
            log_low_open = np.log(data['Low'] / data['Open'])
            
            rs_vol = log_high_close * log_high_open + log_low_close * log_low_open
            rogers_satchell_vol = np.sqrt(np.mean(rs_vol.iloc[-20:]) * 252) * 100 if len(rs_vol) >= 20 else current_vol_20d
        except:
            rogers_satchell_vol = current_vol_20d
            
        # 10. Yang-Zhang Volatility - Combined estimators
        try:
            # Simplified Yang-Zhang implementation
            overnight_vol = (np.log(data['Open'] / data['Close'].shift(1)) ** 2).rolling(20).mean() * 252
            rs_vol_component = rs_vol.rolling(20).mean() * 252 if 'rs_vol' in locals() else overnight_vol
            yang_zhang_vol = np.sqrt(overnight_vol.iloc[-1] + rs_vol_component.iloc[-1]) * 100 if len(overnight_vol.dropna()) > 0 else current_vol_20d
        except:
            yang_zhang_vol = current_vol_20d
            
        # 11. Volatility of Volatility - Second-order volatility
        vol_of_vol = volatility_20d.rolling(10).std().iloc[-1] if len(volatility_20d.dropna()) >= 10 else 2.0
        vol_of_vol = vol_of_vol if not pd.isna(vol_of_vol) else 2.0
        
        # 12. Volatility Momentum - Rate of change in volatility
        if len(volatility_20d.dropna()) >= 10:
            vol_momentum = ((volatility_20d.iloc[-1] - volatility_20d.iloc[-10]) / volatility_20d.iloc[-10] * 100)
            vol_momentum = vol_momentum if not pd.isna(vol_momentum) else 0.0
        else:
            vol_momentum = 0.0
            
        # 13. Volatility Mean Reversion - Tendency to revert to mean
        if len(volatility_20d.dropna()) >= 20:
            vol_mean = volatility_20d.mean()
            vol_mean_reversion = (vol_mean - volatility_20d.iloc[-1]) / vol_mean * 100
            vol_mean_reversion = vol_mean_reversion if not pd.isna(vol_mean_reversion) else 0.0
        else:
            vol_mean_reversion = 0.0
            
        # 14. Volatility Clustering - Persistence analysis
        if len(volatility_20d.dropna()) >= 5:
            vol_clustering = volatility_20d.rolling(5).corr(volatility_20d.shift(1)).iloc[-1]
            vol_clustering = vol_clustering if not pd.isna(vol_clustering) else 0.5
        else:
            vol_clustering = 0.5
            
        # Create indicators dictionary for scoring
        indicators = {
            'historical_20d': current_vol_20d,
            'historical_10d': current_vol_10d,
            'realized_vol': realized_vol,
            'volatility_percentile': vol_percentile,
            'volatility_rank': vol_rank,
            'garch_vol': garch_vol,
            'parkinson_vol': parkinson_vol,
            'garman_klass_vol': garman_klass_vol,
            'rogers_satchell_vol': rogers_satchell_vol,
            'yang_zhang_vol': yang_zhang_vol,
            'volatility_of_volatility': vol_of_vol,
            'volatility_momentum': vol_momentum,
            'volatility_mean_reversion': vol_mean_reversion,
            'volatility_clustering': vol_clustering
        }
        
        # Calculate individual scores for each indicator (0-100 scale)
        scores = {}
        
        # Score volatility measures (higher volatility = higher score up to a point)
        vol_measures = ['historical_20d', 'historical_10d', 'realized_vol', 'garch_vol', 
                       'parkinson_vol', 'garman_klass_vol', 'rogers_satchell_vol', 'yang_zhang_vol']
        
        for measure in vol_measures:
            val = indicators[measure]
            if val <= 15:  # Low volatility
                scores[measure] = (val / 15) * 30  # 0-30 range
            elif val <= 25:  # Normal volatility
                scores[measure] = 30 + ((val - 15) / 10) * 40  # 30-70 range
            elif val <= 40:  # High volatility
                scores[measure] = 70 + ((val - 25) / 15) * 25  # 70-95 range
            else:  # Extreme volatility
                scores[measure] = 95 + min(5, (val - 40) / 20 * 5)  # 95-100 range
                
        # Score percentile and rank directly
        scores['volatility_percentile'] = min(100, max(0, vol_percentile))
        scores['volatility_rank'] = min(100, max(0, vol_rank))
        
        # Score volatility of volatility (lower is better for stability)
        scores['volatility_of_volatility'] = max(0, 100 - (vol_of_vol * 10))
        
        # Score momentum (positive momentum gets higher score)
        scores['volatility_momentum'] = max(0, min(100, 50 + vol_momentum))
        
        # Score mean reversion (stronger mean reversion = higher score)
        scores['volatility_mean_reversion'] = max(0, min(100, 50 + vol_mean_reversion))
        
        # Score clustering (moderate clustering is optimal)
        clustering_optimal = 0.7
        clustering_diff = abs(vol_clustering - clustering_optimal)
        scores['volatility_clustering'] = max(0, 100 - (clustering_diff * 100))
        
        # Calculate weighted composite score
        composite_score = 0
        total_weight = 0
        contributions = {}
        
        for indicator, weight in VOLATILITY_INDICATOR_WEIGHTS.items():
            if indicator in scores:
                contribution = scores[indicator] * weight
                composite_score += contribution
                contributions[indicator] = contribution
                total_weight += weight
                
        # Normalize score if weights don't sum to 1.0
        if total_weight > 0:
            composite_score = composite_score / total_weight
        else:
            composite_score = 50.0
            
        # Determine volatility regime based on composite score and percentile
        if vol_percentile >= 80:
            volatility_regime = "Extreme High Volatility"
            options_strategy = "Sell Premium (Strangles/Straddles)"
        elif vol_percentile >= 65:
            volatility_regime = "High Volatility"
            options_strategy = "Sell Premium (Iron Condors)"
        elif vol_percentile >= 35:
            volatility_regime = "Normal Volatility"
            options_strategy = "Directional Strategies"
        elif vol_percentile >= 20:
            volatility_regime = "Low Volatility"
            options_strategy = "Buy Premium (Long Options)"
        else:
            volatility_regime = "Extremely Low Volatility"
            options_strategy = "Buy Premium (Calendar Spreads)"
            
        # Advanced Metrics
        
        # Volatility acceleration (trend in volatility)
        if len(volatility_20d.dropna()) >= 10:
            vol_5d_ago = volatility_20d.iloc[-5] if len(volatility_20d) >= 5 else current_vol_20d
            vol_10d_ago = volatility_20d.iloc[-10]
            vol_acceleration = ((current_vol_20d - vol_5d_ago) - (vol_5d_ago - vol_10d_ago))
        else:
            vol_acceleration = 0.0
            
        # Volatility consistency (coefficient of variation)
        vol_cv = volatility_20d.tail(30).std() / volatility_20d.tail(30).mean() * 100 if len(volatility_20d.dropna()) >= 30 and volatility_20d.tail(30).mean() > 0 else 0
        
        # Volatility strength factor for technical scoring (0.85 to 1.3)
        if composite_score >= 80:
            vol_strength_factor = 1.3
        elif composite_score >= 65:
            vol_strength_factor = 1.15  
        elif composite_score >= 35:
            vol_strength_factor = 1.0
        else:
            vol_strength_factor = 0.85
            
        # Risk-adjusted return calculation
        if len(returns) >= 20:
            recent_returns = returns.tail(20).mean() * 252 * 100  # Annualized return
            risk_adjusted_return = recent_returns / current_vol_20d if current_vol_20d > 0 else 0
        else:
            risk_adjusted_return = 0.0
            
        # Generate trading implications
        if composite_score >= 75:
            trading_implications = f"High volatility environment. Consider premium selling strategies. Risk management is crucial. Expect larger price swings."
        elif composite_score >= 60:
            trading_implications = f"Above-average volatility. Suitable for swing trading. Monitor position sizing."
        elif composite_score >= 40:
            trading_implications = f"Normal volatility environment. Standard trading approaches applicable."
        elif composite_score >= 25:
            trading_implications = f"Low volatility environment. Consider premium buying strategies. Breakouts may be more significant."
        else:
            trading_implications = f"Very low volatility. Potential for volatility expansion. Consider long options strategies."
            
        return {
            # Core metrics
            'volatility_20d': round(current_vol_20d, 2),
            'volatility_10d': round(current_vol_10d, 2),
            'realized_volatility': round(realized_vol, 2),
            'volatility_percentile': round(vol_percentile, 1),
            'volatility_rank': round(vol_rank, 1),
            'volatility_momentum': round(vol_momentum, 2),
            'volatility_mean_reversion': round(vol_mean_reversion, 2),
            'volatility_clustering': round(vol_clustering, 3),
            'volatility_of_volatility': round(vol_of_vol, 2),
            
            # Advanced estimators
            'garch_volatility': round(garch_vol, 2),
            'parkinson_volatility': round(parkinson_vol, 2),
            'garman_klass_volatility': round(garman_klass_vol, 2),
            'rogers_satchell_volatility': round(rogers_satchell_vol, 2),
            'yang_zhang_volatility': round(yang_zhang_vol, 2),
            
            # Advanced analytics
            'volatility_acceleration': round(vol_acceleration, 2),
            'volatility_consistency': round(float(vol_cv), 2),
            'risk_adjusted_return': round(risk_adjusted_return, 2),
            'volatility_strength_factor': vol_strength_factor,
            
            # Composite analysis
            'volatility_score': round(composite_score, 1),
            'volatility_regime': volatility_regime,
            'options_strategy': options_strategy,
            'trading_implications': trading_implications,
            
            # Component breakdown for UI display
            'indicators': indicators,
            'scores': scores,
            'weights': VOLATILITY_INDICATOR_WEIGHTS,
            'contributions': contributions,
            'analysis_success': True
        }
        
    except Exception as e:
        logger.error(f"Error in volatility analysis: {e}")
        return {
            'error': f'Volatility analysis failed: {str(e)}',
            'volatility_regime': 'Unknown',
            'volatility_score': 50,
            'volatility_20d': 20.0,
            'volatility_10d': 20.0,
            'realized_volatility': 20.0,
            'volatility_percentile': 50.0,
            'volatility_rank': 50.0,
            'volatility_strength_factor': 1.0
        }

@safe_calculation_wrapper  
def calculate_market_wide_volatility_analysis(symbols=['SPY', 'QQQ', 'IWM'], show_debug=False) -> Dict[str, Any]:
    """Calculate market-wide volatility environment across major indices"""
    try:
        import yfinance as yf
        
        market_volatility_data = {}
        
        for symbol in symbols:
            try:
                if show_debug:
                    st.write(f"Fetching volatility data for {symbol}...")
                    
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='3mo')  # 3 months for better volatility analysis
                
                if len(data) >= 30:
                    volatility_analysis = calculate_complete_volatility_analysis(data)
                    if 'error' not in volatility_analysis:
                        market_volatility_data[symbol] = volatility_analysis
                        
            except Exception as e:
                if show_debug:
                    st.write(f"Error fetching {symbol}: {e}")
                continue
                
        if len(market_volatility_data) >= 2:
            # Calculate overall market volatility environment
            avg_volatility_score = sum([data['volatility_score'] for data in market_volatility_data.values()]) / len(market_volatility_data)
            avg_volatility = sum([data['volatility_20d'] for data in market_volatility_data.values()]) / len(market_volatility_data)
            
            # Classify market volatility environment
            if avg_volatility >= 35:
                market_volatility_environment = "High Volatility Market"
            elif avg_volatility >= 25:
                market_volatility_environment = "Above Normal Volatility"
            elif avg_volatility >= 15:
                market_volatility_environment = "Normal Volatility"
            else:
                market_volatility_environment = "Low Volatility Market"
                
            return {
                'market_indices': market_volatility_data,
                'average_volatility_score': round(avg_volatility_score, 1),
                'average_volatility': round(avg_volatility, 2),
                'market_volatility_environment': market_volatility_environment,
                'market_analysis_success': True
            }
        else:
            return {
                'error': 'Insufficient market data for market-wide analysis',
                'market_analysis_success': False
            }
            
    except Exception as e:
        logger.error(f"Error in market-wide volatility analysis: {e}")
        return {
            'error': f'Market volatility analysis failed: {str(e)}',
            'market_analysis_success': False
        }

def get_volatility_trading_implications(vol_regime: str, vol_percentile: float) -> str:
    """Generate detailed trading implications based on volatility analysis"""
    
    if vol_percentile >= 80:
        return "Extreme volatility environment. High premium collection opportunities for sellers. Increased risk of gap moves. Consider wide iron condors or short strangles with adequate margin."
    elif vol_percentile >= 65:
        return "High volatility regime. Favorable for premium selling strategies. Iron condors and credit spreads may perform well. Monitor gamma risk closely."
    elif vol_percentile >= 35:
        return "Normal volatility conditions. Standard technical analysis applies. Consider directional strategies based on trend analysis. Options premiums fairly valued."
    elif vol_percentile >= 20:
        return "Low volatility environment. Premium buying opportunities may emerge. Long options strategies become more attractive. Watch for volatility expansion signals."
    else:
        return "Extremely low volatility. High probability of volatility mean reversion. Consider long straddles or calendar spreads. Breakout potential increases."
