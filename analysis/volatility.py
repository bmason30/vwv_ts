"""
Volatility analysis for VWV Trading System v4.2.1
Comprehensive volatility trend analysis with 5-day and 30-day rolling metrics
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def calculate_volatility_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive volatility analysis with 5-day and 30-day rolling metrics
    
    Returns:
        Dict containing volatility metrics, trends, and regime classification
    """
    try:
        if len(data) < 30:
            return {
                'error': 'Insufficient data for volatility analysis (minimum 30 periods required)',
                'data_points': len(data)
            }

        close = data['Close'].copy()
        
        # Calculate daily returns
        returns = close.pct_change().dropna()
        
        if len(returns) < 30:
            return {
                'error': 'Insufficient returns data for volatility analysis',
                'data_points': len(returns)
            }
        
        # 5-Day Rolling Volatility Analysis (annualized)
        volatility_5d_rolling = returns.rolling(window=5).std() * np.sqrt(252) * 100  # Annualized percentage
        current_5d_vol = volatility_5d_rolling.iloc[-1]
        prev_5d_vol = volatility_5d_rolling.iloc[-2] if len(volatility_5d_rolling) > 1 else current_5d_vol
        
        # 5-Day Volatility Trend Analysis
        vol_5d_trend_direction = "Increasing" if current_5d_vol > prev_5d_vol else "Decreasing"
        vol_5d_trend_strength = abs((current_5d_vol - prev_5d_vol) / prev_5d_vol * 100) if prev_5d_vol > 0 else 0
        
        # Volatility momentum (rate of change over 5 days)
        vol_5d_values = volatility_5d_rolling.tail(5).dropna()
        if len(vol_5d_values) >= 2:
            vol_momentum = (vol_5d_values.iloc[-1] - vol_5d_values.iloc[0]) / vol_5d_values.iloc[0] * 100
        else:
            vol_momentum = 0
        
        # 30-Day Rolling Volatility Analysis
        volatility_30d_rolling = returns.rolling(window=30).std() * np.sqrt(252) * 100  # Annualized percentage
        current_30d_vol = volatility_30d_rolling.iloc[-1]
        
        # 5-Day vs 30-Day Comparison
        vol_ratio_5d_30d = current_5d_vol / current_30d_vol if current_30d_vol > 0 else 1
        vol_deviation_pct = ((current_5d_vol - current_30d_vol) / current_30d_vol * 100) if current_30d_vol > 0 else 0
        
        # Volatility Regime Classification
        if current_5d_vol >= 50:  # Very high volatility
            vol_regime = "Extreme High"
            regime_score = 95
        elif current_5d_vol >= 35:  # High volatility
            vol_regime = "High"
            regime_score = 80
        elif current_5d_vol >= 25:  # Above normal
            vol_regime = "Above Normal"
            regime_score = 65
        elif current_5d_vol >= 15:  # Normal range
            vol_regime = "Normal"
            regime_score = 50
        elif current_5d_vol >= 10:  # Below normal
            vol_regime = "Below Normal"
            regime_score = 35
        else:  # Low volatility
            vol_regime = "Low"
            regime_score = 20
        
        # Volatility Cycle Analysis
        vol_30d_values = volatility_30d_rolling.tail(30).dropna()
        if len(vol_30d_values) >= 10:
            vol_percentile = (vol_30d_values <= current_5d_vol).sum() / len(vol_30d_values) * 100
            
            if vol_percentile >= 90:
                cycle_position = "Peak Volatility"
            elif vol_percentile >= 75:
                cycle_position = "High in Cycle"
            elif vol_percentile >= 25:
                cycle_position = "Mid Cycle"
            elif vol_percentile >= 10:
                cycle_position = "Low in Cycle"
            else:
                cycle_position = "Volatility Trough"
        else:
            cycle_position = "Insufficient Data"
            vol_percentile = 50
        
        # Volatility Breakout Detection
        vol_std_30d = volatility_30d_rolling.rolling(window=30).std().iloc[-1]
        vol_z_score = (current_5d_vol - current_30d_vol) / vol_std_30d if vol_std_30d > 0 else 0
        
        if vol_z_score >= 2.0:
            vol_breakout = "Significant Volatility Spike"
        elif vol_z_score >= 1.5:
            vol_breakout = "Moderate Volatility Increase"
        elif vol_z_score <= -2.0:
            vol_breakout = "Significant Volatility Compression"
        elif vol_z_score <= -1.5:
            vol_breakout = "Moderate Volatility Decrease"
        else:
            vol_breakout = "Normal Volatility Range"
        
        # Options Strategy Adjustment Factor
        # High volatility = higher premiums, low volatility = lower premiums
        if current_5d_vol >= 40:
            options_adjustment = "Sell Premium" # High vol = good for selling
        elif current_5d_vol >= 25:
            options_adjustment = "Neutral Strategy"
        else:
            options_adjustment = "Buy Premium"  # Low vol = good for buying
        
        # Advanced Volatility Metrics
        # Volatility acceleration (change in volatility trend)
        vol_5d_values_extended = volatility_5d_rolling.tail(10).dropna()
        if len(vol_5d_values_extended) >= 5:
            recent_vol_trend = vol_5d_values_extended.tail(3).mean()
            earlier_vol_trend = vol_5d_values_extended.head(3).mean()
            vol_acceleration = (recent_vol_trend - earlier_vol_trend) / earlier_vol_trend * 100 if earlier_vol_trend > 0 else 0
        else:
            vol_acceleration = 0
        
        # Volatility of volatility (how stable the volatility is)
        vol_of_vol = volatility_5d_rolling.tail(10).std() if len(volatility_5d_rolling) >= 10 else 0
        
        # Risk-adjusted return metrics
        current_return = returns.iloc[-1] * 100  # Current day return as percentage
        risk_adjusted_return = current_return / (current_5d_vol / 100) if current_5d_vol > 0 else 0
        
        return {
            # 5-Day Rolling Metrics
            'volatility_5d': round(float(current_5d_vol), 2),
            'vol_5d_trend_direction': vol_5d_trend_direction,
            'vol_5d_trend_strength': round(float(vol_5d_trend_strength), 2),
            'vol_momentum': round(float(vol_momentum), 2),
            
            # 30-Day Comparison Metrics
            'volatility_30d': round(float(current_30d_vol), 2),
            'vol_ratio_5d_30d': round(float(vol_ratio_5d_30d), 2),
            'vol_deviation_pct': round(float(vol_deviation_pct), 2),
            
            # Volatility Regime Analysis
            'vol_regime': vol_regime,
            'regime_score': regime_score,
            'cycle_position': cycle_position,
            'vol_percentile': round(float(vol_percentile), 1),
            'vol_breakout': vol_breakout,
            'vol_z_score': round(float(vol_z_score), 2),
            
            # Options Integration
            'options_adjustment': options_adjustment,
            'vol_composite_score': regime_score,  # For composite technical scoring
            
            # Advanced Metrics
            'vol_acceleration': round(float(vol_acceleration), 2),
            'vol_of_vol': round(float(vol_of_vol), 2),
            'risk_adjusted_return': round(float(risk_adjusted_return), 2),
            'current_return': round(float(current_return), 2),
            
            # Metadata
            'data_points': len(data),
            'returns_count': len(returns),
            'calculation_success': True
        }
        
    except Exception as e:
        logger.error(f"Volatility analysis calculation error: {e}")
        return {
            'error': f'Volatility analysis failed: {str(e)}',
            'data_points': len(data) if data is not None else 0,
            'calculation_success': False
        }

@safe_calculation_wrapper
def get_volatility_interpretation(vol_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Provide human-readable interpretation of volatility analysis results
    """
    try:
        if 'error' in vol_data or not vol_data.get('calculation_success', False):
            return {
                'overall': 'Volatility analysis unavailable',
                'trend': 'Cannot determine trend',
                'regime': 'Cannot assess regime',
                'options': 'Cannot provide options guidance',
                'risk': 'Cannot assess risk level'
            }
        
        regime = vol_data.get('vol_regime', 'Unknown')
        trend_direction = vol_data.get('vol_5d_trend_direction', 'Unknown')
        cycle_position = vol_data.get('cycle_position', 'Unknown')
        options_adjustment = vol_data.get('options_adjustment', 'Unknown')
        
        # Overall interpretation
        volatility_5d = vol_data.get('volatility_5d', 0)
        if regime == 'Extreme High':
            overall = f"Extremely high volatility ({volatility_5d:.1f}%) - crisis or event-driven environment"
        elif regime == 'High':
            overall = f"High volatility ({volatility_5d:.1f}%) - elevated risk environment"
        elif regime == 'Above Normal':
            overall = f"Above normal volatility ({volatility_5d:.1f}%) - increased market stress"
        elif regime == 'Normal':
            overall = f"Normal volatility ({volatility_5d:.1f}%) - typical market conditions"
        elif regime == 'Below Normal':
            overall = f"Below normal volatility ({volatility_5d:.1f}%) - calmer market conditions"
        else:
            overall = f"Low volatility ({volatility_5d:.1f}%) - very stable environment"
        
        # Trend interpretation
        momentum = vol_data.get('vol_momentum', 0)
        if trend_direction == "Increasing" and momentum > 15:
            trend = "Rapidly increasing volatility - market stress building"
        elif trend_direction == "Increasing":
            trend = "Moderately increasing volatility"  
        elif trend_direction == "Decreasing" and momentum < -15:
            trend = "Rapidly decreasing volatility - market calming"
        else:
            trend = "Moderately decreasing volatility"
        
        # Regime assessment
        if cycle_position in ['Peak Volatility', 'High in Cycle']:
            regime_assess = f"{cycle_position} - volatility may mean revert lower"
        elif cycle_position in ['Volatility Trough', 'Low in Cycle']:
            regime_assess = f"{cycle_position} - volatility may increase"
        else:
            regime_assess = f"{cycle_position} - balanced volatility environment"
        
        # Options strategy guidance
        if options_adjustment == "Sell Premium":
            options_guidance = "High volatility favors premium selling strategies (puts/calls)"
        elif options_adjustment == "Buy Premium":
            options_guidance = "Low volatility favors premium buying strategies (long options)"
        else:
            options_guidance = "Neutral volatility - balanced options approach"
        
        # Risk assessment
        if volatility_5d >= 40:
            risk_assessment = "Very high risk environment - use smaller position sizes"
        elif volatility_5d >= 25:
            risk_assessment = "Elevated risk environment - standard risk management"
        elif volatility_5d >= 15:
            risk_assessment = "Normal risk environment - typical position sizing"
        else:
            risk_assessment = "Low risk environment - may consider larger positions"
        
        return {
            'overall': overall,
            'trend': trend,
            'regime': regime_assess,
            'options': options_guidance,
            'risk': risk_assessment
        }
        
    except Exception as e:
        logger.error(f"Volatility interpretation error: {e}")
        return {
            'overall': 'Interpretation error',
            'trend': 'Cannot interpret trend',
            'regime': 'Cannot assess regime',
            'options': 'Review volatility data manually',
            'risk': 'Cannot assess risk level'
        }

@safe_calculation_wrapper
def calculate_market_volatility_comparison(symbols: list = ['SPY', 'QQQ', 'IWM']) -> Dict[str, Any]:
    """
    Calculate market-wide volatility metrics for comparison
    """
    try:
        import yfinance as yf
        
        market_vol_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='2mo')  # Get enough data for 30-day rolling
                
                if len(data) >= 30:
                    vol_analysis = calculate_volatility_analysis(data)
                    if vol_analysis.get('calculation_success', False):
                        market_vol_data[symbol] = {
                            'vol_regime': vol_analysis.get('vol_regime', 'Unknown'),
                            'volatility_5d': vol_analysis.get('volatility_5d', 0),
                            'vol_ratio_5d_30d': vol_analysis.get('vol_ratio_5d_30d', 1.0),
                            'regime_score': vol_analysis.get('regime_score', 50),
                            'cycle_position': vol_analysis.get('cycle_position', 'Unknown'),
                            'options_adjustment': vol_analysis.get('options_adjustment', 'Unknown')
                        }
                        
            except Exception as e:
                logger.error(f"Error fetching market data for {symbol}: {e}")
                continue
        
        if market_vol_data:
            # Calculate overall market volatility environment
            avg_volatility = sum([data['volatility_5d'] for data in market_vol_data.values()]) / len(market_vol_data)
            avg_regime_score = sum([data['regime_score'] for data in market_vol_data.values()]) / len(market_vol_data)
            
            # Market volatility regime classification
            if avg_volatility >= 35:
                market_regime = "High Volatility Market Environment"
                vix_estimate = "High (VIX likely >30)"
            elif avg_volatility >= 25:
                market_regime = "Elevated Volatility Environment"
                vix_estimate = "Elevated (VIX likely 20-30)"
            elif avg_volatility >= 15:
                market_regime = "Normal Volatility Environment"
                vix_estimate = "Normal (VIX likely 15-20)"
            else:
                market_regime = "Low Volatility Environment"
                vix_estimate = "Low (VIX likely <15)"
            
            # Options environment assessment
            options_environments = [data['options_adjustment'] for data in market_vol_data.values()]
            if options_environments.count('Sell Premium') >= 2:
                market_options_env = "Premium Selling Environment"
            elif options_environments.count('Buy Premium') >= 2:
                market_options_env = "Premium Buying Environment"
            else:
                market_options_env = "Mixed Options Environment"
            
            return {
                'individual_metrics': market_vol_data,
                'market_regime': market_regime,
                'market_options_env': market_options_env,
                'avg_volatility': round(float(avg_volatility), 2),
                'avg_regime_score': round(float(avg_regime_score), 1),
                'vix_estimate': vix_estimate,
                'sample_symbols': symbols,
                'calculation_success': True
            }
        else:
            return {
                'error': 'Could not retrieve market volatility data',
                'calculation_success': False
            }
            
    except Exception as e:
        logger.error(f"Market volatility comparison error: {e}")
        return {
            'error': f'Market volatility analysis failed: {str(e)}',
            'calculation_success': False
        }

@safe_calculation_wrapper
def get_volatility_regime_for_options(vol_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get volatility regime information specifically for options strategy adjustment
    """
    try:
        if 'error' in vol_data or not vol_data.get('calculation_success', False):
            return {
                'regime': 'Unknown',
                'multiplier': 1.0,
                'strategy_bias': 'Neutral',
                'confidence': 'Low'
            }
        
        volatility_5d = vol_data.get('volatility_5d', 20)
        vol_regime = vol_data.get('vol_regime', 'Normal')
        cycle_position = vol_data.get('cycle_position', 'Mid Cycle')
        
        # Determine options strategy multiplier based on volatility regime
        if vol_regime == 'Extreme High':
            multiplier = 1.4  # Expand strikes in high vol
            strategy_bias = 'Aggressive Selling'
            confidence = 'High'
        elif vol_regime == 'High':
            multiplier = 1.2
            strategy_bias = 'Premium Selling'
            confidence = 'High'
        elif vol_regime == 'Above Normal':
            multiplier = 1.1
            strategy_bias = 'Slight Selling Bias'
            confidence = 'Medium'
        elif vol_regime == 'Normal':
            multiplier = 1.0
            strategy_bias = 'Neutral'
            confidence = 'Medium'
        elif vol_regime == 'Below Normal':
            multiplier = 0.9
            strategy_bias = 'Slight Buying Bias'
            confidence = 'Medium'
        else:  # Low
            multiplier = 0.8  # Contract strikes in low vol
            strategy_bias = 'Premium Buying'
            confidence = 'High'
        
        return {
            'regime': vol_regime,
            'multiplier': multiplier,
            'strategy_bias': strategy_bias,
            'confidence': confidence,
            'current_volatility': volatility_5d,
            'cycle_position': cycle_position
        }
        
    except Exception as e:
        logger.error(f"Volatility regime for options error: {e}")
        return {
            'regime': 'Error',
            'multiplier': 1.0,
            'strategy_bias': 'Neutral',
            'confidence': 'Low'
        }
