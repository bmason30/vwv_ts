"""
Volatility Analysis Module for VWV Trading System
Advanced volatility trend analysis with 5d/30d comparisons and regime detection
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def calculate_5day_rolling_volatility(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate 5-day rolling volatility average and analysis"""
    try:
        if len(data) < 10:
            return {'error': 'Insufficient data for 5-day volatility analysis'}
        
        # Calculate daily returns
        returns = data['Close'].pct_change().dropna()
        
        # 5-day rolling volatility (annualized)
        volatility_5d = returns.rolling(window=5).std() * np.sqrt(252) * 100  # Annualized percentage
        current_5d_volatility = float(volatility_5d.iloc[-1])
        
        # Current daily volatility (for comparison)
        recent_returns = returns.tail(5)
        current_daily_vol = float(recent_returns.std() * np.sqrt(252) * 100) if len(recent_returns) > 1 else 0
        
        # 5-day volatility trend analysis
        recent_5d_values = volatility_5d.tail(5).dropna()
        
        if len(recent_5d_values) >= 3:
            # Calculate trend using linear regression slope
            x_values = np.arange(len(recent_5d_values))
            y_values = recent_5d_values.values
            
            # Simple linear regression
            n = len(x_values)
            sum_x = np.sum(x_values)
            sum_y = np.sum(y_values)
            sum_xy = np.sum(x_values * y_values)
            sum_x2 = np.sum(x_values ** 2)
            
            # Calculate slope (trend)
            if n * sum_x2 - sum_x ** 2 != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            else:
                slope = 0
            
            # Trend classification
            slope_pct = (slope / current_5d_volatility) * 100 if current_5d_volatility > 0 else 0
            
            if slope_pct > 10:
                trend_direction = "Rapidly Expanding"
                trend_strength = "Strong"
                cycle_phase = "Expansion"
            elif slope_pct > 3:
                trend_direction = "Expanding"
                trend_strength = "Moderate"
                cycle_phase = "Expansion"
            elif slope_pct < -10:
                trend_direction = "Rapidly Contracting"
                trend_strength = "Strong"
                cycle_phase = "Contraction"
            elif slope_pct < -3:
                trend_direction = "Contracting"
                trend_strength = "Moderate"
                cycle_phase = "Contraction"
            else:
                trend_direction = "Stable"
                trend_strength = "Weak"
                cycle_phase = "Neutral"
        else:
            slope = 0
            slope_pct = 0
            trend_direction = "Insufficient Data"
            trend_strength = "N/A"
            cycle_phase = "Unknown"
        
        # Volatility momentum (rate of change)
        volatility_momentum = 0
        if len(recent_5d_values) >= 2:
            volatility_momentum = ((recent_5d_values.iloc[-1] - recent_5d_values.iloc[-2]) / recent_5d_values.iloc[-2]) * 100
        
        # Volatility acceleration (second derivative)
        volatility_acceleration = 0
        if len(recent_5d_values) >= 3:
            recent_changes = recent_5d_values.diff().tail(2)
            if len(recent_changes) == 2:
                volatility_acceleration = recent_changes.iloc[-1] - recent_changes.iloc[-2]
        
        return {
            'current_5d_volatility': round(current_5d_volatility, 2),
            'current_daily_volatility': round(current_daily_vol, 2),
            'trend_slope': round(slope, 3),
            'trend_slope_pct': round(slope_pct, 2),
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'cycle_phase': cycle_phase,
            'volatility_momentum': round(volatility_momentum, 2),
            'volatility_acceleration': round(volatility_acceleration, 3)
        }
        
    except Exception as e:
        logger.error(f"5-day rolling volatility calculation error: {e}")
        return {'error': f'Calculation error: {str(e)}'}

@safe_calculation_wrapper
def calculate_volatility_comparison_30d(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate 5-day vs 30-day volatility comparison analysis"""
    try:
        if len(data) < 35:
            return {'error': 'Insufficient data for 30-day volatility analysis'}
        
        # Calculate daily returns
        returns = data['Close'].pct_change().dropna()
        
        # Rolling volatilities (annualized)
        volatility_5d = returns.rolling(window=5).std().iloc[-1] * np.sqrt(252) * 100
        volatility_30d = returns.rolling(window=30).std().iloc[-1] * np.sqrt(252) * 100
        
        if volatility_30d == 0:
            return {'error': 'Invalid 30-day volatility data'}
        
        # Relative volatility ratio
        relative_ratio = volatility_5d / volatility_30d
        
        # Deviation percentage
        deviation_pct = ((volatility_5d - volatility_30d) / volatility_30d) * 100
        
        # Volatility regime classification
        if relative_ratio >= 2.0:
            regime = "Extreme High Volatility"
            regime_score = 90
            market_environment = "Crisis/Panic"
        elif relative_ratio >= 1.5:
            regime = "High Volatility"
            regime_score = 75
            market_environment = "Stressed"
        elif relative_ratio >= 1.2:
            regime = "Elevated Volatility"
            regime_score = 60
            market_environment = "Uncertain"
        elif relative_ratio >= 0.8:
            regime = "Normal Volatility"
            regime_score = 50
            market_environment = "Stable"
        elif relative_ratio >= 0.6:
            regime = "Low Volatility"
            regime_score = 35
            market_environment = "Calm"
        elif relative_ratio >= 0.4:
            regime = "Very Low Volatility"
            regime_score = 20
            market_environment = "Complacent"
        else:
            regime = "Extremely Low Volatility"
            regime_score = 10
            market_environment = "Suppressed"
        
        # Significance classification
        if abs(deviation_pct) >= 100:
            significance = "Very Significant"
        elif abs(deviation_pct) >= 50:
            significance = "Significant"
        elif abs(deviation_pct) >= 25:
            significance = "Moderate"
        elif abs(deviation_pct) >= 10:
            significance = "Minor"
        else:
            significance = "Negligible"
        
        # Options implications
        if relative_ratio >= 1.5:
            options_implication = "High Premium Environment - Favor Selling"
        elif relative_ratio >= 1.2:
            options_implication = "Elevated Premium - Consider Selling"
        elif relative_ratio <= 0.6:
            options_implication = "Low Premium Environment - Favor Buying"
        else:
            options_implication = "Normal Premium Environment"
        
        return {
            'volatility_5d': round(volatility_5d, 2),
            'volatility_30d': round(volatility_30d, 2),
            'relative_ratio': round(relative_ratio, 2),
            'deviation_pct': round(deviation_pct, 1),
            'regime_classification': regime,
            'regime_score': regime_score,
            'market_environment': market_environment,
            'significance': significance,
            'options_implication': options_implication,
            'above_30d_average': volatility_5d > volatility_30d
        }
        
    except Exception as e:
        logger.error(f"Volatility comparison calculation error: {e}")
        return {'error': f'Comparison error: {str(e)}'}

@safe_calculation_wrapper
def calculate_volatility_regime_detection(data: pd.DataFrame) -> Dict[str, Any]:
    """Detect volatility regimes and unusual patterns"""
    try:
        if len(data) < 60:
            return {'error': 'Insufficient data for regime analysis'}
        
        # Calculate daily returns and rolling volatilities
        returns = data['Close'].pct_change().dropna()
        
        # Multiple timeframe volatilities
        vol_10d = returns.rolling(10).std() * np.sqrt(252) * 100
        vol_20d = returns.rolling(20).std() * np.sqrt(252) * 100
        vol_50d = returns.rolling(50).std() * np.sqrt(252) * 100
        
        current_vol_10d = vol_10d.iloc[-1]
        current_vol_20d = vol_20d.iloc[-1]
        current_vol_50d = vol_50d.iloc[-1]
        
        # Historical volatility percentiles
        vol_percentile_20d = vol_20d.rolling(100).rank(pct=True).iloc[-1] * 100
        vol_percentile_50d = vol_50d.rolling(252).rank(pct=True).iloc[-1] * 100 if len(vol_50d) >= 252 else 50
        
        # Volatility clustering detection
        high_vol_threshold = vol_50d.quantile(0.75)
        low_vol_threshold = vol_50d.quantile(0.25)
        
        recent_high_vol_days = (vol_10d.tail(10) > high_vol_threshold).sum()
        recent_low_vol_days = (vol_10d.tail(10) < low_vol_threshold).sum()
        
        # Regime classification based on multiple factors
        if vol_percentile_20d >= 90:
            regime = "Extreme Volatility Regime"
            regime_strength = "Very Strong"
            regime_score = 95
        elif vol_percentile_20d >= 75:
            regime = "High Volatility Regime"
            regime_strength = "Strong"
            regime_score = 80
        elif vol_percentile_20d >= 60:
            regime = "Elevated Volatility Regime"
            regime_strength = "Moderate"
            regime_score = 65
        elif vol_percentile_20d <= 10:
            regime = "Low Volatility Regime"
            regime_strength = "Very Strong"
            regime_score = 15
        elif vol_percentile_20d <= 25:
            regime = "Suppressed Volatility Regime"
            regime_strength = "Strong"
            regime_score = 25
        elif vol_percentile_20d <= 40:
            regime = "Below Normal Volatility"
            regime_strength = "Moderate"
            regime_score = 35
        else:
            regime = "Normal Volatility Regime"
            regime_strength = "Normal"
            regime_score = 50
        
        # Volatility clustering analysis
        if recent_high_vol_days >= 7:
            clustering = "High Volatility Cluster"
        elif recent_low_vol_days >= 7:
            clustering = "Low Volatility Cluster"
        elif recent_high_vol_days >= 4:
            clustering = "Moderate High Vol Clustering"
        elif recent_low_vol_days >= 4:
            clustering = "Moderate Low Vol Clustering"
        else:
            clustering = "No Clear Clustering"
        
        return {
            'current_vol_10d': round(current_vol_10d, 2),
            'current_vol_20d': round(current_vol_20d, 2),
            'current_vol_50d': round(current_vol_50d, 2),
            'vol_percentile_20d': round(vol_percentile_20d, 1),
            'vol_percentile_50d': round(vol_percentile_50d, 1),
            'regime_classification': regime,
            'regime_strength': regime_strength,
            'regime_score': regime_score,
            'volatility_clustering': clustering,
            'high_vol_days_recent': int(recent_high_vol_days),
            'low_vol_days_recent': int(recent_low_vol_days)
        }
        
    except Exception as e:
        logger.error(f"Volatility regime detection error: {e}")
        return {'error': f'Regime detection error: {str(e)}'}

@safe_calculation_wrapper
def calculate_volatility_composite_score(volatility_analysis: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Calculate composite volatility score for technical integration"""
    try:
        # Extract components
        rolling_5d = volatility_analysis.get('rolling_5d_analysis', {})
        comparison_30d = volatility_analysis.get('comparison_30d_analysis', {})
        regime_analysis = volatility_analysis.get('regime_analysis', {})
        
        scores = []
        weights = []
        components = {}
        
        # 5-day trend component (35% weight)
        if 'cycle_phase' in rolling_5d:
            cycle_phase = rolling_5d['cycle_phase']
            trend_strength = rolling_5d.get('trend_strength', 'Weak')
            volatility_momentum = rolling_5d.get('volatility_momentum', 0)
            
            # Base score by cycle phase
            if cycle_phase == "Expansion":
                if trend_strength == "Strong":
                    trend_score = 75 + min(abs(volatility_momentum) / 2, 15)  # 75-90
                else:
                    trend_score = 65 + min(abs(volatility_momentum) / 2, 10)  # 65-75
            elif cycle_phase == "Contraction":
                if trend_strength == "Strong":
                    trend_score = 25 - min(abs(volatility_momentum) / 2, 15)  # 10-25
                else:
                    trend_score = 35 - min(abs(volatility_momentum) / 2, 10)  # 25-35
            else:  # Neutral/Stable/Unknown
                trend_score = 50
            
            scores.append(max(5, min(95, trend_score)))
            weights.append(0.35)
            components['trend_score'] = round(trend_score, 1)
        
        # 30-day comparison component (40% weight)
        if 'regime_score' in comparison_30d:
            regime_score = comparison_30d['regime_score']
            scores.append(regime_score)
            weights.append(0.40)
            components['regime_score'] = regime_score
        
        # Regime detection component (25% weight)
        if 'regime_score' in regime_analysis:
            regime_detection_score = regime_analysis['regime_score']
            scores.append(regime_detection_score)
            weights.append(0.25)
            components['regime_detection_score'] = regime_detection_score
        
        # Calculate weighted composite score
        if len(scores) == len(weights) and sum(weights) > 0:
            composite_score = sum(score * weight for score, weight in zip(scores, weights)) / sum(weights)
        else:
            composite_score = 50  # Default neutral
        
        # Ensure score is within bounds
        final_score = max(1, min(100, round(composite_score, 1)))
        
        # Score interpretation
        if final_score >= 85:
            interpretation = "Extreme High Volatility"
        elif final_score >= 75:
            interpretation = "High Volatility"
        elif final_score >= 65:
            interpretation = "Elevated Volatility"
        elif final_score >= 35:
            interpretation = "Normal Volatility"
        elif final_score >= 25:
            interpretation = "Low Volatility"
        elif final_score >= 15:
            interpretation = "Very Low Volatility"
        else:
            interpretation = "Suppressed Volatility"
        
        return final_score, {
            'composite_score': final_score,
            'interpretation': interpretation,
            'component_scores': components,
            'total_components': len(scores),
            'weight_distribution': dict(zip(['trend', 'regime_comparison', 'regime_detection'], weights))
        }
        
    except Exception as e:
        logger.error(f"Volatility composite score calculation error: {e}")
        return 50.0, {'error': str(e)}

@safe_calculation_wrapper
def calculate_complete_volatility_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Complete volatility analysis pipeline"""
    try:
        if len(data) < 35:
            return {'error': 'Insufficient data for complete volatility analysis (minimum 35 periods required)'}
        
        # Perform all volatility analyses
        rolling_5d_analysis = calculate_5day_rolling_volatility(data)
        comparison_30d_analysis = calculate_volatility_comparison_30d(data)
        regime_analysis = calculate_volatility_regime_detection(data)
        
        # Calculate composite score
        volatility_analysis = {
            'rolling_5d_analysis': rolling_5d_analysis,
            'comparison_30d_analysis': comparison_30d_analysis,
            'regime_analysis': regime_analysis
        }
        
        composite_score, score_details = calculate_volatility_composite_score(volatility_analysis)
        
        # Combine all results
        complete_analysis = {
            'rolling_5d_analysis': rolling_5d_analysis,
            'comparison_30d_analysis': comparison_30d_analysis,
            'regime_analysis': regime_analysis,
            'composite_score': composite_score,
            'score_details': score_details,
            'analysis_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_points_analyzed': len(data)
        }
        
        return complete_analysis
        
    except Exception as e:
        logger.error(f"Complete volatility analysis error: {e}")
        return {'error': f'Complete analysis error: {str(e)}'}

def format_volatility_analysis_for_display(volatility_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Format volatility analysis for UI display"""
    try:
        if 'error' in volatility_analysis:
            return volatility_analysis
        
        rolling_5d = volatility_analysis.get('rolling_5d_analysis', {})
        comparison_30d = volatility_analysis.get('comparison_30d_analysis', {})
        regime = volatility_analysis.get('regime_analysis', {})
        score_details = volatility_analysis.get('score_details', {})
        
        # Summary metrics for main display
        summary_metrics = {
            'current_5d_volatility': rolling_5d.get('current_5d_volatility', 0),
            'current_30d_volatility': comparison_30d.get('volatility_30d', 0),
            'relative_ratio': comparison_30d.get('relative_ratio', 1.0),
            'deviation_pct': comparison_30d.get('deviation_pct', 0),
            'regime_classification': comparison_30d.get('regime_classification', 'Normal Volatility'),
            'market_environment': comparison_30d.get('market_environment', 'Stable'),
            'trend_direction': rolling_5d.get('trend_direction', 'Unknown'),
            'cycle_phase': rolling_5d.get('cycle_phase', 'Unknown'),
            'options_implication': comparison_30d.get('options_implication', 'Normal Premium Environment'),
            'composite_score': volatility_analysis.get('composite_score', 50),
            'interpretation': score_details.get('interpretation', 'Normal Volatility')
        }
        
        # Detailed breakdown for expandable sections
        detailed_analysis = {
            '5_day_rolling': rolling_5d,
            '30_day_comparison': comparison_30d,
            'regime_detection': regime,
            'scoring_breakdown': score_details
        }
        
        return {
            'summary_metrics': summary_metrics,
            'detailed_analysis': detailed_analysis,
            'display_ready': True
        }
        
    except Exception as e:
        logger.error(f"Volatility analysis formatting error: {e}")
        return {'error': f'Formatting error: {str(e)}'}
