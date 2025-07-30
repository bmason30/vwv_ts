"""
Volume Analysis Module for VWV Trading System
Advanced volume trend analysis with 5d/30d comparisons and regime detection
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def calculate_5day_rolling_volume(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate 5-day rolling volume average and analysis"""
    try:
        if len(data) < 5:
            return {'error': 'Insufficient data for 5-day analysis'}
        
        volume = data['Volume']
        
        # 5-day rolling average
        volume_5d = volume.rolling(window=5).mean()
        current_5d_avg = float(volume_5d.iloc[-1])
        
        # Current volume
        current_volume = float(volume.iloc[-1])
        
        # 5-day trend analysis (last 5 days)
        recent_5d_values = volume_5d.tail(5).dropna()
        
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
            slope_pct = (slope / current_5d_avg) * 100 if current_5d_avg > 0 else 0
            
            if slope_pct > 5:
                trend_direction = "Strongly Increasing"
                trend_strength = "Strong"
            elif slope_pct > 1:
                trend_direction = "Increasing"
                trend_strength = "Moderate"
            elif slope_pct < -5:
                trend_direction = "Strongly Decreasing"
                trend_strength = "Strong"
            elif slope_pct < -1:
                trend_direction = "Decreasing"
                trend_strength = "Moderate"
            else:
                trend_direction = "Stable"
                trend_strength = "Weak"
        else:
            slope = 0
            slope_pct = 0
            trend_direction = "Insufficient Data"
            trend_strength = "N/A"
        
        # Volume momentum (rate of change)
        volume_momentum = 0
        if len(recent_5d_values) >= 2:
            volume_momentum = ((recent_5d_values.iloc[-1] - recent_5d_values.iloc[-2]) / recent_5d_values.iloc[-2]) * 100
        
        return {
            'current_5d_average': round(current_5d_avg, 0),
            'current_volume': round(current_volume, 0),
            'trend_slope': round(slope, 2),
            'trend_slope_pct': round(slope_pct, 2),
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'volume_momentum': round(volume_momentum, 2),
            'volume_vs_5d_pct': round(((current_volume - current_5d_avg) / current_5d_avg) * 100, 2) if current_5d_avg > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"5-day rolling volume calculation error: {e}")
        return {'error': f'Calculation error: {str(e)}'}

@safe_calculation_wrapper
def calculate_volume_comparison_30d(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate 5-day vs 30-day volume comparison analysis"""
    try:
        if len(data) < 30:
            return {'error': 'Insufficient data for 30-day analysis'}
        
        volume = data['Volume']
        
        # Rolling averages
        volume_5d = volume.rolling(window=5).mean().iloc[-1]
        volume_30d = volume.rolling(window=30).mean().iloc[-1]
        
        if volume_30d == 0:
            return {'error': 'Invalid 30-day volume data'}
        
        # Relative volume ratio
        relative_ratio = volume_5d / volume_30d
        
        # Deviation percentage
        deviation_pct = ((volume_5d - volume_30d) / volume_30d) * 100
        
        # Volume regime classification
        if relative_ratio >= 2.0:
            regime = "Extreme High"
            regime_score = 90
        elif relative_ratio >= 1.5:
            regime = "High"
            regime_score = 75
        elif relative_ratio >= 1.2:
            regime = "Above Normal"
            regime_score = 60
        elif relative_ratio >= 0.8:
            regime = "Normal"
            regime_score = 50
        elif relative_ratio >= 0.5:
            regime = "Below Normal"
            regime_score = 35
        elif relative_ratio >= 0.3:
            regime = "Low"
            regime_score = 20
        else:
            regime = "Extreme Low"
            regime_score = 10
        
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
        
        return {
            'volume_5d_avg': round(volume_5d, 0),
            'volume_30d_avg': round(volume_30d, 0),
            'relative_ratio': round(relative_ratio, 2),
            'deviation_pct': round(deviation_pct, 1),
            'regime_classification': regime,
            'regime_score': regime_score,
            'significance': significance,
            'above_30d_average': volume_5d > volume_30d
        }
        
    except Exception as e:
        logger.error(f"Volume comparison calculation error: {e}")
        return {'error': f'Comparison error: {str(e)}'}

@safe_calculation_wrapper
def calculate_volume_breakout_detection(data: pd.DataFrame) -> Dict[str, Any]:
    """Detect volume breakouts and unusual activity"""
    try:
        if len(data) < 50:
            return {'error': 'Insufficient data for breakout analysis'}
        
        volume = data['Volume']
        
        # Calculate statistics over different periods
        volume_mean_20d = volume.rolling(20).mean().iloc[-1]
        volume_std_20d = volume.rolling(20).std().iloc[-1]
        current_volume = volume.iloc[-1]
        
        # Z-score calculation
        if volume_std_20d > 0:
            volume_zscore = (current_volume - volume_mean_20d) / volume_std_20d
        else:
            volume_zscore = 0
        
        # Breakout classification
        if volume_zscore >= 3.0:
            breakout_type = "Extreme Volume Breakout"
            breakout_strength = "Very Strong"
            breakout_score = 95
        elif volume_zscore >= 2.0:
            breakout_type = "Strong Volume Breakout"
            breakout_strength = "Strong"
            breakout_score = 80
        elif volume_zscore >= 1.5:
            breakout_type = "Moderate Volume Increase"
            breakout_strength = "Moderate"
            breakout_score = 65
        elif volume_zscore <= -2.0:
            breakout_type = "Volume Drought"
            breakout_strength = "Strong"
            breakout_score = 20
        elif volume_zscore <= -1.5:
            breakout_type = "Low Volume"
            breakout_strength = "Moderate"
            breakout_score = 35
        else:
            breakout_type = "Normal Volume"
            breakout_strength = "Normal"
            breakout_score = 50
        
        # Volume percentile (position in recent distribution)
        volume_percentile = volume.rolling(50).rank(pct=True).iloc[-1] * 100
        
        return {
            'current_volume': round(current_volume, 0),
            'volume_20d_mean': round(volume_mean_20d, 0),
            'volume_zscore': round(volume_zscore, 2),
            'volume_percentile': round(volume_percentile, 1),
            'breakout_type': breakout_type,
            'breakout_strength': breakout_strength,
            'breakout_score': breakout_score,
            'is_breakout': abs(volume_zscore) >= 1.5
        }
        
    except Exception as e:
        logger.error(f"Volume breakout detection error: {e}")
        return {'error': f'Breakout detection error: {str(e)}'}

@safe_calculation_wrapper
def calculate_volume_composite_score(volume_analysis: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Calculate composite volume score for technical integration"""
    try:
        # Extract components
        rolling_5d = volume_analysis.get('rolling_5d_analysis', {})
        comparison_30d = volume_analysis.get('comparison_30d_analysis', {})
        breakout_analysis = volume_analysis.get('breakout_analysis', {})
        
        scores = []
        weights = []
        components = {}
        
        # 5-day trend component (40% weight)
        if 'trend_direction' in rolling_5d:
            trend_direction = rolling_5d['trend_direction']
            volume_momentum = rolling_5d.get('volume_momentum', 0)
            
            if trend_direction == "Strongly Increasing":
                trend_score = 80 + min(abs(volume_momentum), 15)  # 80-95
            elif trend_direction == "Increasing":
                trend_score = 60 + min(abs(volume_momentum), 15)  # 60-75
            elif trend_direction == "Stable":
                trend_score = 50
            elif trend_direction == "Decreasing":
                trend_score = 40 - min(abs(volume_momentum), 15)  # 25-40
            elif trend_direction == "Strongly Decreasing":
                trend_score = 20 - min(abs(volume_momentum), 15)  # 5-20
            else:
                trend_score = 50
            
            scores.append(trend_score)
            weights.append(0.40)
            components['trend_score'] = round(trend_score, 1)
        
        # 30-day comparison component (35% weight)
        if 'regime_score' in comparison_30d:
            regime_score = comparison_30d['regime_score']
            scores.append(regime_score)
            weights.append(0.35)
            components['regime_score'] = regime_score
        
        # Breakout detection component (25% weight)
        if 'breakout_score' in breakout_analysis:
            breakout_score = breakout_analysis['breakout_score']
            scores.append(breakout_score)
            weights.append(0.25)
            components['breakout_score'] = breakout_score
        
        # Calculate weighted composite score
        if len(scores) == len(weights) and sum(weights) > 0:
            composite_score = sum(score * weight for score, weight in zip(scores, weights)) / sum(weights)
        else:
            composite_score = 50  # Default neutral
        
        # Ensure score is within bounds
        final_score = max(1, min(100, round(composite_score, 1)))
        
        # Score interpretation
        if final_score >= 80:
            interpretation = "Very Strong Volume"
        elif final_score >= 70:
            interpretation = "Strong Volume"
        elif final_score >= 60:
            interpretation = "Above Average Volume"
        elif final_score >= 40:
            interpretation = "Normal Volume"
        elif final_score >= 30:
            interpretation = "Below Average Volume"
        else:
            interpretation = "Weak Volume"
        
        return final_score, {
            'composite_score': final_score,
            'interpretation': interpretation,
            'component_scores': components,
            'total_components': len(scores),
            'weight_distribution': dict(zip(['trend', 'regime', 'breakout'], weights))
        }
        
    except Exception as e:
        logger.error(f"Volume composite score calculation error: {e}")
        return 50.0, {'error': str(e)}

@safe_calculation_wrapper
def calculate_complete_volume_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Complete volume analysis pipeline"""
    try:
        if len(data) < 30:
            return {'error': 'Insufficient data for complete volume analysis (minimum 30 periods required)'}
        
        # Perform all volume analyses
        rolling_5d_analysis = calculate_5day_rolling_volume(data)
        comparison_30d_analysis = calculate_volume_comparison_30d(data)
        breakout_analysis = calculate_volume_breakout_detection(data)
        
        # Calculate composite score
        volume_analysis = {
            'rolling_5d_analysis': rolling_5d_analysis,
            'comparison_30d_analysis': comparison_30d_analysis,
            'breakout_analysis': breakout_analysis
        }
        
        composite_score, score_details = calculate_volume_composite_score(volume_analysis)
        
        # Combine all results
        complete_analysis = {
            'rolling_5d_analysis': rolling_5d_analysis,
            'comparison_30d_analysis': comparison_30d_analysis,
            'breakout_analysis': breakout_analysis,
            'composite_score': composite_score,
            'score_details': score_details,
            'analysis_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_points_analyzed': len(data)
        }
        
        return complete_analysis
        
    except Exception as e:
        logger.error(f"Complete volume analysis error: {e}")
        return {'error': f'Complete analysis error: {str(e)}'}

def format_volume_analysis_for_display(volume_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Format volume analysis for UI display"""
    try:
        if 'error' in volume_analysis:
            return volume_analysis
        
        rolling_5d = volume_analysis.get('rolling_5d_analysis', {})
        comparison_30d = volume_analysis.get('comparison_30d_analysis', {})
        breakout = volume_analysis.get('breakout_analysis', {})
        score_details = volume_analysis.get('score_details', {})
        
        # Summary metrics for main display
        summary_metrics = {
            'current_volume': rolling_5d.get('current_volume', 0),
            'volume_5d_avg': rolling_5d.get('current_5d_average', 0),
            'volume_30d_avg': comparison_30d.get('volume_30d_avg', 0),
            'relative_ratio': comparison_30d.get('relative_ratio', 1.0),
            'deviation_pct': comparison_30d.get('deviation_pct', 0),
            'regime_classification': comparison_30d.get('regime_classification', 'Normal'),
            'trend_direction': rolling_5d.get('trend_direction', 'Unknown'),
            'breakout_type': breakout.get('breakout_type', 'Normal Volume'),
            'composite_score': volume_analysis.get('composite_score', 50),
            'interpretation': score_details.get('interpretation', 'Normal Volume')
        }
        
        # Detailed breakdown for expandable sections
        detailed_analysis = {
            '5_day_rolling': rolling_5d,
            '30_day_comparison': comparison_30d,
            'breakout_detection': breakout,
            'scoring_breakdown': score_details
        }
        
        return {
            'summary_metrics': summary_metrics,
            'detailed_analysis': detailed_analysis,
            'display_ready': True
        }
        
    except Exception as e:
        logger.error(f"Volume analysis formatting error: {e}")
        return {'error': f'Formatting error: {str(e)}'}
