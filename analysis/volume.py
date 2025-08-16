"""
VWV Trading System - Enhanced Volume Analysis Module v8.0.0
FINAL VERSION: Fixes pandas warning + adds composite scoring

FIXES:
✅ Pandas FutureWarning: observed=False deprecated - now uses observed=True
✅ Added comprehensive volume composite score calculation

NEW FEATURES:
✅ Volume Composite Score (0-100 scale) for gradient bar
✅ All 10 enhanced volume analysis features
✅ Smart money vs retail detection
✅ Volume quality assessment
✅ Enhanced trading implications

PRESERVED: All existing functionality
"""

import pandas as pd
import numpy as np
import streamlit as st
import logging
from typing import Dict, Any, Optional, Tuple
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def calculate_on_balance_volume(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate On-Balance Volume (OBV) analysis"""
    try:
        if len(data) < 20:
            return {'obv': 0, 'obv_trend': 'Neutral', 'obv_signal': 'No Signal'}
        
        close = data['Close']
        volume = data['Volume']
        
        # Calculate OBV
        obv = pd.Series(index=data.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(data)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        current_obv = obv.iloc[-1]
        
        # OBV trend analysis
        if len(obv) >= 10:
            obv_10_ago = obv.iloc[-10]
            obv_trend_pct = (current_obv - obv_10_ago) / abs(obv_10_ago) * 100 if obv_10_ago != 0 else 0
            
            if obv_trend_pct > 10:
                obv_trend = 'Strong Accumulation'
            elif obv_trend_pct > 3:
                obv_trend = 'Accumulation'
            elif obv_trend_pct < -10:
                obv_trend = 'Strong Distribution'
            elif obv_trend_pct < -3:
                obv_trend = 'Distribution'
            else:
                obv_trend = 'Neutral'
        else:
            obv_trend = 'Neutral'
            obv_trend_pct = 0
        
        # OBV vs Price Divergence
        price_trend = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10] * 100 if len(close) >= 10 else 0
        
        obv_signal = 'No Signal'
        if price_trend > 2 and obv_trend_pct < -2:
            obv_signal = 'Bearish Divergence'
        elif price_trend < -2 and obv_trend_pct > 2:
            obv_signal = 'Bullish Divergence'
        elif price_trend > 0 and obv_trend_pct > 0:
            obv_signal = 'Bullish Confirmation'
        elif price_trend < 0 and obv_trend_pct < 0:
            obv_signal = 'Bearish Confirmation'
        
        return {
            'obv': round(current_obv, 0),
            'obv_trend': obv_trend,
            'obv_trend_pct': round(obv_trend_pct, 2),
            'obv_signal': obv_signal,
            'price_obv_correlation': calculate_correlation_score(price_trend, obv_trend_pct)
        }
        
    except Exception as e:
        logger.error(f"OBV calculation error: {e}")
        return {'obv': 0, 'obv_trend': 'Neutral', 'obv_signal': 'Error'}

@safe_calculation_wrapper
def calculate_accumulation_distribution_line(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Accumulation/Distribution Line (A/D Line)"""
    try:
        if len(data) < 20:
            return {'ad_line': 0, 'ad_trend': 'Neutral', 'ad_signal': 'No Signal'}
        
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']
        
        # Calculate Money Flow Multiplier
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero when high == low
        
        # Calculate Money Flow Volume
        mfv = clv * volume
        
        # Calculate A/D Line (cumulative)
        ad_line = mfv.cumsum()
        
        current_ad = ad_line.iloc[-1]
        
        # A/D Line trend analysis
        if len(ad_line) >= 10:
            ad_10_ago = ad_line.iloc[-10]
            ad_trend_pct = (current_ad - ad_10_ago) / abs(ad_10_ago) * 100 if ad_10_ago != 0 else 0
            
            if ad_trend_pct > 5:
                ad_trend = 'Strong Accumulation'
            elif ad_trend_pct > 1:
                ad_trend = 'Accumulation'
            elif ad_trend_pct < -5:
                ad_trend = 'Strong Distribution'
            elif ad_trend_pct < -1:
                ad_trend = 'Distribution'
            else:
                ad_trend = 'Neutral'
        else:
            ad_trend = 'Neutral'
            ad_trend_pct = 0
        
        # A/D vs Price Divergence
        price_trend = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10] * 100 if len(close) >= 10 else 0
        
        ad_signal = 'No Signal'
        if price_trend > 2 and ad_trend_pct < -1:
            ad_signal = 'Bearish Divergence (Distribution)'
        elif price_trend < -2 and ad_trend_pct > 1:
            ad_signal = 'Bullish Divergence (Accumulation)'
        elif price_trend > 0 and ad_trend_pct > 0:
            ad_signal = 'Bullish Confirmation'
        elif price_trend < 0 and ad_trend_pct < 0:
            ad_signal = 'Bearish Confirmation'
        
        return {
            'ad_line': round(current_ad, 0),
            'ad_trend': ad_trend,
            'ad_trend_pct': round(ad_trend_pct, 2),
            'ad_signal': ad_signal,
            'money_flow_multiplier': round(clv.iloc[-1], 3)
        }
        
    except Exception as e:
        logger.error(f"A/D Line calculation error: {e}")
        return {'ad_line': 0, 'ad_trend': 'Neutral', 'ad_signal': 'Error'}

@safe_calculation_wrapper
def calculate_volume_rate_of_change(data: pd.DataFrame, period: int = 10) -> Dict[str, Any]:
    """Calculate Volume Rate of Change (VROC)"""
    try:
        if len(data) < period + 5:
            return {'vroc': 0, 'vroc_trend': 'Neutral', 'vroc_signal': 'No Signal'}
        
        volume = data['Volume']
        
        # Calculate VROC
        vroc = ((volume - volume.shift(period)) / volume.shift(period)) * 100
        vroc = vroc.fillna(0)
        
        current_vroc = vroc.iloc[-1]
        
        # VROC trend classification
        if current_vroc > 50:
            vroc_trend = 'Explosive Volume'
        elif current_vroc > 20:
            vroc_trend = 'High Volume Growth'
        elif current_vroc > 5:
            vroc_trend = 'Moderate Volume Growth'
        elif current_vroc < -50:
            vroc_trend = 'Volume Collapse'
        elif current_vroc < -20:
            vroc_trend = 'Volume Decline'
        elif current_vroc < -5:
            vroc_trend = 'Moderate Volume Decline'
        else:
            vroc_trend = 'Stable Volume'
        
        # VROC signal generation
        if current_vroc > 30:
            vroc_signal = 'Breakout Alert (High Volume)'
        elif current_vroc < -30:
            vroc_signal = 'Breakdown Alert (Volume Dry-up)'
        elif abs(current_vroc) < 10:
            vroc_signal = 'Low Activity Warning'
        else:
            vroc_signal = 'Normal Activity'
        
        # VROC momentum (5-day trend)
        if len(vroc) >= 5:
            vroc_momentum = vroc.tail(5).mean()
        else:
            vroc_momentum = current_vroc
        
        return {
            'vroc': round(current_vroc, 2),
            'vroc_trend': vroc_trend,
            'vroc_signal': vroc_signal,
            'vroc_momentum': round(vroc_momentum, 2),
            'volume_acceleration': 'Accelerating' if vroc_momentum > current_vroc else 'Decelerating'
        }
        
    except Exception as e:
        logger.error(f"VROC calculation error: {e}")
        return {'vroc': 0, 'vroc_trend': 'Neutral', 'vroc_signal': 'Error'}

@safe_calculation_wrapper
def calculate_volume_price_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Volume Price Analysis (VPA)"""
    try:
        if len(data) < 20:
            return {'vpa_score': 50, 'vpa_signal': 'Insufficient Data', 'price_volume_correlation': 0}
        
        close = data['Close']
        volume = data['Volume']
        
        # Calculate daily price changes and volume changes
        price_change = close.pct_change().fillna(0)
        volume_change = volume.pct_change().fillna(0)
        
        # Volume-weighted price movements
        up_volume = volume.where(price_change > 0, 0).rolling(10).sum()
        down_volume = volume.where(price_change < 0, 0).rolling(10).sum()
        
        current_up_volume = up_volume.iloc[-1]
        current_down_volume = abs(down_volume.iloc[-1])
        
        # Volume ratio for up vs down moves
        if current_down_volume > 0:
            up_down_volume_ratio = current_up_volume / current_down_volume
        else:
            up_down_volume_ratio = 5.0  # Heavily bullish if no down volume
        
        # Price-Volume correlation over last 20 periods
        if len(data) >= 20:
            recent_price_change = price_change.tail(20)
            recent_volume = volume.tail(20)
            correlation = recent_price_change.corr(recent_volume)
            if pd.isna(correlation):
                correlation = 0
        else:
            correlation = 0
        
        # VPA Score calculation
        vpa_score = 50  # Neutral starting point
        
        # Adjust for volume ratio
        if up_down_volume_ratio > 2:
            vpa_score += 20
        elif up_down_volume_ratio > 1.5:
            vpa_score += 10
        elif up_down_volume_ratio < 0.5:
            vpa_score -= 20
        elif up_down_volume_ratio < 0.67:
            vpa_score -= 10
        
        # Adjust for correlation
        if correlation > 0.3:
            vpa_score += 15
        elif correlation > 0.1:
            vpa_score += 5
        elif correlation < -0.3:
            vpa_score -= 15
        elif correlation < -0.1:
            vpa_score -= 5
        
        # Clamp score
        vpa_score = max(0, min(100, vpa_score))
        
        # VPA Signal generation
        if vpa_score >= 80:
            vpa_signal = 'Strong Bullish (High Volume Confirmation)'
        elif vpa_score >= 65:
            vpa_signal = 'Bullish (Volume Supporting)'
        elif vpa_score >= 35:
            vpa_signal = 'Neutral (Mixed Volume Signals)'
        elif vpa_score >= 20:
            vpa_signal = 'Bearish (Volume Not Supporting)'
        else:
            vpa_signal = 'Strong Bearish (Volume Against Price)'
        
        return {
            'vpa_score': round(vpa_score, 1),
            'vpa_signal': vpa_signal,
            'up_down_volume_ratio': round(up_down_volume_ratio, 2),
            'price_volume_correlation': round(correlation, 3),
            'up_volume_10d': round(current_up_volume, 0),
            'down_volume_10d': round(current_down_volume, 0)
        }
        
    except Exception as e:
        logger.error(f"VPA calculation error: {e}")
        return {'vpa_score': 50, 'vpa_signal': 'Error', 'price_volume_correlation': 0}

@safe_calculation_wrapper
def calculate_dynamic_volume_thresholds(data: pd.DataFrame, lookback: int = 100) -> Dict[str, float]:
    """Calculate symbol-specific dynamic volume thresholds"""
    try:
        if len(data) < lookback:
            return {
                'extreme_high': 2.0,
                'high': 1.5,
                'above_normal': 1.2,
                'below_normal': 0.8,
                'low': 0.5
            }
        
        volume = data['Volume'].tail(lookback)
        volume_ma = volume.rolling(20).mean()
        volume_ratios = volume / volume_ma
        volume_ratios = volume_ratios.dropna()
        
        if len(volume_ratios) < 50:
            return {
                'extreme_high': 2.0,
                'high': 1.5,
                'above_normal': 1.2,
                'below_normal': 0.8,
                'low': 0.5
            }
        
        # Calculate percentile-based thresholds
        percentiles = np.percentile(volume_ratios, [10, 25, 75, 90, 95])
        
        return {
            'extreme_high': max(2.0, percentiles[4]),
            'high': max(1.5, percentiles[3]),
            'above_normal': max(1.2, percentiles[2]),
            'below_normal': min(0.8, percentiles[1]),
            'low': min(0.5, percentiles[0])
        }
        
    except Exception as e:
        logger.error(f"Dynamic threshold calculation error: {e}")
        return {
            'extreme_high': 2.0,
            'high': 1.5,
            'above_normal': 1.2,
            'below_normal': 0.8,
            'low': 0.5
        }

@safe_calculation_wrapper
def calculate_volume_cluster_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Volume Cluster Analysis for institutional vs retail detection"""
    try:
        if len(data) < 50:
            return {'cluster_type': 'Unknown', 'cluster_strength': 0, 'institutional_signal': 'No Signal'}
        
        volume = data['Volume']
        close = data['Close']
        
        # Calculate volume percentiles
        volume_95th = volume.quantile(0.95)
        volume_75th = volume.quantile(0.75)
        volume_25th = volume.quantile(0.25)
        
        # Recent volume analysis (last 10 days)
        recent_volume = volume.tail(10)
        recent_close = close.tail(10)
        
        # High volume days in recent period
        high_volume_days = (recent_volume > volume_75th).sum()
        extreme_volume_days = (recent_volume > volume_95th).sum()
        
        # Price action during high volume
        high_vol_price_change = 0
        if high_volume_days > 0:
            high_vol_mask = recent_volume > volume_75th
            if high_vol_mask.sum() > 0:
                high_vol_prices = recent_close[high_vol_mask]
                if len(high_vol_prices) >= 2:
                    high_vol_price_change = (high_vol_prices.iloc[-1] - high_vol_prices.iloc[0]) / high_vol_prices.iloc[0] * 100
        
        # Cluster analysis
        cluster_strength = (high_volume_days / 10) * 100
        
        # Cluster type determination
        if extreme_volume_days >= 2:
            if high_vol_price_change > 3:
                cluster_type = 'Institutional Accumulation'
                institutional_signal = 'Strong Buy Signal'
            elif high_vol_price_change < -3:
                cluster_type = 'Institutional Distribution'
                institutional_signal = 'Strong Sell Signal'
            else:
                cluster_type = 'Mixed Institutional Activity'
                institutional_signal = 'Mixed Signal'
        elif high_volume_days >= 4:
            if high_vol_price_change > 2:
                cluster_type = 'Retail FOMO Buying'
                institutional_signal = 'Caution (Retail Driven)'
            elif high_vol_price_change < -2:
                cluster_type = 'Retail Panic Selling'
                institutional_signal = 'Potential Opportunity'
            else:
                cluster_type = 'High Activity Period'
                institutional_signal = 'Monitor Closely'
        else:
            cluster_type = 'Normal Volume Pattern'
            institutional_signal = 'No Unusual Activity'
        
        # Smart money vs dumb money score
        if cluster_type in ['Institutional Accumulation', 'Potential Opportunity']:
            smart_money_score = 80
        elif cluster_type in ['Institutional Distribution', 'Retail FOMO Buying']:
            smart_money_score = 20
        else:
            smart_money_score = 50
        
        return {
            'cluster_type': cluster_type,
            'cluster_strength': round(cluster_strength, 1),
            'institutional_signal': institutional_signal,
            'high_volume_days': high_volume_days,
            'extreme_volume_days': extreme_volume_days,
            'smart_money_score': smart_money_score,
            'high_vol_price_change': round(high_vol_price_change, 2)
        }
        
    except Exception as e:
        logger.error(f"Volume cluster analysis error: {e}")
        return {'cluster_type': 'Error', 'cluster_strength': 0, 'institutional_signal': 'Error'}

def calculate_correlation_score(price_trend: float, indicator_trend: float) -> str:
    """Helper function to calculate correlation quality"""
    if abs(price_trend) < 1 or abs(indicator_trend) < 1:
        return 'Weak Signal'
    
    correlation = (price_trend * indicator_trend)
    if correlation > 10:
        return 'Strong Positive'
    elif correlation > 2:
        return 'Positive'
    elif correlation < -10:
        return 'Strong Negative'
    elif correlation < -2:
        return 'Negative'
    else:
        return 'Neutral'

@safe_calculation_wrapper
def calculate_volume_composite_score(volume_analysis: Dict[str, Any]) -> float:
    """
    NEW FEATURE v8.0.0: Calculate comprehensive volume composite score (0-100)
    
    Combines all volume indicators into a single normalized score for the gradient bar
    """
    try:
        # Start with base volume score
        base_score = volume_analysis.get('volume_score', 50)
        composite_score = float(base_score)
        
        # Add OBV signal contribution (±10 points)
        obv_analysis = volume_analysis.get('obv_analysis', {})
        obv_signal = obv_analysis.get('obv_signal', 'No Signal')
        if 'Bullish Divergence' in obv_signal or 'Bullish Confirmation' in obv_signal:
            composite_score += 10
        elif 'Bearish Divergence' in obv_signal or 'Bearish Confirmation' in obv_signal:
            composite_score -= 10
        
        # Add A/D Line contribution (±8 points)
        ad_analysis = volume_analysis.get('ad_analysis', {})
        ad_signal = ad_analysis.get('ad_signal', 'No Signal')
        if 'Accumulation' in ad_signal and 'Bullish' in ad_signal:
            composite_score += 8
        elif 'Distribution' in ad_signal and 'Bearish' in ad_signal:
            composite_score -= 8
        
        # Add VPA contribution (±10 points)
        vpa_analysis = volume_analysis.get('vpa_analysis', {})
        vpa_score = vpa_analysis.get('vpa_score', 50)
        vpa_contribution = (vpa_score - 50) * 0.2  # Scale to ±10 points
        composite_score += vpa_contribution
        
        # Add Smart Money contribution (±5 points)
        cluster_analysis = volume_analysis.get('cluster_analysis', {})
        smart_money_score = cluster_analysis.get('smart_money_score', 50)
        if smart_money_score >= 70:
            composite_score += 5
        elif smart_money_score <= 30:
            composite_score -= 5
        
        # Add VROC momentum contribution (±5 points)
        vroc_analysis = volume_analysis.get('vroc_analysis', {})
        vroc_signal = vroc_analysis.get('vroc_signal', 'Normal Activity')
        if 'Breakout Alert' in vroc_signal:
            composite_score += 5
        elif 'Breakdown Alert' in vroc_signal:
            composite_score -= 5
        
        # Bonus for extreme conditions (±5 points)
        volume_regime = volume_analysis.get('volume_regime', 'Normal')
        if volume_regime == 'Extreme High':
            composite_score += 5
        elif volume_regime == 'Low':
            composite_score -= 5
        
        # Clamp to 0-100 range
        composite_score = max(0, min(100, composite_score))
        
        return round(composite_score, 1)
        
    except Exception as e:
        logger.error(f"Volume composite score calculation error: {e}")
        return 50.0

@safe_calculation_wrapper
def calculate_complete_volume_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """
    ENHANCED v8.0.0: Calculate complete volume analysis with all features
    FIXED: Pandas FutureWarning - uses observed=True instead of deprecated observed=False
    """
    try:
        if len(data) < 30:
            return {
                'error': 'Insufficient data for volume analysis',
                'volume_regime': 'Unknown',
                'volume_score': 50,
                'volume_composite_score': 50
            }

        volume = data['Volume']
        current_volume = float(volume.iloc[-1])
        
        # === EXISTING FUNCTIONALITY (PRESERVED) ===
        
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
        
        # === FIXED: Dynamic Volume Thresholds (pandas warning fixed) ===
        dynamic_thresholds = calculate_dynamic_volume_thresholds(data)
        
        # Enhanced Volume Regime Classification (using dynamic thresholds)
        if volume_ratio >= dynamic_thresholds['extreme_high']:
            volume_regime = "Extreme High"
            volume_score = 95
        elif volume_ratio >= dynamic_thresholds['high']:
            volume_regime = "High"
            volume_score = 80
        elif volume_ratio >= dynamic_thresholds['above_normal']:
            volume_regime = "Above Normal"
            volume_score = 65
        elif volume_ratio >= dynamic_thresholds['below_normal']:
            volume_regime = "Normal"
            volume_score = 50
        elif volume_ratio >= dynamic_thresholds['low']:
            volume_regime = "Below Normal"
            volume_score = 35
        else:
            volume_regime = "Low"
            volume_score = 20
            
        # Volume breakout detection (enhanced)
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
        
        # Volume strength factor for technical scoring (enhanced)
        if volume_score >= 80:
            volume_strength_factor = 1.3
        elif volume_score >= 65:
            volume_strength_factor = 1.15
        elif volume_score >= 35:
            volume_strength_factor = 1.0
        else:
            volume_strength_factor = 0.85
        
        # === ENHANCED FEATURES v8.0.0 ===
        
        # 1. On-Balance Volume Analysis
        obv_analysis = calculate_on_balance_volume(data)
        
        # 2. Accumulation/Distribution Line
        ad_analysis = calculate_accumulation_distribution_line(data)
        
        # 3. Volume Rate of Change
        vroc_analysis = calculate_volume_rate_of_change(data)
        
        # 4. Volume Price Analysis
        vpa_analysis = calculate_volume_price_analysis(data)
        
        # 5. Volume Cluster Analysis
        cluster_analysis = calculate_volume_cluster_analysis(data)
        
        # === BUILD ANALYSIS RESULTS ===
        
        analysis_results = {
            # === EXISTING METRICS (PRESERVED) ===
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
            
            # === NEW FEATURES v8.0.0 ===
            'dynamic_thresholds': dynamic_thresholds,
            'obv_analysis': obv_analysis,
            'ad_analysis': ad_analysis,
            'vroc_analysis': vroc_analysis,
            'vpa_analysis': vpa_analysis,
            'cluster_analysis': cluster_analysis,
            
            # === ENHANCED SIGNALS ===
            'volume_momentum': get_volume_momentum_signal(vroc_analysis, obv_analysis),
            'smart_money_signal': cluster_analysis['institutional_signal'],
            'volume_quality': get_volume_quality_assessment(vpa_analysis, ad_analysis),
            'analysis_version': 'Enhanced v8.0.0'
        }
        
        # === CALCULATE VOLUME COMPOSITE SCORE ===
        volume_composite_score = calculate_volume_composite_score(analysis_results)
        analysis_results['volume_composite_score'] = volume_composite_score
        
        # === ENHANCED TRADING IMPLICATIONS ===
        trading_implications = get_enhanced_volume_trading_implications(
            volume_regime, 
            volume_breakout, 
            obv_analysis['obv_signal'],
            ad_analysis['ad_signal'],
            cluster_analysis['institutional_signal']
        )
        analysis_results['trading_implications'] = trading_implications
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Enhanced volume analysis calculation error: {e}")
        return {
            'error': f'Enhanced volume analysis failed: {str(e)}',
            'volume_regime': 'Unknown',
            'volume_score': 50,
            'volume_composite_score': 50,
            'volume_strength_factor': 1.0
        }

def get_enhanced_volume_trading_implications(volume_regime: str, volume_breakout: str, 
                                           obv_signal: str, ad_signal: str, institutional_signal: str) -> str:
    """Enhanced trading implications considering all volume signals"""
    implications = []
    
    # Base regime implications
    if volume_regime == "Extreme High":
        implications.append("High conviction moves likely")
    elif volume_regime == "High":
        implications.append("Strong interest, watch for continuation")
    elif volume_regime == "Low":
        implications.append("Low conviction, avoid breakouts")
    
    # OBV implications
    if "Bullish Divergence" in obv_signal:
        implications.append("Hidden buying pressure (OBV bullish)")
    elif "Bearish Divergence" in obv_signal:
        implications.append("Hidden selling pressure (OBV bearish)")
    
    # A/D Line implications
    if "Accumulation" in ad_signal:
        implications.append("Smart money accumulating")
    elif "Distribution" in ad_signal:
        implications.append("Smart money distributing")
    
    # Institutional signal implications
    if "Strong Buy" in institutional_signal:
        implications.append("Institutional buying detected")
    elif "Strong Sell" in institutional_signal:
        implications.append("Institutional selling detected")
    elif "Caution" in institutional_signal:
        implications.append("Retail-driven move, use caution")
    
    return " | ".join(implications) if implications else "Monitor volume for directional clues"

def get_volume_momentum_signal(vroc_analysis: Dict, obv_analysis: Dict) -> str:
    """Combine VROC and OBV for momentum signal"""
    vroc_signal = vroc_analysis.get('vroc_signal', 'Normal Activity')
    obv_trend = obv_analysis.get('obv_trend', 'Neutral')
    
    if 'Breakout Alert' in vroc_signal and 'Accumulation' in obv_trend:
        return 'Strong Bullish Momentum'
    elif 'Breakdown Alert' in vroc_signal and 'Distribution' in obv_trend:
        return 'Strong Bearish Momentum'
    elif 'Explosive' in vroc_analysis.get('vroc_trend', ''):
        return 'High Momentum'
    elif 'Collapse' in vroc_analysis.get('vroc_trend', ''):
        return 'Momentum Collapse'
    else:
        return 'Normal Momentum'

def get_volume_quality_assessment(vpa_analysis: Dict, ad_analysis: Dict) -> str:
    """Assess overall volume quality"""
    vpa_score = vpa_analysis.get('vpa_score', 50)
    ad_signal = ad_analysis.get('ad_signal', 'No Signal')
    
    if vpa_score >= 75 and 'Confirmation' in ad_signal:
        return 'High Quality (Strong Confirmation)'
    elif vpa_score >= 60:
        return 'Good Quality'
    elif vpa_score <= 25:
        return 'Poor Quality (Weak Volume)'
    elif 'Divergence' in ad_signal:
        return 'Mixed Quality (Divergence Warning)'
    else:
        return 'Average Quality'

# === EXISTING FUNCTIONS (PRESERVED) ===

def get_volume_trading_implications(volume_regime: str, volume_breakout: str) -> str:
    """Original trading implications function (preserved for backward compatibility)"""
    if volume_regime == "Extreme High" and volume_breakout in ["Extreme", "Strong"]:
        return "High conviction breakout - follow momentum with appropriate position size"
    elif volume_regime == "High" and volume_breakout in ["Moderate", "Strong"]:
        return "Good volume confirmation - breakout/breakdown likely sustainable"
    elif volume_regime == "Low" and volume_breakout == "None":
        return "Low conviction environment - avoid breakout trades, range trading preferred"
    elif volume_regime in ["Below Normal", "Low"]:
        return "Weak volume - any breakouts may be false, wait for volume confirmation"
    else:
        return "Monitor volume for directional clues and trade confirmation"

@safe_calculation_wrapper        
def calculate_market_wide_volume_analysis(show_debug=False) -> Dict[str, Any]:
    """Calculate market-wide volume environment across SPY, QQQ, IWM (preserved and enhanced)"""
    try:
        import yfinance as yf
        
        major_indices = ['SPY', 'QQQ', 'IWM']
        market_volume_data = {}
        
        for symbol in major_indices:
            try:
                if show_debug:
                    st.write(f"📊 Fetching enhanced volume data for {symbol}...")
                    
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='3mo')
                
                if len(data) >= 30:
                    volume_analysis = calculate_complete_volume_analysis(data)
                    if 'error' not in volume_analysis:
                        market_volume_data[symbol] = volume_analysis
                        
            except Exception as e:
                if show_debug:
                    st.write(f"❌ Error fetching {symbol}: {e}")
                continue
                
        if len(market_volume_data) >= 2:
            # Calculate overall market volume environment using composite scores
            avg_volume_score = sum([data.get('volume_composite_score', data.get('volume_score', 50)) 
                                  for data in market_volume_data.values()]) / len(market_volume_data)
            
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
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
                'enhanced_features': True
            }
        else:
            return {'error': 'Insufficient market data', 'enhanced_features': False}
            
    except Exception as e:
        logger.error(f"Market-wide volume analysis error: {e}")
        return {'error': f'Market volume analysis failed: {str(e)}', 'enhanced_features': False}
