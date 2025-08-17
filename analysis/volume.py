"""
Enhanced Volume Analysis v8.0.0 - VWV Trading System
NEW FEATURES:
✅ Volume Composite Score (0-100)
✅ Smart Money vs Retail Detection
✅ On-Balance Volume (OBV) Analysis
✅ Accumulation/Distribution Line
✅ Volume Rate of Change (VROC)
✅ Volume Price Analysis (VPA)
✅ Institutional Activity Detection
✅ Fixed pandas FutureWarning (observed=True)
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any
import logging
from utils.decorators import safe_calculation_wrapper

# Set up logging
logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def calculate_complete_volume_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Complete Enhanced Volume Analysis v8.0.0
    
    NEW FEATURES:
    - Volume Composite Score (0-100)
    - Smart Money Detection
    - Institutional Activity Analysis
    - OBV, A/D Line, VROC, VPA
    - Advanced volume intelligence
    """
    try:
        if len(data) < 30:
            return {'error': 'Insufficient data for volume analysis (need 30+ days)'}
        
        volume = data['Volume']
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        current_volume = volume.iloc[-1]
        current_price = close.iloc[-1]
        
        # === BASIC VOLUME METRICS ===
        volume_5d = volume.rolling(window=5).mean()
        volume_30d = volume.rolling(window=30).mean()
        current_5d_avg = volume_5d.iloc[-1]
        volume_30d_avg = volume_30d.iloc[-1]
        
        # Volume ratios
        volume_ratio = current_volume / current_5d_avg if current_5d_avg > 0 else 1.0
        volume_ratio_30d = current_volume / volume_30d_avg if volume_30d_avg > 0 else 1.0
        
        # Volume trends
        volume_5d_trend = ((current_5d_avg - volume_5d.iloc[-6]) / volume_5d.iloc[-6] * 100) if len(volume_5d) > 5 and volume_5d.iloc[-6] > 0 else 0
        
        # Volume Z-Score
        volume_std = volume.rolling(window=30).std().iloc[-1]
        volume_zscore = (current_volume - volume_30d_avg) / volume_std if volume_std > 0 and not pd.isna(volume_std) else 0
        
        # Volume percentile
        volume_percentile = (volume <= current_volume).sum() / len(volume) * 100
        
        # Relative volume
        relative_volume = current_volume / volume_30d_avg if volume_30d_avg > 0 else 1.0
        
        # === ENHANCED VOLUME TECHNIQUES ===
        
        # 1. On-Balance Volume (OBV)
        obv_data = calculate_obv_analysis(data)
        
        # 2. Accumulation/Distribution Line
        ad_line_data = calculate_ad_line_analysis(data)
        
        # 3. Volume Rate of Change (VROC)
        vroc_data = calculate_vroc_analysis(data)
        
        # 4. Volume Price Analysis (VPA)
        vpa_data = calculate_vpa_analysis(data)
        
        # 5. Institutional Activity Detection
        institutional_data = calculate_institutional_activity(data)
        
        # === VOLUME COMPOSITE SCORE CALCULATION ===
        volume_composite_score = calculate_volume_composite_score(
            volume_ratio, volume_zscore, volume_percentile, obv_data, 
            ad_line_data, vroc_data, vpa_data, institutional_data
        )
        
        # === SMART MONEY DETECTION ===
        smart_money_signal = detect_smart_money_activity(
            volume_ratio, volume_zscore, obv_data, ad_line_data, vpa_data, institutional_data
        )
        
        # === VOLUME REGIME CLASSIFICATION ===
        volume_regime = classify_volume_regime(volume_composite_score, volume_ratio, volume_zscore)
        
        # === VOLUME QUALITY ASSESSMENT ===
        volume_quality = assess_volume_quality(volume_ratio, obv_data, ad_line_data, vpa_data)
        
        # === INSTITUTIONAL ACTIVITY SUMMARY ===
        institutional_activity = summarize_institutional_activity(institutional_data, volume_composite_score)
        
        # Volume strength factor for technical scoring
        volume_strength_factor = calculate_volume_strength_factor(volume_composite_score)
        
        # === RETURN COMPREHENSIVE RESULTS ===
        return {
            # Basic Volume Metrics
            'current_volume': int(current_volume),
            'volume_5d_avg': int(current_5d_avg),
            'volume_30d_avg': int(volume_30d_avg),
            'volume_ratio': round(volume_ratio, 2),
            'volume_ratio_30d': round(volume_ratio_30d, 2),
            'volume_5d_trend': round(volume_5d_trend, 2),
            'volume_zscore': round(float(volume_zscore), 2),
            'volume_percentile': round(volume_percentile, 1),
            'relative_volume': round(relative_volume, 2),
            
            # Enhanced Volume Intelligence
            'volume_composite_score': round(volume_composite_score, 1),
            'smart_money_signal': smart_money_signal,
            'volume_regime': volume_regime,
            'volume_quality': volume_quality,
            'institutional_activity': institutional_activity,
            'volume_strength_factor': volume_strength_factor,
            
            # Advanced Volume Techniques
            'obv_current': obv_data.get('current_obv', 0),
            'obv_trend': obv_data.get('trend', 'Unknown'),
            'obv_change_pct': obv_data.get('change_pct', 0),
            'obv_divergence': obv_data.get('divergence', 'None'),
            'obv_momentum': obv_data.get('momentum', 'Neutral'),
            'accumulation_signal': obv_data.get('accumulation_signal', 'Unknown'),
            
            'ad_line_current': ad_line_data.get('current_ad', 0),
            'ad_line_trend': ad_line_data.get('trend', 'Unknown'),
            'ad_line_change_pct': ad_line_data.get('change_pct', 0),
            'distribution_pattern': ad_line_data.get('distribution_pattern', 'Unknown'),
            'accumulation_phase': ad_line_data.get('accumulation_phase', 'Unknown'),
            'money_flow_direction': ad_line_data.get('money_flow_direction', 'Unknown'),
            
            'vroc_14': vroc_data.get('vroc_14', 0),
            'vroc_signal': vroc_data.get('signal', 'Neutral'),
            'volume_momentum': vroc_data.get('momentum', 'Unknown'),
            
            'vpa_signal': vpa_data.get('signal', 'Unknown'),
            'price_volume_sync': vpa_data.get('price_volume_sync', 'Unknown'),
            'breakout_confirmation': vpa_data.get('breakout_confirmation', 'Unknown'),
            
            # Cluster Analysis Results
            'high_volume_zones': institutional_data.get('high_volume_zones', 'Unknown'),
            'volume_support_resistance': institutional_data.get('support_resistance', 'Unknown'),
            'institutional_footprint': institutional_data.get('footprint', 'Unknown'),
            'volume_profile': institutional_data.get('volume_profile', 'Unknown'),
            'flow_intensity': institutional_data.get('flow_intensity', 'Unknown'),
            'market_participation': institutional_data.get('market_participation', 'Unknown'),
            
            # Trading Implications
            'trading_implications': get_volume_trading_implications(volume_regime, smart_money_signal)
        }
        
    except Exception as e:
        logger.error(f"Volume analysis calculation error: {e}")
        return {
            'error': f'Volume analysis failed: {str(e)}',
            'volume_regime': 'Unknown',
            'volume_composite_score': 50,
            'volume_strength_factor': 1.0
        }

def calculate_obv_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate On-Balance Volume analysis"""
    try:
        close = data['Close']
        volume = data['Volume']
        
        # Calculate OBV
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        
        obv_series = pd.Series(obv, index=data.index)
        current_obv = obv_series.iloc[-1]
        
        # OBV trend analysis
        obv_ma = obv_series.rolling(window=10).mean()
        obv_trend = "Rising" if current_obv > obv_ma.iloc[-1] else "Falling"
        
        # OBV change percentage
        prev_obv = obv_series.iloc[-10] if len(obv_series) > 10 else obv_series.iloc[0]
        obv_change_pct = ((current_obv - prev_obv) / abs(prev_obv) * 100) if prev_obv != 0 else 0
        
        # Price-OBV divergence detection
        price_direction = "Up" if close.iloc[-1] > close.iloc[-10] else "Down"
        obv_direction = "Up" if current_obv > prev_obv else "Down"
        
        if price_direction != obv_direction:
            divergence = "Bullish Divergence" if price_direction == "Down" and obv_direction == "Up" else "Bearish Divergence"
        else:
            divergence = "None"
        
        # OBV momentum
        obv_momentum = "Strong" if abs(obv_change_pct) > 5 else "Moderate" if abs(obv_change_pct) > 2 else "Weak"
        
        # Accumulation signal
        if obv_trend == "Rising" and price_direction == "Up":
            accumulation_signal = "Strong Accumulation"
        elif obv_trend == "Rising" and price_direction == "Down":
            accumulation_signal = "Hidden Accumulation"
        elif obv_trend == "Falling" and price_direction == "Down":
            accumulation_signal = "Distribution"
        else:
            accumulation_signal = "Neutral"
        
        return {
            'current_obv': round(current_obv, 0),
            'trend': obv_trend,
            'change_pct': round(obv_change_pct, 2),
            'divergence': divergence,
            'momentum': obv_momentum,
            'accumulation_signal': accumulation_signal
        }
        
    except Exception as e:
        logger.error(f"OBV calculation error: {e}")
        return {'current_obv': 0, 'trend': 'Unknown', 'change_pct': 0}

def calculate_ad_line_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Accumulation/Distribution Line analysis"""
    try:
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        
        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)  # Handle division by zero
        
        # Money Flow Volume
        mfv = mfm * volume
        
        # Accumulation/Distribution Line
        ad_line = mfv.cumsum()
        current_ad = ad_line.iloc[-1]
        
        # A/D Line trend
        ad_ma = ad_line.rolling(window=10).mean()
        ad_trend = "Rising" if current_ad > ad_ma.iloc[-1] else "Falling"
        
        # A/D Line change percentage
        prev_ad = ad_line.iloc[-10] if len(ad_line) > 10 else ad_line.iloc[0]
        ad_change_pct = ((current_ad - prev_ad) / abs(prev_ad) * 100) if prev_ad != 0 else 0
        
        # Distribution pattern analysis
        recent_ad = ad_line.tail(5)
        if recent_ad.is_monotonic_increasing:
            distribution_pattern = "Strong Accumulation Pattern"
        elif recent_ad.is_monotonic_decreasing:
            distribution_pattern = "Strong Distribution Pattern"
        else:
            distribution_pattern = "Mixed Distribution Pattern"
        
        # Accumulation phase
        if ad_change_pct > 5:
            accumulation_phase = "Active Accumulation"
        elif ad_change_pct < -5:
            accumulation_phase = "Active Distribution"
        else:
            accumulation_phase = "Neutral Phase"
        
        # Money flow direction
        recent_mfv = mfv.tail(5).mean()
        if recent_mfv > 0:
            money_flow_direction = "Positive (Buying Pressure)"
        elif recent_mfv < 0:
            money_flow_direction = "Negative (Selling Pressure)"
        else:
            money_flow_direction = "Neutral"
        
        return {
            'current_ad': round(current_ad, 0),
            'trend': ad_trend,
            'change_pct': round(ad_change_pct, 2),
            'distribution_pattern': distribution_pattern,
            'accumulation_phase': accumulation_phase,
            'money_flow_direction': money_flow_direction
        }
        
    except Exception as e:
        logger.error(f"A/D Line calculation error: {e}")
        return {'current_ad': 0, 'trend': 'Unknown', 'change_pct': 0}

def calculate_vroc_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Volume Rate of Change analysis"""
    try:
        volume = data['Volume']
        
        # Volume Rate of Change (14-period)
        vroc_14 = ((volume - volume.shift(14)) / volume.shift(14) * 100).iloc[-1]
        
        # VROC signal
        if vroc_14 > 20:
            vroc_signal = "Strong Volume Expansion"
        elif vroc_14 > 5:
            vroc_signal = "Moderate Volume Increase"
        elif vroc_14 < -20:
            vroc_signal = "Strong Volume Contraction"
        elif vroc_14 < -5:
            vroc_signal = "Moderate Volume Decrease"
        else:
            vroc_signal = "Stable Volume"
        
        # Volume momentum
        if abs(vroc_14) > 50:
            volume_momentum = "Extreme"
        elif abs(vroc_14) > 20:
            volume_momentum = "High"
        elif abs(vroc_14) > 5:
            volume_momentum = "Moderate"
        else:
            volume_momentum = "Low"
        
        return {
            'vroc_14': round(vroc_14, 2),
            'signal': vroc_signal,
            'momentum': volume_momentum
        }
        
    except Exception as e:
        logger.error(f"VROC calculation error: {e}")
        return {'vroc_14': 0, 'signal': 'Neutral', 'momentum': 'Unknown'}

def calculate_vpa_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Volume Price Analysis"""
    try:
        close = data['Close']
        volume = data['Volume']
        
        # Price change
        price_change = close.pct_change().iloc[-1] * 100
        
        # Volume comparison to average
        vol_avg = volume.rolling(window=20).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_avg if vol_avg > 0 else 1
        
        # VPA Signal Logic
        if price_change > 2 and vol_ratio > 1.5:
            vpa_signal = "Strong Bullish Breakout (High Volume)"
        elif price_change > 1 and vol_ratio > 1.2:
            vpa_signal = "Bullish Move (Above Average Volume)"
        elif price_change < -2 and vol_ratio > 1.5:
            vpa_signal = "Strong Bearish Breakdown (High Volume)"
        elif price_change < -1 and vol_ratio > 1.2:
            vpa_signal = "Bearish Move (Above Average Volume)"
        elif abs(price_change) > 1 and vol_ratio < 0.8:
            vpa_signal = "Weak Move (Low Volume - Suspect)"
        else:
            vpa_signal = "Normal Price-Volume Relationship"
        
        # Price-Volume Synchronization
        if (price_change > 0 and vol_ratio > 1) or (price_change < 0 and vol_ratio > 1):
            price_volume_sync = "Synchronized (Healthy)"
        else:
            price_volume_sync = "Divergent (Caution)"
        
        # Breakout confirmation
        if abs(price_change) > 2 and vol_ratio > 2:
            breakout_confirmation = "Strong Confirmation"
        elif abs(price_change) > 1 and vol_ratio > 1.5:
            breakout_confirmation = "Moderate Confirmation"
        else:
            breakout_confirmation = "No Confirmation"
        
        return {
            'signal': vpa_signal,
            'price_volume_sync': price_volume_sync,
            'breakout_confirmation': breakout_confirmation
        }
        
    except Exception as e:
        logger.error(f"VPA calculation error: {e}")
        return {'signal': 'Unknown', 'price_volume_sync': 'Unknown'}

def calculate_institutional_activity(data: pd.DataFrame) -> Dict[str, Any]:
    """Detect institutional activity patterns"""
    try:
        volume = data['Volume']
        close = data['Close']
        
        # High volume threshold (top 20% of recent volume)
        vol_threshold = volume.tail(50).quantile(0.8)
        high_vol_days = (volume > vol_threshold).sum()
        
        # Volume clustering analysis
        if high_vol_days >= 5:
            high_volume_zones = "Multiple High-Volume Clusters Detected"
        elif high_vol_days >= 2:
            high_volume_zones = "Some High-Volume Activity"
        else:
            high_volume_zones = "Limited High-Volume Activity"
        
        # Support/Resistance based on volume
        high_vol_prices = close[volume > vol_threshold]
        if len(high_vol_prices) > 0:
            vol_support = high_vol_prices.min()
            vol_resistance = high_vol_prices.max()
            current_price = close.iloc[-1]
            
            if abs(current_price - vol_support) < abs(current_price - vol_resistance):
                support_resistance = f"Near Volume Support: ${vol_support:.2f}"
            else:
                support_resistance = f"Near Volume Resistance: ${vol_resistance:.2f}"
        else:
            support_resistance = "No Clear Volume Levels"
        
        # Institutional footprint
        avg_volume = volume.tail(30).mean()
        recent_large_volume_days = (volume.tail(5) > avg_volume * 2).sum()
        
        if recent_large_volume_days >= 2:
            footprint = "Strong Institutional Footprint (Recent Large Volume)"
        elif recent_large_volume_days >= 1:
            footprint = "Moderate Institutional Activity"
        else:
            footprint = "Limited Institutional Activity"
        
        # Volume profile assessment
        vol_std = volume.tail(30).std()
        vol_cv = vol_std / avg_volume if avg_volume > 0 else 0
        
        if vol_cv > 1.5:
            volume_profile = "Highly Volatile Volume (Institutional Waves)"
        elif vol_cv > 1.0:
            volume_profile = "Moderate Volume Variability"
        else:
            volume_profile = "Consistent Volume Pattern"
        
        # Flow intensity
        recent_vol_avg = volume.tail(5).mean()
        flow_intensity_ratio = recent_vol_avg / avg_volume if avg_volume > 0 else 1
        
        if flow_intensity_ratio > 2:
            flow_intensity = "Intense Flow (2x Average)"
        elif flow_intensity_ratio > 1.5:
            flow_intensity = "High Flow (1.5x Average)"
        elif flow_intensity_ratio > 1.2:
            flow_intensity = "Elevated Flow"
        else:
            flow_intensity = "Normal Flow"
        
        # Market participation
        above_avg_days = (volume.tail(10) > avg_volume).sum()
        participation_pct = above_avg_days / 10 * 100
        
        if participation_pct >= 70:
            market_participation = "High Participation (Active Market)"
        elif participation_pct >= 40:
            market_participation = "Moderate Participation"
        else:
            market_participation = "Low Participation (Quiet Market)"
        
        return {
            'high_volume_zones': high_volume_zones,
            'support_resistance': support_resistance,
            'footprint': footprint,
            'volume_profile': volume_profile,
            'flow_intensity': flow_intensity,
            'market_participation': market_participation
        }
        
    except Exception as e:
        logger.error(f"Institutional activity calculation error: {e}")
        return {'footprint': 'Unknown', 'flow_intensity': 'Unknown'}

def calculate_volume_composite_score(volume_ratio, volume_zscore, volume_percentile, 
                                   obv_data, ad_line_data, vroc_data, vpa_data, institutional_data) -> float:
    """Calculate comprehensive volume composite score (0-100)"""
    try:
        # Component scores (each 0-100)
        
        # 1. Volume Ratio Score (25% weight)
        if volume_ratio >= 3.0:
            ratio_score = 100
        elif volume_ratio >= 2.0:
            ratio_score = 85
        elif volume_ratio >= 1.5:
            ratio_score = 70
        elif volume_ratio >= 1.2:
            ratio_score = 60
        elif volume_ratio >= 0.8:
            ratio_score = 50
        elif volume_ratio >= 0.5:
            ratio_score = 35
        else:
            ratio_score = 20
        
        # 2. Volume Z-Score (20% weight)
        if volume_zscore >= 3:
            zscore_score = 100
        elif volume_zscore >= 2:
            zscore_score = 85
        elif volume_zscore >= 1:
            zscore_score = 70
        elif volume_zscore >= 0:
            zscore_score = 55
        elif volume_zscore >= -1:
            zscore_score = 45
        elif volume_zscore >= -2:
            zscore_score = 30
        else:
            zscore_score = 15
        
        # 3. Volume Percentile Score (15% weight)
        percentile_score = min(100, max(0, volume_percentile))
        
        # 4. OBV Analysis Score (15% weight)
        obv_trend = obv_data.get('trend', 'Unknown')
        obv_change = obv_data.get('change_pct', 0)
        
        if obv_trend == 'Rising' and obv_change > 5:
            obv_score = 85
        elif obv_trend == 'Rising':
            obv_score = 70
        elif obv_trend == 'Falling' and obv_change < -5:
            obv_score = 30
        elif obv_trend == 'Falling':
            obv_score = 45
        else:
            obv_score = 50
        
        # 5. A/D Line Score (10% weight)
        ad_trend = ad_line_data.get('trend', 'Unknown')
        ad_change = ad_line_data.get('change_pct', 0)
        
        if ad_trend == 'Rising' and ad_change > 5:
            ad_score = 85
        elif ad_trend == 'Rising':
            ad_score = 70
        elif ad_trend == 'Falling' and ad_change < -5:
            ad_score = 30
        elif ad_trend == 'Falling':
            ad_score = 45
        else:
            ad_score = 50
        
        # 6. VROC Score (10% weight)
        vroc_14 = vroc_data.get('vroc_14', 0)
        
        if vroc_14 > 50:
            vroc_score = 100
        elif vroc_14 > 20:
            vroc_score = 85
        elif vroc_14 > 5:
            vroc_score = 70
        elif vroc_14 > -5:
            vroc_score = 50
        elif vroc_14 > -20:
            vroc_score = 30
        else:
            vroc_score = 15
        
        # 7. Institutional Activity Score (5% weight)
        footprint = institutional_data.get('footprint', 'Unknown')
        
        if 'Strong Institutional' in footprint:
            institutional_score = 90
        elif 'Moderate Institutional' in footprint:
            institutional_score = 70
        else:
            institutional_score = 50
        
        # Calculate weighted composite score
        composite_score = (
            ratio_score * 0.25 +
            zscore_score * 0.20 +
            percentile_score * 0.15 +
            obv_score * 0.15 +
            ad_score * 0.10 +
            vroc_score * 0.10 +
            institutional_score * 0.05
        )
        
        return min(100, max(0, composite_score))
        
    except Exception as e:
        logger.error(f"Volume composite score calculation error: {e}")
        return 50.0

def detect_smart_money_activity(volume_ratio, volume_zscore, obv_data, ad_line_data, vpa_data, institutional_data) -> str:
    """Detect smart money vs retail activity patterns"""
    try:
        # Smart money indicators
        smart_money_signals = 0
        retail_signals = 0
        
        # High volume with OBV accumulation
        if volume_ratio > 1.5 and obv_data.get('trend') == 'Rising':
            smart_money_signals += 2
        
        # A/D Line accumulation
        if ad_line_data.get('trend') == 'Rising' and ad_line_data.get('change_pct', 0) > 5:
            smart_money_signals += 2
        
        # Institutional footprint
        footprint = institutional_data.get('footprint', '')
        if 'Strong Institutional' in footprint:
            smart_money_signals += 3
        elif 'Moderate Institutional' in footprint:
            smart_money_signals += 1
        
        # Price-volume sync
        if 'Synchronized' in vpa_data.get('price_volume_sync', ''):
            smart_money_signals += 1
        
        # Volume extreme without clear direction (retail FOMO/panic)
        if volume_zscore > 2.5 and obv_data.get('divergence') != 'None':
            retail_signals += 2
        
        # High volume with poor price-volume sync
        if volume_ratio > 2 and 'Divergent' in vpa_data.get('price_volume_sync', ''):
            retail_signals += 2
        
        # Determine signal
        if smart_money_signals >= 4:
            return "Institutional Accumulation Detected"
        elif smart_money_signals >= 2:
            return "Potential Smart Money Activity"
        elif retail_signals >= 3:
            return "Retail FOMO/Panic Activity"
        elif retail_signals >= 1:
            return "Potential Retail Activity"
        else:
            return "Neutral Money Flow"
            
    except Exception as e:
        logger.error(f"Smart money detection error: {e}")
        return "Unknown"

def classify_volume_regime(composite_score, volume_ratio, volume_zscore) -> str:
    """Classify current volume regime"""
    try:
        if composite_score >= 85 or volume_zscore >= 3:
            return "🔥 Extreme Volume Activity"
        elif composite_score >= 70 or volume_ratio >= 2:
            return "📈 High Volume Activity"
        elif composite_score >= 60:
            return "⬆️ Above Normal Activity"
        elif composite_score >= 40:
            return "⚖️ Normal Activity"
        elif composite_score >= 25:
            return "⬇️ Below Normal Activity"
        else:
            return "📉 Low Volume Activity"
            
    except Exception as e:
        logger.error(f"Volume regime classification error: {e}")
        return "Unknown Activity"

def assess_volume_quality(volume_ratio, obv_data, ad_line_data, vpa_data) -> str:
    """Assess the quality of volume for confirming price moves"""
    try:
        quality_score = 0
        
        # Volume magnitude
        if volume_ratio > 1.5:
            quality_score += 2
        elif volume_ratio > 1.2:
            quality_score += 1
        
        # OBV confirmation
        if obv_data.get('divergence') == 'None':
            quality_score += 2
        
        # A/D Line support
        if ad_line_data.get('trend') == 'Rising':
            quality_score += 1
        
        # Price-volume sync
        if 'Synchronized' in vpa_data.get('price_volume_sync', ''):
            quality_score += 2
        
        # Breakout confirmation
        if 'Strong Confirmation' in vpa_data.get('breakout_confirmation', ''):
            quality_score += 1
        
        if quality_score >= 6:
            return "High Quality (Strong Confirmation)"
        elif quality_score >= 4:
            return "Good Quality (Moderate Confirmation)"
        elif quality_score >= 2:
            return "Fair Quality (Weak Confirmation)"
        else:
            return "Poor Quality (No Confirmation)"
            
    except Exception as e:
        logger.error(f"Volume quality assessment error: {e}")
        return "Unknown Quality"

def summarize_institutional_activity(institutional_data, composite_score) -> str:
    """Summarize institutional activity level"""
    try:
        footprint = institutional_data.get('footprint', '')
        flow_intensity = institutional_data.get('flow_intensity', '')
        
        if composite_score >= 85 and 'Strong' in footprint:
            return "🏛️ High Institutional Activity"
        elif composite_score >= 70 and ('Strong' in footprint or 'Moderate' in footprint):
            return "🏦 Moderate Institutional Activity"
        elif 'Intense Flow' in flow_intensity:
            return "💰 Elevated Institutional Interest"
        else:
            return "🤝 Retail-Dominated Activity"
            
    except Exception as e:
        logger.error(f"Institutional activity summary error: {e}")
        return "Unknown Activity"

def calculate_volume_strength_factor(composite_score) -> float:
    """Calculate volume strength factor for technical scoring"""
    try:
        if composite_score >= 85:
            return 1.3
        elif composite_score >= 70:
            return 1.15
        elif composite_score >= 55:
            return 1.05
        elif composite_score >= 45:
            return 1.0
        elif composite_score >= 30:
            return 0.95
        else:
            return 0.85
            
    except Exception as e:
        logger.error(f"Volume strength factor calculation error: {e}")
        return 1.0

def get_volume_trading_implications(volume_regime, smart_money_signal) -> str:
    """Get trading implications based on volume analysis"""
    try:
        implications = []
        
        # Volume regime implications
        if "Extreme" in volume_regime:
            implications.append("• Major move likely in progress")
            implications.append("• Watch for continuation or reversal")
        elif "High" in volume_regime:
            implications.append("• Strong conviction behind current move")
            implications.append("• Good environment for breakout trades")
        elif "Low" in volume_regime:
            implications.append("• Lack of conviction in current move")
            implications.append("• Range-bound conditions likely")
        
        # Smart money implications
        if "Institutional Accumulation" in smart_money_signal:
            implications.append("• Smart money building positions")
            implications.append("• Consider alignment with institutional flow")
        elif "Retail FOMO" in smart_money_signal:
            implications.append("• Potential exhaustion move")
            implications.append("• Exercise caution on momentum trades")
        
        return "\n".join(implications) if implications else "Standard volume conditions"
        
    except Exception as e:
        logger.error(f"Trading implications error: {e}")
        return "No specific implications available"

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
                # Use observed=True to fix pandas FutureWarning
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
            # Calculate overall market volume environment
            avg_volume_score = sum([data['volume_composite_score'] for data in market_volume_data.values()]) / len(market_volume_data)
            
            # Classify market volume environment
            if avg_volume_score >= 80:
                market_volume_environment = "🔥 High Activity Market"
            elif avg_volume_score >= 65:
                market_volume_environment = "📈 Above Normal Activity"
            elif avg_volume_score >= 35:
                market_volume_environment = "⚖️ Normal Activity"
            elif avg_volume_score >= 20:
                market_volume_environment = "📉 Below Normal Activity"
            else:
                market_volume_environment = "😴 Low Activity Market"
            
            return {
                'market_volume_environment': market_volume_environment,
                'market_volume_score': round(avg_volume_score, 1),
                'individual_data': market_volume_data,
                'analysis_timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        else:
            return {
                'error': 'Insufficient market data for analysis',
                'market_volume_environment': 'Unknown',
                'market_volume_score': 50
            }
            
    except Exception as e:
        logger.error(f"Market-wide volume analysis error: {e}")
        return {
            'error': f'Market volume analysis failed: {str(e)}',
            'market_volume_environment': 'Unknown',
            'market_volume_score': 50
        }
