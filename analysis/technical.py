"""
Technical analysis indicators and calculations - UPDATED with Volume/Volatility Integration
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, Optional
from utils.decorators import safe_calculation_wrapper
from utils.helpers import statistical_normalize
from config.settings import FIBONACCI_EMA_PERIODS, TECHNICAL_PERIODS
import logging

# Import new analysis modules
from analysis.volume import calculate_complete_volume_analysis
from analysis.volatility import calculate_complete_volatility_analysis

logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def safe_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Safe RSI calculation with proper error handling"""
    try:
        if len(prices) < period + 1:
            return pd.Series([50] * len(prices), index=prices.index)
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)
        return rsi
    except Exception as e:
        logger.error(f"RSI calculation error: {e}")
        return pd.Series([50] * len(prices), index=prices.index)

@safe_calculation_wrapper
def calculate_daily_vwap(data: pd.DataFrame) -> float:
    """Enhanced daily VWAP calculation"""
    try:
        if not hasattr(data, 'index') or not hasattr(data, 'columns'):
            return float(data['Close'].iloc[-1]) if 'Close' in data else 0.0

        if len(data) < 5:
            return float(data['Close'].iloc[-1])

        recent_data = data.tail(20)

        if 'Typical_Price' in recent_data.columns and 'Volume' in recent_data.columns:
            total_pv = (recent_data['Typical_Price'] * recent_data['Volume']).sum()
            total_volume = recent_data['Volume'].sum()

            if total_volume > 0:
                return float(total_pv / total_volume)

        return float(data['Close'].iloc[-1])
    except Exception:
        try:
            return float(data['Close'].iloc[-1])
        except:
            return 0.0

@safe_calculation_wrapper
def calculate_fibonacci_emas(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate Fibonacci EMAs (21, 55, 89, 144, 233)"""
    try:
        if len(data) < 21:
            return {}

        close = data['Close']
        emas = {}

        for period in FIBONACCI_EMA_PERIODS:
            if len(close) >= period:
                ema_value = close.ewm(span=period).mean().iloc[-1]
                emas[f'EMA_{period}'] = round(float(ema_value), 2)

        return emas
    except Exception:
        return {}

@safe_calculation_wrapper
def calculate_point_of_control_enhanced(data: pd.DataFrame) -> Optional[float]:
    """Enhanced Point of Control with better volume weighting"""
    try:
        if len(data) < 20:
            return None

        # Use recent data for daily POC
        recent_data = data.tail(20)

        # Create price bins and sum volume for each bin
        price_range = recent_data['High'].max() - recent_data['Low'].min()
        bin_size = price_range / 50  # 50 price bins

        if bin_size <= 0:
            return float(recent_data['Close'].iloc[-1])

        # Calculate volume profile with better weighting
        volume_profile = {}

        for idx, row in recent_data.iterrows():
            total_volume = row['Volume']
            open_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            
            # Determine if it's a bullish or bearish bar
            is_bullish = close_price >= open_price
            
            # Enhanced volume distribution weighting
            if is_bullish:
                # For bullish bars: more weight to close and high
                price_weights = {
                    open_price: 0.15,   # 15%
                    high_price: 0.30,   # 30%
                    low_price: 0.10,    # 10%
                    close_price: 0.45   # 45%
                }
            else:
                # For bearish bars: more weight to close and low
                price_weights = {
                    open_price: 0.15,   # 15%
                    high_price: 0.10,   # 10%
                    low_price: 0.30,    # 30%
                    close_price: 0.45   # 45%
                }
            
            # Distribute volume according to weights
            for price, weight in price_weights.items():
                bin_key = round(price / bin_size) * bin_size
                volume_profile[bin_key] = volume_profile.get(bin_key, 0) + (total_volume * weight)

        # Find POC (price with highest volume)
        if volume_profile:
            poc_price = max(volume_profile, key=volume_profile.get)
            return round(float(poc_price), 2)
        else:
            return float(recent_data['Close'].iloc[-1])

    except Exception as e:
        logger.error(f"Enhanced POC calculation error: {e}")
        try:
            return float(data['Close'].iloc[-1])
        except:
            return 0.0

@safe_calculation_wrapper
def calculate_comprehensive_technicals(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive technical indicators for individual symbol analysis"""
    try:
        if len(data) < 50:
            return {}

        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']

        # Previous week high/low
        week_data = data.tail(5)  # Last 5 trading days
        prev_week_high = week_data['High'].max()
        prev_week_low = week_data['Low'].min()

        # RSI (14-period)
        rsi_14 = safe_rsi(close, 14).iloc[-1]

        # Money Flow Index (MFI)
        mfi_14 = calculate_mfi(data, 14)

        # MACD (12, 26, 9)
        macd_data = calculate_macd(close, 12, 26, 9)

        # Average True Range (ATR)
        atr_14 = calculate_atr(data, 14)

        # Bollinger Bands (20, 2)
        bb_data = calculate_bollinger_bands(close, 20, 2)

        # Stochastic Oscillator
        stoch_data = calculate_stochastic(data, 14, 3)

        # Williams %R
        williams_r = calculate_williams_r(data, 14)

        # Volume metrics
        volume_sma_20 = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        current_volume = volume.iloc[-1]
        volume_ratio = (current_volume / volume_sma_20) if volume_sma_20 > 0 else 1

        # Price metrics
        current_price = close.iloc[-1]
        price_change_1d = ((current_price - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) > 1 else 0
        price_change_5d = ((current_price - close.iloc[-6]) / close.iloc[-6] * 100) if len(close) > 5 else 0

        # Volatility (20-day)
        returns = close.pct_change().dropna()
        if len(returns) >= 20:
            volatility_20d = returns.rolling(20).std().iloc[-1] * (252 ** 0.5) * 100  # Annualized
        else:
            volatility_20d = returns.std() * (252 ** 0.5) * 100 if len(returns) > 0 else 20

        return {
            'prev_week_high': round(float(prev_week_high), 2),
            'prev_week_low': round(float(prev_week_low), 2),
            'rsi_14': round(float(rsi_14), 2),
            'mfi_14': round(float(mfi_14), 2),
            'macd': macd_data,
            'atr_14': round(float(atr_14), 2),
            'bollinger_bands': bb_data,
            'stochastic': stoch_data,
            'williams_r': williams_r,
            'volume_sma_20': round(float(volume_sma_20), 0),
            'current_volume': round(float(current_volume), 0),
            'volume_ratio': round(float(volume_ratio), 2),
            'price_change_1d': round(float(price_change_1d), 2),
            'price_change_5d': round(float(price_change_5d), 2),
            'volatility_20d': round(float(volatility_20d), 2)
        }

    except Exception as e:
        logger.error(f"Comprehensive technicals calculation error: {e}")
        return {}

@safe_calculation_wrapper
def calculate_mfi(data: pd.DataFrame, period: int = 14) -> float:
    """Calculate Money Flow Index"""
    try:
        if len(data) < period + 1:
            return 50.0
            
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']

        # Positive and negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()

        mfi = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, np.inf)))
        return float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 50.0
    except Exception as e:
        logger.error(f"MFI calculation error: {e}")
        return 50.0

@safe_calculation_wrapper
def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
    """Calculate MACD"""
    try:
        if len(close) < slow:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
            
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return {
            'macd': round(float(macd_line.iloc[-1]), 4),
            'signal': round(float(signal_line.iloc[-1]), 4),
            'histogram': round(float(histogram.iloc[-1]), 4)
        }
    except Exception as e:
        logger.error(f"MACD calculation error: {e}")
        return {'macd': 0, 'signal': 0, 'histogram': 0}

@safe_calculation_wrapper
def calculate_atr(data: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range"""
    try:
        if len(data) < period + 1:
            return 0.0
            
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift(1)).abs()
        low_close = (data['Low'] - data['Close'].shift(1)).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()

        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
    except Exception as e:
        logger.error(f"ATR calculation error: {e}")
        return 0.0

@safe_calculation_wrapper
def calculate_bollinger_bands(close: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, float]:
    """Calculate Bollinger Bands"""
    try:
        if len(close) < period:
            current_close = float(close.iloc[-1])
            return {
                'upper': current_close * 1.02,
                'middle': current_close,
                'lower': current_close * 0.98,
                'position': 50
            }
            
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        current_close = close.iloc[-1]
        
        upper_val = upper_band.iloc[-1]
        lower_val = lower_band.iloc[-1]
        
        if upper_val != lower_val:
            bb_position = ((current_close - lower_val) / (upper_val - lower_val)) * 100
        else:
            bb_position = 50

        return {
            'upper': round(float(upper_val), 2),
            'middle': round(float(sma.iloc[-1]), 2),
            'lower': round(float(lower_val), 2),
            'position': round(float(bb_position), 1)
        }
    except Exception as e:
        logger.error(f"Bollinger Bands calculation error: {e}")
        return {'upper': 0, 'middle': 0, 'lower': 0, 'position': 50}

@safe_calculation_wrapper
def calculate_stochastic(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
    """Calculate Stochastic Oscillator"""
    try:
        if len(data) < k_period:
            return {'k': 50, 'd': 50}
            
        lowest_low = data['Low'].rolling(k_period).min()
        highest_high = data['High'].rolling(k_period).max()

        k_percent = ((data['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
        d_percent = k_percent.rolling(d_period).mean()

        return {
            'k': round(float(k_percent.iloc[-1]), 2),
            'd': round(float(d_percent.iloc[-1]), 2)
        }
    except Exception as e:
        logger.error(f"Stochastic calculation error: {e}")
        return {'k': 50, 'd': 50}

@safe_calculation_wrapper
def calculate_williams_r(data: pd.DataFrame, period: int = 14) -> float:
    """Calculate Williams %R"""
    try:
        if len(data) < period:
            return -50.0
            
        highest_high = data['High'].rolling(period).max()
        lowest_low = data['Low'].rolling(period).min()

        williams_r = ((highest_high - data['Close']) / (highest_high - lowest_low)) * -100
        return float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else -50.0
    except Exception as e:
        logger.error(f"Williams %R calculation error: {e}")
        return -50.0

@safe_calculation_wrapper
def calculate_weekly_deviations(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate weekly 1, 2, 3 standard deviation levels"""
    try:
        if len(data) < 50:
            return {}

        # Resample to weekly data
        weekly_data = data.resample('W-FRI').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()

        if len(weekly_data) < 10:
            return {}

        # Calculate weekly statistics
        weekly_closes = weekly_data['Close']

        # Use last 20 weeks for calculation
        recent_weekly = weekly_closes.tail(20)
        mean_price = recent_weekly.mean()
        std_price = recent_weekly.std()

        if pd.isna(std_price) or std_price == 0:
            return {}

        deviations = {}
        for std_level in [1, 2, 3]:
            upper = mean_price + (std_level * std_price)
            lower = mean_price - (std_level * std_price)

            deviations[f'{std_level}_std'] = {
                'upper': round(float(upper), 2),
                'lower': round(float(lower), 2),
                'range_pct': round(float((std_level * std_price / mean_price) * 100), 2)
            }

        deviations['mean_price'] = round(float(mean_price), 2)
        deviations['std_price'] = round(float(std_price), 2)

        return deviations

    except Exception as e:
        logger.error(f"Weekly deviations calculation error: {e}")
        return {}

def calculate_composite_technical_score(analysis_results: Dict[str, Any]) -> tuple:
    """Calculate composite technical score from all indicators (1-100) - UPDATED with Volume/Volatility"""
    try:
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        volume_analysis = enhanced_indicators.get('volume_analysis', {})
        volatility_analysis = enhanced_indicators.get('volatility_analysis', {})
        current_price = analysis_results['current_price']
        
        scores = []
        weights = []
        
        # 1. PRICE POSITION ANALYSIS (30% total weight - reduced from 35%)
        daily_vwap = enhanced_indicators.get('daily_vwap', current_price)
        poc = enhanced_indicators.get('point_of_control', current_price)
        fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
        
        # VWAP position (10% weight)
        vwap_score = 75 if current_price > daily_vwap else 25
        scores.append(vwap_score)
        weights.append(0.10)
        
        # Point of Control position (10% weight) 
        poc_score = 75 if current_price > poc else 25
        scores.append(poc_score)
        weights.append(0.10)
        
        # EMA confluence analysis (10% weight - reduced from 15%)
        if fibonacci_emas:
            ema_above_count = sum(1 for ema_value in fibonacci_emas.values() if current_price > ema_value)
            ema_confluence_score = (ema_above_count / len(fibonacci_emas)) * 100
            scores.append(ema_confluence_score)
            weights.append(0.10)
        
        # 2. MOMENTUM OSCILLATORS (25% total weight - reduced from 30%)
        rsi = comprehensive_technicals.get('rsi_14', 50)
        mfi = comprehensive_technicals.get('mfi_14', 50)
        williams_r = comprehensive_technicals.get('williams_r', -50)
        stoch_data = comprehensive_technicals.get('stochastic', {})
        stoch_k = stoch_data.get('k', 50)
        
        # RSI scoring (oversold favored in bottom-picking system)
        if rsi < 25:
            rsi_score = 90  # Very oversold - very bullish
        elif rsi < 35:
            rsi_score = 75  # Oversold - bullish
        elif rsi > 75:
            rsi_score = 10  # Very overbought - very bearish
        elif rsi > 65:
            rsi_score = 25  # Overbought - bearish
        else:
            rsi_score = 50 + (50 - rsi) * 0.3  # Neutral zone with slight contrarian bias
        
        scores.append(rsi_score)
        weights.append(0.10)  # Reduced from 0.12
        
        # MFI scoring (money flow consideration)
        if mfi < 20:
            mfi_score = 85
        elif mfi > 80:
            mfi_score = 15
        else:
            mfi_score = 50 + (50 - mfi) * 0.4
        
        scores.append(mfi_score)
        weights.append(0.06)  # Reduced from 0.08
        
        # Williams %R scoring (convert to 0-100 scale)
        williams_normalized = ((williams_r + 100) / 100) * 100  # Convert -100:0 to 0:100
        scores.append(williams_normalized)
        weights.append(0.05)
        
        # Stochastic scoring
        if stoch_k < 20:
            stoch_score = 85
        elif stoch_k > 80:
            stoch_score = 15
        else:
            stoch_score = stoch_k
        
        scores.append(stoch_score)
        weights.append(0.04)  # Reduced from 0.05
        
        # 3. VOLUME ANALYSIS (15% weight - NEW)
        volume_composite_score = volume_analysis.get('composite_score', 50) if volume_analysis else 50
        scores.append(volume_composite_score)
        weights.append(0.15)
        
        # 4. VOLATILITY ANALYSIS (15% weight - NEW)
        volatility_composite_score = volatility_analysis.get('composite_score', 50) if volatility_analysis else 50
        scores.append(volatility_composite_score)
        weights.append(0.15)
        
        # 5. TREND ANALYSIS (15% weight - reduced from 20%)
        macd_data = comprehensive_technicals.get('macd', {})
        histogram = macd_data.get('histogram', 0)
        
        # MACD Histogram trend
        if histogram > 0:
            macd_score = 70 + min(histogram * 1000, 20)  # Bullish with strength adjustment
        elif histogram < 0:
            macd_score = 30 + max(histogram * 1000, -20)  # Bearish with strength adjustment
        else:
            macd_score = 50
        
        scores.append(max(5, min(95, macd_score)))
        weights.append(0.08)  # Reduced from 0.10
        
        # Previous week support/resistance analysis
        prev_week_high = comprehensive_technicals.get('prev_week_high', current_price)
        prev_week_low = comprehensive_technicals.get('prev_week_low', current_price)
        
        if current_price > prev_week_high:
            breakout_score = 85  # Above resistance - bullish breakout
        elif current_price < prev_week_low:
            breakout_score = 15  # Below support - bearish breakdown
        else:
            # Within range - score based on position
            range_size = prev_week_high - prev_week_low
            if range_size > 0:
                position_in_range = (current_price - prev_week_low) / range_size
                breakout_score = 20 + (position_in_range * 60)  # 20-80 range
            else:
                breakout_score = 50
        
        scores.append(breakout_score)
        weights.append(0.07)  # Reduced from 0.10
        
        # Calculate weighted composite score
        if len(scores) == len(weights) and sum(weights) > 0:
            composite_score = sum(score * weight for score, weight in zip(scores, weights)) / sum(weights)
        else:
            composite_score = 50  # Default neutral
        
        # Ensure score is within bounds and add some smoothing
        final_score = max(1, min(100, round(composite_score, 1)))
        
        return final_score, {
            'component_scores': {
                'vwap_position': round(scores[0], 1) if len(scores) > 0 else 50,
                'poc_position': round(scores[1], 1) if len(scores) > 1 else 50, 
                'ema_confluence': round(scores[2], 1) if len(scores) > 2 else 50,
                'rsi_momentum': round(scores[3], 1) if len(scores) > 3 else 50,
                'volume_composite': round(volume_composite_score, 1),
                'volatility_composite': round(volatility_composite_score, 1),
                'trend_direction': round(scores[6], 1) if len(scores) > 6 else 50
            },
            'total_components': len(scores),
            'weight_distribution': {
                'price_position': 0.30,
                'momentum_oscillators': 0.25,
                'volume_analysis': 0.15,
                'volatility_analysis': 0.15,
                'trend_analysis': 0.15
            },
            'new_components_integrated': True
        }
        
    except Exception as e:
        logger.error(f"Composite technical score calculation error: {e}")
        return 50.0, {'error': str(e)}

@safe_calculation_wrapper
def calculate_enhanced_technical_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate enhanced technical analysis with volume and volatility integration"""
    try:
        if len(data) < 50:
            return {'error': 'Insufficient data for enhanced technical analysis'}
        
        # Calculate all traditional technical indicators
        daily_vwap = calculate_daily_vwap(data)
        fibonacci_emas = calculate_fibonacci_emas(data)
        point_of_control = calculate_point_of_control_enhanced(data)
        weekly_deviations = calculate_weekly_deviations(data)
        comprehensive_technicals = calculate_comprehensive_technicals(data)
        
        # Calculate new volume and volatility analyses
        volume_analysis = calculate_complete_volume_analysis(data)
        volatility_analysis = calculate_complete_volatility_analysis(data)
        
        # Combine all indicators
        enhanced_indicators = {
            'daily_vwap': daily_vwap,
            'fibonacci_emas': fibonacci_emas,
            'point_of_control': point_of_control,
            'weekly_deviations': weekly_deviations,
            'comprehensive_technicals': comprehensive_technicals,
            'volume_analysis': volume_analysis,
            'volatility_analysis': volatility_analysis
        }
        
        return enhanced_indicators
        
    except Exception as e:
        logger.error(f"Enhanced technical analysis error: {e}")
        return {'error': f'Enhanced analysis error: {str(e)}'}
