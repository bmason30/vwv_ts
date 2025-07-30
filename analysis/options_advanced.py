"""
Advanced Options Analysis with Sigma Levels & Fibonacci Integration
Multi-factor weighting system with risk level classifications
"""
import pandas as pd
import numpy as np
import math
import logging
from typing import Dict, Any, List, Optional, Tuple
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

# Advanced Options Configuration
ADVANCED_OPTIONS_CONFIG = {
    'base_calculation_days': 5,  # 5-day rolling average for base price
    'fibonacci_levels': [0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272, 1.414, 1.618],
    'volatility_periods': [10, 20, 30],  # Multiple volatility timeframes
    'volume_analysis_period': 20,
    'risk_levels': {
        'conservative': {
            'fibonacci_weight': 0.50,
            'volatility_weight': 0.30, 
            'volume_weight': 0.20,
            'target_pot': 0.15,  # 15% Probability of Touch
            'sigma_multiplier': 0.8,
            'description': 'Low risk, high win rate strategy'
        },
        'moderate': {
            'fibonacci_weight': 0.35,
            'volatility_weight': 0.45,
            'volume_weight': 0.20,
            'target_pot': 0.25,  # 25% Probability of Touch
            'sigma_multiplier': 1.0,
            'description': 'Balanced risk/reward approach'
        },
        'aggressive': {
            'fibonacci_weight': 0.25,
            'volatility_weight': 0.60,
            'volume_weight': 0.15,
            'target_pot': 0.35,  # 35% Probability of Touch
            'sigma_multiplier': 1.3,
            'description': 'Higher risk, higher premium strategy'
        }
    },
    'dte_levels': [7, 14, 21, 30, 45],  # Days to expiration
    'min_data_points': 50
}

@safe_calculation_wrapper
def calculate_5day_rolling_base(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate 5-day rolling average closing price as base for options calculations
    """
    try:
        if len(data) < 5:
            return {'base_price': float(data['Close'].iloc[-1]), 'rolling_periods': 1}
        
        # Calculate 5-day rolling average
        rolling_5day = data['Close'].rolling(window=5).mean()
        base_price = rolling_5day.iloc[-1]
        
        # Additional metrics
        current_price = data['Close'].iloc[-1]
        base_deviation = ((current_price - base_price) / base_price) * 100
        
        return {
            'base_price': round(float(base_price), 2),
            'current_price': round(float(current_price), 2),
            'base_deviation_pct': round(float(base_deviation), 2),
            'rolling_periods': 5
        }
        
    except Exception as e:
        logger.error(f"5-day rolling base calculation error: {e}")
        return {'base_price': float(data['Close'].iloc[-1]), 'rolling_periods': 1}

@safe_calculation_wrapper
def calculate_fibonacci_levels_from_base(base_data: Dict[str, float], volatility_data: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate Fibonacci-based option levels from 5-day average base
    """
    try:
        base_price = base_data['base_price']
        fibonacci_levels = ADVANCED_OPTIONS_CONFIG['fibonacci_levels']
        avg_volatility = volatility_data.get('average_volatility', 20) / 100  # Convert to decimal
        
        fibonacci_strikes = {}
        
        for level in fibonacci_levels:
            # Calculate Fibonacci-based range
            fib_range = base_price * avg_volatility * level
            
            fibonacci_strikes[f'fib_{level}'] = {
                'level': level,
                'put_strike': round(base_price - fib_range, 2),
                'call_strike': round(base_price + fib_range, 2),
                'range_dollars': round(fib_range, 2),
                'range_percent': round((fib_range / base_price) * 100, 2)
            }
        
        return {
            'base_price': base_price,
            'fibonacci_strikes': fibonacci_strikes,
            'volatility_used': avg_volatility * 100
        }
        
    except Exception as e:
        logger.error(f"Fibonacci levels calculation error: {e}")
        return {'fibonacci_strikes': {}, 'base_price': base_data.get('base_price', 0)}

@safe_calculation_wrapper
def calculate_multi_timeframe_volatility(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate volatility across multiple timeframes for sigma calculations
    """
    try:
        if len(data) < 30:
            # Fallback for insufficient data
            returns = data['Close'].pct_change().dropna()
            if len(returns) > 0:
                vol = returns.std() * (252 ** 0.5) * 100
                return {'average_volatility': float(vol), '10d_vol': float(vol), '20d_vol': float(vol), '30d_vol': float(vol)}
            else:
                return {'average_volatility': 20.0, '10d_vol': 20.0, '20d_vol': 20.0, '30d_vol': 20.0}
        
        returns = data['Close'].pct_change().dropna()
        volatilities = {}
        
        for period in ADVANCED_OPTIONS_CONFIG['volatility_periods']:
            if len(returns) >= period:
                vol = returns.rolling(window=period).std().iloc[-1] * (252 ** 0.5) * 100
                volatilities[f'{period}d_vol'] = round(float(vol), 2)
            else:
                # Use whatever data we have
                vol = returns.std() * (252 ** 0.5) * 100
                volatilities[f'{period}d_vol'] = round(float(vol), 2)
        
        # Calculate weighted average volatility
        if volatilities:
            avg_vol = sum(volatilities.values()) / len(volatilities)
        else:
            avg_vol = 20.0  # Default volatility
        
        volatilities['average_volatility'] = round(avg_vol, 2)
        
        return volatilities
        
    except Exception as e:
        logger.error(f"Multi-timeframe volatility calculation error: {e}")
        return {'average_volatility': 20.0, '10d_vol': 20.0, '20d_vol': 20.0, '30d_vol': 20.0}

@safe_calculation_wrapper
def calculate_volume_strength_factor(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate volume strength factor for options sigma calculations
    """
    try:
        volume_period = ADVANCED_OPTIONS_CONFIG['volume_analysis_period']
        
        if len(data) < volume_period:
            return {'volume_factor': 1.0, 'volume_strength': 'Normal', 'current_vs_avg': 1.0}
        
        volume = data['Volume']
        current_volume = volume.iloc[-1]
        avg_volume = volume.rolling(window=volume_period).mean().iloc[-1]
        
        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
        else:
            volume_ratio = 1.0
        
        # Convert volume ratio to factor for sigma calculations
        if volume_ratio > 2.0:
            volume_factor = 1.3  # High volume increases sigma
            volume_strength = 'Very High'
        elif volume_ratio > 1.5:
            volume_factor = 1.15
            volume_strength = 'High'
        elif volume_ratio < 0.5:
            volume_factor = 0.85  # Low volume decreases sigma
            volume_strength = 'Low'
        elif volume_ratio < 0.7:
            volume_factor = 0.95
            volume_strength = 'Below Average'
        else:
            volume_factor = 1.0
            volume_strength = 'Normal'
        
        return {
            'volume_factor': round(volume_factor, 2),
            'volume_strength': volume_strength,
            'current_vs_avg': round(volume_ratio, 2),
            'current_volume': round(float(current_volume), 0),
            'average_volume': round(float(avg_volume), 0)
        }
        
    except Exception as e:
        logger.error(f"Volume strength factor calculation error: {e}")
        return {'volume_factor': 1.0, 'volume_strength': 'Normal', 'current_vs_avg': 1.0}

@safe_calculation_wrapper
def calculate_sigma_levels_with_weighting(
    base_data: Dict[str, float],
    fibonacci_data: Dict[str, Any], 
    volatility_data: Dict[str, float],
    volume_data: Dict[str, float],
    risk_level: str = 'moderate'
) -> Dict[str, Any]:
    """
    Calculate sigma levels using multi-factor weighting system
    """
    try:
        if risk_level not in ADVANCED_OPTIONS_CONFIG['risk_levels']:
            risk_level = 'moderate'
        
        risk_config = ADVANCED_OPTIONS_CONFIG['risk_levels'][risk_level]
        base_price = base_data['base_price']
        
        # Get weighting factors
        fib_weight = risk_config['fibonacci_weight']
        vol_weight = risk_config['volatility_weight'] 
        volume_weight = risk_config['volume_weight']
        sigma_multiplier = risk_config['sigma_multiplier']
        
        # Calculate base sigma from volatility
        avg_volatility = volatility_data['average_volatility'] / 100
        volume_factor = volume_data['volume_factor']
        
        # Multi-factor sigma calculation
        base_sigma = base_price * avg_volatility * sigma_multiplier
        
        # Apply volume adjustment
        volume_adjusted_sigma = base_sigma * volume_factor
        
        # Get fibonacci reference levels
        fibonacci_strikes = fibonacci_data.get('fibonacci_strikes', {})
        
        # Select appropriate fibonacci levels for each risk level
        if risk_level == 'conservative':
            fib_level_put = 0.382  # Closer to money for conservative
            fib_level_call = 0.382
        elif risk_level == 'moderate':
            fib_level_put = 0.500  # Mid-level
            fib_level_call = 0.500
        else:  # aggressive
            fib_level_put = 0.618  # Further from money for aggressive
            fib_level_call = 0.618
        
        # Calculate weighted strikes for different DTEs
        sigma_levels = {}
        
        for dte in ADVANCED_OPTIONS_CONFIG['dte_levels']:
            time_factor = math.sqrt(dte / 365.0)  # Time scaling
            
            # Fibonacci component
            fib_key = f'fib_{fib_level_put}'
            if fib_key in fibonacci_strikes:
                fib_put_strike = fibonacci_strikes[fib_key]['put_strike']
                fib_call_strike = fibonacci_strikes[fib_key]['call_strike']
            else:
                # Fallback calculation
                fib_range = base_price * avg_volatility * fib_level_put
                fib_put_strike = base_price - fib_range
                fib_call_strike = base_price + fib_range
            
            # Volatility component  
            vol_range = volume_adjusted_sigma * time_factor
            vol_put_strike = base_price - vol_range
            vol_call_strike = base_price + vol_range
            
            # Volume component (minimal direct impact on strikes, more on sigma)
            volume_adjustment = 1.0 + (volume_data['volume_factor'] - 1.0) * 0.1
            
            # Weighted combination
            weighted_put_strike = (
                fib_put_strike * fib_weight +
                vol_put_strike * vol_weight +
                base_price * volume_weight * 0.95  # Slight discount for volume
            ) * volume_adjustment
            
            weighted_call_strike = (
                fib_call_strike * fib_weight +
                vol_call_strike * vol_weight +
                base_price * volume_weight * 1.05  # Slight premium for volume
            ) * volume_adjustment
            
            # Calculate Probability of Touch (simplified)
            target_pot = risk_config['target_pot']
            strike_distance_put = abs(base_price - weighted_put_strike) / base_price
            strike_distance_call = abs(weighted_call_strike - base_price) / base_price
            
            # Simplified PoT calculation (to be validated via backtesting)
            pot_put = min(target_pot * 1.2, strike_distance_put * 2 * 100)
            pot_call = min(target_pot * 1.2, strike_distance_call * 2 * 100)
            
            sigma_levels[f'dte_{dte}'] = {
                'dte': dte,
                'put_strike': round(weighted_put_strike, 2),
                'call_strike': round(weighted_call_strike, 2),
                'put_pot': round(pot_put * 100, 1),  # Convert to percentage
                'call_pot': round(pot_call * 100, 1),
                'time_factor': round(time_factor, 3),
                'fibonacci_weight': fib_weight,
                'volatility_weight': vol_weight,
                'volume_weight': volume_weight,
                'expected_move': round(abs(weighted_call_strike - weighted_put_strike), 2)
            }
        
        return {
            'risk_level': risk_level,
            'risk_description': risk_config['description'],
            'base_price': base_price,
            'sigma_levels': sigma_levels,
            'calculation_components': {
                'fibonacci_weight': fib_weight,
                'volatility_weight': vol_weight,
                'volume_weight': volume_weight,
                'sigma_multiplier': sigma_multiplier,
                'volume_factor': volume_data['volume_factor']
            }
        }
        
    except Exception as e:
        logger.error(f"Sigma levels calculation error: {e}")
        return {'sigma_levels': {}, 'risk_level': risk_level, 'base_price': base_data.get('base_price', 0)}

@safe_calculation_wrapper
def calculate_complete_advanced_options(data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """
    Complete advanced options analysis with sigma levels and fibonacci integration
    """
    try:
        if len(data) < ADVANCED_OPTIONS_CONFIG['min_data_points']:
            return {
                'symbol': symbol,
                'error': f'Insufficient data: need at least {ADVANCED_OPTIONS_CONFIG["min_data_points"]} points',
                'status': 'INSUFFICIENT_DATA'
            }
        
        # Step 1: Calculate 5-day rolling base
        base_data = calculate_5day_rolling_base(data)
        
        # Step 2: Calculate multi-timeframe volatility
        volatility_data = calculate_multi_timeframe_volatility(data)
        
        # Step 3: Calculate volume strength factor
        volume_data = calculate_volume_strength_factor(data)
        
        # Step 4: Calculate fibonacci levels from base
        fibonacci_data = calculate_fibonacci_levels_from_base(base_data, volatility_data)
        
        # Step 5: Calculate sigma levels for all risk levels
        risk_level_results = {}
        
        for risk_level in ['conservative', 'moderate', 'aggressive']:
            sigma_data = calculate_sigma_levels_with_weighting(
                base_data, fibonacci_data, volatility_data, volume_data, risk_level
            )
            risk_level_results[risk_level] = sigma_data
        
        # Step 6: Build comprehensive results
        current_date = data.index[-1].strftime('%Y-%m-%d')
        
        return {
            'symbol': symbol,
            'timestamp': current_date,
            'base_analysis': base_data,
            'volatility_analysis': volatility_data,
            'volume_analysis': volume_data,
            'fibonacci_analysis': fibonacci_data,
            'risk_level_analysis': risk_level_results,
            'configuration': ADVANCED_OPTIONS_CONFIG,
            'status': 'OPERATIONAL',
            'backtesting_notes': [
                'Probability of Touch calculations need validation',
                'Sigma multipliers may need adjustment based on historical performance',
                'Volume factor impact should be tested across different market conditions',
                'Fibonacci level selection optimization needed',
                'Time decay factors may need refinement'
            ]
        }
        
    except Exception as e:
        logger.error(f"Complete advanced options analysis error: {e}")
        return {
            'symbol': symbol,
            'error': str(e),
            'status': 'ERROR'
        }

@safe_calculation_wrapper
def format_advanced_options_for_display(advanced_options_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format advanced options data for UI display
    """
    try:
        if 'error' in advanced_options_data:
            return advanced_options_data
        
        risk_level_analysis = advanced_options_data.get('risk_level_analysis', {})
        base_analysis = advanced_options_data.get('base_analysis', {})
        
        # Create summary table data
        display_data = []
        
        for risk_level, risk_data in risk_level_analysis.items():
            sigma_levels = risk_data.get('sigma_levels', {})
            
            for dte_key, level_data in sigma_levels.items():
                display_data.append({
                    'Risk Level': risk_level.capitalize(),
                    'DTE': level_data['dte'],
                    'Put Strike': f"${level_data['put_strike']:.2f}",
                    'Put PoT': f"{level_data['put_pot']:.1f}%",
                    'Call Strike': f"${level_data['call_strike']:.2f}",
                    'Call PoT': f"{level_data['call_pot']:.1f}%",
                    'Expected Move': f"±${level_data['expected_move']:.2f}",
                    'Description': risk_data.get('risk_description', '')
                })
        
        return {
            'display_table': display_data,
            'base_price': base_analysis.get('base_price', 0),
            'current_price': base_analysis.get('current_price', 0),
            'base_deviation': base_analysis.get('base_deviation_pct', 0),
            'volatility_summary': advanced_options_data.get('volatility_analysis', {}),
            'volume_summary': advanced_options_data.get('volume_analysis', {}),
            'fibonacci_summary': advanced_options_data.get('fibonacci_analysis', {}),
            'status': advanced_options_data.get('status', 'UNKNOWN')
        }
        
    except Exception as e:
        logger.error(f"Advanced options display formatting error: {e}")
        return {'error': str(e), 'status': 'FORMATTING_ERROR'}
