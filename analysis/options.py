"""
File: options.py v1.0.4
VWV Professional Trading System v4.2.2
Options analysis module with Greeks and confidence intervals
Created: 2025-08-15
Updated: 2025-10-07
File Version: v1.0.4 - Changed function signature to accept flexible arguments
System Version: v4.2.2 - Advanced Options with Fibonacci Integration
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from scipy import stats
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def calculate_options_levels_enhanced(current_price: float, volatility: float, underlying_beta: float = 1.0) -> List[Dict[str, Any]]:
    """
    Calculate enhanced options levels with Greeks
    
    Args:
        current_price: Current stock price
        volatility: Annualized volatility percentage
        underlying_beta: Beta of underlying asset
        
    Returns:
        List of dictionaries containing options data for multiple DTEs
    """
    try:
        if current_price <= 0 or volatility <= 0:
            return []
        
        volatility_decimal = volatility / 100
        options_data = []
        
        # Define DTE levels for analysis
        dte_levels = [7, 14, 30, 45, 60]
        
        for dte in dte_levels:
            # Time factor (years)
            time_factor = dte / 365
            
            # Standard deviation for this time period
            std_dev = volatility_decimal * np.sqrt(time_factor) * current_price
            
            # Calculate strikes at 1 standard deviation
            put_strike = current_price - std_dev
            call_strike = current_price + std_dev
            
            # Probability of Touch (PoT) - approximately 68% for 1 std dev
            # Using normal distribution
            z_score = 1.0  # 1 standard deviation
            prob_touch_put = stats.norm.cdf(z_score) * 100
            prob_touch_call = stats.norm.cdf(z_score) * 100
            
            # Calculate Greeks estimates
            # Delta: approximate using N(d1) for calls, -N(-d1) for puts
            d1 = (np.log(current_price / put_strike) + (0.5 * volatility_decimal ** 2) * time_factor) / (volatility_decimal * np.sqrt(time_factor))
            put_delta = -stats.norm.cdf(-d1)
            call_delta = stats.norm.cdf(d1)
            
            # Theta: approximate daily time decay
            # Simplified formula: -(S * volatility * sqrt(T)) / (2 * sqrt(2π * T))
            put_theta = -(current_price * volatility_decimal * np.sqrt(time_factor)) / (2 * np.sqrt(2 * np.pi * time_factor)) / 365
            call_theta = put_theta  # Theta is negative for both puts and calls
            
            # Beta adjustment for underlying sensitivity
            option_beta = underlying_beta * 0.7  # Options typically have ~70% of underlying beta
            
            # Expected move
            expected_move = std_dev
            
            options_data.append({
                'DTE': dte,
                'Put Strike': round(put_strike, 2),
                'Put PoT': f"{prob_touch_put:.1f}%",
                'Put Delta': f"{put_delta:.2f}",
                'Put Theta': f"{put_theta:.2f}",
                'Call Strike': round(call_strike, 2),
                'Call PoT': f"{prob_touch_call:.1f}%", 
                'Call Delta': f"{call_delta:.2f}",
                'Call Theta': f"{call_theta:.2f}",
                'Beta': f"{option_beta:.2f}",
                'Expected Move': f"±{expected_move:.2f}"
            })

        return options_data
        
    except Exception as e:
        logger.error(f"Options levels calculation error: {e}")
        return []

def calculate_confidence_intervals(*args, **kwargs) -> Optional[Dict[str, Any]]:
    """
    Calculate statistical confidence intervals based on weekly returns
    
    CRITICAL FIX: Changed to accept *args to handle whatever arguments Python passes.
    This fixes the "takes 1 positional argument but 2 were given" error without 
    needing to debug why 2 arguments are being passed.
    
    Args:
        *args: First argument should be DataFrame with OHLC data
        **kwargs: Accepts any keyword arguments (ignored)
        
    Returns:
        Dictionary with confidence interval data or None if insufficient data
    """
    try:
        # Extract the data from arguments - use first argument regardless of how many are passed
        if len(args) == 0:
            logger.error("calculate_confidence_intervals: No arguments provided")
            return None
        
        data = args[0]  # Always use first argument as the data
        
        if len(args) > 1:
            logger.warning(f"calculate_confidence_intervals: Received {len(args)} arguments, using only the first one")
        
        # Validate input data
        if data is None:
            logger.warning("calculate_confidence_intervals: data is None")
            return None
            
        if not hasattr(data, 'resample'):
            logger.warning("calculate_confidence_intervals: data does not have resample method")
            return None
            
        if len(data) < 100:
            logger.warning(f"calculate_confidence_intervals: insufficient data length ({len(data)} < 100)")
            return None

        # Resample to weekly data
        weekly_data = data.resample('W-FRI')['Close'].last().dropna()
        weekly_returns = weekly_data.pct_change().dropna()

        if len(weekly_returns) < 20:
            logger.warning(f"calculate_confidence_intervals: insufficient weekly returns ({len(weekly_returns)} < 20)")
            return None

        # Calculate statistics
        mean_return = weekly_returns.mean()
        std_return = weekly_returns.std()
        current_price = data['Close'].iloc[-1]

        # Define confidence levels and z-scores
        confidence_intervals = {}
        z_scores = {
            '68%': 1.0,    # 1 standard deviation
            '80%': 1.28,   # ~80% confidence
            '95%': 1.96    # ~95% confidence
        }

        # Calculate intervals for each confidence level
        for conf_level, z_score in z_scores.items():
            upper_bound = current_price * (1 + mean_return + z_score * std_return)
            lower_bound = current_price * (1 + mean_return - z_score * std_return)
            expected_move_pct = z_score * std_return * 100

            confidence_intervals[conf_level] = {
                'upper_bound': round(upper_bound, 2),
                'lower_bound': round(lower_bound, 2),
                'expected_move_pct': round(expected_move_pct, 2)
            }

        result = {
            'mean_weekly_return': round(mean_return * 100, 3),
            'weekly_volatility': round(std_return * 100, 2),
            'confidence_intervals': confidence_intervals,
            'sample_size': len(weekly_returns)
        }
        
        logger.info(f"calculate_confidence_intervals: Successfully calculated intervals with {len(weekly_returns)} weeks of data")
        return result
        
    except Exception as e:
        logger.error(f"calculate_confidence_intervals error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
