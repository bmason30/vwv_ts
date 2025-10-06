"""
FILENAME: options.py
VWV Professional Trading System v4.2.2
File Revision: r9
Date: October 6, 2025
Revision Type: Critical Function Signature Fix

CRITICAL FIXES/CHANGES IN THIS REVISION:
- Fixed calculate_confidence_intervals function signature (was causing positional argument error)
- Changed current_price from positional to keyword-only parameter
- Enhanced error handling with proper dict returns (never None)
- Validated all function signatures for proper parameter handling
- Added comprehensive logging for debugging

FILE REVISION HISTORY:
r9 (Oct 6, 2025) - Fixed function signature causing positional arg error
r8 (Oct 6, 2025) - Enhanced None return handling
r7 (Oct 5, 2025) - Options calculation improvements
r6 (Oct 4, 2025) - Probability calculations enhanced
r5 (Oct 3, 2025) - Core options analysis updates
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy import stats
from utils.decorators import safe_calculation_wrapper
import logging

logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def calculate_options_levels_enhanced(
    current_price: float,
    volatility: float,
    dte_list: List[int] = [7, 14, 21, 30, 45],
    underlying_beta: float = 1.0
) -> List[Dict[str, Any]]:
    """Calculate enhanced options levels with probability of touch"""
    try:
        if current_price <= 0 or volatility <= 0:
            return []
        
        # Adjust volatility for market beta
        adjusted_volatility = volatility * abs(underlying_beta)
        
        options_levels = []
        
        for dte in dte_list:
            # Time factor
            time_factor = np.sqrt(dte / 365)
            
            # Standard deviation move
            std_move = current_price * (adjusted_volatility / 100) * time_factor
            
            # Delta approximation (~0.16 delta for conservative strikes)
            sigma_multiplier = 1.0
            
            put_strike = round(current_price - (std_move * sigma_multiplier), 2)
            call_strike = round(current_price + (std_move * sigma_multiplier), 2)
            
            # Probability of Touch approximation
            pot = round(stats.norm.sf(sigma_multiplier) * 2 * 100, 1)
            
            # Greeks approximations
            delta_put = -0.16
            delta_call = 0.16
            theta_daily = round(std_move * 0.02, 2)
            
            options_levels.append({
                'DTE': dte,
                'Put Strike': f"${put_strike:.2f}",
                'Put Delta': f"{delta_put:.2f}",
                'Put PoT': f"{pot:.1f}%",
                'Call Strike': f"${call_strike:.2f}",
                'Call Delta': f"{delta_call:.2f}",
                'Call PoT': f"{pot:.1f}%",
                'Expected Move': f"±{std_move:.2f}",
                'Theta/Day': f"${theta_daily:.2f}",
                'Beta Adj': f"{underlying_beta:.2f}"
            })
        
        return options_levels
        
    except Exception as e:
        logger.error(f"Options levels calculation error: {e}")
        return []

@safe_calculation_wrapper
def calculate_confidence_intervals(
    data: pd.DataFrame,
    *,
    current_price: float = None,
    confidence_levels: List[float] = None
) -> Dict[str, Any]:
    """
    Calculate statistical confidence intervals
    CRITICAL FIX r9: current_price is now keyword-only parameter
    This prevents positional argument errors
    """
    try:
        # Default confidence levels
        if confidence_levels is None:
            confidence_levels = [0.68, 0.95, 0.997]
        
        # Validate inputs
        if data is None or len(data) < 10:
            logger.warning("Insufficient data for confidence intervals")
            return {
                'error': 'Insufficient data (need at least 10 data points)',
                'confidence_intervals': {},
                'mean_weekly_return': 0.0,
                'weekly_volatility': 0.0,
                'sample_size': len(data) if data is not None else 0
            }
        
        # Get current price
        if current_price is None:
            if 'Close' not in data.columns:
                return {
                    'error': 'No Close price available',
                    'confidence_intervals': {},
                    'mean_weekly_return': 0.0,
                    'weekly_volatility': 0.0,
                    'sample_size': 0
                }
            current_price = float(data['Close'].iloc[-1])
        
        # Resample to weekly data
        try:
            weekly_data = data.resample('W-FRI').agg({'Close': 'last'}).dropna()
        except Exception as e:
            logger.error(f"Resampling error: {e}")
            return {
                'error': f'Resampling failed: {str(e)}',
                'confidence_intervals': {},
                'mean_weekly_return': 0.0,
                'weekly_volatility': 0.0,
                'sample_size': 0
            }
        
        if len(weekly_data) < 10:
            return {
                'error': 'Insufficient weekly data (need at least 10 weeks)',
                'confidence_intervals': {},
                'mean_weekly_return': 0.0,
                'weekly_volatility': 0.0,
                'sample_size': len(weekly_data)
            }
        
        # Calculate weekly returns
        weekly_returns = weekly_data['Close'].pct_change().dropna()
        
        if len(weekly_returns) == 0:
            return {
                'error': 'No valid weekly returns',
                'confidence_intervals': {},
                'mean_weekly_return': 0.0,
                'weekly_volatility': 0.0,
                'sample_size': 0
            }
        
        # Calculate statistics
        mean_return = float(weekly_returns.mean() * 100)
        std_return = float(weekly_returns.std() * 100)
        
        # Build confidence intervals
        intervals = {}
        
        for confidence_level in confidence_levels:
            # Z-score for confidence level
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            
            # Calculate bounds
            lower_return_pct = mean_return - (z_score * std_return)
            upper_return_pct = mean_return + (z_score * std_return)
            
            # Convert to price levels
            lower_price = current_price * (1 + lower_return_pct / 100)
            upper_price = current_price * (1 + upper_return_pct / 100)
            
            # Expected move
            expected_move_pct = z_score * std_return
            
            # Format confidence level
            conf_label = f"{int(confidence_level * 100)}%"
            
            intervals[conf_label] = {
                'lower_bound': round(float(lower_price), 2),
                'upper_bound': round(float(upper_price), 2),
                'expected_move_pct': round(float(expected_move_pct), 2),
                'confidence_level': confidence_level
            }
        
        # Return complete result (ALWAYS a dict)
        return {
            'confidence_intervals': intervals,
            'mean_weekly_return': round(mean_return, 3),
            'weekly_volatility': round(std_return, 2),
            'sample_size': len(weekly_returns),
            'current_price': round(current_price, 2)
        }
        
    except Exception as e:
        logger.error(f"Confidence intervals calculation error: {e}")
        # CRITICAL: Even on error, return a dict (not None)
        return {
            'error': f'Calculation failed: {str(e)}',
            'confidence_intervals': {},
            'mean_weekly_return': 0.0,
            'weekly_volatility': 0.0,
            'sample_size': 0
        }

@safe_calculation_wrapper
def calculate_expected_move(
    current_price: float,
    volatility: float,
    days_to_expiration: int
) -> Dict[str, float]:
    """Calculate expected move for a given timeframe"""
    try:
        if current_price <= 0 or volatility <= 0 or days_to_expiration <= 0:
            return {
                'expected_move_dollars': 0.0,
                'expected_move_pct': 0.0,
                'upper_bound': current_price,
                'lower_bound': current_price
            }
        
        # Time factor
        time_factor = np.sqrt(days_to_expiration / 365)
        
        # Expected move (1 standard deviation)
        expected_move = current_price * (volatility / 100) * time_factor
        expected_move_pct = (expected_move / current_price) * 100
        
        return {
            'expected_move_dollars': round(expected_move, 2),
            'expected_move_pct': round(expected_move_pct, 2),
            'upper_bound': round(current_price + expected_move, 2),
            'lower_bound': round(current_price - expected_move, 2),
            'days': days_to_expiration,
            'volatility_used': volatility
        }
        
    except Exception as e:
        logger.error(f"Expected move calculation error: {e}")
        return {
            'expected_move_dollars': 0.0,
            'expected_move_pct': 0.0,
            'upper_bound': current_price,
            'lower_bound': current_price,
            'error': str(e)
        }

@safe_calculation_wrapper
def calculate_probability_of_profit(
    entry_price: float,
    strike_price: float,
    premium_received: float,
    volatility: float,
    days_to_expiration: int,
    option_type: str = 'put'
) -> float:
    """Calculate approximate probability of profit for option selling"""
    try:
        if entry_price <= 0 or strike_price <= 0 or days_to_expiration <= 0:
            return 0.5
        
        # Breakeven point
        if option_type.lower() == 'put':
            breakeven = strike_price - premium_received
            distance = entry_price - breakeven
        else:  # call
            breakeven = strike_price + premium_received
            distance = breakeven - entry_price
        
        # Standard deviation move
        time_factor = np.sqrt(days_to_expiration / 365)
        std_move = entry_price * (volatility / 100) * time_factor
        
        if std_move == 0:
            return 0.5
        
        # Z-score
        z_score = distance / std_move
        
        # Probability
        probability = stats.norm.cdf(z_score)
        
        return round(float(probability) * 100, 1)
        
    except Exception as e:
        logger.error(f"Probability calculation error: {e}")
        return 50.0
