"""
Options analysis module for VWV Trading System v4.2.2
Complete options analysis with strike levels and Greeks
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def calculate_options_levels_enhanced(current_price: float, volatility: float, underlying_beta: float = 1.0) -> List[Dict[str, Any]]:
    """
    Calculate enhanced options levels with probability of touch
    
    Args:
        current_price: Current stock price
        volatility: Annualized volatility percentage
        underlying_beta: Beta relative to market (default 1.0)
    
    Returns:
        List of option levels with strikes and Greeks
    """
    try:
        # Ensure inputs are floats
        current_price = float(current_price)
        volatility = float(volatility)
        underlying_beta = float(underlying_beta)
        
        if current_price <= 0 or volatility <= 0:
            logger.warning(f"Invalid inputs: price={current_price}, vol={volatility}")
            return []
        
        # Define option levels with sigma distances
        option_levels = [
            {'name': '7 DTE', 'dte': 7, 'sigma': 0.5},
            {'name': '14 DTE', 'dte': 14, 'sigma': 0.7},
            {'name': '21 DTE', 'dte': 21, 'sigma': 0.85},
            {'name': '30 DTE', 'dte': 30, 'sigma': 1.0},
            {'name': '45 DTE', 'dte': 45, 'sigma': 1.2}
        ]
        
        options_data = []
        
        for level in option_levels:
            dte = level['dte']
            sigma = level['sigma']
            
            # Calculate time factor (sqrt of days/365)
            time_factor = np.sqrt(dte / 365.0)
            
            # Calculate expected move based on volatility
            expected_move = current_price * (volatility / 100.0) * sigma * time_factor
            
            # Calculate strike prices
            put_strike = current_price - expected_move
            call_strike = current_price + expected_move
            
            # Beta adjustment for Greeks
            option_beta = underlying_beta * 0.8  # Options typically have lower beta
            
            # Simplified Greeks calculations
            # Delta: measures price sensitivity
            put_delta = -0.16 * option_beta  # Puts have negative delta
            call_delta = 0.16 * option_beta   # Calls have positive delta
            
            # Theta: time decay (more decay as DTE decreases)
            theta_factor = 1.0 / np.sqrt(dte)
            put_theta = -0.02 * theta_factor
            call_theta = -0.02 * theta_factor
            
            # Probability of Touch (simplified)
            # Based on normal distribution approximation
            prob_touch_put = min(99.0, (sigma * 34.0))  # ~34% per sigma
            prob_touch_call = min(99.0, (sigma * 34.0))
            
            options_data.append({
                'Level': level['name'],
                'DTE': dte,
                'Put Strike': f"${put_strike:.2f}",
                'Put PoT': f"{prob_touch_put:.1f}%",
                'Put Delta': f"{put_delta:.2f}",
                'Put Theta': f"{put_theta:.2f}",
                'Call Strike': f"${call_strike:.2f}",
                'Call PoT': f"{prob_touch_call:.1f}%",
                'Call Delta': f"{call_delta:.2f}",
                'Call Theta': f"{call_theta:.2f}",
                'Beta': f"{option_beta:.2f}",
                'Expected Move': f"±${expected_move:.2f}"
            })

        return options_data
        
    except Exception as e:
        logger.error(f"Options levels calculation error: {e}")
        return []

@safe_calculation_wrapper
def calculate_confidence_intervals(data: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Calculate statistical confidence intervals"""
    try:
        if not hasattr(data, 'resample') or len(data) < 100:
            return None

        weekly_data = data.resample('W-FRI')['Close'].last().dropna()
        weekly_returns = weekly_data.pct_change().dropna()

        if len(weekly_returns) < 20:
            return None

        mean_return = weekly_returns.mean()
        std_return = weekly_returns.std()
        current_price = float(data['Close'].iloc[-1])

        confidence_intervals = {}
        z_scores = {'68%': 1.0, '80%': 1.28, '95%': 1.96}

        for conf_level, z_score in z_scores.items():
            upper_bound = current_price * (1 + mean_return + z_score * std_return)
            lower_bound = current_price * (1 + mean_return - z_score * std_return)
            expected_move_pct = z_score * std_return * 100

            confidence_intervals[conf_level] = {
                'upper_bound': round(float(upper_bound), 2),
                'lower_bound': round(float(lower_bound), 2),
                'expected_move_pct': round(float(expected_move_pct), 2)
            }

        return {
            'mean_weekly_return': round(float(mean_return * 100), 3),
            'weekly_volatility': round(float(std_return * 100), 2),
            'confidence_intervals': confidence_intervals,
            'sample_size': int(len(weekly_returns))
        }
    except Exception as e:
        logger.error(f"Confidence intervals calculation error: {e}")
        return None
