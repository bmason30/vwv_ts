"""
Options analysis and Greeks calculations
"""
import math
import logging
from typing import List, Dict, Any
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def calculate_options_levels_enhanced(current_price, volatility, days_to_expiry=[7, 14, 30, 45], risk_free_rate=0.05, underlying_beta=1.0):
    """Enhanced options levels with proper Black-Scholes approximation and Greeks"""
    try:
        # Try to import scipy for more accurate calculations
        try:
            from scipy.stats import norm
            use_scipy = True
        except ImportError:
            use_scipy = False
            logger.warning("scipy not available, using simplified calculations")
        
        options_data = []

        for dte in days_to_expiry:
            T = dte / 365.0  # Time to expiration in years
            vol_annual = volatility / 100.0  # Convert percentage to decimal
            
            if use_scipy:
                # For ~16 delta (0.16), use inverse normal distribution
                delta_16 = 0.16
                z_score = norm.ppf(delta_16)  # ≈ -0.994
                
                # More accurate strike calculation using Black-Scholes framework
                drift = (risk_free_rate - 0.5 * vol_annual**2) * T
                vol_term = vol_annual * math.sqrt(T)
                
                # Put strike (16 delta put)
                put_strike = current_price * math.exp(drift + z_score * vol_term)
                
                # Call strike (16 delta call - using positive z-score)
                call_strike = current_price * math.exp(drift - z_score * vol_term)
                
                # Probability of Touch (more accurate)
                prob_touch_put = 2 * norm.cdf(z_score) * 100
                prob_touch_call = 2 * (1 - norm.cdf(-z_score)) * 100
                
            else:
                # Simplified calculation without scipy
                daily_vol = vol_annual / math.sqrt(252)
                std_move = current_price * daily_vol * math.sqrt(dte)
                
                # Approximate 16-delta strikes
                put_strike = current_price - std_move * 1.2  # Approximate adjustment
                call_strike = current_price + std_move * 1.2
                
                # Simplified probability calculation
                prob_touch_put = min(35, 35 * (std_move / current_price) * 100)
                prob_touch_call = prob_touch_put
            
            # Expected move (1 standard deviation)
            expected_move = current_price * vol_annual * math.sqrt(T)
            
            # Calculate Greeks
            # Delta calculation (approximate for 16-delta options)
            put_delta = -0.16  # Put delta is negative
            call_delta = 0.16   # Call delta is positive
            
            # Theta calculation (time decay per day)
            # Simplified theta estimation: higher for ATM, lower for OTM
            put_moneyness = put_strike / current_price
            call_moneyness = call_strike / current_price
            
            # Theta increases as expiration approaches and decreases for OTM options
            time_factor = math.sqrt(T)
            
            # Simplified theta calculation (option value / days remaining * time decay factor)
            put_theta = -(current_price * vol_annual * 0.4 * put_moneyness) / math.sqrt(dte) if dte > 0 else 0
            call_theta = -(current_price * vol_annual * 0.4 * call_moneyness) / math.sqrt(dte) if dte > 0 else 0
            
            # Beta (underlying's market beta - same for all options on same underlying)
            option_beta = underlying_beta

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

def calculate_confidence_intervals(data):
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
        current_price = data['Close'].iloc[-1]

        confidence_intervals = {}
        z_scores = {'68%': 1.0, '80%': 1.28, '95%': 1.96}

        for conf_level, z_score in z_scores.items():
            upper_bound = current_price * (1 + mean_return + z_score * std_return)
            lower_bound = current_price * (1 + mean_return - z_score * std_return)
            expected_move_pct = z_score * std_return * 100

            confidence_intervals[conf_level] = {
                'upper_bound': round(upper_bound, 2),
                'lower_bound': round(lower_bound, 2),
                'expected_move_pct': round(expected_move_pct, 2)
            }

        return {
            'mean_weekly_return': round(mean_return * 100, 3),
            'weekly_volatility': round(std_return * 100, 2),
            'confidence_intervals': confidence_intervals,
            'sample_size': len(weekly_returns)
        }
    except Exception as e:
        logger.error(f"Confidence intervals calculation error: {e}")
        return None
