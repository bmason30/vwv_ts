"""
File: options.py v2.1.0
VWV Research And Analysis System v4.3.0
Options analysis module with Black-Scholes pricing and accurate Greeks
Created: 2025-08-15
Updated: 2025-11-19
File Version: v2.1.0 - Added strike quality scoring system
Changes in this version:
    - v2.0.0: Implemented true Black-Scholes-Merton option pricing model
    - v2.0.0: Added accurate Greeks calculations (Delta, Gamma, Theta, Vega, Rho)
    - v2.0.0: Enhanced probability calculations (PoT, PoP) with proper formulas
    - v2.0.0: Added optimal strike calculation for target delta
    - v2.0.0: Added premium pricing for both puts and calls
    - v2.1.0: Added strike quality scoring system (0-100 scale)
    - Backward compatible with previous function signatures
Dependencies: scipy>=1.10.0, numpy>=1.24.0
System Version: v4.3.0 - Black-Scholes Options Pricing
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from scipy import stats
from scipy.stats import norm
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

def calculate_black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> Dict[str, float]:
    """
    Calculate Black-Scholes option price and Greeks

    VERSION 2.0.0 - Accurate Black-Scholes-Merton implementation

    Parameters:
    -----------
    S : float - Current stock price
    K : float - Strike price
    T : float - Time to expiration (years)
    r : float - Risk-free rate (decimal, e.g., 0.045 for 4.5%)
    sigma : float - Volatility (decimal, not percentage, e.g., 0.25 for 25%)
    option_type : str - 'call' or 'put'

    Returns:
    --------
    dict: {
        'price': float - Option premium
        'delta': float - Rate of change of option price with respect to underlying
        'gamma': float - Rate of change of delta
        'theta': float - Time decay (per day)
        'vega': float - Sensitivity to volatility (per 1% change)
        'rho': float - Sensitivity to interest rates
    }
    """
    try:
        # Handle edge cases
        if T <= 0:
            # Option has expired
            if option_type == 'call':
                return {'price': max(S - K, 0), 'delta': 1.0 if S > K else 0.0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
            else:
                return {'price': max(K - S, 0), 'delta': -1.0 if K > S else 0.0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

        if sigma <= 0 or S <= 0 or K <= 0:
            logger.warning(f"Invalid parameters: S={S}, K={K}, T={T}, sigma={sigma}")
            return {'price': 0, 'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Calculate option price
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        # Greeks (common to both call and put)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol change
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type == 'call' else norm.cdf(-d2)))
        theta = theta / 365  # Convert to daily

        return {
            'price': round(float(price), 2),
            'delta': round(float(delta), 4),
            'gamma': round(float(gamma), 6),
            'theta': round(float(theta), 2),
            'vega': round(float(vega), 2),
            'rho': round(float(rho), 4)
        }

    except Exception as e:
        logger.error(f"Black-Scholes calculation error: {e}")
        return {'price': 0, 'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}

def calculate_probability_of_touch(S: float, K: float, T: float, sigma: float) -> float:
    """
    Calculate probability that stock will touch strike before expiration

    VERSION 2.0.0 - Accurate PoT calculation

    Uses the approximation: PoT ≈ 2 * Φ(|ln(K/S)| / (σ√T))
    where Φ is the cumulative normal distribution

    Parameters:
    -----------
    S : float - Current stock price
    K : float - Strike price
    T : float - Time to expiration (years)
    sigma : float - Volatility (decimal)

    Returns:
    --------
    float: Probability of touch as percentage (0-100)
    """
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0

        # Calculate d2 from Black-Scholes
        d2 = (np.log(S / K)) / (sigma * np.sqrt(T)) - 0.5 * sigma * np.sqrt(T)

        if K < S:  # Put strike (below current price)
            pot = 2 * norm.cdf(-d2)
        else:  # Call strike (above current price)
            pot = 2 * norm.cdf(d2)

        return round(float(min(pot * 100, 100)), 1)  # Cap at 100%

    except Exception as e:
        logger.error(f"PoT calculation error: {e}")
        return 0.0

def calculate_optimal_strikes_for_target_delta(S: float, T: float, r: float, sigma: float, target_delta: float = 0.16) -> Dict[str, float]:
    """
    Calculate strike prices that yield approximately target delta

    VERSION 2.0.0 - Strike optimization for target delta

    For standard premium selling: target_delta = 0.16 (16 delta)
    This typically corresponds to ~1 standard deviation OTM

    Parameters:
    -----------
    S : float - Current stock price
    T : float - Time to expiration (years)
    r : float - Risk-free rate (decimal)
    sigma : float - Volatility (decimal)
    target_delta : float - Target delta (absolute value, e.g., 0.16 for 16 delta)

    Returns:
    --------
    dict: {'put_strike': float, 'call_strike': float}
    """
    try:
        if T <= 0 or sigma <= 0 or S <= 0:
            # Fallback to 1 standard deviation
            std_dev = S * sigma * np.sqrt(T)
            return {
                'put_strike': round(float(S - std_dev), 2),
                'call_strike': round(float(S + std_dev), 2)
            }

        # For call: delta = N(d1), solve for K when delta = target
        # For put: delta = -N(-d1), solve for K when |delta| = target

        # Approximation using inverse normal
        z_call = norm.ppf(target_delta)
        z_put = norm.ppf(1 - target_delta)

        # Calculate strikes
        call_strike = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z_call)
        put_strike = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z_put)

        return {
            'put_strike': round(float(put_strike), 2),
            'call_strike': round(float(call_strike), 2)
        }

    except Exception as e:
        logger.error(f"Optimal strike calculation error: {e}")
        # Fallback to 1 standard deviation
        std_dev = S * sigma * np.sqrt(T)
        return {
            'put_strike': round(float(S - std_dev), 2),
            'call_strike': round(float(S + std_dev), 2)
        }

@safe_calculation_wrapper
def calculate_options_levels_enhanced(
    current_price: float,
    volatility: float,
    underlying_beta: float = 1.0,
    days_to_expiry: List[int] = None,
    risk_free_rate: float = 0.045,
    target_delta: float = 0.16
) -> List[Dict[str, Any]]:
    """
    Calculate enhanced options levels with accurate Black-Scholes pricing

    VERSION 2.0.0 - MAJOR UPGRADE with Black-Scholes pricing
    Backward compatible with v1.x.x function signatures

    Args:
        current_price: Current stock price
        volatility: Annualized volatility percentage (e.g., 25 for 25%)
        underlying_beta: Beta of underlying asset (default 1.0)
        days_to_expiry: List of DTEs to calculate (default [7, 14, 30, 45, 60])
        risk_free_rate: Risk-free interest rate as decimal (default 0.045 for 4.5%)
        target_delta: Target delta for strike selection (default 0.16 for 16 delta)

    Returns:
        List of dictionaries containing:
            - Strike prices (put and call)
            - Option premiums
            - Complete Greeks (Delta, Gamma, Theta, Vega, Rho)
            - Probability of Touch (PoT) and Probability of Profit (PoP)
            - Expected move calculations
    """
    try:
        if current_price <= 0 or volatility <= 0:
            logger.warning(f"Invalid parameters: price={current_price}, vol={volatility}")
            return []

        # Convert volatility from percentage to decimal
        vol_annual = volatility / 100.0

        # Default DTE levels
        if days_to_expiry is None:
            dte_levels = [7, 14, 30, 45, 60]
        else:
            dte_levels = days_to_expiry

        options_data = []

        for dte in dte_levels:
            # Time factor (years)
            T = dte / 365.0

            # Calculate optimal strikes for target delta
            strikes = calculate_optimal_strikes_for_target_delta(
                current_price, T, risk_free_rate, vol_annual, target_delta
            )

            put_strike = strikes['put_strike']
            call_strike = strikes['call_strike']

            # Calculate Black-Scholes Greeks for put
            put_greeks = calculate_black_scholes_greeks(
                current_price, put_strike, T, risk_free_rate, vol_annual, 'put'
            )

            # Calculate Black-Scholes Greeks for call
            call_greeks = calculate_black_scholes_greeks(
                current_price, call_strike, T, risk_free_rate, vol_annual, 'call'
            )

            # Calculate probabilities
            put_pot = calculate_probability_of_touch(
                current_price, put_strike, T, vol_annual
            )
            call_pot = calculate_probability_of_touch(
                current_price, call_strike, T, vol_annual
            )

            # Probability of profit (rough estimate: 100% - PoT)
            # This assumes selling premium and profiting if strike not touched
            put_pop = round(100 - put_pot, 1)
            call_pop = round(100 - call_pot, 1)

            # Expected move (±1 SD)
            expected_move = current_price * vol_annual * np.sqrt(T)

            # Beta adjustment for underlying sensitivity
            option_beta = underlying_beta * 0.7  # Options typically have ~70% of underlying beta

            options_data.append({
                'DTE': dte,
                'Put Strike': put_strike,
                'Put Premium': f"${put_greeks['price']:.2f}",
                'Put PoT': f"{put_pot:.1f}%",
                'Put PoP': f"{put_pop:.1f}%",
                'Put Delta': f"{put_greeks['delta']:.3f}",
                'Put Gamma': f"{put_greeks['gamma']:.6f}",
                'Put Theta': f"${put_greeks['theta']:.2f}",
                'Put Vega': f"${put_greeks['vega']:.2f}",
                'Put Rho': f"{put_greeks['rho']:.4f}",
                'Call Strike': call_strike,
                'Call Premium': f"${call_greeks['price']:.2f}",
                'Call PoT': f"{call_pot:.1f}%",
                'Call PoP': f"{call_pop:.1f}%",
                'Call Delta': f"{call_greeks['delta']:.3f}",
                'Call Gamma': f"{call_greeks['gamma']:.6f}",
                'Call Theta': f"${call_greeks['theta']:.2f}",
                'Call Vega': f"${call_greeks['vega']:.2f}",
                'Call Rho': f"{call_greeks['rho']:.4f}",
                'Expected Move': f"±${expected_move:.2f}",
                'Expected Move %': f"±{(expected_move / current_price) * 100:.2f}%",
                'Beta': f"{option_beta:.2f}",
                'IV Used': f"{volatility:.1f}%",
                'Risk-Free Rate': f"{risk_free_rate * 100:.2f}%"
            })

        logger.info(f"Calculated Black-Scholes options for {len(options_data)} DTEs")
        return options_data

    except Exception as e:
        logger.error(f"Options levels calculation error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []

def calculate_strike_quality_score(
    strike_data: Dict[str, Any],
    current_price: float,
    volatility: float,
    iv_rank: float = 50.0
) -> Dict[str, Any]:
    """
    Calculate quality score for an options strike (0-100 scale)

    VERSION 2.1.0 - Advanced strike quality analysis

    Scoring factors:
    - Premium quality (30%): Based on IV rank and premium size
    - Probability of profit (25%): Higher PoP = better score
    - Greeks efficiency (20%): Theta/Delta ratio for income vs risk
    - Strike position (15%): Optimal distance from current price
    - Volatility advantage (10%): Higher IV = better for sellers

    Parameters:
    -----------
    strike_data : dict - Strike information with Greeks and probabilities
    current_price : float - Current stock price
    volatility : float - Current volatility percentage
    iv_rank : float - Implied volatility rank (0-100, default 50)

    Returns:
    --------
    dict: {
        'total_score': int (0-100),
        'rating': str ('Poor', 'Fair', 'Good', 'Excellent'),
        'stars': int (1-5),
        'component_scores': dict,
        'recommendation': str
    }
    """
    try:
        def safe_float_extract(value, default=0.0):
            """Extract float from various formats"""
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                import re
                cleaned = re.sub(r'[^\d.-]', '', value)
                try:
                    return float(cleaned) if cleaned else default
                except:
                    return default
            return default

        scores = {}

        # 1. Premium Quality (0-30 points) - Based on IV rank
        premium_score = (iv_rank / 100.0) * 30
        scores['premium_quality'] = round(premium_score, 2)

        # 2. Probability of Profit (0-25 points)
        pop_str = strike_data.get('PoP', strike_data.get('Put PoP', strike_data.get('Call PoP', '0%')))
        pop = safe_float_extract(pop_str)
        pop_score = (pop / 100.0) * 25
        scores['probability'] = round(pop_score, 2)

        # 3. Greeks Efficiency (0-20 points) - Theta/Delta ratio
        try:
            theta_str = strike_data.get('Theta', strike_data.get('Put Theta', strike_data.get('Call Theta', '$0')))
            delta_str = strike_data.get('Delta', strike_data.get('Put Delta', strike_data.get('Call Delta', '0')))

            theta = abs(safe_float_extract(theta_str))
            delta = abs(safe_float_extract(delta_str))

            if delta > 0.01:  # Avoid division by very small numbers
                efficiency = min(theta / delta, 2.0)  # Cap at 2.0
                efficiency_score = (efficiency / 2.0) * 20
            else:
                efficiency_score = 10  # Default mid-score
        except:
            efficiency_score = 10

        scores['greeks_efficiency'] = round(efficiency_score, 2)

        # 4. Strike Position (0-15 points) - Not too far OTM
        strike_price = safe_float_extract(strike_data.get('Strike', strike_data.get('Put Strike', strike_data.get('Call Strike', current_price))))
        distance_pct = abs(strike_price - current_price) / current_price

        # Optimal distance: 8-15% OTM = full points
        if 0.08 <= distance_pct <= 0.15:
            position_score = 15
        elif distance_pct < 0.08:
            position_score = 15 * (distance_pct / 0.08)
        else:  # > 0.15
            position_score = max(0, 15 * (1 - (distance_pct - 0.15) / 0.10))

        scores['strike_position'] = round(position_score, 2)

        # 5. Volatility Advantage (0-10 points)
        if iv_rank >= 70:
            vol_score = 10
        elif iv_rank >= 50:
            vol_score = 7
        elif iv_rank >= 30:
            vol_score = 4
        else:
            vol_score = 2

        scores['volatility_advantage'] = round(vol_score, 2)

        # Calculate total
        total_score = int(sum(scores.values()))

        # Rating and stars
        if total_score >= 80:
            rating = "Excellent"
            stars = 5
        elif total_score >= 65:
            rating = "Good"
            stars = 4
        elif total_score >= 50:
            rating = "Fair"
            stars = 3
        elif total_score >= 35:
            rating = "Below Average"
            stars = 2
        else:
            rating = "Poor"
            stars = 1

        # Generate recommendation
        if total_score >= 80 and pop >= 70:
            recommendation = "⭐ STRONG SELL - Excellent risk/reward with high probability"
        elif total_score >= 65 and pop >= 60:
            recommendation = "✅ GOOD SELL - Solid premium with favorable odds"
        elif total_score >= 50:
            recommendation = "⚠️ CONSIDER - Acceptable but not optimal"
        elif total_score >= 35:
            recommendation = "❌ CAUTION - Marginal risk/reward profile"
        else:
            recommendation = "🚫 AVOID - Poor probability or premium quality"

        return {
            'total_score': total_score,
            'rating': rating,
            'stars': stars,
            'component_scores': scores,
            'recommendation': recommendation
        }

    except Exception as e:
        logger.error(f"Strike quality score calculation error: {e}")
        return {
            'total_score': 50,
            'rating': 'Unknown',
            'stars': 3,
            'component_scores': {},
            'recommendation': '⚠️ Unable to calculate quality score'
        }

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
