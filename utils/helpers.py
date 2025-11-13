"""
Helper utility functions
"""
import pandas as pd
import numpy as np
from typing import Union

def format_large_number(num: Union[int, float]) -> str:
    """Format large numbers with appropriate suffixes"""
    try:
        num = float(num)
        if abs(num) >= 1e12:
            return f"{num/1e12:.1f}T"
        elif abs(num) >= 1e9:
            return f"{num/1e9:.1f}B"
        elif abs(num) >= 1e6:
            return f"{num/1e6:.1f}M"
        elif abs(num) >= 1e3:
            return f"{num/1e3:.1f}K"
        else:
            return f"{num:.0f}"
    except:
        return str(num)

def statistical_normalize(series, lookback_period=252):
    """Simple statistical normalization"""
    try:
        if not hasattr(series, 'rolling') or len(series) < 10:
            if hasattr(series, '__iter__'):
                return 0.5
            else:
                return float(np.clip(series, 0, 1))

        if len(series) < lookback_period:
            lookback_period = len(series)

        percentile = series.rolling(window=lookback_period).rank(pct=True)
        result = percentile.iloc[-1] if not pd.isna(percentile.iloc[-1]) else 0.5
        return float(result)
    except Exception:
        return 0.5

def get_market_status():
    """Get current market status"""
    try:
        from datetime import datetime
        import pytz
        
        # US Eastern Time
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        
        # Market hours: 9:30 AM - 4:00 PM ET, Mon-Fri
        if now.weekday() < 5:  # Monday = 0, Friday = 4
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            
            if market_open <= now <= market_close:
                return "🟢 Market Open"
            elif now < market_open:
                return "🟡 Pre-Market"
            else:
                return "🔴 After Hours"
        else:
            return "🔴 Market Closed"
    except:
        return "🟡 Status Unknown"

def get_correlation_description(corr: float) -> str:
    """Get description of correlation strength"""
    abs_corr = abs(corr)
    if abs_corr >= 0.8:
        return "Very Strong"
    elif abs_corr >= 0.6:
        return "Strong"
    elif abs_corr >= 0.4:
        return "Moderate"
    elif abs_corr >= 0.2:
        return "Weak"
    else:
        return "Very Weak"

def get_etf_description(etf: str) -> str:
    """Get description of ETF"""
    descriptions = {
        'FNGD': '🐻 3x Inverse Technology ETF',
        'FNGU': '📈 3x Leveraged Technology ETF',
        'MAGS': '🏛️ Magnificent Seven ETF'
    }
    return descriptions.get(etf, 'ETF')
