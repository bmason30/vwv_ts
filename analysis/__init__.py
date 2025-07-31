"""
Analysis module for VWV Trading System - Updated with Volume & Volatility
"""

from .technical import (
    safe_rsi,
    calculate_daily_vwap,
    calculate_fibonacci_emas,
    calculate_point_of_control_enhanced,
    calculate_comprehensive_technicals,
    calculate_mfi,
    calculate_macd,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_stochastic,
    calculate_williams_r,
    calculate_weekly_deviations,
    calculate_composite_technical_score
)

from .fundamental import (
    calculate_graham_score,
    calculate_piotroski_score,
    get_graham_grade,
    get_graham_interpretation,
    get_piotroski_grade,
    get_piotroski_interpretation
)

from .market import (
    calculate_market_correlations_enhanced,
    calculate_breakout_breakdown_analysis,
    get_correlation_description
)

from .options import (
    calculate_options_levels_enhanced,
    calculate_confidence_intervals
)

# New Volume Analysis Module
from .volume import (
    calculate_volume_analysis,
    get_volume_interpretation,
    calculate_market_volume_comparison
)

# New Volatility Analysis Module  
from .volatility import (
    calculate_volatility_analysis,
    get_volatility_interpretation,
    calculate_market_volatility_comparison,
    get_volatility_regime_for_options
)

__all__ = [
    # Technical analysis
    'safe_rsi',
    'calculate_daily_vwap',
    'calculate_fibonacci_emas',
    'calculate_point_of_control_enhanced',
    'calculate_comprehensive_technicals',
    'calculate_mfi',
    'calculate_macd',
    'calculate_atr',
    'calculate_bollinger_bands',
    'calculate_stochastic',
    'calculate_williams_r',
    'calculate_weekly_deviations',
    'calculate_composite_technical_score',
    
    # Fundamental analysis
    'calculate_graham_score',
    'calculate_piotroski_score',
    'get_graham_grade',
    'get_graham_interpretation',
    'get_piotroski_grade',
    'get_piotroski_interpretation',
    
    # Market analysis
    'calculate_market_correlations_enhanced',
    'calculate_breakout_breakdown_analysis',
    'get_correlation_description',
    
    # Options analysis
    'calculate_options_levels_enhanced',
    'calculate_confidence_intervals',
    
    # Volume analysis - NEW
    'calculate_volume_analysis',
    'get_volume_interpretation',
    'calculate_market_volume_comparison',
    
    # Volatility analysis - NEW
    'calculate_volatility_analysis',
    'get_volatility_interpretation',
    'calculate_market_volatility_comparison',
    'get_volatility_regime_for_options'
]
