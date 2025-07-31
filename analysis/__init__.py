"""
Analysis module for VWV Trading System v4.2.1
Enhanced with Volume and Volatility Analysis modules
Fixed circular dependencies
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

# Volume and Volatility imports using try/except to avoid circular dependencies
try:
    from .volume import (
        calculate_volume_analysis,
        get_volume_interpretation,
        calculate_market_volume_comparison,
        classify_volume_regime,
        calculate_volume_strength_factor
    )
except ImportError as e:
    print(f"Warning: Volume analysis import failed: {e}")
    # Provide dummy functions to prevent crashes
    def calculate_volume_analysis(*args, **kwargs):
        return {'error': 'Volume analysis not available'}
    def get_volume_interpretation(*args, **kwargs):
        return {'regime_interpretation': 'Not available'}
    def calculate_market_volume_comparison(*args, **kwargs):
        return {'error': 'Market volume comparison not available'}
    def classify_volume_regime(*args, **kwargs):
        return "Unknown", 50
    def calculate_volume_strength_factor(*args, **kwargs):
        return 1.0

try:
    from .volatility import (
        calculate_volatility_analysis,
        get_volatility_interpretation,
        calculate_market_volatility_comparison,
        classify_volatility_regime,
        get_volatility_regime_for_options,
        calculate_volatility_strength_factor
    )
except ImportError as e:
    print(f"Warning: Volatility analysis import failed: {e}")
    # Provide dummy functions to prevent crashes
    def calculate_volatility_analysis(*args, **kwargs):
        return {'error': 'Volatility analysis not available'}
    def get_volatility_interpretation(*args, **kwargs):
        return {'regime_interpretation': 'Not available'}
    def calculate_market_volatility_comparison(*args, **kwargs):
        return {'error': 'Market volatility comparison not available'}
    def classify_volatility_regime(*args, **kwargs):
        return "Unknown", 50
    def get_volatility_regime_for_options(*args, **kwargs):
        return {'strategy': 'Neutral'}
    def calculate_volatility_strength_factor(*args, **kwargs):
        return 1.0

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
    
    # Volume analysis (NEW v4.2.1)
    'calculate_volume_analysis',
    'get_volume_interpretation', 
    'calculate_market_volume_comparison',
    'classify_volume_regime',
    'calculate_volume_strength_factor',
    
    # Volatility analysis (NEW v4.2.1)
    'calculate_volatility_analysis',
    'get_volatility_interpretation',
    'calculate_market_volatility_comparison', 
    'classify_volatility_regime',
    'get_volatility_regime_for_options',
    'calculate_volatility_strength_factor'
]
