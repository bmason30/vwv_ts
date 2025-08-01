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
    calculate_composite_technical_score,
    calculate_enhanced_technical_analysis
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

# Volume and Volatility imports - CORRECTED function names
try:
    from .volume import (
        calculate_complete_volume_analysis,
        calculate_market_wide_volume_analysis,
        get_volume_trading_implications,
        get_volume_regime_color
    )
    # Create aliases for compatibility
    calculate_volume_analysis = calculate_complete_volume_analysis
    calculate_market_volume_comparison = calculate_market_wide_volume_analysis
    
    VOLUME_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Volume analysis import failed: {e}")
    # Provide dummy functions to prevent crashes
    def calculate_complete_volume_analysis(*args, **kwargs):
        return {'error': 'Volume analysis not available'}
    def calculate_volume_analysis(*args, **kwargs):
        return {'error': 'Volume analysis not available'}
    def calculate_market_wide_volume_analysis(*args, **kwargs):
        return {'error': 'Market volume analysis not available'}
    def calculate_market_volume_comparison(*args, **kwargs):
        return {'error': 'Market volume comparison not available'}
    def get_volume_trading_implications(*args, **kwargs):
        return 'Volume analysis not available'
    def get_volume_regime_color(*args, **kwargs):
        return '#808080'
    
    VOLUME_AVAILABLE = False

try:
    from .volatility import (
        calculate_complete_volatility_analysis,
        calculate_market_wide_volatility_analysis,
        get_volatility_trading_implications,
        get_volatility_regime_color
    )
    # Create aliases for compatibility
    calculate_volatility_analysis = calculate_complete_volatility_analysis
    calculate_market_volatility_comparison = calculate_market_wide_volatility_analysis
    
    VOLATILITY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Volatility analysis import failed: {e}")
    # Provide dummy functions to prevent crashes
    def calculate_complete_volatility_analysis(*args, **kwargs):
        return {'error': 'Volatility analysis not available'}
    def calculate_volatility_analysis(*args, **kwargs):
        return {'error': 'Volatility analysis not available'}
    def calculate_market_wide_volatility_analysis(*args, **kwargs):
        return {'error': 'Market volatility analysis not available'}
    def calculate_market_volatility_comparison(*args, **kwargs):
        return {'error': 'Market volatility comparison not available'}
    def get_volatility_trading_implications(*args, **kwargs):
        return 'Volatility analysis not available'
    def get_volatility_regime_color(*args, **kwargs):
        return '#808080'
    
    VOLATILITY_AVAILABLE = False

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
    'calculate_enhanced_technical_analysis',
    
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
    
    # Volume analysis (NEW v4.2.1) - Both original and alias names
    'calculate_complete_volume_analysis',
    'calculate_volume_analysis',  # Alias
    'calculate_market_wide_volume_analysis',
    'calculate_market_volume_comparison',  # Alias
    'get_volume_trading_implications',
    'get_volume_regime_color',
    
    # Volatility analysis (NEW v4.2.1) - Both original and alias names
    'calculate_complete_volatility_analysis',
    'calculate_volatility_analysis',  # Alias
    'calculate_market_wide_volatility_analysis',
    'calculate_market_volatility_comparison',  # Alias
    'get_volatility_trading_implications',
    'get_volatility_regime_color'
]
