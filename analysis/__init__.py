"""
Analysis package initialization file.
This file makes the core analysis functions available for easy import.
"""

# Import from technical.py
from .technical import (
    calculate_daily_vwap,
    calculate_fibonacci_emas,
    calculate_point_of_control_enhanced,
    calculate_comprehensive_technicals,
    calculate_weekly_deviations,
    calculate_composite_technical_score,
    generate_technical_signals,
    calculate_enhanced_technical_analysis
)

# Import from fundamental.py
from .fundamental import (
    calculate_graham_score,
    calculate_piotroski_score
)

# Import from market.py
from .market import (
    calculate_market_correlations_enhanced,
    calculate_breakout_breakdown_analysis
)

# Import from options.py
from .options import (
    calculate_options_levels_enhanced,
    calculate_confidence_intervals
)

# Safe imports for optional modules
try:
    from .volume import calculate_complete_volume_analysis
except ImportError:
    calculate_complete_volume_analysis = None

try:
    from .volatility import calculate_complete_volatility_analysis
except ImportError:
    calculate_complete_volatility_analysis = None

try:
    from .baldwin_indicator import calculate_baldwin_indicator_complete, format_baldwin_for_display
except ImportError:
    calculate_baldwin_indicator_complete = None
    format_baldwin_for_display = None
