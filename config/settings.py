"""
Configuration settings for the VWV Trading System
"""

# Default VWV System Configuration
DEFAULT_VWV_CONFIG = {
    'wvf_period': 22,
    'wvf_multiplier': 2.0,
    'ma_periods': [20, 50, 200],
    'volume_periods': [20, 50],
    'rsi_period': 14,
    'volatility_period': 20,
    'weights': {
        'wvf': 0.8, 
        'ma': 1.2, 
        'volume': 0.6,
        'vwap': 0.4, 
        'momentum': 0.5, 
        'volatility': 0.3
    },
    'scaling_multiplier': 1.5,
    'signal_thresholds': {
        'good': 3.5, 
        'strong': 4.5, 
        'very_strong': 5.5
    },
    'stop_loss_pct': 0.022,
    'take_profit_pct': 0.055
}

# System parameter ranges for UI sliders
PARAMETER_RANGES = {
    'wvf_period': {'min': 10, 'max': 50, 'default': 22},
    'wvf_multiplier': {'min': 0.5, 'max': 3.0, 'default': 2.0, 'step': 0.1},
    'good_threshold': {'min': 2.0, 'max': 5.0, 'default': 3.5, 'step': 0.1},
    'strong_threshold': {'min': 3.0, 'max': 6.0, 'default': 4.5, 'step': 0.1},
    'very_strong_threshold': {'min': 4.0, 'max': 7.0, 'default': 5.5, 'step': 0.1}
}

# Fibonacci EMA periods
FIBONACCI_EMA_PERIODS = [21, 55, 89, 144, 233]

# Technical analysis periods
TECHNICAL_PERIODS = {
    'rsi': 14,
    'mfi': 14,
    'williams_r': 14,
    'stochastic_k': 14,
    'stochastic_d': 3,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'atr': 14,
    'bollinger_period': 20,
    'bollinger_std': 2,
    'volume_sma': 20,
    'volatility_period': 20
}

# Options analysis settings
OPTIONS_SETTINGS = {
    'default_dte': [7, 14, 30, 45],
    'target_delta': 0.16,
    'risk_free_rate': 0.05
}

# Chart settings
CHART_SETTINGS = {
    'default_periods': 100,  # Number of periods to show on chart
    'height': 800,
    'template': 'plotly_white'
}

# Cache settings
CACHE_SETTINGS = {
    'market_data_ttl': 300,  # 5 minutes
    'correlation_data_ttl': 600,  # 10 minutes
    'analysis_cache_size': 50  # Number of analysis results to cache
}

# UI Settings
UI_SETTINGS = {
    'default_symbol': 'SPY',
    'default_period': '1y',
    'max_recent_symbols': 9,
    'periods': ['1mo', '3mo', '6mo', '1y', '2y']
}

def load_default_config():
    """Load default configuration"""
    return DEFAULT_VWV_CONFIG.copy()

def get_parameter_range(param_name):
    """Get parameter range for UI sliders"""
    return PARAMETER_RANGES.get(param_name, {})

def validate_config(config):
    """Validate configuration parameters"""
    errors = []
    
    # Validate required keys
    required_keys = ['wvf_period', 'wvf_multiplier', 'signal_thresholds']
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required config key: {key}")
    
    # Validate thresholds are in ascending order
    if 'signal_thresholds' in config:
        thresholds = config['signal_thresholds']
        if thresholds.get('good', 0) >= thresholds.get('strong', 0):
            errors.append("Strong threshold must be greater than good threshold")
        if thresholds.get('strong', 0) >= thresholds.get('very_strong', 0):
            errors.append("Very strong threshold must be greater than strong threshold")
    
    return errors
