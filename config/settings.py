"""
File: settings.py
Version: 1.0.0
VWV Research And Analysis System
Created: 2025-07-15
Updated: 2025-11-20
Purpose: Configuration settings for the VWV Research And Analysis System
System Version: v1.0.0 - Initial Release of Research And Analysis System
"""

# System identification
SYSTEM_NAME = "VWV Research And Analysis System"
SYSTEM_VERSION = "1.0.0"
SYSTEM_DESCRIPTION = "Professional market research and analysis platform"

# Default VWV System Configuration - Enhanced
DEFAULT_VWV_CONFIG = {
    'wvf_period': 22,
    'wvf_multiplier': 1.2,
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

# Williams VIX Fix Core Configuration
VWV_CORE_CONFIG = {
    'lookback_period': 22,
    'wvf_multiplier': 1.2,
    'ma_fast': 20,
    'ma_medium': 50,
    'ma_slow': 200,
    'volume_fast': 20,
    'volume_slow': 50,
    'vwap_deviation_multiplier': 1.5,
    'rsi_period': 14,
    'rsi_oversold_level': 30.0,
    'volatility_period': 20,
    'volatility_multiplier': 2.0,
    'final_scaling_multiplier': 1.5,
    'component_weights': {
        'wvf': 1.5,
        'ma_confluence': 1.2,
        'volume_confluence': 0.8,
        'vwap_analysis': 0.6,
        'momentum': 0.7,
        'volatility_filter': 0.4
    }
}

# Market Divergence Configuration
DIVERGENCE_CONFIG = {
    'benchmark_etfs': ['MAGS', 'FNGU', 'FNGD', 'SPY', 'QQQ'],
    'ema_short': 21,
    'ema_long': 55,
    'rsi_length': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'volume_ma_length': 20,
    'divergence_threshold': 0.25,
    'strong_divergence_threshold': 0.6,
    'extreme_divergence_threshold': 0.8,
    'scoring_weights': {
        'position': 0.3,
        'momentum': 0.3,
        'slope': 0.2,
        'volume': 0.2
    }
}

# Momentum Divergence Detection Configuration
MOMENTUM_DIVERGENCE_CONFIG = {
    'lookback_period': 20,  # Bars to look back for divergence detection
    'min_swing_distance': 5,  # Minimum bars between peaks/troughs
    'peak_prominence': 0.02,  # 2% minimum prominence for peak detection
    'oscillators': ['rsi', 'mfi', 'stochastic', 'williams_r'],  # Oscillators to check
    'score_weights': {
        'bullish_divergence': 15,  # Points for bullish divergence
        'bearish_divergence': -15,  # Points for bearish divergence
        'hidden_bullish': 10,  # Points for hidden bullish divergence
        'hidden_bearish': -10  # Points for hidden bearish divergence
    },
    'thresholds': {
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'mfi_oversold': 20,
        'mfi_overbought': 80,
        'stochastic_oversold': 20,
        'stochastic_overbought': 80,
        'williams_oversold': -80,
        'williams_overbought': -20
    }
}

# Master Score System Configuration
MASTER_SCORE_CONFIG = {
    'weights': {
        'technical': 0.25,      # Technical analysis weight
        'fundamental': 0.20,    # Fundamental analysis weight
        'vwv_signal': 0.15,     # VWV signal weight
        'momentum': 0.15,       # Momentum indicators weight
        'divergence': 0.10,     # Divergence detection weight
        'volume': 0.10,         # Volume analysis weight
        'volatility': 0.05      # Volatility analysis weight
    },
    'normalization': {
        'technical_max': 100,
        'fundamental_max': 100,
        'vwv_max': 10,
        'momentum_max': 100,
        'divergence_max': 30,
        'volume_max': 5,
        'volatility_max': 5
    },
    'score_thresholds': {
        'extreme_bullish': 80,
        'strong_bullish': 70,
        'moderate_bullish': 60,
        'neutral_high': 55,
        'neutral': 50,
        'neutral_low': 45,
        'moderate_bearish': 40,
        'strong_bearish': 30,
        'extreme_bearish': 20
    },
    'signal_strength': {
        'very_strong': 85,
        'strong': 70,
        'moderate': 55,
        'weak': 45,
        'very_weak': 30
    }
}

# Insider Analysis Configuration
INSIDER_CONFIG = {
    'lookback_days': 30,
    'extended_lookback_days': 90,
    'min_transaction_size': 10000,  # Minimum $10k transaction
    'executive_weight_multiplier': 2.0,  # Weight executive transactions higher
    'score_thresholds': {
        'extreme_bullish': 80,
        'bullish': 60,
        'neutral_high': 20,
        'neutral_low': -20,
        'bearish': -60,
        'extreme_bearish': -80
    },
    'transaction_weights': {
        'ceo': 3.0,
        'cfo': 2.5,
        'president': 2.0,
        'director': 1.5,
        'officer': 1.2,
        'other': 1.0
    }
}

# Enhanced Options Configuration with Sigma Levels
OPTIONS_ENHANCED_CONFIG = {
    'default_dte': [7, 14, 30, 45],
    'target_delta': 0.16,
    'risk_free_rate': 0.05,
    'sigma_levels': {
        'conservative': {
            'multiplier': 0.5,
            'fibonacci_weight': 0.5,
            'volatility_weight': 0.3,
            'volume_weight': 0.2,
            'target_pot': 15  # 15% probability of touch
        },
        'moderate': {
            'multiplier': 1.0,
            'fibonacci_weight': 0.35,
            'volatility_weight': 0.45,
            'volume_weight': 0.2,
            'target_pot': 25  # 25% probability of touch
        },
        'aggressive': {
            'multiplier': 1.5,
            'fibonacci_weight': 0.25,
            'volatility_weight': 0.6,
            'volume_weight': 0.15,
            'target_pot': 35  # 35% probability of touch
        }
    },
    'fibonacci_lookback': 5,  # 5-day rolling average for fibonacci base
    'volatility_lookback': 20,
    'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.786]
}

# Tech Sentiment Configuration
TECH_SENTIMENT_CONFIG = {
    'etf_symbols': ['FNGD', 'FNGU'],
    'ema_short': 20,
    'ema_medium': 50,
    'slope_periods': 5,
    'sentiment_smoothing': 3,
    'thresholds': {
        'extreme_bullish': 70,
        'strong_bullish': 40,
        'moderate_bullish': 15,
        'neutral_high': 5,
        'neutral_low': -5,
        'moderate_bearish': -15,
        'strong_bearish': -40,
        'extreme_bearish': -70
    },
    'weights': {
        'price_position': 0.4,
        'ema_relationship': 0.3,
        'slope_momentum': 0.3
    }
}

# Enhanced Breakout/Breakdown Configuration
BREAKOUT_CONFIG = {
    'symbols': ['SPY', 'QQQ', 'IWM'],
    'timeframes': {
        'short_term': 5,   # 5-day breakouts
        'medium_term': 10, # 10-day breakouts
        'long_term': 20    # 20-day breakouts
    },
    'ma_periods': [20, 50, 200],
    'volume_confirmation_required': True,
    'min_volume_ratio': 1.2,  # 1.2x average volume for confirmation
    'momentum_confirmation': True,
    'breakout_threshold_pct': 0.5,  # 0.5% minimum move for breakout
    'weights': {
        'ma_breakout': 0.4,
        'range_breakout': 0.4,
        'volume_breakout': 0.2
    },
    'rsi_confirmation': {
        'breakout_min': 50,  # RSI > 50 for bullish breakout
        'breakdown_max': 50  # RSI < 50 for bearish breakdown
    }
}

# System parameter ranges for UI sliders
PARAMETER_RANGES = {
    'wvf_period': {'min': 10, 'max': 50, 'default': 22},
    'wvf_multiplier': {'min': 0.5, 'max': 3.0, 'default': 1.2, 'step': 0.1},
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
    'insider_data_ttl': 3600,  # 1 hour for insider data
    'sentiment_data_ttl': 900,  # 15 minutes for sentiment data
    'analysis_cache_size': 50  # Number of analysis results to cache
}

# UI Settings - UPDATED DEFAULT PERIOD TO 3MO
UI_SETTINGS = {
    'default_symbol': 'SPY',
    'default_period': '3mo',  # CHANGED FROM 1y TO 3mo
    'max_recent_symbols': 9,
    'periods': ['1mo', '3mo', '6mo', '1y', '2y'],
    'system_name': 'VWV Research And Analysis System',
    'version': '1.0.0'
}

# Requirements update
REQUIREMENTS_V5 = [
    'streamlit>=1.28.0',
    'pandas>=1.5.0', 
    'numpy>=1.21.0',
    'yfinance>=0.2.20',
    'plotly>=5.15.0',
    'scipy>=1.9.0'  # Added for enhanced options calculations
]

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

def get_vwv_config():
    """Get Williams VIX Fix configuration"""
    return VWV_CORE_CONFIG.copy()

def get_divergence_config():
    """Get market divergence configuration"""
    return DIVERGENCE_CONFIG.copy()

def get_insider_config():
    """Get insider analysis configuration"""
    return INSIDER_CONFIG.copy()

def get_options_config():
    """Get enhanced options configuration"""
    return OPTIONS_ENHANCED_CONFIG.copy()

def get_sentiment_config():
    """Get tech sentiment configuration"""
    return TECH_SENTIMENT_CONFIG.copy()

def get_breakout_config():
    """Get enhanced breakout configuration"""
    return BREAKOUT_CONFIG.copy()

def get_momentum_divergence_config():
    """Get momentum divergence detection configuration"""
    return MOMENTUM_DIVERGENCE_CONFIG.copy()

def get_master_score_config():
    """Get master score system configuration"""
    return MASTER_SCORE_CONFIG.copy()
