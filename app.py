import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import copy
import logging
import json
import hashlib
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# Suppress only specific warnings, not all
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

# Page configuration
st.set_page_config(
    page_title="VWV Professional Trading System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .signal-good { background-color: #d4edda; border-left: 5px solid #28a745; }
    .signal-strong { background-color: #fff3cd; border-left: 5px solid #ffc107; }
    .signal-very-strong { background-color: #f8d7da; border-left: 5px solid #dc3545; }
    .signal-none { background-color: #e2e3e5; border-left: 5px solid #6c757d; }
    .signal-bearish { background-color: #f8d7da; border-left: 5px solid #dc3545; }
    .performance-excellent { background-color: #d1ecf1; border-left: 5px solid #bee5eb; }
    .performance-good { background-color: #fff3cd; border-left: 5px solid #ffc107; }
    .performance-poor { background-color: #f8d7da; border-left: 5px solid #dc3545; }
    .log-entry {
        font-family: 'Courier New', monospace;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 4px;
        margin: 0.2rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ENHANCED: Performance tracking and logging
@dataclass
class PerformanceMetrics:
    operation: str
    start_time: float
    end_time: float
    duration: float
    data_size: int
    memory_usage: Optional[float] = None
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    data_quality_score: float
    
class AdvancedLogger:
    """Enhanced logging system for production trading analysis"""
    
    def __init__(self):
        self.setup_logging()
        self.performance_log = []
        self.analysis_history = []
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger("VWVTradingSystem")
        self.logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            # File handler
            file_handler = logging.FileHandler(
                log_dir / f"vwv_trading_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_handler.setLevel(logging.INFO)
            
            # Console handler for Streamlit
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def log_performance(self, metrics: PerformanceMetrics):
        """Log performance metrics"""
        self.performance_log.append(metrics)
        self.logger.info(f"Performance: {metrics.operation} took {metrics.duration:.3f}s for {metrics.data_size} records")
    
    def log_analysis(self, symbol: str, analysis_result: Dict):
        """Log analysis results"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'signal_type': analysis_result.get('signal_type', 'NONE'),
            'confluence': analysis_result.get('directional_confluence', 0),
            'current_price': analysis_result.get('current_price', 0),
            'system_status': analysis_result.get('system_status', 'UNKNOWN')
        }
        
        self.analysis_history.append(log_entry)
        self.logger.info(f"Analysis: {symbol} - {log_entry['signal_type']} - Confluence: {log_entry['confluence']:.2f}")
    
    def get_performance_summary(self) -> Dict:
        """Get performance analytics summary"""
        if not self.performance_log:
            return {}
        
        durations = [m.duration for m in self.performance_log]
        operations = {}
        
        for metric in self.performance_log:
            op = metric.operation
            if op not in operations:
                operations[op] = []
            operations[op].append(metric.duration)
        
        return {
            'total_operations': len(self.performance_log),
            'avg_duration': np.mean(durations),
            'max_duration': np.max(durations),
            'min_duration': np.min(durations),
            'operations_breakdown': {
                op: {
                    'count': len(times),
                    'avg_time': np.mean(times),
                    'max_time': np.max(times)
                }
                for op, times in operations.items()
            }
        }

# ENHANCED: Optimized data manager with performance tracking
class OptimizedDataManager:
    """Advanced data manager with performance optimization and caching"""
    
    def __init__(self, logger: AdvancedLogger):
        self._market_data_store = {}
        self._analysis_store = {}
        self._data_hashes = {}  # For change detection
        self._cache_timestamps = {}
        self.logger = logger
        self.cache_ttl = 300  # 5 minutes cache TTL
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash for change detection"""
        return hashlib.md5(
            str(data.index[-1]) + str(data['Close'].iloc[-1]) + str(len(data))
        ).hexdigest()
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid"""
        if symbol not in self._cache_timestamps:
            return False
        
        cache_age = time.time() - self._cache_timestamps[symbol]
        return cache_age < self.cache_ttl
    
    def store_market_data(self, symbol: str, market_data: pd.DataFrame):
        """Optimized market data storage with change detection"""
        start_time = time.time()
        
        if not isinstance(market_data, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(market_data)}")
        
        # Calculate hash for change detection
        data_hash = self._calculate_data_hash(market_data)
        
        # Check if data actually changed
        if symbol in self._data_hashes and self._data_hashes[symbol] == data_hash:
            # Data unchanged, extend cache
            self._cache_timestamps[symbol] = time.time()
            self.logger.logger.info(f"Data unchanged for {symbol}, using cache")
            return
        
        # Store with optimized copying strategy
        if len(market_data) > 1000:
            # For large datasets, use more memory-efficient approach
            self._market_data_store[symbol] = market_data.copy(deep=False)  # Shallow copy first
        else:
            # For smaller datasets, use deep copy for safety
            self._market_data_store[symbol] = market_data.copy(deep=True)
        
        self._data_hashes[symbol] = data_hash
        self._cache_timestamps[symbol] = time.time()
        
        # Log performance
        duration = time.time() - start_time
        metrics = PerformanceMetrics(
            operation="store_market_data",
            start_time=start_time,
            end_time=time.time(),
            duration=duration,
            data_size=len(market_data)
        )
        self.logger.log_performance(metrics)
        
        st.write(f"🔒 Optimized storage for {symbol}: {market_data.shape} ({duration:.3f}s)")
    
    def get_market_data_for_analysis(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get optimized copy for analysis"""
        start_time = time.time()
        
        if symbol not in self._market_data_store:
            return None
        
        if not self._is_cache_valid(symbol):
            self.logger.logger.warning(f"Cache expired for {symbol}")
        
        # Optimized copying based on data size
        original_data = self._market_data_store[symbol]
        if len(original_data) > 1000:
            # For large datasets, use view-based approach for analysis
            analysis_copy = original_data.iloc[-1000:].copy(deep=True)  # Last 1000 rows only
        else:
            analysis_copy = original_data.copy(deep=True)
        
        # Log performance
        duration = time.time() - start_time
        metrics = PerformanceMetrics(
            operation="get_analysis_data",
            start_time=start_time,
            end_time=time.time(),
            duration=duration,
            data_size=len(analysis_copy)
        )
        self.logger.log_performance(metrics)
        
        return analysis_copy
    
    def get_market_data_for_chart(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get optimized copy for chart (last 200 periods max)"""
        start_time = time.time()
        
        if symbol not in self._market_data_store:
            return None
        
        original_data = self._market_data_store[symbol]
        
        # For charts, we only need recent data
        chart_periods = min(200, len(original_data))
        chart_copy = original_data.tail(chart_periods).copy(deep=True)
        
        # Verify integrity
        if not isinstance(chart_copy, pd.DataFrame):
            self.logger.logger.error(f"Chart data corrupted for {symbol}: {type(chart_copy)}")
            return None
        
        # Log performance
        duration = time.time() - start_time
        metrics = PerformanceMetrics(
            operation="get_chart_data",
            start_time=start_time,
            end_time=time.time(),
            duration=duration,
            data_size=len(chart_copy)
        )
        self.logger.log_performance(metrics)
        
        return chart_copy
    
    def store_analysis_results(self, symbol: str, analysis_results: Dict):
        """Store analysis results with timestamp"""
        self._analysis_store[symbol] = {
            'results': copy.deepcopy(analysis_results),
            'timestamp': time.time()
        }
        
        # Log analysis
        self.logger.log_analysis(symbol, analysis_results)
    
    def get_analysis_results(self, symbol: str) -> Dict:
        """Get analysis results"""
        if symbol not in self._analysis_store:
            return {}
        return self._analysis_store[symbol]['results']

# ENHANCED: Advanced data validation
class AdvancedDataValidator:
    """Comprehensive data quality validation"""
    
    @staticmethod
    def validate_market_data(data: pd.DataFrame, symbol: str) -> ValidationResult:
        """Comprehensive market data validation"""
        issues = []
        warnings = []
        quality_score = 100.0
        
        # Basic structure validation
        if not isinstance(data, pd.DataFrame):
            issues.append(f"Data is not a DataFrame: {type(data)}")
            return ValidationResult(False, issues, warnings, 0.0)
        
        if len(data) == 0:
            issues.append("DataFrame is empty")
            return ValidationResult(False, issues, warnings, 0.0)
        
        # Required columns validation
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
            quality_score -= 30
        
        # Data quality checks
        if len(data) < 50:
            warnings.append(f"Limited data history: only {len(data)} periods")
            quality_score -= 10
        
        # Price data validation
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data.columns:
                if (data[col] <= 0).any():
                    issues.append(f"Invalid {col} prices: found zero or negative values")
                    quality_score -= 20
                
                if data[col].isna().sum() > len(data) * 0.05:  # More than 5% missing
                    warnings.append(f"High missing data in {col}: {data[col].isna().sum()} values")
                    quality_score -= 15
        
        # Volume validation
        if 'Volume' in data.columns:
            if (data['Volume'] < 0).any():
                issues.append("Invalid volume data: found negative values")
                quality_score -= 15
            
            if data['Volume'].isna().sum() > len(data) * 0.1:
                warnings.append(f"High missing volume data: {data['Volume'].isna().sum()} values")
                quality_score -= 10
        
        # OHLC relationship validation
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            # High should be >= max(Open, Close)
            invalid_high = data['High'] < data[['Open', 'Close']].max(axis=1)
            if invalid_high.any():
                warnings.append(f"Invalid OHLC relationships found: {invalid_high.sum()} cases")
                quality_score -= 10
            
            # Low should be <= min(Open, Close)
            invalid_low = data['Low'] > data[['Open', 'Close']].min(axis=1)
            if invalid_low.any():
                warnings.append(f"Invalid OHLC relationships found: {invalid_low.sum()} cases")
                quality_score -= 10
        
        # Outlier detection
        if 'Close' in data.columns and len(data) > 20:
            returns = data['Close'].pct_change().dropna()
            extreme_returns = abs(returns) > 0.20  # 20% daily moves
            if extreme_returns.sum() > len(returns) * 0.02:  # More than 2% extreme moves
                warnings.append(f"High volatility detected: {extreme_returns.sum()} extreme daily moves")
                quality_score -= 5
        
        # Data freshness check
        if hasattr(data.index, 'max'):
            latest_date = data.index.max()
            if isinstance(latest_date, pd.Timestamp):
                days_old = (datetime.now() - latest_date.to_pydatetime()).days
                if days_old > 7:
                    warnings.append(f"Data may be stale: {days_old} days old")
                    quality_score -= 5
        
        # Final quality score adjustment
        quality_score = max(0.0, min(100.0, quality_score))
        
        is_valid = len(issues) == 0 and quality_score >= 50.0
        
        return ValidationResult(is_valid, issues, warnings, quality_score)

# ENHANCED: Error recovery system
class ErrorRecoverySystem:
    """Advanced error recovery and fallback mechanisms"""
    
    def __init__(self, logger: AdvancedLogger):
        self.logger = logger
        self.fallback_sources = ['yfinance']  # Could add more sources
        self.max_retries = 3
        self.retry_delay = 2.0
    
    def fetch_with_retry(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch data with intelligent retry and fallback"""
        for attempt in range(self.max_retries):
            try:
                self.logger.logger.info(f"Fetching {symbol} data, attempt {attempt + 1}")
                
                # Primary fetch method
                data = self._fetch_yfinance_data(symbol, period)
                
                if data is not None and len(data) > 0:
                    # Validate the fetched data
                    validation = AdvancedDataValidator.validate_market_data(data, symbol)
                    
                    if validation.is_valid:
                        self.logger.logger.info(f"Successfully fetched {symbol} data with quality score: {validation.data_quality_score:.1f}")
                        return data
                    else:
                        self.logger.logger.warning(f"Data quality issues for {symbol}: {validation.issues}")
                        if validation.data_quality_score >= 70.0:  # Accept if reasonably good
                            self.logger.logger.info(f"Accepting data with quality score: {validation.data_quality_score:.1f}")
                            return data
                
                # If we get here, the attempt failed
                self.logger.logger.warning(f"Attempt {attempt + 1} failed for {symbol}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                
            except Exception as e:
                self.logger.logger.error(f"Error in attempt {attempt + 1} for {symbol}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        
        self.logger.logger.error(f"All attempts failed for {symbol}")
        return None
    
    def _fetch_yfinance_data(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """Enhanced yfinance data fetching"""
        try:
            ticker = yf.Ticker(symbol)
            raw_data = ticker.history(period=period)
            
            if len(raw_data) == 0:
                raise ValueError(f"No data returned for {symbol}")
            
            # Clean and prepare data
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in raw_data.columns for col in required_columns):
                available = list(raw_data.columns)
                self.logger.logger.error(f"Missing columns for {symbol}. Available: {available}")
                return None
            
            clean_data = raw_data[required_columns].copy()
            clean_data = clean_data.dropna()
            
            # Add calculated fields
            clean_data['Typical_Price'] = (clean_data['High'] + clean_data['Low'] + clean_data['Close']) / 3
            
            return clean_data
            
        except Exception as e:
            self.logger.logger.error(f"yfinance fetch error for {symbol}: {str(e)}")
            return None

# Initialize enhanced systems
@st.cache_resource
def initialize_enhanced_systems():
    """Initialize all enhanced systems"""
    logger = AdvancedLogger()
    data_manager = OptimizedDataManager(logger)
    error_recovery = ErrorRecoverySystem(logger)
    return logger, data_manager, error_recovery

def safe_rsi(prices, period=14):
    """Safe RSI calculation with proper error handling"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)
        return rsi
    except Exception as e:
        return pd.Series([50] * len(prices), index=prices.index)

def calculate_daily_vwap(data):
    """Optimized daily VWAP calculation"""
    try:
        if not hasattr(data, 'index') or not hasattr(data, 'columns'):
            return float(data['Close'].iloc[-1]) if 'Close' in data else 0.0
        
        if len(data) < 5:
            return float(data['Close'].iloc[-1])
        
        recent_data = data.tail(20)
        
        if 'Typical_Price' in recent_data.columns and 'Volume' in recent_data.columns:
            total_pv = (recent_data['Typical_Price'] * recent_data['Volume']).sum()
            total_volume = recent_data['Volume'].sum()
            
            if total_volume > 0:
                return float(total_pv / total_volume)
        
        return float(data['Close'].iloc[-1])
    except Exception:
        try:
            return float(data['Close'].iloc[-1])
        except:
            return 0.0

def statistical_normalize(series, lookback_period=252):
    """Optimized statistical normalization"""
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

# ENHANCED: VWV Trading System with performance monitoring
class VWVTradingSystemEnhanced:
    def __init__(self, config=None, logger=None):
        """Initialize enhanced trading system"""
        default_config = {
            'wvf_period': 22,
            'wvf_multiplier': 1.2,
            'ma_periods': [20, 50, 200],
            'volume_periods': [20, 50],
            'rsi_period': 14,
            'volatility_period': 20,
            'weights': {
                'wvf': 0.8, 'ma': 1.2, 'volume': 0.6, 
                'vwap': 0.4, 'momentum': 0.5, 'volatility': 0.3
            },
            'scaling_multiplier': 1.5,
            'signal_thresholds': {'good': 3.5, 'strong': 4.5, 'very_strong': 5.5},
            'stop_loss_pct': 0.022,
            'take_profit_pct': 0.055
        }
        
        self.config = {**default_config, **(config or {})}
        self.weights = self.config['weights']
        self.scaling_multiplier = self.config['scaling_multiplier']
        self.signal_thresholds = self.config['signal_thresholds']
        self.stop_loss_pct = self.config['stop_loss_pct']
        self.take_profit_pct = self.config['take_profit_pct']
        self.logger = logger
    
    def _log_performance(self, operation: str, start_time: float, data_size: int):
        """Helper to log performance metrics"""
        if self.logger:
            duration = time.time() - start_time
            metrics = PerformanceMetrics(
                operation=operation,
                start_time=start_time,
                end_time=time.time(),
                duration=duration,
                data_size=data_size
            )
            self.logger.log_performance(metrics)
    
    def calculate_williams_vix_fix(self, data):
        """Optimized Williams VIX Fix calculation"""
        start_time = time.time()
        try:
            period = self.config['wvf_period']
            multiplier = self.config['wvf_multiplier']
            
            if len(data) < period:
                return 0.0
                
            low, close = data['Low'], data['Close']
            highest_close = close.rolling(window=period).max()
            wvf = ((highest_close - low) / highest_close) * 100 * multiplier
            
            result = statistical_normalize(wvf)
            self._log_performance("williams_vix_fix", start_time, len(data))
            return result
        except Exception:
            return 0.0
    
    def calculate_ma_confluence(self, data):
        """Optimized moving average confluence"""
        start_time = time.time()
        try:
            ma_periods = self.config['ma_periods']
            if len(data) < max(ma_periods):
                return 0.0
                
            close = data['Close']
            current_price = close.iloc[-1]
            
            # Vectorized MA calculation
            mas = []
            for period in ma_periods:
                if len(close) >= period:
                    ma = close.rolling(window=period).mean().iloc[-1]
                    mas.append(ma)
            
            if not mas:
                return 0.0
                
            ma_avg = np.mean(mas)
            deviation_pct = (ma_avg - current_price) / ma_avg * 100 if ma_avg > 0 else 0
            
            # Use last 252 periods for normalization
            lookback_data = close.tail(252)
            deviation_series = pd.Series([(ma_avg - p) / ma_avg * 100 for p in lookback_data])
            
            result = statistical_normalize(deviation_series.abs())
            self._log_performance("ma_confluence", start_time, len(data))
            return result
        except Exception:
            return 0.0
    
    def calculate_volume_confluence(self, data):
        """Optimized volume analysis"""
        start_time = time.time()
        try:
            periods = self.config['volume_periods']
            if len(data) < max(periods):
                return 0.0
                
            volume = data['Volume']
            current_vol = volume.iloc[-1]
            
            # Vectorized volume MA calculation
            vol_mas = []
            for period in periods:
                if len(volume) >= period:
                    vol_ma = volume.rolling(window=period).mean().iloc[-1]
                    vol_mas.append(vol_ma)
            
            if not vol_mas:
                return 0.0
                
            avg_vol = np.mean(vol_mas)
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
            
            # Use recent data for normalization
            vol_ratios = volume.tail(252) / volume.rolling(window=periods[0]).mean()
            
            result = statistical_normalize(vol_ratios)
            self._log_performance("volume_confluence", start_time, len(data))
            return result
        except Exception:
            return 0.0
    
    def calculate_vwap_analysis(self, data):
        """Optimized VWAP analysis"""
        start_time = time.time()
        try:
            if len(data) < 20:
                return 0.0
                
            current_price = data['Close'].iloc[-1]
            current_vwap = calculate_daily_vwap(data)
            
            vwap_deviation_pct = abs(current_price - current_vwap) / current_vwap * 100 if current_vwap > 0 else 0
            
            # Optimized historical VWAP calculation
            vwap_deviations = []
            max_lookback = min(252, len(data) - 20)
            
            for i in range(0, max_lookback, 5):  # Sample every 5th period for efficiency
                subset = data.iloc[i:i+20] if i+20 < len(data) else data.iloc[i:]
                if len(subset) >= 5:
                    daily_vwap = calculate_daily_vwap(subset)
                    price = subset['Close'].iloc[-1]
                    deviation = abs(price - daily_vwap) / daily_vwap * 100 if daily_vwap > 0 else 0
                    vwap_deviations.append(deviation)
            
            if vwap_deviations:
                deviation_series = pd.Series(vwap_deviations + [vwap_deviation_pct])
                result = statistical_normalize(deviation_series)
            else:
                result = 0.0
            
            self._log_performance("vwap_analysis", start_time, len(data))
            return result
        except Exception:
            return 0.0
    
    def calculate_momentum(self, data):
        """Optimized momentum calculation"""
        start_time = time.time()
        try:
            period = self.config['rsi_period']
            if len(data) < period + 1:
                return 0.0
                
            close = data['Close']
            rsi = safe_rsi(close, period)
            rsi_value = rsi.iloc[-1]
            
            oversold_signal = (50 - rsi_value) / 50 if rsi_value < 50 else 0
            result = float(np.clip(oversold_signal, 0, 1))
            
            self._log_performance("momentum", start_time, len(data))
            return result
        except Exception:
            return 0.0
    
    def calculate_volatility_filter(self, data):
        """Optimized volatility filter"""
        start_time = time.time()
        try:
            period = self.config['volatility_period']
            if len(data) < period:
                return 0.0
                
            close = data['Close']
            returns = close.pct_change().dropna()
            volatility = returns.rolling(window=period).std() * np.sqrt(252)
            
            result = statistical_normalize(volatility)
            self._log_performance("volatility_filter", start_time, len(data))
            return result
        except Exception:
            return 0.0
    
    def calculate_trend_analysis(self, data):
        """Enhanced trend analysis with performance tracking"""
        start_time = time.time()
        try:
            if len(data) < 100:
                return None
            
            close_prices = data['Close']
            ema_21 = close_prices.ewm(span=21).mean()
            ema_50 = close_prices.ewm(span=50).mean()
            current_price = close_prices.iloc[-1]
            
            price_vs_ema21 = (current_price - ema_21.iloc[-1]) / ema_21.iloc[-1] * 100
            ema21_slope = (ema_21.iloc[-1] - ema_21.iloc[-5]) / ema_21.iloc[-5] * 100
            ema_alignment = 1 if ema_21.iloc[-1] > ema_50.iloc[-1] else -1
            
            if price_vs_ema21 > 2 and ema21_slope > 0 and ema_alignment > 0:
                trend_direction = 'BULLISH'
                trend_strength = abs(price_vs_ema21)
                trend_bias = 1
            elif price_vs_ema21 < -2 and ema21_slope < 0 and ema_alignment < 0:
                trend_direction = 'BEARISH'
                trend_strength = abs(price_vs_ema21)
                trend_bias = -1
            else:
                trend_direction = 'SIDEWAYS'
                trend_strength = 0
                trend_bias = 0
            
            result = {
                'trend_direction': trend_direction,
                'trend_strength': round(trend_strength, 2),
                'trend_bias': trend_bias,
                'price_vs_ema21': round(price_vs_ema21, 2),
                'ema21_slope': round(ema21_slope, 2)
            }
            
            self._log_performance("trend_analysis", start_time, len(data))
            return result
        except Exception:
            return None
    
    def calculate_real_confidence_intervals(self, data):
        """Optimized confidence intervals calculation"""
        start_time = time.time()
        try:
            if not isinstance(data, pd.DataFrame) or len(data) < 100:
                return None
            
            # Use last 2 years of data for efficiency
            analysis_data = data.tail(504) if len(data) > 504 else data
            
            weekly_data = analysis_data.resample('W-FRI')['Close'].last().dropna()
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
            
            result = {
                'mean_weekly_return': round(mean_return * 100, 3),
                'weekly_volatility': round(std_return * 100, 2),
                'confidence_intervals': confidence_intervals,
                'sample_size': len(weekly_returns)
            }
            
            self._log_performance("confidence_intervals", start_time, len(data))
            return result
        except Exception:
            return None
    
    def calculate_confluence_enhanced(self, input_data, symbol='SPY'):
        """Enhanced confluence calculation with comprehensive monitoring"""
        overall_start_time = time.time()
        
        try:
            if not isinstance(input_data, pd.DataFrame):
                raise ValueError(f"Expected DataFrame, got {type(input_data)}")
            
            # Work on isolated copy
            working_data = input_data.copy(deep=True)
            
            # Calculate all components with performance tracking
            components = {
                'wvf': self.calculate_williams_vix_fix(working_data),
                'ma': self.calculate_ma_confluence(working_data),
                'volume': self.calculate_volume_confluence(working_data),
                'vwap': self.calculate_vwap_analysis(working_data),
                'momentum': self.calculate_momentum(working_data),
                'volatility': self.calculate_volatility_filter(working_data)
            }
            
            # Calculate confluence metrics
            raw_confluence = sum(components[comp] * self.weights[comp] for comp in components)
            base_confluence = raw_confluence * self.scaling_multiplier
            
            # Trend analysis
            trend_analysis = self.calculate_trend_analysis(working_data)
            trend_bias = trend_analysis['trend_bias'] if trend_analysis else 0
            
            # Apply directional bias
            directional_confluence = base_confluence * (1 + trend_bias * 0.2)
            
            # Signal determination
            abs_confluence = abs(directional_confluence)
            signal_direction = 'LONG' if directional_confluence >= 0 else 'SHORT'
            
            signal_type = 'NONE'
            signal_strength = 0
            if abs_confluence >= self.signal_thresholds['very_strong']:
                signal_type, signal_strength = 'VERY_STRONG', 3
            elif abs_confluence >= self.signal_thresholds['strong']:
                signal_type, signal_strength = 'STRONG', 2
            elif abs_confluence >= self.signal_thresholds['good']:
                signal_type, signal_strength = 'GOOD', 1
            
            if signal_type != 'NONE':
                signal_type = f"{signal_type}_{signal_direction}"
            
            current_price = round(float(working_data['Close'].iloc[-1]), 2)
            current_date = working_data.index[-1].strftime('%Y-%m-%d')
            
            # Confidence intervals
            confidence_analysis = self.calculate_real_confidence_intervals(working_data)
            
            # Entry information
            entry_info = {}
            if signal_type != 'NONE':
                if signal_direction == 'LONG':
                    stop_price = round(current_price * (1 - self.stop_loss_pct), 2)
                    target_price = round(current_price * (1 + self.take_profit_pct), 2)
                else:
                    stop_price = round(current_price * (1 + self.stop_loss_pct), 2)
                    target_price = round(current_price * (1 - self.take_profit_pct), 2)
                
                entry_info = {
                    'entry_price': current_price,
                    'stop_loss': stop_price,
                    'take_profit': target_price,
                    'direction': signal_direction,
                    'position_multiplier': signal_strength,
                    'risk_reward': round(self.take_profit_pct / self.stop_loss_pct, 2)
                }
            
            # Log overall performance
            self._log_performance("full_confluence_analysis", overall_start_time, len(working_data))
            
            return {
                'symbol': symbol,
                'timestamp': current_date,
                'current_price': current_price,
                'components': {k: round(v, 3) for k, v in components.items()},
                'raw_confluence': round(raw_confluence, 3),
                'base_confluence': round(base_confluence, 3),
                'directional_confluence': round(directional_confluence, 3),
                'signal_type': signal_type,
                'signal_direction': signal_direction,
                'signal_strength': signal_strength,
                'trend_analysis': trend_analysis,
                'confidence_analysis': confidence_analysis,
                'entry_info': entry_info,
                'system_status': 'OPERATIONAL'
            }
            
        except Exception as e:
            if self.logger:
                self.logger.logger.error(f"Confluence calculation error for {symbol}: {str(e)}")
            
            return {
                'symbol': symbol,
                'error': str(e),
                'system_status': 'ERROR'
            }

# ENHANCED: Chart creation with performance optimization
def create_enhanced_chart_optimized(chart_market_data, analysis_results, symbol, logger=None):
    """Optimized chart creation with performance monitoring"""
    start_time = time.time()
    
    try:
        if logger:
            logger.logger.info(f"Creating enhanced chart for {symbol}")
        
        # Data validation
        if isinstance(chart_market_data, dict):
            if logger:
                logger.logger.error(f"Chart received dict for {symbol}: {list(chart_market_data.keys())}")
            st.error(f"❌ CHART RECEIVED DICT INSTEAD OF DATAFRAME!")
            return None
        
        if not isinstance(chart_market_data, pd.DataFrame):
            if logger:
                logger.logger.error(f"Invalid chart data type for {symbol}: {type(chart_market_data)}")
            st.error(f"❌ Invalid chart data type: {type(chart_market_data)}")
            return None
        
        if len(chart_market_data) == 0:
            st.error(f"❌ Chart DataFrame is empty")
            return None
        
        # Validate required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in chart_market_data.columns]
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            return None
        
        # Optimize chart data size
        chart_data = chart_market_data.tail(100)  # Last 100 periods for performance
        current_price = analysis_results['current_price']
        
        # Create optimized subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f'{symbol} Price Chart with Statistical Levels', 
                'Volume', 
                'Confidence Intervals'
            ),
            vertical_spacing=0.08, 
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=chart_data.index, 
            open=chart_data['Open'], 
            high=chart_data['High'],
            low=chart_data['Low'], 
            close=chart_data['Close'], 
            name='Price'
        ), row=1, col=1)
        
        # Moving averages (optimized)
        if len(chart_data) >= 21:
            ema21 = chart_data['Close'].ewm(span=21).mean()
            fig.add_trace(go.Scatter(
                x=chart_data.index, y=ema21, name='EMA21', 
                line=dict(color='orange', width=2)
            ), row=1, col=1)
        
        # Confidence interval levels
        if analysis_results.get('confidence_analysis'):
            conf_data = analysis_results['confidence_analysis']['confidence_intervals']
            
            # 68% confidence
            if '68%' in conf_data:
                fig.add_hline(y=conf_data['68%']['upper_bound'], line_dash="dash", 
                             line_color="green", line_width=2, row=1, col=1)
                fig.add_hline(y=conf_data['68%']['lower_bound'], line_dash="dash", 
                             line_color="green", line_width=2, row=1, col=1)
            
            # 95% confidence
            if '95%' in conf_data:
                fig.add_hline(y=conf_data['95%']['upper_bound'], line_dash="dot", 
                             line_color="red", line_width=1, row=1, col=1)
                fig.add_hline(y=conf_data['95%']['lower_bound'], line_dash="dot", 
                             line_color="red", line_width=1, row=1, col=1)
        
        # Current price line
        fig.add_hline(y=current_price, line_dash="solid", line_color="black", 
                     line_width=3, row=1, col=1)
        
        # Volume chart (optimized colors)
        volume_colors = ['green' if close >= open else 'red' 
                        for close, open in zip(chart_data['Close'], chart_data['Open'])]
        fig.add_trace(go.Bar(
            x=chart_data.index, y=chart_data['Volume'], 
            name='Volume', marker_color=volume_colors
        ), row=2, col=1)
        
        # Confidence intervals bar chart
        if analysis_results.get('confidence_analysis'):
            conf_data = analysis_results['confidence_analysis']['confidence_intervals']
            x_labels = list(conf_data.keys())
            y_values = [conf_data[key]['expected_move_pct'] for key in x_labels]
            
            fig.add_trace(go.Bar(
                x=x_labels, y=y_values, name='Expected Weekly Move %', 
                marker_color='lightblue',
                text=[f"{v:.1f}%" for v in y_values], textposition='outside'
            ), row=3, col=1)
        
        # Layout optimization
        fig.update_layout(
            title=f'{symbol} | Confluence: {analysis_results["directional_confluence"]:.2f} | Signal: {analysis_results["signal_type"]}',
            height=800, 
            showlegend=False, 
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Expected Move %", row=3, col=1)
        
        # Log performance
        if logger:
            duration = time.time() - start_time
            metrics = PerformanceMetrics(
                operation="create_chart",
                start_time=start_time,
                end_time=time.time(),
                duration=duration,
                data_size=len(chart_data)
            )
            logger.log_performance(metrics)
        
        return fig
        
    except Exception as e:
        if logger:
            logger.logger.error(f"Chart creation error for {symbol}: {str(e)}")
        st.error(f"❌ Error creating chart: {str(e)}")
        return None

# ENHANCED: Main application with comprehensive monitoring
def main():
    # Initialize enhanced systems
    logger, data_manager, error_recovery = initialize_enhanced_systems()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 VWV Professional Trading System - PRODUCTION ENHANCED</h1>
        <p>Advanced market analysis with performance optimization, monitoring & logging</p>
        <p><em>Features: Performance tracking, advanced validation, error recovery, comprehensive logging</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.title("📊 Enhanced Analysis Controls")
    
    # Basic controls
    symbol = st.sidebar.text_input("Symbol", value="SPY", help="Enter stock symbol").upper()
    period = st.sidebar.selectbox("Data Period", ['1mo', '3mo', '6mo', '1y', '2y'], index=3)
    
    # Enhanced system parameters
    with st.sidebar.expander("⚙️ System Parameters"):
        st.write("**Williams VIX Fix**")
        wvf_period = st.slider("WVF Period", 10, 50, 22)
        wvf_multiplier = st.slider("WVF Multiplier", 0.5, 2.0, 1.2, 0.1)
        
        st.write("**Signal Thresholds**")
        good_threshold = st.slider("Good Signal", 2.0, 5.0, 3.5, 0.1)
        strong_threshold = st.slider("Strong Signal", 3.0, 6.0, 4.5, 0.1)
        very_strong_threshold = st.slider("Very Strong Signal", 4.0, 7.0, 5.5, 0.1)
    
    # Performance monitoring controls
    with st.sidebar.expander("📈 Performance Monitoring"):
        show_performance = st.checkbox("Show Performance Metrics", value=True)
        show_logs = st.checkbox("Show System Logs", value=False)
        show_data_quality = st.checkbox("Show Data Quality Report", value=True)
    
    # System controls
    show_chart = st.sidebar.checkbox("Show Interactive Chart", value=True)
    analyze_button = st.sidebar.button("📊 Enhanced Analysis", type="primary", use_container_width=True)
    
    # Debug and maintenance tools
    with st.sidebar.expander("🔧 System Tools"):
        if st.button("Clear All Caches"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Caches cleared")
        
        if st.button("Reset Data Manager"):
            logger, data_manager, error_recovery = initialize_enhanced_systems()
            st.success("Systems reset")
        
        if st.button("Performance Summary"):
            perf_summary = logger.get_performance_summary()
            if perf_summary:
                st.json(perf_summary)
            else:
                st.info("No performance data available")
    
    # Create custom configuration
    custom_config = {
        'wvf_period': wvf_period,
        'wvf_multiplier': wvf_multiplier,
        'signal_thresholds': {
            'good': good_threshold,
            'strong': strong_threshold,
            'very_strong': very_strong_threshold
        }
    }
    
    # Initialize enhanced VWV system
    vwv_system = VWVTradingSystemEnhanced(custom_config, logger)
    
    if analyze_button and symbol:
        st.write("## 🔄 Enhanced Analysis Process")
        
        overall_analysis_start = time.time()
        
        with st.spinner(f"Running enhanced analysis for {symbol}..."):
            
            # STEP 1: Enhanced data fetching with retry
            st.write("### Step 1: Enhanced Data Fetching with Retry Logic")
            fetch_start_time = time.time()
            
            fresh_market_data = error_recovery.fetch_with_retry(symbol, period)
            
            if fresh_market_data is None:
                st.error(f"❌ Could not fetch data for {symbol} after {error_recovery.max_retries} attempts")
                return
            
            fetch_duration = time.time() - fetch_start_time
            st.success(f"✅ Data fetched in {fetch_duration:.2f}s")
            
            # STEP 2: Data validation
            st.write("### Step 2: Advanced Data Validation")
            validation_result = AdvancedDataValidator.validate_market_data(fresh_market_data, symbol)
            
            # Display validation results
            if validation_result.is_valid:
                st.success(f"✅ Data validation passed (Quality Score: {validation_result.data_quality_score:.1f}/100)")
            else:
                st.error(f"❌ Data validation failed (Quality Score: {validation_result.data_quality_score:.1f}/100)")
                for issue in validation_result.issues:
                    st.error(f"  • {issue}")
                return
            
            if validation_result.warnings and show_data_quality:
                st.warning("⚠️ Data Quality Warnings:")
                for warning in validation_result.warnings:
                    st.warning(f"  • {warning}")
            
            # Store validated data
            data_manager.store_market_data(symbol, fresh_market_data)
            
            # STEP 3: Enhanced analysis
            st.write("### Step 3: Enhanced Confluence Analysis")
            analysis_input = data_manager.get_market_data_for_analysis(symbol)
            
            if analysis_input is None:
                st.error("❌ Could not prepare analysis data")
                return
            
            analysis_results = vwv_system.calculate_confluence_enhanced(analysis_input, symbol)
            
            if 'error' in analysis_results:
                st.error(f"❌ Analysis failed: {analysis_results['error']}")
                return
            
            # Store results
            data_manager.store_analysis_results(symbol, analysis_results)
            
            # STEP 4: Results display with enhanced metrics
            st.write("### Step 4: Enhanced Results Display")
            
            # Primary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${analysis_results['current_price']}")
            with col2:
                confluence_value = analysis_results['directional_confluence']
                st.metric("Directional Confluence", f"{confluence_value:.2f}")
            with col3:
                signal_icons = {
                    "NONE": "⚪", "GOOD_LONG": "🟢⬆️", "GOOD_SHORT": "🟢⬇️",
                    "STRONG_LONG": "🟡⬆️", "STRONG_SHORT": "🟡⬇️",
                    "VERY_STRONG_LONG": "🔴⬆️", "VERY_STRONG_SHORT": "🔴⬇️"
                }
                signal_display = f"{signal_icons.get(analysis_results['signal_type'], '⚪')} {analysis_results['signal_type']}"
                st.metric("Signal", signal_display)
            with col4:
                trend_dir = analysis_results['trend_analysis']['trend_direction'] if analysis_results['trend_analysis'] else 'N/A'
                st.metric("Trend Direction", trend_dir)
            
            # Enhanced signal display
            if analysis_results['signal_type'] != 'NONE':
                entry_info = analysis_results['entry_info']
                direction = entry_info['direction']
                
                signal_class = "signal-good"
                if "STRONG" in analysis_results['signal_type']:
                    signal_class = "signal-strong"
                elif "VERY_STRONG" in analysis_results['signal_type']:
                    signal_class = "signal-very-strong"
                
                st.markdown(f"""
                <div class="{signal_class}" style="padding: 1rem; margin: 1rem 0; border-radius: 8px;">
                    <h3>🚨 VWV {direction} SIGNAL DETECTED</h3>
                    <p><strong>Signal:</strong> {analysis_results['signal_type']}</p>
                    <p><strong>Direction:</strong> {direction}</p>
                    <p><strong>Entry:</strong> ${entry_info['entry_price']}</p>
                    <p><strong>Stop Loss:</strong> ${entry_info['stop_loss']}</p>
                    <p><strong>Take Profit:</strong> ${entry_info['take_profit']}</p>
                    <p><strong>Risk/Reward:</strong> {entry_info['risk_reward']}:1</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Performance metrics display
            if show_performance:
                st.write("### 📈 Performance Metrics")
                
                total_duration = time.time() - overall_analysis_start
                perf_summary = logger.get_performance_summary()
                
                if perf_summary:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Analysis Time", f"{total_duration:.2f}s")
                    with col2:
                        st.metric("Operations Count", perf_summary['total_operations'])
                    with col3:
                        st.metric("Average Operation Time", f"{perf_summary['avg_duration']:.3f}s")
                    with col4:
                        st.metric("Max Operation Time", f"{perf_summary['max_duration']:.3f}s")
                    
                    # Performance breakdown
                    if perf_summary['operations_breakdown']:
                        st.write("**Operation Performance Breakdown:**")
                        breakdown_data = []
                        for op, stats in perf_summary['operations_breakdown'].items():
                            breakdown_data.append({
                                'Operation': op,
                                'Count': stats['count'],
                                'Avg Time (s)': f"{stats['avg_time']:.3f}",
                                'Max Time (s)': f"{stats['max_time']:.3f}"
                            })
                        
                        df_performance = pd.DataFrame(breakdown_data)
                        st.dataframe(df_performance, use_container_width=True)
                
                # Performance classification
                if total_duration < 2.0:
                    perf_class = "performance-excellent"
                    perf_message = "🚀 Excellent Performance"
                elif total_duration < 5.0:
                    perf_class = "performance-good"
                    perf_message = "✅ Good Performance"
                else:
                    perf_class = "performance-poor"
                    perf_message = "⚠️ Performance Could Be Improved"
                
                st.markdown(f"""
                <div class="{perf_class}" style="padding: 1rem; margin: 1rem 0; border-radius: 8px;">
                    <h4>{perf_message}</h4>
                    <p>Total analysis completed in {total_duration:.2f} seconds</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence intervals (existing code)
            if analysis_results.get('confidence_analysis'):
                st.subheader("📊 Statistical Confidence Intervals")
                conf_data = analysis_results['confidence_analysis']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Weekly Return", f"{conf_data['mean_weekly_return']:.3f}%")
                with col2:
                    st.metric("Weekly Volatility", f"{conf_data['weekly_volatility']:.2f}%")
                with col3:
                    st.metric("Sample Size", f"{conf_data['sample_size']} weeks")
                
                intervals_data = []
                for level, data in conf_data['confidence_intervals'].items():
                    intervals_data.append({
                        'Confidence Level': level,
                        'Upper Bound': f"${data['upper_bound']}",
                        'Lower Bound': f"${data['lower_bound']}",
                        'Expected Move': f"±{data['expected_move_pct']:.2f}%"
                    })
                
                df_intervals = pd.DataFrame(intervals_data)
                st.dataframe(df_intervals, use_container_width=True)
            
            # Components analysis (existing code)
            st.subheader("🔧 VWV Components Analysis")
            comp_data = []
            for comp, value in analysis_results['components'].items():
                weight = vwv_system.weights[comp]
                contribution = round(value * weight, 3)
                comp_data.append({
                    'Component': comp.upper(),
                    'Normalized Value': f"{value:.3f}",
                    'Weight': f"{weight}",
                    'Contribution': f"{contribution:.3f}"
                })
            
            df_components = pd.DataFrame(comp_data)
            st.dataframe(df_components, use_container_width=True)
            
            # Enhanced chart creation
            if show_chart:
                st.write("### Step 5: Enhanced Chart Creation")
                
                chart_market_data = data_manager.get_market_data_for_chart(symbol)
                
                if chart_market_data is None:
                    st.error("❌ Could not get chart data")
                    return
                
                chart = create_enhanced_chart_optimized(chart_market_data, analysis_results, symbol, logger)
                
                if chart is not None:
                    st.plotly_chart(chart, use_container_width=True)
                    st.success("✅ Enhanced chart created successfully")
                else:
                    st.error("❌ Chart creation failed")
            
            # System logs display
            if show_logs:
                st.write("### 📋 System Logs")
                
                if logger.analysis_history:
                    st.write("**Recent Analysis History:**")
                    for entry in logger.analysis_history[-5:]:  # Last 5 analyses
                        st.markdown(f"""
                        <div class="log-entry">
                            {entry['timestamp']} | {entry['symbol']} | {entry['signal_type']} | Confluence: {entry['confluence']:.2f}
                        </div>
                        """, unsafe_allow_html=True)
                
                if logger.performance_log:
                    st.write("**Recent Performance Log:**")
                    for metric in logger.performance_log[-10:]:  # Last 10 operations
                        st.markdown(f"""
                        <div class="log-entry">
                            {metric.operation}: {metric.duration:.3f}s ({metric.data_size} records)
                        </div>
                        """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        ## 🛠️ VWV System - PRODUCTION ENHANCED FEATURES
        
        ### ✅ **Enhanced Features Implemented:**
        
        1. **🚀 Performance Optimization**
           - Optimized data copying strategies
           - Intelligent caching with TTL
           - Vectorized calculations
           - Memory-efficient operations
        
        2. **🔍 Advanced Validation**
           - Comprehensive data quality scoring
           - OHLC relationship validation
           - Outlier detection
           - Data freshness checking
        
        3. **🛡️ Error Recovery**
           - Intelligent retry with exponential backoff
           - Multiple data source fallbacks
           - Graceful degradation
           - Emergency recovery systems
        
        4. **📊 Comprehensive Logging**
           - Performance metrics tracking
           - Analysis history logging
           - File-based logging system
           - Real-time monitoring
        
        5. **⚡ Production Features**
           - Cache optimization
           - Performance classification
           - System health monitoring
           - Advanced debugging tools
        
        ### 📈 **Performance Monitoring:**
        - Real-time operation timing
        - Performance breakdown by component
        - Quality score reporting
        - System efficiency tracking
        
        ### 🔧 **System Tools:**
        - Cache management
        - Performance analytics
        - System reset capabilities
        - Debugging utilities
        
        **Ready for production trading? Enter a symbol and click "Enhanced Analysis"!**
        """)

if __name__ == "__main__":
    main()
