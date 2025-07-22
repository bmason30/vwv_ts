import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import copy

# Suppress only specific warnings, not all
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

# Page configuration
st.set_page_config(
    page_title="VWV Professional Trading System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
</style>
""", unsafe_allow_html=True)

# ENHANCED: Data manager with debug control
class DataManager:
    """Enhanced data manager with debug control"""
    
    def __init__(self):
        self._market_data_store = {}
        self._analysis_store = {}
    
    def store_market_data(self, symbol, market_data, show_debug=False):
        """Data storage with debug control"""
        if not isinstance(market_data, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(market_data)}")
        
        self._market_data_store[symbol] = market_data.copy(deep=True)
        if show_debug:
            st.write(f"🔒 Stored market data for {symbol}: {market_data.shape}")
    
    def get_market_data_for_analysis(self, symbol):
        """Get copy for analysis"""
        if symbol not in self._market_data_store:
            return None
        
        return self._market_data_store[symbol].copy(deep=True)
    
    def get_market_data_for_chart(self, symbol):
        """Get copy for chart"""
        if symbol not in self._market_data_store:
            return None
        
        chart_copy = self._market_data_store[symbol].copy(deep=True)
        
        if not isinstance(chart_copy, pd.DataFrame):
            st.error(f"🚨 Chart data corrupted: {type(chart_copy)}")
            return None
        
        return chart_copy
    
    def store_analysis_results(self, symbol, analysis_results):
        """Store analysis results"""
        self._analysis_store[symbol] = copy.deepcopy(analysis_results)
    
    def get_analysis_results(self, symbol):
        """Get analysis results"""
        return self._analysis_store.get(symbol, {})

# Initialize data manager
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()

# ENHANCED: Data fetching with debug control
def get_market_data_enhanced(symbol='SPY', period='1y', show_debug=False):
    """Enhanced market data fetching with debug control"""
    try:
        if show_debug:
            st.write(f"📡 Fetching data for {symbol}...")
        
        ticker = yf.Ticker(symbol)
        raw_data = ticker.history(period=period)
        
        if raw_data is None or len(raw_data) == 0:
            st.error(f"❌ No data returned for {symbol}")
            return None
        
        if show_debug:
            st.write(f"📊 Retrieved {len(raw_data)} rows")
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = list(raw_data.columns)
        
        if show_debug:
            st.write(f"📋 Available columns: {available_columns}")
        
        missing_columns = [col for col in required_columns if col not in available_columns]
        if missing_columns:
            st.error(f"❌ Missing columns: {missing_columns}")
            return None
        
        # Clean data
        clean_data = raw_data[required_columns].copy()
        clean_data = clean_data.dropna()
        
        if len(clean_data) == 0:
            st.error(f"❌ No data after cleaning")
            return None
        
        # Add typical price
        clean_data['Typical_Price'] = (clean_data['High'] + clean_data['Low'] + clean_data['Close']) / 3
        
        if show_debug:
            st.success(f"✅ Data ready: {clean_data.shape}")
        else:
            st.success(f"✅ Data loaded: {len(clean_data)} periods")
        
        return clean_data
        
    except Exception as e:
        st.error(f"❌ Error fetching {symbol}: {str(e)}")
        return None

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
    """Enhanced daily VWAP calculation"""
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

def calculate_fibonacci_emas(data):
    """Calculate Fibonacci EMAs (21, 55, 89, 144, 233)"""
    try:
        if len(data) < 21:
            return {}
        
        close = data['Close']
        fib_periods = [21, 55, 89, 144, 233]
        emas = {}
        
        for period in fib_periods:
            if len(close) >= period:
                ema_value = close.ewm(span=period).mean().iloc[-1]
                emas[f'EMA_{period}'] = round(float(ema_value), 2)
        
        return emas
    except Exception:
        return {}

def calculate_point_of_control(data):
    """Calculate daily Point of Control (POC) - price level with highest volume"""
    try:
        if len(data) < 20:
            return None
        
        # Use recent data for daily POC
        recent_data = data.tail(20)
        
        # Create price bins and sum volume for each bin
        price_range = recent_data['High'].max() - recent_data['Low'].min()
        bin_size = price_range / 50  # 50 price bins
        
        if bin_size <= 0:
            return float(recent_data['Close'].iloc[-1])
        
        # Calculate volume profile
        volume_profile = {}
        
        for idx, row in recent_data.iterrows():
            # Distribute volume across the OHLC range
            price_levels = [row['Open'], row['High'], row['Low'], row['Close']]
            volume_per_level = row['Volume'] / len(price_levels)
            
            for price in price_levels:
                bin_key = round(price / bin_size) * bin_size
                volume_profile[bin_key] = volume_profile.get(bin_key, 0) + volume_per_level
        
        # Find POC (price with highest volume)
        if volume_profile:
            poc_price = max(volume_profile, key=volume_profile.get)
            return round(float(poc_price), 2)
        else:
            return float(recent_data['Close'].iloc[-1])
            
    except Exception:
        try:
            return float(data['Close'].iloc[-1])
        except:
            return 0.0

def calculate_weekly_deviations(data):
    """Calculate weekly 1, 2, 3 standard deviation levels"""
    try:
        if len(data) < 50:
            return {}
        
        # Resample to weekly data
        weekly_data = data.resample('W-FRI').agg({
            'Open': 'first',
            'High': 'max', 
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        if len(weekly_data) < 10:
            return {}
        
        # Calculate weekly statistics
        weekly_closes = weekly_data['Close']
        current_price = data['Close'].iloc[-1]
        
        # Use last 20 weeks for calculation
        recent_weekly = weekly_closes.tail(20)
        mean_price = recent_weekly.mean()
        std_price = recent_weekly.std()
        
        if pd.isna(std_price) or std_price == 0:
            return {}
        
        deviations = {}
        for std_level in [1, 2, 3]:
            upper = mean_price + (std_level * std_price)
            lower = mean_price - (std_level * std_price)
            
            deviations[f'{std_level}_std'] = {
                'upper': round(float(upper), 2),
                'lower': round(float(lower), 2),
                'range_pct': round(float((std_level * std_price / mean_price) * 100), 2)
            }
        
        deviations['mean_price'] = round(float(mean_price), 2)
        deviations['std_price'] = round(float(std_price), 2)
        
        return deviations
        
    except Exception:
        return {}

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

# ENHANCED: VWV Trading System with enhanced indicators
class VWVTradingSystem:
    def __init__(self, config=None):
        """Initialize simple trading system"""
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
    
    def calculate_williams_vix_fix(self, data):
        """Williams VIX Fix calculation"""
        try:
            period = self.config['wvf_period']
            multiplier = self.config['wvf_multiplier']
            
            if len(data) < period:
                return 0.0
                
            low, close = data['Low'], data['Close']
            highest_close = close.rolling(window=period).max()
            wvf = ((highest_close - low) / highest_close) * 100 * multiplier
            
            return statistical_normalize(wvf)
        except Exception:
            return 0.0
    
    def calculate_ma_confluence(self, data):
        """Moving average confluence"""
        try:
            ma_periods = self.config['ma_periods']
            if len(data) < max(ma_periods):
                return 0.0
                
            close = data['Close']
            current_price = close.iloc[-1]
            
            mas = []
            for period in ma_periods:
                if len(close) >= period:
                    ma = close.rolling(window=period).mean().iloc[-1]
                    mas.append(ma)
            
            if not mas:
                return 0.0
                
            ma_avg = np.mean(mas)
            deviation_pct = (ma_avg - current_price) / ma_avg * 100 if ma_avg > 0 else 0
            
            deviation_series = pd.Series([(ma_avg - p) / ma_avg * 100 for p in close.tail(252)])
            return statistical_normalize(deviation_series.abs())
        except Exception:
            return 0.0
    
    def calculate_volume_confluence(self, data):
        """Volume analysis"""
        try:
            periods = self.config['volume_periods']
            if len(data) < max(periods):
                return 0.0
                
            volume = data['Volume']
            current_vol = volume.iloc[-1]
            
            vol_mas = []
            for period in periods:
                if len(volume) >= period:
                    vol_ma = volume.rolling(window=period).mean().iloc[-1]
                    vol_mas.append(vol_ma)
            
            if not vol_mas:
                return 0.0
                
            avg_vol = np.mean(vol_mas)
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
            
            vol_ratios = volume.tail(252) / volume.rolling(window=periods[0]).mean()
            return statistical_normalize(vol_ratios)
        except Exception:
            return 0.0
    
    def calculate_vwap_analysis(self, data):
        """VWAP analysis"""
        try:
            if len(data) < 20:
                return 0.0
                
            current_price = data['Close'].iloc[-1]
            current_vwap = calculate_daily_vwap(data)
            
            vwap_deviation_pct = abs(current_price - current_vwap) / current_vwap * 100 if current_vwap > 0 else 0
            
            vwap_deviations = []
            for i in range(min(252, len(data) - 20)):
                subset = data.iloc[i:i+20] if i+20 < len(data) else data.iloc[i:]
                if len(subset) >= 5:
                    daily_vwap = calculate_daily_vwap(subset)
                    price = subset['Close'].iloc[-1]
                    deviation = abs(price - daily_vwap) / daily_vwap * 100 if daily_vwap > 0 else 0
                    vwap_deviations.append(deviation)
            
            if vwap_deviations:
                deviation_series = pd.Series(vwap_deviations + [vwap_deviation_pct])
                return statistical_normalize(deviation_series)
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def calculate_momentum(self, data):
        """Momentum calculation"""
        try:
            period = self.config['rsi_period']
            if len(data) < period + 1:
                return 0.0
                
            close = data['Close']
            rsi = safe_rsi(close, period)
            rsi_value = rsi.iloc[-1]
            
            oversold_signal = (50 - rsi_value) / 50 if rsi_value < 50 else 0
            return float(np.clip(oversold_signal, 0, 1))
        except Exception:
            return 0.0
    
    def calculate_volatility_filter(self, data):
        """Volatility filter"""
        try:
            period = self.config['volatility_period']
            if len(data) < period:
                return 0.0
                
            close = data['Close']
            returns = close.pct_change().dropna()
            volatility = returns.rolling(window=period).std() * np.sqrt(252)
            
            return statistical_normalize(volatility)
        except Exception:
            return 0.0
    
    def calculate_trend_analysis(self, data):
        """Trend analysis"""
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
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': round(trend_strength, 2),
                'trend_bias': trend_bias,
                'price_vs_ema21': round(price_vs_ema21, 2),
                'ema21_slope': round(ema21_slope, 2)
            }
        except Exception:
            return None
    
    def calculate_real_confidence_intervals(self, data):
        """Confidence intervals calculation"""
        try:
            if not isinstance(data, pd.DataFrame) or len(data) < 100:
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
        except Exception:
            return None
    
    def calculate_confluence(self, input_data, symbol='SPY'):
        """Enhanced confluence calculation with additional technical indicators"""
        try:
            if not isinstance(input_data, pd.DataFrame):
                raise ValueError(f"Expected DataFrame, got {type(input_data)}")
            
            # Work on isolated copy
            working_data = input_data.copy(deep=True)
            
            # Calculate enhanced technical indicators
            daily_vwap = calculate_daily_vwap(working_data)
            fibonacci_emas = calculate_fibonacci_emas(working_data)
            point_of_control = calculate_point_of_control(working_data)
            weekly_deviations = calculate_weekly_deviations(working_data)
            
            # Calculate original VWV components
            components = {
                'wvf': self.calculate_williams_vix_fix(working_data),
                'ma': self.calculate_ma_confluence(working_data),
                'volume': self.calculate_volume_confluence(working_data),
                'vwap': self.calculate_vwap_analysis(working_data),
                'momentum': self.calculate_momentum(working_data),
                'volatility': self.calculate_volatility_filter(working_data)
            }
            
            # Calculate confluence
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
                'enhanced_indicators': {
                    'daily_vwap': daily_vwap,
                    'fibonacci_emas': fibonacci_emas,
                    'point_of_control': point_of_control,
                    'weekly_deviations': weekly_deviations
                },
                'system_status': 'OPERATIONAL'
            }
            
        except Exception as e:
            return {
                'symbol': symbol,
                'error': str(e),
                'system_status': 'ERROR'
            }

# ENHANCED: Chart creation with Fibonacci EMAs and enhanced indicators
def create_enhanced_chart(chart_market_data, analysis_results, symbol):
    """Enhanced chart creation with Fibonacci EMAs and technical levels"""
    
    if isinstance(chart_market_data, dict):
        st.error(f"❌ CHART RECEIVED DICT INSTEAD OF DATAFRAME!")
        return None
    
    if not isinstance(chart_market_data, pd.DataFrame):
        st.error(f"❌ Invalid chart data type: {type(chart_market_data)}")
        return None
    
    if len(chart_market_data) == 0:
        st.error(f"❌ Chart DataFrame is empty")
        return None
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in chart_market_data.columns]
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        return None
    
    try:
        chart_data = chart_market_data.tail(100)
        current_price = analysis_results['current_price']
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(f'{symbol} Price Chart with Technical Levels', 'Volume', 'Weekly Deviations'),
            vertical_spacing=0.08, row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=chart_data.index, open=chart_data['Open'], high=chart_data['High'],
            low=chart_data['Low'], close=chart_data['Close'], name='Price'
        ), row=1, col=1)
        
        # Fibonacci EMAs
        fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
        ema_colors = {'EMA_21': 'orange', 'EMA_55': 'blue', 'EMA_89': 'purple', 'EMA_144': 'red', 'EMA_233': 'brown'}
        
        for ema_name, ema_value in fibonacci_emas.items():
            period = int(ema_name.split('_')[1])
            if len(chart_data) >= period:
                ema_series = chart_data['Close'].ewm(span=period).mean()
                color = ema_colors.get(ema_name, 'gray')
                fig.add_trace(go.Scatter(
                    x=chart_data.index, y=ema_series, name=ema_name, 
                    line=dict(color=color, width=1.5)
                ), row=1, col=1)
        
        # Daily VWAP
        daily_vwap = enhanced_indicators.get('daily_vwap')
        if daily_vwap:
            fig.add_hline(y=daily_vwap, line_dash="solid", 
                         line_color="cyan", line_width=2, row=1, col=1)
        
        # Point of Control
        poc = enhanced_indicators.get('point_of_control')
        if poc:
            fig.add_hline(y=poc, line_dash="dot", 
                         line_color="magenta", line_width=2, row=1, col=1)
        
        # Weekly Standard Deviations
        weekly_devs = enhanced_indicators.get('weekly_deviations', {})
        if weekly_devs:
            for std_level in [1, 2, 3]:
                std_data = weekly_devs.get(f'{std_level}_std')
                if std_data:
                    opacity = 0.6 - (std_level - 1) * 0.15  # Fade with distance
                    fig.add_hline(y=std_data['upper'], line_dash="dash", 
                                 line_color="green", line_width=1, opacity=opacity, row=1, col=1)
                    fig.add_hline(y=std_data['lower'], line_dash="dash", 
                                 line_color="red", line_width=1, opacity=opacity, row=1, col=1)
        
        # Confidence interval levels
        if analysis_results.get('confidence_analysis'):
            conf_data = analysis_results['confidence_analysis']['confidence_intervals']
            
            if '68%' in conf_data:
                fig.add_hline(y=conf_data['68%']['upper_bound'], line_dash="dash", 
                             line_color="lightgreen", line_width=1, row=1, col=1)
                fig.add_hline(y=conf_data['68%']['lower_bound'], line_dash="dash", 
                             line_color="lightgreen", line_width=1, row=1, col=1)
        
        # Current price line
        fig.add_hline(y=current_price, line_dash="solid", line_color="black", 
                     line_width=3, row=1, col=1)
        
        # Volume chart
        colors = ['green' if close >= open else 'red' 
                  for close, open in zip(chart_data['Close'], chart_data['Open'])]
        fig.add_trace(go.Bar(x=chart_data.index, y=chart_data['Volume'], 
                            name='Volume', marker_color=colors), row=2, col=1)
        
        # Weekly deviations chart
        if weekly_devs:
            std_levels = []
            upper_values = []
            lower_values = []
            range_pcts = []
            
            for std_level in [1, 2, 3]:
                std_data = weekly_devs.get(f'{std_level}_std')
                if std_data:
                    std_levels.append(f"{std_level}σ")
                    upper_values.append(std_data['upper'])
                    lower_values.append(std_data['lower'])
                    range_pcts.append(std_data['range_pct'])
            
            if std_levels:
                fig.add_trace(go.Scatter(
                    x=std_levels, y=upper_values, name='Upper Bounds', 
                    mode='markers+lines', marker_color='green'
                ), row=3, col=1)
                fig.add_trace(go.Scatter(
                    x=std_levels, y=lower_values, name='Lower Bounds', 
                    mode='markers+lines', marker_color='red'
                ), row=3, col=1)
        
        fig.update_layout(
            title=f'{symbol} | Confluence: {analysis_results["directional_confluence"]:.2f} | Signal: {analysis_results["signal_type"]}',
            height=800, showlegend=True, template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Weekly Levels ($)", row=3, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"❌ Error creating chart: {str(e)}")
        return None

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 VWV Professional Trading System</h1>
        <p>Advanced market analysis with enhanced technical indicators</p>
        <p><em>Features: Daily VWAP, Fibonacci EMAs, Point of Control, Weekly Deviations</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.title("📊 Trading Analysis")
    
    # Basic controls
    symbol = st.sidebar.text_input("Symbol", value="SPY", help="Enter stock symbol").upper()
    period = st.sidebar.selectbox("Data Period", ['1mo', '3mo', '6mo', '1y', '2y'], index=3)
    
    # Debug toggle
    show_debug = st.sidebar.checkbox("🐛 Show Debug Info", value=False)
    
    # System parameters
    with st.sidebar.expander("⚙️ System Parameters"):
        st.write("**Williams VIX Fix**")
        wvf_period = st.slider("WVF Period", 10, 50, 22)
        wvf_multiplier = st.slider("WVF Multiplier", 0.5, 2.0, 1.2, 0.1)
        
        st.write("**Signal Thresholds**")
        good_threshold = st.slider("Good Signal", 2.0, 5.0, 3.5, 0.1)
        strong_threshold = st.slider("Strong Signal", 3.0, 6.0, 4.5, 0.1)
        very_strong_threshold = st.slider("Very Strong Signal", 4.0, 7.0, 5.5, 0.1)
    
    # Create configuration
    custom_config = {
        'wvf_period': wvf_period,
        'wvf_multiplier': wvf_multiplier,
        'signal_thresholds': {
            'good': good_threshold,
            'strong': strong_threshold,
            'very_strong': very_strong_threshold
        }
    }
    
    # Initialize system
    vwv_system = VWVTradingSystem(custom_config)
    
    # Controls
    show_chart = st.sidebar.checkbox("Show Interactive Chart", value=True)
    test_button = st.sidebar.button("🧪 Test Data Fetch", use_container_width=True)
    analyze_button = st.sidebar.button("📊 Analyze Symbol", type="primary", use_container_width=True)
    
    # Data manager
    data_manager = st.session_state.data_manager
    
    # Test data fetch
    if test_button and symbol:
        if show_debug:
            st.write("## 🧪 Data Fetch Test")
        else:
            st.write("## 🧪 Quick Test")
        
        test_data = get_market_data_enhanced(symbol, period, show_debug)
        
        if test_data is not None:
            st.success(f"✅ Data fetch successful for {symbol}!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Data Points", len(test_data))
            with col2:
                st.metric("Current Price", f"${test_data['Close'].iloc[-1]:.2f}")
            with col3:
                st.metric("Date Range", f"{(test_data.index[-1] - test_data.index[0]).days} days")
            with col4:
                st.metric("Avg Volume", f"{test_data['Volume'].mean():,.0f}")
            
            if show_debug:
                st.write("**Sample Data:**")
                st.dataframe(test_data.tail().round(2), use_container_width=True)
        else:
            st.error(f"❌ Data fetch failed for {symbol}")
    
    # Full analysis
    if analyze_button and symbol:
        st.write("## 📊 VWV Trading Analysis")
        
        with st.spinner(f"Analyzing {symbol}..."):
            
            # Step 1: Fetch data
            if show_debug:
                st.write("### Step 1: Data Fetching")
            
            market_data = get_market_data_enhanced(symbol, period, show_debug)
            
            if market_data is None:
                st.error(f"❌ Could not fetch data for {symbol}")
                return
            
            # Store data
            data_manager.store_market_data(symbol, market_data, show_debug)
            
            # Step 2: Analysis
            if show_debug:
                st.write("### Step 2: Analysis Processing")
            
            analysis_input = data_manager.get_market_data_for_analysis(symbol)
            
            if analysis_input is None:
                st.error("❌ Could not prepare analysis data")
                return
            
            analysis_results = vwv_system.calculate_confluence(analysis_input, symbol)
            
            if 'error' in analysis_results:
                st.error(f"❌ Analysis failed: {analysis_results['error']}")
                return
            
            # Store results
            data_manager.store_analysis_results(symbol, analysis_results)
            
            # Step 3: Display results
            if show_debug:
                st.write("### Step 3: Results Display")
            
            # Primary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${analysis_results['current_price']}")
            with col2:
                st.metric("Directional Confluence", f"{analysis_results['directional_confluence']:.2f}")
            with col3:
                signal_icons = {
                    "NONE": "⚪", "GOOD_LONG": "🟢⬆️", "GOOD_SHORT": "🟢⬇️",
                    "STRONG_LONG": "🟡⬆️", "STRONG_SHORT": "🟡⬇️",
                    "VERY_STRONG_LONG": "🔴⬆️", "VERY_STRONG_SHORT": "🔴⬇️"
                }
                st.metric("Signal", f"{signal_icons.get(analysis_results['signal_type'], '⚪')} {analysis_results['signal_type']}")
            with col4:
                trend_dir = analysis_results['trend_analysis']['trend_direction'] if analysis_results['trend_analysis'] else 'N/A'
                st.metric("Trend Direction", trend_dir)
            
            # Enhanced Technical Indicators
            st.subheader("📊 Enhanced Technical Indicators")
            enhanced_indicators = analysis_results.get('enhanced_indicators', {})
            
            # Daily VWAP and POC
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                daily_vwap = enhanced_indicators.get('daily_vwap', 0)
                st.metric("Daily VWAP", f"${daily_vwap:.2f}")
            with col2:
                poc = enhanced_indicators.get('point_of_control', 0)
                st.metric("Point of Control", f"${poc:.2f}")
            with col3:
                current_price = analysis_results['current_price']
                vwap_distance = ((current_price - daily_vwap) / daily_vwap * 100) if daily_vwap > 0 else 0
                st.metric("Distance from VWAP", f"{vwap_distance:.2f}%")
            with col4:
                poc_distance = ((current_price - poc) / poc * 100) if poc > 0 else 0
                st.metric("Distance from POC", f"{poc_distance:.2f}%")
            
            # Fibonacci EMAs
            fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
            if fibonacci_emas:
                st.write("**Fibonacci EMAs:**")
                ema_cols = st.columns(len(fibonacci_emas))
                for i, (ema_name, ema_value) in enumerate(fibonacci_emas.items()):
                    period = ema_name.split('_')[1]
                    distance_pct = ((current_price - ema_value) / ema_value * 100) if ema_value > 0 else 0
                    with ema_cols[i]:
                        st.metric(f"EMA {period}", f"${ema_value:.2f}", f"{distance_pct:+.1f}%")
            
            # Weekly Standard Deviations
            weekly_devs = enhanced_indicators.get('weekly_deviations', {})
            if weekly_devs:
                st.write("**Weekly Standard Deviations:**")
                
                # Mean and std info
                col1, col2 = st.columns(2)
                with col1:
                    mean_price = weekly_devs.get('mean_price', 0)
                    st.metric("Weekly Mean Price", f"${mean_price:.2f}")
                with col2:
                    std_price = weekly_devs.get('std_price', 0)
                    st.metric("Weekly Std Dev", f"${std_price:.2f}")
                
                # Deviation levels
                std_data = []
                for std_level in [1, 2, 3]:
                    std_info = weekly_devs.get(f'{std_level}_std')
                    if std_info:
                        # Determine which zone current price is in
                        if current_price > std_info['upper']:
                            zone = f"Above +{std_level}σ"
                        elif current_price < std_info['lower']:
                            zone = f"Below -{std_level}σ"
                        else:
                            zone = f"Within ±{std_level}σ"
                        
                        std_data.append({
                            'Level': f"±{std_level}σ",
                            'Upper': f"${std_info['upper']:.2f}",
                            'Lower': f"${std_info['lower']:.2f}",
                            'Range %': f"±{std_info['range_pct']:.1f}%",
                            'Current Zone': zone if std_level == 1 else ""
                        })
                
                if std_data:
                    df_std = pd.DataFrame(std_data)
                    st.dataframe(df_std, use_container_width=True)
            
            # Signal display
            if analysis_results['signal_type'] != 'NONE':
                entry_info = analysis_results['entry_info']
                direction = entry_info['direction']
                
                st.success(f"""
                🚨 **VWV {direction} SIGNAL DETECTED**
                
                **Signal:** {analysis_results['signal_type']}  
                **Direction:** {direction}  
                **Entry:** ${entry_info['entry_price']}  
                **Stop Loss:** ${entry_info['stop_loss']}  
                **Take Profit:** ${entry_info['take_profit']}  
                **Risk/Reward:** {entry_info['risk_reward']}:1
                """)
            
            # Confidence intervals
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
            
            # Components analysis (show only if debug is on)
            if show_debug:
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
            
            # Chart
            if show_chart:
                if show_debug:
                    st.write("### Step 4: Enhanced Chart Creation")
                else:
                    st.subheader("📈 Technical Chart")
                
                chart_market_data = data_manager.get_market_data_for_chart(symbol)
                
                if chart_market_data is None:
                    st.error("❌ Could not get chart data")
                    return
                
                chart = create_enhanced_chart(chart_market_data, analysis_results, symbol)
                
                if chart is not None:
                    st.plotly_chart(chart, use_container_width=True)
                    if show_debug:
                        st.success("✅ Chart created successfully")
                else:
                    st.error("❌ Chart creation failed")
    
    else:
        st.markdown("""
        ## 🛠️ VWV Professional Trading System
        
        ### ✅ **Enhanced Features:**
        
        1. **🔧 Core VWV Analysis**
           - Williams VIX Fix calculation
           - Moving average confluence
           - Volume analysis with VWAP
           - Momentum and volatility filters
           - Directional signal generation
        
        2. **📊 Enhanced Technical Indicators**
           - **Daily VWAP**: Volume-weighted average price
           - **Fibonacci EMAs**: 21, 55, 89, 144, 233 period exponential moving averages
           - **Point of Control**: Price level with highest volume activity
           - **Weekly Standard Deviations**: 1σ, 2σ, 3σ support/resistance levels
        
        3. **🎯 Professional Features**
           - Bidirectional signals (LONG/SHORT)
           - Statistical confidence intervals
           - Risk/reward calculations
           - Comprehensive technical chart
           - Debug mode for detailed analysis
        
        4. **📈 Chart Enhancements**
           - All Fibonacci EMAs displayed
           - VWAP and Point of Control levels
           - Weekly standard deviation bands
           - Interactive technical analysis
        
        ### 📊 **How to Use:**
        
        1. **Test First**: Click "🧪 Test Data Fetch" to verify symbol works
        2. **Analyze**: Click "📊 Analyze Symbol" for complete analysis
        3. **Review Signals**: Check VWV confluence and directional signals
        4. **Study Levels**: Use technical levels for entry/exit planning
        
        ### 🔧 **System Controls:**
        
        - **Debug Toggle**: Shows/hides detailed process information
        - **System Parameters**: Customize VWV calculation settings
        - **Chart Options**: Control technical chart display
        
        **Start with: SPY, AAPL, MSFT, GOOGL, QQQ, or TSLA**
        """)

if __name__ == "__main__":
    main()
