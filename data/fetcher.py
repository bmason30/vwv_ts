"""
Data fetching and validation functionality - v2.0.0 ENHANCED
FIXES APPLIED:
- Fixed hardcoded period parameter (CRITICAL BUG)
- Added minimum data validation
- Quality checks now block bad data
- Improved ETF detection with caching
- Added retry logic with exponential backoff
- Added data freshness validation
- Enhanced error handling and logging
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

class DataQualityChecker:
    """Comprehensive data quality validation"""
    
    @staticmethod
    def validate_market_data(data: pd.DataFrame, symbol: str = "", period: str = "") -> Dict[str, Any]:
        """Validate market data quality and completeness"""
        issues = []
        quality_score = 100
        
        # Check for empty data
        if data.empty or len(data.columns) == 0:
            return {
                'quality_score': 0, 
                'issues': ['Data is empty'],
                'data_points': 0, 
                'date_range': None, 
                'is_acceptable': False
            }
        
        # Check for missing data
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
        if missing_pct > 5:
            issues.append(f"High missing data: {missing_pct:.1f}%")
            quality_score -= 20
        elif missing_pct > 1:
            issues.append(f"Some missing data: {missing_pct:.1f}%")
            quality_score -= 5
            
        # Check for price anomalies
        returns = data['Close'].pct_change().dropna()
        if len(returns) > 0:
            extreme_returns = (abs(returns) > 0.2).sum()  # >20% moves
            if extreme_returns > len(returns) * 0.02:  # More than 2% of data
                issues.append(f"Excessive extreme returns: {extreme_returns}")
                quality_score -= 15
        
        # Check volume consistency
        if 'Volume' in data.columns:
            zero_volume_days = (data['Volume'] == 0).sum()
            if zero_volume_days > len(data) * 0.05:  # More than 5%
                issues.append(f"High zero-volume days: {zero_volume_days}")
                quality_score -= 10
                
        # Check price consistency (High >= Low, etc.)
        price_inconsistencies = ((data['High'] < data['Low']) | 
                                 (data['Close'] > data['High']) | 
                                 (data['Close'] < data['Low'])).sum()
        if price_inconsistencies > 0:
            issues.append(f"Price inconsistencies: {price_inconsistencies}")
            quality_score -= 25
        
        # Check for minimum data points based on period
        min_days_required = {
            '1mo': 15,
            '3mo': 45,
            '6mo': 90,
            '1y': 180,
            '2y': 360,
            '5y': 900,
            'max': 1000
        }
        
        min_days = min_days_required.get(period, 20)
        if len(data) < min_days:
            issues.append(f"Insufficient data: {len(data)} days (expected ~{min_days})")
            quality_score -= 30
        
        # Check data freshness (last update should be recent)
        if len(data) > 0:
            last_date = data.index[-1]
            days_since_update = (datetime.now() - last_date).days
            
            # Allow up to 7 days for stale data (accounts for weekends/holidays)
            if days_since_update > 7:
                issues.append(f"Stale data: last update {days_since_update} days ago")
                quality_score -= 20
            elif days_since_update > 3:
                issues.append(f"Data may be slightly stale: {days_since_update} days old")
                quality_score -= 5
            
        return {
            'quality_score': max(0, quality_score),
            'issues': issues,
            'data_points': len(data),
            'date_range': (data.index[0].strftime('%Y-%m-%d'), data.index[-1].strftime('%Y-%m-%d')) if len(data) > 0 else None,
            'is_acceptable': quality_score >= 70,  # Minimum 70/100 to be acceptable
            'days_since_update': days_since_update if len(data) > 0 else None
        }

@st.cache_data(ttl=86400)  # Cache ETF detection for 24 hours
def is_etf(symbol: str) -> bool:
    """
    Detect if a symbol is an ETF with caching
    Uses multiple detection methods for accuracy
    """
    try:
        symbol_upper = symbol.upper()
        
        # Known individual stocks (definitely NOT ETFs)
        known_stocks = {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 
            'NFLX', 'JPM', 'JNJ', 'UNH', 'V', 'PG', 'HD', 'MA', 'BAC', 'ABBV', 
            'PFE', 'COST', 'DIS', 'ADBE', 'CRM', 'TMO', 'CSCO', 'ACN', 'MCD',
            'WMT', 'DHR', 'VZ', 'CMCSA', 'INTC', 'AMD', 'QCOM', 'TXN', 'ORCL',
            'NKE', 'HON', 'UPS', 'IBM', 'AMGN', 'NEE', 'CVX', 'PM', 'RTX'
        }
        if symbol_upper in known_stocks:
            return False
        
        # Known ETFs (definitely ETFs)
        common_etfs = {
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'AGG', 'BND', 
            'TLT', 'GLD', 'SLV', 'USO', 'UNG', 'XLF', 'XLE', 'XLK', 'XLV', 
            'XLI', 'XLP', 'ARKK', 'ARKG', 'ARKW', 'MAGS', 'JEPI', 'DIVO',
            'SCHD', 'SPYI', 'HYG', 'JNK', 'SPHB', 'EWW', 'FXI', 'INDA',
            'UUP', 'UDN', 'GDX', 'URNM', 'PHYS', 'FNGD', 'FNGU',
            'VGT', 'VIG', 'VYM', 'VTV', 'VUG', 'VXUS', 'IEMG', 'EFA',
            'VNQ', 'IYR', 'VCIT', 'VCSH', 'BSV', 'BIV', 'BLV'
        }
        if symbol_upper in common_etfs:
            return True
        
        # Try yfinance lookup as fallback
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check quoteType
            quote_type = info.get('quoteType', '').upper()
            if quote_type == 'ETF':
                return True
            
            # Check category
            category = info.get('category', '')
            if category and 'ETF' in category.upper():
                return True
            
            # Check if it has fundamental data (stocks have, ETFs usually don't)
            # This is a heuristic, not definitive
            has_fundamentals = any([
                info.get('trailingPE'),
                info.get('forwardPE'),
                info.get('priceToBook')
            ])
            
            if not has_fundamentals and info.get('totalAssets'):
                # Has assets but no PE ratio -> likely an ETF
                return True
                
        except Exception as e:
            logger.debug(f"yfinance lookup failed for {symbol}: {e}")
        
        # If uncertain, assume it's a stock (safer default)
        return False
        
    except Exception as e:
        logger.error(f"ETF detection error for {symbol}: {e}")
        return False

@safe_calculation_wrapper
def get_market_data_enhanced(
    symbol: str = 'SPY', 
    period: str = '1y', 
    show_debug: bool = False,
    max_retries: int = 3
) -> Optional[pd.DataFrame]:
    """
    Enhanced market data fetching with comprehensive validation
    
    Args:
        symbol: Stock/ETF ticker symbol
        period: Time period ('1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
        show_debug: Show detailed debug information
        max_retries: Maximum number of retry attempts
    
    Returns:
        DataFrame with OHLCV data and Typical_Price, or None if fetch fails
    """
    
    for attempt in range(max_retries):
        try:
            if show_debug:
                st.write(f"📡 Fetching {period} of data for {symbol}... (attempt {attempt + 1}/{max_retries})")

            # Fetch data from yfinance
            ticker = yf.Ticker(symbol)
            raw_data = ticker.history(period=period)  # ✅ FIXED: Use the period parameter!

            # Check if data was returned
            if raw_data.empty:
                if show_debug:
                    st.error(f"❌ No data returned for {symbol}")
                
                # Retry with exponential backoff
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    if show_debug:
                        st.info(f"⏳ Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return None

            # Verify required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in raw_data.columns]
            
            if missing_columns:
                if show_debug:
                    st.error(f"❌ Missing required columns for {symbol}: {missing_columns}")
                return None

            # Clean the data
            clean_data = raw_data[required_columns].copy()
            
            # Remove rows with NaN values
            initial_length = len(clean_data)
            clean_data = clean_data.dropna()
            dropped_rows = initial_length - len(clean_data)
            
            if dropped_rows > 0 and show_debug:
                st.info(f"ℹ️ Dropped {dropped_rows} rows with missing data")

            # Check if we have any data left after cleaning
            if clean_data.empty:
                if show_debug:
                    st.error(f"❌ No valid data remaining after cleaning for {symbol}")
                return None

            # Add Typical Price calculation
            clean_data['Typical_Price'] = (clean_data['High'] + clean_data['Low'] + clean_data['Close']) / 3
            
            # Validate data quality
            quality_check = DataQualityChecker.validate_market_data(clean_data, symbol, period)
            
            # Log quality issues if debug enabled
            if show_debug and quality_check['issues']:
                st.warning(f"⚠️ Data quality issues for {symbol}:")
                for issue in quality_check['issues']:
                    st.write(f"  • {issue}")
                st.write(f"**Quality Score:** {quality_check['quality_score']}/100")
            
            # Block data if quality is too low
            if not quality_check['is_acceptable']:
                if show_debug:
                    st.error(f"❌ Data quality too low for {symbol} (score: {quality_check['quality_score']}/100)")
                    st.write("**Issues:**")
                    for issue in quality_check['issues']:
                        st.write(f"  • {issue}")
                
                # Retry if we have attempts left
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    if show_debug:
                        st.info(f"⏳ Retrying with fresh data in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return None

            # Success! Log details if debug enabled
            if show_debug:
                st.success(f"✅ Data ready for {symbol}")
                st.write(f"**Data Points:** {quality_check['data_points']}")
                st.write(f"**Date Range:** {quality_check['date_range'][0]} to {quality_check['date_range'][1]}")
                st.write(f"**Quality Score:** {quality_check['quality_score']}/100")
                if quality_check['days_since_update'] is not None:
                    st.write(f"**Data Freshness:** {quality_check['days_since_update']} days old")

            return clean_data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol} (attempt {attempt + 1}): {e}")
            
            if show_debug:
                st.error(f"❌ Error fetching {symbol}: {str(e)}")
            
            # Retry with exponential backoff
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                if show_debug:
                    st.info(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            
            # Final failure
            return None
    
    # Should never reach here, but just in case
    return None


# Utility function for bulk data fetching (used in screening/correlation)
def get_multiple_symbols_data(
    symbols: list, 
    period: str = '1y',
    show_progress: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple symbols with progress tracking
    
    Args:
        symbols: List of ticker symbols
        period: Time period to fetch
        show_progress: Show progress bar in Streamlit
    
    Returns:
        Dictionary mapping symbols to their DataFrames
    """
    results = {}
    
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    for idx, symbol in enumerate(symbols):
        try:
            if show_progress:
                progress = (idx + 1) / len(symbols)
                progress_bar.progress(progress)
                status_text.text(f"Fetching {symbol}... ({idx + 1}/{len(symbols)})")
            
            data = get_market_data_enhanced(symbol, period, show_debug=False)
            if data is not None:
                results[symbol] = data
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            continue
    
    if show_progress:
        progress_bar.empty()
        status_text.empty()
    
    return results
