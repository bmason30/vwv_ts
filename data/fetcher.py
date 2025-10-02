"""
Data fetching and validation functionality - v2.0.1 FIXED
Fixed timezone comparison issue causing IndentationError
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import logging
import time
from datetime import datetime, timedelta, timezone
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
        days_since_update = None
        
        # Check for empty data
        if data.empty or len(data.columns) == 0:
            return {
                'quality_score': 0, 
                'issues': ['Data is empty'],
                'data_points': 0, 
                'date_range': None, 
                'is_acceptable': False,
                'days_since_update': None
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
            extreme_returns = (abs(returns) > 0.2).sum()
            if extreme_returns > len(returns) * 0.02:
                issues.append(f"Excessive extreme returns: {extreme_returns}")
                quality_score -= 15
        
        # Check volume consistency
        if 'Volume' in data.columns:
            zero_volume_days = (data['Volume'] == 0).sum()
            if zero_volume_days > len(data) * 0.05:
                issues.append(f"High zero-volume days: {zero_volume_days}")
                quality_score -= 10
                
        # Check price consistency
        price_inconsistencies = ((data['High'] < data['Low']) | 
                                 (data['Close'] > data['High']) | 
                                 (data['Close'] < data['Low'])).sum()
        if price_inconsistencies > 0:
            issues.append(f"Price inconsistencies: {price_inconsistencies}")
            quality_score -= 25
        
        # Check for minimum data points
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
        
        # Check data freshness - FIXED TIMEZONE ISSUE
        if len(data) > 0:
            try:
                last_date = data.index[-1]
                now_utc = datetime.now(timezone.utc)
                
                # Make last_date timezone-aware
                if last_date.tzinfo is None:
                    last_date = last_date.replace(tzinfo=timezone.utc)
                else:
                    last_date = last_date.astimezone(timezone.utc)
                
                days_since_update = (now_utc - last_date).days
                
                if days_since_update > 7:
                    issues.append(f"Stale data: last update {days_since_update} days ago")
                    quality_score -= 20
                elif days_since_update > 3:
                    issues.append(f"Data slightly stale: {days_since_update} days old")
                    quality_score -= 5
            except Exception as e:
                logger.warning(f"Could not check data freshness: {e}")
                days_since_update = None
            
        return {
            'quality_score': max(0, quality_score),
            'issues': issues,
            'data_points': len(data),
            'date_range': (data.index[0].strftime('%Y-%m-%d'), data.index[-1].strftime('%Y-%m-%d')) if len(data) > 0 else None,
            'is_acceptable': quality_score >= 70,
            'days_since_update': days_since_update
        }

@st.cache_data(ttl=86400)
def is_etf(symbol: str) -> bool:
    """Detect if a symbol is an ETF with caching"""
    try:
        symbol_upper = symbol.upper()
        
        # Known stocks
        known_stocks = {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 
            'NFLX', 'JPM', 'JNJ', 'UNH', 'V', 'PG', 'HD', 'MA', 'BAC', 'ABBV', 
            'PFE', 'COST', 'DIS', 'ADBE', 'CRM', 'TMO', 'CSCO', 'ACN', 'MCD',
            'WMT', 'DHR', 'VZ', 'CMCSA', 'INTC', 'AMD', 'QCOM', 'TXN', 'ORCL',
            'NKE', 'HON', 'UPS', 'IBM', 'AMGN', 'NEE', 'CVX', 'PM', 'RTX'
        }
        if symbol_upper in known_stocks:
            return False
        
        # Known ETFs
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
        
        # Try yfinance lookup
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            quote_type = info.get('quoteType', '').upper()
            if quote_type == 'ETF':
                return True
            category = info.get('category', '')
            if category and 'ETF' in category.upper():
                return True
            has_fundamentals = any([
                info.get('trailingPE'),
                info.get('forwardPE'),
                info.get('priceToBook')
            ])
            if not has_fundamentals and info.get('totalAssets'):
                return True
        except Exception as e:
            logger.debug(f"yfinance lookup failed for {symbol}: {e}")
        
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
    """Enhanced market data fetching with comprehensive validation"""
    
    for attempt in range(max_retries):
        try:
            if show_debug:
                st.write(f"📡 Fetching {period} of data for {symbol}... (attempt {attempt + 1}/{max_retries})")

            ticker = yf.Ticker(symbol)
            raw_data = ticker.history(period=period)

            if raw_data.empty:
                if show_debug:
                    st.error(f"❌ No data returned for {symbol}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    if show_debug:
                        st.info(f"⏳ Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return None

            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in raw_data.columns]
            
            if missing_columns:
                if show_debug:
                    st.error(f"❌ Missing columns for {symbol}: {missing_columns}")
                return None

            clean_data = raw_data[required_columns].copy()
            initial_length = len(clean_data)
            clean_data = clean_data.dropna()
            dropped_rows = initial_length - len(clean_data)
            
            if dropped_rows > 0 and show_debug:
                st.info(f"ℹ️ Dropped {dropped_rows} rows with missing data")

            if clean_data.empty:
                if show_debug:
                    st.error(f"❌ No valid data after cleaning for {symbol}")
                return None

            clean_data['Typical_Price'] = (clean_data['High'] + clean_data['Low'] + clean_data['Close']) / 3
            
            quality_check = DataQualityChecker.validate_market_data(clean_data, symbol, period)
            
            if show_debug and quality_check['issues']:
                st.warning(f"⚠️ Data quality issues for {symbol}:")
                for issue in quality_check['issues']:
                    st.write(f"  • {issue}")
                st.write(f"**Quality Score:** {quality_check['quality_score']}/100")
            
            if not quality_check['is_acceptable']:
                if show_debug:
                    st.error(f"❌ Data quality too low for {symbol} (score: {quality_check['quality_score']}/100)")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    if show_debug:
                        st.info(f"⏳ Retrying with fresh data in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return None

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
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                if show_debug:
                    st.info(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            
            return None
    
    return None

def get_multiple_symbols_data(
    symbols: list, 
    period: str = '1y',
    show_progress: bool = False
) -> Dict[str, pd.DataFrame]:
    """Fetch data for multiple symbols with progress tracking"""
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
            
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            continue
    
    if show_progress:
        progress_bar.empty()
        status_text.empty()
    
    return results
