"""
Data fetching and validation functionality
Updated to suppress messages during bulk screening
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import logging
from typing import Optional, Dict, Any
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

class DataQualityChecker:
    """Comprehensive data quality validation"""
    
    @staticmethod
    def validate_market_data(data: pd.DataFrame) -> Dict[str, Any]:
        """Validate market data quality and completeness"""
        issues = []
        quality_score = 100
        
        # Check for missing data
        if data.empty or len(data.columns) == 0:
            return {
                'quality_score': 0, 'issues': ['Data is empty'],
                'data_points': 0, 'date_range': None, 'is_acceptable': False
            }
        
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
        if missing_pct > 5:
            issues.append(f"High missing data: {missing_pct:.1f}%")
            quality_score -= 20
            
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
            
        return {
            'quality_score': max(0, quality_score),
            'issues': issues,
            'data_points': len(data),
            'date_range': (data.index[0].strftime('%Y-%m-%d'), data.index[-1].strftime('%Y-%m-%d')) if len(data) > 0 else None,
            'is_acceptable': quality_score >= 70
        }

def is_etf(symbol: str) -> bool:
    """Detect if a symbol is an ETF"""
    try:
        known_stocks = {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'JPM', 'JNJ', 'UNH', 'V', 'PG', 'HD', 'MA', 'BAC', 'ABBV', 'PFE'
        }
        if symbol.upper() in known_stocks:
            return False
        
        common_etfs = {
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'AGG', 'BND', 'TLT', 'GLD', 'SLV', 'USO', 'UNG', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'ARKK'
        }
        if symbol.upper() in common_etfs:
            return True
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            quote_type = info.get('quoteType', '').upper()
            if quote_type == 'ETF':
                return True
            category = info.get('category', '')
            if category and 'ETF' in category:
                return True
        except Exception as e:
            logger.debug(f"yfinance lookup failed for {symbol}: {e}")
        
        return False
    except Exception as e:
        logger.error(f"ETF detection error for {symbol}: {e}")
        return False

@safe_calculation_wrapper
def get_market_data_enhanced(symbol: str = 'SPY', period: str = '1y', show_debug: bool = False) -> Optional[pd.DataFrame]:
    """Enhanced market data fetching with debug control"""
    try:
        if show_debug:
            st.write(f"📡 Fetching data for {symbol}...")

        ticker = yf.Ticker(symbol)
        raw_data = ticker.history(period="6mo")

        if raw_data.empty:
            if show_debug:
                st.error(f"❌ No data returned for {symbol}")
            return None

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in raw_data.columns for col in required_columns):
            if show_debug:
                st.error(f"❌ Missing required columns for {symbol}")
            return None

        clean_data = raw_data[required_columns].copy().dropna()

        if clean_data.empty:
            if show_debug:
                st.error(f"❌ No data after cleaning for {symbol}")
            return None

        clean_data['Typical_Price'] = (clean_data['High'] + clean_data['Low'] + clean_data['Close']) / 3
        
        quality_check = DataQualityChecker.validate_market_data(clean_data)
        if not quality_check['is_acceptable'] and show_debug:
            st.warning(f"⚠️ Data quality issues for {symbol}: {quality_check['issues']}")

        if show_debug:
            st.success(f"✅ Data ready for {symbol}: {clean_data.shape}")

        return clean_data

    except Exception as e:
        if show_debug:
            st.error(f"❌ Error fetching {symbol}: {str(e)}")
        return None
