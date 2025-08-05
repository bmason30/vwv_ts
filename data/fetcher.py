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
        
        # Check for missing 
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
        if missing_pct > 5:
            issues.append(f"High missing : {missing_pct:.1f}%")
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
            'date_range': (data.index[0], data.index[-1]) if len(data) > 0 else None,
            'is_acceptable': quality_score >= 70
        }

def is_etf(symbol: str) -> bool:
    """Detect if a symbol is an ETF"""
    try:
        # Known individual stocks that should never be considered ETFs
        known_stocks = {
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'JPM', 'JNJ', 'UNH', 'V', 'PG', 'HD', 'MA', 'BAC', 'ABBV', 'PFE',
            'KO', 'ADBE', 'PEP', 'TMO', 'COST', 'AVGO', 'NKE', 'MRK', 'ABT', 'CRM',
            'LLY', 'ACN', 'TXN', 'DHR', 'WMT', 'NEE', 'VZ', 'ORCL', 'CMCSA', 'PM',
            'DIS', 'BMY', 'RTX', 'HON', 'QCOM', 'UPS', 'T', 'AIG', 'LOW', 'MDT'
        }
        
        if symbol.upper() in known_stocks:
            return False
        
        # Common ETF patterns and known ETFs
        etf_suffixes = ['ETF', 'FUND']
        common_etfs = {
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'AGG', 'BND', 'TLT',
            'GLD', 'SLV', 'USO', 'UNG', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP',
            'XLY', 'XLU', 'XLRE', 'XLB', 'EFA', 'EEM', 'FXI', 'EWJ', 'EWG', 'EWU',
            'ARKK', 'ARKQ', 'ARKW', 'ARKG', 'ARKF', 'FNGU', 'FNGD', 'MAGS', 'SOXX',
            'SMH', 'IBB', 'XBI', 'JETS', 'HACK', 'ESPO', 'ICLN', 'PBW', 'KWEB',
            'SPHB', 'SOXL', 'QQI', 'DIVO', 'URNM', 'GDX', 'FETH'
        }
        
        if symbol.upper() in common_etfs:
            return True
        
        for suffix in etf_suffixes:
            if symbol.upper().endswith(suffix):
                return True
        
        # Try to get security type from yfinance (more reliable but slower)
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            quote_type = info.get('quoteType', '').upper()
            
            if quote_type == 'ETF':
                return True
                
            category = info.get('category', '').upper()
            if 'ETF' in category and ('EXCHANGE' in category or 'TRADED' in category):
                return True
                
            fund_family = info.get('fundFamily', '').upper()
            if fund_family and ('ETF' in fund_family or 'FUND' in fund_family):
                if quote_type in ['ETF', 'MUTUALFUND']:
                    return True
                    
            long_name = info.get('longName', '').upper()
            etf_name_indicators = ['EXCHANGE TRADED FUND', 'ETF']
            for indicator in etf_name_indicators:
                if indicator in long_name and quote_type != 'EQUITY':
                    return True
                    
        except Exception as e:
            # This 'except' block correctly closes the inner 'try'
            logger.debug(f"yfinance lookup failed for {symbol}: {e}")
        
        # Default to False if the API call fails or doesn't confirm it's an ETF
        return False
        
    except Exception as e:
        # This 'except' block correctly closes the outer 'try'
        logger.error(f"ETF detection error for {symbol}: {e}")
        return False
