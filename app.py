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
from typing import Optional, Dict, Any
import hashlib

# Suppress only specific warnings, not all
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        position: relative;
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        overflow: hidden;
        min-height: 200px;
        
        /* Great Wave inspired layered background */
        background: 
            /* Sky gradient */
            linear-gradient(to bottom, #1a4d3a 0%, #2d6b4f 40%, #1e5540 100%),
            /* Main wave shape using clip-path technique */
            radial-gradient(ellipse 120% 60% at 20% 100%, #0a2f1f 0%, #0a2f1f 25%, transparent 26%),
            radial-gradient(ellipse 150% 80% at 80% 95%, #134a35 0%, #134a35 30%, transparent 31%),
            radial-gradient(ellipse 200% 100% at 50% 90%, #0f3b28 0%, #0f3b28 20%, transparent 21%);
        
        /* Wave foam effect */
        background-image:
            /* Small foam dots */
            radial-gradient(circle at 15% 75%, rgba(255,255,255,0.4) 1px, transparent 2px),
            radial-gradient(circle at 25% 78%, rgba(255,255,255,0.5) 1.5px, transparent 2.5px),
            radial-gradient(circle at 35% 72%, rgba(255,255,255,0.3) 1px, transparent 2px),
            radial-gradient(circle at 45% 80%, rgba(255,255,255,0.4) 1.2px, transparent 2.2px),
            radial-gradient(circle at 55% 76%, rgba(255,255,255,0.3) 1px, transparent 2px),
            radial-gradient(circle at 65% 82%, rgba(255,255,255,0.5) 1.3px, transparent 2.3px),
            radial-gradient(circle at 75% 74%, rgba(255,255,255,0.4) 1px, transparent 2px),
            radial-gradient(circle at 85% 79%, rgba(255,255,255,0.3) 1.4px, transparent 2.4px),
            /* Wave crest highlights */
            linear-gradient(135deg, transparent 60%, rgba(255,255,255,0.1) 65%, transparent 70%),
            linear-gradient(45deg, transparent 40%, rgba(255,255,255,0.05) 45%, transparent 50%);
        
        background-size:
            100% 100%,  /* Sky */
            100% 100%,  /* Wave shapes */
            100% 100%,
            100% 100%,
            30px 30px,  /* Foam dots */
            35px 35px,
            40px 40px,
            25px 25px,
            45px 45px,
            50px 50px,
            30px 30px,
            35px 35px,
            100% 100%,  /* Wave highlights */
            100% 100%;
    }
    
    /* Candlestick chart background */
    .candlestick-chart {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 1;
        opacity: 0.3;
    }
    
    .candle {
        position: absolute;
        width: 6px;
        background: #28a745;
        border-radius: 1px;
        transform: translateX(-50%);
    }
    
    .candle::before {
        content: '';
        position: absolute;
        left: 50%;
        top: -8px;
        width: 1px;
        height: calc(100% + 16px);
        background: currentColor;
        transform: translateX(-50%);
    }
    
    .candle-green {
        background: #28a745;
        color: #28a745;
        box-shadow: 0 0 3px rgba(40, 167, 69, 0.4);
    }
    
    .candle-red {
        background: #dc3545;
        color: #dc3545;
        box-shadow: 0 0 3px rgba(220, 53, 69, 0.4);
    }
    
    /* Dollar signs in foam */
    .foam-dollars {
        position: absolute;
        bottom: 40px;
        left: 50%;
        transform: translateX(-50%);
        font-size: 14px;
        color: rgba(255,255,255,0.6);
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        letter-spacing: 20px;
        opacity: 0.8;
        z-index: 1;
    }
    
    /* Header content */
    .header-content {
        position: relative;
        z-index: 3;
        background: rgba(0,0,0,0.2);
        padding: 1.5rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .header-content h1 {
        font-size: 2.8rem;
        margin-bottom: 1rem;
        color: #ffffff;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .header-content p {
        color: #f0f8f0;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.7);
        margin: 0.5rem 0;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    .header-content em {
        color: #e0f0e0;
        font-style: italic;
        font-size: 1rem;
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

class DataQualityChecker:
    """Comprehensive data quality validation"""
    
    @staticmethod
    def validate_market_data(data: pd.DataFrame) -> Dict[str, Any]:
        """Validate market data quality and completeness"""
        issues = []
        quality_score = 100
        
        # Check for missing data
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
            'date_range': (data.index[0], data.index[-1]) if len(data) > 0 else None,
            'is_acceptable': quality_score >= 70
        }

def safe_calculation_wrapper(func):
    """Decorator for safe financial calculations"""
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if result is None:
                logger.warning(f"Function {func.__name__} returned None")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper

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

@st.cache_data(ttl=300)  # 5-minute cache
def get_cached_market_data(symbol: str, period: str):
    """Cached market data retrieval"""
    return get_market_data_enhanced(symbol, period, show_debug=False)

def generate_cache_key(symbol: str, analysis_config: dict) -> str:
    """Generate unique cache key for analysis results"""
    config_str = str(sorted(analysis_config.items()))
    return hashlib.md5(f"{symbol}_{config_str}".encode()).hexdigest()

# ENHANCED: Data fetching with debug control
def is_etf(symbol):
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
        
        # If it's a known individual stock, it's definitely not an ETF
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
        
        # Check if symbol is in known ETFs list
        if symbol.upper() in common_etfs:
            return True
        
        # Check for ETF suffixes
        for suffix in etf_suffixes:
            if symbol.upper().endswith(suffix):
                return True
        
        # Try to get security type from yfinance (more reliable but slower)
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # More specific checks for ETF identification
            quote_type = info.get('quoteType', '').upper()
            
            # Only consider it an ETF if quoteType is specifically "ETF"
            if quote_type == 'ETF':
                return True
                
            # Check category field but be more specific
            category = info.get('category', '').upper()
            if 'ETF' in category and ('EXCHANGE' in category or 'TRADED' in category):
                return True
                
            # Check fund family but only if other indicators suggest ETF
            fund_family = info.get('fundFamily', '').upper()
            if fund_family and ('ETF' in fund_family or 'FUND' in fund_family):
                # Double-check with security type
                if quote_type in ['ETF', 'MUTUALFUND']:
                    return True
                    
            # Be very specific about name checks to avoid false positives
            long_name = info.get('longName', '').upper()
            short_name = info.get('shortName', '').upper()
            
            # Only flag as ETF if name explicitly contains ETF-specific terms
            etf_name_indicators = ['EXCHANGE TRADED FUND', 'ETF']
            for indicator in etf_name_indicators:
                if indicator in long_name and quote_type != 'EQUITY':
                    return True
                    
        except Exception as e:
            # If yfinance lookup fails, use conservative pattern matching
            logger.debug(f"yfinance lookup failed for {symbol}: {e}")
        
        # Default to False (assume it's a stock) if uncertain
        return False
        
    except Exception as e:
        logger.error(f"ETF detection error for {symbol}: {e}")
        return False

@safe_calculation_wrapper
def calculate_graham_score(symbol, show_debug=False):
    """Calculate Benjamin Graham Score based on value investing criteria"""
    try:
        if show_debug:
            st.write(f"📊 Calculating Graham Score for {symbol}...")
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        
        if not info or len(financials.columns) == 0 or len(balance_sheet.columns) == 0:
            return {'score': 0, 'total_possible': 10, 'criteria': [], 'error': 'Insufficient fundamental data'}
        
        score = 0
        total_possible = 10
        criteria = []
        
        # Get key metrics
        pe_ratio = info.get('trailingPE', info.get('forwardPE', None))
        pb_ratio = info.get('priceToBook', None)
        debt_to_equity = info.get('debtToEquity', None)
        current_ratio = info.get('currentRatio', None)
        quick_ratio = info.get('quickRatio', None)
        
        # Calculate additional metrics from financial statements if available
        try:
            # Get most recent year data
            latest_financials = financials.iloc[:, 0] if len(financials.columns) > 0 else None
            prev_financials = financials.iloc[:, 1] if len(financials.columns) > 1 else None
            
            # Earnings growth
            earnings_growth = None
            if latest_financials is not None and prev_financials is not None:
                if 'Net Income' in latest_financials and 'Net Income' in prev_financials:
                    latest_earnings = latest_financials.get('Net Income', 0)
                    prev_earnings = prev_financials.get('Net Income', 0)
                    if prev_earnings != 0:
                        earnings_growth = (latest_earnings - prev_earnings) / abs(prev_earnings)
            
            # Revenue growth
            revenue_growth = None
            if latest_financials is not None and prev_financials is not None:
                if 'Total Revenue' in latest_financials and 'Total Revenue' in prev_financials:
                    latest_revenue = latest_financials.get('Total Revenue', 0)
                    prev_revenue = prev_financials.get('Total Revenue', 0)
                    if prev_revenue != 0:
                        revenue_growth = (latest_revenue - prev_revenue) / abs(prev_revenue)
        
        except:
            earnings_growth = None
            revenue_growth = None
        
        # Graham Criteria Evaluation
        
        # 1. P/E ratio < 15
        if pe_ratio and pe_ratio < 15:
            score += 1
            criteria.append(f"✅ P/E < 15 ({pe_ratio:.2f})")
        else:
            pe_display = f"{pe_ratio:.2f}" if pe_ratio else "N/A"
            criteria.append(f"❌ P/E < 15 ({pe_display})")
        
        # 2. P/B ratio < 1.5
        if pb_ratio and pb_ratio < 1.5:
            score += 1
            criteria.append(f"✅ P/B < 1.5 ({pb_ratio:.2f})")
        else:
            pb_display = f"{pb_ratio:.2f}" if pb_ratio else "N/A"
            criteria.append(f"❌ P/B < 1.5 ({pb_display})")
        
        # 3. P/E × P/B < 22.5
        if pe_ratio and pb_ratio and (pe_ratio * pb_ratio) < 22.5:
            score += 1
            criteria.append(f"✅ P/E × P/B < 22.5 ({pe_ratio * pb_ratio:.2f})")
        else:
            pe_pb_product = f"{pe_ratio * pb_ratio:.2f}" if (pe_ratio and pb_ratio) else "N/A"
            criteria.append(f"❌ P/E × P/B < 22.5 ({pe_pb_product})")
        
        # 4. Debt-to-Equity < 0.5 (50%)
        if debt_to_equity is not None:
            debt_ratio = debt_to_equity / 100  # Convert percentage to decimal
            if debt_ratio < 0.5:
                score += 1
                criteria.append(f"✅ Debt/Equity < 50% ({debt_to_equity:.1f}%)")
            else:
                criteria.append(f"❌ Debt/Equity < 50% ({debt_to_equity:.1f}%)")
        else:
            criteria.append("❌ Debt/Equity < 50% (N/A)")
        
        # 5. Current Ratio > 1.5
        if current_ratio and current_ratio > 1.5:
            score += 1
            criteria.append(f"✅ Current Ratio > 1.5 ({current_ratio:.2f})")
        else:
            current_display = f"{current_ratio:.2f}" if current_ratio else "N/A"
            criteria.append(f"❌ Current Ratio > 1.5 ({current_display})")
        
        # 6. Quick Ratio > 1.0
        if quick_ratio and quick_ratio > 1.0:
            score += 1
            criteria.append(f"✅ Quick Ratio > 1.0 ({quick_ratio:.2f})")
        else:
            quick_display = f"{quick_ratio:.2f}" if quick_ratio else "N/A"
            criteria.append(f"❌ Quick Ratio > 1.0 ({quick_display})")
        
        # 7. Positive earnings growth
        if earnings_growth is not None and earnings_growth > 0:
            score += 1
            criteria.append(f"✅ Earnings Growth > 0% ({earnings_growth*100:.1f}%)")
        else:
            earnings_display = f"{earnings_growth*100:.1f}%" if earnings_growth is not None else "N/A"
            criteria.append(f"❌ Earnings Growth > 0% ({earnings_display})")
        
        # 8. Positive revenue growth
        if revenue_growth is not None and revenue_growth > 0:
            score += 1
            criteria.append(f"✅ Revenue Growth > 0% ({revenue_growth*100:.1f}%)")
        else:
            revenue_display = f"{revenue_growth*100:.1f}%" if revenue_growth is not None else "N/A"
            criteria.append(f"❌ Revenue Growth > 0% ({revenue_display})")
        
        # 9. Positive net income (current year)
        net_income_positive = info.get('netIncomeToCommon', 0) > 0
        if net_income_positive:
            score += 1
            criteria.append("✅ Positive Net Income")
        else:
            criteria.append("❌ Positive Net Income")
        
        # 10. Dividend paying (bonus point)
        dividend_yield = info.get('dividendYield', 0)
        if dividend_yield and dividend_yield > 0:
            score += 1
            criteria.append(f"✅ Dividend Paying ({dividend_yield*100:.2f}%)")
        else:
            criteria.append("❌ Dividend Paying (0.0%)")
        
        return {
            'score': score,
            'total_possible': total_possible,
            'percentage': (score / total_possible) * 100,
            'criteria': criteria,
            'grade': get_graham_grade(score),
            'interpretation': get_graham_interpretation(score)
        }
        
    except Exception as e:
        logger.error(f"Graham score calculation error: {e}")
        return {'score': 0, 'total_possible': 10, 'criteria': [], 'error': f'Calculation error: {str(e)}'}

@safe_calculation_wrapper
def calculate_piotroski_score(symbol, show_debug=False):
    """Calculate Piotroski F-Score (0-9 points)"""
    try:
        if show_debug:
            st.write(f"📊 Calculating Piotroski F-Score for {symbol}...")
            
        ticker = yf.Ticker(symbol)
        info = ticker.info
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow
        
        if len(financials.columns) < 2 or len(balance_sheet.columns) < 2:
            return {'score': 0, 'total_possible': 9, 'criteria': [], 'error': 'Need at least 2 years of financial data'}
        
        score = 0
        total_possible = 9
        criteria = []
        
        # Get current and previous year data
        current_year = financials.iloc[:, 0]
        previous_year = financials.iloc[:, 1]
        current_bs = balance_sheet.iloc[:, 0]
        previous_bs = balance_sheet.iloc[:, 1]
        
        # PROFITABILITY CRITERIA (4 points)
        
        # 1. Positive Net Income
        net_income = current_year.get('Net Income', 0)
        if net_income > 0:
            score += 1
            criteria.append(f"✅ Positive Net Income (${net_income/1e9:.2f}B)")
        else:
            criteria.append(f"❌ Positive Net Income (${net_income/1e9:.2f}B)")
        
        # 2. Positive Operating Cash Flow
        try:
            if len(cashflow.columns) > 0:
                operating_cf = cashflow.iloc[:, 0].get('Operating Cash Flow', 0)
                if operating_cf > 0:
                    score += 1
                    criteria.append(f"✅ Positive Operating CF (${operating_cf/1e9:.2f}B)")
                else:
                    criteria.append(f"❌ Positive Operating CF (${operating_cf/1e9:.2f}B)")
            else:
                criteria.append("❌ Positive Operating CF (N/A)")
        except:
            criteria.append("❌ Positive Operating CF (N/A)")
        
        # 3. ROA Improvement (Return on Assets)
        try:
            # Calculate ROA = Net Income / Total Assets
            current_assets = current_bs.get('Total Assets', 1)
            previous_assets = previous_bs.get('Total Assets', 1)
            prev_net_income = previous_year.get('Net Income', 0)
            
            current_roa = net_income / current_assets if current_assets != 0 else 0
            previous_roa = prev_net_income / previous_assets if previous_assets != 0 else 0
            
            if current_roa > previous_roa:
                score += 1
                criteria.append(f"✅ ROA Improved ({current_roa*100:.2f}% vs {previous_roa*100:.2f}%)")
            else:
                criteria.append(f"❌ ROA Improved ({current_roa*100:.2f}% vs {previous_roa*100:.2f}%)")
        except:
            criteria.append("❌ ROA Improved (N/A)")
        
        # 4. Operating Cash Flow > Net Income (Quality of Earnings)
        try:
            if len(cashflow.columns) > 0:
                operating_cf = cashflow.iloc[:, 0].get('Operating Cash Flow', 0)
                if operating_cf > net_income:
                    score += 1
                    criteria.append("✅ Operating CF > Net Income")
                else:
                    criteria.append("❌ Operating CF > Net Income")
            else:
                criteria.append("❌ Operating CF > Net Income (N/A)")
        except:
            criteria.append("❌ Operating CF > Net Income (N/A)")
        
        # LEVERAGE/LIQUIDITY CRITERIA (3 points)
        
        # 5. Decrease in Debt-to-Assets ratio
        try:
            current_debt = current_bs.get('Total Debt', current_bs.get('Long Term Debt', 0))
            previous_debt = previous_bs.get('Total Debt', previous_bs.get('Long Term Debt', 0))
            
            current_debt_ratio = current_debt / current_assets if current_assets != 0 else 0
            previous_debt_ratio = previous_debt / previous_assets if previous_assets != 0 else 0
            
            if current_debt_ratio < previous_debt_ratio:
                score += 1
                criteria.append(f"✅ Debt Ratio Decreased ({current_debt_ratio*100:.1f}% vs {previous_debt_ratio*100:.1f}%)")
            else:
                criteria.append(f"❌ Debt Ratio Decreased ({current_debt_ratio*100:.1f}% vs {previous_debt_ratio*100:.1f}%)")
        except:
            criteria.append("❌ Debt Ratio Decreased (N/A)")
        
        # 6. Increase in Current Ratio
        try:
            current_current_assets = current_bs.get('Current Assets', 0)
            current_current_liab = current_bs.get('Current Liabilities', 1)
            prev_current_assets = previous_bs.get('Current Assets', 0)
            prev_current_liab = previous_bs.get('Current Liabilities', 1)
            
            current_ratio_now = current_current_assets / current_current_liab
            current_ratio_prev = prev_current_assets / prev_current_liab
            
            if current_ratio_now > current_ratio_prev:
                score += 1
                criteria.append(f"✅ Current Ratio Increased ({current_ratio_now:.2f} vs {current_ratio_prev:.2f})")
            else:
                criteria.append(f"❌ Current Ratio Increased ({current_ratio_now:.2f} vs {current_ratio_prev:.2f})")
        except:
            criteria.append("❌ Current Ratio Increased (N/A)")
        
        # 7. No Share Dilution (shares outstanding didn't increase)
        try:
            current_shares = info.get('sharesOutstanding', info.get('impliedSharesOutstanding', 0))
            # For previous shares, we'll use a proxy or skip if not available
            # This is often hard to get from yfinance, so we'll be conservative
            shares_metric_available = False  # Set to False since historical shares data is limited
            
            if shares_metric_available:
                # Would compare current_shares vs previous_shares here
                pass
            else:
                # Conservative approach - check if company has been buying back shares recently
                shares_change = info.get('netSharesPurchased', 0)  # This is often not available
                if shares_change <= 0:  # No dilution or buybacks
                    score += 1
                    criteria.append("✅ No Share Dilution")
                else:
                    criteria.append("❌ No Share Dilution")
        except:
            # Default to neutral/conservative scoring
            criteria.append("⚪ No Share Dilution (N/A - Assumed Neutral)")
        
        # OPERATING EFFICIENCY CRITERIA (2 points)
        
        # 8. Increase in Gross Margin
        try:
            current_revenue = current_year.get('Total Revenue', 0)
            current_gross_profit = current_year.get('Gross Profit', 0)
            prev_revenue = previous_year.get('Total Revenue', 0)
            prev_gross_profit = previous_year.get('Gross Profit', 0)
            
            current_gross_margin = current_gross_profit / current_revenue if current_revenue != 0 else 0
            prev_gross_margin = prev_gross_profit / prev_revenue if prev_revenue != 0 else 0
            
            if current_gross_margin > prev_gross_margin:
                score += 1
                criteria.append(f"✅ Gross Margin Increased ({current_gross_margin*100:.1f}% vs {prev_gross_margin*100:.1f}%)")
            else:
                criteria.append(f"❌ Gross Margin Increased ({current_gross_margin*100:.1f}% vs {prev_gross_margin*100:.1f}%)")
        except:
            criteria.append("❌ Gross Margin Increased (N/A)")
        
        # 9. Increase in Asset Turnover Ratio
        try:
            current_asset_turnover = current_revenue / current_assets if current_assets != 0 else 0
            prev_asset_turnover = prev_revenue / previous_assets if previous_assets != 0 else 0
            
            if current_asset_turnover > prev_asset_turnover:
                score += 1
                criteria.append(f"✅ Asset Turnover Increased ({current_asset_turnover:.2f} vs {prev_asset_turnover:.2f})")
            else:
                criteria.append(f"❌ Asset Turnover Increased ({current_asset_turnover:.2f} vs {prev_asset_turnover:.2f})")
        except:
            criteria.append("❌ Asset Turnover Increased (N/A)")
        
        return {
            'score': score,
            'total_possible': total_possible,
            'percentage': (score / total_possible) * 100,
            'criteria': criteria,
            'grade': get_piotroski_grade(score),
            'interpretation': get_piotroski_interpretation(score)
        }
        
    except Exception as e:
        logger.error(f"Piotroski score calculation error: {e}")
        return {'score': 0, 'total_possible': 9, 'criteria': [], 'error': f'Calculation error: {str(e)}'}

def get_graham_grade(score):
    """Convert Graham score to letter grade"""
    percentage = (score / 10) * 100
    if percentage >= 80: return "A"
    elif percentage >= 70: return "B"
    elif percentage >= 60: return "C"
    elif percentage >= 50: return "D"
    else: return "F"

def get_graham_interpretation(score):
    """Interpret Graham score"""
    if score >= 8: return "Excellent value investment candidate"
    elif score >= 6: return "Good value investment potential"
    elif score >= 4: return "Moderate value investment appeal"
    elif score >= 2: return "Limited value investment appeal"
    else: return "Poor value investment candidate"

def get_piotroski_grade(score):
    """Convert Piotroski score to letter grade"""
    if score >= 8: return "A"
    elif score >= 7: return "B+"
    elif score >= 6: return "B"
    elif score >= 5: return "B-"
    elif score >= 4: return "C"
    elif score >= 3: return "D+"
    elif score >= 2: return "D"
    else: return "F"

def get_piotroski_interpretation(score):
    """Interpret Piotroski F-Score"""
    if score >= 8: return "Very strong fundamental quality"
    elif score >= 6: return "Strong fundamental quality" 
    elif score >= 4: return "Average fundamental quality"
    elif score >= 2: return "Weak fundamental quality"
    else: return "Very weak fundamental quality"

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

        # Data quality check
        quality_check = DataQualityChecker.validate_market_data(clean_data)
        
        if not quality_check['is_acceptable']:
            st.warning(f"⚠️ Data quality issues detected for {symbol}: {quality_check['issues']}")

        if show_debug:
            st.success(f"✅ Data ready: {clean_data.shape} | Quality Score: {quality_check['quality_score']}")
        else:
            st.success(f"✅ Data loaded: {len(clean_data)} periods | Quality: {quality_check['quality_score']}/100")

        return clean_data

    except Exception as e:
        st.error(f"❌ Error fetching {symbol}: {str(e)}")
        return None

@safe_calculation_wrapper
def safe_rsi(prices, period=14):
    """Safe RSI calculation with proper error handling"""
    try:
        if len(prices) < period + 1:
            return pd.Series([50] * len(prices), index=prices.index)
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)
        return rsi
    except Exception as e:
        logger.error(f"RSI calculation error: {e}")
        return pd.Series([50] * len(prices), index=prices.index)

@safe_calculation_wrapper
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

@safe_calculation_wrapper
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

@safe_calculation_wrapper
def calculate_point_of_control_enhanced(data):
    """Enhanced Point of Control with better volume weighting"""
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

        # Calculate volume profile with better weighting
        volume_profile = {}

        for idx, row in recent_data.iterrows():
            total_volume = row['Volume']
            open_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            
            # Determine if it's a bullish or bearish bar
            is_bullish = close_price >= open_price
            
            # Enhanced volume distribution weighting
            if is_bullish:
                # For bullish bars: more weight to close and high
                price_weights = {
                    open_price: 0.15,   # 15%
                    high_price: 0.30,   # 30%
                    low_price: 0.10,    # 10%
                    close_price: 0.45   # 45%
                }
            else:
                # For bearish bars: more weight to close and low
                price_weights = {
                    open_price: 0.15,   # 15%
                    high_price: 0.10,   # 10%
                    low_price: 0.30,    # 30%
                    close_price: 0.45   # 45%
                }
            
            # Distribute volume according to weights
            for price, weight in price_weights.items():
                bin_key = round(price / bin_size) * bin_size
                volume_profile[bin_key] = volume_profile.get(bin_key, 0) + (total_volume * weight)

        # Find POC (price with highest volume)
        if volume_profile:
            poc_price = max(volume_profile, key=volume_profile.get)
            return round(float(poc_price), 2)
        else:
            return float(recent_data['Close'].iloc[-1])

    except Exception as e:
        logger.error(f"Enhanced POC calculation error: {e}")
        try:
            return float(data['Close'].iloc[-1])
        except:
            return 0.0

@safe_calculation_wrapper
def calculate_comprehensive_technicals(data):
    """Calculate comprehensive technical indicators for individual symbol analysis"""
    try:
        if len(data) < 50:
            return {}

        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']

        # Previous week high/low
        week_data = data.tail(5)  # Last 5 trading days
        prev_week_high = week_data['High'].max()
        prev_week_low = week_data['Low'].min()

        # RSI (14-period)
        rsi_14 = safe_rsi(close, 14).iloc[-1]

        # Money Flow Index (MFI)
        mfi_14 = calculate_mfi(data, 14)

        # MACD (12, 26, 9)
        macd_data = calculate_macd(close, 12, 26, 9)

        # Average True Range (ATR)
        atr_14 = calculate_atr(data, 14)

        # Bollinger Bands (20, 2)
        bb_data = calculate_bollinger_bands(close, 20, 2)

        # Stochastic Oscillator
        stoch_data = calculate_stochastic(data, 14, 3)

        # Williams %R
        williams_r = calculate_williams_r(data, 14)

        # Volume metrics
        volume_sma_20 = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
        current_volume = volume.iloc[-1]
        volume_ratio = (current_volume / volume_sma_20) if volume_sma_20 > 0 else 1

        # Price metrics
        current_price = close.iloc[-1]
        price_change_1d = ((current_price - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) > 1 else 0
        price_change_5d = ((current_price - close.iloc[-6]) / close.iloc[-6] * 100) if len(close) > 5 else 0

        # Volatility (20-day)
        returns = close.pct_change().dropna()
        if len(returns) >= 20:
            volatility_20d = returns.rolling(20).std().iloc[-1] * (252 ** 0.5) * 100  # Annualized
        else:
            volatility_20d = returns.std() * (252 ** 0.5) * 100 if len(returns) > 0 else 20

        return {
            'prev_week_high': round(float(prev_week_high), 2),
            'prev_week_low': round(float(prev_week_low), 2),
            'rsi_14': round(float(rsi_14), 2),
            'mfi_14': round(float(mfi_14), 2),
            'macd': macd_data,
            'atr_14': round(float(atr_14), 2),
            'bollinger_bands': bb_data,
            'stochastic': stoch_data,
            'williams_r': williams_r,
            'volume_sma_20': round(float(volume_sma_20), 0),
            'current_volume': round(float(current_volume), 0),
            'volume_ratio': round(float(volume_ratio), 2),
            'price_change_1d': round(float(price_change_1d), 2),
            'price_change_5d': round(float(price_change_5d), 2),
            'volatility_20d': round(float(volatility_20d), 2)
        }

    except Exception as e:
        logger.error(f"Comprehensive technicals calculation error: {e}")
        return {}

@safe_calculation_wrapper
def calculate_mfi(data, period=14):
    """Calculate Money Flow Index"""
    try:
        if len(data) < period + 1:
            return 50.0
            
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']

        # Positive and negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()

        mfi = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, np.inf)))
        return float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 50.0
    except Exception as e:
        logger.error(f"MFI calculation error: {e}")
        return 50.0

@safe_calculation_wrapper
def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    try:
        if len(close) < slow:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
            
        ema_fast = close.ewm(span=fast).mean()
        ema_slow = close.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return {
            'macd': round(float(macd_line.iloc[-1]), 4),
            'signal': round(float(signal_line.iloc[-1]), 4),
            'histogram': round(float(histogram.iloc[-1]), 4)
        }
    except Exception as e:
        logger.error(f"MACD calculation error: {e}")
        return {'macd': 0, 'signal': 0, 'histogram': 0}

@safe_calculation_wrapper
def calculate_atr(data, period=14):
    """Calculate Average True Range"""
    try:
        if len(data) < period + 1:
            return 0.0
            
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift(1)).abs()
        low_close = (data['Low'] - data['Close'].shift(1)).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()

        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0
    except Exception as e:
        logger.error(f"ATR calculation error: {e}")
        return 0.0

@safe_calculation_wrapper
def calculate_bollinger_bands(close, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    try:
        if len(close) < period:
            current_close = float(close.iloc[-1])
            return {
                'upper': current_close * 1.02,
                'middle': current_close,
                'lower': current_close * 0.98,
                'position': 50
            }
            
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        current_close = close.iloc[-1]
        
        upper_val = upper_band.iloc[-1]
        lower_val = lower_band.iloc[-1]
        
        if upper_val != lower_val:
            bb_position = ((current_close - lower_val) / (upper_val - lower_val)) * 100
        else:
            bb_position = 50

        return {
            'upper': round(float(upper_val), 2),
            'middle': round(float(sma.iloc[-1]), 2),
            'lower': round(float(lower_val), 2),
            'position': round(float(bb_position), 1)
        }
    except Exception as e:
        logger.error(f"Bollinger Bands calculation error: {e}")
        return {'upper': 0, 'middle': 0, 'lower': 0, 'position': 50}

@safe_calculation_wrapper
def calculate_stochastic(data, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    try:
        if len(data) < k_period:
            return {'k': 50, 'd': 50}
            
        lowest_low = data['Low'].rolling(k_period).min()
        highest_high = data['High'].rolling(k_period).max()

        k_percent = ((data['Close'] - lowest_low) / (highest_high - lowest_low)) * 100
        d_percent = k_percent.rolling(d_period).mean()

        return {
            'k': round(float(k_percent.iloc[-1]), 2),
            'd': round(float(d_percent.iloc[-1]), 2)
        }
    except Exception as e:
        logger.error(f"Stochastic calculation error: {e}")
        return {'k': 50, 'd': 50}

@safe_calculation_wrapper
def calculate_williams_r(data, period=14):
    """Calculate Williams %R"""
    try:
        if len(data) < period:
            return -50.0
            
        highest_high = data['High'].rolling(period).max()
        lowest_low = data['Low'].rolling(period).min()

        williams_r = ((highest_high - data['Close']) / (highest_high - lowest_low)) * -100
        return float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else -50.0
    except Exception as e:
        logger.error(f"Williams %R calculation error: {e}")
        return -50.0

# Initialize correlation data cache
if 'correlation_cache' not in st.session_state:
    st.session_state.correlation_cache = {}

@st.cache_data(ttl=600)  # 10-minute cache for correlation data
def get_correlation_etf_data(etf_symbols, period='1y'):
    """Cached function to fetch correlation ETF data"""
    etf_data = {}
    
    for etf in etf_symbols:
        try:
            etf_ticker = yf.Ticker(etf)
            etf_history = etf_ticker.history(period=period)
            
            if len(etf_history) > 50:
                etf_returns = etf_history['Close'].pct_change().dropna()
                etf_data[etf] = etf_returns
                
        except Exception as e:
            logger.error(f"Error fetching {etf} data: {e}")
            continue
    
    return etf_data

@safe_calculation_wrapper
def calculate_market_correlations_enhanced(symbol_data, symbol, period='1y', show_debug=False):
    """Enhanced market correlations with caching to avoid redundant API calls"""
    try:
        comparison_etfs = ['FNGD', 'FNGU', 'MAGS']
        correlations = {}

        if show_debug:
            st.write(f"📊 Calculating correlations for {symbol}...")

        # Get symbol returns
        symbol_returns = symbol_data['Close'].pct_change().dropna()
        
        # Get cached ETF data
        etf_data = get_correlation_etf_data(comparison_etfs, period)

        for etf in comparison_etfs:
            try:
                if etf not in etf_data:
                    correlations[etf] = {'correlation': 0, 'beta': 0, 'relationship': 'No data available'}
                    continue
                
                etf_returns = etf_data[etf]
                
                # Align dates
                aligned_data = pd.concat([symbol_returns, etf_returns], axis=1, join='inner')
                aligned_data.columns = [symbol, etf]

                if len(aligned_data) > 30:
                    correlation = aligned_data[symbol].corr(aligned_data[etf])

                    # Calculate beta
                    covariance = aligned_data[symbol].cov(aligned_data[etf])
                    etf_variance = aligned_data[etf].var()
                    beta = covariance / etf_variance if etf_variance != 0 else 0

                    correlations[etf] = {
                        'correlation': round(float(correlation), 3),
                        'beta': round(float(beta), 3),
                        'relationship': get_correlation_description(correlation)
                    }

                    if show_debug:
                        st.write(f"  • {etf}: {correlation:.3f} correlation")
                else:
                    correlations[etf] = {'correlation': 0, 'beta': 0, 'relationship': 'Insufficient data'}

            except Exception as e:
                correlations[etf] = {'correlation': 0, 'beta': 0, 'relationship': f'Error: {str(e)[:20]}...'}
                if show_debug:
                    st.write(f"  • {etf}: Error - {str(e)}")

        return correlations

    except Exception as e:
        if show_debug:
            st.write(f"❌ Correlation calculation error: {str(e)}")
        return {}

def get_correlation_description(corr):
    """Get description of correlation strength"""
    abs_corr = abs(corr)
    if abs_corr >= 0.8:
        return "Very Strong"
    elif abs_corr >= 0.6:
        return "Strong"
    elif abs_corr >= 0.4:
        return "Moderate"
    elif abs_corr >= 0.2:
        return "Weak"
    else:
        return "Very Weak"

@safe_calculation_wrapper
def calculate_breakout_breakdown_analysis(show_debug=False):
    """Calculate breakout/breakdown ratios for major indices"""
    try:
        indices = ['SPY', 'QQQ', 'IWM']
        results = {}
        
        for index in indices:
            try:
                if show_debug:
                    st.write(f"📊 Analyzing breakouts/breakdowns for {index}...")
                
                # Get recent data (3 months for reliable signals)
                ticker = yf.Ticker(index)
                data = ticker.history(period='3mo')
                
                if len(data) < 50:
                    continue
                    
                current_price = data['Close'].iloc[-1]
                
                # Multi-timeframe resistance levels
                resistance_10 = data['High'].rolling(10).max().iloc[-2]   # 10-day high
                resistance_20 = data['High'].rolling(20).max().iloc[-2]   # 20-day high
                resistance_50 = data['High'].rolling(50).max().iloc[-2]   # 50-day high
                
                # Multi-timeframe support levels  
                support_10 = data['Low'].rolling(10).min().iloc[-2]       # 10-day low
                support_20 = data['Low'].rolling(20).min().iloc[-2]       # 20-day low
                support_50 = data['Low'].rolling(50).min().iloc[-2]       # 50-day low
                
                # Breakout signals (price above resistance)
                breakout_10 = 1 if current_price > resistance_10 else 0
                breakout_20 = 1 if current_price > resistance_20 else 0
                breakout_50 = 1 if current_price > resistance_50 else 0
                
                # Breakdown signals (price below support)
                breakdown_10 = 1 if current_price < support_10 else 0
                breakdown_20 = 1 if current_price < support_20 else 0
                breakdown_50 = 1 if current_price < support_50 else 0
                
                # Calculate ratios
                total_breakouts = breakout_10 + breakout_20 + breakout_50
                total_breakdowns = breakdown_10 + breakdown_20 + breakdown_50
                
                breakout_ratio = (total_breakouts / 3) * 100  # Percentage of timeframes showing breakout
                breakdown_ratio = (total_breakdowns / 3) * 100  # Percentage of timeframes showing breakdown
                net_ratio = breakout_ratio - breakdown_ratio  # Net bias
                
                # Determine overall signal strength
                if net_ratio > 66:
                    signal_strength = "Very Bullish"
                elif net_ratio > 33:
                    signal_strength = "Bullish" 
                elif net_ratio > -33:
                    signal_strength = "Neutral"
                elif net_ratio > -66:
                    signal_strength = "Bearish"
                else:
                    signal_strength = "Very Bearish"
                
                results[index] = {
                    'current_price': round(current_price, 2),
                    'breakout_ratio': round(breakout_ratio, 1),
                    'breakdown_ratio': round(breakdown_ratio, 1),
                    'net_ratio': round(net_ratio, 1),
                    'signal_strength': signal_strength,
                    'breakout_levels': {
                        '10d': round(resistance_10, 2),
                        '20d': round(resistance_20, 2), 
                        '50d': round(resistance_50, 2)
                    },
                    'breakdown_levels': {
                        '10d': round(support_10, 2),
                        '20d': round(support_20, 2),
                        '50d': round(support_50, 2)
                    },
                    'active_breakouts': [f"{days}d" for days, signal in 
                                       [('10', breakout_10), ('20', breakout_20), ('50', breakout_50)] if signal],
                    'active_breakdowns': [f"{days}d" for days, signal in 
                                        [('10', breakdown_10), ('20', breakdown_20), ('50', breakdown_50)] if signal]
                }
                
            except Exception as e:
                if show_debug:
                    st.write(f"❌ Error analyzing {index}: {e}")
                continue
        
        # Calculate overall market sentiment
        if results:
            overall_breakout = sum([results[idx]['breakout_ratio'] for idx in results]) / len(results)
            overall_breakdown = sum([results[idx]['breakdown_ratio'] for idx in results]) / len(results)
            overall_net = overall_breakout - overall_breakdown
            
            # Market regime classification
            if overall_net > 50:
                market_regime = "🚀 Strong Breakout Environment"
            elif overall_net > 20:
                market_regime = "📈 Bullish Breakout Bias"
            elif overall_net > -20:
                market_regime = "⚖️ Balanced Market"
            elif overall_net > -50:
                market_regime = "📉 Bearish Breakdown Bias"
            else:
                market_regime = "🔻 Strong Breakdown Environment"
            
            results['OVERALL'] = {
                'breakout_ratio': round(overall_breakout, 1),
                'breakdown_ratio': round(overall_breakdown, 1), 
                'net_ratio': round(overall_net, 1),
                'market_regime': market_regime,
                'sample_size': len(results)
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Breakout/breakdown analysis error: {e}")
        return {}

def calculate_composite_technical_score(analysis_results):
    """Calculate composite technical score from all indicators (1-100)"""
    try:
        enhanced_indicators = analysis_results.get('enhanced_indicators', {})
        comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
        current_price = analysis_results['current_price']
        
        scores = []
        weights = []
        
        # 1. PRICE POSITION ANALYSIS (35% total weight)
        daily_vwap = enhanced_indicators.get('daily_vwap', current_price)
        poc = enhanced_indicators.get('point_of_control', current_price)
        fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
        
        # VWAP position (10% weight)
        vwap_score = 75 if current_price > daily_vwap else 25
        scores.append(vwap_score)
        weights.append(0.10)
        
        # Point of Control position (10% weight) 
        poc_score = 75 if current_price > poc else 25
        scores.append(poc_score)
        weights.append(0.10)
        
        # EMA confluence analysis (15% weight)
        if fibonacci_emas:
            ema_above_count = sum(1 for ema_value in fibonacci_emas.values() if current_price > ema_value)
            ema_confluence_score = (ema_above_count / len(fibonacci_emas)) * 100
            scores.append(ema_confluence_score)
            weights.append(0.15)
        
        # 2. MOMENTUM OSCILLATORS (30% total weight)
        rsi = comprehensive_technicals.get('rsi_14', 50)
        mfi = comprehensive_technicals.get('mfi_14', 50)
        williams_r = comprehensive_technicals.get('williams_r', -50)
        stoch_data = comprehensive_technicals.get('stochastic', {})
        stoch_k = stoch_data.get('k', 50)
        
        # RSI scoring (oversold favored in bottom-picking system)
        if rsi < 25:
            rsi_score = 90  # Very oversold - very bullish
        elif rsi < 35:
            rsi_score = 75  # Oversold - bullish
        elif rsi > 75:
            rsi_score = 10  # Very overbought - very bearish
        elif rsi > 65:
            rsi_score = 25  # Overbought - bearish
        else:
            rsi_score = 50 + (50 - rsi) * 0.3  # Neutral zone with slight contrarian bias
        
        scores.append(rsi_score)
        weights.append(0.12)
        
        # MFI scoring (money flow consideration)
        if mfi < 20:
            mfi_score = 85
        elif mfi > 80:
            mfi_score = 15
        else:
            mfi_score = 50 + (50 - mfi) * 0.4
        
        scores.append(mfi_score)
        weights.append(0.08)
        
        # Williams %R scoring (convert to 0-100 scale)
        williams_normalized = ((williams_r + 100) / 100) * 100  # Convert -100:0 to 0:100
        scores.append(williams_normalized)
        weights.append(0.05)
        
        # Stochastic scoring
        if stoch_k < 20:
            stoch_score = 85
        elif stoch_k > 80:
            stoch_score = 15
        else:
            stoch_score = stoch_k
        
        scores.append(stoch_score)
        weights.append(0.05)
        
        # 3. VOLUME ANALYSIS (15% weight)
        volume_ratio = comprehensive_technicals.get('volume_ratio', 1)
        if volume_ratio > 2.0:
            volume_score = 85  # Very high volume
        elif volume_ratio > 1.5:
            volume_score = 70  # High volume
        elif volume_ratio < 0.3:
            volume_score = 15  # Very low volume
        elif volume_ratio < 0.7:
            volume_score = 30  # Low volume
        else:
            volume_score = 50 + (volume_ratio - 1) * 20  # Neutral zone
        
        scores.append(max(10, min(90, volume_score)))  # Cap extreme values
        weights.append(0.15)
        
        # 4. TREND ANALYSIS (20% weight)
        macd_data = comprehensive_technicals.get('macd', {})
        histogram = macd_data.get('histogram', 0)
        
        # MACD Histogram trend
        if histogram > 0:
            macd_score = 70 + min(histogram * 1000, 20)  # Bullish with strength adjustment
        elif histogram < 0:
            macd_score = 30 + max(histogram * 1000, -20)  # Bearish with strength adjustment
        else:
            macd_score = 50
        
        scores.append(max(5, min(95, macd_score)))
        weights.append(0.10)
        
        # Previous week support/resistance analysis
        prev_week_high = comprehensive_technicals.get('prev_week_high', current_price)
        prev_week_low = comprehensive_technicals.get('prev_week_low', current_price)
        
        if current_price > prev_week_high:
            breakout_score = 85  # Above resistance - bullish breakout
        elif current_price < prev_week_low:
            breakout_score = 15  # Below support - bearish breakdown
        else:
            # Within range - score based on position
            range_size = prev_week_high - prev_week_low
            if range_size > 0:
                position_in_range = (current_price - prev_week_low) / range_size
                breakout_score = 20 + (position_in_range * 60)  # 20-80 range
            else:
                breakout_score = 50
        
        scores.append(breakout_score)
        weights.append(0.10)
        
        # Calculate weighted composite score
        if len(scores) == len(weights) and sum(weights) > 0:
            composite_score = sum(score * weight for score, weight in zip(scores, weights)) / sum(weights)
        else:
            composite_score = 50  # Default neutral
        
        # Ensure score is within bounds and add some smoothing
        final_score = max(1, min(100, round(composite_score, 1)))
        
        return final_score, {
            'component_scores': {
                'vwap_position': round(scores[0], 1) if len(scores) > 0 else 50,
                'poc_position': round(scores[1], 1) if len(scores) > 1 else 50, 
                'ema_confluence': round(scores[2], 1) if len(scores) > 2 else 50,
                'rsi_momentum': round(scores[3], 1) if len(scores) > 3 else 50,
                'volume_strength': round(scores[5], 1) if len(scores) > 5 else 50,
                'trend_direction': round(scores[6], 1) if len(scores) > 6 else 50
            },
            'total_components': len(scores),
            'weight_distribution': dict(zip(['vwap', 'poc', 'ema', 'rsi', 'mfi', 'williams', 'stoch', 'volume', 'macd', 'breakout'], weights))
        }
        
    except Exception as e:
        logger.error(f"Composite technical score calculation error: {e}")
        return 50.0, {'error': str(e)}

def create_technical_score_bar(score, details=None):
    """Create professional gradient bar for technical score"""
    
    # Determine interpretation and color
    if score >= 80:
        interpretation = "Very Bullish"
        primary_color = "#00A86B"  # Jade green
    elif score >= 65:
        interpretation = "Bullish" 
        primary_color = "#32CD32"  # Lime green
    elif score >= 55:
        interpretation = "Slightly Bullish"
        primary_color = "#9ACD32"  # Yellow green
    elif score >= 45:
        interpretation = "Neutral"
        primary_color = "#FFD700"  # Gold
    elif score >= 35:
        interpretation = "Slightly Bearish"
        primary_color = "#FF8C00"  # Dark orange
    elif score >= 20:
        interpretation = "Bearish"
        primary_color = "#FF4500"  # Orange red
    else:
        interpretation = "Very Bearish"
        primary_color = "#DC143C"  # Crimson
    
    # Create professional gradient bar HTML
    html = f"""
    <div style="margin: 1.5rem 0; padding: 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                border-radius: 12px; border: 1px solid #dee2e6; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div>
                <span style="font-weight: 700; color: #495057; font-size: 1.2em;">Technical Composite Score</span>
                <div style="font-size: 0.9em; color: #6c757d; margin-top: 0.2rem;">
                    Aggregated signal from all technical indicators
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-weight: 700; color: {primary_color}; font-size: 2em;">{score}</div>
                <div style="font-size: 0.9em; color: {primary_color}; font-weight: 600;">{interpretation}</div>
            </div>
        </div>
        
        <div style="position: relative; width: 100%; height: 24px; background: linear-gradient(to right, 
                    #DC143C 0%, #FF4500 15%, #FF8C00 30%, #FFD700 50%, 
                    #9ACD32 70%, #32CD32 85%, #00A86B 100%); 
                    border-radius: 12px; border: 1px solid #ced4da; overflow: hidden;">
            
            <!-- Score indicator -->
            <div style="position: absolute; left: {score}%; top: 50%; transform: translate(-50%, -50%); 
                        width: 6px; height: 30px; background: white; border: 2px solid #343a40; 
                        border-radius: 3px; box-shadow: 0 2px 4px rgba(0,0,0,0.3); z-index: 10;">
            </div>
            
            <!-- Progress fill -->
            <div style="position: absolute; left: 0; top: 0; height: 100%; width: {score}%; 
                        background: linear-gradient(to right, transparent 0%, {primary_color} 100%); 
                        opacity: 0.3; border-radius: 12px;">
            </div>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.75em; color: #6c757d;">
            <span style="font-weight: 600;">Very Bearish</span>
            <span style="font-weight: 600;">Neutral</span>
            <span style="font-weight: 600;">Very Bullish</span>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 0.2rem; font-size: 0.7em; color: #adb5bd;">
            <span>1</span>
            <span>25</span>
            <span>50</span>
            <span>75</span>
            <span>100</span>
        </div>
    </div>
    """
    
    return html

@safe_calculation_wrapper
def calculate_options_levels_enhanced(current_price, volatility, days_to_expiry=[7, 14, 30, 45], risk_free_rate=0.05, underlying_beta=1.0):
    """Enhanced options levels with proper Black-Scholes approximation and Greeks"""
    try:
        from scipy.stats import norm
        import math
        
        options_data = []

        for dte in days_to_expiry:
            T = dte / 365.0  # Time to expiration in years
            vol_annual = volatility / 100.0  # Convert percentage to decimal
            
            # For ~16 delta (0.16), use inverse normal distribution
            delta_16 = 0.16
            z_score = norm.ppf(delta_16)  # ≈ -0.994
            
            # More accurate strike calculation using Black-Scholes framework
            drift = (risk_free_rate - 0.5 * vol_annual**2) * T
            vol_term = vol_annual * math.sqrt(T)
            
            # Put strike (16 delta put)
            put_strike = current_price * math.exp(drift + z_score * vol_term)
            
            # Call strike (16 delta call - using positive z-score)
            call_strike = current_price * math.exp(drift - z_score * vol_term)
            
            # Probability of Touch (more accurate)
            prob_touch_put = 2 * norm.cdf(z_score) * 100
            prob_touch_call = 2 * (1 - norm.cdf(-z_score)) * 100
            
            # Expected move (1 standard deviation)
            expected_move = current_price * vol_annual * math.sqrt(T)
            
            # Calculate Greeks
            # Delta calculation (approximate for 16-delta options)
            put_delta = -0.16  # Put delta is negative
            call_delta = 0.16   # Call delta is positive
            
            # Theta calculation (time decay per day)
            # Simplified theta estimation: higher for ATM, lower for OTM
            put_moneyness = put_strike / current_price
            call_moneyness = call_strike / current_price
            
            # Theta increases as expiration approaches and decreases for OTM options
            time_factor = math.sqrt(T)
            
            # Simplified theta calculation (option value / days remaining * time decay factor)
            put_theta = -(current_price * vol_annual * 0.4 * put_moneyness) / math.sqrt(dte) if dte > 0 else 0
            call_theta = -(current_price * vol_annual * 0.4 * call_moneyness) / math.sqrt(dte) if dte > 0 else 0
            
            # Beta (underlying's market beta - same for all options on same underlying)
            option_beta = underlying_beta

            options_data.append({
                'DTE': dte,
                'Put Strike': round(put_strike, 2),
                'Put PoT': f"{prob_touch_put:.1f}%",
                'Put Delta': f"{put_delta:.2f}",
                'Put Theta': f"{put_theta:.2f}",
                'Call Strike': round(call_strike, 2),
                'Call PoT': f"{prob_touch_call:.1f}%", 
                'Call Delta': f"{call_delta:.2f}",
                'Call Theta': f"{call_theta:.2f}",
                'Beta': f"{option_beta:.2f}",
                'Expected Move': f"±{expected_move:.2f}"
            })

        return options_data

    except ImportError:
        # Fallback to simplified calculation if scipy is not available
        options_data = []
        for dte in days_to_expiry:
            daily_vol = volatility / 100 / (252 ** 0.5)
            std_move = current_price * daily_vol * (dte ** 0.5)
            
            put_strike = current_price - std_move
            call_strike = current_price + std_move
            
            prob_touch_put = min(32, 32 * (std_move / current_price) * 100)
            prob_touch_call = prob_touch_put
            
            # Simplified Greeks for fallback
            put_delta = -0.16
            call_delta = 0.16
            put_theta = -(std_move * 0.1) / dte if dte > 0 else 0
            call_theta = put_theta
            option_beta = underlying_beta

            options_data.append({
                'DTE': dte,
                'Put Strike': round(put_strike, 2),
                'Put PoT': f"{prob_touch_put:.1f}%",
                'Put Delta': f"{put_delta:.2f}",
                'Put Theta': f"{put_theta:.2f}",
                'Call Strike': round(call_strike, 2),
                'Call PoT': f"{prob_touch_call:.1f}%",
                'Call Delta': f"{call_delta:.2f}",
                'Call Theta': f"{call_theta:.2f}",
                'Beta': f"{option_beta:.2f}",
                'Expected Move': f"±{std_move:.2f}"
            })
        
        return options_data
        
    except Exception as e:
        logger.error(f"Options levels calculation error: {e}")
        return []

@safe_calculation_wrapper
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

    except Exception as e:
        logger.error(f"Weekly deviations calculation error: {e}")
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

# ENHANCED: VWV Trading System with corrected Williams VIX Fix
class VWVTradingSystem:
    def __init__(self, config=None):
        """Initialize enhanced trading system"""
        default_config = {
            'wvf_period': 22,
            'wvf_multiplier': 2.0,  # Standard Bollinger Band multiplier
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

    @safe_calculation_wrapper
    def detect_market_regime(self, data):
        """Detect market regime for dynamic weight adjustment"""
        try:
            if len(data) < 50:
                return {'regime': 'NORMAL', 'volatility_regime': 'NORMAL', 'trend_regime': 'SIDEWAYS'}

            close = data['Close']
            
            # Calculate volatility regime (20-day rolling volatility)
            returns = close.pct_change().dropna()
            current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)  # Annualized
            vol_history = returns.rolling(20).std() * np.sqrt(252)
            vol_percentile = vol_history.rolling(252).rank(pct=True).iloc[-1] if len(vol_history) >= 252 else 0.5
            
            if vol_percentile > 0.8:
                volatility_regime = 'HIGH'
            elif vol_percentile < 0.2:
                volatility_regime = 'LOW'
            else:
                volatility_regime = 'NORMAL'
            
            # Calculate trend regime (EMA slopes and alignment)
            ema_20 = close.ewm(span=20).mean()
            ema_50 = close.ewm(span=50).mean()
            
            # EMA slope strength
            ema20_slope = (ema_20.iloc[-1] - ema_20.iloc[-5]) / ema_20.iloc[-5] * 100
            ema50_slope = (ema_50.iloc[-1] - ema_50.iloc[-10]) / ema_50.iloc[-10] * 100
            
            # EMA alignment
            ema_aligned = ema_20.iloc[-1] > ema_50.iloc[-1]
            
            if abs(ema20_slope) > 2 and abs(ema50_slope) > 1 and ema_aligned:
                if ema20_slope > 0:
                    trend_regime = 'STRONG_UPTREND'
                else:
                    trend_regime = 'STRONG_DOWNTREND'
            elif abs(ema20_slope) > 1:
                if ema20_slope > 0:
                    trend_regime = 'UPTREND'
                else:
                    trend_regime = 'DOWNTREND'
            else:
                trend_regime = 'SIDEWAYS'
            
            # Overall regime classification
            if volatility_regime == 'HIGH' and trend_regime in ['STRONG_UPTREND', 'STRONG_DOWNTREND']:
                overall_regime = 'TRENDING_VOLATILE'
            elif volatility_regime == 'HIGH':
                overall_regime = 'HIGH_VOLATILITY'
            elif trend_regime in ['STRONG_UPTREND', 'STRONG_DOWNTREND']:
                overall_regime = 'TRENDING'
            elif volatility_regime == 'LOW' and trend_regime == 'SIDEWAYS':
                overall_regime = 'LOW_VOLATILITY'
            else:
                overall_regime = 'NORMAL'
            
            return {
                'regime': overall_regime,
                'volatility_regime': volatility_regime,
                'trend_regime': trend_regime,
                'volatility_percentile': vol_percentile,
                'trend_strength': abs(ema20_slope)
            }
            
        except Exception as e:
            logger.error(f"Market regime detection error: {e}")
            return {'regime': 'NORMAL', 'volatility_regime': 'NORMAL', 'trend_regime': 'SIDEWAYS'}

    def get_dynamic_weights(self, market_regime):
        """Get dynamic weights based on market regime"""
        base_weights = self.weights.copy()
        
        regime = market_regime['regime']
        vol_regime = market_regime['volatility_regime']
        trend_regime = market_regime['trend_regime']
        
        # Adjust weights based on regime
        if regime == 'HIGH_VOLATILITY' or vol_regime == 'HIGH':
            # In high volatility, WVF becomes more important
            base_weights['wvf'] *= 1.3
            base_weights['volatility'] *= 1.2
            base_weights['ma'] *= 0.9
            
        elif regime == 'TRENDING' or trend_regime in ['STRONG_UPTREND', 'STRONG_DOWNTREND']:
            # In trending markets, MA confluence becomes more important
            base_weights['ma'] *= 1.4
            base_weights['momentum'] *= 1.2
            base_weights['wvf'] *= 0.8
            
        elif regime == 'LOW_VOLATILITY':
            # In low volatility, volume and VWAP become more important
            base_weights['volume'] *= 1.3
            base_weights['vwap'] *= 1.2
            base_weights['volatility'] *= 0.7
            
        elif regime == 'TRENDING_VOLATILE':
            # In trending volatile markets, balance trend and volatility indicators
            base_weights['wvf'] *= 1.2
            base_weights['ma'] *= 1.3
            base_weights['momentum'] *= 1.1
        
        return base_weights
        
    def calculate_williams_vix_fix_enhanced(self, data):
        """Enhanced Williams VIX Fix with proper binary signal logic"""
        try:
            period = self.config['wvf_period']  # Default 22
            multiplier = self.config['wvf_multiplier']  # Default 2.0
            
            if len(data) < period * 2:
                return {'binary_signal': 0, 'normalized_strength': 0.0, 'wvf_value': 0, 'upper_band': 0}

            close = data['Close']
            low = data['Low']

            # Original WVF formula: ((Highest Close - Low) / Highest Close) × 100
            highest_close = close.rolling(window=period).max()
            wvf_raw = ((highest_close - low) / highest_close) * 100

            # Apply Bollinger Band to WVF for signals
            wvf_sma = wvf_raw.rolling(window=period).mean()
            wvf_std = wvf_raw.rolling(window=period).std()
            wvf_upper_band = wvf_sma + (wvf_std * multiplier)

            # Get current values
            current_wvf = wvf_raw.iloc[-1] if not pd.isna(wvf_raw.iloc[-1]) else 0
            current_upper = wvf_upper_band.iloc
