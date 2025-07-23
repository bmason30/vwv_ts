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
    
    /* Remove bitcoin sun styling */
    
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
        # Common ETF patterns and known ETFs
        etf_suffixes = ['ETF', 'FUND']
        common_etfs = {
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'AGG', 'BND', 'TLT',
            'GLD', 'SLV', 'USO', 'UNG', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP',
            'XLY', 'XLU', 'XLRE', 'XLB', 'EFA', 'EEM', 'FXI', 'EWJ', 'EWG', 'EWU',
            'ARKK', 'ARKQ', 'ARKW', 'ARKG', 'ARKF', 'FNGU', 'FNGD', 'MAGS', 'SOXX',
            'SMH', 'IBB', 'XBI', 'JETS', 'HACK', 'ESPO', 'ICLN', 'PBW', 'KWEB'
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
            
            # Check various fields that might indicate ETF
            security_type = info.get('quoteType', '').upper()
            category = info.get('category', '').upper() 
            fund_family = info.get('fundFamily', '').upper()
            
            if 'ETF' in security_type or 'ETF' in category or 'ETF' in fund_family:
                return True
                
            # Check long name for ETF indicators
            long_name = info.get('longName', '').upper()
            short_name = info.get('shortName', '').upper()
            
            etf_keywords = ['ETF', 'EXCHANGE TRADED', 'INDEX FUND', 'TRUST']
            for keyword in etf_keywords:
                if keyword in long_name or keyword in short_name:
                    return True
                    
        except:
            # If yfinance lookup fails, use pattern matching
            pass
        
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
            criteria.append(f"❌ Current Ratio > 1.5 ({current_ratio:.2f if current_ratio else 'N/A'})")
        
        # 6. Quick Ratio > 1.0
        if quick_ratio and quick_ratio > 1.0:
            score += 1
            criteria.append(f"✅ Quick Ratio > 1.0 ({quick_ratio:.2f})")
        else:
            criteria.append(f"❌ Quick Ratio > 1.0 ({quick_ratio:.2f if quick_ratio else 'N/A'})")
        
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
            'williams_r': round(float(williams_r), 2),
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

@safe_calculation_wrapper
def calculate_market_correlations(symbol_data, symbol, period='1y', show_debug=False):
    """Calculate correlations with market ETFs"""
    try:
        comparison_etfs = ['FNGD', 'FNGU', 'MAGS']
        correlations = {}

        if show_debug:
            st.write(f"📊 Calculating correlations for {symbol}...")

        # Get symbol returns
        symbol_returns = symbol_data['Close'].pct_change().dropna()

        for etf in comparison_etfs:
            try:
                # Fetch ETF data
                etf_ticker = yf.Ticker(etf)
                etf_data = etf_ticker.history(period=period)

                if len(etf_data) > 50:
                    etf_returns = etf_data['Close'].pct_change().dropna()

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
                else:
                    correlations[etf] = {'correlation': 0, 'beta': 0, 'relationship': 'No data available'}

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
def calculate_options_levels_enhanced(current_price, volatility, days_to_expiry=[7, 14, 30, 45], risk_free_rate=0.05):
    """Enhanced options levels with proper Black-Scholes approximation"""
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
            # PoT ≈ 2 * N(d) for puts, 2 * (1 - N(-d)) for calls
            prob_touch_put = 2 * norm.cdf(z_score) * 100
            prob_touch_call = 2 * (1 - norm.cdf(-z_score)) * 100
            
            # Expected move (1 standard deviation)
            expected_move = current_price * vol_annual * math.sqrt(T)

            options_data.append({
                'DTE': dte,
                'Put Strike': round(put_strike, 2),
                'Put PoT': f"{prob_touch_put:.1f}%",
                'Call Strike': round(call_strike, 2),
                'Call PoT': f"{prob_touch_call:.1f}%",
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

            options_data.append({
                'DTE': dte,
                'Put Strike': round(put_strike, 2),
                'Put PoT': f"{prob_touch_put:.1f}%",
                'Call Strike': round(call_strike, 2),
                'Call PoT': f"{prob_touch_call:.1f}%",
                'Expected Move': f"±{std_move:.2f}"
            })
        
        return options_data
        
    except Exception as e:
        logger.error(f"Options levels calculation error: {e}")
        return []

@safe_calculation_wrapper
def calculate_position_sizing(account_balance, risk_per_trade_pct=2.0, entry_price=None, stop_loss=None):
    """Professional position sizing based on risk management principles"""
    try:
        if not entry_price or not stop_loss or account_balance <= 0:
            return None
            
        risk_amount = account_balance * (risk_per_trade_pct / 100)
        price_risk_per_share = abs(entry_price - stop_loss)
        
        if price_risk_per_share == 0:
            return None
            
        position_size = risk_amount / price_risk_per_share
        position_value = position_size * entry_price
        
        # Maximum position size (typically 10-20% of account)
        max_position_pct = 15.0
        max_position_value = account_balance * (max_position_pct / 100)
        
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
            actual_risk = position_size * price_risk_per_share
            actual_risk_pct = (actual_risk / account_balance) * 100
        else:
            actual_risk_pct = risk_per_trade_pct
            
        return {
            'position_size': round(position_size, 0),
            'position_value': round(position_value, 2),
            'risk_amount': round(risk_amount, 2),
            'actual_risk_pct': round(actual_risk_pct, 2),
            'max_position_limited': position_value > max_position_value
        }
    except Exception as e:
        logger.error(f"Position sizing calculation error: {e}")
        return None

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
    def calculate_williams_vix_fix_corrected(self, data):
        """Corrected Williams VIX Fix per original Larry Williams formula"""
        try:
            period = self.config['wvf_period']  # Default 22
            multiplier = self.config['wvf_multiplier']  # Default 2.0
            
            if len(data) < period * 2:
                return 0.0

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
            current_upper = wvf_upper_band.iloc[-1] if not pd.isna(wvf_upper_band.iloc[-1]) else 0

            # Normalize the signal strength
            if current_upper > 0 and current_wvf > current_upper:
                signal_strength = (current_wvf - current_upper) / current_upper
            else:
                signal_strength = 0

            return float(np.clip(signal_strength, 0, 1))

        except Exception as e:
            logger.error(f"Williams VIX Fix calculation error: {e}")
            return 0.0

    @safe_calculation_wrapper
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
            deviation_pct = abs((current_price - ma_avg) / ma_avg * 100) if ma_avg > 0 else 0

            # Create deviation series for normalization
            deviation_series = []
            for i in range(min(252, len(close))):
                if i + max(ma_periods) < len(close):
                    subset_close = close.iloc[i:i+max(ma_periods)]
                    subset_mas = []
                    for period in ma_periods:
                        if len(subset_close) >= period:
                            subset_ma = subset_close.rolling(window=period).mean().iloc[-1]
                            subset_mas.append(subset_ma)
                    if subset_mas:
                        subset_avg = np.mean(subset_mas)
                        subset_price = subset_close.iloc[-1]
                        subset_deviation = abs((subset_price - subset_avg) / subset_avg * 100) if subset_avg > 0 else 0
                        deviation_series.append(subset_deviation)

            if deviation_series:
                deviation_df = pd.Series(deviation_series + [deviation_pct])
                return statistical_normalize(deviation_df)
            else:
                return 0.5

        except Exception as e:
            logger.error(f"MA confluence calculation error: {e}")
            return 0.0

    @safe_calculation_wrapper
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

            # Create volume ratio series for normalization
            if len(volume) >= periods[0]:
                vol_ratios = volume.tail(252) / volume.rolling(window=periods[0]).mean()
                vol_ratios = vol_ratios.dropna()
                if len(vol_ratios) > 0:
                    return statistical_normalize(vol_ratios)

            return 0.5
        except Exception as e:
            logger.error(f"Volume confluence calculation error: {e}")
            return 0.0

    @safe_calculation_wrapper
    def calculate_vwap_analysis(self, data):
        """VWAP analysis"""
        try:
            if len(data) < 20:
                return 0.0

            current_price = data['Close'].iloc[-1]
            current_vwap = calculate_daily_vwap(data)

            if current_vwap == 0:
                return 0.0

            vwap_deviation_pct = abs(current_price - current_vwap) / current_vwap * 100

            # Create VWAP deviation series for normalization
            vwap_deviations = []
            for i in range(min(252, len(data) - 20)):
                end_idx = len(data) - i
                start_idx = max(0, end_idx - 20)
                subset = data.iloc[start_idx:end_idx]
                if len(subset) >= 5:
                    daily_vwap = calculate_daily_vwap(subset)
                    price = subset['Close'].iloc[-1]
                    if daily_vwap > 0:
                        deviation = abs(price - daily_vwap) / daily_vwap * 100
                        vwap_deviations.append(deviation)

            if vwap_deviations:
                deviation_series = pd.Series(vwap_deviations + [vwap_deviation_pct])
                return statistical_normalize(deviation_series)
            else:
                return 0.5
        except Exception as e:
            logger.error(f"VWAP analysis calculation error: {e}")
            return 0.0

    @safe_calculation_wrapper
    def calculate_momentum(self, data):
        """Momentum calculation"""
        try:
            period = self.config['rsi_period']
            if len(data) < period + 1:
                return 0.0

            close = data['Close']
            rsi = safe_rsi(close, period)
            rsi_value = rsi.iloc[-1]

            # Convert RSI to oversold signal strength (higher values for more oversold)
            oversold_signal = max(0, (50 - rsi_value) / 50) if rsi_value < 50 else 0
            return float(np.clip(oversold_signal, 0, 1))
        except Exception as e:
            logger.error(f"Momentum calculation error: {e}")
            return 0.0

    @safe_calculation_wrapper
    def calculate_volatility_filter(self, data):
        """Volatility filter"""
        try:
            period = self.config['volatility_period']
            if len(data) < period:
                return 0.0

            close = data['Close']
            returns = close.pct_change().dropna()
            
            if len(returns) < period:
                return 0.5
                
            volatility = returns.rolling(window=period).std() * np.sqrt(252)
            
            if len(volatility) == 0:
                return 0.5

            return statistical_normalize(volatility)
        except Exception as e:
            logger.error(f"Volatility filter calculation error: {e}")
            return 0.0

    @safe_calculation_wrapper
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
            ema21_slope = (ema_21.iloc[-1] - ema_21.iloc[-5]) / ema_21.iloc[-5] * 100 if len(ema_21) > 5 else 0
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
        except Exception as e:
            logger.error(f"Trend analysis calculation error: {e}")
            return None

    @safe_calculation_wrapper
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
        except Exception as e:
            logger.error(f"Confidence intervals calculation error: {e}")
            return None

    def calculate_confluence(self, input_data, symbol='SPY', show_debug=False):
        """Enhanced confluence calculation with comprehensive analysis"""
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
            comprehensive_technicals = calculate_comprehensive_technicals(working_data)

            # Calculate market correlations
            market_correlations = calculate_market_correlations(working_data, symbol, show_debug=show_debug)

            # Calculate fundamental analysis scores (skip for ETFs)
            is_etf_symbol = is_etf(symbol)
            
            if is_etf_symbol:
                graham_score = {'score': 0, 'total_possible': 10, 'criteria': [], 'error': 'ETF - Fundamental analysis not applicable'}
                piotroski_score = {'score': 0, 'total_possible': 9, 'criteria': [], 'error': 'ETF - Fundamental analysis not applicable'}
                if show_debug:
                    st.write(f"📊 Skipping fundamental analysis for ETF: {symbol}")
            else:
                graham_score = calculate_graham_score(symbol, show_debug)
                piotroski_score = calculate_piotroski_score(symbol, show_debug)

            # Calculate VWV components with corrected WVF
            components = {
                'wvf': self.calculate_williams_vix_fix_corrected(working_data),
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

            # Calculate options levels
            volatility = comprehensive_technicals.get('volatility_20d', 20)
            options_levels = calculate_options_levels_enhanced(current_price, volatility)

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
                    'weekly_deviations': weekly_deviations,
                    'comprehensive_technicals': comprehensive_technicals,
                    'market_correlations': market_correlations,
                    'options_levels': options_levels,
                    'graham_score': graham_score,
                    'piotroski_score': piotroski_score
                },
                'system_status': 'OPERATIONAL'
            }

        except Exception as e:
            logger.error(f"Confluence calculation error for {symbol}: {e}")
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

            for std_level in [1, 2, 3]:
                std_data = weekly_devs.get(f'{std_level}_std')
                if std_data:
                    std_levels.append(f"{std_level}σ")
                    upper_values.append(std_data['upper'])
                    lower_values.append(std_data['lower'])

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

def get_etf_description(etf):
    """Get description of ETF"""
    descriptions = {
        'FNGD': '🐻 3x Inverse Technology ETF',
        'FNGU': '🚀 3x Leveraged Technology ETF',
        'MAGS': '🏛️ Magnificent Seven ETF'
    }
    return descriptions.get(etf, 'ETF')

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="candlestick-chart">
            <div class="candle candle-green" style="left: 10%; height: 30px; top: 45%;"></div>
            <div class="candle candle-red" style="left: 20%; height: 40px; top: 40%;"></div>
            <div class="candle candle-green" style="left: 30%; height: 25px; top: 50%;"></div>
            <div class="candle candle-green" style="left: 40%; height: 35px; top: 42%;"></div>
            <div class="candle candle-red" style="left: 50%; height: 45px; top: 38%;"></div>
            <div class="candle candle-green" style="left: 60%; height: 20px; top: 52%;"></div>
            <div class="candle candle-green" style="left: 70%; height: 50px; top: 35%;"></div>
            <div class="candle candle-red" style="left: 80%; height: 30px; top: 45%;"></div>
            <div class="candle candle-green" style="left: 90%; height: 35px; top: 42%;"></div>
        </div>
        <div class="foam-dollars">$ $ $ $ $ $ $</div>
        <div class="header-content">
            <h1>VWV Professional Trading System</h1>
            <p>Advanced market analysis with enhanced technical indicators</p>
            <p><em>Features: Daily VWAP, Fibonacci EMAs, Point of Control, Weekly Deviations</em></p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.title("📊 Trading Analysis")
    
    # Quick Links section
    with st.sidebar.expander("🔗 Quick Links"):
        st.write("**Popular Symbols**")
        
        # Symbol descriptions dictionary
        symbol_descriptions = {
            'QQQ': 'Invesco QQQ Trust - Nasdaq-100 ETF',
            'SPY': 'SPDR S&P 500 ETF - Large Cap US Stocks',
            'IWM': 'iShares Russell 2000 ETF - Small Cap US Stocks',
            'GLD': 'SPDR Gold Shares - Physical Gold ETF',
            'GDX': 'VanEck Gold Miners ETF - Gold Mining Stocks',
            'SLV': 'iShares Silver Trust - Physical Silver ETF',
            'URNM': 'North Shore Global Uranium Mining ETF',
            'TSLA': 'Tesla Inc - Electric Vehicles & Clean Energy',
            'AAPL': 'Apple Inc - Consumer Electronics & Technology',
            'AMZN': 'Amazon.com Inc - E-commerce & Cloud Computing',
            'NVDA': 'NVIDIA Corporation - Graphics & AI Chips',
            'NFLX': 'Netflix Inc - Streaming Entertainment',
            'MSFT': 'Microsoft Corporation - Software & Cloud Services',
            'META': 'Meta Platforms Inc - Social Media & Metaverse',
            'GOOG': 'Alphabet Inc - Google Search & Cloud',
            'AIG': 'American International Group - Insurance',
            'DIVO': 'Amplify CWP Enhanced Dividend Income ETF',
            'UNH': 'UnitedHealth Group - Healthcare & Insurance',
            'FNGD': 'MicroSectors FANG+ 3X Inverse Leveraged ETN',
            'FNGU': 'MicroSectors FANG+ 3X Leveraged ETN',
            'SPHB': 'Invesco S&P 500 High Beta ETF',
            'TLT': 'iShares 20+ Year Treasury Bond ETF',
            'SOXL': 'Direxion Semiconductor Bull 3X ETF',
            'QQI': 'Invesco QQQ Trust Series I',
            'MAGS': 'Roundhill Magnificent Seven ETF',
            'DIS': 'The Walt Disney Company - Entertainment',
            'FETH': 'Fidelity Ethereum ETF - Crypto Exposure'
        }
        
        # Organize symbols by category
        categories = {
            '📈 Major ETFs': ['QQQ', 'SPY', 'IWM', 'MAGS', 'SPHB', 'TLT'],
            '🥇 Commodities': ['GLD', 'GDX', 'SLV', 'URNM', 'FETH'],
            '🚀 Tech Giants': ['TSLA', 'AAPL', 'AMZN', 'NVDA', 'MSFT', 'META', 'GOOG'],
            '📺 Other Stocks': ['NFLX', 'AIG', 'DIVO', 'UNH', 'DIS'],
            '⚡ Leveraged': ['FNGD', 'FNGU', 'SOXL', 'QQI']
        }
        
        # Display symbols by category
        for category, symbols in categories.items():
            st.write(f"**{category}**")
            
            # Create rows of 3 buttons each
            for i in range(0, len(symbols), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(symbols):
                        sym = symbols[i + j]
                        with col:
                            if st.button(sym, help=symbol_descriptions[sym], key=f"quick_link_{sym}", use_container_width=True):
                                st.session_state.selected_symbol = sym
                                st.rerun()
            st.write("")  # Add spacing between categories

    # Basic controls - use session state for symbol if set by quick links
    if 'selected_symbol' in st.session_state:
        default_symbol = st.session_state.selected_symbol
        # Clear the session state after using it
        del st.session_state.selected_symbol
    else:
        default_symbol = "SPY"
        
    symbol = st.sidebar.text_input("Symbol", value=default_symbol, help="Enter stock symbol").upper()
    period = st.sidebar.selectbox("Data Period", ['1mo', '3mo', '6mo', '1y', '2y'], index=3)
    
    # Main analyze button - positioned right after data period
    analyze_button = st.sidebar.button("📊 Analyze Symbol", type="primary", use_container_width=True)

    # Debug toggle
    show_debug = st.sidebar.checkbox("🐛 Show Debug Info", value=False)
    
    # Test button - only show if debug is enabled
    if show_debug:
        test_button = st.sidebar.button("🧪 Test Data Fetch", use_container_width=True)
    else:
        test_button = False

    # Position sizing controls
    with st.sidebar.expander("💰 Position Sizing"):
        account_balance = st.number_input("Account Balance ($)", min_value=1000, value=100000, step=1000)
        risk_per_trade = st.slider("Risk per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.1)

    # System parameters
    with st.sidebar.expander("⚙️ System Parameters"):
        st.write("**Williams VIX Fix**")
        wvf_period = st.slider("WVF Period", 10, 50, 22)
        wvf_multiplier = st.slider("WVF Multiplier", 0.5, 3.0, 2.0, 0.1)

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
    elif analyze_button and symbol:
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
            
            analysis_results = vwv_system.calculate_confluence(analysis_input, symbol, show_debug)
            
            if 'error' in analysis_results:
                st.error(f"❌ Analysis failed: {analysis_results['error']}")
                return
            
            # Store results
            data_manager.store_analysis_results(symbol, analysis_results)
            
            # ============================================================
            # SECTION 1: INDIVIDUAL SYMBOL ANALYSIS
            # ============================================================
            st.header(f"📊 {symbol} - Individual Technical Analysis")
            
            enhanced_indicators = analysis_results.get('enhanced_indicators', {})
            comprehensive_technicals = enhanced_indicators.get('comprehensive_technicals', {})
            fibonacci_emas = enhanced_indicators.get('fibonacci_emas', {})
            
            # Primary metrics row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${analysis_results['current_price']}")
            with col2:
                price_change_1d = comprehensive_technicals.get('price_change_1d', 0)
                st.metric("1-Day Change", f"{price_change_1d:+.2f}%")
            with col3:
                price_change_5d = comprehensive_technicals.get('price_change_5d', 0)
                st.metric("5-Day Change", f"{price_change_5d:+.2f}%")
            with col4:
                volatility = comprehensive_technicals.get('volatility_20d', 0)
                st.metric("20D Volatility", f"{volatility:.1f}%")
            
            # Comprehensive Technical Analysis Table
            st.subheader("📋 Comprehensive Technical Indicators")
            
            current_price = analysis_results['current_price']
            daily_vwap = enhanced_indicators.get('daily_vwap', 0)
            point_of_control = enhanced_indicators.get('point_of_control', 0)
            
            def determine_signal(indicator_name, current_price, indicator_value, distance_pct_str, bb_data=None):
                """Determine if indicator is Bullish, Neutral, or Bearish"""
                try:
                    # Handle N/A cases
                    if distance_pct_str == "N/A" or indicator_value == 0:
                        return "Neutral"
                    
                    # Extract percentage value
                    if "%" in distance_pct_str:
                        distance_pct = float(distance_pct_str.replace("+", "").replace("%", ""))
                    else:
                        distance_pct = 0
                    
                    # Current Price - always neutral (reference point)
                    if "Current Price" in indicator_name:
                        return "Neutral"
                    
                    # VWAP and POC - above is bullish, below is bearish
                    elif "VWAP" in indicator_name or "Point of Control" in indicator_name:
                        return "Bullish" if distance_pct > 0 else "Bearish"
                    
                    # Previous Week High - breakout is very bullish
                    elif "Prev Week High" in indicator_name:
                        if distance_pct > 0:
                            return "Bullish"  # Breakout above resistance
                        elif distance_pct > -2:
                            return "Neutral"  # Near resistance
                        else:
                            return "Bearish"  # Well below resistance
                    
                    # Previous Week Low - breakdown is very bearish
                    elif "Prev Week Low" in indicator_name:
                        if distance_pct > 5:
                            return "Bullish"  # Well above support
                        elif distance_pct > 0:
                            return "Neutral"  # Just above support
                        else:
                            return "Bearish"  # Below support (breakdown)
                    
                    # EMAs - above is bullish, below is bearish
                    elif "EMA" in indicator_name:
                        if distance_pct > 1:
                            return "Bullish"
                        elif distance_pct > -1:
                            return "Neutral"
                        else:
                            return "Bearish"
                    
                    # Bollinger Bands - more complex logic
                    elif "BB" in indicator_name:
                        if "Upper" in indicator_name:
                            if distance_pct > 0:
                                return "Bearish"  # Above upper band (overbought)
                            elif distance_pct > -5:
                                return "Bullish"  # Near upper band (strong momentum)
                            else:
                                return "Neutral"
                        elif "Middle" in indicator_name:
                            return "Bullish" if distance_pct > 0 else "Bearish"
                        elif "Lower" in indicator_name:
                            if distance_pct < 0:
                                return "Bullish"  # Below lower band (oversold)
                            elif distance_pct < 5:
                                return "Bearish"  # Near lower band (weak momentum)
                            else:
                                return "Neutral"
                    
                    # Default logic for other indicators
                    else:
                        if distance_pct > 2:
                            return "Bullish"
                        elif distance_pct > -2:
                            return "Neutral"
                        else:
                            return "Bearish"
                
                except:
                    return "Neutral"

            # Build comprehensive indicators table
            indicators_data = []
            
            # Current Price
            indicators_data.append(("Current Price", f"${current_price:.2f}", "📍 Reference", "0.0%", "Current", 
                                  determine_signal("Current Price", current_price, current_price, "0.0%")))
            
            # Daily VWAP
            vwap_distance = f"{((current_price - daily_vwap) / daily_vwap * 100):+.2f}%" if daily_vwap > 0 else "N/A"
            vwap_status = "Above" if current_price > daily_vwap else "Below"
            indicators_data.append(("Daily VWAP", f"${daily_vwap:.2f}", "📊 Volume Weighted", vwap_distance, vwap_status,
                                  determine_signal("Daily VWAP", current_price, daily_vwap, vwap_distance)))
            
            # Point of Control
            poc_distance = f"{((current_price - point_of_control) / point_of_control * 100):+.2f}%" if point_of_control > 0 else "N/A"
            poc_status = "Above" if current_price > point_of_control else "Below"
            indicators_data.append(("Point of Control", f"${point_of_control:.2f}", "📊 Volume Profile", poc_distance, poc_status,
                                  determine_signal("Point of Control", current_price, point_of_control, poc_distance)))
            
            # Previous Week High
            prev_high = comprehensive_technicals.get('prev_week_high', 0)
            high_distance = f"{((current_price - prev_high) / prev_high * 100):+.2f}%" if prev_high > 0 else "N/A"
            high_status = "Above" if current_price > prev_high else "Below"
            indicators_data.append(("Prev Week High", f"${prev_high:.2f}", "📈 Resistance", high_distance, high_status,
                                  determine_signal("Prev Week High", current_price, prev_high, high_distance)))
            
            # Previous Week Low
            prev_low = comprehensive_technicals.get('prev_week_low', 0)
            low_distance = f"{((current_price - prev_low) / prev_low * 100):+.2f}%" if prev_low > 0 else "N/A"
            low_status = "Above" if current_price > prev_low else "Below"
            indicators_data.append(("Prev Week Low", f"${prev_low:.2f}", "📉 Support", low_distance, low_status,
                                  determine_signal("Prev Week Low", current_price, prev_low, low_distance)))
            
            # Add Fibonacci EMAs
            for ema_name, ema_value in fibonacci_emas.items():
                period = ema_name.split('_')[1]
                distance_pct = f"{((current_price - ema_value) / ema_value * 100):+.2f}%" if ema_value > 0 else "N/A"
                status = "Above" if current_price > ema_value else "Below"
                signal = determine_signal(f"EMA {period}", current_price, ema_value, distance_pct)
                indicators_data.append((f"EMA {period}", f"${ema_value:.2f}", "📈 Trend", distance_pct, status, signal))
            
            # Add Bollinger Bands
            bb_data = comprehensive_technicals.get('bollinger_bands', {})
            if bb_data:
                # BB Upper
                bb_upper = bb_data.get('upper', 0)
                upper_distance = f"{((current_price - bb_upper) / bb_upper * 100):+.2f}%" if bb_upper > 0 else "N/A"
                upper_status = f"Position: {bb_data.get('position', 50):.1f}%"
                upper_signal = determine_signal("BB Upper", current_price, bb_upper, upper_distance, bb_data)
                
                # BB Middle
                bb_middle = bb_data.get('middle', 0)
                middle_distance = f"{((current_price - bb_middle) / bb_middle * 100):+.2f}%" if bb_middle > 0 else "N/A"
                middle_status = "Above" if current_price > bb_middle else "Below"
                middle_signal = determine_signal("BB Middle", current_price, bb_middle, middle_distance, bb_data)
                
                # BB Lower
                bb_lower = bb_data.get('lower', 0)
                lower_distance = f"{((current_price - bb_lower) / bb_lower * 100):+.2f}%" if bb_lower > 0 else "N/A"
                lower_status = "Above" if current_price > bb_lower else "Below"
                lower_signal = determine_signal("BB Lower", current_price, bb_lower, lower_distance, bb_data)
                
                indicators_data.extend([
                    ("BB Upper", f"${bb_upper:.2f}", "📊 Volatility", upper_distance, upper_status, upper_signal),
                    ("BB Middle", f"${bb_middle:.2f}", "📊 SMA 20", middle_distance, middle_status, middle_signal),
                    ("BB Lower", f"${bb_lower:.2f}", "📊 Volatility", lower_distance, lower_status, lower_signal),
                ])
            
            # Convert to DataFrame and display with emoji indicators for signals
            for i, row in enumerate(indicators_data):
                if len(row) == 6:  # Has signal column
                    signal = row[5]
                    if signal == 'Bullish':
                        indicators_data[i] = row[:5] + ('🟢 Bullish',)
                    elif signal == 'Bearish':
                        indicators_data[i] = row[:5] + ('🔴 Bearish',)
                    else:
                        indicators_data[i] = row[:5] + ('🟡 Neutral',)
            
            df_technical = pd.DataFrame(indicators_data, columns=['Indicator', 'Value', 'Type', 'Distance %', 'Status', 'Signal'])
            st.dataframe(df_technical, use_container_width=True, hide_index=True)
            
            # Oscillators and Momentum
            st.subheader("📈 Momentum & Oscillator Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                rsi = comprehensive_technicals.get('rsi_14', 50)
                rsi_status = "🔴 Overbought" if rsi > 70 else "🟢 Oversold" if rsi < 30 else "⚪ Neutral"
                st.metric("RSI (14)", f"{rsi:.1f}", rsi_status)
            
            with col2:
                mfi = comprehensive_technicals.get('mfi_14', 50)
                mfi_status = "🔴 Overbought" if mfi > 80 else "🟢 Oversold" if mfi < 20 else "⚪ Neutral"
                st.metric("MFI (14)", f"{mfi:.1f}", mfi_status)
            
            with col3:
                williams_r = comprehensive_technicals.get('williams_r', -50)
                wr_status = "🔴 Overbought" if williams_r > -20 else "🟢 Oversold" if williams_r < -80 else "⚪ Neutral"
                st.metric("Williams %R", f"{williams_r:.1f}", wr_status)
            
            with col4:
                stoch_data = comprehensive_technicals.get('stochastic', {})
                stoch_k = stoch_data.get('k', 50)
                stoch_status = "🔴 Overbought" if stoch_k > 80 else "🟢 Oversold" if stoch_k < 20 else "⚪ Neutral"
                st.metric("Stochastic %K", f"{stoch_k:.1f}", stoch_status)
            
            # MACD Analysis
            macd_data = comprehensive_technicals.get('macd', {})
            if macd_data:
                col1, col2, col3 = st.columns(3)
                with col1:
                    macd_line = macd_data.get('macd', 0)
                    st.metric("MACD Line", f"{macd_line:.4f}")
                with col2:
                    signal_line = macd_data.get('signal', 0)
                    st.metric("Signal Line", f"{signal_line:.4f}")
                with col3:
                    histogram = macd_data.get('histogram', 0)
                    hist_trend = "📈 Bullish" if histogram > 0 else "📉 Bearish"
                    st.metric("MACD Histogram", f"{histogram:.4f}", hist_trend)
            
            # Volume Analysis
            st.subheader("📊 Volume Analysis")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_volume = comprehensive_technicals.get('current_volume', 0)
                st.metric("Current Volume", f"{current_volume:,.0f}")
            
            with col2:
                avg_volume = comprehensive_technicals.get('volume_sma_20', 0)
                st.metric("20D Avg Volume", f"{avg_volume:,.0f}")
            
            with col3:
                volume_ratio = comprehensive_technicals.get('volume_ratio', 1)
                vol_status = "🔴 High" if volume_ratio > 1.5 else "🟢 Low" if volume_ratio < 0.5 else "⚪ Normal"
                st.metric("Volume Ratio", f"{volume_ratio:.2f}x", vol_status)
            
            with col4:
                atr = comprehensive_technicals.get('atr_14', 0)
                st.metric("ATR (14)", f"${atr:.2f}")
            
            # ============================================================
            # SECTION 1.5: FUNDAMENTAL ANALYSIS (Skip for ETFs)
            # ============================================================
            
            # Check if symbol is ETF
            enhanced_indicators = analysis_results.get('enhanced_indicators', {})
            graham_data = enhanced_indicators.get('graham_score', {})
            piotroski_data = enhanced_indicators.get('piotroski_score', {})
            
            # Only show fundamental analysis for stocks, not ETFs
            is_etf_symbol = ('ETF' in graham_data.get('error', '') or 
                           'ETF' in piotroski_data.get('error', ''))
            
            if not is_etf_symbol and ('error' not in graham_data or 'error' not in piotroski_data):
                st.header("📊 Fundamental Analysis - Value Investment Scores")
                
                # Display scores overview
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if 'error' not in graham_data:
                        st.metric(
                            "Graham Score", 
                            f"{graham_data.get('score', 0)}/10",
                            f"Grade: {graham_data.get('grade', 'N/A')}"
                        )
                    else:
                        st.metric("Graham Score", "N/A", "Data Limited")
                
                with col2:
                    if 'error' not in piotroski_data:
                        st.metric(
                            "Piotroski F-Score", 
                            f"{piotroski_data.get('score', 0)}/9",
                            f"Grade: {piotroski_data.get('grade', 'N/A')}"
                        )
                    else:
                        st.metric("Piotroski F-Score", "N/A", "Data Limited")
                
                with col3:
                    if 'error' not in graham_data:
                        st.metric(
                            "Graham %", 
                            f"{graham_data.get('percentage', 0):.0f}%",
                            graham_data.get('interpretation', '')[:20] + "..."
                        )
                    else:
                        st.metric("Graham %", "0%", "No Data")
                
                with col4:
                    if 'error' not in piotroski_data:
                        st.metric(
                            "Piotroski %", 
                            f"{piotroski_data.get('percentage', 0):.0f}%",
                            piotroski_data.get('interpretation', '')[:20] + "..."
                        )
                    else:
                        st.metric("Piotroski %", "0%", "No Data")
                
                # Detailed breakdown
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🏛️ Benjamin Graham Value Score")
                    if 'error' not in graham_data and graham_data.get('criteria'):
                        st.write(f"**Overall Assessment:** {graham_data.get('interpretation', 'N/A')}")
                        st.write("**Criteria Breakdown:**")
                        for criterion in graham_data['criteria']:
                            st.write(f"• {criterion}")
                    else:
                        st.warning(f"⚠️ Graham analysis unavailable: {graham_data.get('error', 'Unknown error')}")
                        st.info("💡 **Graham Score evaluates:**\n"
                               "• P/E and P/B ratios\n"
                               "• Debt levels and liquidity\n" 
                               "• Earnings and revenue growth\n"
                               "• Dividend policy")
                
                with col2:
                    st.subheader("🏆 Piotroski F-Score Quality")
                    if 'error' not in piotroski_data and piotroski_data.get('criteria'):
                        st.write(f"**Overall Assessment:** {piotroski_data.get('interpretation', 'N/A')}")
                        st.write("**Criteria Breakdown:**")
                        for criterion in piotroski_data['criteria']:
                            st.write(f"• {criterion}")
                    else:
                        st.warning(f"⚠️ Piotroski analysis unavailable: {piotroski_data.get('error', 'Unknown error')}")
                        st.info("💡 **Piotroski F-Score evaluates:**\n"
                               "• Profitability trends\n"
                               "• Leverage and liquidity changes\n"
                               "• Operating efficiency improvements\n"
                               "• Overall financial quality")
                
                # Combined interpretation
                if 'error' not in graham_data and 'error' not in piotroski_data:
                    combined_score = (graham_data.get('percentage', 0) + piotroski_data.get('percentage', 0)) / 2
                    
                    if combined_score >= 75:
                        st.success(f"🟢 **Strong Fundamental Profile** ({combined_score:.0f}% Combined Score)")
                        st.write("Both value and quality metrics indicate a fundamentally sound investment candidate.")
                    elif combined_score >= 50:
                        st.info(f"🟡 **Moderate Fundamental Profile** ({combined_score:.0f}% Combined Score)")
                        st.write("Mixed fundamental signals - some strengths and weaknesses present.")
                    else:
                        st.error(f"🔴 **Weak Fundamental Profile** ({combined_score:.0f}% Combined Score)")
                        st.write("Fundamental analysis suggests caution - multiple areas of concern identified.")
            
            elif is_etf_symbol:
                st.info(f"ℹ️ **{symbol} is an ETF** - Fundamental analysis (Graham Score & Piotroski F-Score) is not applicable to Exchange-Traded Funds. ETFs represent baskets of securities and don't have individual company financials to analyze.")

            # ============================================================
            # SECTION 2: MARKET COMPARISON ANALYSIS
            # ============================================================
            st.header("🌐 Market Correlation & Comparison Analysis")
            
            market_correlations = enhanced_indicators.get('market_correlations', {})
            
            if market_correlations:
                st.subheader("📊 ETF Correlation Analysis")
                
                correlation_table_data = []
                for etf, etf_data in market_correlations.items():
                    correlation_table_data.append({
                        'ETF': etf,
                        'Correlation': f"{etf_data.get('correlation', 0):.3f}",
                        'Beta': f"{etf_data.get('beta', 0):.3f}",
                        'Relationship': etf_data.get('relationship', 'Unknown'),
                        'Description': get_etf_description(etf)
                    })
                
                df_correlations = pd.DataFrame(correlation_table_data)
                st.dataframe(df_correlations, use_container_width=True, hide_index=True)
                
                # Correlation interpretation
                st.write("**Correlation Interpretation:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write("• **FNGD**: 🐻 3x Inverse Tech ETF")
                    st.write("• Negative correlation expected")
                with col2:
                    st.write("• **FNGU**: 🚀 3x Leveraged Tech ETF") 
                    st.write("• Positive correlation for tech stocks")
                with col3:
                    st.write("• **MAGS**: 🏛️ Mega-cap Growth ETF")
                    st.write("• Broad market correlation")
            else:
                st.warning("⚠️ Market correlation data not available")
            
            # Broader market context
            st.subheader("📈 Market Context")
            trend_analysis = analysis_results.get('trend_analysis', {})
            if trend_analysis:
                col1, col2, col3 = st.columns(3)
                with col1:
                    trend_dir = trend_analysis.get('trend_direction', 'N/A')
                    trend_strength = trend_analysis.get('trend_strength', 0)
                    st.metric("Trend Direction", trend_dir, f"Strength: {trend_strength:.1f}")
                
                with col2:
                    price_vs_ema21 = trend_analysis.get('price_vs_ema21', 0)
                    st.metric("Price vs EMA21", f"{price_vs_ema21:+.2f}%")
                
                with col3:
                    ema21_slope = trend_analysis.get('ema21_slope', 0)
                    slope_trend = "📈 Rising" if ema21_slope > 0 else "📉 Falling"
                    st.metric("EMA21 Slope", f"{ema21_slope:+.2f}%", slope_trend)
            
            # ============================================================
            # SECTION 3: OPTIONS ANALYSIS
            # ============================================================
            st.header("🎯 Options Trading Analysis")
            
            options_levels = enhanced_indicators.get('options_levels', [])
            
            if options_levels:
                st.subheader("💰 Premium Selling Levels")
                st.write("**Enhanced option strike levels with Black-Scholes approximation**")
                
                df_options = pd.DataFrame(options_levels)
                st.dataframe(df_options, use_container_width=True, hide_index=True)
                
                # Options context
                col1, col2 = st.columns(2)
                with col1:
                    st.info("**Put Selling Strategy:**\n"
                           "• Sell puts below current price\n"
                           "• Collect premium if stock stays above strike\n"
                           "• PoT = Probability of Touch")
                
                with col2:
                    st.info("**Call Selling Strategy:**\n"
                           "• Sell calls above current price\n" 
                           "• Collect premium if stock stays below strike\n"
                           "• Lower PoT = Higher probability of profit")
            else:
                st.warning("⚠️ Options analysis not available - insufficient data")
            
            # VWV Signal Analysis
            st.subheader("🎯 VWV Trading Signal")
            
            # Signal display
            if analysis_results['signal_type'] != 'NONE':
                entry_info = analysis_results['entry_info']
                direction = entry_info['direction']
                
                # Calculate position sizing
                position_info = calculate_position_sizing(
                    account_balance, 
                    risk_per_trade, 
                    entry_info['entry_price'], 
                    entry_info['stop_loss']
                )
                
                st.success(f"""
                🚨 **VWV {direction} SIGNAL DETECTED**
                
                **Signal Strength:** {analysis_results['signal_type']}  
                **Direction:** {direction}  
                **Entry Price:** ${entry_info['entry_price']}  
                **Stop Loss:** ${entry_info['stop_loss']}  
                **Take Profit:** ${entry_info['take_profit']}  
                **Risk/Reward Ratio:** {entry_info['risk_reward']}:1  
                **Directional Confluence:** {analysis_results['directional_confluence']:.2f}
                """)
                
                # Position sizing information
                if position_info:
                    st.subheader("💰 Position Sizing")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Position Size", f"{position_info['position_size']:.0f} shares")
                    with col2:
                        st.metric("Position Value", f"${position_info['position_value']:,.2f}")
                    with col3:
                        st.metric("Risk Amount", f"${position_info['risk_amount']:,.2f}")
                    with col4:
                        st.metric("Risk %", f"{position_info['actual_risk_pct']:.2f}%")
                    
                    if position_info['max_position_limited']:
                        st.warning("⚠️ Position size limited by maximum position rules (15% of account)")
                    
            else:
                st.info("⚪ **No VWV Signal** - Market conditions do not meet signal criteria")
            
            # Show confluence components only if debug is on
            if show_debug:
                st.subheader("🔧 VWV Components Breakdown")
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
                st.dataframe(df_components, use_container_width=True, hide_index=True)
            
            # ============================================================
            # SECTION 4: INTERACTIVE CHART
            # ============================================================
            if show_chart:
                st.header("📈 Technical Analysis Chart")
                
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
            
            # Statistical confidence intervals (if available)
            if analysis_results.get('confidence_analysis'):
                st.subheader("📊 Statistical Confidence Intervals")
                confidence_data = analysis_results['confidence_analysis']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Weekly Return", f"{confidence_data['mean_weekly_return']:.3f}%")
                with col2:
                    st.metric("Weekly Volatility", f"{confidence_data['weekly_volatility']:.2f}%")
                with col3:
                    st.metric("Sample Size", f"{confidence_data['sample_size']} weeks")
                
                final_intervals_data = []
                for level, level_data in confidence_data['confidence_intervals'].items():
                    final_intervals_data.append({
                        'Confidence Level': level,
                        'Upper Bound': f"${level_data['upper_bound']}",
                        'Lower Bound': f"${level_data['lower_bound']}",
                        'Expected Move': f"±{level_data['expected_move_pct']:.2f}%"
                    })
                
                df_intervals = pd.DataFrame(final_intervals_data)
                st.dataframe(df_intervals, use_container_width=True, hide_index=True)
    
    else:
        st.markdown("""
        ## 🛠️ VWV Professional Trading System - Enhanced
        
        ### ✅ **Comprehensive Analysis Structure:**
        
        1. **📊 Individual Symbol Analysis**
           - Exhaustive technical indicator table with signal analysis
           - Price levels: VWAP, EMAs, Support/Resistance
           - Momentum indicators: RSI, MFI, MACD, Stochastic
           - Volume analysis and volatility metrics
           - Previous week high/low levels
        
        1.5. **📊 Fundamental Analysis**
           - **Benjamin Graham Score**: Classic value investing criteria (0-10)
           - **Piotroski F-Score**: Financial quality assessment (0-9)
           - P/E, P/B, debt ratios, profitability trends
           - Combined fundamental strength rating
        
        2. **🌐 Market Correlation Analysis**
           - Correlation with FNGD (3x Inverse Tech)
           - Correlation with FNGU (3x Leveraged Tech)  
           - Correlation with MAGS (Mega-cap Growth)
           - Beta calculations and relationship strength
           - Broader market context and trend analysis
        
        3. **🎯 Options Trading Analysis**
           - Enhanced premium selling strike levels (Black-Scholes)
           - Probability of touch calculations
           - Put and call selling strategies
           - Risk-adjusted option levels for multiple expiries
        
        4. **📈 Interactive Technical Chart**
           - All Fibonacci EMAs displayed
           - VWAP and Point of Control levels
           - Weekly standard deviation bands
           - Comprehensive visual analysis
        
        ### 🔧 **Enhanced Features:**
        
        **📊 Technical Analysis**
        - **Williams VIX Fix**: ✅ **CORRECTED** - Proper formula implementation
        - **Options Pricing**: ✅ **ENHANCED** - Black-Scholes approximation  
        - **Signal Analysis**: New bullish/neutral/bearish column in indicators table
        
        **💰 Fundamental Analysis**
        - **Graham Score**: Benjamin Graham's value investing criteria
        - **Piotroski F-Score**: 9-point financial quality assessment
        - **Combined Rating**: Integrated fundamental strength evaluation
        
        **🚀 Risk Management**
        - **Position Sizing**: Professional risk-based calculations
        - **Account Integration**: Real position sizes with P&L projections
        
        **Start analyzing: SPY, AAPL, MSFT, GOOGL, QQQ, TSLA, or any major symbol**
        
        **System Status: ✅ ENHANCED WITH FUNDAMENTAL ANALYSIS**
        """)

if __name__ == "__main__":
    main()
