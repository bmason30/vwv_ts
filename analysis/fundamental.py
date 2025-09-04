"""
Filename: analysis/fundamental.py
VWV Trading System v4.2.1
Created/Updated: 2025-09-04 14:01:47 EDT
Version: 1.1.0 - Initial integration and deployment compatibility fix
Purpose: Provides fundamental analysis using Graham and Piotroski F-Score.
"""
import yfinance as yf
import logging
from utils.decorators import safe_calculation_wrapper

logger = logging.getLogger(__name__)

@safe_calculation_wrapper
def calculate_graham_score(symbol, show_debug=False):
    """Calculate Benjamin Graham Score based on value investing criteria"""
    try:
        if show_debug:
            import streamlit as st
            st.write(f"📊 Calculating Graham Score for {symbol}...")
        
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info:
            return {'score': 0, 'total_possible': 10, 'criteria': [], 'error': 'Insufficient fundamental data'}
        
        score, total_possible, criteria = 0, 10, []
        
        pe_ratio = info.get('trailingPE')
        pb_ratio = info.get('priceToBook')
        debt_to_equity = info.get('debtToEquity')
        current_ratio = info.get('currentRatio')
        
        # 1. P/E ratio < 15
        if pe_ratio and pe_ratio < 15:
            score += 1; criteria.append(f"✅ P/E < 15 ({pe_ratio:.2f})")
        else: criteria.append(f"❌ P/E < 15 ({pe_ratio or 'N/A'})")
        
        # 2. P/B ratio < 1.5
        if pb_ratio and pb_ratio < 1.5:
            score += 1; criteria.append(f"✅ P/B < 1.5 ({pb_ratio:.2f})")
        else: criteria.append(f"❌ P/B < 1.5 ({pb_ratio or 'N/A'})")
        
        # ... Other criteria calculations ...
        
        return {
            'score': score, 'total_possible': total_possible, 'criteria': criteria,
            'grade': get_graham_grade(score), 'interpretation': get_graham_interpretation(score)
        }
        
    except Exception as e:
        logger.error(f"Graham score calculation error for {symbol}: {e}")
        return {'score': 0, 'total_possible': 10, 'criteria': [], 'error': str(e)}

@safe_calculation_wrapper
def calculate_piotroski_score(symbol, show_debug=False):
    """Calculate Piotroski F-Score (0-9 points)"""
    try:
        if show_debug:
            import streamlit as st
            st.write(f"📊 Calculating Piotroski F-Score for {symbol}...")
        
        ticker = yf.Ticker(symbol)
        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cashflow = ticker.cashflow
        
        if len(financials.columns) < 2 or len(balance_sheet.columns) < 2:
            return {'score': 0, 'total_possible': 9, 'criteria': [], 'error': 'Need at least 2 years of financial data'}
        
        score, total_possible, criteria = 0, 9, []
        
        # ... Piotroski F-Score logic ...
        
        return {
            'score': score, 'total_possible': total_possible, 'criteria': criteria,
            'grade': get_piotroski_grade(score), 'interpretation': get_piotroski_interpretation(score)
        }
        
    except Exception as e:
        logger.error(f"Piotroski score calculation error for {symbol}: {e}")
        return {'score': 0, 'total_possible': 9, 'criteria': [], 'error': str(e)}

def get_graham_grade(score):
    if score >= 8: return "A";
    elif score >= 6: return "B";
    elif score >= 4: return "C";
    else: return "F"

def get_graham_interpretation(score):
    if score >= 8: return "Excellent value candidate"
    elif score >= 6: return "Good value potential"
    else: return "Limited value appeal"

def get_piotroski_grade(score):
    if score >= 8: return "A";
    elif score >= 6: return "B";
    elif score >= 4: return "C";
    else: return "F"

def get_piotroski_interpretation(score):
    if score >= 8: return "Very strong fundamentals"
    elif score >= 6: return "Strong fundamentals"
    else: return "Weak fundamentals"
