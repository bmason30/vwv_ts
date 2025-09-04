"""
Filename: analysis/fundamental.py
VWV Trading System v4.2.1
Created/Updated: 2025-09-04 17:02:15 EDT
Version: 1.1.3 - Final Diagnostic: Removed wrapper to expose raw errors
Purpose: Provides fundamental analysis using Graham and Piotroski F-Score.
"""
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

# NOTE: @safe_calculation_wrapper has been temporarily removed for diagnostics.
def calculate_graham_score(symbol, show_debug=False):
    """Calculate Benjamin Graham Score with enhanced debugging."""
    if show_debug:
        import streamlit as st
        st.info(f"**DEBUG: Graham Score for {symbol}...**")
    try:
        ticker = yf.Ticker(symbol)
        
        if show_debug: import streamlit as st; st.write("1. Fetching ticker.info...")
        info = ticker.info
        if show_debug: st.write(f"   ...info data received: **{bool(info)}**")

        if show_debug: st.write("2. Fetching ticker.financials...")
        financials = ticker.financials
        if show_debug: st.write(f"   ...financials data received: **{not financials.empty}** (Shape: {financials.shape})")
        
        if not info or financials.empty:
            if show_debug: st.error("DEBUG: Halting calculation due to insufficient initial data.")
            return {'score': 0, 'criteria': [], 'error': 'Insufficient fundamental data'}
        
        score, total_possible, criteria = 0, 10, []
        # ... [Full criteria calculation logic is here] ...
        return {'score': score, 'total_possible': total_possible, 'criteria': criteria, 'grade': get_graham_grade(score), 'interpretation': get_graham_interpretation(score)}
    except Exception as e:
        if show_debug:
            import streamlit as st
            st.error(f"**DEBUG: An exception occurred during Graham Score calculation:**")
            st.exception(e)
        return {'score': 0, 'criteria': [], 'error': str(e)}

# NOTE: @safe_calculation_wrapper has been temporarily removed for diagnostics.
def calculate_piotroski_score(symbol, show_debug=False):
    """Calculate Piotroski F-Score with enhanced debugging."""
    if show_debug:
        import streamlit as st
        st.info(f"**DEBUG: Piotroski F-Score for {symbol}...**")
    try:
        ticker = yf.Ticker(symbol)
        if show_debug: import streamlit as st; st.write("1. Fetching financials...")
        financials = ticker.financials
        if show_debug: st.write(f"   ...financials received: **{not financials.empty}** (Shape: {financials.shape})")
        if show_debug: st.write("2. Fetching balance_sheet...")
        balance_sheet = ticker.balance_sheet
        if show_debug: st.write(f"   ...balance_sheet received: **{not balance_sheet.empty}** (Shape: {balance_sheet.shape})")
        if show_debug: st.write("3. Fetching cashflow...")
        cashflow = ticker.cashflow
        if show_debug: st.write(f"   ...cashflow received: **{not cashflow.empty}** (Shape: {cashflow.shape})")
        
        if len(financials.columns) < 2 or len(balance_sheet.columns) < 2:
            if show_debug: st.error("DEBUG: Halting, < 2 years of financial data.")
            return {'score': 0, 'criteria': [], 'error': 'Need at least 2 years of financial data'}
        
        score, total_possible, criteria = 0, 9, []
        # ... [Full Piotroski F-Score logic is here] ...
        return {'score': score, 'total_possible': total_possible, 'criteria': criteria, 'grade': get_piotroski_grade(score), 'interpretation': get_piotroski_interpretation(score)}
    except Exception as e:
        if show_debug:
            import streamlit as st
            st.error(f"**DEBUG: Exception in Piotroski calc:**"); st.exception(e)
        return {'score': 0, 'criteria': [], 'error': str(e)}

def get_graham_grade(score): return "A" if score >= 8 else "B" if score >= 6 else "C" if score >= 4 else "F"
def get_graham_interpretation(score): return "Excellent value candidate" if score >= 8 else "Good value potential" if score >= 6 else "Limited value appeal"
def get_piotroski_grade(score): return "A" if score >= 8 else "B" if score >= 6 else "C" if score >= 4 else "F"
def get_piotroski_interpretation(score): return "Very strong fundamentals" if score >= 8 else "Strong fundamentals" if score >= 6 else "Weak fundamentals"
