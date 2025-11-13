"""
Filename: analysis/fundamental.py
VWV Trading System v4.2.1
Created/Updated: 2025-09-04 14:55:10 EDT
Version: 1.1.1 - Restored missing financial data fetching calls
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
        financials = ticker.financials # RESTORED
        
        if not info or financials.empty:
            return {'score': 0, 'total_possible': 10, 'criteria': [], 'error': 'Insufficient fundamental data'}
        
        score, total_possible, criteria = 0, 10, []

        pe_ratio = info.get('trailingPE')
        pb_ratio = info.get('priceToBook')
        debt_to_equity = info.get('debtToEquity')
        current_ratio = info.get('currentRatio')
        dividend_yield = info.get('dividendYield', 0)
        profit_margins = info.get('profitMargins')

        # Criterion 1: P/E Ratio < 15
        if pe_ratio and pe_ratio < 15:
            score += 1
            criteria.append("✓ P/E < 15")
        else:
            criteria.append(f"✗ P/E {pe_ratio:.2f if pe_ratio else 'N/A'} >= 15")

        # Criterion 2: P/B Ratio < 1.5
        if pb_ratio and pb_ratio < 1.5:
            score += 1
            criteria.append("✓ P/B < 1.5")
        else:
            criteria.append(f"✗ P/B {pb_ratio:.2f if pb_ratio else 'N/A'} >= 1.5")

        # Criterion 3: Debt to Equity < 0.5
        if debt_to_equity and debt_to_equity < 0.5:
            score += 1
            criteria.append("✓ Debt/Equity < 0.5")
        else:
            criteria.append(f"✗ Debt/Equity {debt_to_equity:.2f if debt_to_equity else 'N/A'} >= 0.5")

        # Criterion 4: Current Ratio > 2.0
        if current_ratio and current_ratio > 2.0:
            score += 1
            criteria.append("✓ Current Ratio > 2.0")
        else:
            criteria.append(f"✗ Current Ratio {current_ratio:.2f if current_ratio else 'N/A'} <= 2.0")

        # Criterion 5: Dividend Yield > 0
        if dividend_yield and dividend_yield > 0:
            score += 1
            criteria.append(f"✓ Pays Dividends ({dividend_yield*100:.2f}%)")
        else:
            criteria.append("✗ No Dividends")

        # Criterion 6: Profit Margins > 10%
        if profit_margins and profit_margins > 0.10:
            score += 1
            criteria.append(f"✓ Profit Margin > 10% ({profit_margins*100:.1f}%)")
        else:
            criteria.append(f"✗ Profit Margin <= 10% ({profit_margins*100:.1f}% if profit_margins else 'N/A')")

        # Criterion 7: Earnings Growth (check if available in financials)
        try:
            if not financials.empty and len(financials.columns) >= 2:
                net_income_col = [col for col in financials.index if 'Net Income' in str(col)]
                if net_income_col:
                    recent_income = financials.loc[net_income_col[0]].iloc[0]
                    older_income = financials.loc[net_income_col[0]].iloc[-1]
                    if recent_income > older_income:
                        score += 1
                        criteria.append("✓ Positive Earnings Growth")
                    else:
                        criteria.append("✗ Negative Earnings Growth")
                else:
                    criteria.append("✗ Earnings data unavailable")
            else:
                criteria.append("✗ Insufficient financial history")
        except:
            criteria.append("✗ Earnings growth check failed")

        # Criterion 8: Price reasonableness (P/E * P/B < 22.5 - Graham's formula)
        if pe_ratio and pb_ratio:
            graham_number = pe_ratio * pb_ratio
            if graham_number < 22.5:
                score += 1
                criteria.append(f"✓ Graham Number < 22.5 ({graham_number:.1f})")
            else:
                criteria.append(f"✗ Graham Number >= 22.5 ({graham_number:.1f})")
        else:
            criteria.append("✗ Cannot calculate Graham Number")

        # Criterion 9: Positive Book Value
        book_value = info.get('bookValue')
        if book_value and book_value > 0:
            score += 1
            criteria.append(f"✓ Positive Book Value (${book_value:.2f})")
        else:
            criteria.append("✗ Non-positive Book Value")

        # Criterion 10: Market Cap adequate (> $2B)
        market_cap = info.get('marketCap')
        if market_cap and market_cap > 2_000_000_000:
            score += 1
            criteria.append(f"✓ Market Cap > $2B")
        else:
            criteria.append("✗ Market Cap <= $2B")

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
        financials = ticker.financials       # RESTORED
        balance_sheet = ticker.balance_sheet # RESTORED
        cashflow = ticker.cashflow           # RESTORED
        
        if len(financials.columns) < 2 or len(balance_sheet.columns) < 2:
            return {'score': 0, 'total_possible': 9, 'criteria': [], 'error': 'Need at least 2 years of financial data'}

        score, total_possible, criteria = 0, 9, []

        try:
            # Get current (most recent) and previous year data
            current_year = 0
            previous_year = 1

            # PROFITABILITY SIGNALS (4 points max)

            # 1. ROA > 0 (Positive Return on Assets)
            try:
                net_income_idx = [idx for idx in financials.index if 'Net Income' in str(idx)]
                total_assets_idx = [idx for idx in balance_sheet.index if 'Total Assets' in str(idx)]

                if net_income_idx and total_assets_idx:
                    net_income = financials.loc[net_income_idx[0]].iloc[current_year]
                    total_assets = balance_sheet.loc[total_assets_idx[0]].iloc[current_year]

                    if total_assets != 0:
                        roa = net_income / total_assets
                        if roa > 0:
                            score += 1
                            criteria.append(f"✓ Positive ROA ({roa*100:.2f}%)")
                        else:
                            criteria.append(f"✗ Negative ROA ({roa*100:.2f}%)")
                    else:
                        criteria.append("✗ ROA: Cannot calculate (zero assets)")
                else:
                    criteria.append("✗ ROA: Data unavailable")
            except:
                criteria.append("✗ ROA: Calculation failed")

            # 2. Operating Cash Flow > 0
            try:
                ocf_idx = [idx for idx in cashflow.index if 'Operating Cash Flow' in str(idx) or 'Total Cash From Operating' in str(idx)]

                if ocf_idx:
                    ocf = cashflow.loc[ocf_idx[0]].iloc[current_year]
                    if ocf > 0:
                        score += 1
                        criteria.append("✓ Positive Operating Cash Flow")
                    else:
                        criteria.append("✗ Negative Operating Cash Flow")
                else:
                    criteria.append("✗ OCF: Data unavailable")
            except:
                criteria.append("✗ OCF: Calculation failed")

            # 3. Change in ROA (ROA current > ROA previous)
            try:
                if net_income_idx and total_assets_idx:
                    net_income_curr = financials.loc[net_income_idx[0]].iloc[current_year]
                    total_assets_curr = balance_sheet.loc[total_assets_idx[0]].iloc[current_year]
                    net_income_prev = financials.loc[net_income_idx[0]].iloc[previous_year]
                    total_assets_prev = balance_sheet.loc[total_assets_idx[0]].iloc[previous_year]

                    if total_assets_curr != 0 and total_assets_prev != 0:
                        roa_curr = net_income_curr / total_assets_curr
                        roa_prev = net_income_prev / total_assets_prev

                        if roa_curr > roa_prev:
                            score += 1
                            criteria.append("✓ Improving ROA")
                        else:
                            criteria.append("✗ Declining ROA")
                    else:
                        criteria.append("✗ Δ ROA: Cannot calculate")
                else:
                    criteria.append("✗ Δ ROA: Data unavailable")
            except:
                criteria.append("✗ Δ ROA: Calculation failed")

            # 4. Quality of Earnings (OCF > Net Income)
            try:
                if net_income_idx and ocf_idx:
                    net_income = financials.loc[net_income_idx[0]].iloc[current_year]
                    ocf = cashflow.loc[ocf_idx[0]].iloc[current_year]

                    if ocf > net_income:
                        score += 1
                        criteria.append("✓ OCF > Net Income (Quality)")
                    else:
                        criteria.append("✗ OCF <= Net Income")
                else:
                    criteria.append("✗ Quality: Data unavailable")
            except:
                criteria.append("✗ Quality: Calculation failed")

            # LEVERAGE, LIQUIDITY & SOURCE OF FUNDS (3 points max)

            # 5. Change in Long-Term Debt (Decrease is good)
            try:
                ltd_idx = [idx for idx in balance_sheet.index if 'Long Term Debt' in str(idx)]

                if ltd_idx:
                    ltd_curr = balance_sheet.loc[ltd_idx[0]].iloc[current_year]
                    ltd_prev = balance_sheet.loc[ltd_idx[0]].iloc[previous_year]

                    if ltd_curr < ltd_prev:
                        score += 1
                        criteria.append("✓ Decreasing Long-Term Debt")
                    else:
                        criteria.append("✗ Increasing Long-Term Debt")
                else:
                    criteria.append("✗ LTD: Data unavailable")
            except:
                criteria.append("✗ LTD: Calculation failed")

            # 6. Change in Current Ratio (Increase is good)
            try:
                current_assets_idx = [idx for idx in balance_sheet.index if 'Current Assets' in str(idx)]
                current_liabilities_idx = [idx for idx in balance_sheet.index if 'Current Liabilities' in str(idx)]

                if current_assets_idx and current_liabilities_idx:
                    ca_curr = balance_sheet.loc[current_assets_idx[0]].iloc[current_year]
                    cl_curr = balance_sheet.loc[current_liabilities_idx[0]].iloc[current_year]
                    ca_prev = balance_sheet.loc[current_assets_idx[0]].iloc[previous_year]
                    cl_prev = balance_sheet.loc[current_liabilities_idx[0]].iloc[previous_year]

                    if cl_curr != 0 and cl_prev != 0:
                        cr_curr = ca_curr / cl_curr
                        cr_prev = ca_prev / cl_prev

                        if cr_curr > cr_prev:
                            score += 1
                            criteria.append("✓ Improving Current Ratio")
                        else:
                            criteria.append("✗ Declining Current Ratio")
                    else:
                        criteria.append("✗ Current Ratio: Cannot calculate")
                else:
                    criteria.append("✗ Current Ratio: Data unavailable")
            except:
                criteria.append("✗ Current Ratio: Calculation failed")

            # 7. No New Shares Issued
            try:
                shares_idx = [idx for idx in balance_sheet.index if 'Shares Outstanding' in str(idx) or 'Common Stock Shares' in str(idx)]

                if shares_idx:
                    shares_curr = balance_sheet.loc[shares_idx[0]].iloc[current_year]
                    shares_prev = balance_sheet.loc[shares_idx[0]].iloc[previous_year]

                    if shares_curr <= shares_prev:
                        score += 1
                        criteria.append("✓ No New Share Dilution")
                    else:
                        criteria.append("✗ Share Dilution Occurred")
                else:
                    criteria.append("✗ Shares: Data unavailable")
            except:
                criteria.append("✗ Shares: Calculation failed")

            # OPERATING EFFICIENCY (2 points max)

            # 8. Change in Gross Margin (Increase is good)
            try:
                gross_profit_idx = [idx for idx in financials.index if 'Gross Profit' in str(idx)]
                total_revenue_idx = [idx for idx in financials.index if 'Total Revenue' in str(idx)]

                if gross_profit_idx and total_revenue_idx:
                    gp_curr = financials.loc[gross_profit_idx[0]].iloc[current_year]
                    rev_curr = financials.loc[total_revenue_idx[0]].iloc[current_year]
                    gp_prev = financials.loc[gross_profit_idx[0]].iloc[previous_year]
                    rev_prev = financials.loc[total_revenue_idx[0]].iloc[previous_year]

                    if rev_curr != 0 and rev_prev != 0:
                        gm_curr = gp_curr / rev_curr
                        gm_prev = gp_prev / rev_prev

                        if gm_curr > gm_prev:
                            score += 1
                            criteria.append("✓ Improving Gross Margin")
                        else:
                            criteria.append("✗ Declining Gross Margin")
                    else:
                        criteria.append("✗ Gross Margin: Cannot calculate")
                else:
                    criteria.append("✗ Gross Margin: Data unavailable")
            except:
                criteria.append("✗ Gross Margin: Calculation failed")

            # 9. Change in Asset Turnover (Increase is good)
            try:
                if total_revenue_idx and total_assets_idx:
                    rev_curr = financials.loc[total_revenue_idx[0]].iloc[current_year]
                    assets_curr = balance_sheet.loc[total_assets_idx[0]].iloc[current_year]
                    rev_prev = financials.loc[total_revenue_idx[0]].iloc[previous_year]
                    assets_prev = balance_sheet.loc[total_assets_idx[0]].iloc[previous_year]

                    if assets_curr != 0 and assets_prev != 0:
                        at_curr = rev_curr / assets_curr
                        at_prev = rev_prev / assets_prev

                        if at_curr > at_prev:
                            score += 1
                            criteria.append("✓ Improving Asset Turnover")
                        else:
                            criteria.append("✗ Declining Asset Turnover")
                    else:
                        criteria.append("✗ Asset Turnover: Cannot calculate")
                else:
                    criteria.append("✗ Asset Turnover: Data unavailable")
            except:
                criteria.append("✗ Asset Turnover: Calculation failed")

        except Exception as e:
            logger.error(f"Piotroski calculation error: {e}")
            return {'score': 0, 'total_possible': 9, 'criteria': ['Error in calculation'], 'error': str(e)}

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
