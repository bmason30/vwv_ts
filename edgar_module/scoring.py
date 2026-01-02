"""
Advanced Financial Scoring Module
Calculates Piotroski F-Score, Altman Z-Score, and Benjamin Graham Number
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AdvancedScoring:
    """
    Advanced financial scoring and analysis
    """

    @staticmethod
    def calculate_piotroski_fscore(
        current_financials: Dict[str, pd.DataFrame],
        prior_financials: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Calculate Piotroski F-Score (0-9 scale)

        Components:
        Profitability (4 points):
        - Positive Net Income (1 pt)
        - Positive ROA (1 pt)
        - Positive Operating Cash Flow (1 pt)
        - Cash Flow > Net Income (1 pt)

        Leverage/Liquidity (3 points):
        - Lower Long-term Debt ratio (1 pt)
        - Higher Current Ratio (1 pt)
        - No New Shares Issued (1 pt)

        Efficiency (2 points):
        - Higher Gross Margin (1 pt)
        - Higher Asset Turnover (1 pt)

        Args:
            current_financials: Dict with balance_sheet, income_statement, cash_flow
            prior_financials: Same structure for prior year

        Returns:
            Dict with score and component breakdown
        """
        score = 0
        components = {}

        try:
            curr_bs = current_financials.get('balance_sheet', pd.DataFrame())
            curr_income = current_financials.get('income_statement', pd.DataFrame())
            curr_cf = current_financials.get('cash_flow', pd.DataFrame())

            prior_bs = prior_financials.get('balance_sheet', pd.DataFrame())
            prior_income = prior_financials.get('income_statement', pd.DataFrame())

            if curr_bs.empty or curr_income.empty:
                return {'score': None, 'components': {}, 'error': 'Insufficient data'}

            # Get latest values
            curr_bs_latest = curr_bs.iloc[0]
            curr_income_latest = curr_income.iloc[0]

            # PROFITABILITY (4 points)

            # 1. Positive Net Income
            net_income = curr_income_latest.get('NetIncomeLoss', np.nan)
            if pd.notna(net_income) and net_income > 0:
                score += 1
                components['positive_net_income'] = True
            else:
                components['positive_net_income'] = False

            # 2. Positive ROA
            assets = curr_bs_latest.get('Assets', np.nan)
            if pd.notna(net_income) and pd.notna(assets) and assets > 0:
                roa = net_income / assets
                if roa > 0:
                    score += 1
                    components['positive_roa'] = True
                    components['roa'] = roa
                else:
                    components['positive_roa'] = False
                    components['roa'] = roa
            else:
                components['positive_roa'] = False

            # 3. Positive Operating Cash Flow
            if not curr_cf.empty:
                curr_cf_latest = curr_cf.iloc[0]
                ocf = curr_cf_latest.get('NetCashProvidedByUsedInOperatingActivities', np.nan)
                if pd.notna(ocf) and ocf > 0:
                    score += 1
                    components['positive_ocf'] = True
                    components['ocf'] = ocf
                else:
                    components['positive_ocf'] = False

                # 4. Cash Flow > Net Income (quality of earnings)
                if pd.notna(ocf) and pd.notna(net_income):
                    if ocf > net_income:
                        score += 1
                        components['ocf_gt_ni'] = True
                    else:
                        components['ocf_gt_ni'] = False
            else:
                components['positive_ocf'] = False
                components['ocf_gt_ni'] = False

            # LEVERAGE/LIQUIDITY (3 points)

            # 5. Lower Long-term Debt Ratio (compared to prior year)
            if not prior_bs.empty:
                prior_bs_latest = prior_bs.iloc[0]

                curr_debt = curr_bs_latest.get('LongTermDebt', np.nan)
                curr_assets = curr_bs_latest.get('Assets', np.nan)
                prior_debt = prior_bs_latest.get('LongTermDebt', np.nan)
                prior_assets = prior_bs_latest.get('Assets', np.nan)

                if all(pd.notna([curr_debt, curr_assets, prior_debt, prior_assets])):
                    if curr_assets > 0 and prior_assets > 0:
                        curr_debt_ratio = curr_debt / curr_assets
                        prior_debt_ratio = prior_debt / prior_assets
                        if curr_debt_ratio < prior_debt_ratio:
                            score += 1
                            components['lower_debt_ratio'] = True
                        else:
                            components['lower_debt_ratio'] = False
                        components['debt_ratio_change'] = curr_debt_ratio - prior_debt_ratio

            # 6. Higher Current Ratio
            if not prior_bs.empty:
                curr_current_assets = curr_bs_latest.get('AssetsCurrent', np.nan)
                curr_current_liab = curr_bs_latest.get('LiabilitiesCurrent', np.nan)
                prior_current_assets = prior_bs_latest.get('AssetsCurrent', np.nan)
                prior_current_liab = prior_bs_latest.get('LiabilitiesCurrent', np.nan)

                if all(pd.notna([curr_current_assets, curr_current_liab, prior_current_assets, prior_current_liab])):
                    if curr_current_liab > 0 and prior_current_liab > 0:
                        curr_current_ratio = curr_current_assets / curr_current_liab
                        prior_current_ratio = prior_current_assets / prior_current_liab
                        if curr_current_ratio > prior_current_ratio:
                            score += 1
                            components['higher_current_ratio'] = True
                        else:
                            components['higher_current_ratio'] = False
                        components['current_ratio'] = curr_current_ratio

            # 7. No New Shares Issued
            # This would require shares outstanding data - approximate with equity change
            curr_equity = curr_bs_latest.get('StockholdersEquity', np.nan)
            if not prior_bs.empty:
                prior_equity = prior_bs_latest.get('StockholdersEquity', np.nan)
                if pd.notna(curr_equity) and pd.notna(prior_equity):
                    # If equity didn't increase significantly, likely no new shares
                    equity_change_pct = (curr_equity - prior_equity) / prior_equity if prior_equity != 0 else 0
                    if equity_change_pct <= 0.05:  # Less than 5% increase
                        score += 1
                        components['no_new_shares'] = True
                    else:
                        components['no_new_shares'] = False

            # EFFICIENCY (2 points)

            # 8. Higher Gross Margin
            if not prior_income.empty:
                prior_income_latest = prior_income.iloc[0]

                # Get revenue (try multiple concepts)
                curr_revenue = curr_income_latest.get('Revenues',
                    curr_income_latest.get('RevenueFromContractWithCustomerExcludingAssessedTax', np.nan))
                curr_gross_profit = curr_income_latest.get('GrossProfit', np.nan)

                prior_revenue = prior_income_latest.get('Revenues',
                    prior_income_latest.get('RevenueFromContractWithCustomerExcludingAssessedTax', np.nan))
                prior_gross_profit = prior_income_latest.get('GrossProfit', np.nan)

                if all(pd.notna([curr_revenue, curr_gross_profit, prior_revenue, prior_gross_profit])):
                    if curr_revenue > 0 and prior_revenue > 0:
                        curr_gross_margin = curr_gross_profit / curr_revenue
                        prior_gross_margin = prior_gross_profit / prior_revenue
                        if curr_gross_margin > prior_gross_margin:
                            score += 1
                            components['higher_gross_margin'] = True
                        else:
                            components['higher_gross_margin'] = False
                        components['gross_margin'] = curr_gross_margin

            # 9. Higher Asset Turnover
            if not prior_income.empty:
                if all(pd.notna([curr_revenue, curr_assets, prior_revenue, prior_assets])):
                    if curr_assets > 0 and prior_assets > 0:
                        curr_turnover = curr_revenue / curr_assets
                        prior_turnover = prior_revenue / prior_assets
                        if curr_turnover > prior_turnover:
                            score += 1
                            components['higher_asset_turnover'] = True
                        else:
                            components['higher_asset_turnover'] = False
                        components['asset_turnover'] = curr_turnover

            return {
                'score': score,
                'max_score': 9,
                'components': components,
                'interpretation': get_piotroski_interpretation(score)
            }

        except Exception as e:
            logger.error(f"Error calculating Piotroski F-Score: {e}")
            return {'score': None, 'components': {}, 'error': str(e)}

    @staticmethod
    def calculate_altman_zscore(
        financials: Dict[str, pd.DataFrame],
        market_cap: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate Altman Z-Score for bankruptcy prediction

        Z = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E

        Where:
        A = Working Capital / Total Assets
        B = Retained Earnings / Total Assets
        C = EBIT / Total Assets
        D = Market Value of Equity / Total Liabilities
        E = Sales / Total Assets

        Interpretation:
        Z > 2.99: Safe zone
        1.81 < Z < 2.99: Grey zone
        Z < 1.81: Distress zone

        Args:
            financials: Dict with balance_sheet, income_statement
            market_cap: Market capitalization (if available)

        Returns:
            Dict with Z-score and components
        """
        try:
            bs = financials.get('balance_sheet', pd.DataFrame())
            income = financials.get('income_statement', pd.DataFrame())

            if bs.empty or income.empty:
                return {'zscore': None, 'error': 'Insufficient data'}

            bs_latest = bs.iloc[0]
            income_latest = income.iloc[0]

            # Get required values
            current_assets = bs_latest.get('AssetsCurrent', np.nan)
            current_liab = bs_latest.get('LiabilitiesCurrent', np.nan)
            total_assets = bs_latest.get('Assets', np.nan)
            total_liab = bs_latest.get('Liabilities', np.nan)
            equity = bs_latest.get('StockholdersEquity', np.nan)

            # Revenue
            revenue = income_latest.get('Revenues',
                income_latest.get('RevenueFromContractWithCustomerExcludingAssessedTax', np.nan))

            # Operating income as proxy for EBIT
            ebit = income_latest.get('OperatingIncomeLoss', np.nan)

            # Calculate components
            if pd.notna(total_assets) and total_assets > 0:
                # A: Working Capital / Total Assets
                if pd.notna(current_assets) and pd.notna(current_liab):
                    working_capital = current_assets - current_liab
                    a = 1.2 * (working_capital / total_assets)
                else:
                    a = 0

                # B: Retained Earnings / Total Assets
                # Using equity as proxy for retained earnings (conservative)
                if pd.notna(equity):
                    b = 1.4 * (equity / total_assets)
                else:
                    b = 0

                # C: EBIT / Total Assets
                if pd.notna(ebit):
                    c = 3.3 * (ebit / total_assets)
                else:
                    c = 0

                # D: Market Value of Equity / Total Liabilities
                if pd.notna(total_liab) and total_liab > 0:
                    if market_cap is not None:
                        d = 0.6 * (market_cap / total_liab)
                    elif pd.notna(equity):
                        # Use book value if market cap not available
                        d = 0.6 * (equity / total_liab)
                    else:
                        d = 0
                else:
                    d = 0

                # E: Sales / Total Assets
                if pd.notna(revenue):
                    e = 1.0 * (revenue / total_assets)
                else:
                    e = 0

                zscore = a + b + c + d + e

                # Determine zone
                if zscore > 2.99:
                    zone = 'Safe Zone'
                    risk = 'Low'
                elif zscore > 1.81:
                    zone = 'Grey Zone'
                    risk = 'Moderate'
                else:
                    zone = 'Distress Zone'
                    risk = 'High'

                return {
                    'zscore': zscore,
                    'components': {
                        'A_working_capital': a,
                        'B_retained_earnings': b,
                        'C_ebit': c,
                        'D_equity_to_liab': d,
                        'E_sales_turnover': e
                    },
                    'zone': zone,
                    'bankruptcy_risk': risk,
                    'interpretation': f"{zone} - {risk} bankruptcy risk"
                }
            else:
                return {'zscore': None, 'error': 'Total assets not available'}

        except Exception as e:
            logger.error(f"Error calculating Altman Z-Score: {e}")
            return {'zscore': None, 'error': str(e)}

    @staticmethod
    def calculate_graham_number(
        financials: Dict[str, pd.DataFrame],
        current_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate Benjamin Graham Number (maximum fair value)

        Graham Number = sqrt(22.5 × EPS × BVPS)

        Where:
        EPS = Earnings Per Share
        BVPS = Book Value Per Share

        Graham's criteria:
        - P/E ratio should not exceed 15
        - Price-to-Book should not exceed 1.5
        - Combined: P/E × P/B ≤ 22.5

        Args:
            financials: Dict with balance_sheet, income_statement
            current_price: Current stock price (if available)

        Returns:
            Dict with Graham Number and valuation metrics
        """
        try:
            bs = financials.get('balance_sheet', pd.DataFrame())
            income = financials.get('income_statement', pd.DataFrame())

            if bs.empty or income.empty:
                return {'graham_number': None, 'error': 'Insufficient data'}

            bs_latest = bs.iloc[0]
            income_latest = income.iloc[0]

            # Get EPS
            eps = income_latest.get('EarningsPerShareBasic',
                income_latest.get('EarningsPerShareDiluted', np.nan))

            # Calculate BVPS
            equity = bs_latest.get('StockholdersEquity', np.nan)
            shares = income_latest.get('WeightedAverageNumberOfSharesOutstandingBasic',
                income_latest.get('WeightedAverageNumberOfDilutedSharesOutstanding', np.nan))

            if pd.notna(eps) and pd.notna(equity) and pd.notna(shares) and shares > 0:
                bvps = equity / shares

                # Calculate Graham Number
                if eps > 0 and bvps > 0:
                    graham_number = np.sqrt(22.5 * eps * bvps)

                    result = {
                        'graham_number': graham_number,
                        'eps': eps,
                        'bvps': bvps
                    }

                    # If current price available, calculate metrics
                    if current_price is not None and current_price > 0:
                        pe_ratio = current_price / eps if eps > 0 else None
                        pb_ratio = current_price / bvps if bvps > 0 else None
                        discount = ((graham_number - current_price) / graham_number * 100) if graham_number > 0 else None

                        result.update({
                            'current_price': current_price,
                            'pe_ratio': pe_ratio,
                            'pb_ratio': pb_ratio,
                            'price_to_graham': (current_price / graham_number * 100) if graham_number > 0 else None,
                            'discount_pct': discount,
                            'meets_graham_criteria': (
                                pe_ratio is not None and
                                pb_ratio is not None and
                                pe_ratio <= 15 and
                                pb_ratio <= 1.5
                            ),
                            'valuation': get_graham_valuation(current_price, graham_number)
                        })

                    return result
                else:
                    return {'graham_number': None, 'error': 'Negative or zero EPS/BVPS'}
            else:
                return {'graham_number': None, 'error': 'Missing EPS or equity data'}

        except Exception as e:
            logger.error(f"Error calculating Graham Number: {e}")
            return {'graham_number': None, 'error': str(e)}


def get_piotroski_interpretation(score: int) -> str:
    """Interpret Piotroski F-Score"""
    if score >= 8:
        return "Very Strong - High quality value stock"
    elif score >= 6:
        return "Strong - Good financial health"
    elif score >= 4:
        return "Moderate - Average quality"
    elif score >= 2:
        return "Weak - Poor financial health"
    else:
        return "Very Weak - Avoid"


def get_graham_valuation(price: float, graham_number: float) -> str:
    """Determine valuation relative to Graham Number"""
    if graham_number <= 0:
        return "N/A"

    ratio = price / graham_number

    if ratio < 0.67:
        return "Deeply Undervalued"
    elif ratio < 0.85:
        return "Undervalued"
    elif ratio < 1.15:
        return "Fairly Valued"
    elif ratio < 1.33:
        return "Overvalued"
    else:
        return "Significantly Overvalued"
