"""
Layer B: The Normalization Engine (The "Processor") - Financials Component
Handles parsing and normalization of XBRL financial data
Extracts Balance Sheets, Income Statements, and Cash Flow Statements
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from .session import EDGARSession
from .exceptions import DataNotAvailable, ParseError
from .config import COMMON_FINANCIAL_CONCEPTS, SUPPORTED_TAXONOMIES

logger = logging.getLogger(__name__)


class FinancialsParser:
    """
    Parser for SEC Company Facts API
    Extracts structured financial data from XBRL
    """

    def __init__(self, session: EDGARSession):
        """
        Initialize financials parser

        Args:
            session: EDGARSession instance for API calls
        """
        self.session = session

    def get_company_facts(self, cik: str) -> Dict[str, Any]:
        """
        Get all company facts (raw XBRL data)

        Args:
            cik: 10-digit CIK or shorter CIK (will be padded)

        Returns:
            Raw company facts dictionary
        """
        return self.session.get_company_facts(cik)

    def _extract_concept_data(
        self,
        facts: Dict[str, Any],
        concept: str,
        taxonomy: str = 'us-gaap'
    ) -> Optional[pd.DataFrame]:
        """
        Extract data for a specific XBRL concept

        Args:
            facts: Company facts dictionary
            concept: XBRL concept name (e.g., 'Assets', 'Revenues')
            taxonomy: XBRL taxonomy (default: 'us-gaap')

        Returns:
            DataFrame with concept data or None if not found
        """
        try:
            taxonomy_data = facts.get('facts', {}).get(taxonomy, {})

            if concept not in taxonomy_data:
                return None

            concept_data = taxonomy_data[concept]

            # Get the units (usually USD for financial data)
            units = concept_data.get('units', {})

            # Try USD first, then other currencies
            currency_keys = ['USD', 'usd']
            data_list = None

            for key in currency_keys:
                if key in units:
                    data_list = units[key]
                    break

            # If no USD, try the first available unit
            if data_list is None and units:
                first_unit = next(iter(units.keys()))
                data_list = units[first_unit]
                logger.debug(f"Using unit '{first_unit}' for concept {concept}")

            if not data_list:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data_list)

            return df

        except Exception as e:
            logger.error(f"Error extracting concept {concept}: {e}")
            return None

    def _normalize_financials(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize financial data by handling duplicates and selecting appropriate values

        Args:
            df: Raw concept DataFrame

        Returns:
            Normalized DataFrame
        """
        if df is None or df.empty:
            return pd.DataFrame()

        # Convert end date to datetime
        if 'end' in df.columns:
            df['end'] = pd.to_datetime(df['end'])

        # Filter for annual data (10-K) and quarterly data (10-Q) by form
        if 'form' in df.columns:
            # Prioritize 10-K and 10-Q filings
            df = df[df['form'].isin(['10-K', '10-Q', '10-K/A', '10-Q/A'])].copy()

        # Handle duplicates - keep the most recent filing for each end date
        if 'end' in df.columns and 'filed' in df.columns:
            df['filed'] = pd.to_datetime(df['filed'])
            df = df.sort_values('filed', ascending=False)
            df = df.drop_duplicates(subset=['end'], keep='first')

        # Sort by end date
        if 'end' in df.columns:
            df = df.sort_values('end', ascending=False)

        return df.reset_index(drop=True)

    def get_concept_timeseries(
        self,
        cik: str,
        concept: str,
        taxonomy: str = 'us-gaap',
        form_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get time series data for a specific financial concept

        Args:
            cik: 10-digit CIK or shorter CIK (will be padded)
            concept: XBRL concept name (e.g., 'Assets', 'Revenues')
            taxonomy: XBRL taxonomy (default: 'us-gaap')
            form_type: Filter by form type ('10-K' for annual, '10-Q' for quarterly)

        Returns:
            DataFrame with columns: end, value, form, filed, fy, fp, frame
        """
        facts = self.get_company_facts(cik)
        df = self._extract_concept_data(facts, concept, taxonomy)

        if df is None or df.empty:
            logger.warning(f"No data found for concept {concept}")
            return pd.DataFrame()

        df = self._normalize_financials(df)

        # Filter by form type if specified
        if form_type and 'form' in df.columns:
            df = df[df['form'] == form_type].copy()

        # Select relevant columns
        cols_to_keep = ['end', 'val', 'form', 'filed', 'fy', 'fp', 'frame']
        available_cols = [col for col in cols_to_keep if col in df.columns]

        result = df[available_cols].copy()

        # Rename 'val' to 'value' for clarity
        if 'val' in result.columns:
            result = result.rename(columns={'val': 'value'})

        return result

    def get_balance_sheet(
        self,
        cik: str,
        annual: bool = True
    ) -> pd.DataFrame:
        """
        Get balance sheet data

        Args:
            cik: 10-digit CIK or shorter CIK (will be padded)
            annual: If True, return annual data (10-K), else quarterly (10-Q)

        Returns:
            DataFrame with balance sheet line items as columns
        """
        form_type = '10-K' if annual else '10-Q'

        # Define balance sheet concepts
        bs_concepts = {
            'Assets': 'Assets',
            'AssetsCurrent': 'AssetsCurrent',
            'LiabilitiesAndStockholdersEquity': 'LiabilitiesAndStockholdersEquity',
            'Liabilities': 'Liabilities',
            'LiabilitiesCurrent': 'LiabilitiesCurrent',
            'StockholdersEquity': 'StockholdersEquity',
            'CashAndCashEquivalentsAtCarryingValue': 'CashAndCashEquivalentsAtCarryingValue',
            'PropertyPlantAndEquipmentNet': 'PropertyPlantAndEquipmentNet',
            'Goodwill': 'Goodwill',
            'IntangibleAssetsNetExcludingGoodwill': 'IntangibleAssetsNetExcludingGoodwill',
            'LongTermDebt': 'LongTermDebt',
        }

        # Fetch data for each concept
        balance_sheet_data = {}
        facts = self.get_company_facts(cik)

        for label, concept in bs_concepts.items():
            df = self._extract_concept_data(facts, concept, 'us-gaap')
            if df is not None and not df.empty:
                df = self._normalize_financials(df)

                # Filter by form type
                if 'form' in df.columns:
                    df = df[df['form'] == form_type].copy()

                if not df.empty and 'end' in df.columns and 'val' in df.columns:
                    # Create a series indexed by end date
                    series = df.set_index('end')['val']
                    balance_sheet_data[label] = series

        if not balance_sheet_data:
            raise DataNotAvailable(f"No balance sheet data found for CIK {cik}")

        # Combine into a single DataFrame
        result = pd.DataFrame(balance_sheet_data)
        result.index.name = 'date'

        # Sort by date (most recent first)
        result = result.sort_index(ascending=False)

        return result

    def get_income_statement(
        self,
        cik: str,
        annual: bool = True
    ) -> pd.DataFrame:
        """
        Get income statement data

        Args:
            cik: 10-digit CIK or shorter CIK (will be padded)
            annual: If True, return annual data (10-K), else quarterly (10-Q)

        Returns:
            DataFrame with income statement line items as columns
        """
        form_type = '10-K' if annual else '10-Q'

        # Define income statement concepts
        is_concepts = {
            'Revenues': 'Revenues',
            'RevenueFromContractWithCustomerExcludingAssessedTax': 'RevenueFromContractWithCustomerExcludingAssessedTax',
            'CostOfRevenue': 'CostOfRevenue',
            'GrossProfit': 'GrossProfit',
            'OperatingExpenses': 'OperatingExpenses',
            'OperatingIncomeLoss': 'OperatingIncomeLoss',
            'InterestExpense': 'InterestExpense',
            'IncomeTaxExpenseBenefit': 'IncomeTaxExpenseBenefit',
            'NetIncomeLoss': 'NetIncomeLoss',
            'EarningsPerShareBasic': 'EarningsPerShareBasic',
            'EarningsPerShareDiluted': 'EarningsPerShareDiluted',
            'WeightedAverageNumberOfSharesOutstandingBasic': 'WeightedAverageNumberOfSharesOutstandingBasic',
            'WeightedAverageNumberOfDilutedSharesOutstanding': 'WeightedAverageNumberOfDilutedSharesOutstanding',
        }

        # Fetch data for each concept
        income_statement_data = {}
        facts = self.get_company_facts(cik)

        for label, concept in is_concepts.items():
            df = self._extract_concept_data(facts, concept, 'us-gaap')
            if df is not None and not df.empty:
                df = self._normalize_financials(df)

                # Filter by form type
                if 'form' in df.columns:
                    df = df[df['form'] == form_type].copy()

                if not df.empty and 'end' in df.columns and 'val' in df.columns:
                    # Create a series indexed by end date
                    series = df.set_index('end')['val']
                    income_statement_data[label] = series

        if not income_statement_data:
            raise DataNotAvailable(f"No income statement data found for CIK {cik}")

        # Combine into a single DataFrame
        result = pd.DataFrame(income_statement_data)
        result.index.name = 'date'

        # Sort by date (most recent first)
        result = result.sort_index(ascending=False)

        return result

    def get_cash_flow_statement(
        self,
        cik: str,
        annual: bool = True
    ) -> pd.DataFrame:
        """
        Get cash flow statement data

        Args:
            cik: 10-digit CIK or shorter CIK (will be padded)
            annual: If True, return annual data (10-K), else quarterly (10-Q)

        Returns:
            DataFrame with cash flow statement line items as columns
        """
        form_type = '10-K' if annual else '10-Q'

        # Define cash flow statement concepts
        cf_concepts = {
            'NetCashProvidedByUsedInOperatingActivities': 'NetCashProvidedByUsedInOperatingActivities',
            'NetCashProvidedByUsedInInvestingActivities': 'NetCashProvidedByUsedInInvestingActivities',
            'NetCashProvidedByUsedInFinancingActivities': 'NetCashProvidedByUsedInFinancingActivities',
            'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect':
                'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect',
            'DepreciationDepletionAndAmortization': 'DepreciationDepletionAndAmortization',
            'PaymentsToAcquirePropertyPlantAndEquipment': 'PaymentsToAcquirePropertyPlantAndEquipment',
            'PaymentsForRepurchaseOfCommonStock': 'PaymentsForRepurchaseOfCommonStock',
            'PaymentsOfDividends': 'PaymentsOfDividends',
        }

        # Fetch data for each concept
        cash_flow_data = {}
        facts = self.get_company_facts(cik)

        for label, concept in cf_concepts.items():
            df = self._extract_concept_data(facts, concept, 'us-gaap')
            if df is not None and not df.empty:
                df = self._normalize_financials(df)

                # Filter by form type
                if 'form' in df.columns:
                    df = df[df['form'] == form_type].copy()

                if not df.empty and 'end' in df.columns and 'val' in df.columns:
                    # Create a series indexed by end date
                    series = df.set_index('end')['val']
                    cash_flow_data[label] = series

        if not cash_flow_data:
            raise DataNotAvailable(f"No cash flow statement data found for CIK {cik}")

        # Combine into a single DataFrame
        result = pd.DataFrame(cash_flow_data)
        result.index.name = 'date'

        # Sort by date (most recent first)
        result = result.sort_index(ascending=False)

        return result

    def get_all_financials(
        self,
        cik: str,
        annual: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Get all financial statements at once

        Args:
            cik: 10-digit CIK or shorter CIK (will be padded)
            annual: If True, return annual data (10-K), else quarterly (10-Q)

        Returns:
            Dictionary with keys 'balance_sheet', 'income_statement', 'cash_flow'
        """
        result = {}

        try:
            result['balance_sheet'] = self.get_balance_sheet(cik, annual)
        except DataNotAvailable:
            logger.warning(f"Balance sheet not available for CIK {cik}")
            result['balance_sheet'] = pd.DataFrame()

        try:
            result['income_statement'] = self.get_income_statement(cik, annual)
        except DataNotAvailable:
            logger.warning(f"Income statement not available for CIK {cik}")
            result['income_statement'] = pd.DataFrame()

        try:
            result['cash_flow'] = self.get_cash_flow_statement(cik, annual)
        except DataNotAvailable:
            logger.warning(f"Cash flow statement not available for CIK {cik}")
            result['cash_flow'] = pd.DataFrame()

        return result

    def get_key_metrics(self, cik: str, annual: bool = True) -> pd.DataFrame:
        """
        Calculate key financial metrics from available data

        Args:
            cik: 10-digit CIK or shorter CIK (will be padded)
            annual: If True, use annual data (10-K), else quarterly (10-Q)

        Returns:
            DataFrame with key metrics over time
        """
        financials = self.get_all_financials(cik, annual)

        bs = financials.get('balance_sheet', pd.DataFrame())
        income = financials.get('income_statement', pd.DataFrame())

        metrics = pd.DataFrame(index=bs.index if not bs.empty else income.index)

        # Calculate metrics where data is available
        if not bs.empty:
            # Working Capital
            if 'AssetsCurrent' in bs.columns and 'LiabilitiesCurrent' in bs.columns:
                metrics['WorkingCapital'] = bs['AssetsCurrent'] - bs['LiabilitiesCurrent']

            # Debt to Equity Ratio
            if 'LongTermDebt' in bs.columns and 'StockholdersEquity' in bs.columns:
                metrics['DebtToEquityRatio'] = bs['LongTermDebt'] / bs['StockholdersEquity']

            # Current Ratio
            if 'AssetsCurrent' in bs.columns and 'LiabilitiesCurrent' in bs.columns:
                metrics['CurrentRatio'] = bs['AssetsCurrent'] / bs['LiabilitiesCurrent']

        if not income.empty:
            # Profit Margin
            revenue_col = None
            if 'Revenues' in income.columns:
                revenue_col = 'Revenues'
            elif 'RevenueFromContractWithCustomerExcludingAssessedTax' in income.columns:
                revenue_col = 'RevenueFromContractWithCustomerExcludingAssessedTax'

            if revenue_col and 'NetIncomeLoss' in income.columns:
                metrics['ProfitMargin'] = income['NetIncomeLoss'] / income[revenue_col]

            # Gross Margin
            if revenue_col and 'GrossProfit' in income.columns:
                metrics['GrossMargin'] = income['GrossProfit'] / income[revenue_col]

        # Combine balance sheet and income statement for ROE calculation
        if not bs.empty and not income.empty:
            # Return on Equity (ROE)
            if 'NetIncomeLoss' in income.columns and 'StockholdersEquity' in bs.columns:
                # Align the indices
                common_dates = bs.index.intersection(income.index)
                if len(common_dates) > 0:
                    metrics.loc[common_dates, 'ReturnOnEquity'] = (
                        income.loc[common_dates, 'NetIncomeLoss'] /
                        bs.loc[common_dates, 'StockholdersEquity']
                    )

            # Return on Assets (ROA)
            if 'NetIncomeLoss' in income.columns and 'Assets' in bs.columns:
                common_dates = bs.index.intersection(income.index)
                if len(common_dates) > 0:
                    metrics.loc[common_dates, 'ReturnOnAssets'] = (
                        income.loc[common_dates, 'NetIncomeLoss'] /
                        bs.loc[common_dates, 'Assets']
                    )

        return metrics
