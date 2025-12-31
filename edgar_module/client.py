"""
Layer C: The vwv Integration Layer (The "Controller")
Public-facing API for the EDGAR module
"""
import logging
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from .session import EDGARSession
from .metadata import CIKLookup, SubmissionsParser
from .financials import FinancialsParser
from .comparison import CompanyComparison
from .exceptions import CIKNotFound, DataNotAvailable
from .config import DEFAULT_USER_AGENT

logger = logging.getLogger(__name__)


class EDGARClient:
    """
    High-level client for accessing SEC EDGAR financial data

    This is the main entry point for the edgar_module. It provides
    simple, intuitive methods for retrieving and analyzing financial data.

    Example:
        >>> client = EDGARClient(user_agent="MyApp/1.0 (contact@example.com)")
        >>>
        >>> # Get company financials by ticker
        >>> financials = client.get_financials('AAPL')
        >>> balance_sheet = financials['balance_sheet']
        >>>
        >>> # Compare metrics across companies
        >>> comparison = client.compare_metrics(['AAPL', 'MSFT', 'GOOGL'], 'Revenues')
    """

    def __init__(
        self,
        user_agent: Optional[str] = None,
        enable_cache: bool = True
    ):
        """
        Initialize the EDGAR client

        Args:
            user_agent: Custom User-Agent string for SEC requests.
                       Should include app name and contact info.
                       If None, uses a default User-Agent.
            enable_cache: Whether to enable local file caching (default: True)
        """
        self.session = EDGARSession(user_agent=user_agent, enable_cache=enable_cache)
        self.cik_lookup = CIKLookup(self.session)
        self.submissions = SubmissionsParser(self.session)
        self.financials = FinancialsParser(self.session)
        self.comparison = CompanyComparison(self.session)

        logger.info("EDGARClient initialized")

    def set_user_agent(self, user_agent: str) -> None:
        """
        Update the User-Agent string

        Args:
            user_agent: New User-Agent string
        """
        self.session.user_agent = user_agent
        self.session.session.headers.update({'User-Agent': user_agent})
        logger.info(f"User-Agent updated to: {user_agent}")

    def ticker_to_cik(self, ticker: str) -> str:
        """
        Convert a ticker symbol to a CIK

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')

        Returns:
            10-digit CIK string

        Raises:
            CIKNotFound: If ticker is not found
        """
        return self.cik_lookup.ticker_to_cik(ticker)

    def get_company_info(self, ticker_or_cik: str) -> Dict[str, Any]:
        """
        Get company metadata

        Args:
            ticker_or_cik: Stock ticker or CIK

        Returns:
            Dictionary with company metadata
        """
        cik = self._resolve_identifier(ticker_or_cik)
        return self.submissions.get_company_metadata(cik)

    def get_filings(
        self,
        ticker_or_cik: str,
        form_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get company filings with optional filtering

        Args:
            ticker_or_cik: Stock ticker or CIK
            form_type: Filter by form type (e.g., '10-K', '10-Q', '8-K')
            start_date: Filter filings after this date (YYYY-MM-DD)
            end_date: Filter filings before this date (YYYY-MM-DD)

        Returns:
            DataFrame with filing information
        """
        cik = self._resolve_identifier(ticker_or_cik)
        return self.submissions.get_filings(cik, form_type, start_date, end_date)

    def get_financials(
        self,
        ticker_or_cik: str,
        annual: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Get all financial statements for a company

        Args:
            ticker_or_cik: Stock ticker or CIK
            annual: If True, return annual data (10-K), else quarterly (10-Q)

        Returns:
            Dictionary with keys:
                - 'balance_sheet': Balance sheet DataFrame
                - 'income_statement': Income statement DataFrame
                - 'cash_flow': Cash flow statement DataFrame
        """
        cik = self._resolve_identifier(ticker_or_cik)
        return self.financials.get_all_financials(cik, annual)

    def get_balance_sheet(
        self,
        ticker_or_cik: str,
        annual: bool = True
    ) -> pd.DataFrame:
        """
        Get balance sheet data

        Args:
            ticker_or_cik: Stock ticker or CIK
            annual: If True, return annual data (10-K), else quarterly (10-Q)

        Returns:
            DataFrame with balance sheet line items
        """
        cik = self._resolve_identifier(ticker_or_cik)
        return self.financials.get_balance_sheet(cik, annual)

    def get_income_statement(
        self,
        ticker_or_cik: str,
        annual: bool = True
    ) -> pd.DataFrame:
        """
        Get income statement data

        Args:
            ticker_or_cik: Stock ticker or CIK
            annual: If True, return annual data (10-K), else quarterly (10-Q)

        Returns:
            DataFrame with income statement line items
        """
        cik = self._resolve_identifier(ticker_or_cik)
        return self.financials.get_income_statement(cik, annual)

    def get_cash_flow(
        self,
        ticker_or_cik: str,
        annual: bool = True
    ) -> pd.DataFrame:
        """
        Get cash flow statement data

        Args:
            ticker_or_cik: Stock ticker or CIK
            annual: If True, return annual data (10-K), else quarterly (10-Q)

        Returns:
            DataFrame with cash flow statement line items
        """
        cik = self._resolve_identifier(ticker_or_cik)
        return self.financials.get_cash_flow_statement(cik, annual)

    def get_key_metrics(
        self,
        ticker_or_cik: str,
        annual: bool = True
    ) -> pd.DataFrame:
        """
        Calculate key financial metrics

        Args:
            ticker_or_cik: Stock ticker or CIK
            annual: If True, use annual data (10-K), else quarterly (10-Q)

        Returns:
            DataFrame with calculated metrics (ROE, ROA, margins, etc.)
        """
        cik = self._resolve_identifier(ticker_or_cik)
        return self.financials.get_key_metrics(cik, annual)

    def get_metric_timeseries(
        self,
        ticker_or_cik: str,
        concept: str,
        annual: bool = True
    ) -> pd.DataFrame:
        """
        Get time series for a specific financial concept

        Args:
            ticker_or_cik: Stock ticker or CIK
            concept: XBRL concept name (e.g., 'Assets', 'Revenues')
            annual: If True, return annual data (10-K), else quarterly (10-Q)

        Returns:
            DataFrame with time series data
        """
        cik = self._resolve_identifier(ticker_or_cik)
        form_type = '10-K' if annual else '10-Q'
        return self.financials.get_concept_timeseries(cik, concept, form_type=form_type)

    def compare_metrics(
        self,
        tickers_or_ciks: List[str],
        concept: str,
        annual_only: bool = True
    ) -> pd.DataFrame:
        """
        Compare a specific metric across multiple companies

        Args:
            tickers_or_ciks: List of stock tickers or CIKs
            concept: XBRL concept name (e.g., 'Assets', 'Revenues')
            annual_only: If True, only include annual data

        Returns:
            DataFrame with companies as columns and dates as index
        """
        ciks = [self._resolve_identifier(t) for t in tickers_or_ciks]
        return self.comparison.compare_companies(ciks, concept, annual_only=annual_only)

    def compare_peers(
        self,
        tickers_or_ciks: List[str],
        concepts: List[str],
        annual_only: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Compare multiple metrics across multiple companies

        Args:
            tickers_or_ciks: List of stock tickers or CIKs
            concepts: List of XBRL concept names
            annual_only: If True, only include annual data

        Returns:
            Dictionary mapping concept names to comparison DataFrames
        """
        ciks = [self._resolve_identifier(t) for t in tickers_or_ciks]
        return self.comparison.get_peer_comparison(ciks, concepts, annual_only)

    def get_industry_percentile(
        self,
        ticker_or_cik: str,
        concept: str,
        year: int
    ) -> Dict[str, Any]:
        """
        Get a company's percentile ranking within its industry

        Args:
            ticker_or_cik: Stock ticker or CIK
            concept: XBRL concept name (e.g., 'Assets', 'Revenues')
            year: Calendar year

        Returns:
            Dictionary with percentile information
        """
        cik = self._resolve_identifier(ticker_or_cik)
        return self.comparison.get_industry_percentiles(cik, concept, year)

    def search_filings(
        self,
        ticker_or_cik: str,
        keywords: Optional[List[str]] = None,
        form_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Search filings (currently returns filtered filings by type)

        Note: Full-text search of filing content is not yet implemented.
        This method currently filters by form type only.

        Args:
            ticker_or_cik: Stock ticker or CIK
            keywords: Keywords to search for (NOT YET IMPLEMENTED)
            form_type: Filter by form type

        Returns:
            DataFrame with matching filings
        """
        if keywords:
            logger.warning("Keyword search is not yet implemented. Filtering by form_type only.")

        return self.get_filings(ticker_or_cik, form_type=form_type)

    def get_filing_url(
        self,
        ticker_or_cik: str,
        form_type: str = '10-K',
        latest: bool = True
    ) -> str:
        """
        Get the URL for a specific filing

        Args:
            ticker_or_cik: Stock ticker or CIK
            form_type: Form type (e.g., '10-K', '10-Q')
            latest: If True, get the most recent filing

        Returns:
            URL to the filing on SEC website
        """
        cik = self._resolve_identifier(ticker_or_cik)
        filing = self.submissions.get_latest_filing(cik, form_type)

        accession = filing['accessionNumber'].replace('-', '')
        primary_doc = filing['primaryDocument']

        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"
        return url

    def clear_cache(self) -> int:
        """
        Clear all cached data

        Returns:
            Number of cache files deleted
        """
        return self.session.clear_cache()

    def _resolve_identifier(self, ticker_or_cik: str) -> str:
        """
        Resolve a ticker or CIK to a 10-digit CIK

        Args:
            ticker_or_cik: Stock ticker or CIK

        Returns:
            10-digit CIK string
        """
        identifier = ticker_or_cik.strip()

        # Check if it's already a CIK (numeric)
        if identifier.isdigit():
            return str(int(identifier)).zfill(10)

        # Otherwise, treat as ticker and convert to CIK
        return self.cik_lookup.ticker_to_cik(identifier)

    def close(self) -> None:
        """Close the client session"""
        self.session.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def __repr__(self) -> str:
        """String representation"""
        return f"EDGARClient(user_agent='{self.session.user_agent}', cache_enabled={self.session.enable_cache})"


# Convenience function for quick access
def get_financials(
    ticker: str,
    annual: bool = True,
    user_agent: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to quickly get financials for a company

    Args:
        ticker: Stock ticker symbol
        annual: If True, return annual data (10-K), else quarterly (10-Q)
        user_agent: Optional custom User-Agent

    Returns:
        Dictionary with financial statements

    Example:
        >>> from edgar_module import get_financials
        >>> financials = get_financials('AAPL')
        >>> print(financials['balance_sheet'].head())
    """
    with EDGARClient(user_agent=user_agent) as client:
        return client.get_financials(ticker, annual)
