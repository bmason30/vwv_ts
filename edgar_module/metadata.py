"""
Layer B: The Normalization Engine (The "Processor") - Metadata Component
Handles ticker-to-CIK mapping and submissions metadata parsing
"""
import logging
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime

from .session import EDGARSession
from .exceptions import CIKNotFound, DataNotAvailable

logger = logging.getLogger(__name__)


class CIKLookup:
    """
    Utility for converting stock tickers to 10-digit Central Index Keys (CIKs)
    Uses SEC's master ticker map with local caching
    """

    def __init__(self, session: EDGARSession):
        """
        Initialize CIK lookup

        Args:
            session: EDGARSession instance for API calls
        """
        self.session = session
        self._ticker_map: Optional[Dict[str, int]] = None
        self._reverse_map: Optional[Dict[int, str]] = None

    def _load_ticker_map(self) -> None:
        """Load and cache the ticker-to-CIK mapping from SEC"""
        if self._ticker_map is not None:
            return

        logger.info("Loading ticker-to-CIK mapping from SEC...")
        data = self.session.get_ticker_map()

        # The SEC ticker map format: {0: {ticker, cik, title}, 1: {...}, ...}
        self._ticker_map = {}
        self._reverse_map = {}

        for entry in data.values():
            ticker = entry.get('ticker', '').upper()
            cik = entry.get('cik_str')

            if ticker and cik is not None:
                self._ticker_map[ticker] = int(cik)
                self._reverse_map[int(cik)] = ticker

        logger.info(f"Loaded {len(self._ticker_map)} ticker mappings")

    def ticker_to_cik(self, ticker: str) -> str:
        """
        Convert a ticker symbol to a 10-digit CIK

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')

        Returns:
            10-digit zero-padded CIK string

        Raises:
            CIKNotFound: If ticker is not found in SEC database
        """
        self._load_ticker_map()

        ticker_upper = ticker.upper().strip()

        if ticker_upper not in self._ticker_map:
            raise CIKNotFound(ticker, f"Ticker '{ticker}' not found in SEC database")

        cik_int = self._ticker_map[ticker_upper]
        return str(cik_int).zfill(10)

    def cik_to_ticker(self, cik: int) -> Optional[str]:
        """
        Convert a CIK to a ticker symbol (if available)

        Args:
            cik: CIK as integer or string

        Returns:
            Ticker symbol or None if not found
        """
        self._load_ticker_map()

        cik_int = int(cik)
        return self._reverse_map.get(cik_int)

    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get basic company information from ticker map

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with ticker, cik, and title
        """
        self._load_ticker_map()
        ticker_upper = ticker.upper().strip()

        data = self.session.get_ticker_map()

        for entry in data.values():
            if entry.get('ticker', '').upper() == ticker_upper:
                return {
                    'ticker': entry.get('ticker'),
                    'cik': str(entry.get('cik_str')).zfill(10),
                    'title': entry.get('title'),
                }

        raise CIKNotFound(ticker)


class SubmissionsParser:
    """
    Parser for SEC submissions/filings data
    Provides filtering and normalization of filing metadata
    """

    def __init__(self, session: EDGARSession):
        """
        Initialize submissions parser

        Args:
            session: EDGARSession instance for API calls
        """
        self.session = session

    def get_filings(
        self,
        cik: str,
        form_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get filings for a company with optional filtering

        Args:
            cik: 10-digit CIK or shorter CIK (will be padded)
            form_type: Filter by form type (e.g., '10-K', '10-Q', '8-K')
            start_date: Filter filings after this date (YYYY-MM-DD)
            end_date: Filter filings before this date (YYYY-MM-DD)

        Returns:
            DataFrame with filing information
        """
        data = self.session.get_submissions(cik)

        # Extract the filings from the response
        recent_filings = data.get('filings', {}).get('recent', {})

        if not recent_filings:
            raise DataNotAvailable(f"No filings found for CIK {cik}")

        # Convert to DataFrame
        df = pd.DataFrame(recent_filings)

        # Ensure we have the required columns
        required_cols = ['accessionNumber', 'filingDate', 'form', 'primaryDocument']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing columns in filings data: {missing_cols}")

        # Convert filingDate to datetime
        if 'filingDate' in df.columns:
            df['filingDate'] = pd.to_datetime(df['filingDate'])

        # Filter by form type
        if form_type:
            df = df[df['form'] == form_type].copy()

        # Filter by date range
        if start_date and 'filingDate' in df.columns:
            start_dt = pd.to_datetime(start_date)
            df = df[df['filingDate'] >= start_dt].copy()

        if end_date and 'filingDate' in df.columns:
            end_dt = pd.to_datetime(end_date)
            df = df[df['filingDate'] <= end_dt].copy()

        # Sort by filing date (most recent first)
        if 'filingDate' in df.columns:
            df = df.sort_values('filingDate', ascending=False)

        return df.reset_index(drop=True)

    def get_latest_filing(self, cik: str, form_type: str = '10-K') -> Dict[str, Any]:
        """
        Get the most recent filing of a specific type

        Args:
            cik: 10-digit CIK or shorter CIK (will be padded)
            form_type: Form type to search for (default: '10-K')

        Returns:
            Dictionary with filing information

        Raises:
            DataNotAvailable: If no filings of the specified type are found
        """
        df = self.get_filings(cik, form_type=form_type)

        if df.empty:
            raise DataNotAvailable(f"No {form_type} filings found for CIK {cik}")

        # Return the most recent filing as a dictionary
        return df.iloc[0].to_dict()

    def get_company_metadata(self, cik: str) -> Dict[str, Any]:
        """
        Get company metadata from submissions endpoint

        Args:
            cik: 10-digit CIK or shorter CIK (will be padded)

        Returns:
            Dictionary with company metadata
        """
        data = self.session.get_submissions(cik)

        # Extract relevant metadata
        metadata = {
            'cik': data.get('cik'),
            'entityType': data.get('entityType'),
            'sic': data.get('sic'),
            'sicDescription': data.get('sicDescription'),
            'name': data.get('name'),
            'tickers': data.get('tickers', []),
            'exchanges': data.get('exchanges', []),
            'ein': data.get('ein'),
            'stateOfIncorporation': data.get('stateOfIncorporation'),
            'fiscalYearEnd': data.get('fiscalYearEnd'),
            'category': data.get('category'),
            'phone': data.get('phone'),
            'addresses': {
                'mailing': data.get('addresses', {}).get('mailing', {}),
                'business': data.get('addresses', {}).get('business', {}),
            },
            'formerNames': data.get('formerNames', []),
        }

        return metadata

    def get_filing_count_by_type(self, cik: str) -> pd.Series:
        """
        Get count of filings by form type

        Args:
            cik: 10-digit CIK or shorter CIK (will be padded)

        Returns:
            Series with form types as index and counts as values
        """
        df = self.get_filings(cik)

        if 'form' not in df.columns:
            return pd.Series(dtype=int)

        return df['form'].value_counts()

    def get_filings_timeline(self, cik: str, form_type: Optional[str] = None) -> pd.DataFrame:
        """
        Get a timeline view of filings with year aggregation

        Args:
            cik: 10-digit CIK or shorter CIK (will be padded)
            form_type: Optional filter by form type

        Returns:
            DataFrame with year and count columns
        """
        df = self.get_filings(cik, form_type=form_type)

        if df.empty or 'filingDate' not in df.columns:
            return pd.DataFrame(columns=['year', 'count'])

        df['year'] = df['filingDate'].dt.year
        timeline = df.groupby('year').size().reset_index(name='count')
        timeline = timeline.sort_values('year', ascending=False)

        return timeline
