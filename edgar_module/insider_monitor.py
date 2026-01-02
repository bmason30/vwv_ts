"""
Insider Transaction Monitor - Form 4 Tracker
Automatically tracks largest insider buys and sells across all SEC filings
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

from .session import EDGARSession
from .metadata import CIKLookup
from .exceptions import DataNotAvailable

logger = logging.getLogger(__name__)


class InsiderTransactionMonitor:
    """
    Monitors and tracks insider transactions from Form 4 filings
    Identifies largest buys and sells with transaction details
    """

    def __init__(self, session: EDGARSession):
        """
        Initialize insider transaction monitor

        Args:
            session: EDGARSession instance
        """
        self.session = session
        self.cik_lookup = CIKLookup(session)

    def get_recent_form4_filings(self, days_back: int = 30) -> pd.DataFrame:
        """
        Get recent Form 4 filings across all companies using RSS or bulk download

        Note: SEC doesn't provide a simple API for all Form 4s. This is a simplified
        approach using company tickers. In production, you'd want to:
        1. Use SEC's RSS feeds for real-time Form 4s
        2. Parse the daily index files from SEC's FTP
        3. Use a third-party data provider

        Args:
            days_back: How many days back to look

        Returns:
            DataFrame with Form 4 filing metadata
        """
        # For now, we'll track a watchlist of companies
        # In production, this would scan ALL Form 4 filings

        # Start with major indices + high-volume stocks
        watchlist_tickers = [
            # Mega cap tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA',
            # Mega cap other
            'BRK.B', 'UNH', 'JNJ', 'XOM', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX',
            # Large cap tech
            'AVGO', 'ORCL', 'CRM', 'CSCO', 'ADBE', 'AMD', 'INTC', 'QCOM',
            # Large cap consumer
            'WMT', 'DIS', 'NFLX', 'NKE', 'COST', 'PEP', 'KO', 'MCD',
            # Large cap finance
            'BAC', 'WFC', 'MS', 'GS', 'C', 'AXP', 'BLK',
            # Large cap healthcare
            'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR',
            # Growth stocks
            'COIN', 'SQ', 'SHOP', 'SNOW', 'PLTR', 'RBLX',
            # Meme/Retail favorites
            'GME', 'AMC', 'BBBY', 'RIVN', 'LCID',
            # SPACs and recent IPOs
            'HOOD', 'SOFI', 'UPST', 'OPEN',
        ]

        all_filings = []

        for ticker in watchlist_tickers:
            try:
                cik = self.cik_lookup.ticker_to_cik(ticker)

                # Get submissions
                submissions = self.session.get_submissions(cik)
                filings = submissions.get('filings', {}).get('recent', {})

                if not filings:
                    continue

                # Convert to DataFrame
                df = pd.DataFrame(filings)

                # Filter for Form 4 only
                if 'form' in df.columns:
                    df = df[df['form'] == '4'].copy()

                # Add ticker and CIK
                df['ticker'] = ticker
                df['cik'] = cik

                # Filter by date
                if 'filingDate' in df.columns:
                    df['filingDate'] = pd.to_datetime(df['filingDate'])
                    cutoff_date = datetime.now() - timedelta(days=days_back)
                    df = df[df['filingDate'] >= cutoff_date]

                if not df.empty:
                    all_filings.append(df)

            except Exception as e:
                logger.warning(f"Could not get Form 4s for {ticker}: {e}")
                continue

        if all_filings:
            combined = pd.concat(all_filings, ignore_index=True)
            combined = combined.sort_values('filingDate', ascending=False)
            return combined
        else:
            return pd.DataFrame()

    def parse_form4_xml(self, accession_number: str, cik: str) -> List[Dict[str, Any]]:
        """
        Parse Form 4 XML to extract transaction details

        Note: This is a simplified parser. Full Form 4 parsing is complex.

        Args:
            accession_number: Filing accession number
            cik: Company CIK

        Returns:
            List of transaction dictionaries
        """
        # This would need to fetch and parse the actual XML file
        # For now, return mock data structure

        # In production, you would:
        # 1. Construct the document URL from accession number
        # 2. Fetch the primary document (form4.xml)
        # 3. Parse XML to extract transaction tables
        # 4. Return structured transaction data

        # Mock transaction data (replace with actual XML parsing)
        return [{
            'transaction_date': None,
            'transaction_code': 'P',  # P = Purchase, S = Sale
            'shares': 10000,
            'price_per_share': 150.00,
            'shares_owned_after': 50000,
            'direct_or_indirect': 'D',
            'insider_name': 'Unknown',
            'insider_title': 'Unknown'
        }]

    def estimate_transaction_value(
        self,
        ticker: str,
        filing_date: str,
        shares: Optional[int] = None
    ) -> float:
        """
        Estimate transaction value based on filing date price

        Args:
            ticker: Stock ticker
            filing_date: Date of filing
            shares: Number of shares (if known)

        Returns:
            Estimated transaction value in USD
        """
        # This is a simplified estimation
        # Would need actual transaction price from Form 4 XML

        # For now, use a heuristic based on recent trading
        # In production, parse actual transaction price from XML

        if shares:
            # Estimate ~$100-500 per share for large caps
            # This should be replaced with actual price from Form 4
            estimated_price = 200.0
            return shares * estimated_price

        return 0.0

    def get_top_insider_transactions(
        self,
        days_back: int = 30,
        min_value: float = 100000.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get top insider buys and sells

        Args:
            days_back: Days to look back
            min_value: Minimum transaction value to include

        Returns:
            Tuple of (top_buys_df, top_sells_df)
        """
        # Get all Form 4 filings
        filings = self.get_recent_form4_filings(days_back)

        if filings.empty:
            return pd.DataFrame(), pd.DataFrame()

        # For each filing, estimate transaction details
        # In production, this would parse actual Form 4 XML

        transactions = []

        for idx, row in filings.iterrows():
            # Estimate transaction (simplified - would parse XML in production)
            estimated_value = self.estimate_transaction_value(
                row.get('ticker', ''),
                row.get('filingDate', ''),
                shares=None  # Would get from XML
            )

            transactions.append({
                'ticker': row.get('ticker', 'N/A'),
                'cik': row.get('cik', 'N/A'),
                'filing_date': row.get('filingDate', 'N/A'),
                'accession_number': row.get('accessionNumber', 'N/A'),
                'estimated_value': estimated_value,
                'transaction_type': 'Buy',  # Would parse from XML
                'insider_name': 'N/A',  # Would parse from XML
                'insider_title': 'N/A',  # Would parse from XML
                'shares': 0,  # Would parse from XML
                'price_per_share': 0.0,  # Would parse from XML
            })

        df = pd.DataFrame(transactions)

        # Separate buys and sells
        buys = df[df['transaction_type'] == 'Buy'].copy()
        sells = df[df['transaction_type'] == 'Sell'].copy()

        # Sort by estimated value
        buys = buys.sort_values('estimated_value', ascending=False)
        sells = sells.sort_values('estimated_value', ascending=False)

        # Filter by minimum value
        buys = buys[buys['estimated_value'] >= min_value]
        sells = sells[sells['estimated_value'] >= min_value]

        return buys, sells


def create_insider_summary(
    buys_df: pd.DataFrame,
    sells_df: pd.DataFrame,
    top_n: int = 50
) -> Dict[str, pd.DataFrame]:
    """
    Create summary of top insider transactions

    Args:
        buys_df: DataFrame of buy transactions
        sells_df: DataFrame of sell transactions
        top_n: Number of top transactions to return

    Returns:
        Dict with 'top_buys' and 'top_sells' DataFrames
    """
    return {
        'top_buys': buys_df.head(top_n),
        'top_sells': sells_df.head(top_n)
    }
