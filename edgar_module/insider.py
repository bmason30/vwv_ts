"""
Insider Transaction Analysis
Parses Form 4 filings to identify insider buying and selling activity
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from .session import EDGARSession
from .metadata import SubmissionsParser

logger = logging.getLogger(__name__)


class InsiderTransactionAnalyzer:
    """
    Analyzes insider transactions from Form 4 filings
    """

    def __init__(self, session: EDGARSession):
        """
        Initialize insider transaction analyzer

        Args:
            session: EDGARSession instance
        """
        self.session = session
        self.submissions = SubmissionsParser(session)

    def get_recent_insider_activity(
        self,
        cik: str,
        days_back: int = 180
    ) -> pd.DataFrame:
        """
        Get recent Form 4 insider transaction filings

        Args:
            cik: Company CIK
            days_back: How many days back to look (default: 180)

        Returns:
            DataFrame with Form 4 filings
        """
        try:
            # Get filings
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

            filings = self.submissions.get_filings(
                cik,
                form_type='4',
                start_date=start_date,
                end_date=end_date
            )

            return filings

        except Exception as e:
            logger.error(f"Error getting insider activity for CIK {cik}: {e}")
            return pd.DataFrame()

    def analyze_insider_sentiment(
        self,
        cik: str,
        days_back: int = 180
    ) -> Dict[str, Any]:
        """
        Analyze insider sentiment from Form 4 filings

        Note: Full parsing of Form 4 XML requires additional complexity.
        This provides a simplified analysis based on filing frequency.

        Args:
            cik: Company CIK
            days_back: How many days back to analyze

        Returns:
            Dict with insider activity metrics
        """
        try:
            filings = self.get_recent_insider_activity(cik, days_back)

            if filings.empty:
                return {
                    'form4_count': 0,
                    'activity_level': 'None',
                    'recent_filings': []
                }

            # Count filings
            form4_count = len(filings)

            # Determine activity level
            if form4_count >= 10:
                activity_level = 'High'
            elif form4_count >= 5:
                activity_level = 'Moderate'
            elif form4_count >= 1:
                activity_level = 'Low'
            else:
                activity_level = 'None'

            # Get recent filing dates
            recent_filings = []
            if 'filingDate' in filings.columns:
                recent_dates = filings.head(5)['filingDate'].tolist()
                recent_filings = [str(d) for d in recent_dates]

            return {
                'form4_count': form4_count,
                'activity_level': activity_level,
                'recent_filings': recent_filings,
                'days_analyzed': days_back,
                'interpretation': self._interpret_activity(form4_count, days_back)
            }

        except Exception as e:
            logger.error(f"Error analyzing insider sentiment: {e}")
            return {'error': str(e)}

    def _interpret_activity(self, count: int, days: int) -> str:
        """Interpret insider activity level"""
        monthly_rate = (count / days) * 30

        if monthly_rate >= 5:
            return "Very active insider trading (bullish or bearish signal)"
        elif monthly_rate >= 2:
            return "Moderate insider activity"
        elif monthly_rate >= 0.5:
            return "Low insider activity"
        else:
            return "Minimal insider activity"

    def get_insider_summary(self, cik: str) -> str:
        """
        Get a text summary of insider activity

        Args:
            cik: Company CIK

        Returns:
            Summary string
        """
        analysis = self.analyze_insider_sentiment(cik)

        if 'error' in analysis:
            return "N/A"

        count = analysis.get('form4_count', 0)
        level = analysis.get('activity_level', 'Unknown')

        if count == 0:
            return "No Activity"
        else:
            return f"{count} Form 4s ({level})"


def estimate_insider_position_size(
    form4_count: int,
    market_cap: Optional[float] = None
) -> str:
    """
    Estimate relative insider position size based on filing frequency
    and market cap

    This is a heuristic - actual transaction amounts require XML parsing

    Args:
        form4_count: Number of Form 4 filings
        market_cap: Company market capitalization

    Returns:
        Estimated position size description
    """
    if form4_count == 0:
        return "None"

    # Adjust for market cap if available
    if market_cap is not None:
        if market_cap > 100_000_000_000:  # Large cap
            threshold_high = 15
            threshold_mod = 8
        elif market_cap > 10_000_000_000:  # Mid cap
            threshold_high = 10
            threshold_mod = 5
        else:  # Small cap
            threshold_high = 7
            threshold_mod = 3
    else:
        threshold_high = 10
        threshold_mod = 5

    if form4_count >= threshold_high:
        return "Significant"
    elif form4_count >= threshold_mod:
        return "Moderate"
    else:
        return "Small"
