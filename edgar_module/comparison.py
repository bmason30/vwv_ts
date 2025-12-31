"""
Layer B: The Normalization Engine (The "Processor") - Comparison Component
Handles cross-company comparison using Company Concept API
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

from .session import EDGARSession
from .exceptions import DataNotAvailable

logger = logging.getLogger(__name__)


class CompanyComparison:
    """
    Utility for comparing specific financial metrics across multiple companies
    Uses the Company Concept API
    """

    def __init__(self, session: EDGARSession):
        """
        Initialize company comparison

        Args:
            session: EDGARSession instance for API calls
        """
        self.session = session

    def get_concept_for_company(
        self,
        cik: str,
        concept: str,
        taxonomy: str = 'us-gaap'
    ) -> pd.DataFrame:
        """
        Get a specific concept's data for one company

        Args:
            cik: 10-digit CIK or shorter CIK (will be padded)
            concept: XBRL concept name (e.g., 'Assets', 'Revenues')
            taxonomy: XBRL taxonomy (default: 'us-gaap')

        Returns:
            DataFrame with concept data over time
        """
        try:
            # The company_concept endpoint returns data for a specific concept
            data = self.session.get_company_concept(cik, concept)

            # Extract the units (usually USD)
            units = data.get('units', {})

            # Try to get USD data
            usd_data = units.get('USD', units.get('usd', None))

            if usd_data is None and units:
                # Get the first available unit
                first_unit = next(iter(units.keys()))
                usd_data = units[first_unit]
                logger.debug(f"Using unit '{first_unit}' for concept {concept}")

            if not usd_data:
                raise DataNotAvailable(f"No data found for concept {concept} in CIK {cik}")

            # Convert to DataFrame
            df = pd.DataFrame(usd_data)

            # Convert dates
            if 'end' in df.columns:
                df['end'] = pd.to_datetime(df['end'])

            if 'filed' in df.columns:
                df['filed'] = pd.to_datetime(df['filed'])

            # Filter for standard forms (10-K, 10-Q)
            if 'form' in df.columns:
                df = df[df['form'].isin(['10-K', '10-Q', '10-K/A', '10-Q/A'])].copy()

            # Remove duplicates - keep most recent filing for each end date
            if 'end' in df.columns and 'filed' in df.columns:
                df = df.sort_values('filed', ascending=False)
                df = df.drop_duplicates(subset=['end'], keep='first')

            # Sort by end date
            if 'end' in df.columns:
                df = df.sort_values('end', ascending=False)

            return df.reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error getting concept {concept} for CIK {cik}: {e}")
            raise DataNotAvailable(f"Failed to get concept {concept} for CIK {cik}: {str(e)}")

    def compare_companies(
        self,
        ciks: List[str],
        concept: str,
        taxonomy: str = 'us-gaap',
        annual_only: bool = True
    ) -> pd.DataFrame:
        """
        Compare a specific concept across multiple companies

        Args:
            ciks: List of CIKs to compare
            concept: XBRL concept name (e.g., 'Assets', 'Revenues')
            taxonomy: XBRL taxonomy (default: 'us-gaap')
            annual_only: If True, only include annual (10-K) data

        Returns:
            DataFrame with companies as columns and dates as index
        """
        company_data = {}

        for cik in ciks:
            try:
                df = self.get_concept_for_company(cik, concept, taxonomy)

                # Filter for annual data if requested
                if annual_only and 'form' in df.columns:
                    df = df[df['form'].isin(['10-K', '10-K/A'])].copy()

                if not df.empty and 'end' in df.columns and 'val' in df.columns:
                    # Use end date as index and value as data
                    series = df.set_index('end')['val']
                    series.name = cik
                    company_data[cik] = series

            except DataNotAvailable as e:
                logger.warning(f"Data not available for CIK {cik}: {e}")
                continue

        if not company_data:
            raise DataNotAvailable(f"No data available for concept {concept} across specified companies")

        # Combine into a single DataFrame
        result = pd.DataFrame(company_data)
        result.index.name = 'date'

        # Sort by date (most recent first)
        result = result.sort_index(ascending=False)

        return result

    def get_peer_comparison(
        self,
        ciks: List[str],
        concepts: List[str],
        annual_only: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Compare multiple concepts across multiple companies

        Args:
            ciks: List of CIKs to compare
            concepts: List of XBRL concept names
            annual_only: If True, only include annual (10-K) data

        Returns:
            Dictionary mapping concept names to comparison DataFrames
        """
        results = {}

        for concept in concepts:
            try:
                df = self.compare_companies(ciks, concept, annual_only=annual_only)
                results[concept] = df
            except DataNotAvailable as e:
                logger.warning(f"Could not compare concept {concept}: {e}")
                results[concept] = pd.DataFrame()

        return results

    def calculate_relative_metrics(
        self,
        comparison_df: pd.DataFrame,
        metric: str = 'growth'
    ) -> pd.DataFrame:
        """
        Calculate relative metrics from comparison data

        Args:
            comparison_df: DataFrame from compare_companies()
            metric: Type of metric to calculate ('growth', 'pct_change', 'rank')

        Returns:
            DataFrame with calculated metrics
        """
        if comparison_df.empty:
            return pd.DataFrame()

        if metric == 'growth':
            # Calculate year-over-year growth for each company
            result = comparison_df.pct_change(periods=-1) * 100

        elif metric == 'pct_change':
            # Calculate percentage change from first period
            first_values = comparison_df.iloc[-1]  # Most recent is first after sorting
            result = ((comparison_df - first_values) / first_values) * 100

        elif metric == 'rank':
            # Rank companies at each date (1 = highest value)
            result = comparison_df.rank(axis=1, ascending=False)

        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'growth', 'pct_change', or 'rank'")

        return result

    def get_industry_frames(
        self,
        concept: str,
        year: int,
        taxonomy: str = 'us-gaap'
    ) -> pd.DataFrame:
        """
        Get frames data for an entire industry/sector for a specific year

        Args:
            concept: XBRL concept name (e.g., 'Assets', 'Revenues')
            year: Calendar year
            taxonomy: XBRL taxonomy (default: 'us-gaap')

        Returns:
            DataFrame with all companies reporting this concept for the year
        """
        try:
            data = self.session.get_frames(concept, year)

            # Extract the data array
            data_list = data.get('data', [])

            if not data_list:
                raise DataNotAvailable(f"No frame data found for {concept} in {year}")

            # Convert to DataFrame
            df = pd.DataFrame(data_list)

            # Convert dates if present
            if 'end' in df.columns:
                df['end'] = pd.to_datetime(df['end'])

            if 'filed' in df.columns:
                df['filed'] = pd.to_datetime(df['filed'])

            return df

        except Exception as e:
            logger.error(f"Error getting frames for {concept} in {year}: {e}")
            raise DataNotAvailable(f"Failed to get frames: {str(e)}")

    def get_industry_percentiles(
        self,
        cik: str,
        concept: str,
        year: int
    ) -> Dict[str, Any]:
        """
        Calculate where a company stands relative to industry

        Args:
            cik: Company CIK
            concept: XBRL concept name
            year: Calendar year

        Returns:
            Dictionary with percentile information
        """
        # Get industry data
        industry_df = self.get_industry_frames(concept, year)

        if industry_df.empty or 'val' not in industry_df.columns:
            raise DataNotAvailable(f"No industry data available for {concept} in {year}")

        # Get company's value
        company_data = self.get_concept_for_company(cik, concept)
        company_data['year'] = pd.to_datetime(company_data['end']).dt.year
        company_value = company_data[company_data['year'] == year]['val'].iloc[0] if len(
            company_data[company_data['year'] == year]) > 0 else None

        if company_value is None:
            raise DataNotAvailable(f"No data for CIK {cik} in year {year}")

        # Calculate percentiles
        industry_values = industry_df['val'].dropna()

        percentile = (industry_values < company_value).sum() / len(industry_values) * 100

        return {
            'company_value': company_value,
            'industry_median': industry_values.median(),
            'industry_mean': industry_values.mean(),
            'industry_min': industry_values.min(),
            'industry_max': industry_values.max(),
            'percentile': percentile,
            'rank': (industry_values >= company_value).sum(),
            'total_companies': len(industry_values),
        }
