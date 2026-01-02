"""
EDGAR Multi-Ticker Screener
Combines Piotroski, Altman, Graham scores with insider activity
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .session import EDGARSession
from .metadata import CIKLookup
from .financials import FinancialsParser
from .scoring import AdvancedScoring
from .insider import InsiderTransactionAnalyzer
from .exceptions import CIKNotFound, DataNotAvailable

logger = logging.getLogger(__name__)


class EDGARScreener:
    """
    Multi-ticker screener with advanced scoring
    """

    def __init__(self, session: EDGARSession):
        """
        Initialize screener

        Args:
            session: EDGARSession instance
        """
        self.session = session
        self.cik_lookup = CIKLookup(session)
        self.financials_parser = FinancialsParser(session)
        self.scoring = AdvancedScoring()
        self.insider = InsiderTransactionAnalyzer(session)

    def screen_ticker(
        self,
        ticker: str,
        include_price_data: bool = False,
        price_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Screen a single ticker with all scoring metrics

        Args:
            ticker: Stock ticker
            include_price_data: Whether to include price-based metrics
            price_data: Dict with 'price', 'market_cap', 'sma_50' if available

        Returns:
            Dict with all screening data
        """
        result = {
            'ticker': ticker,
            'cik': None,
            'company_name': None,
            'piotroski_score': None,
            'altman_zscore': None,
            'graham_number': None,
            'insider_activity': None,
            'vwv_alpha_score': None,
            'error': None
        }

        try:
            # Get CIK
            cik = self.cik_lookup.ticker_to_cik(ticker)
            result['cik'] = cik

            # Get company info
            try:
                submissions = self.session.get_submissions(cik)
                result['company_name'] = submissions.get('name', ticker)
            except:
                result['company_name'] = ticker

            # Get financials (current and prior year)
            try:
                current_financials = self.financials_parser.get_all_financials(cik, annual=True)

                # Get 2 years of data for prior year comparison
                bs = current_financials.get('balance_sheet', pd.DataFrame())
                income = current_financials.get('income_statement', pd.DataFrame())
                cf = current_financials.get('cash_flow', pd.DataFrame())

                # Extract prior year data (second row if available)
                if not bs.empty and len(bs) >= 2:
                    prior_financials = {
                        'balance_sheet': bs.iloc[[1]],
                        'income_statement': income.iloc[[1]] if len(income) >= 2 else pd.DataFrame(),
                        'cash_flow': cf.iloc[[1]] if len(cf) >= 2 else pd.DataFrame()
                    }
                else:
                    prior_financials = {
                        'balance_sheet': pd.DataFrame(),
                        'income_statement': pd.DataFrame(),
                        'cash_flow': pd.DataFrame()
                    }

                # Calculate Piotroski F-Score
                piotroski = self.scoring.calculate_piotroski_fscore(
                    current_financials,
                    prior_financials
                )
                result['piotroski_score'] = piotroski.get('score')
                result['piotroski_details'] = piotroski

                # Calculate Altman Z-Score
                market_cap = price_data.get('market_cap') if price_data else None
                altman = self.scoring.calculate_altman_zscore(
                    current_financials,
                    market_cap=market_cap
                )
                result['altman_zscore'] = altman.get('zscore')
                result['altman_zone'] = altman.get('zone')
                result['altman_details'] = altman

                # Calculate Graham Number
                current_price = price_data.get('price') if price_data else None
                graham = self.scoring.calculate_graham_number(
                    current_financials,
                    current_price=current_price
                )
                result['graham_number'] = graham.get('graham_number')
                result['graham_details'] = graham

                if current_price and graham.get('graham_number'):
                    result['price_to_graham_pct'] = (
                        current_price / graham['graham_number'] * 100
                    )

                # Calculate VWV Alpha Score (weighted combination)
                vwv_score = self._calculate_vwv_alpha_score(piotroski, altman, graham)
                result['vwv_alpha_score'] = vwv_score

            except DataNotAvailable as e:
                result['error'] = f"Financial data not available: {str(e)}"
            except Exception as e:
                result['error'] = f"Error in financial analysis: {str(e)}"

            # Get insider activity
            try:
                insider_summary = self.insider.get_insider_summary(cik)
                result['insider_activity'] = insider_summary
            except Exception as e:
                result['insider_activity'] = "N/A"
                logger.warning(f"Could not get insider activity for {ticker}: {e}")

            # Add price data if provided
            if price_data:
                result['current_price'] = price_data.get('price')
                result['market_cap'] = price_data.get('market_cap')
                result['sma_50'] = price_data.get('sma_50')

                # Calculate distance from SMA
                if price_data.get('price') and price_data.get('sma_50'):
                    sma_distance = (
                        (price_data['price'] - price_data['sma_50']) /
                        price_data['sma_50'] * 100
                    )
                    result['sma_50_distance_pct'] = sma_distance

        except CIKNotFound:
            result['error'] = f"Ticker {ticker} not found in SEC database"
        except Exception as e:
            result['error'] = f"Screening error: {str(e)}"
            logger.error(f"Error screening {ticker}: {e}")

        return result

    def screen_multiple_tickers(
        self,
        tickers: List[str],
        include_price_data: bool = False,
        price_data_dict: Optional[Dict[str, Dict[str, Any]]] = None,
        max_workers: int = 3
    ) -> pd.DataFrame:
        """
        Screen multiple tickers in parallel

        Args:
            tickers: List of stock tickers
            include_price_data: Whether to include price data
            price_data_dict: Dict mapping ticker to price data
            max_workers: Max concurrent workers (default: 3 to respect rate limits)

        Returns:
            DataFrame with screening results
        """
        results = []

        # Use ThreadPoolExecutor for parallel processing (with rate limit consideration)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {}

            for ticker in tickers:
                price_data = None
                if price_data_dict and ticker in price_data_dict:
                    price_data = price_data_dict[ticker]

                future = executor.submit(
                    self.screen_ticker,
                    ticker,
                    include_price_data,
                    price_data
                )
                future_to_ticker[future] = ticker

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
                    results.append({
                        'ticker': ticker,
                        'error': str(e)
                    })

        # Convert to DataFrame
        if results:
            df = pd.DataFrame(results)

            # Sort by VWV Alpha Score (descending)
            if 'vwv_alpha_score' in df.columns:
                df = df.sort_values('vwv_alpha_score', ascending=False, na_position='last')

            return df
        else:
            return pd.DataFrame()

    def _calculate_vwv_alpha_score(
        self,
        piotroski: Dict[str, Any],
        altman: Dict[str, Any],
        graham: Dict[str, Any]
    ) -> Optional[float]:
        """
        Calculate VWV Alpha Score - weighted combination of all metrics

        Weighting:
        - Piotroski F-Score: 40% (quality and value)
        - Altman Z-Score: 30% (financial health)
        - Graham criteria: 30% (value)

        Returns:
            Score from 0-100
        """
        try:
            score = 0
            weights_used = 0

            # Piotroski contribution (0-40 points)
            piotroski_score = piotroski.get('score')
            if piotroski_score is not None:
                # Scale 0-9 to 0-40
                score += (piotroski_score / 9) * 40
                weights_used += 40

            # Altman contribution (0-30 points)
            altman_score = altman.get('zscore')
            if altman_score is not None:
                # Convert Z-score to 0-30 scale
                # Z > 2.99 = 30 pts, 1.81-2.99 = 15 pts, < 1.81 = 0 pts
                if altman_score > 2.99:
                    altman_points = 30
                elif altman_score > 1.81:
                    # Linear scale in grey zone
                    altman_points = 15 + ((altman_score - 1.81) / (2.99 - 1.81)) * 15
                else:
                    # Minimal points in distress zone
                    altman_points = max(0, (altman_score / 1.81) * 5)

                score += altman_points
                weights_used += 30

            # Graham contribution (0-30 points)
            graham_details = graham.get('graham_details') if isinstance(graham, dict) and 'graham_details' in graham else graham
            meets_criteria = False

            if isinstance(graham_details, dict):
                meets_criteria = graham_details.get('meets_graham_criteria', False)

            if meets_criteria:
                score += 30
                weights_used += 30
            elif graham.get('graham_number') is not None:
                # Partial credit based on price/graham ratio
                price_to_graham = graham_details.get('price_to_graham', 100) if isinstance(graham_details, dict) else 100
                if price_to_graham is not None and price_to_graham < 150:
                    # Give credit if reasonably valued
                    graham_points = max(0, (150 - price_to_graham) / 150 * 30)
                    score += graham_points
                    weights_used += 30

            # Normalize to 100-point scale if not all metrics available
            if weights_used > 0:
                normalized_score = (score / weights_used) * 100
                return round(normalized_score, 1)
            else:
                return None

        except Exception as e:
            logger.error(f"Error calculating VWV Alpha Score: {e}")
            return None

    def get_top_scorers(
        self,
        tickers: List[str],
        top_n: int = 10,
        min_score: Optional[float] = 50.0
    ) -> pd.DataFrame:
        """
        Get top N stocks by VWV Alpha Score

        Args:
            tickers: List of tickers to screen
            top_n: Number of top stocks to return
            min_score: Minimum VWV Alpha Score to include

        Returns:
            DataFrame with top scorers
        """
        df = self.screen_multiple_tickers(tickers)

        if df.empty:
            return df

        # Filter by minimum score
        if min_score is not None and 'vwv_alpha_score' in df.columns:
            df = df[df['vwv_alpha_score'] >= min_score]

        # Get top N
        if len(df) > top_n:
            df = df.head(top_n)

        return df

    def export_screening_results(
        self,
        df: pd.DataFrame,
        filename: str = 'edgar_screening_results.csv'
    ) -> None:
        """
        Export screening results to CSV

        Args:
            df: DataFrame from screening
            filename: Output filename
        """
        try:
            # Select key columns for export
            export_cols = [
                'ticker', 'company_name', 'piotroski_score',
                'altman_zscore', 'altman_zone', 'graham_number',
                'price_to_graham_pct', 'insider_activity',
                'vwv_alpha_score', 'sma_50_distance_pct'
            ]

            available_cols = [col for col in export_cols if col in df.columns]
            export_df = df[available_cols]

            export_df.to_csv(filename, index=False)
            logger.info(f"Screening results exported to {filename}")

        except Exception as e:
            logger.error(f"Error exporting results: {e}")
