"""
Layer A: The SEC Interface (The "Fetcher")
Manages raw communication with SEC servers
"""
import requests
import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path
import json

from .config import (
    ENDPOINTS,
    DEFAULT_USER_AGENT,
    REQUEST_TIMEOUT,
    CACHE_DIR,
    CACHE_EXPIRY_DAYS
)
from .rate_limiter import rate_limited
from .exceptions import (
    APIError,
    NetworkError,
    RateLimitExceeded,
    InvalidCIK
)

logger = logging.getLogger(__name__)


class EDGARSession:
    """
    Manages HTTP sessions and requests to SEC EDGAR APIs
    Handles User-Agent, rate limiting, caching, and error handling
    """

    def __init__(self, user_agent: Optional[str] = None, enable_cache: bool = True):
        """
        Initialize EDGAR session

        Args:
            user_agent: Custom User-Agent string. If None, uses DEFAULT_USER_AGENT
            enable_cache: Whether to enable local file caching
        """
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
        })
        self.enable_cache = enable_cache

        if self.enable_cache:
            self.cache_dir = Path(CACHE_DIR)
            self.cache_dir.mkdir(exist_ok=True)

        logger.info(f"EDGAR session initialized with User-Agent: {self.user_agent}")

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key"""
        # Sanitize cache key for filesystem
        safe_key = cache_key.replace('/', '_').replace(':', '_')
        return self.cache_dir / f"{safe_key}.json"

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load data from cache if available and not expired"""
        if not self.enable_cache:
            return None

        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            # Check if cache is expired
            cache_age_days = (time.time() - cache_path.stat().st_mtime) / 86400
            if cache_age_days > CACHE_EXPIRY_DAYS:
                logger.debug(f"Cache expired for {cache_key}")
                return None

            with open(cache_path, 'r') as f:
                data = json.load(f)
                logger.debug(f"Loaded from cache: {cache_key}")
                return data

        except Exception as e:
            logger.warning(f"Failed to load cache for {cache_key}: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Save data to cache"""
        if not self.enable_cache:
            return

        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
                logger.debug(f"Saved to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache for {cache_key}: {e}")

    @rate_limited
    def _make_request(self, url: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Make an HTTP GET request to SEC API with rate limiting

        Args:
            url: The URL to request
            use_cache: Whether to use cached data if available

        Returns:
            JSON response as dictionary

        Raises:
            APIError: If the API returns an error status code
            NetworkError: If network connectivity issues occur
            RateLimitExceeded: If rate limit is exceeded (should not happen with decorator)
        """
        # Check cache first
        if use_cache:
            cached_data = self._load_from_cache(url)
            if cached_data is not None:
                return cached_data

        try:
            logger.debug(f"Requesting: {url}")
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)

            # Handle rate limiting (429)
            if response.status_code == 429:
                logger.warning("SEC API rate limit hit (429)")
                raise RateLimitExceeded("SEC API returned 429 Too Many Requests")

            # Handle other HTTP errors
            if response.status_code == 404:
                raise APIError(404, f"Resource not found: {url}")

            if response.status_code != 200:
                raise APIError(response.status_code, f"HTTP {response.status_code}: {response.text[:200]}")

            # Parse JSON response
            data = response.json()

            # Save to cache
            if use_cache:
                self._save_to_cache(url, data)

            return data

        except requests.exceptions.Timeout:
            raise NetworkError(f"Request timeout after {REQUEST_TIMEOUT} seconds")
        except requests.exceptions.ConnectionError:
            raise NetworkError("Connection error - check your internet connection")
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Network error: {str(e)}")
        except json.JSONDecodeError:
            raise APIError(response.status_code, "Invalid JSON response from SEC API")

    def get_submissions(self, cik: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get company submissions data

        Args:
            cik: 10-digit CIK (will be padded if shorter)
            use_cache: Whether to use cached data

        Returns:
            Submissions data dictionary
        """
        cik_padded = self._format_cik(cik)
        url = ENDPOINTS['submissions'].format(cik=cik_padded)
        return self._make_request(url, use_cache=use_cache)

    def get_company_facts(self, cik: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get company facts (all XBRL financial data)

        Args:
            cik: 10-digit CIK (will be padded if shorter)
            use_cache: Whether to use cached data

        Returns:
            Company facts data dictionary
        """
        cik_padded = self._format_cik(cik)
        url = ENDPOINTS['company_facts'].format(cik=cik_padded)
        return self._make_request(url, use_cache=use_cache)

    def get_company_concept(self, cik: str, concept: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get specific concept data for a company

        Args:
            cik: 10-digit CIK (will be padded if shorter)
            concept: XBRL concept (e.g., 'Assets', 'Revenues')
            use_cache: Whether to use cached data

        Returns:
            Company concept data dictionary
        """
        cik_padded = self._format_cik(cik)
        url = ENDPOINTS['company_concept'].format(cik=cik_padded, concept=concept)
        return self._make_request(url, use_cache=use_cache)

    def get_frames(self, concept: str, year: int, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get frames data for a specific concept and year (cross-company)

        Args:
            concept: XBRL concept (e.g., 'Assets', 'Revenues')
            year: Calendar year
            use_cache: Whether to use cached data

        Returns:
            Frames data dictionary
        """
        url = ENDPOINTS['frames'].format(concept=concept, year=year)
        return self._make_request(url, use_cache=use_cache)

    def get_ticker_map(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get the ticker-to-CIK mapping

        Args:
            use_cache: Whether to use cached data

        Returns:
            Ticker map dictionary
        """
        url = ENDPOINTS['ticker_map']
        return self._make_request(url, use_cache=use_cache)

    def get_ticker_exchange(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get the ticker exchange mapping with additional metadata

        Args:
            use_cache: Whether to use cached data

        Returns:
            Ticker exchange data dictionary
        """
        url = ENDPOINTS['ticker_exchange']
        return self._make_request(url, use_cache=use_cache)

    @staticmethod
    def _format_cik(cik: str) -> str:
        """
        Format CIK to 10-digit zero-padded string

        Args:
            cik: CIK string or integer

        Returns:
            10-digit zero-padded CIK string

        Raises:
            InvalidCIK: If CIK is invalid
        """
        try:
            # Remove leading zeros and convert to int, then back to 10-digit string
            cik_str = str(cik).strip()
            cik_int = int(cik_str)

            if cik_int < 0:
                raise InvalidCIK(cik, "CIK cannot be negative")

            formatted = str(cik_int).zfill(10)
            return formatted

        except (ValueError, TypeError):
            raise InvalidCIK(cik, f"CIK must be numeric, got: {cik}")

    def clear_cache(self) -> int:
        """
        Clear all cached files

        Returns:
            Number of files deleted
        """
        if not self.enable_cache or not self.cache_dir.exists():
            return 0

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        logger.info(f"Cleared {count} cache files")
        return count

    def close(self) -> None:
        """Close the session"""
        self.session.close()
        logger.info("EDGAR session closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
