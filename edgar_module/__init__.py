"""
EDGAR Module for VWV Trading System
High-level abstractions for SEC EDGAR financial data extraction

This module provides three layers of functionality:
- Layer A: SEC Interface (Fetcher) - Raw communication with SEC APIs
- Layer B: Normalization Engine (Processor) - Data flattening and standardization
- Layer C: VWV Integration (Controller) - Public-facing API

Usage:
    from edgar_module import EDGARClient

    # Initialize client
    client = EDGARClient(user_agent="MyApp/1.0 (contact@example.com)")

    # Get company financials by ticker
    financials = client.get_financials('AAPL')
    balance_sheet = financials['balance_sheet']

    # Compare companies
    comparison = client.compare_metrics(['AAPL', 'MSFT', 'GOOGL'], 'Revenues')

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "VWV Trading System"

# Layer A: SEC Interface (Fetcher)
from .session import EDGARSession
from .rate_limiter import RateLimiter, rate_limited, get_rate_limiter

# Layer B: Normalization Engine (Processor)
from .metadata import CIKLookup, SubmissionsParser
from .financials import FinancialsParser
from .comparison import CompanyComparison
from .scoring import AdvancedScoring
from .insider import InsiderTransactionAnalyzer
from .insider_monitor import InsiderTransactionMonitor
from .screener import EDGARScreener

# Layer C: VWV Integration (Controller)
from .client import EDGARClient, get_financials

# Exceptions
from .exceptions import (
    EDGARException,
    RateLimitExceeded,
    CIKNotFound,
    InvalidCIK,
    APIError,
    DataNotAvailable,
    ParseError,
    InvalidFormType,
    NetworkError,
)

# Configuration
from .config import (
    ENDPOINTS,
    COMMON_FORM_TYPES,
    COMMON_FINANCIAL_CONCEPTS,
)

__all__ = [
    # Version
    '__version__',
    '__author__',

    # Layer A: SEC Interface
    'EDGARSession',
    'RateLimiter',
    'rate_limited',
    'get_rate_limiter',

    # Layer B: Normalization Engine
    'CIKLookup',
    'SubmissionsParser',
    'FinancialsParser',
    'CompanyComparison',
    'AdvancedScoring',
    'InsiderTransactionAnalyzer',
    'InsiderTransactionMonitor',
    'EDGARScreener',

    # Layer C: VWV Integration (Main API)
    'EDGARClient',
    'get_financials',

    # Exceptions
    'EDGARException',
    'RateLimitExceeded',
    'CIKNotFound',
    'InvalidCIK',
    'APIError',
    'DataNotAvailable',
    'ParseError',
    'InvalidFormType',
    'NetworkError',

    # Configuration
    'ENDPOINTS',
    'COMMON_FORM_TYPES',
    'COMMON_FINANCIAL_CONCEPTS',
]
