"""
SEC EDGAR Module Configuration
Manages API endpoints, rate limits, and global settings
"""

# SEC API Endpoints
SEC_BASE_URL = "https://data.sec.gov"
SEC_ARCHIVE_URL = "https://www.sec.gov/Archives"

# API Endpoint Registry
ENDPOINTS = {
    'submissions': f"{SEC_BASE_URL}/submissions/CIK{{cik}}.json",
    'company_facts': f"{SEC_BASE_URL}/api/xbrl/companyfacts/CIK{{cik}}.json",
    'company_concept': f"{SEC_BASE_URL}/api/xbrl/companyconcept/CIK{{cik}}/us-gaap/{{concept}}.json",
    'frames': f"{SEC_BASE_URL}/api/xbrl/frames/us-gaap/{{concept}}/USD/CY{{year}}.json",
    'ticker_map': f"{SEC_BASE_URL}/files/company_tickers.json",
    'ticker_exchange': f"{SEC_BASE_URL}/files/company_tickers_exchange.json"
}

# Rate Limiting Configuration
# SEC requires no more than 10 requests per second
MAX_REQUESTS_PER_SECOND = 10
REQUEST_TIMEOUT = 30  # seconds

# User-Agent Configuration
# SEC requires a User-Agent header with contact information
DEFAULT_USER_AGENT = "VWV Trading System edgar_module/1.0 (https://github.com/vwv; contact@example.com)"

# Caching Configuration
CACHE_DIR = ".edgar_cache"
CACHE_EXPIRY_DAYS = 30  # How long to keep cached metadata

# Data Processing Configuration
SUPPORTED_TAXONOMIES = ['us-gaap', 'ifrs-full', 'dei']
COMMON_FINANCIAL_CONCEPTS = {
    'Assets': 'us-gaap:Assets',
    'AssetsCurrent': 'us-gaap:AssetsCurrent',
    'Liabilities': 'us-gaap:Liabilities',
    'LiabilitiesCurrent': 'us-gaap:LiabilitiesCurrent',
    'StockholdersEquity': 'us-gaap:StockholdersEquity',
    'Revenues': 'us-gaap:Revenues',
    'RevenueFromContractWithCustomerExcludingAssessedTax': 'us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax',
    'CostOfRevenue': 'us-gaap:CostOfRevenue',
    'GrossProfit': 'us-gaap:GrossProfit',
    'OperatingIncomeLoss': 'us-gaap:OperatingIncomeLoss',
    'NetIncomeLoss': 'us-gaap:NetIncomeLoss',
    'EarningsPerShareBasic': 'us-gaap:EarningsPerShareBasic',
    'EarningsPerShareDiluted': 'us-gaap:EarningsPerShareDiluted',
    'CashAndCashEquivalentsAtCarryingValue': 'us-gaap:CashAndCashEquivalentsAtCarryingValue',
}

# Form Types
COMMON_FORM_TYPES = {
    '10-K': 'Annual Report',
    '10-Q': 'Quarterly Report',
    '8-K': 'Current Report',
    '10-K/A': 'Annual Report Amendment',
    '10-Q/A': 'Quarterly Report Amendment',
    'S-1': 'Registration Statement',
    '20-F': 'Annual Report (Foreign)',
    '6-K': 'Current Report (Foreign)',
    'DEF 14A': 'Proxy Statement',
}

# Unit Scaling Factors
UNIT_MULTIPLIERS = {
    'shares': 1,
    'USD': 1,
    'USD/shares': 1,
    'pure': 1,
}

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
