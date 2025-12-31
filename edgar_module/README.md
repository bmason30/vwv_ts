# EDGAR Module for VWV Trading System

A comprehensive Python module for extracting and analyzing financial data from the SEC EDGAR database. Provides high-level abstractions for retrieving company filings, XBRL financial facts, and cross-company comparisons.

## Features

- **Zero Dependency on Web Scraping**: All data comes directly from SEC's official APIs
- **Automatic Rate Limiting**: Complies with SEC's 10 requests/second limit
- **Intelligent Caching**: Reduces redundant API calls with local file-based cache
- **Pandas Integration**: All data methods return pandas DataFrames for easy analysis
- **Three-Layer Architecture**:
  - **Layer A: SEC Interface** - Raw API communication
  - **Layer B: Normalization Engine** - Data processing and standardization
  - **Layer C: VWV Integration** - High-level public API

## Installation

The module requires the following dependencies:

```bash
pip install requests pandas numpy
```

## Quick Start

### Basic Usage

```python
from edgar_module import EDGARClient

# Initialize client with your contact information
client = EDGARClient(user_agent="MyApp/1.0 (contact@example.com)")

# Get company financials by ticker
financials = client.get_financials('AAPL', annual=True)

# Access individual statements
balance_sheet = financials['balance_sheet']
income_statement = financials['income_statement']
cash_flow = financials['cash_flow']

# Display latest balance sheet
print(balance_sheet.head())
```

### Convenience Function

```python
from edgar_module import get_financials

# Quick access without creating a client
financials = get_financials('AAPL')
print(financials['income_statement'])
```

## Core Functionality

### 1. Company Metadata

```python
# Get company information
info = client.get_company_info('AAPL')
print(f"Name: {info['name']}")
print(f"CIK: {info['cik']}")
print(f"Industry: {info['sicDescription']}")

# Convert ticker to CIK
cik = client.ticker_to_cik('AAPL')
```

### 2. Filing Retrieval

```python
# Get all recent filings
filings = client.get_filings('AAPL')

# Filter by form type
filings_10k = client.get_filings('AAPL', form_type='10-K')

# Filter by date range
filings_2023 = client.get_filings(
    'AAPL',
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# Get URL for latest 10-K
url = client.get_filing_url('AAPL', form_type='10-K')
```

### 3. Financial Statements

```python
# Get individual statements
balance_sheet = client.get_balance_sheet('AAPL', annual=True)
income_statement = client.get_income_statement('AAPL', annual=True)
cash_flow = client.get_cash_flow('AAPL', annual=True)

# Get quarterly data instead of annual
quarterly_bs = client.get_balance_sheet('AAPL', annual=False)

# Get all statements at once
all_financials = client.get_financials('AAPL', annual=True)
```

### 4. Key Metrics

```python
# Calculate financial ratios and metrics
metrics = client.get_key_metrics('AAPL', annual=True)

# Available metrics include:
# - Working Capital
# - Current Ratio
# - Debt to Equity Ratio
# - Profit Margin
# - Gross Margin
# - Return on Equity (ROE)
# - Return on Assets (ROA)

print(metrics.head())
```

### 5. Time Series Analysis

```python
# Get time series for a specific concept
revenue_ts = client.get_metric_timeseries('AAPL', 'Revenues', annual=True)

# Plot revenue over time
import matplotlib.pyplot as plt
revenue_ts.set_index('end')['value'].plot()
plt.title('Apple Revenue Over Time')
plt.show()
```

### 6. Cross-Company Comparison

```python
# Compare a metric across companies
revenue_comparison = client.compare_metrics(
    ['AAPL', 'MSFT', 'GOOGL'],
    'Revenues',
    annual_only=True
)

print(revenue_comparison.head())

# Compare multiple metrics
peer_comparison = client.compare_peers(
    ['AAPL', 'MSFT', 'GOOGL'],
    ['Assets', 'Revenues', 'NetIncomeLoss'],
    annual_only=True
)

for metric, data in peer_comparison.items():
    print(f"\n{metric}:")
    print(data.head())
```

### 7. Industry Percentiles

```python
# See where a company ranks in its industry
percentile = client.get_industry_percentile('AAPL', 'Revenues', 2023)

print(f"Company Revenue: ${percentile['company_value']:,.0f}")
print(f"Industry Median: ${percentile['industry_median']:,.0f}")
print(f"Percentile Rank: {percentile['percentile']:.1f}%")
print(f"Rank: {percentile['rank']} out of {percentile['total_companies']}")
```

## Architecture Details

### Layer A: SEC Interface (Fetcher)

Handles raw communication with SEC servers:

```python
from edgar_module import EDGARSession

# Direct session access for advanced use
session = EDGARSession(user_agent="MyApp/1.0 (contact@example.com)")
data = session.get_company_facts("0000320193")  # Apple's CIK
```

Features:
- **User-Agent Management**: Ensures all requests include required contact info
- **Rate Limiting**: Enforces 10 requests/second limit automatically
- **Caching**: Local file-based cache for metadata (configurable expiry)
- **Error Handling**: Specific exceptions for different failure modes

### Layer B: Normalization Engine (Processor)

Processes and standardizes SEC data:

```python
from edgar_module import FinancialsParser, CIKLookup

session = EDGARSession()

# CIK lookup utilities
lookup = CIKLookup(session)
cik = lookup.ticker_to_cik('AAPL')

# Financial data parsing
parser = FinancialsParser(session)
balance_sheet = parser.get_balance_sheet(cik, annual=True)
```

Features:
- **CIK Lookup**: Ticker-to-CIK conversion using SEC master map
- **Schema Mapping**: Handles different GAAP/IFRS taxonomies
- **Unit Conversion**: Automatic handling of currency and scaling
- **Data Normalization**: Flattens nested JSON/XBRL structures

### Layer C: VWV Integration (Controller)

Public-facing API optimized for the VWV trading system:

```python
from edgar_module import EDGARClient

client = EDGARClient()
# All high-level methods available here
```

Features:
- **Ticker or CIK**: All methods accept either format
- **Pandas Integration**: All methods return DataFrames
- **Filtering**: Built-in filtering by form type, date range, etc.
- **Comparison Tools**: Cross-company and industry analysis

## Configuration

### Custom User-Agent

Always set a custom User-Agent with your contact information:

```python
client = EDGARClient(
    user_agent="MyCompany FinanceApp/2.0 (john@mycompany.com)"
)
```

### Cache Management

```python
# Disable caching
client = EDGARClient(enable_cache=False)

# Clear cache
num_deleted = client.clear_cache()
print(f"Cleared {num_deleted} cache files")
```

### Cache Configuration

Edit `edgar_module/config.py`:

```python
CACHE_DIR = ".edgar_cache"  # Cache directory
CACHE_EXPIRY_DAYS = 30      # Cache expiration
```

## Common Financial Concepts

The module supports standard XBRL concepts:

### Balance Sheet
- `Assets` - Total assets
- `AssetsCurrent` - Current assets
- `Liabilities` - Total liabilities
- `LiabilitiesCurrent` - Current liabilities
- `StockholdersEquity` - Shareholders' equity
- `CashAndCashEquivalentsAtCarryingValue` - Cash
- `PropertyPlantAndEquipmentNet` - PP&E
- `Goodwill` - Goodwill
- `LongTermDebt` - Long-term debt

### Income Statement
- `Revenues` - Total revenue
- `CostOfRevenue` - Cost of revenue
- `GrossProfit` - Gross profit
- `OperatingIncomeLoss` - Operating income
- `NetIncomeLoss` - Net income
- `EarningsPerShareBasic` - Basic EPS
- `EarningsPerShareDiluted` - Diluted EPS

### Cash Flow Statement
- `NetCashProvidedByUsedInOperatingActivities` - Operating cash flow
- `NetCashProvidedByUsedInInvestingActivities` - Investing cash flow
- `NetCashProvidedByUsedInFinancingActivities` - Financing cash flow
- `PaymentsToAcquirePropertyPlantAndEquipment` - CapEx
- `PaymentsOfDividends` - Dividends paid

## Error Handling

```python
from edgar_module.exceptions import (
    CIKNotFound,
    DataNotAvailable,
    RateLimitExceeded,
    APIError
)

try:
    financials = client.get_financials('INVALID_TICKER')
except CIKNotFound as e:
    print(f"Ticker not found: {e.message}")
except DataNotAvailable as e:
    print(f"Data not available: {e.message}")
except RateLimitExceeded as e:
    print(f"Rate limit hit: {e.message}")
except APIError as e:
    print(f"API error {e.status_code}: {e.message}")
```

## Best Practices

1. **Always Set User-Agent**: Include your app name and contact email
2. **Use Caching**: Enable caching to reduce API calls (default: enabled)
3. **Handle Missing Data**: Not all companies report all metrics
4. **Respect Rate Limits**: The module handles this automatically
5. **Close Sessions**: Use context manager or call `client.close()`

```python
# Recommended: Use context manager
with EDGARClient(user_agent="MyApp/1.0 (me@example.com)") as client:
    data = client.get_financials('AAPL')
    # Session automatically closed
```

## Testing

Run the comprehensive test suite:

```bash
python test_edgar_module.py
```

The test suite covers:
- Ticker-to-CIK conversion
- Filing retrieval and filtering
- Financial statement extraction
- Key metrics calculation
- Cross-company comparison
- Rate limiting functionality

## Limitations

1. **No Text Extraction**: The module retrieves structured data only, not full text from filings
2. **XBRL Only**: Only works with companies that file in XBRL format
3. **US Companies**: Primarily designed for US companies (10-K, 10-Q forms)
4. **Rate Limits**: SEC enforces 10 requests/second (handled automatically)
5. **Data Availability**: Not all companies report all metrics

## Future Enhancements

Planned for future versions:
- Full-text search within filings
- MD&A (Management Discussion & Analysis) extraction
- Automatic detection of restatements
- Support for foreign filers (20-F, 6-K)
- OAuth integration for "EDGAR Next" (late 2025)

## API Reference

### EDGARClient

Main client class for accessing EDGAR data.

**Methods:**

- `ticker_to_cik(ticker)` - Convert ticker to CIK
- `get_company_info(ticker_or_cik)` - Get company metadata
- `get_filings(ticker_or_cik, form_type, start_date, end_date)` - Get filings
- `get_financials(ticker_or_cik, annual)` - Get all financial statements
- `get_balance_sheet(ticker_or_cik, annual)` - Get balance sheet
- `get_income_statement(ticker_or_cik, annual)` - Get income statement
- `get_cash_flow(ticker_or_cik, annual)` - Get cash flow statement
- `get_key_metrics(ticker_or_cik, annual)` - Calculate key metrics
- `get_metric_timeseries(ticker_or_cik, concept, annual)` - Get time series
- `compare_metrics(tickers_or_ciks, concept, annual_only)` - Compare companies
- `compare_peers(tickers_or_ciks, concepts, annual_only)` - Multi-metric comparison
- `get_industry_percentile(ticker_or_cik, concept, year)` - Industry ranking
- `get_filing_url(ticker_or_cik, form_type, latest)` - Get filing URL
- `clear_cache()` - Clear cached data
- `close()` - Close the session

## Support and Contributing

For issues, questions, or contributions, please refer to the main VWV Trading System repository.

## License

This module is part of the VWV Trading System.

## Acknowledgments

Data provided by the U.S. Securities and Exchange Commission (SEC).
This module uses the SEC's EDGAR APIs and complies with all SEC usage guidelines.

---

**Version**: 1.0.0
**Last Updated**: 2025-01-01
**Author**: VWV Trading System
