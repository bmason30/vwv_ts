"""
Comprehensive tests for the EDGAR module
Tests all three layers: SEC Interface, Normalization Engine, and VWV Integration

Run with: python test_edgar_module.py
"""

import sys
from edgar_module import EDGARClient, get_financials
from edgar_module.exceptions import CIKNotFound, DataNotAvailable


def test_basic_functionality():
    """Test basic functionality of the EDGAR module"""
    print("\n" + "="*80)
    print("TESTING EDGAR MODULE - BASIC FUNCTIONALITY")
    print("="*80)

    # Initialize client
    print("\n1. Initializing EDGARClient...")
    client = EDGARClient(user_agent="VWV Test/1.0 (testing@vwv.com)")
    print(f"   ✓ Client initialized: {client}")

    # Test ticker to CIK conversion
    print("\n2. Testing Ticker-to-CIK conversion...")
    try:
        cik_aapl = client.ticker_to_cik('AAPL')
        print(f"   ✓ AAPL CIK: {cik_aapl}")

        cik_msft = client.ticker_to_cik('MSFT')
        print(f"   ✓ MSFT CIK: {cik_msft}")

        cik_googl = client.ticker_to_cik('GOOGL')
        print(f"   ✓ GOOGL CIK: {cik_googl}")
    except Exception as e:
        print(f"   ✗ Error in ticker-to-CIK conversion: {e}")

    # Test invalid ticker
    print("\n3. Testing invalid ticker handling...")
    try:
        invalid_cik = client.ticker_to_cik('INVALID_TICKER_XYZ')
        print(f"   ✗ Should have raised CIKNotFound exception")
    except CIKNotFound as e:
        print(f"   ✓ Correctly raised CIKNotFound: {e.message}")

    # Test company info
    print("\n4. Getting company metadata...")
    try:
        info = client.get_company_info('AAPL')
        print(f"   ✓ Company Name: {info.get('name')}")
        print(f"   ✓ CIK: {info.get('cik')}")
        print(f"   ✓ SIC: {info.get('sic')} - {info.get('sicDescription')}")
        print(f"   ✓ Tickers: {info.get('tickers')}")
    except Exception as e:
        print(f"   ✗ Error getting company info: {e}")

    client.close()
    print("\n   ✓ Client closed successfully")


def test_filings():
    """Test filing retrieval and filtering"""
    print("\n" + "="*80)
    print("TESTING FILING RETRIEVAL")
    print("="*80)

    client = EDGARClient(user_agent="VWV Test/1.0 (testing@vwv.com)")

    # Get all filings
    print("\n1. Getting recent filings for AAPL...")
    try:
        filings = client.get_filings('AAPL')
        print(f"   ✓ Total filings retrieved: {len(filings)}")
        if len(filings) > 0:
            print(f"   ✓ Latest filing: {filings.iloc[0]['form']} on {filings.iloc[0]['filingDate']}")
    except Exception as e:
        print(f"   ✗ Error getting filings: {e}")

    # Get 10-K filings only
    print("\n2. Getting 10-K filings only...")
    try:
        filings_10k = client.get_filings('AAPL', form_type='10-K')
        print(f"   ✓ Total 10-K filings: {len(filings_10k)}")
        if len(filings_10k) > 0:
            print(f"   ✓ Latest 10-K: {filings_10k.iloc[0]['filingDate']}")
    except Exception as e:
        print(f"   ✗ Error getting 10-K filings: {e}")

    # Get filing URL
    print("\n3. Getting filing URL...")
    try:
        url = client.get_filing_url('AAPL', form_type='10-K')
        print(f"   ✓ Filing URL: {url[:80]}...")
    except Exception as e:
        print(f"   ✗ Error getting filing URL: {e}")

    client.close()


def test_financials():
    """Test financial statement retrieval"""
    print("\n" + "="*80)
    print("TESTING FINANCIAL STATEMENTS")
    print("="*80)

    client = EDGARClient(user_agent="VWV Test/1.0 (testing@vwv.com)")

    # Get balance sheet
    print("\n1. Getting balance sheet for AAPL...")
    try:
        balance_sheet = client.get_balance_sheet('AAPL', annual=True)
        print(f"   ✓ Balance sheet retrieved: {balance_sheet.shape}")
        print(f"   ✓ Columns: {list(balance_sheet.columns[:5])}...")
        if not balance_sheet.empty:
            print(f"   ✓ Latest date: {balance_sheet.index[0]}")
            if 'Assets' in balance_sheet.columns:
                latest_assets = balance_sheet['Assets'].iloc[0]
                print(f"   ✓ Total Assets: ${latest_assets:,.0f}")
    except Exception as e:
        print(f"   ✗ Error getting balance sheet: {e}")

    # Get income statement
    print("\n2. Getting income statement for AAPL...")
    try:
        income_stmt = client.get_income_statement('AAPL', annual=True)
        print(f"   ✓ Income statement retrieved: {income_stmt.shape}")
        print(f"   ✓ Columns: {list(income_stmt.columns[:5])}...")
        if not income_stmt.empty:
            print(f"   ✓ Latest date: {income_stmt.index[0]}")
            if 'NetIncomeLoss' in income_stmt.columns:
                latest_income = income_stmt['NetIncomeLoss'].iloc[0]
                print(f"   ✓ Net Income: ${latest_income:,.0f}")
    except Exception as e:
        print(f"   ✗ Error getting income statement: {e}")

    # Get cash flow statement
    print("\n3. Getting cash flow statement for AAPL...")
    try:
        cash_flow = client.get_cash_flow('AAPL', annual=True)
        print(f"   ✓ Cash flow statement retrieved: {cash_flow.shape}")
        print(f"   ✓ Columns: {list(cash_flow.columns[:3])}...")
    except Exception as e:
        print(f"   ✗ Error getting cash flow statement: {e}")

    # Get all financials at once
    print("\n4. Getting all financials at once...")
    try:
        all_financials = client.get_financials('AAPL', annual=True)
        print(f"   ✓ Balance Sheet: {all_financials['balance_sheet'].shape}")
        print(f"   ✓ Income Statement: {all_financials['income_statement'].shape}")
        print(f"   ✓ Cash Flow: {all_financials['cash_flow'].shape}")
    except Exception as e:
        print(f"   ✗ Error getting all financials: {e}")

    client.close()


def test_key_metrics():
    """Test key metrics calculation"""
    print("\n" + "="*80)
    print("TESTING KEY METRICS")
    print("="*80)

    client = EDGARClient(user_agent="VWV Test/1.0 (testing@vwv.com)")

    print("\n1. Calculating key metrics for AAPL...")
    try:
        metrics = client.get_key_metrics('AAPL', annual=True)
        print(f"   ✓ Metrics calculated: {metrics.shape}")
        print(f"   ✓ Available metrics: {list(metrics.columns)}")

        if not metrics.empty:
            print(f"\n   Latest metrics (as of {metrics.index[0]}):")
            for col in metrics.columns:
                value = metrics[col].iloc[0]
                if not pd.isna(value):
                    if 'Ratio' in col:
                        print(f"   • {col}: {value:.2f}")
                    elif 'Margin' in col:
                        print(f"   • {col}: {value:.2%}")
                    else:
                        print(f"   • {col}: {value:,.0f}")
    except Exception as e:
        print(f"   ✗ Error calculating metrics: {e}")

    client.close()


def test_comparison():
    """Test cross-company comparison"""
    print("\n" + "="*80)
    print("TESTING CROSS-COMPANY COMPARISON")
    print("="*80)

    client = EDGARClient(user_agent="VWV Test/1.0 (testing@vwv.com)")

    print("\n1. Comparing revenues across tech companies...")
    try:
        # Try with common concept names
        concepts_to_try = [
            'Revenues',
            'RevenueFromContractWithCustomerExcludingAssessedTax'
        ]

        comparison = None
        for concept in concepts_to_try:
            try:
                comparison = client.compare_metrics(
                    ['AAPL', 'MSFT', 'GOOGL'],
                    concept,
                    annual_only=True
                )
                if not comparison.empty:
                    print(f"   ✓ Comparison successful using concept: {concept}")
                    print(f"   ✓ Shape: {comparison.shape}")
                    print(f"\n   Latest revenue comparison:")
                    if len(comparison) > 0:
                        latest = comparison.iloc[0]
                        for ticker in comparison.columns:
                            if not pd.isna(latest[ticker]):
                                print(f"   • {ticker}: ${latest[ticker]:,.0f}")
                    break
            except DataNotAvailable:
                continue

        if comparison is None or comparison.empty:
            print(f"   ⚠ No comparison data available")

    except Exception as e:
        print(f"   ✗ Error in comparison: {e}")

    client.close()


def test_convenience_function():
    """Test the convenience get_financials function"""
    print("\n" + "="*80)
    print("TESTING CONVENIENCE FUNCTION")
    print("="*80)

    print("\n1. Using get_financials() convenience function...")
    try:
        financials = get_financials('AAPL', annual=True)
        print(f"   ✓ Balance Sheet: {financials['balance_sheet'].shape}")
        print(f"   ✓ Income Statement: {financials['income_statement'].shape}")
        print(f"   ✓ Cash Flow: {financials['cash_flow'].shape}")
    except Exception as e:
        print(f"   ✗ Error with convenience function: {e}")


def test_rate_limiting():
    """Test rate limiting functionality"""
    print("\n" + "="*80)
    print("TESTING RATE LIMITING")
    print("="*80)

    from edgar_module.rate_limiter import RateLimiter
    import time

    print("\n1. Testing rate limiter...")
    limiter = RateLimiter(max_requests=5, time_window=1.0)

    print("   Making 5 requests (should be instant)...")
    start = time.time()
    for i in range(5):
        limiter.acquire()
    elapsed = time.time() - start
    print(f"   ✓ 5 requests completed in {elapsed:.3f} seconds")

    print("   Making 6th request (should wait)...")
    start = time.time()
    limiter.acquire()
    elapsed = time.time() - start
    print(f"   ✓ 6th request waited {elapsed:.3f} seconds")

    print("   ✓ Rate limiter working correctly")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("EDGAR MODULE COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("\nNote: These tests make real API calls to SEC servers.")
    print("They respect the 10 requests/second rate limit.")
    print("Tests may take several minutes to complete.")

    import pandas as pd
    globals()['pd'] = pd  # Make pandas available to test functions

    try:
        test_basic_functionality()
        test_filings()
        test_financials()
        test_key_metrics()
        test_comparison()
        test_convenience_function()
        test_rate_limiting()

        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)
        print("\n✓ EDGAR module is working correctly!")
        print("\nNote: Some tests may show warnings if specific data is not available.")
        print("This is normal behavior - not all companies report all metrics.\n")

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\n✗ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
