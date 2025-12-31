"""
Example usage of the EDGAR module for VWV Trading System

This file demonstrates common use cases and workflows.
"""

from edgar_module import EDGARClient, get_financials
import pandas as pd


def example_1_basic_usage():
    """Example 1: Basic financial data retrieval"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Financial Data Retrieval")
    print("="*80 + "\n")

    # Initialize client
    client = EDGARClient(user_agent="VWV Examples/1.0 (demo@vwv.com)")

    # Get company information
    info = client.get_company_info('AAPL')
    print(f"Company: {info['name']}")
    print(f"CIK: {info['cik']}")
    print(f"Industry: {info['sicDescription']}\n")

    # Get annual financials
    financials = client.get_financials('AAPL', annual=True)

    # Display balance sheet summary
    print("Balance Sheet (last 5 years):")
    bs = financials['balance_sheet']
    if 'Assets' in bs.columns and 'Liabilities' in bs.columns:
        summary = bs[['Assets', 'Liabilities', 'StockholdersEquity']].head()
        print(summary.to_string())

    client.close()


def example_2_key_metrics():
    """Example 2: Calculate and display key financial metrics"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Key Financial Metrics")
    print("="*80 + "\n")

    client = EDGARClient(user_agent="VWV Examples/1.0 (demo@vwv.com)")

    # Get key metrics for Apple
    metrics = client.get_key_metrics('AAPL', annual=True)

    print("Key Metrics for Apple Inc. (last 3 years):\n")

    # Display relevant metrics
    if not metrics.empty:
        display_metrics = ['CurrentRatio', 'DebtToEquityRatio', 'ProfitMargin', 'ReturnOnEquity']
        available_metrics = [m for m in display_metrics if m in metrics.columns]

        if available_metrics:
            print(metrics[available_metrics].head(3).to_string())
        else:
            print("Metrics data not available in expected format")
            print(f"Available columns: {list(metrics.columns)}")

    client.close()


def example_3_peer_comparison():
    """Example 3: Compare metrics across peer companies"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Peer Company Comparison")
    print("="*80 + "\n")

    client = EDGARClient(user_agent="VWV Examples/1.0 (demo@vwv.com)")

    # Compare tech giants
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    print(f"Comparing: {', '.join(tickers)}\n")

    # Try different revenue concepts
    revenue_concepts = [
        'Revenues',
        'RevenueFromContractWithCustomerExcludingAssessedTax'
    ]

    for concept in revenue_concepts:
        try:
            comparison = client.compare_metrics(tickers, concept, annual_only=True)

            if not comparison.empty:
                print(f"Revenue Comparison (using {concept}):")
                print("\nLatest Annual Revenues (USD):")

                # Get latest year data
                latest = comparison.iloc[0]
                for ticker in tickers:
                    if ticker in comparison.columns:
                        cik = client.ticker_to_cik(ticker)
                        if cik in comparison.columns:
                            value = latest[cik]
                            if not pd.isna(value):
                                print(f"  {ticker}: ${value:,.0f}")
                break
        except Exception as e:
            continue

    client.close()


def example_4_time_series_analysis():
    """Example 4: Analyze trends over time"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Time Series Analysis")
    print("="*80 + "\n")

    client = EDGARClient(user_agent="VWV Examples/1.0 (demo@vwv.com)")

    # Get revenue time series
    print("Apple Revenue Trend:\n")

    try:
        revenue_ts = client.get_metric_timeseries('AAPL', 'Revenues', annual=True)

        if not revenue_ts.empty and 'end' in revenue_ts.columns and 'value' in revenue_ts.columns:
            # Calculate year-over-year growth
            revenue_ts = revenue_ts.sort_values('end')
            revenue_ts['yoy_growth'] = revenue_ts['value'].pct_change() * 100

            # Display last 5 years
            recent = revenue_ts.tail(5)
            for _, row in recent.iterrows():
                year = row['end'].year if hasattr(row['end'], 'year') else row['end']
                revenue = row['value']
                growth = row['yoy_growth']

                if pd.isna(growth):
                    print(f"  {year}: ${revenue:,.0f}")
                else:
                    print(f"  {year}: ${revenue:,.0f} ({growth:+.1f}% YoY)")
    except Exception as e:
        print(f"  Could not retrieve revenue trend: {e}")

    client.close()


def example_5_filing_search():
    """Example 5: Search and filter filings"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Filing Search and Filtering")
    print("="*80 + "\n")

    client = EDGARClient(user_agent="VWV Examples/1.0 (demo@vwv.com)")

    # Get all 10-K filings for Tesla
    print("Recent 10-K Filings for Tesla:\n")

    filings = client.get_filings('TSLA', form_type='10-K')

    if not filings.empty:
        # Display last 3 filings
        for _, filing in filings.head(3).iterrows():
            date = filing['filingDate']
            form = filing['form']
            print(f"  {form} filed on {date}")

        # Get URL for latest 10-K
        print("\nLatest 10-K URL:")
        url = client.get_filing_url('TSLA', form_type='10-K')
        print(f"  {url}")
    else:
        print("  No 10-K filings found")

    client.close()


def example_6_convenience_function():
    """Example 6: Using convenience functions"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Convenience Functions")
    print("="*80 + "\n")

    # Quick access without creating a client
    print("Using get_financials() convenience function:\n")

    financials = get_financials('MSFT', annual=True)

    print("Microsoft Financial Statements:")
    print(f"  Balance Sheet: {financials['balance_sheet'].shape}")
    print(f"  Income Statement: {financials['income_statement'].shape}")
    print(f"  Cash Flow: {financials['cash_flow'].shape}")


def example_7_industry_ranking():
    """Example 7: Industry percentile ranking"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Industry Percentile Ranking")
    print("="*80 + "\n")

    client = EDGARClient(user_agent="VWV Examples/1.0 (demo@vwv.com)")

    try:
        # Get industry ranking for Apple's revenue
        ranking = client.get_industry_percentile('AAPL', 'Revenues', 2023)

        print("Apple's Industry Position (2023 Revenues):\n")
        print(f"  Company Revenue: ${ranking['company_value']:,.0f}")
        print(f"  Industry Median: ${ranking['industry_median']:,.0f}")
        print(f"  Industry Mean: ${ranking['industry_mean']:,.0f}")
        print(f"  Percentile Rank: {ranking['percentile']:.1f}%")
        print(f"  Position: #{ranking['rank']} out of {ranking['total_companies']} companies")
    except Exception as e:
        print(f"  Industry ranking not available: {e}")

    client.close()


def example_8_error_handling():
    """Example 8: Proper error handling"""
    print("\n" + "="*80)
    print("EXAMPLE 8: Error Handling")
    print("="*80 + "\n")

    from edgar_module.exceptions import CIKNotFound, DataNotAvailable

    client = EDGARClient(user_agent="VWV Examples/1.0 (demo@vwv.com)")

    # Handle invalid ticker
    print("Testing invalid ticker handling:")
    try:
        client.get_financials('INVALID_TICKER_XYZ')
    except CIKNotFound as e:
        print(f"  ✓ Caught CIKNotFound: {e.message}\n")

    # Handle missing data gracefully
    print("Testing graceful handling of missing data:")
    try:
        # Some companies may not have all financial statements
        financials = client.get_financials('AAPL')

        for stmt_name, stmt_data in financials.items():
            if stmt_data.empty:
                print(f"  ⚠ {stmt_name}: No data available")
            else:
                print(f"  ✓ {stmt_name}: {stmt_data.shape}")
    except DataNotAvailable as e:
        print(f"  ⚠ Data not available: {e.message}")

    client.close()


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("EDGAR MODULE - USAGE EXAMPLES")
    print("="*80)
    print("\nThese examples demonstrate common use cases for the EDGAR module.")
    print("Note: Examples make real API calls and respect SEC rate limits.\n")

    examples = [
        example_1_basic_usage,
        example_2_key_metrics,
        example_3_peer_comparison,
        example_4_time_series_analysis,
        example_5_filing_search,
        example_6_convenience_function,
        example_7_industry_ranking,
        example_8_error_handling,
    ]

    for example in examples:
        try:
            example()
        except KeyboardInterrupt:
            print("\n\nExamples interrupted by user.")
            break
        except Exception as e:
            print(f"\n  ✗ Example failed with error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("EXAMPLES COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
