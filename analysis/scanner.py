"""
Multi-Symbol Master Score Scanner Module
Version: 1.0.0
Purpose: Batch analysis of multiple symbols with Master Score ranking

Scans all Quick Links symbols and calculates Master Scores for comparison.
"""

import streamlit as st
import pandas as pd
import logging
from typing import List, Dict, Any
from datetime import datetime
import time

logger = logging.getLogger(__name__)

# Import configuration
from config.constants import QUICK_LINK_CATEGORIES


def get_quick_links_symbols() -> List[str]:
    """
    Extract all symbols from Quick Links configuration

    Returns:
        List of unique ticker symbols
    """
    try:
        symbols = []
        for category, symbol_list in QUICK_LINK_CATEGORIES.items():
            symbols.extend(symbol_list)

        # Remove duplicates and sort
        unique_symbols = sorted(list(set(symbols)))

        logger.info(f"Loaded {len(unique_symbols)} unique symbols from Quick Links")
        return unique_symbols

    except Exception as e:
        logger.error(f"Error getting Quick Links symbols: {e}")
        return ["SPY", "QQQ", "AAPL"]  # Minimal fallback


def scan_single_symbol(
    ticker: str,
    period: str = "3mo",
    show_debug: bool = False
) -> Dict[str, Any]:
    """
    Run complete analysis on a single symbol and extract Master Score

    Args:
        ticker: Stock symbol to analyze
        period: Data period (default 3mo)
        show_debug: Debug mode flag

    Returns:
        Dict with ticker, master_score, technical_score, current_price, sentiment
    """
    try:
        # Import analysis function
        from app import perform_enhanced_analysis

        # Run analysis
        analysis_results, _ = perform_enhanced_analysis(ticker, period, show_debug)

        if not analysis_results:
            logger.warning(f"No analysis results for {ticker}")
            return {
                'ticker': ticker,
                'master_score': None,
                'technical_score': None,
                'current_price': None,
                'sentiment': 'No Data',
                'error': 'Analysis returned no results',
                'timestamp': datetime.now()
            }

        # Extract master score data
        master_score_data = analysis_results.get('master_score', {})

        if isinstance(master_score_data, dict):
            master_score = master_score_data.get('score', 0)
        else:
            master_score = 0

        # Extract technical score
        tech_score_data = analysis_results.get('enhanced_indicators', {}).get('composite_technical_score', {})
        if isinstance(tech_score_data, dict):
            technical_score = tech_score_data.get('score', 0)
        else:
            technical_score = 0

        # Extract current price
        current_price = analysis_results.get('current_price', 0)

        # Determine sentiment based on Master Score
        if master_score >= 70:
            sentiment = "🟢 Bullish"
        elif master_score >= 55:
            sentiment = "🟡 Neutral"
        elif master_score >= 40:
            sentiment = "🟠 Moderate Bearish"
        else:
            sentiment = "🔴 Bearish"

        result = {
            'ticker': ticker,
            'master_score': round(master_score, 1) if master_score else 0,
            'technical_score': round(technical_score, 1) if technical_score else 0,
            'current_price': round(current_price, 2) if current_price else 0,
            'sentiment': sentiment,
            'error': None,
            'timestamp': datetime.now()
        }

        logger.info(f"✅ {ticker}: Master Score = {master_score:.1f}")
        return result

    except Exception as e:
        logger.error(f"❌ Error analyzing {ticker}: {e}")
        return {
            'ticker': ticker,
            'master_score': None,
            'technical_score': None,
            'current_price': None,
            'sentiment': 'Error',
            'error': str(e),
            'timestamp': datetime.now()
        }


def scan_all_symbols(
    symbols: List[str],
    period: str = "3mo",
    show_debug: bool = False,
    progress_placeholder=None,
    status_placeholder=None
) -> pd.DataFrame:
    """
    Scan all symbols and return results DataFrame

    Args:
        symbols: List of ticker symbols to scan
        period: Data period for analysis
        show_debug: Debug mode flag
        progress_placeholder: Streamlit placeholder for progress bar
        status_placeholder: Streamlit placeholder for status text

    Returns:
        DataFrame with scan results
    """
    try:
        results = []
        total = len(symbols)

        logger.info(f"🔍 Starting scan of {total} symbols")

        for idx, ticker in enumerate(symbols, 1):
            # Update progress
            if progress_placeholder:
                progress = idx / total
                progress_placeholder.progress(progress)

            if status_placeholder:
                status_placeholder.text(f"🔍 Scanning {idx}/{total}: {ticker}...")

            # Scan symbol
            result = scan_single_symbol(
                ticker=ticker,
                period=period,
                show_debug=show_debug
            )

            results.append(result)

            # Small delay to avoid overwhelming the system
            time.sleep(0.2)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Sort by Master Score (descending, nulls last)
        df = df.sort_values('master_score', ascending=False, na_position='last')

        logger.info(f"✅ Scan complete: {len(df)} symbols processed")

        return df

    except Exception as e:
        logger.error(f"Error in batch scan: {e}")
        return pd.DataFrame()


def display_scanner_results(results_df: pd.DataFrame, sort_by: str = "Master Score"):
    """
    Display scanner results in organized table

    Args:
        results_df: DataFrame with scan results
        sort_by: Column to sort by
    """

    st.write("---")
    st.subheader("📊 Scanner Results")

    # Summary statistics
    total_symbols = len(results_df)
    successful = results_df['master_score'].notna().sum()
    failed = total_symbols - successful

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Scanned", total_symbols)

    with col2:
        st.metric("Successful", successful)

    with col3:
        st.metric("Failed", failed)

    with col4:
        if successful > 0:
            avg_score = results_df['master_score'].mean()
            st.metric("Avg Master Score", f"{avg_score:.1f}")
        else:
            st.metric("Avg Master Score", "N/A")

    # Sort results based on user selection
    sort_column_map = {
        "Master Score": "master_score",
        "Technical": "technical_score",
        "Ticker": "ticker"
    }

    sort_column = sort_column_map.get(sort_by, "master_score")

    if sort_by == "Ticker":
        display_df = results_df.sort_values(sort_column, ascending=True)
    else:
        display_df = results_df.sort_values(sort_column, ascending=False, na_position='last')

    # Filter options
    col1, col2 = st.columns(2)

    with col1:
        filter_sentiment = st.multiselect(
            "Filter by Sentiment",
            options=["🟢 Bullish", "🟡 Neutral", "🟠 Moderate Bearish", "🔴 Bearish", "Error", "No Data"],
            default=["🟢 Bullish", "🟡 Neutral", "🟠 Moderate Bearish", "🔴 Bearish"],
            key="scanner_filter_sentiment"
        )

    with col2:
        min_score = st.slider(
            "Minimum Master Score",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            key="scanner_min_score"
        )

    # Apply filters
    filtered_df = display_df[
        (display_df['sentiment'].isin(filter_sentiment)) &
        ((display_df['master_score'] >= min_score) | (display_df['master_score'].isna()))
    ]

    st.write(f"**Showing {len(filtered_df)} of {len(display_df)} symbols**")

    # Prepare display DataFrame
    display_columns = {
        'ticker': 'Ticker',
        'master_score': 'Master Score',
        'technical_score': 'Technical',
        'current_price': 'Price',
        'sentiment': 'Sentiment'
    }

    display_data = filtered_df[list(display_columns.keys())].copy()
    display_data.columns = list(display_columns.values())

    # Format for display
    if not display_data.empty:
        # Format numbers
        for col in ['Master Score', 'Technical']:
            if col in display_data.columns:
                display_data[col] = display_data[col].apply(
                    lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
                )

        if 'Price' in display_data.columns:
            display_data['Price'] = display_data['Price'].apply(
                lambda x: f"${x:.2f}" if pd.notna(x) and x > 0 else "N/A"
            )

    # Display table
    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True,
        height=400
    )

    # Top performers
    st.write("---")
    st.write("**🏆 Top 5 Performers by Master Score:**")

    top_5 = filtered_df.nlargest(5, 'master_score', keep='all')

    if not top_5.empty:
        for idx, row in top_5.iterrows():
            col1, col2, col3, col4 = st.columns([1, 2, 2, 2])

            with col1:
                st.write(f"**{row['ticker']}**")

            with col2:
                if pd.notna(row['master_score']):
                    st.write(f"Master: {row['master_score']:.1f}")
                else:
                    st.write("Master: N/A")

            with col3:
                st.write(row['sentiment'])

            with col4:
                # Click to analyze button
                if st.button(f"📊 Analyze {row['ticker']}", key=f"analyze_{row['ticker']}"):
                    st.session_state.pending_symbol = row['ticker']
                    st.session_state.last_analyzed_symbol = None
                    st.rerun()
    else:
        st.info("No symbols meet the filter criteria")

    # Export and clear options
    st.write("---")
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.session_state.scanner_timestamp:
            timestamp_str = st.session_state.scanner_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            st.write(f"_Last scan: {timestamp_str}_")

    with col2:
        # Export to CSV
        csv = results_df.to_csv(index=False)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vwv_scanner_{timestamp_str}.csv"

        st.download_button(
            label="📥 Export CSV",
            data=csv,
            file_name=filename,
            mime="text/csv",
            key="download_scanner_csv",
            use_container_width=True
        )

    with col3:
        # Clear results
        if st.button("🔄 Clear Results", use_container_width=True):
            st.session_state.scanner_results = None
            st.session_state.scanner_timestamp = None
            st.rerun()


def display_scanner_module(show_debug: bool = False):
    """
    Display Multi-Symbol Master Score Scanner module

    Args:
        show_debug: Debug mode flag
    """

    # Check session state for visibility
    if not st.session_state.get('show_scanner', True):
        return

    # Initialize session state
    if 'scanner_running' not in st.session_state:
        st.session_state.scanner_running = False
    if 'scanner_results' not in st.session_state:
        st.session_state.scanner_results = None
    if 'scanner_timestamp' not in st.session_state:
        st.session_state.scanner_timestamp = None

    with st.expander("🔍 Multi-Symbol Master Score Scanner", expanded=True):
        st.write("Scan all Quick Links symbols to compare Master Scores and identify trading opportunities")

        # Get symbols
        symbols = get_quick_links_symbols()

        # Scanner controls
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.write(f"**Symbols to scan:** {len(symbols)} tickers")
            st.caption(f"{', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")

        with col2:
            period_selection = st.selectbox(
                "Analysis Period",
                options=["1mo", "3mo", "6mo", "1y"],
                index=1,  # Default to 3mo
                key="scanner_period"
            )

        with col3:
            sort_by = st.selectbox(
                "Sort By",
                options=["Master Score", "Technical", "Ticker"],
                index=0,
                key="scanner_sort"
            )

        # Scan button with callback
        def on_scan_click():
            """Callback for scan button"""
            st.session_state.scanner_running = True
            st.session_state.scanner_results = None
            st.session_state.scanner_timestamp = None

        # Scan button
        if st.button(
            "🚀 Scan All Symbols",
            type="primary",
            on_click=on_scan_click,
            use_container_width=True,
            help=f"Run Master Score analysis on all {len(symbols)} Quick Links symbols"
        ):
            pass  # Callback handles state change

        # Execute scan if flag is set
        if st.session_state.scanner_running:

            # Create placeholders for progress
            progress_placeholder = st.empty()
            status_placeholder = st.empty()

            try:
                # Estimate time
                est_time = len(symbols) * 2

                with st.spinner(f"🔍 Scanning {len(symbols)} symbols... Estimated time: ~{est_time}s"):

                    # Run scan
                    results_df = scan_all_symbols(
                        symbols=symbols,
                        period=period_selection,
                        show_debug=show_debug,
                        progress_placeholder=progress_placeholder,
                        status_placeholder=status_placeholder
                    )

                    if not results_df.empty:
                        st.session_state.scanner_results = results_df
                        st.session_state.scanner_timestamp = datetime.now()
                        st.success(f"✅ Scan complete! Analyzed {len(results_df)} symbols")
                        logger.info(f"Scanner results stored: {len(results_df)} symbols")
                    else:
                        st.error("❌ Scan failed - no results returned")
                        logger.error("Scanner returned empty results")

            except Exception as e:
                st.error(f"❌ Scanner error: {str(e)}")
                logger.error(f"Scanner execution error: {e}", exc_info=True)
                if show_debug:
                    st.exception(e)

            finally:
                # Clear progress indicators
                progress_placeholder.empty()
                status_placeholder.empty()

                # Reset running flag
                st.session_state.scanner_running = False

        # Display results if available
        if st.session_state.scanner_results is not None:
            display_scanner_results(
                results_df=st.session_state.scanner_results,
                sort_by=sort_by
            )
