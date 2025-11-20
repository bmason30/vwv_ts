"""
Multi-Symbol Master Score Scanner Module
Version: 1.1.0
Purpose: Batch analysis of multiple symbols with Master Score ranking

Scans Quick Links symbols OR custom manual symbol lists and calculates Master Scores for comparison.
Supports manual comma-separated symbol input (2-20 symbols).
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


def parse_manual_symbol_input(input_text: str) -> tuple[List[str], str]:
    """
    Parse and validate manual symbol input

    Args:
        input_text: Comma-separated symbol string (e.g., "AAPL, MSFT, GOOGL")

    Returns:
        Tuple of (valid_symbols_list, error_message)
        If valid: (symbols, "")
        If invalid: ([], error_message)
    """
    try:
        if not input_text or not input_text.strip():
            return [], "Please enter at least 2 symbols separated by commas"

        # Split by commas and clean up
        symbols = [s.strip().upper() for s in input_text.split(',')]

        # Remove empty strings
        symbols = [s for s in symbols if s]

        # Remove duplicates while preserving order
        seen = set()
        unique_symbols = []
        for symbol in symbols:
            if symbol not in seen:
                seen.add(symbol)
                unique_symbols.append(symbol)

        # Validate count
        if len(unique_symbols) < 2:
            return [], "Minimum 2 symbols required. Please enter at least 2 different symbols."

        if len(unique_symbols) > 20:
            return [], f"Maximum 20 symbols allowed. You entered {len(unique_symbols)} symbols."

        # Validate symbol format (1-5 alphanumeric characters)
        invalid_symbols = []
        for symbol in unique_symbols:
            if not symbol.isalnum() or len(symbol) < 1 or len(symbol) > 5:
                invalid_symbols.append(symbol)

        if invalid_symbols:
            return [], f"Invalid symbol format: {', '.join(invalid_symbols)}. Symbols must be 1-5 alphanumeric characters."

        logger.info(f"Validated {len(unique_symbols)} manual symbols: {', '.join(unique_symbols)}")
        return unique_symbols, ""

    except Exception as e:
        logger.error(f"Error parsing manual symbol input: {e}")
        return [], f"Error parsing symbols: {str(e)}"


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
    if 'scanner_mode' not in st.session_state:
        st.session_state.scanner_mode = "Quick Links"
    if 'scanner_manual_input' not in st.session_state:
        st.session_state.scanner_manual_input = ""

    with st.expander("🔍 Multi-Symbol Master Score Scanner", expanded=True):
        st.write("Scan Quick Links symbols or enter custom symbols to compare Master Scores and identify trading opportunities")

        # Mode selection
        mode = st.radio(
            "Symbol Source",
            options=["Quick Links", "Manual Input"],
            index=0 if st.session_state.scanner_mode == "Quick Links" else 1,
            horizontal=True,
            key="scanner_mode_select",
            help="Choose between predefined Quick Links or manually entered symbols"
        )

        # Update session state
        if mode != st.session_state.scanner_mode:
            st.session_state.scanner_mode = mode
            st.session_state.scanner_results = None  # Clear results when mode changes

        # Get symbols based on mode
        symbols = []
        input_error = ""

        if mode == "Quick Links":
            symbols = get_quick_links_symbols()
        else:
            # Manual input mode
            st.write("**Enter symbols separated by commas** (e.g., AAPL, MSFT, GOOGL, TSLA)")

            manual_input = st.text_area(
                "Symbol List",
                value=st.session_state.scanner_manual_input,
                height=80,
                key="manual_symbol_input",
                placeholder="Example: AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA",
                help="Enter 2-20 symbols separated by commas. Spaces are optional."
            )

            # Update session state
            st.session_state.scanner_manual_input = manual_input

            # Parse and validate
            if manual_input and manual_input.strip():
                symbols, input_error = parse_manual_symbol_input(manual_input)

                if input_error:
                    st.error(f"❌ {input_error}")
                else:
                    st.success(f"✅ Valid input: {len(symbols)} symbols ready to scan")
            else:
                input_error = "Please enter at least 2 symbols"

        # Scanner controls
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            if symbols:
                st.write(f"**Symbols to scan:** {len(symbols)} tickers")
                st.caption(f"{', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
            else:
                if mode == "Manual Input":
                    st.write("**No symbols ready** - Enter symbols above")
                else:
                    st.write("**No symbols available** - Quick Links empty")

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

        # Determine button state and label
        button_disabled = (len(symbols) == 0 or input_error != "")

        if mode == "Quick Links":
            button_label = f"🚀 Scan {len(symbols)} Quick Links Symbols"
            button_help = f"Run Master Score analysis on {len(symbols)} predefined symbols"
        else:
            if symbols:
                button_label = f"🚀 Scan {len(symbols)} Custom Symbols"
                button_help = f"Run Master Score analysis on {len(symbols)} manually entered symbols"
            else:
                button_label = "🚀 Scan Custom Symbols"
                button_help = "Enter valid symbols above to enable scanning"

        # Scan button
        if st.button(
            button_label,
            type="primary",
            on_click=on_scan_click if not button_disabled else None,
            use_container_width=True,
            disabled=button_disabled,
            help=button_help
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
