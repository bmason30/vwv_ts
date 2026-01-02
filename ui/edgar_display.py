"""
SEC EDGAR Insider Transaction Monitor UI
Displays top 50 largest insider buys and sells with sortable tables
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Optional, Dict, Any, List

from edgar_module import EDGARSession
from edgar_module.insider_monitor import InsiderTransactionMonitor, create_insider_summary
from edgar_module.exceptions import (
    DataNotAvailable,
    RateLimitExceeded,
    APIError
)


def initialize_insider_monitor() -> InsiderTransactionMonitor:
    """Initialize insider transaction monitor with proper User-Agent"""
    if 'edgar_session' not in st.session_state:
        st.session_state.edgar_session = EDGARSession(
            user_agent="VWV Trading System/2.0 (vwv.insider@trading.com)",
            enable_cache=True
        )
    if 'insider_monitor' not in st.session_state:
        st.session_state.insider_monitor = InsiderTransactionMonitor(st.session_state.edgar_session)

    return st.session_state.insider_monitor


def display_transaction_table(
    df: pd.DataFrame,
    transaction_type: str = "Buy"
) -> None:
    """
    Display insider transaction table with sorting capability

    Args:
        df: DataFrame with transaction data
        transaction_type: "Buy" or "Sell"
    """
    if df.empty:
        st.warning(f"No {transaction_type.lower()} transactions found")
        return

    emoji = "🟢" if transaction_type == "Buy" else "🔴"
    st.markdown(f"### {emoji} Top {len(df)} Largest Insider {transaction_type}s")

    # Format the display DataFrame
    display_df = df.copy()

    # Rename columns for display
    column_mapping = {
        'ticker': 'Ticker',
        'insider_name': 'Insider Name',
        'insider_title': 'Title',
        'filing_date': 'Filing Date',
        'shares': 'Shares',
        'price_per_share': 'Price/Share',
        'estimated_value': 'Transaction Value',
        'transaction_type': 'Type'
    }

    # Select and rename columns
    display_cols = [col for col in column_mapping.keys() if col in display_df.columns]
    display_df = display_df[display_cols].copy()
    display_df = display_df.rename(columns=column_mapping)

    # Format numeric columns
    if 'Transaction Value' in display_df.columns:
        display_df['Transaction Value'] = display_df['Transaction Value'].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) and x > 0 else "N/A"
        )

    if 'Shares' in display_df.columns:
        display_df['Shares'] = display_df['Shares'].apply(
            lambda x: f"{x:,}" if pd.notna(x) and x > 0 else "N/A"
        )

    if 'Price/Share' in display_df.columns:
        display_df['Price/Share'] = display_df['Price/Share'].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) and x > 0 else "N/A"
        )

    if 'Filing Date' in display_df.columns:
        display_df['Filing Date'] = pd.to_datetime(display_df['Filing Date']).dt.strftime('%Y-%m-%d')

    # Display with dataframe (sortable by clicking column headers)
    st.dataframe(
        display_df,
        use_container_width=True,
        height=600
    )

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label=f"📥 Download {transaction_type}s (CSV)",
        data=csv,
        file_name=f"insider_{transaction_type.lower()}s_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key=f"download_{transaction_type}"
    )


def display_summary_stats(buys_df: pd.DataFrame, sells_df: pd.DataFrame) -> None:
    """Display summary statistics for insider activity"""

    st.markdown("### 📊 Insider Activity Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_buys = len(buys_df)
        st.metric("Total Buys", f"{total_buys:,}")

    with col2:
        total_sells = len(sells_df)
        st.metric("Total Sells", f"{total_sells:,}")

    with col3:
        if not buys_df.empty and 'estimated_value' in buys_df.columns:
            total_buy_value = buys_df['estimated_value'].sum()
            st.metric("Total Buy Volume", f"${total_buy_value/1e6:.1f}M")
        else:
            st.metric("Total Buy Volume", "N/A")

    with col4:
        if not sells_df.empty and 'estimated_value' in sells_df.columns:
            total_sell_value = sells_df['estimated_value'].sum()
            st.metric("Total Sell Volume", f"${total_sell_value/1e6:.1f}M")
        else:
            st.metric("Total Sell Volume", "N/A")


def render_edgar_page():
    """Main EDGAR insider transaction monitor page"""

    st.markdown("## 📄 SEC Insider Transaction Monitor")
    st.markdown("**Track Top 50 Largest Insider Buys & Sells from Form 4 Filings**")

    # Initialize monitor
    monitor = initialize_insider_monitor()

    # Minimal sidebar settings
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        days_back = st.slider(
            "Days to Look Back",
            min_value=7,
            max_value=90,
            value=30,
            step=7,
            help="How many days of Form 4 filings to analyze"
        )

        min_value = st.number_input(
            "Min Transaction Value ($)",
            min_value=0,
            max_value=10000000,
            value=100000,
            step=50000,
            help="Minimum transaction value to display"
        )

        st.markdown("---")

        # Cache management
        if st.button("Clear Cache", key="edgar_clear_cache"):
            count = st.session_state.edgar_session.clear_cache()
            st.success(f"Cleared {count} cache files")

    # TOP OF PAGE: Run button
    st.markdown("### Monitor Insider Transactions")

    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:
        if st.button("🔍 SCAN FOR INSIDER ACTIVITY", key="edgar_run_scan", type="primary", use_container_width=True):
            st.session_state.edgar_scan_clicked = True

    st.markdown("---")

    # Main content
    if st.session_state.get('edgar_scan_clicked', False):

        with st.spinner(f"Scanning Form 4 filings from last {days_back} days... This may take a moment."):

            try:
                # Get top buys and sells
                buys_df, sells_df = monitor.get_top_insider_transactions(
                    days_back=days_back,
                    min_value=min_value
                )

                # Create summary
                summary = create_insider_summary(buys_df, sells_df, top_n=50)

                # Store in session state
                st.session_state.insider_buys = summary['top_buys']
                st.session_state.insider_sells = summary['top_sells']

                # Display summary stats
                display_summary_stats(summary['top_buys'], summary['top_sells'])

                st.markdown("---")

                # Display tables in tabs
                tab1, tab2 = st.tabs(["🟢 Top Buys", "🔴 Top Sells"])

                with tab1:
                    display_transaction_table(summary['top_buys'], "Buy")

                with tab2:
                    display_transaction_table(summary['top_sells'], "Sell")

                # Note about data source
                st.info("""
                **Note:** This monitor currently tracks Form 4 filings from a watchlist of major stocks.

                **Current Limitations:**
                - Covers ~80 major tickers (mega/large cap stocks)
                - Transaction details are estimated (full Form 4 XML parsing pending)
                - Real-time updates require SEC RSS feed integration

                **Roadmap:**
                - Full Form 4 XML parsing for actual transaction details
                - Complete coverage of all SEC filers
                - Real-time RSS feed monitoring
                - Historical transaction trends
                """)

            except RateLimitExceeded:
                st.error("⚠️ SEC rate limit exceeded. Please wait a moment and try again.")
            except Exception as e:
                st.error(f"Error scanning insider activity: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

    else:
        # Welcome screen
        st.info("👆 Click 'SCAN FOR INSIDER ACTIVITY' to view recent insider transactions")

        st.markdown("### 📚 How This Works")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **What You'll See:**
            - **Top 50 Largest Buys**: Insider purchases sorted by transaction value
            - **Top 50 Largest Sells**: Insider sales sorted by transaction value

            **For Each Transaction:**
            - Ticker symbol
            - Insider name and title
            - Filing date
            - Number of shares
            - Price per share
            - Total transaction value
            """)

        with col2:
            st.markdown("""
            **Use Cases:**
            - Identify bullish insider sentiment (large buys)
            - Spot potential red flags (large sells by executives)
            - Track insider activity by company
            - Monitor trending stocks among insiders

            **Sorting:**
            - Click any column header to sort
            - Multi-column sorting available
            - Download results as CSV for further analysis
            """)

        st.markdown("### 💡 Interpreting Insider Activity")

        st.markdown("""
        **🟢 Insider Buys (Bullish Signals):**
        - **C-Level Executives**: CEO, CFO, COO buying = strong confidence
        - **Directors**: Board members buying = positive outlook
        - **Cluster Buying**: Multiple insiders buying simultaneously = very bullish
        - **Large Positions**: Buys >$500K = significant commitment

        **🔴 Insider Sells (Interpret Carefully):**
        - **Planned Sales**: Often routine (stock options, diversification)
        - **Multiple Executives**: Several insiders selling = potential concern
        - **Unusual Timing**: Sells before earnings/news = red flag
        - **C-Level Mass Exits**: Top executives selling heavily = warning sign

        **⚖️ Context Matters:**
        - Compare sell vs. remaining holdings
        - Check if part of 10b5-1 trading plan
        - Consider overall market conditions
        - Review company recent news/events
        """)
