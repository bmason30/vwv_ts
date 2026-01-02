"""
SEC EDGAR Financial Screener UI Module
Streamlit interface for multi-ticker screening with advanced scores
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Optional, Dict, Any, List

from edgar_module import EDGARSession, EDGARScreener
from edgar_module.exceptions import (
    CIKNotFound,
    DataNotAvailable,
    RateLimitExceeded,
    APIError
)
from data.fetcher import get_market_data_enhanced


def initialize_edgar_screener() -> EDGARScreener:
    """Initialize EDGAR screener with proper User-Agent"""
    if 'edgar_session' not in st.session_state:
        st.session_state.edgar_session = EDGARSession(
            user_agent="VWV Trading System/2.0 (vwv.screener@trading.com)",
            enable_cache=True
        )
    if 'edgar_screener' not in st.session_state:
        st.session_state.edgar_screener = EDGARScreener(st.session_state.edgar_session)

    return st.session_state.edgar_screener


def get_price_data_for_tickers(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Get current price and SMA data for tickers using existing vwv data fetcher

    Args:
        tickers: List of ticker symbols

    Returns:
        Dict mapping ticker to price data
    """
    price_data = {}

    for ticker in tickers:
        try:
            # Get market data using existing vwv function
            data = get_market_data_enhanced(ticker, period='3mo', show_debug=False)

            if data is not None and not data.empty:
                current_price = data['Close'].iloc[-1]

                # Calculate 50-day SMA
                if len(data) >= 50:
                    sma_50 = data['Close'].tail(50).mean()
                else:
                    sma_50 = data['Close'].mean()

                # Estimate market cap (would need shares outstanding - using placeholder)
                # In production, this should come from yfinance .info or SEC data
                market_cap = None  # Not calculating for now

                price_data[ticker] = {
                    'price': current_price,
                    'sma_50': sma_50,
                    'market_cap': market_cap
                }
        except Exception as e:
            st.warning(f"Could not get price data for {ticker}: {str(e)}")
            continue

    return price_data


def display_screening_results(df: pd.DataFrame) -> None:
    """Display screening results in a formatted table"""

    if df.empty:
        st.warning("No screening results to display")
        return

    st.markdown("### 📊 Screening Results")

    # Create display DataFrame with formatted columns
    display_df = df.copy()

    # Select columns to display
    display_cols = [
        'ticker',
        'company_name',
        'piotroski_score',
        'altman_zscore',
        'altman_zone',
        'graham_number',
        'price_to_graham_pct',
        'insider_activity',
        'vwv_alpha_score',
        'sma_50_distance_pct'
    ]

    available_cols = [col for col in display_cols if col in display_df.columns]
    display_df = display_df[available_cols]

    # Rename columns for display
    rename_map = {
        'ticker': 'Ticker',
        'company_name': 'Company',
        'piotroski_score': 'Piotroski (0-9)',
        'altman_zscore': 'Altman Z',
        'altman_zone': 'Risk Zone',
        'graham_number': 'Graham #',
        'price_to_graham_pct': 'P/G %',
        'insider_activity': 'Insider',
        'vwv_alpha_score': 'VWV Alpha',
        'sma_50_distance_pct': 'vs 50D SMA'
    }

    display_df = display_df.rename(columns=rename_map)

    # Format numeric columns
    if 'Altman Z' in display_df.columns:
        display_df['Altman Z'] = display_df['Altman Z'].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )

    if 'Graham #' in display_df.columns:
        display_df['Graham #'] = display_df['Graham #'].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
        )

    if 'P/G %' in display_df.columns:
        display_df['P/G %'] = display_df['P/G %'].apply(
            lambda x: f"{x:.0f}%" if pd.notna(x) else "N/A"
        )

    if 'VWV Alpha' in display_df.columns:
        display_df['VWV Alpha'] = display_df['VWV Alpha'].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
        )

    if 'vs 50D SMA' in display_df.columns:
        display_df['vs 50D SMA'] = display_df['vs 50D SMA'].apply(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
        )

    # Display with color coding
    st.dataframe(
        display_df,
        use_container_width=True,
        height=min(600, len(display_df) * 35 + 38)
    )

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name=f"edgar_screening_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


def display_score_distributions(df: pd.DataFrame) -> None:
    """Display visualizations of score distributions"""

    if df.empty:
        return

    st.markdown("### 📈 Score Distributions")

    col1, col2 = st.columns(2)

    with col1:
        if 'piotroski_score' in df.columns:
            # Piotroski histogram
            fig = px.histogram(
                df,
                x='piotroski_score',
                nbins=10,
                title='Piotroski F-Score Distribution',
                labels={'piotroski_score': 'Piotroski Score (0-9)'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'vwv_alpha_score' in df.columns:
            # VWV Alpha histogram
            fig = px.histogram(
                df.dropna(subset=['vwv_alpha_score']),
                x='vwv_alpha_score',
                nbins=20,
                title='VWV Alpha Score Distribution',
                labels={'vwv_alpha_score': 'VWV Alpha Score (0-100)'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Scatter plot: Piotroski vs Altman
    if 'piotroski_score' in df.columns and 'altman_zscore' in df.columns:
        st.markdown("### 📊 Quality vs. Financial Health")

        scatter_df = df.dropna(subset=['piotroski_score', 'altman_zscore'])

        if not scatter_df.empty:
            fig = px.scatter(
                scatter_df,
                x='piotroski_score',
                y='altman_zscore',
                hover_data=['ticker', 'company_name'],
                title='Piotroski F-Score vs Altman Z-Score',
                labels={
                    'piotroski_score': 'Piotroski Score (Quality)',
                    'altman_zscore': 'Altman Z-Score (Health)'
                }
            )

            # Add zone lines for Altman
            fig.add_hline(y=2.99, line_dash="dash", line_color="green",
                         annotation_text="Safe Zone")
            fig.add_hline(y=1.81, line_dash="dash", line_color="orange",
                         annotation_text="Distress Zone")

            st.plotly_chart(fig, use_container_width=True)


def display_top_picks(df: pd.DataFrame, n: int = 5) -> None:
    """Display top N stocks by VWV Alpha Score"""

    if df.empty or 'vwv_alpha_score' not in df.columns:
        return

    st.markdown(f"### 🏆 Top {n} Value Picks (by VWV Alpha Score)")

    # Filter out rows with null scores
    top_df = df.dropna(subset=['vwv_alpha_score']).head(n)

    if top_df.empty:
        st.info("No stocks with complete scoring data")
        return

    for idx, row in top_df.iterrows():
        with st.expander(f"**{row['ticker']}** - {row.get('company_name', 'N/A')} (Score: {row['vwv_alpha_score']:.1f})"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Piotroski F-Score", f"{row.get('piotroski_score', 'N/A')}/9")
                st.metric("Altman Z-Score", f"{row.get('altman_zscore', 'N/A'):.2f}" if pd.notna(row.get('altman_zscore')) else "N/A")

            with col2:
                st.metric("Graham Number", f"${row.get('graham_number', 0):.2f}" if pd.notna(row.get('graham_number')) else "N/A")
                st.metric("Price/Graham", f"{row.get('price_to_graham_pct', 0):.0f}%" if pd.notna(row.get('price_to_graham_pct')) else "N/A")

            with col3:
                st.metric("Insider Activity", row.get('insider_activity', 'N/A'))
                st.metric("vs 50D SMA", f"{row.get('sma_50_distance_pct', 0):+.1f}%" if pd.notna(row.get('sma_50_distance_pct')) else "N/A")

            # Interpretation
            risk_zone = row.get('altman_zone', 'N/A')
            if risk_zone == 'Safe Zone':
                st.success(f"✅ {risk_zone} - Low bankruptcy risk")
            elif risk_zone == 'Grey Zone':
                st.warning(f"⚠️ {risk_zone} - Moderate risk")
            elif risk_zone == 'Distress Zone':
                st.error(f"🚨 {risk_zone} - High bankruptcy risk")


def render_edgar_page():
    """Main EDGAR screener page renderer for Streamlit"""

    st.markdown("## 📄 SEC EDGAR Value Screener")
    st.markdown("**Advanced Financial Screening with Piotroski, Altman, Graham & Insider Analysis**")

    # Initialize screener
    screener = initialize_edgar_screener()

    # Minimal sidebar settings
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        # Options
        include_price_data = st.checkbox(
            "Include Price Data (50D SMA)",
            value=True,
            help="Fetch current prices and calculate distance from 50-day moving average"
        )

        min_alpha_score = st.slider(
            "Minimum VWV Alpha Score",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            help="Filter results to show only stocks above this score"
        )

        st.markdown("---")

        # Cache management
        if st.button("Clear Cache", key="edgar_clear_cache"):
            count = st.session_state.edgar_session.clear_cache()
            st.success(f"Cleared {count} cache files")

    # TOP OF PAGE: Ticker input and Run button
    st.markdown("### Enter Tickers to Screen")

    col1, col2 = st.columns([3, 1])

    with col1:
        tickers_input = st.text_area(
            "Stock Tickers (one per line or comma-separated)",
            value="AAPL\nMSFT\nGOOGL\nTSLA\nAMZN",
            height=100,
            key="edgar_screener_tickers",
            label_visibility="collapsed"
        )

    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("🔍 RUN SCREENING", key="edgar_run_screen", type="primary", use_container_width=True):
            st.session_state.edgar_screen_clicked = True

    # Parse tickers
    tickers = []
    for line in tickers_input.split('\n'):
        for ticker in line.split(','):
            ticker = ticker.strip().upper()
            if ticker:
                tickers.append(ticker)

    st.info(f"📊 Ready to screen **{len(tickers)}** tickers")
    st.markdown("---")

    # Main content
    if st.session_state.get('edgar_screen_clicked', False) and len(tickers) > 0:

        with st.spinner(f"Screening {len(tickers)} tickers... This may take a few minutes."):

            # Get price data if requested
            price_data_dict = None
            if include_price_data:
                with st.spinner("Fetching current price data..."):
                    price_data_dict = get_price_data_for_tickers(tickers)

            # Run screening
            try:
                results_df = screener.screen_multiple_tickers(
                    tickers,
                    include_price_data=include_price_data,
                    price_data_dict=price_data_dict,
                    max_workers=3  # Respect rate limits
                )

                if not results_df.empty:
                    # Store results in session state
                    st.session_state.screening_results = results_df

                    # Filter by minimum score
                    if 'vwv_alpha_score' in results_df.columns:
                        filtered_df = results_df[
                            results_df['vwv_alpha_score'] >= min_alpha_score
                        ]
                    else:
                        filtered_df = results_df

                    # Display results
                    if not filtered_df.empty:
                        display_top_picks(filtered_df, n=5)
                        st.markdown("---")
                        display_screening_results(filtered_df)
                        st.markdown("---")
                        display_score_distributions(results_df)
                    else:
                        st.warning(f"No stocks meet the minimum VWV Alpha Score of {min_alpha_score}")
                        st.info("Try lowering the minimum score or add more tickers to screen")

                else:
                    st.error("No screening results returned. Check your tickers and try again.")

            except RateLimitExceeded:
                st.error("⚠️ SEC rate limit exceeded. Please wait a moment and try again with fewer tickers.")
            except Exception as e:
                st.error(f"Screening error: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

    elif st.session_state.get('edgar_screen_clicked', False):
        st.warning("Please enter at least one ticker symbol to screen")

    else:
        # Welcome screen
        st.info("👈 Enter ticker symbols and click 'Run Screening' to analyze stocks")

        st.markdown("### 📚 What This Screener Does")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Financial Health Scores:**
            - **Piotroski F-Score (0-9)**: Quality and value indicator
              - 8-9: Very Strong
              - 6-7: Strong
              - 4-5: Moderate
              - 0-3: Weak

            - **Altman Z-Score**: Bankruptcy prediction
              - Z > 2.99: Safe Zone
              - 1.81-2.99: Grey Zone
              - Z < 1.81: Distress Zone
            """)

        with col2:
            st.markdown("""
            **Valuation Metrics:**
            - **Graham Number**: Maximum fair value
              - Compares current price to intrinsic value
              - Based on EPS and Book Value

            - **VWV Alpha Score (0-100)**: Combined quality score
              - Weighted combination of all metrics
              - Higher = Better value/quality

            - **Insider Activity**: Recent Form 4 filings
            """)

        st.markdown("### 💡 How to Use")
        st.markdown("""
        1. Enter ticker symbols in the sidebar (one per line or comma-separated)
        2. Optionally enable price data for SMA analysis
        3. Set minimum VWV Alpha Score filter
        4. Click "Run Screening"
        5. Review top picks and detailed results
        6. Download results as CSV for further analysis
        """)
