"""
SEC EDGAR Financial Data UI Module
Streamlit interface for the EDGAR module
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Optional, Dict, Any, List

from edgar_module import EDGARClient
from edgar_module.exceptions import (
    CIKNotFound,
    DataNotAvailable,
    RateLimitExceeded,
    APIError
)


def initialize_edgar_client() -> EDGARClient:
    """Initialize EDGAR client with proper User-Agent"""
    if 'edgar_client' not in st.session_state:
        st.session_state.edgar_client = EDGARClient(
            user_agent="VWV Trading System/1.0 (vwv@trading.com)",
            enable_cache=True
        )
    return st.session_state.edgar_client


def display_company_info(client: EDGARClient, ticker: str) -> None:
    """Display company information card"""
    try:
        info = client.get_company_info(ticker)

        st.markdown("### Company Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Company Name", info.get('name', 'N/A'))
            st.metric("CIK", info.get('cik', 'N/A'))

        with col2:
            st.metric("Industry", info.get('sicDescription', 'N/A')[:30] + '...')
            st.metric("SIC Code", info.get('sic', 'N/A'))

        with col3:
            tickers = ', '.join(info.get('tickers', []))
            st.metric("Tickers", tickers if tickers else 'N/A')
            st.metric("State", info.get('stateOfIncorporation', 'N/A'))

    except Exception as e:
        st.error(f"Error fetching company info: {str(e)}")


def display_financial_statements(
    client: EDGARClient,
    ticker: str,
    annual: bool = True
) -> None:
    """Display financial statements"""
    try:
        with st.spinner(f"Fetching {'annual' if annual else 'quarterly'} financial data..."):
            financials = client.get_financials(ticker, annual=annual)

        tabs = st.tabs(["📊 Balance Sheet", "💰 Income Statement", "💵 Cash Flow", "📈 Key Metrics"])

        # Balance Sheet Tab
        with tabs[0]:
            bs = financials.get('balance_sheet', pd.DataFrame())
            if not bs.empty:
                st.markdown("#### Balance Sheet")

                # Select key metrics to display
                key_cols = ['Assets', 'AssetsCurrent', 'Liabilities', 'LiabilitiesCurrent', 'StockholdersEquity']
                available_cols = [col for col in key_cols if col in bs.columns]

                if available_cols:
                    display_df = bs[available_cols].head(5)
                    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d')

                    # Format as currency
                    for col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")

                    st.dataframe(display_df, use_container_width=True)

                    # Chart
                    if 'Assets' in bs.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=bs.index,
                            y=bs['Assets'],
                            mode='lines+markers',
                            name='Total Assets'
                        ))
                        fig.update_layout(
                            title='Total Assets Over Time',
                            xaxis_title='Date',
                            yaxis_title='Assets (USD)',
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No key balance sheet metrics available")
            else:
                st.warning("Balance sheet data not available")

        # Income Statement Tab
        with tabs[1]:
            income = financials.get('income_statement', pd.DataFrame())
            if not income.empty:
                st.markdown("#### Income Statement")

                # Select key metrics
                key_cols = ['Revenues', 'GrossProfit', 'OperatingIncomeLoss', 'NetIncomeLoss']
                # Also try alternative revenue concept
                if 'Revenues' not in income.columns:
                    key_cols[0] = 'RevenueFromContractWithCustomerExcludingAssessedTax'

                available_cols = [col for col in key_cols if col in income.columns]

                if available_cols:
                    display_df = income[available_cols].head(5)
                    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d')

                    # Format as currency
                    for col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")

                    st.dataframe(display_df, use_container_width=True)

                    # Chart - Revenue and Net Income
                    fig = go.Figure()
                    revenue_col = 'Revenues' if 'Revenues' in income.columns else 'RevenueFromContractWithCustomerExcludingAssessedTax'
                    if revenue_col in income.columns:
                        fig.add_trace(go.Scatter(
                            x=income.index,
                            y=income[revenue_col],
                            mode='lines+markers',
                            name='Revenue'
                        ))
                    if 'NetIncomeLoss' in income.columns:
                        fig.add_trace(go.Scatter(
                            x=income.index,
                            y=income['NetIncomeLoss'],
                            mode='lines+markers',
                            name='Net Income'
                        ))

                    fig.update_layout(
                        title='Revenue and Net Income Over Time',
                        xaxis_title='Date',
                        yaxis_title='Amount (USD)',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No key income statement metrics available")
            else:
                st.warning("Income statement data not available")

        # Cash Flow Tab
        with tabs[2]:
            cf = financials.get('cash_flow', pd.DataFrame())
            if not cf.empty:
                st.markdown("#### Cash Flow Statement")

                # Select key metrics
                key_cols = [
                    'NetCashProvidedByUsedInOperatingActivities',
                    'NetCashProvidedByUsedInInvestingActivities',
                    'NetCashProvidedByUsedInFinancingActivities'
                ]
                available_cols = [col for col in key_cols if col in cf.columns]

                if available_cols:
                    display_df = cf[available_cols].head(5)
                    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d')

                    # Rename for display
                    display_df.columns = ['Operating CF', 'Investing CF', 'Financing CF']

                    # Format as currency
                    for col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")

                    st.dataframe(display_df, use_container_width=True)

                    # Chart
                    fig = go.Figure()
                    for col in available_cols:
                        short_name = col.replace('NetCashProvidedByUsedIn', '').replace('Activities', '')
                        fig.add_trace(go.Bar(
                            x=cf.index,
                            y=cf[col],
                            name=short_name
                        ))

                    fig.update_layout(
                        title='Cash Flow Components',
                        xaxis_title='Date',
                        yaxis_title='Cash Flow (USD)',
                        hovermode='x unified',
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No key cash flow metrics available")
            else:
                st.warning("Cash flow data not available")

        # Key Metrics Tab
        with tabs[3]:
            try:
                metrics = client.get_key_metrics(ticker, annual=annual)
                if not metrics.empty:
                    st.markdown("#### Key Financial Metrics")

                    # Display latest metrics
                    latest_date = metrics.index[0]
                    st.markdown(f"**As of:** {pd.to_datetime(latest_date).strftime('%Y-%m-%d')}")

                    metric_cols = st.columns(4)

                    idx = 0
                    for col_name in metrics.columns:
                        if idx >= 4:
                            break
                        with metric_cols[idx]:
                            value = metrics[col_name].iloc[0]
                            if pd.notna(value):
                                if 'Ratio' in col_name:
                                    st.metric(col_name, f"{value:.2f}")
                                elif 'Margin' in col_name or 'Return' in col_name:
                                    st.metric(col_name, f"{value:.2%}")
                                else:
                                    st.metric(col_name, f"${value:,.0f}")
                                idx += 1

                    # Chart metrics over time
                    st.markdown("#### Metrics Trends")

                    # Select a few key metrics to chart
                    chart_metrics = []
                    for m in ['ProfitMargin', 'ReturnOnEquity', 'ReturnOnAssets', 'CurrentRatio']:
                        if m in metrics.columns:
                            chart_metrics.append(m)

                    if chart_metrics:
                        fig = go.Figure()
                        for metric in chart_metrics:
                            fig.add_trace(go.Scatter(
                                x=metrics.index,
                                y=metrics[metric],
                                mode='lines+markers',
                                name=metric
                            ))

                        fig.update_layout(
                            title='Key Metrics Over Time',
                            xaxis_title='Date',
                            yaxis_title='Value',
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Key metrics not available")
            except Exception as e:
                st.error(f"Error calculating key metrics: {str(e)}")

    except CIKNotFound:
        st.error(f"Company ticker '{ticker}' not found in SEC database")
    except DataNotAvailable as e:
        st.warning(f"Financial data not available: {str(e)}")
    except Exception as e:
        st.error(f"Error fetching financial statements: {str(e)}")


def display_peer_comparison(
    client: EDGARClient,
    tickers: List[str],
    metric: str = 'Revenues'
) -> None:
    """Display peer comparison"""
    try:
        with st.spinner("Comparing companies..."):
            comparison = client.compare_metrics(tickers, metric, annual_only=True)

        if not comparison.empty:
            st.markdown(f"### {metric} Comparison")

            # Convert CIKs back to tickers for display
            ticker_map = {}
            for ticker in tickers:
                try:
                    cik = client.ticker_to_cik(ticker)
                    ticker_map[cik] = ticker
                except:
                    pass

            # Rename columns from CIK to ticker
            display_df = comparison.copy()
            display_df.columns = [ticker_map.get(col, col) for col in display_df.columns]

            # Show table
            display_table = display_df.head(5).copy()
            display_table.index = pd.to_datetime(display_table.index).strftime('%Y-%m-%d')

            # Format as currency
            for col in display_table.columns:
                display_table[col] = display_table[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")

            st.dataframe(display_table, use_container_width=True)

            # Chart
            fig = go.Figure()
            for col in display_df.columns:
                fig.add_trace(go.Scatter(
                    x=display_df.index,
                    y=display_df[col],
                    mode='lines+markers',
                    name=col
                ))

            fig.update_layout(
                title=f'{metric} Comparison Over Time',
                xaxis_title='Date',
                yaxis_title=f'{metric} (USD)',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No comparison data available")

    except Exception as e:
        st.error(f"Error in peer comparison: {str(e)}")


def display_filings(client: EDGARClient, ticker: str) -> None:
    """Display recent filings"""
    try:
        with st.spinner("Fetching filings..."):
            filings = client.get_filings(ticker)

        if not filings.empty:
            st.markdown("### Recent SEC Filings")

            # Filter options
            col1, col2 = st.columns([1, 3])
            with col1:
                form_types = ['All'] + sorted(filings['form'].unique().tolist())
                selected_form = st.selectbox("Form Type", form_types, key="filing_form_filter")

            # Filter filings
            if selected_form != 'All':
                display_filings = filings[filings['form'] == selected_form].copy()
            else:
                display_filings = filings.copy()

            # Display table
            display_cols = ['filingDate', 'form', 'accessionNumber']
            if all(col in display_filings.columns for col in display_cols):
                display_table = display_filings[display_cols].head(20).copy()
                display_table['filingDate'] = pd.to_datetime(display_table['filingDate']).dt.strftime('%Y-%m-%d')
                display_table.columns = ['Filing Date', 'Form Type', 'Accession Number']

                st.dataframe(display_table, use_container_width=True, height=400)

                # Link to latest 10-K
                st.markdown("#### Quick Links")
                col1, col2, col3 = st.columns(3)

                try:
                    with col1:
                        url_10k = client.get_filing_url(ticker, '10-K')
                        st.markdown(f"[Latest 10-K Filing]({url_10k})")
                except:
                    pass

                try:
                    with col2:
                        url_10q = client.get_filing_url(ticker, '10-Q')
                        st.markdown(f"[Latest 10-Q Filing]({url_10q})")
                except:
                    pass

                try:
                    with col3:
                        url_8k = client.get_filing_url(ticker, '8-K')
                        st.markdown(f"[Latest 8-K Filing]({url_8k})")
                except:
                    pass
            else:
                st.warning("Filing data format not recognized")
        else:
            st.warning("No filings found")

    except Exception as e:
        st.error(f"Error fetching filings: {str(e)}")


def render_edgar_page():
    """Main EDGAR page renderer for Streamlit"""
    st.markdown("## 📄 SEC EDGAR Financial Data")

    # Initialize client
    client = initialize_edgar_client()

    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎯 Analysis Settings")

        mode = st.radio(
            "Analysis Mode",
            ["Single Company", "Peer Comparison", "Filings"],
            key="edgar_mode"
        )

        st.markdown("---")

        if mode == "Single Company":
            ticker = st.text_input(
                "Company Ticker",
                value="AAPL",
                key="edgar_ticker"
            ).upper().strip()

            data_type = st.radio(
                "Data Frequency",
                ["Annual (10-K)", "Quarterly (10-Q)"],
                key="edgar_frequency"
            )
            annual = data_type == "Annual (10-K)"

            if st.button("🔍 Analyze", key="edgar_analyze", type="primary", use_container_width=True):
                st.session_state.edgar_analyze_clicked = True

        elif mode == "Peer Comparison":
            tickers_input = st.text_input(
                "Company Tickers (comma-separated)",
                value="AAPL,MSFT,GOOGL",
                key="edgar_peer_tickers"
            )
            tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

            metric = st.selectbox(
                "Metric to Compare",
                ["Revenues", "Assets", "NetIncomeLoss", "StockholdersEquity"],
                key="edgar_peer_metric"
            )

            if st.button("🔍 Compare", key="edgar_compare", type="primary", use_container_width=True):
                st.session_state.edgar_compare_clicked = True

        else:  # Filings
            ticker = st.text_input(
                "Company Ticker",
                value="AAPL",
                key="edgar_filings_ticker"
            ).upper().strip()

            if st.button("🔍 Get Filings", key="edgar_filings_btn", type="primary", use_container_width=True):
                st.session_state.edgar_filings_clicked = True

    # Main content
    if mode == "Single Company":
        if st.session_state.get('edgar_analyze_clicked', False):
            if ticker:
                display_company_info(client, ticker)
                st.markdown("---")
                display_financial_statements(client, ticker, annual)
            else:
                st.warning("Please enter a company ticker")
        else:
            st.info("👈 Enter a company ticker and click 'Analyze' to view SEC financial data")

    elif mode == "Peer Comparison":
        if st.session_state.get('edgar_compare_clicked', False):
            if len(tickers) >= 2:
                display_peer_comparison(client, tickers, metric)
            else:
                st.warning("Please enter at least 2 company tickers for comparison")
        else:
            st.info("👈 Enter company tickers and click 'Compare' to analyze peers")

    else:  # Filings
        if st.session_state.get('edgar_filings_clicked', False):
            if ticker:
                display_filings(client, ticker)
            else:
                st.warning("Please enter a company ticker")
        else:
            st.info("👈 Enter a company ticker and click 'Get Filings' to view SEC filings")

    # Cache management
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔧 Settings")
        if st.button("Clear Cache", key="edgar_clear_cache"):
            count = client.clear_cache()
            st.success(f"Cleared {count} cache files")
