"""
File: app.py
VWV Professional Trading System v4.2.2 - Modular Architecture
Version: v4.2.2-MODULAR-ISOLATION-2025-08-27-19-45-00-EST
PURPOSE: Module independence - changes to one module cannot break others
ARCHITECTURE: Registry-based module loading with safe isolation
Last Updated: August 27, 2025 - 7:45 PM EST
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import logging

# Core system imports (always safe)
from config.settings import DEFAULT_VWV_CONFIG, UI_SETTINGS, PARAMETER_RANGES
from config.constants import SYMBOL_DESCRIPTIONS, QUICK_LINK_CATEGORIES, MAJOR_INDICES
from data.manager import get_data_manager
from data.fetcher import get_market_data_enhanced, is_etf
from ui.components import create_header, format_large_number
from utils.helpers import get_market_status, get_etf_description
from utils.decorators import safe_calculation_wrapper

# New modular architecture imports
from core.module_loader import get_module_loader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='yfinance')

# Page configuration
st.set_page_config(
    page_title="VWV Professional Trading System v4.2.2",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class VWVTradingApp:
    """Main trading application with modular architecture"""
    
    def __init__(self):
        self.module_loader = get_module_loader()
        self.registry = self.module_loader.get_registry()
        self.data_manager = get_data_manager()
    
    def create_sidebar_controls(self):
        """Create sidebar controls with module registry integration"""
        st.sidebar.title("VWV Trading System v4.2.2")
        st.sidebar.caption("Modular Architecture - Module Independence Guaranteed")
        
        # Symbol input
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            symbol_input = st.text_input("Enter Symbol", value="", placeholder="AAPL, TSLA, SPY...")
        with col2:
            analyze_button = st.button("Analyze", use_container_width=True, type="primary")

        # Time period selection
        period = st.sidebar.selectbox(
            "Time Period",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            index=0,
            help="Select analysis time period"
        )

        # Quick Links
        quick_link_clicked = None
        with st.sidebar.expander("Quick Links", expanded=False):
            st.write("**Popular Symbols by Category**")
            
            for category, symbols in QUICK_LINK_CATEGORIES.items():
                with st.expander(f"{category}", expanded=False):
                    cols = st.columns(2)
                    for i, symbol in enumerate(symbols):
                        col = cols[i % 2]
                        with col:
                            description = SYMBOL_DESCRIPTIONS.get(symbol, symbol)
                            if st.button(f"{symbol}", key=f"quick_{symbol}", help=description, use_container_width=True):
                                quick_link_clicked = symbol

        # Recently viewed (session state managed internally)
        if 'recently_viewed' not in st.session_state:
            st.session_state.recently_viewed = []
            
        if st.session_state.recently_viewed:
            with st.sidebar.expander("Recently Viewed", expanded=False):
                recent_cols = st.columns(2)
                for i, recent_symbol in enumerate(st.session_state.recently_viewed[-6:]):
                    col = recent_cols[i % 2]
                    with col:
                        if st.button(f"{recent_symbol}", key=f"recent_{recent_symbol}", use_container_width=True):
                            quick_link_clicked = recent_symbol

        # Module toggles using registry
        with st.sidebar.expander("Analysis Modules", expanded=False):
            self.registry.create_module_toggles()

        # Debug toggle
        show_debug = st.sidebar.checkbox("Debug Mode", value=False)

        # Market status
        market_status = get_market_status()
        if market_status:
            st.sidebar.info(f"Market: {market_status}")

        # Module status display
        if show_debug:
            with st.sidebar.expander("Module Status", expanded=False):
                modules = self.registry.get_all_modules()
                for module_id, status in modules.items():
                    icon = "✅" if status['available'] else "❌"
                    st.write(f"{icon} {status['display_name']}: {status['status']}")

        # Determine final symbol
        final_symbol = None
        final_analyze = False

        if quick_link_clicked:
            final_symbol = quick_link_clicked.upper()
            final_analyze = True
        elif analyze_button and symbol_input:
            final_symbol = symbol_input.upper().strip()
            final_analyze = True

        return {
            'symbol': final_symbol,
            'analyze_button': final_analyze,
            'period': period,
            'show_debug': show_debug
        }
    
    def add_to_recently_viewed(self, symbol):
        """Add symbol to recently viewed list"""
        if symbol and symbol not in st.session_state.recently_viewed:
            st.session_state.recently_viewed.append(symbol)
            if len(st.session_state.recently_viewed) > 10:
                st.session_state.recently_viewed.pop(0)
    
    def show_interactive_charts(self, chart_data, analysis_results, show_debug=False):
        """Display interactive charts with safe fallback"""
        st.subheader("Interactive Charts")
        
        if chart_data is not None and len(chart_data) > 0:
            try:
                # Try to use advanced charts module if available
                try:
                    from charts.plotting import display_trading_charts
                    display_trading_charts(chart_data, analysis_results)
                except ImportError:
                    # Fallback to simple chart
                    st.line_chart(chart_data['Close'])
                    if show_debug:
                        st.info("Using fallback chart display - advanced charts module not available")
                
                if show_debug:
                    st.write(f"Chart data points: {len(chart_data)}")
                    
            except Exception as e:
                st.error(f"Chart display failed: {e}")
                if show_debug:
                    st.exception(e)
        else:
            st.warning("No chart data available")
    
    @safe_calculation_wrapper
    def perform_comprehensive_analysis(self, symbol, period, show_debug=False):
        """Perform analysis using registered modules only"""
        try:
            # Get market data
            data = get_market_data_enhanced(symbol, period)
            if data is None or len(data) == 0:
                raise ValueError(f"No data available for {symbol}")
            
            # Store data
            self.data_manager.store_market_data(symbol, data, period)
            
            # Initialize results structure
            analysis_results = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'enhanced_indicators': {},
                'system_status': 'MODULAR v4.2.2'
            }
            
            # Execute analysis for each available module
            available_modules = self.registry.get_all_modules()
            
            for module_id, module_status in available_modules.items():
                if module_status['available'] and module_status['has_analysis']:
                    try:
                        if show_debug:
                            st.write(f"Executing {module_status['display_name']}...")
                        
                        # Execute analysis based on module type
                        if module_id == 'technical_analysis':
                            result = self.registry.execute_analysis(module_id, data)
                        elif module_id == 'fundamental_analysis':
                            result = self.registry.execute_analysis(module_id, symbol)
                        elif module_id == 'market_analysis':
                            result = self.registry.execute_analysis(module_id, symbol, period)
                        elif module_id == 'baldwin_indicator':
                            result = self.registry.execute_analysis(module_id, show_debug=show_debug)
                        elif module_id in ['volatility_analysis', 'volume_analysis', 'options_analysis']:
                            result = self.registry.execute_analysis(module_id, data)
                        else:
                            continue
                        
                        # Store result if successful
                        if result and result.get('status') != 'execution_error':
                            analysis_results['enhanced_indicators'][module_id] = result
                            
                    except Exception as e:
                        logger.error(f"Module {module_id} analysis failed: {e}")
                        if show_debug:
                            st.error(f"Module {module_id} failed: {e}")
            
            # Store complete results
            self.data_manager.store_analysis_results(symbol, analysis_results)
            
            # Get chart data
            chart_data = self.data_manager.get_market_data_for_chart(symbol)
            
            return analysis_results, chart_data
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            st.error(f"Analysis failed: {str(e)}")
            return None, None
    
    def display_all_analysis(self, analysis_results, show_debug=False):
        """Display all analysis using registered display functions"""
        
        # Get available modules with display functions
        available_modules = self.registry.get_all_modules()
        
        # Define display order (modules will only show if available and enabled)
        display_order = [
            'technical_analysis',
            'volatility_analysis', 
            'volume_analysis',
            'fundamental_analysis',
            'baldwin_indicator',
            'market_analysis',
            'options_analysis'
        ]
        
        for module_id in display_order:
            if module_id in available_modules:
                module_info = available_modules[module_id]
                
                # Check if module is available and has display function
                if module_info['available'] and module_info['has_display']:
                    # Check if user has enabled this module
                    session_key = f'show_{module_id}'
                    if st.session_state.get(session_key, True):
                        try:
                            # Execute display function
                            if module_id == 'baldwin_indicator':
                                # Baldwin is special - can run independently
                                self.registry.execute_display(module_id, analysis_results, show_debug)
                            else:
                                # Regular analysis modules
                                self.registry.execute_display(module_id, analysis_results, show_debug)
                                
                        except Exception as e:
                            logger.error(f"Display failed for {module_id}: {e}")
                            if show_debug:
                                st.error(f"Display error for {module_id}: {e}")
    
    def run(self):
        """Main application execution"""
        # Create header
        create_header()
        
        # Show architecture info
        st.write("## VWV Trading System v4.2.2 - Modular Architecture")
        st.caption("Module Independence: Changes to one module cannot break others")
        
        # Create sidebar and get controls
        controls = self.create_sidebar_controls()
        
        # Module status summary
        modules = self.registry.get_all_modules()
        available_count = sum(1 for m in modules.values() if m['available'])
        total_count = len(modules)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Available Modules", f"{available_count}/{total_count}")
        with col2:
            st.metric("Architecture", "Registry-Based")
        with col3:
            st.metric("Isolation", "Guaranteed")
        
        # Main analysis flow
        if controls['analyze_button'] and controls['symbol']:
            # Add to recently viewed
            self.add_to_recently_viewed(controls['symbol'])
            
            st.write(f"## Analysis Report - {controls['symbol']}")
            
            with st.spinner(f"Analyzing {controls['symbol']}..."):
                # Perform comprehensive analysis
                analysis_results, chart_data = self.perform_comprehensive_analysis(
                    controls['symbol'],
                    controls['period'],
                    controls['show_debug']
                )
                
                if analysis_results and chart_data is not None:
                    # Show charts first
                    self.show_interactive_charts(chart_data, analysis_results, controls['show_debug'])
                    
                    # Display all available analysis
                    self.display_all_analysis(analysis_results, controls['show_debug'])
                    
                    # Debug information
                    if controls['show_debug']:
                        with st.expander("System Debug Information", expanded=False):
                            st.write("### Module Execution Results")
                            for module_id, result in analysis_results.get('enhanced_indicators', {}).items():
                                status = result.get('status', 'unknown')
                                st.write(f"**{module_id}**: {status}")
                            
                            st.write("### Full Analysis Results")
                            st.json(analysis_results, expanded=False)
        
        else:
            # Welcome screen
            st.write("## Welcome to VWV Professional Trading System")
            st.write("**New Modular Architecture Features:**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("**Module Independence**: Each analysis module is isolated - changes to one cannot break others")
                st.info("**Safe Fallbacks**: Modules gracefully degrade when dependencies are unavailable")
                
            with col2:
                st.info("**Registry System**: All modules are registered and managed centrally")
                st.info("**Error Isolation**: Module failures don't cascade to other modules")
            
            # Show module capabilities
            st.write("### Available Analysis Modules")
            modules = self.registry.get_all_modules()
            
            module_data = []
            for module_id, status in modules.items():
                module_data.append({
                    'Module': status['display_name'],
                    'Status': "Available" if status['available'] else "Unavailable",
                    'Description': status['description'],
                    'Version': status['version']
                })
            
            if module_data:
                df_modules = pd.DataFrame(module_data)
                st.dataframe(df_modules, use_container_width=True, hide_index=True)
            
            st.write("Select a symbol from the sidebar to begin analysis.")

def main():
    """Application entry point"""
    try:
        app = VWVTradingApp()
        app.run()
    except Exception as e:
        st.error(f"Application failed to start: {e}")
        st.write("Please check the logs for detailed error information.")

if __name__ == "__main__":
    main()
