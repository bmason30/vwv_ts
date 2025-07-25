"""
Data management and storage functionality
"""
import pandas as pd
import streamlit as st
import copy
from typing import Optional, Dict, Any
import hashlib

class DataManager:
    """Enhanced data manager with debug control"""

    def __init__(self):
        self._market_data_store = {}
        self._analysis_store = {}

    def store_market_data(self, symbol: str, market_data: pd.DataFrame, show_debug: bool = False):
        """Data storage with debug control"""
        if not isinstance(market_data, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(market_data)}")

        self._market_data_store[symbol] = market_data.copy(deep=True)
        if show_debug:
            st.write(f"🔒 Stored market data for {symbol}: {market_data.shape}")

    def get_market_data_for_analysis(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get copy for analysis"""
        if symbol not in self._market_data_store:
            return None
        return self._market_data_store[symbol].copy(deep=True)

    def get_market_data_for_chart(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get copy for chart"""
        if symbol not in self._market_data_store:
            return None

        chart_copy = self._market_data_store[symbol].copy(deep=True)

        if not isinstance(chart_copy, pd.DataFrame):
            st.error(f"🚨 Chart data corrupted: {type(chart_copy)}")
            return None

        return chart_copy

    def store_analysis_results(self, symbol: str, analysis_results: Dict[str, Any]):
        """Store analysis results"""
        self._analysis_store[symbol] = copy.deepcopy(analysis_results)

    def get_analysis_results(self, symbol: str) -> Dict[str, Any]:
        """Get analysis results"""
        return self._analysis_store.get(symbol, {})
    
    def clear_data(self, symbol: str = None):
        """Clear stored data for symbol or all data"""
        if symbol:
            self._market_data_store.pop(symbol, None)
            self._analysis_store.pop(symbol, None)
        else:
            self._market_data_store.clear()
            self._analysis_store.clear()
    
    def get_stored_symbols(self) -> list:
        """Get list of symbols with stored data"""
        return list(self._market_data_store.keys())
    
    def get_data_summary(self) -> dict:
        """Get summary of stored data"""
        return {
            'market_data_count': len(self._market_data_store),
            'analysis_results_count': len(self._analysis_store),
            'symbols': list(self._market_data_store.keys())
        }

def generate_cache_key(symbol: str, analysis_config: dict) -> str:
    """Generate unique cache key for analysis results"""
    config_str = str(sorted(analysis_config.items()))
    return hashlib.md5(f"{symbol}_{config_str}".encode()).hexdigest()

# Global data manager instance for session state
def get_data_manager() -> DataManager:
    """Get or create data manager instance"""
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DataManager()
    return st.session_state.data_manager
