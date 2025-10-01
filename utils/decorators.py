"""
Utility decorators for safe calculations and error handling - v2.0 FIXED
Now shows errors in Streamlit when they occur
"""
import logging
import functools
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_calculation_wrapper(func):
    """Decorator for safe financial calculations with Streamlit error display"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if result is None:
                logger.warning(f"Function {func.__name__} returned None")
                # Check if we're in a Streamlit context and show_debug is True
                if hasattr(st, 'session_state'):
                    try:
                        if st.session_state.get('show_debug', False):
                            st.warning(f"⚠️ {func.__name__} returned None")
                    except:
                        pass  # Not in Streamlit context
            return result
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            logger.error(error_msg)
            
            # Try to show error in Streamlit if available
            try:
                st.error(f"❌ {error_msg}")
                # Show full traceback in debug mode
                if hasattr(st, 'session_state') and st.session_state.get('show_debug', False):
                    import traceback
                    st.code(traceback.format_exc())
            except:
                pass  # Not in Streamlit context, just log
            
            return None
    return wrapper
