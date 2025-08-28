"""
File: core/module_registry.py
Robust Module Registry System for VWV Trading System
Version: v4.2.2-ARCHITECTURE-REFACTOR-2025-08-27-19-30-00-EST
PURPOSE: Prevent cross-module contamination and enable safe module independence
Last Updated: August 27, 2025 - 7:30 PM EST
"""

from typing import Dict, Any, Callable, Optional, List
import logging
import streamlit as st
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ModuleStatus(Enum):
    """Module availability status"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    DISABLED = "disabled"

@dataclass
class ModuleInfo:
    """Information about a registered module"""
    name: str
    display_name: str
    description: str
    version: str
    status: ModuleStatus
    analysis_function: Optional[Callable] = None
    display_function: Optional[Callable] = None
    dependencies: List[str] = None
    error_message: Optional[str] = None

class ModuleRegistry:
    """Central registry for all analysis and display modules"""
    
    def __init__(self):
        self.modules: Dict[str, ModuleInfo] = {}
        self._session_state_initialized = False
    
    def register_module(self, 
                       module_id: str,
                       display_name: str, 
                       description: str,
                       version: str,
                       analysis_function: Optional[Callable] = None,
                       display_function: Optional[Callable] = None,
                       dependencies: List[str] = None) -> bool:
        """Register a module with safe error handling"""
        
        try:
            # Test if analysis function is callable
            status = ModuleStatus.AVAILABLE
            error_msg = None
            
            if analysis_function:
                if not callable(analysis_function):
                    status = ModuleStatus.ERROR
                    error_msg = f"Analysis function for {module_id} is not callable"
            
            if display_function:
                if not callable(display_function):
                    status = ModuleStatus.ERROR  
                    error_msg = f"Display function for {module_id} is not callable"
            
            # Check dependencies
            if dependencies:
                for dep in dependencies:
                    if dep not in self.modules or self.modules[dep].status != ModuleStatus.AVAILABLE:
                        status = ModuleStatus.UNAVAILABLE
                        error_msg = f"Missing dependency: {dep}"
                        break
            
            self.modules[module_id] = ModuleInfo(
                name=module_id,
                display_name=display_name,
                description=description,
                version=version,
                status=status,
                analysis_function=analysis_function,
                display_function=display_function,
                dependencies=dependencies or [],
                error_message=error_msg
            )
            
            logger.info(f"Module registered: {module_id} - Status: {status.value}")
            return status == ModuleStatus.AVAILABLE
            
        except Exception as e:
            logger.error(f"Failed to register module {module_id}: {e}")
            self.modules[module_id] = ModuleInfo(
                name=module_id,
                display_name=display_name,
                description=description,
                version=version,
                status=ModuleStatus.ERROR,
                error_message=str(e)
            )
            return False
    
    def is_available(self, module_id: str) -> bool:
        """Check if module is available for use"""
        if module_id not in self.modules:
            return False
        return self.modules[module_id].status == ModuleStatus.AVAILABLE
    
    def get_analysis_function(self, module_id: str) -> Optional[Callable]:
        """Safely get analysis function"""
        if self.is_available(module_id):
            return self.modules[module_id].analysis_function
        return None
    
    def get_display_function(self, module_id: str) -> Optional[Callable]:
        """Safely get display function"""
        if self.is_available(module_id):
            return self.modules[module_id].display_function
        return None
    
    def execute_analysis(self, module_id: str, *args, **kwargs) -> Dict[str, Any]:
        """Safely execute analysis function with error handling"""
        if not self.is_available(module_id):
            return {
                'error': f'Module {module_id} not available',
                'status': 'unavailable'
            }
        
        try:
            analysis_func = self.get_analysis_function(module_id)
            if analysis_func:
                result = analysis_func(*args, **kwargs)
                if isinstance(result, dict):
                    result['module_id'] = module_id
                    result['status'] = result.get('status', 'success')
                return result
            else:
                return {
                    'error': f'No analysis function for {module_id}',
                    'status': 'no_function'
                }
        except Exception as e:
            logger.error(f"Analysis execution failed for {module_id}: {e}")
            return {
                'error': f'Analysis execution failed: {str(e)}',
                'status': 'execution_error',
                'module_id': module_id
            }
    
    def execute_display(self, module_id: str, *args, **kwargs) -> bool:
        """Safely execute display function with error handling"""
        if not self.is_available(module_id):
            st.warning(f"Module {self.modules.get(module_id, {}).get('display_name', module_id)} not available")
            return False
        
        try:
            display_func = self.get_display_function(module_id)
            if display_func:
                display_func(*args, **kwargs)
                return True
            else:
                st.warning(f"No display function for {module_id}")
                return False
        except Exception as e:
            logger.error(f"Display execution failed for {module_id}: {e}")
            st.error(f"Display error for {module_id}: {str(e)}")
            return False
    
    def get_module_status(self, module_id: str) -> Dict[str, Any]:
        """Get detailed module status information"""
        if module_id not in self.modules:
            return {
                'status': 'not_registered',
                'available': False,
                'error': f'Module {module_id} not registered'
            }
        
        module = self.modules[module_id]
        return {
            'status': module.status.value,
            'available': module.status == ModuleStatus.AVAILABLE,
            'display_name': module.display_name,
            'description': module.description,
            'version': module.version,
            'has_analysis': module.analysis_function is not None,
            'has_display': module.display_function is not None,
            'dependencies': module.dependencies,
            'error': module.error_message
        }
    
    def get_all_modules(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered modules"""
        return {
            module_id: self.get_module_status(module_id) 
            for module_id in self.modules.keys()
        }
    
    def initialize_session_state(self):
        """Initialize Streamlit session state for all modules"""
        if self._session_state_initialized:
            return
        
        for module_id, module in self.modules.items():
            session_key = f'show_{module_id}'
            if session_key not in st.session_state:
                # Default to enabled for available modules, disabled for unavailable
                st.session_state[session_key] = module.status == ModuleStatus.AVAILABLE
        
        self._session_state_initialized = True
    
    def create_module_toggles(self):
        """Create UI toggles for enabling/disabling modules"""
        st.write("**Analysis Modules:**")
        
        for module_id, module in self.modules.items():
            session_key = f'show_{module_id}'
            
            # Status indicator
            status_icon = {
                ModuleStatus.AVAILABLE: "✅",
                ModuleStatus.UNAVAILABLE: "⚠️", 
                ModuleStatus.ERROR: "❌",
                ModuleStatus.DISABLED: "🔘"
            }.get(module.status, "❓")
            
            # Create toggle with status
            label = f"{status_icon} {module.display_name}"
            if module.error_message:
                label += f" ({module.error_message[:30]}...)"
            
            # Only allow toggling if module is available
            if module.status == ModuleStatus.AVAILABLE:
                st.session_state[session_key] = st.checkbox(
                    label, 
                    value=st.session_state.get(session_key, True),
                    key=f"toggle_{module_id}",
                    help=module.description
                )
            else:
                st.checkbox(
                    label,
                    value=False,
                    disabled=True,
                    key=f"toggle_disabled_{module_id}",
                    help=f"{module.description} - {module.error_message or 'Not available'}"
                )

# Global module registry instance
_module_registry = ModuleRegistry()

def get_module_registry() -> ModuleRegistry:
    """Get the global module registry instance"""
    return _module_registry

# Convenience functions for registration
def register_analysis_module(module_id: str, 
                           display_name: str,
                           description: str, 
                           version: str,
                           analysis_function: Callable,
                           display_function: Optional[Callable] = None,
                           dependencies: List[str] = None) -> bool:
    """Register an analysis module"""
    return _module_registry.register_module(
        module_id, display_name, description, version,
        analysis_function, display_function, dependencies
    )

def safe_module_import(module_path: str, function_names: List[str]) -> Dict[str, Optional[Callable]]:
    """Safely import functions from a module"""
    functions = {}
    
    try:
        module = __import__(module_path, fromlist=function_names)
        for func_name in function_names:
            if hasattr(module, func_name):
                functions[func_name] = getattr(module, func_name)
            else:
                functions[func_name] = None
                logger.warning(f"Function {func_name} not found in {module_path}")
    except ImportError as e:
        logger.warning(f"Module {module_path} could not be imported: {e}")
        for func_name in function_names:
            functions[func_name] = None
    except Exception as e:
        logger.error(f"Error importing from {module_path}: {e}")
        for func_name in function_names:
            functions[func_name] = None
    
    return functions
