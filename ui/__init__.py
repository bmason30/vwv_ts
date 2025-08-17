"""
UI components module for VWV Trading System v8.0.0
ENHANCED: Added Volume Score Bar export
"""

from .components import (
    create_technical_score_bar,
    create_volume_score_bar,  # NEW v8.0.0
    create_header
)

__all__ = [
    'create_technical_score_bar',
    'create_volume_score_bar',  # NEW v8.0.0
    'create_header'
]
