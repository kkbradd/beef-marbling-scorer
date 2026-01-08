"""
High-level rule utilities used during inference.

This module provides a unified import surface so callers can do:

    from src.features.rules import (
        choose_marbling,
        marbling_to_usda,
        assign_bms,
        assign_aus_meat,
    )
"""

from .marbling_rules import choose_marbling, marbling_to_usda
from .bms_rules import assign_bms
from .aus_rules import assign_aus_meat

__all__ = ["choose_marbling", "marbling_to_usda", "assign_bms", "assign_aus_meat"]


