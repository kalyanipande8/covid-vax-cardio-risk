"""Feature engineering stubs.

Add real feature functions as you explore the datasets. Keep functions
small and testable.
"""
from typing import Any, Dict


def build_basic_features(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of the input row with example derived features.

    This is a placeholder demonstrating where to add feature logic.
    """
    out = dict(row)
    # Example: add a flag for missing age
    out["age_missing"] = row.get("age") is None
    return out
