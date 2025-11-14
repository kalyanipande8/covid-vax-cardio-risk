"""Starter data loading utilities.

This file avoids importing heavy dependencies at module import time. Each
function performs lazy imports so the module can be imported in a bare
Python environment for testing.
"""
from typing import Optional


def example() -> str:
    """A trivial function used by the smoke test.

    Returns:
        str: a small message confirming the module loads correctly.
    """
    return "smoke test OK"


def load_csv_lazy(path: str, delimiter: str = ","):
    """Load a CSV file using pandas (lazy import).

    This function imports pandas only when called, so importing this module
    doesn't require pandas to be installed.
    """
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas is required to use load_csv_lazy: ") from e

    return pd.read_csv(path, sep=delimiter)
