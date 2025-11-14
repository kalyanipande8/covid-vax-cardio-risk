"""Minimal python package for covid-vax-cardio-risk.

Keep top-level package small; heavy imports are performed lazily inside functions
to avoid forcing environment dependencies at import time.
"""

__all__ = ["data_loader", "features", "models"]
__version__ = "0.0.1"
