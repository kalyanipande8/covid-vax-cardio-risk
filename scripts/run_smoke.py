"""Run a quick smoke test for the starter package.

This script adds the project root to `sys.path` so it can be executed directly
from the repository root without requiring a packaging/install step.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work when running this
# script directly (e.g. `python scripts/run_smoke.py`).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.python import data_loader  # noqa: E402


def main() -> None:
    print(data_loader.example())


if __name__ == "__main__":
    main()
