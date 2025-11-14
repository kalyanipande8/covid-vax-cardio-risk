"""Inspect `data/aggregated.csv.gz` and print schema and sample rows.

This script uses pandas (lazy import). Run from the project root. It is
intended to give a quick overview so we can design preprocessing steps.
"""
from __future__ import annotations
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data" / "aggregated.csv.gz"


def main() -> None:
    # Prefer pandas if available for a richer inspection. If pandas is not
    # installed in the environment, fall back to reading the compressed file
    # with the standard library to at least show headers and a few rows.
    if not DATA_PATH.exists():
        print(f"File not found: {DATA_PATH}")
        return

    try:
        import pandas as pd

        # Attempt to read a small sample (pandas autodetects compression by suffix)
        df = pd.read_csv(DATA_PATH, compression='gzip', nrows=20)

        print("--- Columns ---")
        for c in df.columns:
            print(c)

        print("\n--- Dtypes ---")
        print(df.dtypes)

        print("\n--- Head ---")
        print(df.head(10).to_string(index=False))

    except Exception:
        # Fallback: use gzip + csv to show header and first few rows
        import gzip
        import csv

        print("pandas is not available; using gzip+csv fallback to inspect file")
        with gzip.open(DATA_PATH, mode="rt", encoding="utf-8", errors="replace") as fh:
            reader = csv.reader(fh)
            try:
                header = next(reader)
            except StopIteration:
                print("file is empty")
                return

            print("--- Columns (inferred from header) ---")
            for c in header:
                print(c)

            print("\n--- Sample rows ---")
            for i, row in enumerate(reader):
                print(row)
                if i >= 9:
                    break


if __name__ == "__main__":
    main()
