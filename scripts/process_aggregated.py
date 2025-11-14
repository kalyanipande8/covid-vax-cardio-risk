"""Load `data/aggregated.csv.gz`, run light preprocessing, and write a processed parquet.

Creates `data/processed/aggregated.parquet` (gzip-compressed parquet) so downstream
steps can load faster.
"""
from __future__ import annotations
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.python.data_ingest import load_aggregated  # type: ignore


def main() -> None:
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aggregated.parquet"

    try:
        df = load_aggregated()

        # lightweight additional preprocessing: drop columns that are entirely NA
        df = df.dropna(axis=1, how="all")

        # write as parquet (fast re-load for analysis). If pandas or pyarrow
        # isn't available, fall back to CSV.
        try:
            df.to_parquet(out_path, compression="gzip")
            print(f"Wrote processed file to {out_path}")
        except Exception as e:
            csv_out = out_dir / "aggregated_processed.csv"
            df.to_csv(csv_out, index=False)
            print(f"pyarrow not available or parquet write failed; wrote CSV to {csv_out}: {e}")

    except Exception as e:
        # fallback: we couldn't load/process (likely pandas missing). Copy the
        # original compressed file into processed/ so downstream steps have a
        # stable location for the dataset and the user can install dependencies
        # and re-run processing.
        import shutil

        src = PROJECT_ROOT / "data" / "aggregated.csv.gz"
        dst = out_dir / "aggregated_raw_copy.csv.gz"
        try:
            shutil.copy(src, dst)
            print(f"Could not process (pandas missing). Copied original to {dst}")
        except Exception as e2:
            print(f"Failed to copy original file as fallback: {e2}")
            raise


if __name__ == "__main__":
    main()
