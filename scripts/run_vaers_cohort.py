"""Small runner script to preprocess VAERS, map vaccines, and extract cardiac cohort.

Usage:
    python scripts/run_vaers_cohort.py --data-dir data --out-dir data/processed

This script attempts to use parquet outputs when possible and falls back
to gzipped CSV when necessary.
"""
from __future__ import annotations

import argparse
import os


def main(data_dir: str, out_dir: str, include_serious_only: bool):
    # Ensure project root is on sys.path so `src.python` imports resolve
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # local imports to keep script lightweight for users who only inspect
    from src.python.vaers_preprocess import run_pipeline
    from src.python.vaers_cohort import map_vaccine_manufacturer, extract_cardiac_cohort, save_cohort

    # 1) run preprocessing
    processed_path = os.path.join(out_dir, "vaers_combined.parquet")
    try:
        print("Running VAERS preprocessing pipeline...")
        run_pipeline(data_dir=data_dir, out_path=processed_path)
    except Exception as e:
        # If parquet write failed, run_pipeline will have attempted CSV fallback
        print("Preprocessing pipeline error (continuing if fallback wrote CSV):", e)

    # 2) load whichever processed file exists
    import glob
    import pandas as pd

    candidates = [processed_path, processed_path.replace(".parquet", ".csv.gz")]
    proc_file = None
    for c in candidates:
        if os.path.exists(c):
            proc_file = c
            break

    if not proc_file:
        raise FileNotFoundError("No processed VAERS file found. Please run preprocessing or check data/processed.")

    print(f"Loading processed VAERS from: {proc_file}")
    df = pd.read_parquet(proc_file) if proc_file.endswith(".parquet") else pd.read_csv(proc_file, compression="gzip")

    # 3) map vaccine manufacturers
    print("Mapping vaccine manufacturers...")
    df = map_vaccine_manufacturer(df)

    # save mapped full table (all vaccines)
    mapped_path = os.path.join(out_dir, "vaers_mapped.parquet")
    try:
        df.to_parquet(mapped_path, index=False)
        print("Wrote mapped full table to", mapped_path)
    except Exception:
        mapped_path = mapped_path.replace(".parquet", ".csv.gz")
        df.to_csv(mapped_path, index=False, compression="gzip")
        print("Wrote mapped full table to", mapped_path)

    # also produce a COVID-only mapped file and use that for cohort extraction
    try:
        covid_df = filter_covid_vaccines(df)
    except Exception:
        covid_df = df.iloc[0:0]

    mapped_covid_path = os.path.join(out_dir, "vaers_mapped_covid.parquet")
    try:
        covid_df.to_parquet(mapped_covid_path, index=False)
        print("Wrote COVID-only mapped table to", mapped_covid_path)
    except Exception:
        mapped_covid_path = mapped_covid_path.replace(".parquet", ".csv.gz")
        covid_df.to_csv(mapped_covid_path, index=False, compression="gzip")
        print("Wrote COVID-only mapped table to", mapped_covid_path)

    # 4) extract cardiac cohort from COVID-only mapped data
    print("Extracting cardiac cohort (COVID vaccines only)...")
    cohort = extract_cardiac_cohort(covid_df, include_serious_only=include_serious_only)

    cohort_path = os.path.join(out_dir, "vaers_cardiac.parquet")
    outp = save_cohort(cohort, cohort_path)
    print("Wrote cardiac cohort to", outp)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data", help="Directory with VAERS CSV files")
    p.add_argument("--out-dir", default="data/processed", help="Directory to write processed outputs")
    p.add_argument("--serious-only", action="store_true", help="Restrict cohort to serious reports only")
    args = p.parse_args()
    main(args.data_dir, args.out_dir, args.serious_only)
