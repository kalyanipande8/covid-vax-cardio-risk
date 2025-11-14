"""VAERS preprocessing utilities.

Functions to load yearly VAERS CSV files, clean/standardize columns,
flag likely cardiac events, and write a processed dataset.

This module is defensive: if `pandas` is not installed it will raise
an informative error explaining how to install dependencies.
"""
from __future__ import annotations

import glob
import os
import re
from typing import List


try:
    import pandas as pd
except Exception as e:  # pragma: no cover - environment dependent
    raise ImportError(
        "pandas is required for VAERS preprocessing. Install with: `pip install pandas pyarrow`"
    ) from e


CARDIAC_KEYWORDS = [
    r"myocard", r"pericard", r"cardiomyopath", r"heart attack", r"mi\b", r"myopericard",
    r"arrhythm", r"atrial fibrillation", r"afib", r"tachycard", r"bradycard", r"cardiac",
    r"ischemi", r"infarct", r"troponin",
]


def list_vaers_files(data_dir: str = "data") -> List[str]:
    """Return a sorted list of VAERS CSV file paths under `data_dir`.

    Matches filenames like `*VAERSDATA.csv`.
    """
    pattern = os.path.join(data_dir, "*VAERSDATA.csv")
    files = sorted(glob.glob(pattern))
    return files


def _compile_cardiac_re():
    return re.compile("|".join(CARDIAC_KEYWORDS), flags=re.IGNORECASE)


def load_and_concat_vaers(paths: List[str]) -> pd.DataFrame:
    """Load multiple VAERS CSV files and concatenate into one DataFrame.

    Uses a conservative dtype inference and parses obvious date columns.
    """
    if not paths:
        raise ValueError("No VAERS files found to load")

    dfs = []
    parse_dates = ["RECVDATE", "VAX_DATE", "ONSET_DATE", "DATEDIED"]

    for p in paths:
        try:
            df = pd.read_csv(p, dtype=str, low_memory=False)
        except Exception:
            # try with latin-1 fallback
            df = pd.read_csv(p, dtype=str, low_memory=False, encoding="latin-1")

        # ensure consistent columns
        df.columns = [c.strip() for c in df.columns]

        # keep original filename for traceability
        df["_source_file"] = os.path.basename(p)

        # attempt to parse date-like columns if present
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=False)

        # numeric conversion where sensible
        if "AGE_YRS" in df.columns:
            df["AGE_YRS"] = pd.to_numeric(df["AGE_YRS"], errors="coerce")

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True, sort=False)
    return out


def flag_cardiac_events(df: pd.DataFrame, text_field: str = "SYMPTOM_TEXT") -> pd.Series:
    """Return a boolean Series marking rows likely related to cardiac events.

    This is a simple keyword match on the provided `text_field`. It's intentionally
    permissive to aid cohort discovery; downstream manual review is required.
    """
    cre = _compile_cardiac_re()
    if text_field not in df.columns:
        return pd.Series(False, index=df.index)

    txt = df[text_field].fillna("").astype(str)
    return txt.str.contains(cre)


def derive_seriousness(df: pd.DataFrame) -> pd.Series:
    """Create a boolean `serious` Series based on common VAERS markers.

    Marks True when `DIED` == 'Y' or hospitalization/ER/ life-threatening flags are set.
    """
    cols = {"DIED": "Y", "HOSPITAL": "Y", "ER_VISIT": "Y", "L_THREAT": "Y"}
    ser = pd.Series(False, index=df.index)
    for c, val in cols.items():
        if c in df.columns:
            ser = ser | (df[c].fillna("").astype(str).str.upper() == val)
    return ser


def preprocess_vaers(df: pd.DataFrame) -> pd.DataFrame:
    """Run standard cleaning and add derived columns useful for cohorting.

    Adds:
      - `cardiac_flag`: permissive keyword match in `SYMPTOM_TEXT`
      - `serious`: boolean per `derive_seriousness`
      - normalizes `SEX` to {M,F,U}
    """
    out = df.copy()

    # normalize SEX
    if "SEX" in out.columns:
        out["SEX"] = out["SEX"].fillna("U").astype(str).str.upper().str.strip()
        out.loc[out["SEX"].isin(["M", "MALE"]) == False, "SEX"] = out.loc[out["SEX"].isin(["M", "MALE"]) == False, "SEX"].replace({"FEMALE": "F"})
        out["SEX"] = out["SEX"].map({"M": "M", "F": "F"}).fillna("U")

    # cardiac flag
    out["cardiac_flag"] = flag_cardiac_events(out, text_field="SYMPTOM_TEXT")

    # serious flag
    out["serious"] = derive_seriousness(out)

    # short summary column for quick review
    if "SYMPTOM_TEXT" in out.columns:
        out["_symptom_snippet"] = out["SYMPTOM_TEXT"].fillna("").astype(str).str.slice(0, 200)

    return out


def write_processed(df: pd.DataFrame, out_path: str) -> None:
    """Write processed DataFrame to `out_path`.

    If the path ends with `.parquet` the function will attempt to use
    parquet (pyarrow); otherwise it will write a gzipped CSV.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if out_path.endswith(".parquet"):
        try:
            df.to_parquet(out_path, index=False)
            return
        except Exception:
            # fallback to CSV
            out_path = out_path.replace(".parquet", ".csv.gz")

    # CSV gzip fallback
    df.to_csv(out_path, index=False, compression="gzip")


def run_pipeline(data_dir: str = "data", out_path: str = "data/processed/vaers_combined.parquet") -> str:
    """High-level helper to run the VAERS preprocessing pipeline.

    Returns the path written.
    """
    files = list_vaers_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No VAERS files found in {data_dir}")

    print(f"Found {len(files)} VAERS files. Loading...")
    df = load_and_concat_vaers(files)
    print(f"Loaded {len(df):,} rows. Running preprocessing...")
    df2 = preprocess_vaers(df)
    print(f"Writing processed data to {out_path}")
    write_processed(df2, out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Preprocess VAERS CSV files into a combined dataset")
    p.add_argument("--data-dir", default="data", help="Directory containing VAERS CSV files")
    p.add_argument("--out", default="data/processed/vaers_combined.parquet", help="Output path (parquet preferred)")
    args = p.parse_args()

    try:
        out = run_pipeline(data_dir=args.data_dir, out_path=args.out)
        print("Done. Wrote:", out)
    except Exception as e:
        print("VAERS preprocessing failed:", e)
        raise
