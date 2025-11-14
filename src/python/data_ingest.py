"""Helpers for loading and inspecting datasets kept in `data/`.

This module contains lightweight, lazy-loading helpers so the rest of the
project can import it without forcing heavy dependencies at import time.
"""
from pathlib import Path
from typing import List, Dict, Optional


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def list_data_files(subdir: Optional[str] = "raw") -> List[Path]:
    """List files inside `data/<subdir>` (default: `data/raw`).

    Returns a list of Path objects sorted by name.
    """
    folder = DATA_DIR / subdir
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file()])


def load_csv(path: str, **kwargs):
    """Load a CSV with pandas (lazy import).

    This raises a RuntimeError with a helpful message if pandas isn't
    available in the environment.
    """
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas is required to load CSV files") from e

    return pd.read_csv(path, **kwargs)


def inspect_csv(path: str, nrows: int = 5) -> Dict[str, object]:
    """Return a small inspection summary (head, dtypes, shape).

    The function uses lazy import of pandas to avoid hard dependency at import
    time.
    """
    df = load_csv(path, nrows=nrows)
    return {"head": df.head(nrows), "dtypes": df.dtypes.to_dict(), "shape": df.shape}


def load_aggregated(path: Optional[str] = None, parse_dates: bool = True, nrows: Optional[int] = None):
    """Load the project `aggregated.csv.gz` (country-level aggregated dataset).

    The function attempts to be robust:
      - automatically detect gzip compression by filename
      - parse a `date`-like column to datetime
      - coerce numeric columns and strip string columns

    Returns a pandas.DataFrame. Raises a RuntimeError if pandas isn't
    available.
    """
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas is required to load aggregated data") from e

    DATA_PATH = Path(path) if path else DATA_DIR / "aggregated.csv.gz"
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Aggregated file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, compression="gzip", nrows=nrows)

    # common cleanups
    # 1) parse date-like column
    date_cols = [c for c in df.columns if c.lower() in ("date", "day", "report_date")]
    if parse_dates and date_cols:
        for c in date_cols:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                # leave as-is if parsing fails
                pass

    # 2) coerce numeric columns where possible
    obj_cols = df.select_dtypes(include=[object]).columns.tolist()
    for c in obj_cols:
        # skip columns likely to be non-numeric identifiers (contain 'code' or 'id' or 'name')
        low = c.lower()
        if any(x in low for x in ("code", "id", "name", "iso", "uid", "slug")):
            continue
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            # leave as object if coercion fails
            pass

    # 3) strip whitespace in string columns
    str_cols = df.select_dtypes(include=[object]).columns
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip().replace({"": pd.NA})

    return df
"""Data ingestion helpers for CSV/Parquet sources with light validation.

Functions here intentionally perform lazy imports so the module can be
imported even if pandas is not installed. Use these helpers to standardize
loading logic across Python workflows.
"""
from typing import Optional, Dict


def load_tabular(path: str, engine: Optional[str] = None, **read_kwargs):
    """Load a CSV or Parquet file into a pandas.DataFrame.

    Args:
        path: Path to the file (supports .csv, .parquet, .feather)
        engine: Optional engine to pass to pandas reader (e.g., 'pyarrow')
        **read_kwargs: Extra kwargs forwarded to pandas reader

    Returns:
        pandas.DataFrame

    Raises:
        RuntimeError: if pandas (or pyarrow for parquet) is not available.
    """
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas is required to load tabular data") from e

    path_lower = path.lower()
    if path_lower.endswith(".csv"):
        return pd.read_csv(path, **read_kwargs)
    if path_lower.endswith(".parquet"):
        return pd.read_parquet(path, engine=engine, **read_kwargs)
    if path_lower.endswith(".feather"):
        return pd.read_feather(path, **read_kwargs)

    raise ValueError("Unsupported file extension for path: {}".format(path))


def basic_profile(df) -> Dict[str, int]:
    """Return a tiny profile (rows, columns, missing counts).

    Designed for quick CLI summaries and smoke checks.
    """
    try:
        # local import to avoid hard dependency at import time
        import pandas as pd  # noqa: F401
    except Exception:
        pass

    profile = {
        "n_rows": int(getattr(df, "shape", (0, 0))[0]),
        "n_cols": int(getattr(df, "shape", (0, 0))[1]),
        "n_missing_total": int(df.isnull().sum().sum()) if hasattr(df, "isnull") else 0,
    }
    return profile
