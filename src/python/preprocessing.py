"""Configurable preprocessing pipeline for tabular data.

Start here and extend with domain-specific steps (vaccination timing,
comorbidity flags, date parsing, etc.). Keep functions deterministic and easy
to unit-test.
"""
from typing import Optional, List


class Preprocessor:
    """Simple preprocessor wrapper.

    Usage:
        p = Preprocessor(fill_method="median")
        df_clean = p.fit_transform(df)
    """

    def __init__(self, fill_method: str = "median", drop_threshold: float = 0.9):
        self.fill_method = fill_method
        self.drop_threshold = drop_threshold

    def fit(self, df):
        # compute any statistics needed for transform (e.g., medians)
        try:
            import pandas as pd  # lazy
        except Exception:
            raise RuntimeError("pandas is required for preprocessing")

        self._medians = df.median(numeric_only=True).to_dict()
        return self

    def transform(self, df):
        # perform cleaning steps
        try:
            import pandas as pd  # lazy
        except Exception:
            raise RuntimeError("pandas is required for preprocessing")

        out = df.copy()
        # drop columns with too many missing values
        thresh = int((1.0 - self.drop_threshold) * len(out))
        out = out.dropna(axis=1, thresh=thresh)

        # fill numeric columns using medians computed during fit
        for col, val in getattr(self, "_medians", {}).items():
            if col in out.columns:
                out[col] = out[col].fillna(val)

        # example: convert boolean-like columns
        return out

    def fit_transform(self, df):
        return self.fit(df).transform(df)


def basic_clean(df, drop_cols: Optional[List[str]] = None):
    """Convenience function performing common quick-clean operations.

    - drop specified columns
    - strip whitespace in string columns
    - convert empty strings to NA
    """
    try:
        import pandas as pd
        import numpy as np
    except Exception:
        raise RuntimeError("pandas and numpy are required for basic_clean")

    out = df.copy()
    if drop_cols:
        out = out.drop(columns=[c for c in drop_cols if c in out.columns])

    # strip whitespace for object columns
    obj_cols = out.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        out[c] = out[c].astype(str).str.strip().replace({"": pd.NA})

    return out
"""Configurable preprocessing pipeline utilities.

This module contains generic, dataset-agnostic functions that perform common
preprocessing tasks: date parsing, missing-value summaries, simple imputations,
categorical encoding, and scaling. Each function uses lazy imports so the
module can be imported without immediately installing heavy dependencies.
"""
from typing import List, Optional, Dict, Any


def parse_dates(df, columns: List[str]) -> "pandas.DataFrame":
    """Parse date-like columns to pandas datetime dtype (in-place).

    Args:
        df: DataFrame-like with __setitem__ semantics
        columns: list of column names to parse
    Returns:
        The same DataFrame object with parsed date columns.
    """
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas is required for parse_dates") from e

    for c in columns:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def summarize_missing(df) -> Dict[str, Any]:
    """Return column-wise missing value summary.

    Returns a dict with counts and fractions per column.
    """
    try:
        import pandas as pd  # noqa: F401
    except Exception:
        pass

    out = {}
    n = int(getattr(df, "shape", (0, 0))[0])
    for col in getattr(df, "columns", []):
        missing = int(df[col].isnull().sum()) if hasattr(df[col], "isnull") else 0
        out[col] = {"missing_count": missing, "missing_fraction": missing / n if n else None}
    return out


def impute_numeric(df, columns: List[str], strategy: str = "median"):
    """Impute numeric columns in-place.

    Supported strategies: 'median', 'mean', 'zero'.
    """
    try:
        import pandas as pd  # noqa: F401
    except Exception:
        pass

    for c in columns:
        if c not in df.columns:
            continue
        if strategy == "median":
            value = df[c].median()
        elif strategy == "mean":
            value = df[c].mean()
        elif strategy == "zero":
            value = 0
        else:
            raise ValueError("Unknown imputation strategy: {}".format(strategy))
        df[c] = df[c].fillna(value)
    return df


def encode_categorical(df, columns: List[str], drop_first: bool = True):
    """One-hot encode categorical columns using pandas.get_dummies (in-place replacement).

    Note: For large cardinality columns consider using target encoding or hashing.
    """
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas is required for encode_categorical") from e

    df = pd.get_dummies(df, columns=[c for c in columns if c in df.columns], drop_first=drop_first)
    return df


def scale_numeric(df, columns: List[str]):
    """Scale numeric columns using sklearn's StandardScaler.

    Returns: (df, scaler) — df is a copy with scaled columns, scaler is fitted object.
    """
    try:
        import pandas as pd  # noqa: F401
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise RuntimeError("scikit-learn is required for scale_numeric") from e

    scaler = StandardScaler()
    present = [c for c in columns if c in df.columns]
    if present:
        df_loc = df.copy()
        df_loc[present] = scaler.fit_transform(df_loc[present])
        return df_loc, scaler
    return df, None
