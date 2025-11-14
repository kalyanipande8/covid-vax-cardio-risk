"""VAERS cohort extraction and vaccine mapping helpers.

Provides functions to normalize vaccine manufacturer information and
to extract a cardiac-focused cohort from a VAERS DataFrame produced by
`vaers_preprocess.run_pipeline`.
"""
from __future__ import annotations

import re
import os
from typing import Optional

import pandas as pd


# heuristic mapping rules (lowercase keys -> normalized manufacturer)
_MANU_PATTERNS = {
    r"moderna|mRNA-1273|spikevax|mRNA": "Moderna",
    r"pfizer|bnt162b2|comirnaty|tozinameran": "Pfizer/BioNTech",
    r"janssen|johnson|janssen": "Janssen/Johnson & Johnson",
    r"astra|vaxzevria|oxford": "AstraZeneca",
    r"novavax|nvx-cov|nuvaxovid|nvx": "Novavax",
    r"gsk|arexvy|shingrix": "GSK",
    r"pfizer.*covid": "Pfizer/BioNTech",
    r"sanofi|fluzone|flulaval|influenza": "Sanofi/Other Flu",
    r"merck|gardasil|varivax": "Merck",
}


def map_vaccine_manufacturer(df: pd.DataFrame, source_cols: Optional[list] = None) -> pd.DataFrame:
    """Add a `vax_manufacturer` column inferred from common VAERS fields.

    The function checks the provided `source_cols` (defaults commonly used
    VAERS columns) and uses regex heuristics to normalize manufacturer names.
    """
    src = source_cols or ["VAX_MANU", "VAX_NAME", "VAX_TYPE", "VAX_PRODUCT", "SYMPTOM_TEXT"]

    # create a combined text field to search
    found = pd.Series("", index=df.index)
    for c in src:
        if c in df.columns:
            found = found.str.cat(df[c].fillna("").astype(str), sep=" ")

    found = found.str.lower().fillna("")

    def _match(s: str) -> str:
        for pat, label in _MANU_PATTERNS.items():
            if re.search(pat, s):
                return label
        return "Unknown"

    df = df.copy()
    df["vax_manufacturer"] = found.apply(_match)
    return df


def extract_cardiac_cohort(df: pd.DataFrame, include_serious_only: bool = False) -> pd.DataFrame:
    """Return a DataFrame containing a cardiac-focused cohort.

    Selection logic (permissive):
      - `cardiac_flag` == True (from preprocessing keyword match), OR
      - `SYMPTOM_TEXT` contains cardiac keywords (redundant but defensive), OR
      - if `include_serious_only` then require `serious`==True as well

    The function returns a copy with a `_reason` column describing why
    each row was included.
    """
    df = df.copy()

    cardiac_patterns = [
        r"myocard", r"pericard", r"cardiomyopath", r"heart attack", r"\bmi\b",
        r"myopericard", r"atrial fibrillation", r"afib", r"arrhythm", r"tachycard",
        r"bradycard", r"ischemi", r"infarct", r"troponin", r"cardiac",
    ]
    cre = re.compile("|".join(cardiac_patterns), flags=re.IGNORECASE)

    reasons = []
    mask = pd.Series(False, index=df.index)

    # use precomputed flag if available
    if "cardiac_flag" in df.columns:
        m1 = df["cardiac_flag"].fillna(False).astype(bool)
        mask = mask | m1
        reasons.append((m1, "cardiac_flag"))

    # search symptom text
    if "SYMPTOM_TEXT" in df.columns:
        txt = df["SYMPTOM_TEXT"].fillna("").astype(str)
        m2 = txt.str.contains(cre)
        mask = mask | m2
        reasons.append((m2, "symptom_text_match"))

    # optional: require serious
    if include_serious_only and "serious" in df.columns:
        mask = mask & df["serious"].fillna(False).astype(bool)

    cohort = df[mask].copy()

    # compute a reason column
    reason_col = []
    for idx in cohort.index:
        rs = []
        for m, label in reasons:
            if m.loc[idx]:
                rs.append(label)
        reason_col.append(";".join(rs) if rs else "match")

    cohort["_reason"] = reason_col
    return cohort


def save_cohort(df: pd.DataFrame, out_path: str) -> str:
    """Persist cohort dataframe to disk (parquet preferred). Returns path written."""
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    try:
        df.to_parquet(out_path, index=False)
        return out_path
    except Exception:
        out_path = out_path.replace(".parquet", ".csv.gz")
        df.to_csv(out_path, index=False, compression="gzip")
        return out_path


def filter_covid_vaccines(df: pd.DataFrame, allowed: Optional[list] = None) -> pd.DataFrame:
    """Return a filtered DataFrame containing only rows with COVID vaccine manufacturers.

    By default, this includes common COVID vaccine manufacturers recognized
    by the mapping heuristics (Moderna, Pfizer/BioNTech, Janssen/J&J,
    AstraZeneca, Novavax). Pass an `allowed` list to override.
    """
    if allowed is None:
        allowed = ["Moderna", "Pfizer/BioNTech", "Janssen/Johnson & Johnson", "AstraZeneca", "Novavax"]
    if "vax_manufacturer" not in df.columns:
        # no manufacturer information: return empty frame to be explicit
        return df.iloc[0:0]
    return df[df["vax_manufacturer"].isin(allowed)].copy()


def label_vaccine_related_cardiac(df: pd.DataFrame, window_days: int = 28, require_serious: bool = True) -> pd.Series:
    """Create a boolean label for vaccine-related cardiac complications.

    Logic (default):
      - `cardiac_flag` is True (keyword/symptom-based)
      - `VAX_DATE` and `ONSET_DATE` are present and 0 <= onset - vax <= window_days
            - by default require `serious` == True so the endpoint focuses on
                hospitalization/death/life-threatening events unless overridden

    Returns a boolean Series aligned to `df`.
    """
    # ensure date columns exist
    out = pd.Series(False, index=df.index)
    if not ("VAX_DATE" in df.columns and "ONSET_DATE" in df.columns):
        return out

    vax = pd.to_datetime(df["VAX_DATE"], errors="coerce")
    onset = pd.to_datetime(df["ONSET_DATE"], errors="coerce")
    # compute latency in days
    latency = (onset - vax).dt.days

    has_card = df.get("cardiac_flag", pd.Series(False, index=df.index)).fillna(False).astype(bool)

    within_window = latency.notna() & (latency >= 0) & (latency <= int(window_days))

    label = has_card & within_window
    if require_serious and "serious" in df.columns:
        label = label & df["serious"].fillna(False).astype(bool)

    out.loc[label.index] = label
    return out
