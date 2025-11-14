"""Train a small model on the VAERS cardiac cohort and save SHAP plots.

This script samples the processed `vaers_mapped` dataset to a manageable
size, trains a RandomForestClassifier on `cardiac_flag`, and writes two
SHAP summary plots to `notebooks/shap_summary.png` and `notebooks/shap_bar.png`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ensure repo root on path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

try:
    import shap
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover - runtime dep
    raise ImportError("This script requires `shap` and `matplotlib`. Install: pip install shap matplotlib") from e


def build_features(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    X = pd.DataFrame(index=df.index)
    # Age
    if 'AGE_YRS' in df.columns:
        X['age_yrs'] = pd.to_numeric(df['AGE_YRS'], errors='coerce').fillna(-1)
        X['age_bin'] = pd.cut(X['age_yrs'], bins=[-1,5,18,40,65,120], labels=['infant','child','adult','mid','senior'])
        X = pd.concat([X, pd.get_dummies(X['age_bin'], prefix='age')], axis=1)
    else:
        X['age_unknown'] = 1

    # Sex
    if 'SEX' in df.columns:
        X['is_male'] = (df['SEX'].fillna('U').str.upper() == 'M').astype(int)
    else:
        X['is_male'] = 0

    # Vaccine manufacturer one-hot (top N)
    if 'vax_manufacturer' in df.columns:
        vman = df['vax_manufacturer'].fillna('Unknown')
        vm_dummies = pd.get_dummies(vman, prefix='vax')
        top = vm_dummies.sum().sort_values(ascending=False).head(8).index.tolist()
        X = pd.concat([X, vm_dummies[top]], axis=1)
    else:
        X['vax_unknown'] = 1

    # Parse common comorbidities from free-text fields into binary flags.
    def parse_comorbidities(df):
        parts = []
        for c in ['PHM', 'HISTORY', 'CUR_ILL', 'OTHER_MEDS']:
            if c in df.columns:
                parts.append(df[c].fillna('').astype(str))
        if not parts:
            return pd.DataFrame(index=df.index)
        text = pd.Series('', index=df.index)
        for s in parts:
            text = text.str.cat(s, sep=' ')
        text = text.str.lower()
        flags = pd.DataFrame(index=df.index)
        flags['hx_cad'] = text.str.contains(r'\b(cad|coronary|angina|stent|myocardial infarct|\bmi\b|coronary artery)\b', regex=True)
        flags['hx_af'] = text.str.contains(r'atrial fibrill|afib|\baf\b', regex=True)
        flags['hx_hf'] = text.str.contains(r'heart failure|cardiomyopath|cardiomyopathy|\bhf\b', regex=True)
        flags['hx_htn'] = text.str.contains(r'hypertension|high blood pressure|\bhtn\b', regex=True)
        flags['hx_dm'] = text.str.contains(r'diabet|\bdm\b|type 2', regex=True)
        flags['hx_hyperlip'] = text.str.contains(r'hyperlipid|cholesterol|\bhld\b|ldl|triglycerid', regex=True)
        return flags.astype(int)

    com = parse_comorbidities(df)
    if not com.empty:
        X = pd.concat([X, com], axis=1)

    # target (placeholder) - caller will override if using refined label
    if 'cardiac_flag' in df.columns:
        y = df['cardiac_flag'].fillna(False).astype(int)
    else:
        # don't raise here; caller sometimes overrides with refined label
        y = pd.Series(0, index=df.index)

    # keep only numeric columns
    X = X.select_dtypes(include=[np.number]).fillna(0)
    return X, y


def sample_for_training(df: pd.DataFrame, label_col: str = 'cardiac_flag', max_samples: int = 40000, min_pos: int = 2000):
    # ensure there are positives
    y = df[label_col].fillna(False).astype(int)
    pos_idx = df[y == 1].index
    neg_idx = df[y == 0].index

    n_pos = len(pos_idx)
    n_pos_use = min(n_pos, min_pos)
    if n_pos_use == 0:
        raise ValueError('No positive cardiac_flag examples found to train on')

    # choose up to min_pos positives (or all if fewer), then sample negatives to reach max_samples
    pos_sample = np.random.choice(pos_idx, size=n_pos_use, replace=False)
    n_neg = max_samples - n_pos_use
    n_neg = max(1000, n_neg)
    n_neg = min(n_neg, len(neg_idx))
    neg_sample = np.random.choice(neg_idx, size=n_neg, replace=False)

    idx = np.concatenate([pos_sample, neg_sample])
    return df.loc[idx]


def main():
    # prefer COVID-only mapped file when available
    processed = Path('data/processed/vaers_mapped_covid.parquet')
    if not processed.exists():
        processed = Path('data/processed/vaers_mapped_covid.csv.gz')
    if not processed.exists():
        # fallback to full mapped table if COVID-only not present
        processed = Path('data/processed/vaers_mapped.parquet')
    if not processed.exists():
        processed = Path('data/processed/vaers_mapped.csv.gz')
    if not processed.exists():
        raise FileNotFoundError('Processed mapped file not found; run scripts/run_vaers_cohort.py first')

    print('Loading processed mapped VAERS...')
    df = pd.read_parquet(processed) if processed.suffix == '.parquet' else pd.read_csv(processed, compression='gzip')
    print(f'Loaded {len(df):,} rows')
    # label vaccine-related cardiac events (default 28-day window)
    from src.python.vaers_cohort import label_vaccine_related_cardiac
    # use a 28-day window and require serious events by default
    df['vaccine_related_cardiac'] = label_vaccine_related_cardiac(df, window_days=28, require_serious=True)

    # sample down for training speed and balance using the refined label
    df_sample = sample_for_training(df, label_col='vaccine_related_cardiac', max_samples=40000, min_pos=2000)
    print(f"Sampled {len(df_sample):,} rows for training (positives: {int(df_sample['vaccine_related_cardiac'].sum())})")

    X, y = build_features(df_sample)
    # override y to the refined label
    y = df_sample['vaccine_related_cardiac'].fillna(False).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print('Training RandomForestClassifier...')
    m = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    m.fit(X_train, y_train)

    models_dir = Path('models')
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / 'rf_vaers_cardiac.joblib'
    joblib.dump(m, model_path)
    print('Saved model to', model_path)

    # SHAP
    print('Computing SHAP values (TreeExplainer)...')
    explainer = shap.TreeExplainer(m)
    # use a small subset of test for SHAP speed
    Xshap = X_test.sample(min(2000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(Xshap)

    # defensive: SHAP can return different shapes/containers depending on version
    n_samples, n_features = Xshap.shape[0], Xshap.shape[1]
    arr = None

    # Debug-print shapes for tracing in case of mismatch
    try:
        if isinstance(shap_values, list):
            shapes = [getattr(a, 'shape', None) for a in shap_values]
            print('shap_values is list with shapes:', shapes)
            # find any array candidate matching sample dimension
            for a in shap_values:
                if getattr(a, 'ndim', 0) == 2:
                    if a.shape[0] == n_samples:
                        arr = a
                        break
                    if a.shape[1] == n_samples:
                        arr = a.T
                        break
            # last fallback: if first element looks like feature axis, try transpose
            if arr is None and len(shap_values) > 0 and getattr(shap_values[0], 'ndim', 0) == 2:
                a0 = shap_values[0]
                if a0.shape[0] == n_features and a0.shape[1] == n_samples:
                    arr = a0.T
        else:
                print('shap_values is ndarray with shape:', getattr(shap_values, 'shape', None))
                a = shap_values
                if getattr(a, 'ndim', 0) == 2:
                    # common 2D: (n_samples, n_features) or transposed
                    if a.shape[0] == n_samples and a.shape[1] == n_features:
                        arr = a
                    elif a.shape[1] == n_samples and a.shape[0] == n_features:
                        arr = a.T
                    else:
                        # best-effort: prefer rows==samples
                        if a.shape[0] == n_samples:
                            arr = a
                        elif a.shape[1] == n_samples:
                            arr = a.T
                elif getattr(a, 'ndim', 0) == 3:
                    # handle various 3D orders robustly by matching axes to samples/features
                    d0, d1, d2 = a.shape
                    # case: (n_samples, n_features, n_classes)
                    if d0 == n_samples and d1 == n_features:
                        cls_idx = 1 if d2 > 1 else 0
                        arr = a[:, :, cls_idx]
                    # case: (n_samples, n_classes, n_features)
                    elif d0 == n_samples and d2 == n_features:
                        cls_idx = 1 if d1 > 1 else 0
                        arr = a[:, cls_idx, :]
                    # case: (n_classes, n_samples, n_features) or similar
                    elif d1 == n_samples and d2 == n_features:
                        cls_idx = 0 if d0 <= 1 else 1
                        arr = a[cls_idx, :, :].T
                    else:
                        # fallback attempts: try to find axis matching n_features
                        if d2 == n_features:
                            arr = a[:, :, 1] if d1 > 1 else a[:, :, 0]
                        elif d1 == n_features:
                            arr = a[:, 1, :] if d2 > 1 else a[:, 0, :]
                        else:
                            arr = None
    except Exception as _e:
        print('Error while inspecting SHAP outputs:', _e)

    if arr is None:
        raise ValueError(f'Could not find SHAP array matching samples {n_samples} (observed shapes: {[getattr(a, "shape", None) for a in (shap_values if isinstance(shap_values, list) else [shap_values])]})')

    out_dir = Path('notebooks')
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_png = out_dir / 'shap_summary.png'
    bar_png = out_dir / 'shap_bar.png'

    print('Saving SHAP summary plot to', summary_png)
    plt.figure(figsize=(8,6))
    shap.summary_plot(arr, Xshap, show=False)
    plt.tight_layout()
    plt.savefig(summary_png, dpi=150)
    plt.close()

    print('Saving SHAP bar plot to', bar_png)
    plt.figure(figsize=(8,6))
    shap.summary_plot(arr, Xshap, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(bar_png, dpi=150)
    plt.close()

    print('Done. Plots written:', summary_png, bar_png)


if __name__ == '__main__':
    main()
