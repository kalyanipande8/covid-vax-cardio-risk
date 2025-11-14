"""Train a small demo model on the aggregated dataset.

Usage example:
  PYTHONPATH=. python3 scripts/train_demo_model.py \
      --input data/processed/aggregated.parquet \
      --target new_confirmed \
      --model random_forest \
      --outdir models/demo

The script is defensive: if required packages are missing it prints
installation hints and exits gracefully.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def ensure_deps():
    try:
        import pandas as pd  # noqa: F401
        import numpy as np  # noqa: F401
        from sklearn.ensemble import RandomForestRegressor  # noqa: F401
        from sklearn.model_selection import train_test_split  # noqa: F401
        from sklearn.metrics import mean_squared_error  # noqa: F401
    except Exception:
        print("Required Python packages are missing. Install with:")
        print("  pip install -r requirements.txt")
        print("  pip install pyarrow scikit-learn joblib")
        sys.exit(2)


def find_input_path(proposed: Path | None) -> Path:
    candidates = []
    if proposed:
        candidates.append(proposed)
    candidates += [
        PROJECT_ROOT / "data" / "processed" / "aggregated.parquet",
        PROJECT_ROOT / "data" / "processed" / "aggregated_processed.csv",
        PROJECT_ROOT / "data" / "processed" / "aggregated_raw_copy.csv.gz",
        PROJECT_ROOT / "data" / "aggregated.csv.gz",
    ]

    for p in candidates:
        if p and p.exists():
            return p
    raise FileNotFoundError("No aggregated input file found. Place your file in data/processed or data/")


def load_df(path: Path):
    import pandas as pd

    if path.suffix == ".parquet" or path.suffixes[-2:] == [".parquet"]:
        return pd.read_parquet(path)
    # let pandas handle compression by suffix
    return pd.read_csv(path, compression=("gzip" if str(path).endswith(".gz") else None))


def select_features(df, target: str):
    # drop obviously non-feature columns
    drop_like = ["country", "iso", "name", "code", "id", "place_id", "locality_name", "subregion"]
    cols = [c for c in df.columns if c != target]
    features = []
    for c in cols:
        low = c.lower()
        if any(x in low for x in drop_like):
            continue
        # keep numeric columns only
        if pd.api.types.is_numeric_dtype(df[c]):
            features.append(c)
    return df[features], df[target]


def main(argv=None):
    ensure_deps()
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import joblib

    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, help="Path to input file (parquet or csv.gz)")
    p.add_argument("--target", type=str, required=True, help="Target column to predict (numeric)")
    p.add_argument("--model", choices=["random_forest"], default="random_forest")
    p.add_argument("--outdir", type=Path, default=Path("models/demo"))
    p.add_argument("--test-size", type=float, default=0.2)
    args = p.parse_args(argv)

    try:
        input_path = find_input_path(args.input)
    except FileNotFoundError as e:
        print(e)
        sys.exit(2)

    print(f"Loading data from {input_path}")
    df = load_df(input_path)
    if args.target not in df.columns:
        print(f"Target column '{args.target}' not found in the dataset. Available cols: {list(df.columns)[:20]}")
        sys.exit(2)

    # basic preprocessing: drop rows where target is missing
    df = df.dropna(subset=[args.target])

    X, y = select_features(df, args.target)

    if X.shape[0] < 10:
        print("Not enough rows after filtering to train a model")
        sys.exit(2)

    # fill missing numeric values with median
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    if args.model == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        raise NotImplementedError(args.model)

    print("Training model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "model.pkl"
    metrics_path = outdir / "metrics.json"

    joblib.dump(model, model_path)
    with open(metrics_path, "w") as fh:
        json.dump({"mse": float(mse), "r2": float(r2), "n_train": int(X_train.shape[0]), "n_test": int(X_test.shape[0])}, fh, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"MSE: {mse:.4f}, R2: {r2:.4f}")


if __name__ == "__main__":
    main()
