# covid-vax-cardio-risk

Repository and reproducible workflow for investigating vaccine-associated cardiac events using VAERS and related datasets.

This project provides ingestion, preprocessing, cohort extraction, baseline modeling, and explainability (SHAP) examples implemented in Python. It is organized so you can reproduce analyses, extend labels/outcomes, and swap models.

## Quick Links

- **Scripts:** `scripts/` — runnable helpers (preprocessing, cohort extraction, SHAP example)
- **Code:** `src/python/` — data ingestion, preprocessing, cohort & labeling logic, feature builders
- **Data:** `data/` — local raw files (not checked into git)
- **Outputs:** `data/processed/`, `models/`, `notebooks/`

## Requirements & Environment

- Python 3.10+ recommended. Use the provided `environment.yml` or `requirements.txt`.

Conda (recommended):
```bash
conda env create -f environment.yml
conda activate covid-vax-cardio-risk
```

Pip / venv:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data (important)

- Raw VAERS CSVs and other large datasets are NOT included in this repo. Place your local copies under `data/` (for example `data/2019VAERSDATA.csv`, `data/aggregated.csv.gz`).
- The repository `.gitignore` intentionally excludes `data/*.csv`, `data/processed/`, and `models/` to avoid pushing large files to Git.

If you want to version large files, consider Git LFS or an external data bucket.

## Project Structure

- `scripts/`
  - `run_vaers_cohort.py`: preprocess VAERS files, map vaccine manufacturers, create a COVID-only mapped file (`vaers_mapped_covid.parquet`), and extract a cardiac cohort.
  - `run_shap_example.py`: sample data, train a RandomForestClassifier, compute SHAP explanations, and save plots to `notebooks/`.
- `src/python/`
  - `vaers_preprocess.py`: loading/concatenation and basic text flags (cardiac_flag, serious)
  - `vaers_cohort.py`: vaccine manufacturer mapping, COVID filter, cohort extraction, and refined labeling (`vaccine_related_cardiac`)
  - helper modules: `features.py`, `models.py`, `preprocessing.py`
- `data/processed/` — output parquet files (ignored by git)
- `models/` — saved model artifacts (ignored by git)

## Quick Start — run the demo pipeline

1) Preprocess VAERS and create COVID-only mapped dataset + cardiac cohort

```bash
python scripts/run_vaers_cohort.py --data-dir data --out-dir data/processed
```

2) Train demo model and compute SHAP (prefers COVID-only mapped file)

```bash
python scripts/run_shap_example.py
```

3) Run tests

```bash
pytest -q
```

## Configuration & Customization

- To change which manufacturers are considered COVID vaccines, edit `_MANU_PATTERNS` and `filter_covid_vaccines()` in `src/python/vaers_cohort.py`.
- To change the outcome definition window or seriousness requirement, modify `label_vaccine_related_cardiac(..., window_days=..., require_serious=...)` in `src/python/vaers_cohort.py`.
- To tune modeling parameters, edit `scripts/run_shap_example.py` (sampling sizes, classifier hyperparameters).

## Notes on Labels and Features

- The default refined label `vaccine_related_cardiac` uses a 28-day window and requires `serious` events by default. This focuses on hospitalization/death/life-threatening outcomes unless overridden.
- `scripts/run_shap_example.py` now builds simple comorbidity flags from free-text fields (`PHM`, `HISTORY`, `CUR_ILL`, `OTHER_MEDS`) — these are regex-based heuristics and should be validated for your dataset.

## GitHub / CI

- A helper script is provided: `tools/create_and_push_repo.sh` — it initializes git and uses the `gh` CLI to create & push a remote repo (if `gh` is available).
- A basic CI workflow is available at `.github/workflows/ci.yml` which runs `pytest` on push/PR.

## Troubleshooting

- SHAP plotting: if you see shape mismatch errors, update `shap` to a compatible version or inspect the defensive handling in `scripts/run_shap_example.py` which chooses the correct SHAP slice for plotting.
- Push failures due to large files: ensure raw CSVs are ignored via `.gitignore` and remove tracked large files with `git rm --cached`. Use Git LFS if you want to version large binaries.

## License

- MIT License — see `LICENSE`.

## Contributing

- Contributions welcome. Open an issue or PR. For data access or reproducibility requests, contact the repo owner.

## Next actions I can help with

- add `data/README.md` with instructions to download VAERS files and how to place them locally,
- add a CLI flag to `run_vaers_cohort.py` to override the COVID manufacturer list at runtime,
- add a small notebook demonstrating loading `data/processed/vaers_mapped_covid.parquet` and computing SHAP plots interactively.
# covid-vax-cardio-risk

Repository for building models and explainability workflows to study possible cardiac events related to vaccination using VAERS and related datasets.

This repo contains ingestion, preprocessing, cohort extraction, a demo model and SHAP explainability scripts under `src/python/` and `scripts/`.

Quick setup

1. Create a local git repository, commit files, and push to GitHub.

Recommended (using GitHub CLI `gh`):

```bash
# from repository root
git init
git add .
git commit -m "Initial import: covid-vax-cardio-risk"
# create remote repo under your account (replace name if desired)
gh repo create kalyanipande8/covid-vax-cardio-risk --public --source=. --remote=origin --push
```

Alternative (manual HTTPS or SSH):

```bash
git init
git add .
git commit -m "Initial import: covid-vax-cardio-risk"
# create repo on github.com under your account kalyanipande8 (via web)
# then add remote (SSH):
git remote add origin git@github.com:kalyanipande8/covid-vax-cardio-risk.git
# or HTTPS:
git remote add origin https://github.com/kalyanipande8/covid-vax-cardio-risk.git
git branch -M main
git push -u origin main
```

Notes and best practices

- Large files and processed outputs are ignored by `.gitignore`. Keep raw data out of the repo; store it in a data registry or cloud storage and reference it in `data/` only for local runs.
- If you want to include `data/processed` artifacts in the repo, remove that path from `.gitignore` and use Git LFS for large binaries.
- Consider adding a `LICENSE` and CONTRIBUTING guide before publishing.

Run the example pipeline locally

```bash
# build processed VAERS mapped data
python scripts/run_vaers_cohort.py --data-dir data --out-dir data/processed

# run the SHAP example (trains a small RF and writes SHAP PNGs)
python scripts/run_shap_example.py
```

If you want me to create and push the repository for you, I can output the exact commands to run locally or prepare a script that uses the `gh` CLI — but I cannot create the GitHub repo directly from here without credentials. Follow the `gh` command above (recommended) to create and push in one step.

Automated create & push script

I added a small helper script `tools/create_and_push_repo.sh` to initialize the repo, commit, and create+push the remote using `gh` (if available). Example usage:

```bash
# make script executable first (once):
chmod +x tools/create_and_push_repo.sh

# create and push the repo under your account (defaults to kalyanipande8/covid-vax-cardio-risk):
./tools/create_and_push_repo.sh kalyanipande8 covid-vax-cardio-risk
```

Notes:
- The script will use the `gh` CLI if installed and authenticated; otherwise it will set `origin` to your SSH remote and instruct you to create the repo on GitHub manually and push.
- The script sets the branch to `main` and creates an initial commit if one does not exist.

