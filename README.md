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

