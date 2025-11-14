Data directory
=================

Place all project data files in this folder. Keep raw/unprocessed datasets here and
use the `data/processed/` and `data/interim/` subfolders for pipeline outputs.

Recommended structure
- data/raw/        -> original raw files (do not modify)
- data/interim/    -> intermediate files produced during preprocessing
- data/processed/  -> cleaned datasets ready for modeling

Notes
- Do not commit sensitive or personal data to the repository. If you must keep
  private datasets in the repo for testing, add them to `.gitignore` and use a
  secure storage mechanism for real data.
- Name files clearly: e.g. `covid_cases_global_YYYYMMDD.csv`, `vaccinations_global_YYYYMMDD.csv`.
- Keep metadata describing the provenance and any preprocessing steps.

Quick commands
```
# list raw files
ls -la data/raw

# inspect a CSV header
head -n 5 data/raw/your_file.csv
```
# data/

Place raw datasets in this folder. Do NOT commit large or sensitive files to Git.

Recommended files for this project (examples):

- `vaccinations.csv` — vaccine administration records (date, person id, vaccine type, dose)
- `cases.csv` — COVID-19 case reports (date, person id, severity)
- `cardio_outcomes.csv` — cardiovascular diagnoses or events (date, person id, diagnosis_code)
- `demographics.csv` — demographic information (person id, age, sex, location, comorbidities)

Notes and best practices
- Keep original raw files (e.g., `vaccinations.csv`) and create cleaned copies under `data/processed/`.
- If data are sensitive (PHI), keep them encrypted and outside the repository; add pointers or scripts to fetch them from a secure store.
- Use `src/python/data_ingest.py` and `src/R/data_ingest.R` templates to load and standardize formats.

Provenance
- For each dataset, record source, date downloaded, and any transformations performed in this README or a companion metadata file.
