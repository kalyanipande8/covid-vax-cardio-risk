#!/usr/bin/env bash
# Simple bootstrap for Python and R development environments (informational).

echo "This script provides recommended commands to create Python and R environments." 
echo "It does not modify your system by default; run the commands shown manually after review."

echo
echo "Python (conda):"
echo "  conda env create -f environment.yml"
echo "  conda activate covid-vax-cardio-risk"

echo
echo "Python (venv + pip):"
echo "  python3 -m venv .venv"
echo "  source .venv/bin/activate"
echo "  pip install -r requirements.txt"

echo
echo "R (renv):"
echo "  # from an R session: install.packages('renv'); renv::init(); renv::restore()"

echo
echo "After environment setup, run tests: pytest -q"
