#!/usr/bin/env bash
set -euo pipefail

# Script to initialize local git repository and create & push to GitHub.
# Usage: ./tools/create_and_push_repo.sh [GITHUB_OWNER] [REPO_NAME]
# Example: ./tools/create_and_push_repo.sh kalyanipande8 covid-vax-cardio-risk

OWNER=${1:-kalyanipande8}
REPO=${2:-covid-vax-cardio-risk}
REMOTE_SSH="git@github.com:${OWNER}/${REPO}.git"

echo "Preparing to create and push repo: ${OWNER}/${REPO}"

if [ ! -d .git ]; then
  git init
  echo "Initialized empty git repository"
fi

# Create initial commit if none exists
if git rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "Existing commit found"
else
  git add .
  git commit -m "Initial import: covid-vax-cardio-risk" || true
fi

git branch -M main || true

if command -v gh >/dev/null 2>&1; then
  echo "gh CLI found. Attempting to create repo '${OWNER}/${REPO}' and push."
  if gh repo view "${OWNER}/${REPO}" >/dev/null 2>&1; then
    echo "Remote repo already exists on GitHub. Setting origin to existing repo."
    git remote remove origin 2>/dev/null || true
    git remote add origin "${REMOTE_SSH}" || git remote set-url origin "${REMOTE_SSH}"
    git push -u origin main
  else
    # create and push in one command (requires gh auth)
    gh repo create "${OWNER}/${REPO}" --public --source=. --remote=origin --push
  fi
else
  echo "gh CLI not found. I'll set remote to ${REMOTE_SSH} but you must create the repo on GitHub or push to an existing remote."
  git remote remove origin 2>/dev/null || true
  git remote add origin "${REMOTE_SSH}" || true
  echo "To finish: create repo on github.com or change remote, then run: git push -u origin main"
fi

echo "Done. Repository prepared for ${OWNER}/${REPO}."
