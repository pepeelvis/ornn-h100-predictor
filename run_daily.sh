#!/bin/bash
# Daily driver for the Ornn H100 index predictor. Invoked by the
# com.ornn.h100predictor LaunchAgent at 15:00 America/New_York, ahead of
# the index's ~16:30 ET publish. Safe to also run by hand.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(pwd)"
source .venv/bin/activate
cd src
python predict_today.py
python build_dashboard.py

# Publish the updated log + dashboard to the live GitHub Pages site.
cd "$ROOT"
if [ -n "$(git status --porcelain predictions_log.csv docs/)" ]; then
    git add predictions_log.csv docs/
    git commit -m "Daily update: $(date +%Y-%m-%d)"
    git push origin main
fi
