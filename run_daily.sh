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
# This workspace's .git is a linked worktree pointing at a separate primary
# checkout; a transient collision there (e.g. Clawdmeter's own checkpoint
# writes) can make git fail here with "not a git repository" (happened once,
# 2026-08-09). One retry after a short pause absorbs that; if it's still
# broken after that, fail loudly instead of silently skipping today's publish.
cd "$ROOT"
publish() {
    if [ -n "$(git status --porcelain predictions_log.csv docs/)" ]; then
        git add predictions_log.csv docs/
        git commit -m "Daily update: $(date +%Y-%m-%d)"
    fi
    # Unconditional: also catches a prior attempt that committed locally but
    # failed to push (retry below would otherwise see a clean tree and no-op).
    git push origin main
}
if ! publish; then
    echo "Publish failed, retrying in 10s..." >&2
    sleep 10
    publish
fi
