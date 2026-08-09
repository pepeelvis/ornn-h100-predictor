#!/bin/bash
# Daily driver for the Ornn H100 index predictor. Invoked by the
# com.ornn.h100predictor LaunchAgent at 15:00 America/New_York, ahead of
# the index's ~16:30 ET publish. Safe to also run by hand.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate
cd src
python predict_today.py
