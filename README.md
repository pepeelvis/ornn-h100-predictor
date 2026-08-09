# Ornn H100 Index Predictor

Forecasts where the [Ornn Compute Price Index — H100 SXM](https://dashboard.ornnai.com/) (Bloomberg: `ORNNH100`)
will publish at its next daily print (~16:30 ET / 20:00 UTC).

## How it works

- **Data**: pulled from the same endpoint Ornn's own dashboard calls
  (`GET https://dashboard.ornnai.com/api/gpu/{gpu}/index-history`). No
  Ornn account required, but the public tier caps history at the trailing
  ~3 months (`"access": "public-3mo"` in the response) — that's the ceiling
  on how far back this predictor can see.
- **Forecast**: an average of two models, each re-fit daily on all
  available history:
  - **Holt's linear-trend exponential smoothing** on the H100 series alone.
  - **Ridge regression** on H100 lags/rolling stats/momentum plus lag
    features from correlated GPU series (H200, A100 SXM4, RTX 5090, B200),
    with alpha selected by time-series cross-validation.
  - A naive carry-forward (`tomorrow = today`) is tracked alongside as a
    baseline — the whole point is to know if the model is actually earning
    its keep. As of the last backtest, the blend beat naive by ~5% MAE.
- **Backtest**: walk-forward over the trailing 25 days (re-fit fresh at
  each step, predict one day out, compare to what actually published).
  Backtest error also sets the 80% interval width.
- **Uncertainty**: `blend ± 1.28 * (backtest residual std)` as an 80%
  interval, not a hard bound.

## Usage

```bash
source .venv/bin/activate
cd src
python predict_today.py
```

This fetches fresh data, backtests, prints today's forecast, and appends a
row to `predictions_log.csv` in the repo root. On each run it also
reconciles any past prediction whose target date has since published,
filling in `actual` and `abs_error` — so the log doubles as an accuracy
track record over time.

## Daily automation

A macOS LaunchAgent (`com.ornn.h100predictor`, installed at
`~/Library/LaunchAgents/com.ornn.h100predictor.plist`) runs
`run_daily.sh` every day at **15:00 America/New_York** — ahead of the
~16:30 ET publish, and after the prior day's actual has already landed
(so reconciliation happens on the same run). The Mac's system timezone is
already `America/New_York`, so this tracks ET correctly across DST without
any manual adjustment.

```bash
# check status / next run
launchctl list | grep ornn

# run it manually right now
launchctl kickstart -k gui/$(id -u)/com.ornn.h100predictor

# logs
tail -f ~/Library/Logs/ornn-h100predictor.log
tail -f ~/Library/Logs/ornn-h100predictor.err.log

# disable
launchctl bootout gui/$(id -u)/com.ornn.h100predictor
```

Caveat: `StartCalendarInterval` doesn't wake a sleeping Mac — if the machine
is asleep at 15:00 it fires on next wake instead.

## Known limitations

- **91-day public window.** No Ornn Data login is wired in, so the model
  never sees more than ~3 months of history. `predictions_log.csv` is the
  only way this project accumulates a longer track record over time.
- **Small-sample model.** With ~90 usable rows, the Ridge model deliberately
  uses a light feature set (a handful of H100 lags/rolling stats + lag-1/2
  of correlated GPUs) to avoid overfitting — it is not trying to be
  sophisticated.
- **Publish time is inferred, not documented.** Every data point in the
  public API carries a `20:00:00.000Z` timestamp, which is 16:00 ET during
  EDT — not confirmed to be exactly 16:30 ET. Treated as "the daily print"
  here; if Ornn's actual publish timing differs, the target-date alignment
  may be off by a day at the boundary.
- **No live intraday updates.** The API only exposes one value per
  calendar day, so there's nothing to refine the forecast with between
  runs — running more than once a day is a no-op until the next print.
