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

A GitHub Actions workflow (`.github/workflows/daily-predict.yml`) runs the
prediction + dashboard build on a schedule (19:00 UTC, i.e. 15:00 ET in
summer / 14:00 ET in winter -- either way comfortably ahead of the ~16:30 ET
publish) and commits/pushes `predictions_log.csv`, `lighter_basis_log.csv`,
and `docs/` if anything changed. Runs on GitHub's own infra, so it doesn't
depend on any local machine being on, awake, or networked.

```bash
# trigger a run manually
gh workflow run daily-predict.yml --repo pepeelvis/ornn-h100-predictor

# check recent runs
gh run list --workflow=daily-predict.yml --repo pepeelvis/ornn-h100-predictor
```

This project previously ran via a macOS LaunchAgent on a local machine
(retired 2026-08-21). That setup repeatedly caused gaps in the public track
record -- the LaunchAgent silently unloaded after any restart, and it once
fired right as the machine woke from sleep, before networking was back,
crashing the run outright. Moving to GitHub Actions eliminates that whole
class of failure.

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

## Cross-market tools

Beyond the single-index predictor above, `src/` also has tools that look at
the H100 price across *other* venues that reference Ornn's OCPI:

- **`kalshi_curve.py`** — builds a market-implied forward curve from
  Kalshi's live, publicly-readable GPU compute prediction markets
  (weekly/monthly/yearly strike ladders on H100/H200/B200/A100/RTX5090,
  settled against OCPI). No auth needed. Extracts an implied median/mean
  price per expiry from each strike ladder's implied survival function
  (isotonic-cleaned to handle thin/stale strikes). Run directly for a
  standalone table, or via `market_curves.py` for the combined view.
- **`lighter_perp.py`** — pulls the H100 perpetual future's mark/index price
  and funding context from Lighter (on-chain perp DEX) via its public,
  unauthenticated market-data API (`mainnet.zklighter.elliot.ai`). Note:
  Lighter's *trading frontend* is geo-restricted and needs a VPN from some
  regions for compliance reasons — that restriction is about executing
  trades, not reading public market data, so this module doesn't need one.
- **`market_curves.py`** — combines the above with the current OCPI spot
  into one H100 term-structure snapshot (`data/h100_term_structure.csv`).
- **`spread_model.py`** + **`gx_fetch.py`** — models the spread between
  Ornn's OCPI H100 index and Compute Desk/General Index's GX Hopper US
  index as a mean-reverting AR(1) process, with a walk-forward backtest and
  a next-day spread forecast. GX has no free tier — `gx_fetch.py` prints
  exact setup steps (free GX Go trial signup, index code lookup, API key)
  if credentials aren't configured via `GX_API_TOKEN`/`GX_HOPPER_CODE` (or
  `GX_USERNAME`/`GX_PASSWORD`) environment variables.
