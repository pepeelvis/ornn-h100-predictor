"""Render predictions_log.csv into a static dashboard (docs/index.html) for
GitHub Pages. Pure stdlib + pandas -- no server, no build step, just a
self-contained HTML file with an inline Chart.js chart.
"""
from __future__ import annotations

import html as _html
import json as _json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG_PATH = os.path.join(ROOT, "predictions_log.csv")
OUT_PATH = os.path.join(ROOT, "docs", "index.html")

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ornn H100 SXM Index -- Daily Predictor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {{ color-scheme: light dark; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 920px; margin: 2.5rem auto; padding: 0 1.25rem;
    color: #1a1a1a; background: #fff;
  }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.1rem; }}
  .sub {{ color: #666; margin-bottom: 1.75rem; font-size: 0.92rem; }}
  .stats {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.75rem; }}
  .stat {{
    border: 1px solid #e0e0e0; border-radius: 10px; padding: 0.9rem 1.15rem;
    min-width: 150px; flex: 1;
  }}
  .stat .label {{ font-size: 0.78rem; color: #777; text-transform: uppercase; letter-spacing: 0.03em; }}
  .stat .value {{ font-size: 1.5rem; font-weight: 600; margin-top: 0.2rem; }}
  .stat .note {{ font-size: 0.78rem; color: #888; margin-top: 0.15rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-top: 0.5rem; }}
  th, td {{ text-align: right; padding: 0.4rem 0.6rem; border-bottom: 1px solid #eee; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: #666; font-weight: 500; font-size: 0.78rem; text-transform: uppercase; }}
  tr:last-child td {{ border-bottom: none; }}
  .pending {{ color: #999; font-style: italic; }}
  footer {{ margin-top: 2.5rem; color: #999; font-size: 0.8rem; }}
  canvas {{ max-height: 340px; }}
  @media (prefers-color-scheme: dark) {{
    body {{ color: #eee; background: #111; }}
    .stat {{ border-color: #333; }}
    .stat .label, .stat .note {{ color: #999; }}
    th {{ color: #999; }}
    th, td {{ border-color: #292929; }}
  }}
</style>
</head>
<body>
<h1>Ornn H100 SXM Index -- Daily Predictor</h1>
<p class="sub">Inverse-error-weighted blend of Holt trend-smoothing, Ridge regression on lag / cross-GPU features, and a 7-day mean-reversion model, backtested via walk-forward validation against a naive carry-forward baseline. Runs daily at 15:00 ET, ~90 min before the index's ~16:30 ET publish. Last updated {last_updated}.</p>

<div class="stats">
  <div class="stat">
    <div class="label">Resolved predictions</div>
    <div class="value">{n_resolved}</div>
    <div class="note">of {n_total} logged</div>
  </div>
  <div class="stat">
    <div class="label">Blend MAE</div>
    <div class="value">{blend_mae}</div>
    <div class="note">index units ($/GPU-hr)</div>
  </div>
  <div class="stat">
    <div class="label">Naive MAE</div>
    <div class="value">{naive_mae}</div>
    <div class="note">carry-forward baseline</div>
  </div>
  <div class="stat">
    <div class="label">Improvement vs. naive</div>
    <div class="value">{improvement}</div>
    <div class="note">{improvement_note}</div>
  </div>
</div>

<canvas id="chart"></canvas>

<h2 style="font-size:1.05rem; margin-top:2rem;">Track record</h2>
<table>
  <thead>
    <tr><th>Target date</th><th>Blend forecast</th><th>80% interval</th><th>Actual</th><th>Abs. error</th></tr>
  </thead>
  <tbody>
    {table_rows}
  </tbody>
</table>

{curve_section}

{spread_section}

<footer>
  Generated from <code>predictions_log.csv</code> by <code>build_dashboard.py</code>, pushed automatically each daily run.
  Data source: Ornn's public H100 SXM index (public tier caps at ~3 months of history).
  Small sample size -- treat early accuracy numbers as provisional.
</footer>

<script>
const labels = {labels_json};
const actual = {actual_json};
const blend = {blend_json};
const lower = {lower_json};
const upper = {upper_json};

new Chart(document.getElementById('chart'), {{
  type: 'line',
  data: {{
    labels: labels,
    datasets: [
      {{
        label: '80% interval',
        data: upper,
        borderWidth: 0,
        pointRadius: 0,
        backgroundColor: 'rgba(99,132,255,0.12)',
        fill: '+1',
      }},
      {{
        label: '80% interval (lower)',
        data: lower,
        borderWidth: 0,
        pointRadius: 0,
        backgroundColor: 'rgba(99,132,255,0.12)',
        fill: false,
      }},
      {{
        label: 'Blend forecast',
        data: blend,
        borderColor: '#6384ff',
        backgroundColor: '#6384ff',
        borderWidth: 2,
        pointRadius: 3,
        tension: 0.15,
      }},
      {{
        label: 'Actual',
        data: actual,
        borderColor: '#1a1a1a',
        backgroundColor: '#1a1a1a',
        borderWidth: 2,
        pointRadius: 3,
        spanGaps: true,
      }},
    ],
  }},
  options: {{
    plugins: {{
      legend: {{ labels: {{ filter: (item) => item.text !== '80% interval (lower)' }} }},
    }},
    scales: {{
      y: {{ title: {{ display: true, text: '$ / GPU-hr' }} }},
    }},
  }},
}});
</script>
</body>
</html>
"""


def fmt(v, digits=4):
    return "--" if pd.isna(v) else f"{v:.{digits}f}"


def build_curve_section() -> str:
    """H100 forward curve: Ornn spot -> Lighter perp -> Kalshi implied
    ladder prices. Best-effort -- if Kalshi/Lighter are unreachable, the
    rest of the dashboard (the actual predictor track record) must still
    build and publish, so any failure here degrades to a placeholder."""
    try:
        from market_curves import build_h100_term_structure

        df = build_h100_term_structure()
    except Exception as e:  # pragma: no cover - network dependent
        return (
            '<h2 style="font-size:1.05rem; margin-top:2rem;">H100 forward curve</h2>\n'
            f'<p class="pending">Temporarily unavailable ({_html.escape(str(e))}).</p>'
        )

    rows = "\n    ".join(
        f"<tr><td>{r['venue']}</td><td>{r['instrument']}</td><td>{int(r['days_out'])}</td>"
        f"<td>{r['implied_price']:.3f}</td><td>{r['vs_spot_pct']:+.1f}%</td></tr>"
        for _, r in df.iterrows()
    )
    labels = _json.dumps([r["instrument"] for _, r in df.iterrows()])
    prices = _json.dumps([round(float(r["implied_price"]), 4) for _, r in df.iterrows()])

    return f"""<h2 style="font-size:1.05rem; margin-top:2rem;">H100 forward curve</h2>
<p class="sub" style="margin-bottom:0.75rem;">Ornn OCPI spot, Lighter's H100 perp mark price, and Kalshi's implied
forward prices (inverted from its live GPU compute strike ladders, which settle against OCPI). Snapshot at page
build time, not a logged history.</p>
<canvas id="curveChart"></canvas>
<table>
  <thead><tr><th>Venue</th><th>Instrument</th><th>Days out</th><th>Implied price</th><th>vs. spot</th></tr></thead>
  <tbody>
    {rows}
  </tbody>
</table>
<script>
new Chart(document.getElementById('curveChart'), {{
  type: 'bar',
  data: {{
    labels: {labels},
    datasets: [{{
      label: 'Implied H100 price ($/GPU-hr)',
      data: {prices},
      backgroundColor: '#6384ff',
    }}],
  }},
  options: {{
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ y: {{ title: {{ display: true, text: '$ / GPU-hr' }}, beginAtZero: false }} }},
  }},
}});
</script>"""


def build_spread_section() -> str:
    """OCPI vs. GX Hopper US spread model. GX has no free tier, so this
    degrades to setup instructions rather than failing the whole build when
    GX_API_TOKEN/GX_HOPPER_CODE aren't configured."""
    header = '<h2 style="font-size:1.05rem; margin-top:2rem;">OCPI vs. GX Hopper US spread</h2>'
    try:
        from fetch import fetch_gpu_history
        from gx_fetch import fetch_gx_index
        from spread_model import build_spread, fit_ar1, forecast_next_spread, signal, walk_forward_backtest

        ocpi = fetch_gpu_history("H100 SXM")
        gx = fetch_gx_index()
    except RuntimeError as e:
        escaped = _html.escape(str(e)).replace("\n", "<br>")
        return f'{header}\n<p class="pending">{escaped}</p>'
    except Exception as e:  # pragma: no cover - network dependent
        return f'{header}\n<p class="pending">Temporarily unavailable ({_html.escape(str(e))}).</p>'

    try:
        df = build_spread(ocpi, gx)
        params = fit_ar1(df["spread"])
        fc = forecast_next_spread(df["spread"], params)
        z = df["spread_zscore"].iloc[-1]
        bt = walk_forward_backtest(df["spread"])
    except Exception as e:  # pragma: no cover - data dependent
        return f'{header}\n<p class="pending">Not enough overlapping OCPI/GX history yet ({_html.escape(str(e))}).</p>'

    labels = _json.dumps([d.isoformat() for d in df.index])
    spread_json = _json.dumps([round(float(v), 4) for v in df["spread"]])

    return f"""{header}
<p class="sub" style="margin-bottom:0.75rem;">Ornn's OCPI H100 series (single-SKU) minus Compute Desk's GX Hopper US
index (blends H100+H200), modeled as a mean-reverting AR(1) process. {signal(z)}</p>
<div class="stats">
  <div class="stat"><div class="label">Latest spread</div><div class="value">{fmt(df['spread'].iloc[-1], 3)}</div><div class="note">$/GPU-hr</div></div>
  <div class="stat"><div class="label">Z-score</div><div class="value">{fmt(z, 2)}</div><div class="note">vs 20d rolling</div></div>
  <div class="stat"><div class="label">Half-life</div><div class="value">{fmt(params['half_life_days'], 1)}</div><div class="note">days to revert 50%</div></div>
  <div class="stat"><div class="label">Next-day forecast</div><div class="value">{fmt(fc, 3)}</div><div class="note">AR(1) MAE {fmt(bt['mae_ar1'], 3)} vs naive {fmt(bt['mae_naive'], 3)}</div></div>
</div>
<canvas id="spreadChart"></canvas>
<script>
new Chart(document.getElementById('spreadChart'), {{
  type: 'line',
  data: {{
    labels: {labels},
    datasets: [{{
      label: 'OCPI - GX Hopper spread',
      data: {spread_json},
      borderColor: '#ff6384',
      backgroundColor: '#ff6384',
      borderWidth: 2,
      pointRadius: 2,
    }}],
  }},
  options: {{
    scales: {{ y: {{ title: {{ display: true, text: '$ / GPU-hr' }} }} }},
  }},
}});
</script>"""


def main():
    log = pd.read_csv(LOG_PATH, parse_dates=["target_date", "last_known_date"])
    log = log.sort_values("target_date")

    resolved = log.dropna(subset=["actual"])
    n_resolved, n_total = len(resolved), len(log)

    if n_resolved > 0:
        blend_mae = (resolved["actual"] - resolved["blend"]).abs().mean()
        naive_mae = (resolved["actual"] - resolved["naive"]).abs().mean()
        improvement = f"{(1 - blend_mae / naive_mae) * 100:+.1f}%" if naive_mae else "--"
        improvement_note = "lower blend MAE is better" if naive_mae else ""
        blend_mae_s, naive_mae_s = fmt(blend_mae), fmt(naive_mae)
    else:
        blend_mae_s = naive_mae_s = improvement = "--"
        improvement_note = "not enough resolved days yet"

    rows = []
    for _, r in log.iterrows():
        target = r["target_date"].date().isoformat()
        interval = f"[{fmt(r['lower_80'], 3)}, {fmt(r['upper_80'], 3)}]"
        if pd.notna(r["actual"]):
            rows.append(
                f"<tr><td>{target}</td><td>{fmt(r['blend'], 3)}</td><td>{interval}</td>"
                f"<td>{fmt(r['actual'], 3)}</td><td>{fmt(r['abs_error'])}</td></tr>"
            )
        else:
            rows.append(
                f"<tr><td>{target}</td><td>{fmt(r['blend'], 3)}</td><td>{interval}</td>"
                f'<td class="pending">pending</td><td class="pending">--</td></tr>'
            )
    # newest first in the table
    table_rows = "\n    ".join(reversed(rows))

    labels_json = [r["target_date"].date().isoformat() for _, r in log.iterrows()]
    actual_json = [None if pd.isna(r["actual"]) else round(float(r["actual"]), 4) for _, r in log.iterrows()]
    blend_json = [round(float(r["blend"]), 4) for _, r in log.iterrows()]
    lower_json = [round(float(r["lower_80"]), 4) for _, r in log.iterrows()]
    upper_json = [round(float(r["upper_80"]), 4) for _, r in log.iterrows()]

    print("Building forward-curve section (Kalshi + Lighter)...")
    curve_section = build_curve_section()
    print("Building OCPI/GX spread section...")
    spread_section = build_spread_section()

    html = PAGE_TEMPLATE.format(
        last_updated=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M %Z") or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        n_resolved=n_resolved,
        n_total=n_total,
        blend_mae=blend_mae_s,
        naive_mae=naive_mae_s,
        improvement=improvement,
        improvement_note=improvement_note,
        table_rows=table_rows,
        curve_section=curve_section,
        spread_section=spread_section,
        labels_json=_json.dumps(labels_json),
        actual_json=_json.dumps(actual_json),
        blend_json=_json.dumps(blend_json),
        lower_json=_json.dumps(lower_json),
        upper_json=_json.dumps(upper_json),
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(html)
    print(f"Dashboard written to {OUT_PATH}")


if __name__ == "__main__":
    main()
