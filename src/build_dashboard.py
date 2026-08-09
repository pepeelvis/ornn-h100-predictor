"""Render predictions_log.csv into a static dashboard (docs/index.html) for
GitHub Pages. Pure stdlib + pandas -- no server, no build step, just a
self-contained HTML file with an inline Chart.js chart.
"""
from __future__ import annotations

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
<p class="sub">Blend of Holt trend-smoothing + Ridge regression on lag / cross-GPU features, backtested via walk-forward validation against a naive carry-forward baseline. Runs daily at 15:00 ET, ~90 min before the index's ~16:30 ET publish. Last updated {last_updated}.</p>

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

    import json as _json

    html = PAGE_TEMPLATE.format(
        last_updated=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M %Z") or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        n_resolved=n_resolved,
        n_total=n_total,
        blend_mae=blend_mae_s,
        naive_mae=naive_mae_s,
        improvement=improvement,
        improvement_note=improvement_note,
        table_rows=table_rows,
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
