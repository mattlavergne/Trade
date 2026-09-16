"""Console and HTML reporting.

The HTML report is a single self-contained file with no external assets, so it
can be dropped straight onto static hosting (e.g. mattlavergne.com) or opened
locally. Charts are inline SVG -- no JS libraries, nothing to load, nothing to
break.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .costs import breakeven_move_pct, get_venue
from .engine import BacktestResult

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def print_result(result: BacktestResult) -> None:
    """Print a backtest result to the console, warnings included."""
    m, b = result.metrics, result.benchmark_metrics
    venue = get_venue(result.venue)

    print()
    print("=" * 78)
    print(f"  {result.strategy}  |  {result.symbol} {result.timeframe}  |  venue: {result.venue}")
    print("=" * 78)
    print(f"  Period            {result.equity_curve.index[0].date()} -> {result.equity_curve.index[-1].date()}")
    print(f"  Starting equity   ${m.starting_equity:,.2f}")
    print(f"  Ending equity     ${m.ending_equity:,.2f}")
    print()
    print(f"  {'':22s}{'STRATEGY':>14s}{'BUY & HOLD':>14s}{'DIFFERENCE':>14s}")
    print(f"  {'-' * 62}")
    rows = [
        ("Total return", f"{m.total_return_pct:.2f}%", f"{b.total_return_pct:.2f}%",
         f"{result.excess_return_pct:+.2f}%"),
        ("Annualised return", f"{m.annualised_return_pct:.2f}%", f"{b.annualised_return_pct:.2f}%",
         f"{m.annualised_return_pct - b.annualised_return_pct:+.2f}%"),
        ("Max drawdown", f"{m.max_drawdown_pct:.2f}%", f"{b.max_drawdown_pct:.2f}%",
         f"{m.max_drawdown_pct - b.max_drawdown_pct:+.2f}%"),
        ("Sharpe", f"{m.sharpe:.2f}", f"{b.sharpe:.2f}", f"{m.sharpe - b.sharpe:+.2f}"),
        ("Volatility (ann.)", f"{m.annualised_volatility_pct:.2f}%",
         f"{b.annualised_volatility_pct:.2f}%", ""),
    ]
    for label, s, bench, diff in rows:
        print(f"  {label:22s}{s:>14s}{bench:>14s}{diff:>14s}")

    print()
    print(f"  Round trips       {m.num_round_trips}")
    print(f"  Fills             {m.num_fills}")
    print(f"  Rejected orders   {m.num_rejections}")
    print(f"  Time in market    {m.exposure_pct:.1f}%")
    print(f"  Win rate          {m.win_rate_pct:.1f}%")
    print(f"  Profit factor     {m.profit_factor:.2f}")
    print(f"  Avg trade PnL     ${m.avg_trade_pnl:.4f}  (t = {m.trade_pnl_tstat:.2f})")
    print()
    print(f"  Fees paid         ${m.total_fees:,.2f}")
    print(f"  Slippage paid     ${m.total_slippage:,.2f}")
    print(f"  Total cost drag   {m.fee_drag_pct:.2f}% of starting capital")
    print(f"  Breakeven move    {breakeven_move_pct(venue):.2f}% per round trip on {venue.name}")

    verdict = "BEAT" if result.beat_benchmark else "LOST TO"
    print()
    print(f"  VERDICT: this strategy {verdict} buy-and-hold by {abs(result.excess_return_pct):.2f} "
          "percentage points, net of all costs.")

    if m.warnings:
        print()
        print("  READ THIS BEFORE BELIEVING ANY NUMBER ABOVE:")
        for warning in m.warnings:
            print(f"    - {warning}")
    print("=" * 78)
    print()


def _sparkline_svg(
    series: pd.Series, benchmark: pd.Series, width: int = 900, height: int = 300
) -> str:
    """Two-line inline SVG chart of equity vs benchmark. No dependencies."""
    pad = 44
    combined = pd.concat([series, benchmark])
    lo, hi = float(combined.min()), float(combined.max())
    span = (hi - lo) or 1.0

    def path_for(values: pd.Series) -> str:
        n = len(values)
        points = []
        for i, value in enumerate(values.to_numpy()):
            x = pad + (width - 2 * pad) * (i / max(n - 1, 1))
            y = height - pad - (height - 2 * pad) * ((float(value) - lo) / span)
            points.append(f"{x:.1f},{y:.1f}")
        return "M" + " L".join(points)

    start_y = height - pad - (height - 2 * pad) * ((float(series.iloc[0]) - lo) / span)
    return f"""<svg viewBox="0 0 {width} {height}" role="img" aria-label="Equity curve versus buy and hold">
  <rect x="0" y="0" width="{width}" height="{height}" fill="none"/>
  <line x1="{pad}" y1="{start_y:.1f}" x2="{width - pad}" y2="{start_y:.1f}"
        stroke="var(--grid)" stroke-dasharray="4 4" stroke-width="1"/>
  <path d="{path_for(benchmark)}" fill="none" stroke="var(--bench)" stroke-width="2"
        stroke-linejoin="round" stroke-linecap="round"/>
  <path d="{path_for(series)}" fill="none" stroke="var(--accent)" stroke-width="2.5"
        stroke-linejoin="round" stroke-linecap="round"/>
  <text x="{pad}" y="{pad - 16}" fill="var(--muted)" font-size="13">${hi:,.2f}</text>
  <text x="{pad}" y="{height - pad + 22}" fill="var(--muted)" font-size="13">${lo:,.2f}</text>
</svg>"""


def _metric_card(label: str, value: str, note: str = "", tone: str = "") -> str:
    tone_class = f" {tone}" if tone else ""
    note_html = f'<div class="note">{html.escape(note)}</div>' if note else ""
    return (
        f'<div class="card{tone_class}"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>{note_html}</div>'
    )


def write_html_report(result: BacktestResult, path: Path | None = None) -> Path:
    """Render a self-contained HTML report and return its path."""
    m, b = result.metrics, result.benchmark_metrics
    venue = get_venue(result.venue)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if path is None:
        stem = result.strategy.split("(")[0]
        path = REPORTS_DIR / f"{stem}_{result.venue}_{result.timeframe}.html"

    beat = result.beat_benchmark
    verdict_tone = "good" if beat else "bad"
    verdict_text = (
        f"{'Beat' if beat else 'Lost to'} buy-and-hold by "
        f"{abs(result.excess_return_pct):.2f} percentage points, net of all costs."
    )

    cards = "".join([
        _metric_card("Net return", f"{m.total_return_pct:.2f}%",
                     f"buy & hold: {b.total_return_pct:.2f}%",
                     "good" if m.total_return_pct > 0 else "bad"),
        _metric_card("Max drawdown", f"{m.max_drawdown_pct:.2f}%",
                     f"buy & hold: {b.max_drawdown_pct:.2f}%"),
        _metric_card("Sharpe", f"{m.sharpe:.2f}", f"buy & hold: {b.sharpe:.2f}"),
        _metric_card("Round trips", f"{m.num_round_trips}", f"{m.num_fills} fills"),
        _metric_card("Costs paid", f"${m.total_fees + m.total_slippage:,.2f}",
                     f"{m.fee_drag_pct:.1f}% of starting capital",
                     "bad" if m.fee_drag_pct > 10 else ""),
        _metric_card("Breakeven move", f"{breakeven_move_pct(venue):.2f}%",
                     f"required per round trip on {venue.name}"),
        _metric_card("Avg trade PnL", f"${m.avg_trade_pnl:.4f}", f"t-stat {m.trade_pnl_tstat:.2f}"),
        _metric_card("Time in market", f"{m.exposure_pct:.1f}%",
                     f"win rate {m.win_rate_pct:.1f}%"),
    ])

    warnings_html = ""
    if m.warnings:
        items = "".join(f"<li>{html.escape(w)}</li>" for w in m.warnings)
        warnings_html = (
            '<section class="warnings"><h2>Read this before believing any number above</h2>'
            f"<ul>{items}</ul></section>"
        )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Backtest Report</title>
<style>
  :root {{
    --bg: #ffffff; --fg: #16191d; --muted: #6b7280; --line: #e5e7eb;
    --card: #f8f9fb; --accent: #2563eb; --bench: #9ca3af; --grid: #d1d5db;
    --good: #047857; --bad: #b91c1c; --warn-bg: #fef3c7; --warn-line: #f59e0b;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #0f1114; --fg: #e8eaed; --muted: #9aa1ab; --line: #262b32;
      --card: #171a1f; --accent: #60a5fa; --bench: #6b7280; --grid: #374151;
      --good: #34d399; --bad: #f87171; --warn-bg: #2a2110; --warn-line: #b45309;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #0f1114; --fg: #e8eaed; --muted: #9aa1ab; --line: #262b32;
    --card: #171a1f; --accent: #60a5fa; --bench: #6b7280; --grid: #374151;
    --good: #34d399; --bad: #f87171; --warn-bg: #2a2110; --warn-line: #b45309;
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--fg);
         font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif; }}
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 40px 16px 72px; }}
  h1 {{ font-size: 24px; margin: 0 0 4px; letter-spacing: -0.01em; }}
  .sub {{ color: var(--muted); font-size: 14px; margin-bottom: 28px; }}
  .verdict {{ padding: 14px 18px; border-radius: 10px; font-weight: 600;
              border: 1px solid var(--line); margin-bottom: 28px; }}
  .verdict.good {{ color: var(--good); }}
  .verdict.bad {{ color: var(--bad); }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
           gap: 12px; margin-bottom: 32px; }}
  .card {{ background: var(--card); border: 1px solid var(--line);
           border-radius: 10px; padding: 14px 16px; }}
  .card .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: .06em;
                  color: var(--muted); }}
  .card .value {{ font-size: 24px; font-weight: 650; margin-top: 6px;
                  font-variant-numeric: tabular-nums; }}
  .card.good .value {{ color: var(--good); }}
  .card.bad .value {{ color: var(--bad); }}
  .card .note {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  .chart {{ border: 1px solid var(--line); border-radius: 10px; padding: 8px;
            background: var(--card); margin-bottom: 12px; }}
  .chart svg {{ display: block; width: 100%; height: auto; }}
  .legend {{ display: flex; gap: 20px; font-size: 13px; color: var(--muted);
             margin-bottom: 32px; }}
  .swatch {{ display: inline-block; width: 22px; height: 3px; vertical-align: middle;
             margin-right: 7px; border-radius: 2px; }}
  .warnings {{ background: var(--warn-bg); border-left: 3px solid var(--warn-line);
               border-radius: 8px; padding: 16px 20px; }}
  .warnings h2 {{ font-size: 15px; margin: 0 0 10px; }}
  .warnings ul {{ margin: 0; padding-left: 20px; }}
  .warnings li {{ margin-bottom: 8px; }}
  footer {{ margin-top: 40px; padding-top: 18px; border-top: 1px solid var(--line);
            color: var(--muted); font-size: 13px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{html.escape(result.strategy)}</h1>
  <div class="sub">
    {html.escape(result.symbol)} &middot; {html.escape(result.timeframe)} bars &middot;
    venue {html.escape(result.venue)} &middot;
    {result.equity_curve.index[0].date()} to {result.equity_curve.index[-1].date()} &middot;
    ${m.starting_equity:,.2f} &rarr; ${m.ending_equity:,.2f}
  </div>

  <div class="verdict {verdict_tone}">{html.escape(verdict_text)}</div>

  <div class="grid">{cards}</div>

  <div class="chart">{_sparkline_svg(result.equity_curve, result.benchmark_curve)}</div>
  <div class="legend">
    <span><span class="swatch" style="background:var(--accent)"></span>Strategy equity</span>
    <span><span class="swatch" style="background:var(--bench)"></span>Buy &amp; hold</span>
  </div>

  {warnings_html}

  <footer>
    Generated {generated} &middot; simulated fills only, no real orders were placed.
    Past performance on historical bars is not evidence of future profitability.
  </footer>
</div>
</body>
</html>"""

    path.write_text(doc, encoding="utf-8")
    return path
