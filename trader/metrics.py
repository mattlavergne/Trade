"""Performance metrics, with the caveats attached rather than omitted.

Every metric here is computed net of fees and slippage. Where a number is
statistically meaningless -- too few trades, too short a sample -- the report
says so instead of printing a confident-looking figure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = {
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "1h": 8_760,
    "4h": 2_190,
    "1d": 365,
}

# Below this many round trips, per-trade statistics are noise. Roughly the
# point where a t-test on mean trade PnL has any power at all.
MIN_TRADES_FOR_SIGNIFICANCE = 30


@dataclass
class Metrics:
    starting_equity: float
    ending_equity: float
    total_return_pct: float
    annualised_return_pct: float
    annualised_volatility_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    calmar: float
    num_fills: int
    num_round_trips: int
    num_rejections: int
    total_fees: float
    total_slippage: float
    fee_drag_pct: float
    win_rate_pct: float
    profit_factor: float
    avg_trade_pnl: float
    trade_pnl_tstat: float
    exposure_pct: float
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


def _round_trip_pnls(fills: list) -> list[float]:
    """Pair buys and sells FIFO into completed round trips and return net PnL.

    Fees and slippage are already embedded in each fill's cash effect, so this
    measures what actually hit the account, not a gross fantasy.
    """
    from .broker import Side

    lots: list[tuple[float, float, float]] = []  # (qty, price, fee_per_unit)
    pnls: list[float] = []

    for fill in fills:
        if fill.side is Side.BUY:
            fee_per_unit = fill.fee / fill.quantity if fill.quantity else 0.0
            lots.append((fill.quantity, fill.fill_price, fee_per_unit))
            continue

        remaining = fill.quantity
        sell_fee_per_unit = fill.fee / fill.quantity if fill.quantity else 0.0
        while remaining > 1e-12 and lots:
            lot_qty, lot_price, lot_fee = lots[0]
            matched = min(remaining, lot_qty)
            gross = (fill.fill_price - lot_price) * matched
            costs = matched * (lot_fee + sell_fee_per_unit)
            pnls.append(gross - costs)
            remaining -= matched
            if matched >= lot_qty - 1e-12:
                lots.pop(0)
            else:
                lots[0] = (lot_qty - matched, lot_price, lot_fee)

    return pnls


def compute(
    equity_curve: pd.Series,
    fills: list,
    rejections: list,
    *,
    timeframe: str,
    total_fees: float,
    total_slippage: float,
    exposure: pd.Series | None = None,
    risk_free_rate: float = 0.0,
) -> Metrics:
    """Compute the full metric set from an equity curve and fill log."""
    if len(equity_curve) < 2:
        raise ValueError("equity curve needs at least 2 points")

    periods = PERIODS_PER_YEAR.get(timeframe, 365)
    start, end = float(equity_curve.iloc[0]), float(equity_curve.iloc[-1])
    total_return = (end / start - 1.0) * 100.0

    returns = equity_curve.pct_change().dropna()
    years = len(equity_curve) / periods

    annualised_return = (
        ((end / start) ** (1.0 / years) - 1.0) * 100.0 if years > 0 and end > 0 else float("nan")
    )
    vol = float(returns.std(ddof=1)) * math.sqrt(periods) * 100.0 if len(returns) > 1 else 0.0

    excess = returns - risk_free_rate / periods
    sharpe = (
        float(excess.mean() / excess.std(ddof=1)) * math.sqrt(periods)
        if len(excess) > 1 and excess.std(ddof=1) > 0
        else 0.0
    )
    downside = excess[excess < 0]
    sortino = (
        float(excess.mean() / downside.std(ddof=1)) * math.sqrt(periods)
        if len(downside) > 1 and downside.std(ddof=1) > 0
        else 0.0
    )

    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max - 1.0) * 100.0
    max_dd = float(drawdown.min())
    calmar = annualised_return / abs(max_dd) if max_dd < 0 else 0.0

    pnls = _round_trip_pnls(fills)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = (len(wins) / len(pnls) * 100.0) if pnls else 0.0
    gross_win, gross_loss = sum(wins), abs(sum(losses))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    avg_pnl = float(np.mean(pnls)) if pnls else 0.0
    if len(pnls) > 1 and np.std(pnls, ddof=1) > 0:
        tstat = float(np.mean(pnls) / (np.std(pnls, ddof=1) / math.sqrt(len(pnls))))
    else:
        tstat = 0.0

    exposure_pct = float(exposure.mean() * 100.0) if exposure is not None and len(exposure) else 0.0
    fee_drag = (total_fees + total_slippage) / start * 100.0

    warnings: list[str] = []
    if len(pnls) < MIN_TRADES_FOR_SIGNIFICANCE:
        warnings.append(
            f"Only {len(pnls)} round trips. Below ~{MIN_TRADES_FOR_SIGNIFICANCE} the per-trade "
            "statistics (win rate, profit factor, average PnL) are noise, not evidence."
        )
    if abs(tstat) < 2.0 and len(pnls) >= MIN_TRADES_FOR_SIGNIFICANCE:
        warnings.append(
            f"Mean trade PnL t-statistic is {tstat:.2f} (|t| < 2). The edge is not "
            "statistically distinguishable from zero."
        )
    if years < 0.5:
        warnings.append(
            f"Sample covers only {years * 365:.0f} days. Annualised figures extrapolated from "
            "this little data are close to meaningless."
        )
    if fee_drag > 10.0:
        warnings.append(
            f"Costs consumed {fee_drag:.1f}% of starting capital. This strategy trades too often "
            "for its account size and fee tier."
        )
    if rejections:
        warnings.append(
            f"{len(rejections)} orders were rejected (mostly below venue minimum notional). "
            "The strategy as designed cannot be fully executed at this account size."
        )

    return Metrics(
        starting_equity=start,
        ending_equity=end,
        total_return_pct=total_return,
        annualised_return_pct=annualised_return,
        annualised_volatility_pct=vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_pct=max_dd,
        calmar=calmar,
        num_fills=len(fills),
        num_round_trips=len(pnls),
        num_rejections=len(rejections),
        total_fees=total_fees,
        total_slippage=total_slippage,
        fee_drag_pct=fee_drag,
        win_rate_pct=win_rate,
        profit_factor=profit_factor,
        avg_trade_pnl=avg_pnl,
        trade_pnl_tstat=tstat,
        exposure_pct=exposure_pct,
        warnings=warnings,
    )
