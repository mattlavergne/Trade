"""Perpetual futures funding rates: the cash-and-carry edge.

A perpetual future has no expiry, so exchanges tether it to spot with a periodic
`funding` payment between longs and shorts. When the perp trades above spot
(the usual state, because retail leverage demand is structurally long), longs
pay shorts.

You can harvest that without taking a directional view:

    long 1 BTC spot  +  short 1 BTC perpetual  =  no price exposure

The position is delta-neutral; you collect funding every period regardless of
where BTC goes. This is a genuine, well-known institutional trade, not a
prediction. What makes it real rather than free money:

  - **Basis risk.** Spot and perp converge over time but diverge in between.
    Your mark-to-market moves even though your net delta is zero.
  - **Liquidation risk.** The short perp leg is margined. A violent rally can
    liquidate it before the spot leg can be sold to cover, turning a neutral
    position into a realised loss.
  - **Negative funding.** In a sustained sell-off, shorts pay longs and the
    trade bleeds.
  - **Counterparty risk.** Your capital sits on an exchange. This is the risk
    that has actually destroyed the most capital in crypto history, and it is
    not diversifiable by any amount of cleverness in the code.
  - **Access.** Perpetual futures are not legally available to US persons on
    the venues where this trade works best. This is a hard blocker, not a
    technicality.
"""

from __future__ import annotations

import logging
import os
import time

import ccxt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Funding is exchanged every 8 hours on the major perpetual venues.
FUNDING_PERIODS_PER_DAY = 3
FUNDING_PERIODS_PER_YEAR = FUNDING_PERIODS_PER_DAY * 365


def make_perp_exchange(venue: str = "okx") -> ccxt.Exchange:
    exchange = getattr(ccxt, venue)({
        "enableRateLimit": True, "timeout": 30_000,
        "options": {"defaultType": "swap"},
    })
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        exchange.httpsProxy = proxy
    return exchange


def fetch_funding_history(
    symbol: str = "BTC/USDT:USDT",
    *,
    venue: str = "okx",
    days: int = 365,
    max_pages: int = 300,
) -> pd.Series:
    """Fetch historical funding rates, paginating backwards.

    Returns a Series of per-period funding rates indexed by UTC timestamp. A
    positive value means longs paid shorts, i.e. the carry trade earned.
    """
    exchange = make_perp_exchange(venue)
    target = exchange.milliseconds() - days * 86_400_000
    rows: list[dict] = []
    cursor: int | None = None

    for _ in range(max_pages):
        params = {"after": cursor} if cursor else {}
        batch = exchange.fetch_funding_rate_history(symbol, limit=100, params=params)
        if not batch:
            break
        rows.extend(batch)
        oldest = min(r["timestamp"] for r in batch)
        if oldest <= target or oldest == cursor:
            break
        cursor = oldest
        time.sleep(exchange.rateLimit / 1000.0)

    if not rows:
        return pd.Series(dtype=float, name=symbol)

    frame = pd.DataFrame([
        {"ts": pd.to_datetime(r["timestamp"], unit="ms", utc=True),
         "rate": float(r["fundingRate"])}
        for r in rows
    ])
    series = frame.drop_duplicates("ts").set_index("ts").sort_index()["rate"]
    series.name = symbol
    return series


def summarise_funding(rates: pd.Series) -> dict:
    """Descriptive statistics for a funding stream.

    The Sharpe reported here is for the funding stream ALONE. It is typically
    enormous (30+) and it is NOT the Sharpe of the carry trade, because it
    ignores basis risk, execution cost and liquidation risk. Quoting it as a
    strategy Sharpe would be dishonest.
    """
    clean = rates.dropna()
    if len(clean) < 2:
        return {}
    annualised = float(clean.mean()) * FUNDING_PERIODS_PER_YEAR
    return {
        "periods": len(clean),
        "span_days": (clean.index[-1] - clean.index[0]).days,
        "mean_bps": float(clean.mean()) * 1e4,
        "annualised_pct": annualised * 100.0,
        "pct_positive": float((clean > 0).mean()) * 100.0,
        "worst_period_bps": float(clean.min()) * 1e4,
        "funding_stream_sharpe": float(
            clean.mean() / clean.std() * np.sqrt(FUNDING_PERIODS_PER_YEAR)
        ) if clean.std() > 0 else 0.0,
    }


def carry_trade_pnl(
    funding: pd.Series,
    *,
    entry_cost_bps: float = 20.0,
    exit_cost_bps: float = 20.0,
    basis_vol_bps: float = 15.0,
    capital_efficiency: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate a delta-neutral cash-and-carry position over a funding history.

    Args:
        entry_cost_bps: Round-trip cost of establishing BOTH legs (spot buy and
            perp short). Two taker fees plus two spreads.
        exit_cost_bps: Same to unwind.
        basis_vol_bps: Per-period standard deviation of spot-perp basis moves.
            The position is delta-neutral but not variance-free; this is the
            noise the textbook description leaves out.
        capital_efficiency: Fraction of capital actually earning funding. You
            must post margin on the perp leg, so not all capital is deployed.
            0.5 is realistic for a conservative margin buffer that avoids
            liquidation.

    Returns a DataFrame with per-period and cumulative returns on capital.
    """
    clean = funding.dropna()
    if clean.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    basis_noise = rng.normal(0.0, basis_vol_bps * 1e-4, size=len(clean))

    gross = clean.to_numpy() * capital_efficiency
    net = gross + basis_noise
    net[0] -= entry_cost_bps * 1e-4
    net[-1] -= exit_cost_bps * 1e-4

    return pd.DataFrame(
        {
            "funding_rate": clean.to_numpy(),
            "gross_return": gross,
            "basis_noise": basis_noise,
            "net_return": net,
            "cumulative": (1.0 + net).cumprod(),
        },
        index=clean.index,
    )


def breakeven_holding_periods(
    mean_funding_rate: float,
    *,
    entry_cost_bps: float = 20.0,
    exit_cost_bps: float = 20.0,
    capital_efficiency: float = 0.5,
) -> float:
    """How many funding periods you must hold just to cover round-trip costs.

    The number that decides whether this trade is viable for you. If it exceeds
    what you can realistically hold, the trade is theoretical.
    """
    per_period = mean_funding_rate * capital_efficiency
    if per_period <= 0:
        return float("inf")
    return (entry_cost_bps + exit_cost_bps) * 1e-4 / per_period
