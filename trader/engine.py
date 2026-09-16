"""Backtest engine.

The execution model, stated plainly so it can be checked:

    bar t:   strategy sees OHLCV up to and including bar t, emits exposure e_t
    bar t+1: broker rebalances toward e_t at bar t+1's OPEN price

There is exactly one bar of latency between deciding and trading. This is
conservative but defensible: it is what you get if your bot wakes up on each
bar close and sends a market order. Anything faster requires you to prove your
infrastructure can do it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .broker import PaperBroker
from .costs import VenueCosts
from .metrics import Metrics, compute
from .strategy import Strategy

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    timeframe: str
    venue: str
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    exposure: pd.Series
    metrics: Metrics
    benchmark_metrics: Metrics
    broker: PaperBroker

    @property
    def beat_benchmark(self) -> bool:
        return self.metrics.total_return_pct > self.benchmark_metrics.total_return_pct

    @property
    def excess_return_pct(self) -> float:
        return self.metrics.total_return_pct - self.benchmark_metrics.total_return_pct


def _run_curve(
    bars: pd.DataFrame,
    exposure: pd.Series,
    broker: PaperBroker,
) -> pd.Series:
    """Walk the bars, rebalancing to the previous bar's target exposure."""
    opens = bars["open"].to_numpy()
    closes = bars["close"].to_numpy()
    index = bars.index
    targets = exposure.to_numpy()

    equity = []
    for i in range(len(bars)):
        if i > 0:
            # Act on bar i-1's signal at bar i's open. This ordering is the
            # no-look-ahead guarantee; do not "optimise" it away.
            target_fraction = targets[i - 1]
            price = opens[i]
            desired_value = broker.equity(price) * target_fraction
            broker.target_position_value(index[i], desired_value, price)
        equity.append(broker.equity(closes[i]))

    return pd.Series(equity, index=index, name="equity")


def run_backtest(
    bars: pd.DataFrame,
    strategy: Strategy,
    venue: VenueCosts,
    *,
    starting_cash: float = 50.0,
    timeframe: str = "1h",
    symbol: str = "BTC/USD",
    is_taker: bool = True,
) -> BacktestResult:
    """Run `strategy` over `bars` and compare it to buying and holding.

    The benchmark is run through the same broker and cost model, so the
    comparison is apples to apples: buy-and-hold pays entry and exit fees too.
    """
    if len(bars) < 2:
        raise ValueError("need at least 2 bars to backtest")

    from .strategies.buy_and_hold import BuyAndHold

    exposure = strategy.target_exposure(bars)
    if not exposure.index.equals(bars.index):
        raise ValueError("strategy returned exposure not aligned to the bar index")
    if exposure.max() > 1.0 + 1e-9 or exposure.min() < -1e-9:
        raise ValueError("exposure must lie in [0, 1]; this harness does not model leverage")

    broker = PaperBroker(starting_cash=starting_cash, venue=venue, is_taker=is_taker)
    equity_curve = _run_curve(bars, exposure, broker)

    bench_broker = PaperBroker(starting_cash=starting_cash, venue=venue, is_taker=is_taker)
    bench_exposure = BuyAndHold().target_exposure(bars)
    benchmark_curve = _run_curve(bars, bench_exposure, bench_broker)

    metrics = compute(
        equity_curve, broker.fills, broker.rejections,
        timeframe=timeframe, total_fees=broker.total_fees,
        total_slippage=broker.total_slippage, exposure=exposure,
    )
    bench_metrics = compute(
        benchmark_curve, bench_broker.fills, bench_broker.rejections,
        timeframe=timeframe, total_fees=bench_broker.total_fees,
        total_slippage=bench_broker.total_slippage, exposure=bench_exposure,
    )

    return BacktestResult(
        strategy=strategy.describe(),
        symbol=symbol,
        timeframe=timeframe,
        venue=venue.name,
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        exposure=exposure,
        metrics=metrics,
        benchmark_metrics=bench_metrics,
        broker=broker,
    )
