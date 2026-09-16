"""Engine tests. The look-ahead test is the most important one here:
if it ever fails, every backtest this repo produces is worthless.
"""

import numpy as np
import pandas as pd
import pytest

import trader.strategies  # noqa: F401
from trader.costs import get_venue
from trader.engine import run_backtest
from trader.strategy import Strategy, register


@register("_test_oracle")
class OracleStrategy(Strategy):
    """Deliberately cheats: holds only on the bar the spike occurs.

    The engine must NOT let it profit, because it acts on bar t's signal at
    bar t+1's open -- by which time the spike is over.
    """

    def target_exposure(self, bars: pd.DataFrame) -> pd.Series:
        exposure = pd.Series(0.0, index=bars.index)
        spike = bars["close"].idxmax()
        exposure.loc[spike] = 1.0
        return exposure


@register("_test_always_flat")
class AlwaysFlat(Strategy):
    def target_exposure(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=bars.index)


def test_lookahead_is_impossible(spike_bars):
    """A strategy that 'knows' the spike bar cannot capture the spike."""
    result = run_backtest(
        spike_bars, OracleStrategy(), get_venue("frictionless"),
        starting_cash=50.0, timeframe="1h",
    )
    # Signal at bar 150 executes at bar 151's open, which is back at 100.
    # Buying the top and selling flat must lose, never gain 10x.
    assert result.metrics.ending_equity <= 50.0 + 1e-6, (
        f"LOOK-AHEAD BIAS DETECTED: ended with "
        f"${result.metrics.ending_equity:.2f} from $50. The engine is letting "
        f"strategies trade on information they could not have had."
    )


def test_flat_strategy_never_trades_and_never_loses(flat_bars):
    result = run_backtest(
        flat_bars, AlwaysFlat(), get_venue("coinbase"), starting_cash=50.0, timeframe="1h"
    )
    assert result.metrics.num_fills == 0
    assert result.metrics.ending_equity == pytest.approx(50.0)
    assert result.metrics.total_fees == 0.0


def test_buy_and_hold_tracks_price_minus_costs(trending_bars):
    from trader.strategies.buy_and_hold import BuyAndHold

    result = run_backtest(
        trending_bars, BuyAndHold(), get_venue("frictionless"),
        starting_cash=50.0, timeframe="1h",
    )
    # Entered at bar 1's open, so the captured move is open[1] -> close[-1].
    expected = trending_bars["close"].iloc[-1] / trending_bars["open"].iloc[1]
    assert result.metrics.ending_equity == pytest.approx(50.0 * expected, rel=1e-6)


def test_fees_strictly_reduce_returns(trending_bars):
    from trader.strategy import get_strategy

    free = run_backtest(trending_bars, get_strategy("sma_cross"), get_venue("frictionless"),
                        starting_cash=50.0, timeframe="1h")
    paid = run_backtest(trending_bars, get_strategy("sma_cross"), get_venue("coinbase"),
                        starting_cash=50.0, timeframe="1h")
    if free.metrics.num_fills > 0:
        assert paid.metrics.ending_equity < free.metrics.ending_equity


def test_benchmark_pays_the_same_costs(trending_bars):
    """Buy-and-hold must be charged fees too, or the comparison is rigged."""
    from trader.strategy import get_strategy

    result = run_backtest(trending_bars, get_strategy("sma_cross"), get_venue("coinbase"),
                          starting_cash=50.0, timeframe="1h")
    assert result.benchmark_metrics.total_fees > 0.0


def test_exposure_outside_unit_interval_is_rejected(flat_bars):
    class Leveraged(Strategy):
        name = "leveraged"

        def target_exposure(self, bars):
            return pd.Series(2.0, index=bars.index)

    with pytest.raises(ValueError, match="leverage"):
        run_backtest(flat_bars, Leveraged(), get_venue("kraken"),
                     starting_cash=50.0, timeframe="1h")


def test_misaligned_exposure_is_rejected(flat_bars):
    class Misaligned(Strategy):
        name = "misaligned"

        def target_exposure(self, bars):
            return pd.Series(0.0, index=bars.index[:-5])

    with pytest.raises(ValueError, match="aligned"):
        run_backtest(flat_bars, Misaligned(), get_venue("kraken"),
                     starting_cash=50.0, timeframe="1h")
