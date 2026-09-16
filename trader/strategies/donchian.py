"""Donchian channel breakout: long on a new N-bar high, flat on a new N-bar low.

A classic trend-following rule (the Turtle system's core). Trades rarely, which
makes it one of the few shapes of strategy that a fee-heavy small account can
plausibly afford to run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategy import Strategy, register


@register("donchian")
class Donchian(Strategy):
    def __init__(self, entry_lookback: int = 55, exit_lookback: int = 20) -> None:
        if entry_lookback < 2 or exit_lookback < 2:
            raise ValueError("lookbacks must be >= 2")
        self.entry_lookback = int(entry_lookback)
        self.exit_lookback = int(exit_lookback)

    def target_exposure(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"]
        # shift(1) so the channel is built from *prior* bars only; comparing
        # today's close against a high that includes today is look-ahead bias.
        upper = bars["high"].rolling(self.entry_lookback, min_periods=self.entry_lookback).max().shift(1)
        lower = bars["low"].rolling(self.exit_lookback, min_periods=self.exit_lookback).min().shift(1)

        exposure = np.zeros(len(close), dtype=float)
        holding = False
        closes, ups, lows = close.to_numpy(), upper.to_numpy(), lower.to_numpy()
        for i in range(len(closes)):
            if np.isnan(ups[i]) or np.isnan(lows[i]):
                holding = False
            elif holding:
                holding = closes[i] > lows[i]
            else:
                holding = closes[i] > ups[i]
            exposure[i] = 1.0 if holding else 0.0

        return self._clip(pd.Series(exposure, index=bars.index))
