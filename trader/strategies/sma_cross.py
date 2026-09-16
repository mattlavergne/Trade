"""Simple moving-average crossover: long while fast SMA > slow SMA.

The canonical beginner trend strategy. Included because it is the thing most
people build first, and because watching it lose to fees is instructive.
"""

from __future__ import annotations

import pandas as pd

from ..strategy import Strategy, register


@register("sma_cross")
class SmaCross(Strategy):
    def __init__(self, fast: int = 20, slow: int = 50) -> None:
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")
        self.fast = int(fast)
        self.slow = int(slow)

    def target_exposure(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"]
        fast = close.rolling(self.fast, min_periods=self.fast).mean()
        slow = close.rolling(self.slow, min_periods=self.slow).mean()
        return self._clip((fast > slow).astype(float))
