"""Buy and hold: the benchmark every active strategy must beat.

If a strategy cannot beat buying the asset once and sitting on it, it is
generating trades, fees and stress in exchange for nothing. This is the most
commonly skipped comparison in amateur backtests and the most damning one.
"""

from __future__ import annotations

import pandas as pd

from ..strategy import Strategy, register


@register("buy_and_hold")
class BuyAndHold(Strategy):
    def target_exposure(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=bars.index)
