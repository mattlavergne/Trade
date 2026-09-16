"""Z-score mean reversion: buy when price is unusually far below its mean.

Goes long when the close is `entry_z` standard deviations below its rolling
mean and exits once it recovers to `exit_z`. Mean reversion trades frequently,
which makes it the strategy most savagely punished by fees -- exactly the
lesson worth seeing in the numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategy import Strategy, register


@register("mean_reversion")
class MeanReversion(Strategy):
    def __init__(self, lookback: int = 48, entry_z: float = -1.5, exit_z: float = -0.2) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        if entry_z >= exit_z:
            raise ValueError(f"entry_z ({entry_z}) must be < exit_z ({exit_z})")
        self.lookback = int(lookback)
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)

    def target_exposure(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"]
        mean = close.rolling(self.lookback, min_periods=self.lookback).mean()
        std = close.rolling(self.lookback, min_periods=self.lookback).std()
        z = (close - mean) / std.replace(0.0, np.nan)

        # Stateful entry/exit: hold the position until z recovers past exit_z.
        exposure = np.zeros(len(close), dtype=float)
        holding = False
        for i, value in enumerate(z.to_numpy()):
            if np.isnan(value):
                holding = False
            elif holding:
                holding = value < self.exit_z
            else:
                holding = value <= self.entry_z
            exposure[i] = 1.0 if holding else 0.0

        return self._clip(pd.Series(exposure, index=bars.index))
