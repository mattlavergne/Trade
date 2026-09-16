"""Cross-sectional strategies with documented academic support.

Each of these corresponds to a return premium that has survived out-of-sample
scrutiny across decades and asset classes. That is not a promise they will work
now -- published anomalies decay, sometimes to nothing, once they are known. It
means they start from evidence rather than from a curve fit.

References:
  Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum", JFE.
  Asness, Moskowitz & Pedersen (2013), "Value and Momentum Everywhere", JF.
  Moreira & Muir (2017), "Volatility-Managed Portfolios", JF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..portfolio import PortfolioStrategy
from ..universe import Panel


class TimeSeriesMomentum(PortfolioStrategy):
    """Long assets whose own recent return is positive; flat or short otherwise.

    The most robust documented anomaly in systematic trading, verified across
    more than a century of data and dozens of markets. It also has brutal
    multi-year drawdowns, which is precisely why it has not been arbitraged
    away: most people cannot sit through them.
    """

    name = "tsmom"

    def __init__(self, lookback: int = 90, vol_scale: bool = True, allow_short: bool = False) -> None:
        if lookback < 5:
            raise ValueError("lookback must be >= 5")
        self.lookback = int(lookback)
        self.vol_scale = bool(vol_scale)
        self.allow_short = bool(allow_short)

    def signals(self, panel: Panel) -> pd.DataFrame:
        close = panel.close
        past_return = close.pct_change(self.lookback)

        if self.vol_scale:
            # Scale conviction by return / volatility, so a modest move in a
            # calm asset counts as much as a large move in a wild one. Then
            # squash to [-1, 1] -- conviction, never size.
            vol = panel.returns().rolling(self.lookback, min_periods=self.lookback // 2).std()
            signal = np.tanh(past_return / (vol * np.sqrt(self.lookback) + 1e-9))
        else:
            signal = np.sign(past_return)

        signal = pd.DataFrame(signal, index=close.index, columns=close.columns)
        if not self.allow_short:
            signal = signal.clip(lower=0.0)
        return signal.clip(-1.0, 1.0).fillna(0.0)


class CrossSectionalMomentum(PortfolioStrategy):
    """Long the strongest assets in the cross-section, short the weakest.

    Differs from time-series momentum in being relative: it holds the best
    available assets even in a falling market, which makes it roughly
    market-neutral and a genuinely different return stream.
    """

    name = "xsmom"

    def __init__(self, lookback: int = 60, top_n: int = 4, allow_short: bool = False) -> None:
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        self.lookback = int(lookback)
        self.top_n = int(top_n)
        self.allow_short = bool(allow_short)

    def signals(self, panel: Panel) -> pd.DataFrame:
        past_return = panel.close.pct_change(self.lookback)
        ranks = past_return.rank(axis=1, ascending=False, na_option="keep")
        num_assets = past_return.notna().sum(axis=1)

        long_leg = (ranks <= self.top_n).astype(float)
        if not self.allow_short:
            return long_leg.where(num_assets.gt(self.top_n), 0.0).fillna(0.0)

        short_leg = ranks.gt(num_assets.values[:, None] - self.top_n).astype(float)
        combined = long_leg - short_leg
        return combined.where(num_assets.gt(2 * self.top_n), 0.0).fillna(0.0)


class TrendFilteredMomentum(PortfolioStrategy):
    """Cross-sectional momentum, but only while the asset is above its own
    long-term trend.

    A regime filter. Cross-sectional momentum picks the best horse in the race;
    this refuses to bet when the whole field is running backwards. Combining a
    relative signal with an absolute one is standard practice in managed futures
    and materially reduces bear-market drawdowns.
    """

    name = "trend_filtered"

    def __init__(self, lookback: int = 60, top_n: int = 4, trend_window: int = 200) -> None:
        self.lookback = int(lookback)
        self.top_n = int(top_n)
        self.trend_window = int(trend_window)

    def signals(self, panel: Panel) -> pd.DataFrame:
        close = panel.close
        base = CrossSectionalMomentum(
            lookback=self.lookback, top_n=self.top_n, allow_short=False
        ).signals(panel)
        trend = close > close.rolling(self.trend_window, min_periods=self.trend_window // 2).mean()
        return (base * trend.astype(float)).fillna(0.0)


class Ensemble(PortfolioStrategy):
    """Average the signals of several strategies.

    Diversification across *signals* works the same way as diversification
    across assets: if the components are imperfectly correlated, the blend has a
    higher Sharpe than its average member. It also removes the temptation to
    pick the one that happened to do best in the backtest -- which is the
    selection bias the Deflated Sharpe exists to punish.
    """

    name = "ensemble"

    def __init__(self, components: list[PortfolioStrategy] | None = None,
                 weights: list[float] | None = None) -> None:
        self.components = components or [
            TimeSeriesMomentum(lookback=90),
            CrossSectionalMomentum(lookback=60, top_n=4),
            TrendFilteredMomentum(lookback=60, top_n=4, trend_window=200),
        ]
        if weights is not None and len(weights) != len(self.components):
            raise ValueError("weights must match number of components")
        self.weights = weights or [1.0 / len(self.components)] * len(self.components)

    def signals(self, panel: Panel) -> pd.DataFrame:
        total = None
        for strategy, weight in zip(self.components, self.weights):
            contribution = strategy.signals(panel) * weight
            total = contribution if total is None else total.add(contribution, fill_value=0.0)
        return total.clip(-1.0, 1.0).fillna(0.0)

    def describe(self) -> str:
        parts = ", ".join(s.name for s in self.components)
        return f"ensemble({parts})"


class EqualWeightHold(PortfolioStrategy):
    """Equal-weight buy and hold. The benchmark that must be beaten."""

    name = "hold"

    def signals(self, panel: Panel) -> pd.DataFrame:
        return pd.DataFrame(1.0, index=panel.index, columns=panel.symbols)


PORTFOLIO_STRATEGIES = {
    "tsmom": TimeSeriesMomentum,
    "xsmom": CrossSectionalMomentum,
    "trend_filtered": TrendFilteredMomentum,
    "ensemble": Ensemble,
    "hold": EqualWeightHold,
}
