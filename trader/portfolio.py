"""Multi-asset portfolio backtest engine.

Differences from the single-asset engine, all of them material:

- Trades a cross-section of assets, so results reflect diversification rather
  than one market's luck.
- Sizes positions by volatility targeting instead of all-in/all-out.
- Applies a no-trade band so small drift does not generate constant turnover.
- Scales exposure down during drawdowns and halts at a hard limit.
- Charges costs on turnover per asset, every rebalance.

The no-look-ahead rule is unchanged and non-negotiable: signals computed from
bars through day t are executed at day t+1's open.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import BPS, VenueCosts
from .risk import RiskConfig, drawdown_scalar, ewma_volatility
from .universe import Panel

log = logging.getLogger(__name__)


class PortfolioStrategy:
    """Base class for cross-sectional strategies.

    `signals` returns a DataFrame (date x symbol) of desired exposures, where
    +1 is maximum long conviction, 0 is flat, -1 is maximum short. The portfolio
    engine converts conviction into position size via volatility targeting; a
    strategy should express *direction and confidence*, never position size.
    """

    name: str = "unnamed"

    def signals(self, panel: Panel) -> pd.DataFrame:
        raise NotImplementedError

    def describe(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in sorted(vars(self).items()))
        return f"{self.name}({params})"


@dataclass
class PortfolioResult:
    strategy: str
    equity_curve: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    gross_equity_curve: pd.Series
    benchmark_curve: pd.Series
    exposure: pd.Series
    config: RiskConfig
    venue: str
    metadata: dict = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        return float(self.costs.sum())

    @property
    def total_return_pct(self) -> float:
        return float(self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1.0) * 100.0

    @property
    def benchmark_return_pct(self) -> float:
        return float(self.benchmark_curve.iloc[-1] / self.benchmark_curve.iloc[0] - 1.0) * 100.0

    def daily_returns(self) -> pd.Series:
        return self.equity_curve.pct_change().dropna()


def run_portfolio_backtest(
    panel: Panel,
    strategy: PortfolioStrategy,
    venue: VenueCosts,
    *,
    config: RiskConfig | None = None,
    starting_cash: float = 10_000.0,
    warmup: int = 100,
) -> PortfolioResult:
    """Run a cross-sectional strategy over a price panel.

    Args:
        warmup: Bars skipped before trading, so rolling estimates are populated.
            Trading during warmup uses half-formed statistics and inflates
            results.
    """
    config = config or RiskConfig()
    signals = strategy.signals(panel)
    if not signals.index.equals(panel.index):
        raise ValueError("strategy signals must align to the panel index")
    if signals.abs().to_numpy()[~np.isnan(signals.to_numpy())].max(initial=0.0) > 1.0 + 1e-9:
        raise ValueError("signals must lie in [-1, 1]; size is the engine's job, not the strategy's")

    returns = panel.returns()
    asset_vol = ewma_volatility(returns, halflife=config.vol_lookback)
    tradeable = panel.tradeable()

    opens = panel.open
    closes = panel.close
    index = panel.index
    symbols = panel.symbols

    # Hot loop runs on numpy: walk-forward executes thousands of backtests and
    # pandas indexing per bar dominates the runtime otherwise.
    from .risk import apply_no_trade_band_np, volatility_scaled_weights_np

    returns_np = returns.fillna(0.0).to_numpy()
    signals_np = np.nan_to_num(signals.to_numpy(), nan=0.0)
    vol_np = asset_vol.to_numpy()
    tradeable_np = tradeable.to_numpy()
    num_symbols = len(symbols)

    equity = starting_cash
    gross_equity = starting_cash  # same path with costs switched off
    peak = starting_cash
    current = np.zeros(num_symbols, dtype=float)

    cost_rate = (venue.fee_bps(is_taker=True) + venue.half_spread_bps) * BPS
    n = len(index)

    equity_path = np.empty(n)
    gross_path = np.empty(n)
    weight_matrix = np.zeros((n, num_symbols))
    turnover_path = np.zeros(n)
    cost_path = np.zeros(n)
    exposure_path = np.zeros(n)

    for i in range(n):
        if i <= warmup:
            equity_path[i] = equity
            gross_path[i] = gross_equity
            continue

        # --- mark the existing book to today's move ------------------------
        portfolio_return = float(current @ returns_np[i])
        equity *= 1.0 + portfolio_return
        gross_equity *= 1.0 + portfolio_return
        peak = max(peak, equity)

        # --- decide the new target book ------------------------------------
        # Signal from bar i-1 (yesterday's close), executed today. One full bar
        # of latency, exactly as in the single-asset engine.
        raw_signal = signals_np[i - 1].copy()
        raw_signal[~tradeable_np[i]] = 0.0

        target = volatility_scaled_weights_np(
            raw_signal, vol_np[i - 1], config.target_volatility,
            config.max_leverage, config.max_position_weight,
        )

        drawdown = equity / peak - 1.0
        target = target * drawdown_scalar(drawdown, config)

        banded = apply_no_trade_band_np(current, target, config.no_trade_band)

        # --- charge costs on turnover --------------------------------------
        turnover = float(np.abs(banded - current).sum())
        cost = equity * turnover * cost_rate
        equity -= cost

        current = banded
        equity_path[i] = equity
        gross_path[i] = gross_equity
        weight_matrix[i] = current
        turnover_path[i] = turnover
        cost_path[i] = cost
        exposure_path[i] = float(np.abs(current).sum())

    equity_curve = pd.Series(equity_path, index=index, name="equity")
    gross_curve = pd.Series(gross_path, index=index, name="gross_equity")

    # Benchmark: equal-weight buy and hold of the whole universe, charged one
    # round trip so the comparison is not rigged in the strategy's favour.
    equal_weight_returns = returns.mean(axis=1).fillna(0.0)
    equal_weight_returns.iloc[: warmup + 1] = 0.0
    entry_cost = (venue.fee_bps(is_taker=True) + venue.half_spread_bps) * BPS
    benchmark = starting_cash * (1.0 - entry_cost) * (1.0 + equal_weight_returns).cumprod()

    return PortfolioResult(
        strategy=strategy.describe(),
        equity_curve=equity_curve,
        weights=pd.DataFrame(weight_matrix, index=index, columns=symbols),
        turnover=pd.Series(turnover_path, index=index),
        costs=pd.Series(cost_path, index=index),
        gross_equity_curve=gross_curve,
        benchmark_curve=benchmark,
        exposure=pd.Series(exposure_path, index=index),
        config=config,
        venue=venue.name,
        metadata={"warmup": warmup, "symbols": symbols, "starting_cash": starting_cash},
    )
