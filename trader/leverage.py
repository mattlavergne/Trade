"""Position-size and leverage analysis: how much risk is too much.

"Take more risk to make more money" is true only up to a point, and false
after it. This module finds the point.

The mathematics, for log-compounded capital:

    g(L) = L*mu - (L*sigma)^2 / 2

Leverage multiplies arithmetic return LINEARLY but volatility drag
QUADRATICALLY. So growth rises, peaks, and then falls -- and past the peak you
are taking strictly more risk for strictly less money. The peak is the Kelly
criterion:

    L* = mu / sigma^2 = Sharpe / sigma        g(L*) = Sharpe^2 / 2

Two facts that matter more than the formula:

1. **At 2x Kelly, growth is exactly zero** with double the volatility. There is
   no consolation prize for overbetting.
2. **Kelly assumes you know mu and sigma.** You do not. Sharpe estimated over T
   years has standard error ~sqrt((1 + S^2/2) / T), which over a handful of
   years is enormous. Overestimating your edge by 2x means betting 2x Kelly,
   which means zero growth. This is why practitioners bet a FRACTION of Kelly --
   typically a quarter to a half -- and why an unproven edge implies an optimal
   leverage near zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 365


@dataclass
class LeverageProfile:
    """Growth and risk as a function of leverage, with honest error bars."""

    arithmetic_return: float
    volatility: float
    sharpe: float
    years: float
    kelly_leverage: float
    growth_at_kelly: float
    sharpe_stderr: float

    @property
    def sharpe_ci(self) -> tuple[float, float]:
        """95% confidence interval for the Sharpe ratio."""
        return (self.sharpe - 1.96 * self.sharpe_stderr,
                self.sharpe + 1.96 * self.sharpe_stderr)

    @property
    def kelly_ci(self) -> tuple[float, float]:
        low, high = self.sharpe_ci
        return (max(0.0, low / self.volatility), high / self.volatility)

    @property
    def edge_is_established(self) -> bool:
        """True only if the Sharpe is significantly above zero."""
        return self.sharpe_ci[0] > 0.0

    def growth(self, leverage: float | np.ndarray):
        """Expected log growth rate at a given leverage."""
        return leverage * self.arithmetic_return - (leverage * self.volatility) ** 2 / 2

    def recommended_leverage(self, kelly_fraction: float = 0.25) -> float:
        """Fractional Kelly, floored at zero when the edge is not established.

        An unproven edge means the growth-optimal bet is not merely small; its
        sign is unknown. Betting on it is a choice to gamble, not to invest.
        """
        if not self.edge_is_established:
            return 0.0
        return max(0.0, self.kelly_leverage * kelly_fraction)


def analyse(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> LeverageProfile:
    """Build a leverage profile from a return series."""
    clean = returns.dropna()
    if len(clean) < 30:
        raise ValueError("need at least 30 observations to estimate leverage safely")

    mu = float(clean.mean()) * periods_per_year
    sigma = float(clean.std(ddof=1)) * np.sqrt(periods_per_year)
    sharpe = mu / sigma if sigma > 0 else 0.0
    years = len(clean) / periods_per_year

    # Lo (2002), "The Statistics of Sharpe Ratios": asymptotic standard error
    # for i.i.d. returns. Real returns are autocorrelated, so the true error is
    # usually LARGER than this. It is a floor on your uncertainty, not a ceiling.
    stderr = float(np.sqrt((1.0 + sharpe**2 / 2.0) / years)) if years > 0 else float("inf")

    kelly = mu / sigma**2 if sigma > 0 else 0.0
    return LeverageProfile(
        arithmetic_return=mu,
        volatility=sigma,
        sharpe=sharpe,
        years=years,
        kelly_leverage=kelly,
        growth_at_kelly=sharpe**2 / 2.0,
        sharpe_stderr=stderr,
    )


def outcome_distribution(
    returns: pd.Series,
    leverage: float,
    *,
    horizon_days: int = 365,
    num_paths: int = 20_000,
    block_size: int = 20,
    ruin_threshold: float = 0.01,
    seed: int = 0,
) -> dict:
    """Block-bootstrap the distribution of outcomes at a given leverage.

    Returns probabilities that matter for a decision -- doubling, halving,
    near-total loss -- rather than a single expected value that hides them.

    Args:
        ruin_threshold: Equity fraction at or below which the account is
            treated as destroyed. Leveraged positions are liquidated well
            before reaching zero, so this models a realistic wipeout.
    """
    values = returns.dropna().to_numpy()
    n = len(values)
    if n < block_size * 2:
        raise ValueError("not enough data to bootstrap")

    rng = np.random.default_rng(seed)
    num_blocks = int(np.ceil(horizon_days / block_size))
    finals = np.empty(num_paths)

    for i in range(num_paths):
        starts = rng.integers(0, n, size=num_blocks)
        sample = np.concatenate([
            np.take(values, range(s, s + block_size), mode="wrap") for s in starts
        ])[:horizon_days]
        equity = np.cumprod(1.0 + leverage * sample)
        finals[i] = equity[-1] if (equity > ruin_threshold).all() else 0.0

    survived = finals > 0
    log_growth = np.where(survived, np.log(np.maximum(finals, 1e-12)), -20.0)
    return {
        "leverage": leverage,
        "median": float(np.median(finals)),
        "p_double": float((finals >= 2.0).mean()),
        "p_up": float((finals > 1.0).mean()),
        "p_lose_half": float((finals <= 0.5).mean()),
        "p_ruin": float((finals <= 0.1).mean()),
        "mean_log_growth": float(log_growth.mean()),
        "p5": float(np.percentile(finals, 5)),
        "p95": float(np.percentile(finals, 95)),
    }


def leverage_table(
    returns: pd.Series,
    levels: list[float] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Outcome distributions across a range of leverage levels."""
    levels = levels or [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    return pd.DataFrame([outcome_distribution(returns, L, **kwargs) for L in levels])


def zero_edge_control(returns: pd.Series) -> pd.Series:
    """The same series with its mean removed: identical risk, no edge.

    The control experiment. Run any leverage analysis against this to see what
    your results look like if the edge you measured was noise. If the answers
    are similar, your edge is not driving the outcome -- your risk-taking is.
    """
    clean = returns.dropna()
    return clean - clean.mean()
