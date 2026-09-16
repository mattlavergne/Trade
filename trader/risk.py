"""Risk management: position sizing, volatility targeting, and circuit breakers.

Risk management does not create edge. It does something arguably more valuable:
it stops a losing streak from becoming a terminal event, and it keeps the bet
size proportionate to the opportunity so that no single trade can define the
outcome.

Volatility targeting is the exception -- it is a genuine, well-documented
improvement to risk-adjusted returns (Moreira & Muir, "Volatility-Managed
Portfolios", Journal of Finance 2017). Scaling exposure inversely to recent
realised volatility raises Sharpe across essentially every asset class studied,
because volatility is strongly autocorrelated while returns are not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 365  # crypto trades every day


def realised_volatility(
    returns: pd.DataFrame | pd.Series,
    lookback: int = 30,
    *,
    annualise: bool = True,
) -> pd.DataFrame | pd.Series:
    """Rolling realised volatility, annualised by default."""
    vol = returns.rolling(lookback, min_periods=max(2, lookback // 2)).std()
    return vol * np.sqrt(TRADING_DAYS) if annualise else vol


def ewma_volatility(
    returns: pd.DataFrame | pd.Series,
    halflife: int = 20,
    *,
    annualise: bool = True,
) -> pd.DataFrame | pd.Series:
    """EWMA volatility. Reacts faster to regime shifts than a flat window."""
    vol = returns.ewm(halflife=halflife, min_periods=10).std()
    return vol * np.sqrt(TRADING_DAYS) if annualise else vol


@dataclass
class RiskConfig:
    """Risk limits applied to every rebalance.

    Attributes:
        target_volatility: Desired annualised portfolio volatility. 0.20 means
            the system sizes positions so the portfolio is expected to move
            about 20% per year.
        max_leverage: Hard cap on gross exposure as a multiple of equity. 1.0
            means never borrow. Vol targeting can demand leverage when markets
            are calm; this is what stops it.
        max_position_weight: Cap on any single asset's share of equity, so one
            name cannot dominate the book.
        vol_lookback: Days used to estimate volatility.
        no_trade_band: Skip rebalancing an asset whose weight is within this
            FRACTION of its target weight. 0.25 means "only trade when drift
            exceeds a quarter of the intended position". The single most
            effective cost reducer in the system -- without it, tiny daily
            drift generates constant turnover and fees eat the strategy alive.

            This is deliberately relative, not absolute. An absolute band
            silently prevents a diversified book from ever entering: with 14
            assets at a 20% vol target, individual weights are ~2%, so any
            absolute band above that blocks every trade forever.
        max_drawdown_stop: Flatten everything if drawdown from peak exceeds
            this. A circuit breaker, not an optimisation.
        drawdown_scale_start: Drawdown level at which to begin cutting exposure
            linearly, well before the hard stop.
    """

    target_volatility: float = 0.20
    max_leverage: float = 1.0
    max_position_weight: float = 0.35
    vol_lookback: int = 30
    no_trade_band: float = 0.25
    max_drawdown_stop: float = 0.35
    drawdown_scale_start: float = 0.15

    def __post_init__(self) -> None:
        if not 0 < self.target_volatility <= 3.0:
            raise ValueError("target_volatility must be in (0, 3]")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")
        if not 0 < self.max_position_weight <= 1.0:
            raise ValueError("max_position_weight must be in (0, 1]")
        if not 0 <= self.no_trade_band < 1.0:
            raise ValueError("no_trade_band must be in [0, 1)")
        if not 0 < self.drawdown_scale_start < self.max_drawdown_stop <= 1.0:
            raise ValueError("need 0 < drawdown_scale_start < max_drawdown_stop <= 1")


def volatility_scaled_weights(
    signals: pd.Series,
    asset_vol: pd.Series,
    config: RiskConfig,
) -> pd.Series:
    """Convert raw signals into portfolio weights at the target volatility.

    Each asset is sized inversely to its own volatility (so a calm asset and a
    wild one contribute comparable risk), then the whole book is scaled to hit
    the portfolio volatility target, then hard caps are applied.

    This is inverse-volatility weighting, which assumes correlations are roughly
    equal across assets. That assumption is wrong in a crisis, when everything
    correlates to 1 -- which is precisely why the drawdown circuit breaker
    exists as a separate, independent control.
    """
    signals = signals.fillna(0.0)
    vol = asset_vol.reindex(signals.index)

    # Assets with no usable vol estimate are untradeable, not zero-risk.
    usable = vol.notna() & (vol > 1e-8) & signals.notna()
    if not usable.any():
        return pd.Series(0.0, index=signals.index)

    raw = pd.Series(0.0, index=signals.index)
    raw[usable] = signals[usable] / vol[usable]

    gross = raw.abs().sum()
    if gross < 1e-12:
        return pd.Series(0.0, index=signals.index)

    # Normalise to unit gross, then scale so expected portfolio vol hits target.
    normalised = raw / gross
    portfolio_vol = (normalised.abs() * vol.fillna(0.0)).sum()
    scale = config.target_volatility / portfolio_vol if portfolio_vol > 1e-8 else 0.0
    weights = normalised * scale

    weights = weights.clip(-config.max_position_weight, config.max_position_weight)

    gross_exposure = weights.abs().sum()
    if gross_exposure > config.max_leverage:
        weights *= config.max_leverage / gross_exposure

    return weights


def drawdown_scalar(current_drawdown: float, config: RiskConfig) -> float:
    """Exposure multiplier based on current drawdown from peak.

    Full exposure until `drawdown_scale_start`, then linearly down to zero at
    `max_drawdown_stop`. Cutting risk while losing is the opposite of what a
    human does under stress, which is exactly why it belongs in code.

    Args:
        current_drawdown: Non-positive number, e.g. -0.20 for a 20% drawdown.
    """
    depth = abs(min(current_drawdown, 0.0))
    if depth <= config.drawdown_scale_start:
        return 1.0
    if depth >= config.max_drawdown_stop:
        return 0.0
    span = config.max_drawdown_stop - config.drawdown_scale_start
    return float(1.0 - (depth - config.drawdown_scale_start) / span)


def apply_no_trade_band(
    current: pd.Series, target: pd.Series, band: float, *, floor: float = 1e-4
) -> pd.Series:
    """Keep the current weight wherever drift is small relative to the target.

    An asset trades only when |target - current| exceeds `band` times the
    intended position size. Turnover is a certain, immediate cost; a small
    signal change is an uncertain benefit. Trading only on meaningful drift is
    about as close to free Sharpe as this business offers.

    Args:
        band: Drift threshold as a fraction of target position size.
        floor: Absolute weight below which a position is closed rather than
            maintained, so dust positions do not linger.
    """
    target = target.reindex(current.index).fillna(0.0)
    drift = (target - current).abs()

    # Reference size: the larger of where we are and where we want to be, so
    # both entries (current == 0) and exits (target == 0) can clear the band.
    reference = pd.concat([target.abs(), current.abs()], axis=1).max(axis=1)
    threshold = (band * reference).clip(lower=floor)

    updated = target.where(drift > threshold, current)
    # Always honour a full exit; never leave dust behind.
    return updated.where(~((target.abs() < floor) & (drift > floor)), 0.0)


# ---------------------------------------------------------------------------
# NumPy fast paths
#
# The pandas implementations above are the readable reference. Walk-forward
# analysis runs thousands of backtests, and pandas indexing inside a bar loop
# dominates the runtime, so the hot loop uses these instead. They must stay
# behaviourally identical -- tests/test_risk_fastpath.py asserts exactly that.
# ---------------------------------------------------------------------------


def volatility_scaled_weights_np(
    signals: np.ndarray,
    asset_vol: np.ndarray,
    target_volatility: float,
    max_leverage: float,
    max_position_weight: float,
) -> np.ndarray:
    """NumPy equivalent of `volatility_scaled_weights`."""
    signals = np.nan_to_num(signals, nan=0.0)
    usable = np.isfinite(asset_vol) & (asset_vol > 1e-8)

    raw = np.zeros_like(signals, dtype=float)
    raw[usable] = signals[usable] / asset_vol[usable]

    gross = np.abs(raw).sum()
    if gross < 1e-12:
        return np.zeros_like(signals, dtype=float)

    normalised = raw / gross
    portfolio_vol = float((np.abs(normalised) * np.nan_to_num(asset_vol)).sum())
    scale = target_volatility / portfolio_vol if portfolio_vol > 1e-8 else 0.0

    weights = np.clip(normalised * scale, -max_position_weight, max_position_weight)

    gross_exposure = np.abs(weights).sum()
    if gross_exposure > max_leverage:
        weights = weights * (max_leverage / gross_exposure)
    return weights


def apply_no_trade_band_np(
    current: np.ndarray, target: np.ndarray, band: float, floor: float = 1e-4
) -> np.ndarray:
    """NumPy equivalent of `apply_no_trade_band`."""
    target = np.nan_to_num(target, nan=0.0)
    drift = np.abs(target - current)
    reference = np.maximum(np.abs(target), np.abs(current))
    threshold = np.maximum(band * reference, floor)

    updated = np.where(drift > threshold, target, current)
    return np.where((np.abs(target) < floor) & (drift > floor), 0.0, updated)
