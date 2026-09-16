"""The numpy fast paths must be behaviourally identical to the pandas reference.

The portfolio engine's hot loop uses the numpy versions for speed. If they ever
drift from the readable pandas implementations that the docs and other tests
describe, results silently stop meaning what they claim to mean.
"""

import numpy as np
import pandas as pd
import pytest

from trader.risk import (
    RiskConfig,
    apply_no_trade_band,
    apply_no_trade_band_np,
    volatility_scaled_weights,
    volatility_scaled_weights_np,
)


@pytest.mark.parametrize("seed", range(12))
def test_vol_scaled_weights_match(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 20))
    names = [f"S{i}" for i in range(n)]
    signals = pd.Series(rng.uniform(-1, 1, n), index=names)
    vol = pd.Series(np.abs(rng.normal(0.5, 0.3, n)), index=names)
    if seed % 3 == 0:
        vol.iloc[0] = np.nan          # missing volatility estimate
    if seed % 4 == 0:
        vol.iloc[-1] = 0.0            # degenerate volatility

    cfg = RiskConfig(
        target_volatility=float(rng.uniform(0.05, 0.8)),
        max_leverage=float(rng.uniform(0.5, 2.0)),
        max_position_weight=float(rng.uniform(0.1, 1.0)),
    )

    reference = volatility_scaled_weights(signals, vol, cfg).to_numpy()
    fast = volatility_scaled_weights_np(
        signals.to_numpy(), vol.to_numpy(),
        cfg.target_volatility, cfg.max_leverage, cfg.max_position_weight,
    )
    np.testing.assert_allclose(reference, fast, atol=1e-12)


@pytest.mark.parametrize("seed", range(12))
def test_no_trade_band_matches(seed):
    rng = np.random.default_rng(seed + 100)
    n = int(rng.integers(3, 25))
    names = [f"S{i}" for i in range(n)]
    current = pd.Series(rng.normal(0, 0.08, n), index=names)
    target = pd.Series(rng.normal(0, 0.08, n), index=names)
    if seed % 2 == 0:
        target.iloc[0] = 0.0          # full exit
        current.iloc[1] = 0.0         # fresh entry
    band = float(rng.uniform(0.0, 0.9))

    reference = apply_no_trade_band(current, target, band).to_numpy()
    fast = apply_no_trade_band_np(current.to_numpy(), target.to_numpy(), band)
    np.testing.assert_allclose(reference, fast, atol=1e-12)


def test_fast_path_handles_all_zero_signals():
    zeros = np.zeros(5)
    vol = np.full(5, 0.5)
    assert np.abs(volatility_scaled_weights_np(zeros, vol, 0.2, 1.0, 0.3)).sum() == 0.0


def test_fast_path_respects_leverage_cap():
    signals = np.ones(10)
    vol = np.full(10, 0.01)
    weights = volatility_scaled_weights_np(signals, vol, 5.0, 1.0, 1.0)
    assert np.abs(weights).sum() <= 1.0 + 1e-12


def test_fast_path_respects_position_cap():
    signals = np.array([1.0, 0.01, 0.01])
    vol = np.array([0.05, 2.0, 2.0])
    weights = volatility_scaled_weights_np(signals, vol, 0.3, 1.0, 0.2)
    assert np.abs(weights).max() <= 0.2 + 1e-12
