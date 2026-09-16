"""Risk controls must bind. A limit that can be exceeded is not a limit."""

import numpy as np
import pandas as pd
import pytest

from trader.risk import (
    RiskConfig,
    apply_no_trade_band,
    drawdown_scalar,
    ewma_volatility,
    realised_volatility,
    volatility_scaled_weights,
)


def _signals(values):
    return pd.Series(values, index=[f"A{i}" for i in range(len(values))], dtype=float)


def test_max_leverage_is_never_exceeded():
    signals = _signals([1.0] * 10)
    vol = pd.Series(0.01, index=signals.index)  # very calm -> wants huge size
    cfg = RiskConfig(target_volatility=0.50, max_leverage=1.0)
    w = volatility_scaled_weights(signals, vol, cfg)
    assert w.abs().sum() <= cfg.max_leverage + 1e-9


def test_single_position_cap_binds():
    signals = _signals([1.0, 0.01, 0.01])
    vol = pd.Series([0.05, 2.0, 2.0], index=signals.index)
    cfg = RiskConfig(max_position_weight=0.20)
    w = volatility_scaled_weights(signals, vol, cfg)
    assert w.abs().max() <= 0.20 + 1e-9


def test_low_vol_assets_get_larger_positions():
    """Inverse-vol sizing: the calm asset should carry more weight."""
    signals = _signals([1.0, 1.0])
    vol = pd.Series([0.10, 1.00], index=signals.index)
    w = volatility_scaled_weights(signals, vol, RiskConfig())
    assert w.iloc[0] > w.iloc[1]


def test_assets_without_vol_estimate_get_zero_weight():
    signals = _signals([1.0, 1.0])
    vol = pd.Series([0.5, np.nan], index=signals.index)
    w = volatility_scaled_weights(signals, vol, RiskConfig())
    assert w.iloc[1] == 0.0


def test_zero_signals_produce_zero_weights():
    signals = _signals([0.0, 0.0, 0.0])
    vol = pd.Series(0.5, index=signals.index)
    assert volatility_scaled_weights(signals, vol, RiskConfig()).abs().sum() == 0.0


def test_drawdown_scalar_is_monotonic_and_bounded():
    cfg = RiskConfig()
    depths = [0.0, -0.05, -0.15, -0.20, -0.25, -0.30, -0.35, -0.60]
    values = [drawdown_scalar(d, cfg) for d in depths]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert values == sorted(values, reverse=True), "exposure must never rise as drawdown deepens"
    assert values[0] == 1.0
    assert values[-1] == 0.0


def test_hard_drawdown_stop_flattens_completely():
    cfg = RiskConfig(max_drawdown_stop=0.30, drawdown_scale_start=0.10)
    assert drawdown_scalar(-0.31, cfg) == 0.0


def test_no_trade_band_allows_entry_from_zero():
    """Regression: an absolute band silently blocked all entries."""
    current = pd.Series({"A": 0.0})
    target = pd.Series({"A": 0.02})
    assert apply_no_trade_band(current, target, 0.25)["A"] == pytest.approx(0.02)


def test_no_trade_band_suppresses_small_drift():
    current = pd.Series({"A": 0.100})
    target = pd.Series({"A": 0.105})
    assert apply_no_trade_band(current, target, 0.25)["A"] == pytest.approx(0.100)


def test_no_trade_band_honours_full_exit():
    current = pd.Series({"A": 0.10})
    target = pd.Series({"A": 0.0})
    assert apply_no_trade_band(current, target, 0.25)["A"] == 0.0


def test_wider_band_never_increases_turnover():
    rng = np.random.default_rng(0)
    current = pd.Series(rng.normal(0, 0.05, 50))
    target = pd.Series(rng.normal(0, 0.05, 50))
    turnovers = [
        (apply_no_trade_band(current, target, b) - current).abs().sum()
        for b in (0.0, 0.25, 0.5, 0.9)
    ]
    assert turnovers == sorted(turnovers, reverse=True)


def test_invalid_risk_configs_are_rejected():
    with pytest.raises(ValueError):
        RiskConfig(target_volatility=0.0)
    with pytest.raises(ValueError):
        RiskConfig(max_leverage=-1.0)
    with pytest.raises(ValueError):
        RiskConfig(drawdown_scale_start=0.5, max_drawdown_stop=0.3)


def test_volatility_estimators_scale_with_noise():
    rng = np.random.default_rng(0)
    calm = pd.Series(rng.normal(0, 0.001, 300))
    wild = pd.Series(rng.normal(0, 0.050, 300))
    assert realised_volatility(calm).iloc[-1] < realised_volatility(wild).iloc[-1]
    assert ewma_volatility(calm).iloc[-1] < ewma_volatility(wild).iloc[-1]
