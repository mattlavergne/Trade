"""Strategies must not peek at the future and must stay within [0, 1]."""

import pandas as pd
import pytest

import trader.strategies  # noqa: F401
from trader.strategy import available_strategies, get_strategy

BUILT_IN = ["buy_and_hold", "sma_cross", "mean_reversion", "donchian"]


@pytest.mark.parametrize("name", BUILT_IN)
def test_exposure_is_bounded_and_aligned(name, trending_bars):
    exposure = get_strategy(name).target_exposure(trending_bars)
    assert exposure.index.equals(trending_bars.index)
    assert exposure.min() >= 0.0 and exposure.max() <= 1.0
    assert not exposure.isna().any()


@pytest.mark.parametrize("name", BUILT_IN)
def test_exposure_depends_only_on_the_past(name, trending_bars):
    """Truncating future bars must not change any past exposure value.

    This is the causality check. If a strategy's signal at bar 100 changes
    when bars 101+ are removed, it was reading the future.
    """
    strategy = get_strategy(name)
    full = strategy.target_exposure(trending_bars)
    cut = 150
    truncated = get_strategy(name).target_exposure(trending_bars.iloc[:cut])
    pd.testing.assert_series_equal(
        full.iloc[:cut], truncated, check_names=False,
        obj=f"{name} exposure changed when future bars were removed",
    )


def test_registry_contains_all_built_ins():
    assert set(BUILT_IN).issubset(set(available_strategies()))


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        get_strategy("sma_cross", fast=50, slow=20)
    with pytest.raises(ValueError):
        get_strategy("mean_reversion", entry_z=1.0, exit_z=-1.0)


def test_unknown_strategy_raises():
    with pytest.raises(KeyError, match="Unknown strategy"):
        get_strategy("get_rich_quick")
