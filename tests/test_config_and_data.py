"""Configuration and offline market-data safeguards."""

from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_risk_decision_engine.config import StressWindow, load_config
from portfolio_risk_decision_engine.market_data import (
    load_processed,
    normalize_yfinance_close,
    prices_to_returns,
    save_processed,
    validate_prices,
)


def test_config_rejects_weight_sum_and_duplicate_ticker(tmp_path):
    source = tmp_path / "portfolio.json"
    source.write_text(
        (Path(__file__).parents[1] / "configs/portfolio.json").read_text()
    )
    config = load_config(source)
    assert np.isclose(config.portfolio_vector().sum(), 1)
    text = source.read_text().replace('"QQQ": 0.25', '"QQQ": 0.20')
    source.write_text(text)
    with pytest.raises(ValueError, match="sum to 1"):
        load_config(source)
    source.write_text(text.replace('"QQQ": 0.20', '"QQQ": 0.25, "SPY": 0.50'))
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_config(source)


def test_config_rejects_invalid_dates(tmp_path):
    source = Path(__file__).parents[1] / "configs/portfolio.json"
    target = tmp_path / "bad.json"
    target.write_text(
        source.read_text().replace(
            '"end_exclusive": "2026-01-01"', '"end_exclusive": "2017-01-01"'
        )
    )
    with pytest.raises(ValueError, match="precede"):
        load_config(target)


@pytest.fixture
def short_config():
    source = Path(__file__).parents[1] / "configs/portfolio.json"
    config = load_config(source)
    return replace(
        config,
        start=date(2024, 1, 1),
        end_exclusive=date(2024, 1, 18),
        minimum_price_observations=3,
        stress_windows=(StressWindow("test", date(2024, 1, 2), date(2024, 1, 5)),),
        walk_forward_start=date(2024, 1, 16),
        lookback_days=5,
        rebalance_every_days=3,
    )


def test_yfinance_column_orders_and_missing_data(short_config):
    dates = pd.bdate_range("2024-01-02", periods=10)
    tickers = short_config.tickers
    raw = pd.DataFrame(
        np.arange(10 * len(tickers), dtype=float).reshape(10, len(tickers)) + 100,
        index=dates,
        columns=pd.MultiIndex.from_product([["Close"], tickers]),
    )
    expected = normalize_yfinance_close(raw, tickers)
    swapped = raw.copy()
    swapped.columns = swapped.columns.swaplevel(0, 1)
    pd.testing.assert_frame_equal(normalize_yfinance_close(swapped, tickers), expected)
    validate_prices(expected, short_config)
    broken = expected.copy()
    broken.iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        validate_prices(broken, short_config)
    broken = expected.copy()
    broken.iloc[3, 0] = 0
    with pytest.raises(ValueError, match="positive"):
        validate_prices(broken, short_config)
    returns = prices_to_returns(expected)
    assert len(returns) == len(expected) - 1
    assert np.isclose(returns.iloc[0, 0], expected.iloc[1, 0] / expected.iloc[0, 0] - 1)


def test_saved_returns_are_checked_against_prices(short_config, tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=10)
    prices = pd.DataFrame(
        {ticker: 100 + np.arange(len(dates)) for ticker in short_config.tickers},
        index=dates,
    )
    save_processed(prices, short_config, tmp_path)
    loaded_prices, loaded_returns = load_processed(short_config, tmp_path)
    pd.testing.assert_frame_equal(loaded_prices, prices.astype(float), check_freq=False)
    pd.testing.assert_frame_equal(
        loaded_returns, prices_to_returns(prices.astype(float)), check_freq=False
    )
    corrupted = loaded_returns.copy()
    corrupted.iloc[0, 0] += 0.01
    corrupted.to_parquet(tmp_path / "returns.parquet")
    with pytest.raises(ValueError, match="do not match"):
        load_processed(short_config, tmp_path)
