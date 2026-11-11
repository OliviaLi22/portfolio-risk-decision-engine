"""Yahoo Finance ingestion and strict local price/return validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PortfolioConfig


def normalize_yfinance_close(
    raw: pd.DataFrame, tickers: tuple[str, ...]
) -> pd.DataFrame:
    """Extract auto-adjusted Close from either yfinance MultiIndex ordering."""
    if raw.empty:
        raise ValueError("Yahoo Finance returned no price rows")
    if isinstance(raw.columns, pd.MultiIndex):
        levels = raw.columns.names
        close_level = next(
            (
                i
                for i in range(raw.columns.nlevels)
                if "Close" in raw.columns.get_level_values(i)
            ),
            None,
        )
        if close_level is None:
            raise ValueError(f"Yahoo Finance result has no Close field: {levels}")
        close = raw.xs("Close", axis=1, level=close_level)
    elif "Close" in raw.columns and len(tickers) == 1:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    else:
        raise ValueError("Unexpected Yahoo Finance columns; expected adjusted Close")
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    close.columns = close.columns.astype(str)
    if len(set(close.columns)) != len(close.columns):
        raise ValueError("Yahoo Finance returned duplicate ticker columns")
    missing = set(tickers) - set(close.columns)
    if missing:
        raise ValueError(f"Yahoo Finance omitted tickers: {sorted(missing)}")
    return close.loc[:, list(tickers)]


def validate_prices(prices: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    """Reject missing, duplicate, nonpositive, and out-of-range observations."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex")
    if prices.index.tz is not None:
        prices = prices.copy()
        prices.index = prices.index.tz_localize(None)
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("price dates must be unique and increasing")
    if len(set(prices.columns)) != len(prices.columns):
        raise ValueError("price ticker columns must be unique")
    if set(prices.columns) != set(config.tickers):
        raise ValueError(
            f"price ticker coverage differs from config: {set(config.tickers) - set(prices.columns)} missing"
        )
    if len(prices) < config.minimum_price_observations:
        raise ValueError(
            f"need at least {config.minimum_price_observations} price observations"
        )
    if (prices.index < pd.Timestamp(config.start)).any() or (
        prices.index >= pd.Timestamp(config.end_exclusive)
    ).any():
        raise ValueError(
            "price dates fall outside configured [start, end_exclusive) range"
        )
    if (prices.index.min() - pd.Timestamp(config.start)).days > 7:
        raise ValueError("prices start too late for the configured analysis period")
    if (pd.Timestamp(config.end_exclusive) - prices.index.max()).days > 7:
        raise ValueError("prices end too early for the configured analysis period")
    if prices.isna().any().any():
        raise ValueError("prices contain missing values; gaps are not forward-filled")
    try:
        numeric = prices.loc[:, list(config.tickers)].astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("prices must be numeric") from exc
    if not np.isfinite(numeric.to_numpy()).all() or (numeric <= 0).any().any():
        raise ValueError("prices must be finite and strictly positive")
    for window in config.stress_windows:
        if not (
            (numeric.index >= pd.Timestamp(window.start))
            & (numeric.index <= pd.Timestamp(window.end))
        ).any():
            raise ValueError(f"no prices in stress window {window.name}")
    return numeric


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple close-to-close returns. The first date has no return."""
    return prices.pct_change(fill_method=None).iloc[1:]


def download_prices(config: PortfolioConfig) -> pd.DataFrame:
    """Download adjusted ETF closes; only this function needs network access."""
    import yfinance as yf

    raw = yf.download(
        list(config.tickers),
        start=config.start.isoformat(),
        end=config.end_exclusive.isoformat(),
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="column",
    )
    return validate_prices(normalize_yfinance_close(raw, config.tickers), config)


def save_processed(
    prices: pd.DataFrame, config: PortfolioConfig, directory: Path
) -> None:
    validated = validate_prices(prices, config)
    directory.mkdir(parents=True, exist_ok=True)
    validated.to_parquet(directory / "prices.parquet")
    prices_to_returns(validated).to_parquet(directory / "returns.parquet")


def load_processed(
    config: PortfolioConfig, directory: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices_path = directory / "prices.parquet"
    returns_path = directory / "returns.parquet"
    if not prices_path.exists() or not returns_path.exists():
        raise FileNotFoundError(
            "processed Parquet files missing; run download-data first"
        )
    prices = validate_prices(pd.read_parquet(prices_path), config)
    returns = pd.read_parquet(returns_path)
    expected = prices_to_returns(prices)
    if not returns.index.equals(expected.index) or list(returns.columns) != list(
        expected.columns
    ):
        raise ValueError("saved returns dates or columns do not match prices")
    if not np.allclose(returns.to_numpy(), expected.to_numpy(), rtol=1e-12, atol=1e-14):
        raise ValueError("saved returns do not match saved prices")
    return prices, returns
