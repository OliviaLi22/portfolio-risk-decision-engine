"""Return and risk measures for daily constant-weight portfolios."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _check_weights(returns: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    vector = np.asarray(weights, dtype=float)
    if vector.shape != (returns.shape[1],) or not np.isfinite(vector).all():
        raise ValueError("weights must be finite and match the return columns")
    if (vector < -1e-10).any() or not np.isclose(vector.sum(), 1.0, atol=1e-8):
        raise ValueError("weights must be long-only and sum to 1")
    return vector


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Daily constant-mix return, assuming costless rebalancing each close."""
    vector = _check_weights(returns, weights)
    return pd.Series(returns.to_numpy() @ vector, index=returns.index, name="portfolio")


def annualized_volatility(
    daily_returns: pd.Series, annualization_days: int = 252
) -> float:
    return float(daily_returns.std(ddof=1) * np.sqrt(annualization_days))


def tracking_error(
    portfolio_daily: pd.Series,
    benchmark_daily: pd.Series,
    annualization_days: int = 252,
) -> float:
    """Ex-post annualized standard deviation of daily active returns."""
    if not portfolio_daily.index.equals(benchmark_daily.index):
        raise ValueError("portfolio and benchmark dates must match")
    return annualized_volatility(portfolio_daily - benchmark_daily, annualization_days)


def active_return(portfolio_daily: pd.Series, benchmark_daily: pd.Series) -> pd.Series:
    if not portfolio_daily.index.equals(benchmark_daily.index):
        raise ValueError("portfolio and benchmark dates must match")
    return portfolio_daily - benchmark_daily


def cumulative_returns(daily_returns: pd.Series) -> pd.Series:
    return (1 + daily_returns).cumprod() - 1


def maximum_drawdown(daily_returns: pd.Series) -> float:
    wealth = (1 + daily_returns).cumprod()
    wealth = pd.concat(
        [pd.Series([1.0]), wealth.reset_index(drop=True)], ignore_index=True
    )
    return float((wealth / wealth.cummax() - 1).min())


def covariance_matrix(returns: pd.DataFrame) -> np.ndarray:
    if len(returns) < 2 or returns.isna().any().any():
        raise ValueError("need at least two complete return observations")
    covariance = returns.cov().to_numpy()
    if not np.isfinite(covariance).all():
        raise ValueError("covariance contains nonfinite values")
    return covariance


def predicted_tracking_error(
    weights: np.ndarray,
    benchmark_weights: np.ndarray,
    covariance: np.ndarray,
    annualization_days: int = 252,
) -> float:
    """Ex-ante TE from an estimated daily covariance matrix."""
    active = np.asarray(weights) - np.asarray(benchmark_weights)
    variance = float(active @ covariance @ active)
    return float(np.sqrt(max(variance, 0) * annualization_days))


def risk_contributions(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Component contribution to portfolio annualization-free volatility."""
    weights = np.asarray(weights, dtype=float)
    variance = float(weights @ covariance @ weights)
    if variance <= 0:
        raise ValueError("portfolio variance must be positive")
    return weights * (covariance @ weights) / np.sqrt(variance)


def risk_diagnostics(
    returns: pd.DataFrame,
    weights: np.ndarray,
    benchmark_weights: np.ndarray,
    annualization_days: int = 252,
) -> dict[str, float]:
    portfolio = portfolio_returns(returns, weights)
    benchmark = portfolio_returns(returns, benchmark_weights)
    active = active_return(portfolio, benchmark)
    correlation = float(portfolio.corr(benchmark))
    return {
        "annualized_volatility": annualized_volatility(portfolio, annualization_days),
        "ex_post_tracking_error": tracking_error(
            portfolio, benchmark, annualization_days
        ),
        "annualized_arithmetic_active_return": float(
            active.mean() * annualization_days
        ),
        "correlation_with_benchmark": correlation,
        "cumulative_return": float(cumulative_returns(portfolio).iloc[-1]),
        "benchmark_cumulative_return": float(cumulative_returns(benchmark).iloc[-1]),
        "maximum_drawdown": maximum_drawdown(portfolio),
        "benchmark_maximum_drawdown": maximum_drawdown(benchmark),
        "largest_asset_weight": float(np.max(weights)),
        "weight_herfindahl_index": float(np.sum(np.square(weights))),
        "active_weight_l1": float(np.sum(np.abs(weights - benchmark_weights))),
    }
