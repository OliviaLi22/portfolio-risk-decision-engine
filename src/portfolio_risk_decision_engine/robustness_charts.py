"""Illustrative figures for observed portfolio outcomes and robustness evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

from .robustness import RobustnessResult
from .validation import WalkForwardResult

_INK = "#18383b"
_MUTED = "#667d80"
_GRID = "#e4ece9"
_POSITIVE = "#168274"
_NEGATIVE = "#cc7569"
_STRATEGIES = {
    "no_rebalancing": ("No rebalancing", "#829eaa"),
    "rules_based": ("Rules based", "#a08ac0"),
    "optimized": ("Optimized", "#168274"),
    "benchmark": ("SPY benchmark", "#d58b79"),
}


def _style_axes(ax: Axes, *, horizontal: bool = False) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="x" if horizontal else "y", color=_GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(_GRID)
    ax.tick_params(colors=_MUTED, length=0, pad=8)
    ax.xaxis.label.set_color(_MUTED)
    ax.yaxis.label.set_color(_MUTED)
    ax.title.set_color(_INK)


def _save(fig: Figure, directory: Path, name: str) -> str:
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return f"{name}.png"


def _heading(fig: Figure, title: str, subtitle: str) -> None:
    fig.text(0.035, 0.985, title, color=_INK, fontsize=18, fontweight="bold", va="top")
    fig.text(0.035, 0.94, subtitle, color=_MUTED, fontsize=10, va="top")


def _date_range(walk: WalkForwardResult) -> str:
    index = walk.daily_returns.index
    return f"{index[0]:%d %b %Y} – {index[-1]:%d %b %Y}"


def _strategy_tradeoffs(
    walk: WalkForwardResult, terminal_comparison: list[dict], directory: Path
) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4))
    _heading(
        fig, "A portfolio decision changes several kinds of risk", _date_range(walk)
    )
    specifications = [
        (
            "realized_tracking_error",
            "Realized tracking error",
            "Annualized · lower is smaller benchmark deviation",
        ),
        (
            "cumulative_net_return",
            "Cumulative net return",
            "Full OOS period · costs included",
        ),
        (
            "mean_decision_stress_loss",
            "Mean historical stress-loss exposure",
            "Past windows assessed at each decision · lower exposure",
        ),
    ]
    for ax, (metric, title, description) in zip(
        axes.flat, specifications, strict=False
    ):
        keys = [key for key in _STRATEGIES if key in walk.summary.index]
        values = [walk.summary.loc[key, metric] for key in keys]
        bars = ax.bar(
            np.arange(len(keys)),
            values,
            color=[_STRATEGIES[key][1] for key in keys],
            width=0.62,
        )
        ax.set_xticks(
            np.arange(len(keys)), [_STRATEGIES[key][0] for key in keys], fontsize=9
        )
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=25)
        ax.text(
            0, 1.025, description, transform=ax.transAxes, color=_MUTED, fontsize=8.5
        )
        ax.yaxis.set_major_formatter(
            PercentFormatter(
                1, decimals=1 if metric == "realized_tracking_error" else 0
            )
        )
        ax.margins(y=0.2)
        ax.bar_label(
            bars,
            labels=[f"{value:.2%}" for value in values],
            padding=4,
            color=_INK,
            fontsize=9,
        )
        _style_axes(ax)
    ax = axes[1, 1]
    terminal = pd.DataFrame(terminal_comparison).set_index("strategy")
    keys = [key for key in _STRATEGIES if key in terminal.index]
    values = terminal.loc[keys, "weight_hhi"].to_numpy(dtype=float)
    bars = ax.bar(
        np.arange(len(keys)),
        values,
        color=[_STRATEGIES[key][1] for key in keys],
        width=0.56,
    )
    ax.set_xticks(
        np.arange(len(keys)), [_STRATEGIES[key][0] for key in keys], fontsize=9
    )
    ax.set_title(
        "Final-date weight concentration",
        loc="left",
        fontsize=12,
        fontweight="bold",
        pad=25,
    )
    ax.text(
        0,
        1.025,
        "Retrospective allocation · HHI = sum of squared weights",
        transform=ax.transAxes,
        color=_MUTED,
        fontsize=8.5,
    )
    ax.set_ylabel("Weight HHI")
    ax.margins(y=0.23)
    ax.bar_label(
        bars,
        labels=[f"{value:.3f}" for value in values],
        padding=4,
        color=_INK,
        fontsize=9,
    )
    _style_axes(ax)
    fig.text(
        0.035,
        0.025,
        "OOS = out of sample. Historical loss exposure is not a subsequently realized crisis loss. The concentration panel is a separate final-date diagnostic.",
        fontsize=8.4,
        color=_MUTED,
    )
    fig.subplots_adjust(
        left=0.07, right=0.98, top=0.84, bottom=0.09, hspace=0.56, wspace=0.21
    )
    return _save(fig, directory, "robustness_strategy_tradeoffs")


def _risk_path(
    walk: WalkForwardResult, directory: Path, annualization_days: int
) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.8), sharex=True)
    _heading(
        fig,
        "Risk unfolds through time",
        "Trailing 63 observed sessions and net OOS drawdowns · " + _date_range(walk),
    )
    daily = walk.daily_returns
    for key, (label, color) in _STRATEGIES.items():
        if key not in daily:
            continue
        width = 2.2 if key == "optimized" else 1.5
        if key != "benchmark":
            active = daily[key] - daily["benchmark"]
            rolling = active.rolling(63, min_periods=63).std(ddof=1) * np.sqrt(
                annualization_days
            )
            axes[0].plot(
                rolling.index, rolling, label=label, color=color, linewidth=width
            )
        wealth = (1 + daily[key]).cumprod()
        # Initial wealth of one must also be included in the running peak.
        drawdown = wealth / wealth.cummax().clip(lower=1) - 1
        axes[1].plot(
            drawdown.index, drawdown, label=label, color=color, linewidth=width
        )
    axes[0].set_title(
        "Rolling realized tracking error", loc="left", fontweight="bold", fontsize=12
    )
    axes[0].set_ylabel("Annualized tracking error")
    axes[0].legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
    axes[1].set_title(
        "Drawdown from prior wealth peak", loc="left", fontweight="bold", fontsize=12
    )
    axes[1].set_ylabel("Net drawdown")
    axes[1].legend(frameon=False, ncol=4, loc="lower left", fontsize=9)
    for ax in axes:
        _style_axes(ax)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
        ax.margins(x=0)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.text(
        0.035,
        0.025,
        "Tracking error uses only the current and prior 62 observed daily active returns; the first 62 sessions have no rolling estimate. Costs enter net returns.",
        fontsize=8.5,
        color=_MUTED,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.10, hspace=0.33)
    return _save(fig, directory, "robustness_risk_path")


def _short_label(row: pd.Series) -> str:
    label = str(row["label"])
    substitutions = {
        "Base configuration": "Base configuration",
        "Covariance lookback:": "Lookback:",
        "Trading cost:": "Trading cost:",
        "Stress objective weight:": "Stress weight:",
        "Predicted TE budget:": "TE budget:",
        "Covariance diagonal shrinkage:": "Diagonal shrinkage:",
        "Rebalance every": "Rebalance every",
    }
    for original, replacement in substitutions.items():
        label = label.replace(original, replacement)
    if row["status"] != "completed":
        label += f" — {row['status']}"
    return label


def _sensitivity(robustness: RobustnessResult, directory: Path) -> str:
    rows = robustness.sensitivity.reset_index(drop=True)
    height = max(8.3, len(rows) * 0.40 + 2.4)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, height), sharey=True)
    _heading(
        fig,
        "Which assumptions change the conclusion?",
        "Each alternative changes one setting · positive values favor the optimized strategy versus no rebalancing",
    )
    specifications = [
        (
            "te_improvement_vs_no_rebalancing",
            "Tracking-error improvement",
            "No rebalance TE − optimized TE",
        ),
        (
            "net_return_difference_vs_no_rebalancing",
            "Net-return difference",
            "Optimized − no rebalance",
        ),
        (
            "stress_improvement_vs_no_rebalancing",
            "Historical stress-exposure reduction",
            "No rebalance loss − optimized loss",
        ),
    ]
    y = np.arange(len(rows))
    for ax, (column, title, subtitle) in zip(axes, specifications, strict=True):
        values = pd.to_numeric(rows[column], errors="coerce").to_numpy() * 100
        finite = np.isfinite(values) & rows.status.eq("completed").to_numpy()
        ax.barh(
            y[finite],
            values[finite],
            color=[_POSITIVE if value >= 0 else _NEGATIVE for value in values[finite]],
            height=0.60,
        )
        ax.axvline(0, color=_INK, linewidth=1)
        limit = max(np.abs(values[finite]).max(initial=0) * 1.24, 0.5)
        ax.set_xlim(-limit, limit)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=27)
        ax.text(0, 1.025, subtitle, transform=ax.transAxes, color=_MUTED, fontsize=8.5)
        ax.set_xlabel("Percentage points")
        for index, value in enumerate(values):
            if not finite[index]:
                ax.axhspan(index - 0.38, index + 0.38, color="#f0f3f2", zorder=0)
                ax.text(
                    0,
                    index,
                    "Not estimated",
                    ha="center",
                    va="center",
                    color=_MUTED,
                    fontsize=8,
                    fontstyle="italic",
                )
            else:
                ax.annotate(
                    f"{value:+.2f}",
                    (value, index),
                    xytext=(4 if value >= 0 else -4, 0),
                    textcoords="offset points",
                    ha="left" if value >= 0 else "right",
                    va="center",
                    fontsize=8,
                    color=_INK,
                )
        _style_axes(ax, horizontal=True)
        ax.set_yticks(y)
        ax.set_ylim(len(rows) - 0.5, -0.5)
    axes[0].set_yticklabels(
        [_short_label(row) for _, row in rows.iterrows()], fontsize=9
    )
    # Emphasize the reference configuration without implying independent trials.
    for ax in axes:
        ax.axhspan(-0.38, 0.38, color="#168274", alpha=0.06, zorder=0)
    fig.text(
        0.035,
        0.025,
        "Failed variants have no plotted outcome. These settings are exploratory alternatives, not independent trials or a probability of future success. Scales differ by panel.",
        fontsize=8.7,
        color=_MUTED,
    )
    fig.subplots_adjust(left=0.22, right=0.98, top=0.83, bottom=0.10, wspace=0.23)
    return _save(fig, directory, "robustness_sensitivity")


def _tradeoff_map(
    walk: WalkForwardResult, robustness: RobustnessResult, directory: Path
) -> str:
    rows = robustness.sensitivity.loc[robustness.sensitivity.status == "completed"]
    fig, ax = plt.subplots(figsize=(12.8, 7.6))
    _heading(
        fig,
        "Tracking error and historical stress exposure pull in different directions",
        "Observed OOS outcomes across completed configurations · each point represents one tested setting",
    )
    _style_axes(ax)
    x = rows.realized_tracking_error.to_numpy() * 100
    y = rows.mean_decision_stress_loss.to_numpy() * 100
    ax.scatter(
        x,
        y,
        s=55,
        color="#bccdca",
        edgecolor="white",
        linewidth=0.8,
        label="Other completed variants",
        zorder=3,
    )
    selected = {
        "base": ("Base optimizer", _POSITIVE, (35, 28)),
        "stress_0": ("Stress weight 0", "#a08ac0", (22, 22)),
        "stress_1": ("Stress weight 1", "#a08ac0", (-95, -35)),
        "shrinkage_25": ("25% diagonal shrinkage", _NEGATIVE, (-105, -35)),
        "te_0.03": ("3% TE cap", "#617d8a", (38, -32)),
    }
    for key, (label, color, offset) in selected.items():
        sample = rows.loc[rows.variant == key]
        if sample.empty:
            continue
        point = sample.iloc[0]
        xy = (
            point.realized_tracking_error * 100,
            point.mean_decision_stress_loss * 100,
        )
        ax.scatter(
            *xy,
            s=125 if key == "base" else 85,
            color=color,
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
        )
        ax.annotate(
            label,
            xy,
            xytext=offset,
            textcoords="offset points",
            fontsize=10,
            fontweight="bold" if key == "base" else "normal",
            color=_INK,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.9},
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 3},
            zorder=6,
        )
    base = walk.summary.loc["no_rebalancing"]
    xy = (base.realized_tracking_error * 100, base.mean_decision_stress_loss * 100)
    ax.scatter(*xy, marker="D", s=100, color=_INK, edgecolor="white", zorder=5)
    ax.annotate(
        "No rebalancing",
        xy,
        xytext=(28, 18),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        color=_INK,
        arrowprops={"arrowstyle": "-", "color": _INK, "lw": 0.9},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 3},
    )
    ax.set_xlabel("Realized annualized tracking error (%)", labelpad=12)
    ax.set_ylabel("Mean historical stress-loss exposure (%)", labelpad=12)
    ax.margins(x=0.16, y=0.17)
    ax.annotate(
        "Lower in both dimensions",
        xy=(0.05, 0.08),
        xytext=(0.22, 0.22),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": _POSITIVE, "lw": 1.5},
        color=_POSITIVE,
        fontsize=10,
        ha="center",
    )
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.text(
        0.035,
        0.025,
        "Past-window loss exposure is a diagnostic, not future crisis protection. Some settings overlap. Infeasible settings have no point; this is not an efficient frontier.",
        fontsize=8.7,
        color=_MUTED,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.85, bottom=0.13)
    return _save(fig, directory, "robustness_tradeoff_map")


def _annual(robustness: RobustnessResult, directory: Path) -> str:
    annual = robustness.annual
    years = sorted(annual.year.unique())
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.2))
    _heading(
        fig,
        "The aggregate result is built from different calendar years",
        "Annual slices of the same walk-forward paths · modeled trading costs included",
    )
    metrics = [
        ("cumulative_net_return", "Calendar-year net return", True),
        ("realized_tracking_error", "Annualized realized tracking error", False),
    ]
    for ax, (metric, title, include_benchmark) in zip(axes, metrics, strict=True):
        keys = [key for key in _STRATEGIES if include_benchmark or key != "benchmark"]
        width = 0.74 / len(keys)
        for i, key in enumerate(keys):
            values = (
                annual.loc[annual.strategy == key]
                .set_index("year")[metric]
                .reindex(years)
            )
            positions = np.arange(len(years)) + (i - (len(keys) - 1) / 2) * width
            ax.bar(
                positions,
                values,
                width=width * 0.93,
                color=_STRATEGIES[key][1],
                label=_STRATEGIES[key][0],
            )
        ax.set_xticks(np.arange(len(years)), years)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=14)
        ax.yaxis.set_major_formatter(
            PercentFormatter(
                1, decimals=1 if metric == "realized_tracking_error" else 0
            )
        )
        ax.margins(y=0.2)
        _style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.055),
        fontsize=9,
    )
    fig.text(
        0.035,
        0.015,
        "Annual returns compound and do not add to the cumulative period return. Annual tracking error is measured against SPY and is zero for SPY itself.",
        fontsize=8.5,
        color=_MUTED,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.79, bottom=0.19, wspace=0.22)
    return _save(fig, directory, "robustness_annual")


def _bootstrap(robustness: RobustnessResult, directory: Path) -> str:
    bootstrap = robustness.bootstrap
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 6.7))
    _heading(
        fig,
        "What uncertainty is supported by the observed paths?",
        f"Conditional 95% percentile intervals · {bootstrap['replicates']:,} paired resamples · {bootstrap['block_days']}-session blocks",
    )
    metrics = [
        (
            "annual_return_difference",
            "Annual arithmetic net-return difference",
            "Optimized minus no rebalancing",
        ),
        (
            "tracking_error_improvement",
            "Realized tracking-error improvement",
            "No rebalance TE minus optimized TE",
        ),
    ]
    for ax, (key, title, subtitle) in zip(axes, metrics, strict=True):
        bounds = bootstrap[key]
        lower, upper, estimate = (
            float(bounds[field]) * 100 for field in ("lower", "upper", "estimate")
        )
        color = _NEGATIVE if upper < 0 else _POSITIVE if lower > 0 else "#829eaa"
        ax.axvline(0, linestyle="--", color=_INK, linewidth=1, label="No difference")
        ax.plot(
            [lower, upper], [0, 0], color=color, linewidth=7, solid_capstyle="round"
        )
        ax.plot([lower, lower], [-0.07, 0.07], color=color, linewidth=2)
        ax.plot([upper, upper], [-0.07, 0.07], color=color, linewidth=2)
        ax.scatter(
            [estimate],
            [0],
            s=105,
            color=_INK,
            edgecolor="white",
            linewidth=1.3,
            zorder=5,
        )
        ax.text(lower, 0.14, f"{lower:+.2f} pp", ha="center", color=_INK, fontsize=10)
        ax.text(upper, 0.14, f"{upper:+.2f} pp", ha="center", color=_INK, fontsize=10)
        ax.text(
            estimate,
            -0.2,
            f"Observed estimate {estimate:+.2f} pp",
            ha="center",
            va="top",
            color=_INK,
            fontsize=9,
        )
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=22)
        ax.text(0, 1.035, subtitle, transform=ax.transAxes, color=_MUTED, fontsize=9)
        ax.set_ylim(-0.42, 0.42)
        ax.set_yticks([])
        _style_axes(ax, horizontal=True)
        ax.spines["left"].set_visible(False)
        min_value, max_value = min(lower, estimate, 0), max(upper, estimate, 0)
        padding = max((max_value - min_value) * 0.13, 0.1)
        ax.set_xlim(min_value - padding, max_value + padding)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+.1f}"))
        ax.set_xlabel("Percentage points · positive favors optimized", fontsize=9)
    fig.text(
        0.035,
        0.025,
        "Fixed observed decisions are resampled together. Covariance and decisions are not refitted; strategy selection and future-regime uncertainty are outside these intervals.",
        fontsize=8.6,
        color=_MUTED,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.12, hspace=0.95)
    return _save(fig, directory, "robustness_bootstrap")


def create_robustness_figures(
    walk: WalkForwardResult,
    robustness: RobustnessResult,
    terminal_comparison: list[dict],
    figures_dir: Path,
    annualization_days: int = 252,
) -> dict[str, str]:
    """Save six data-driven PNG/SVG figures and return the PNG basenames.

    Temporal plots use trailing observations only. Failed sensitivity variants
    stay visibly unestimated. Bootstrap plots describe fixed observed decision
    paths and do not imply uncertainty about fully refitted future strategies.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    ):
        return {
            "strategy_tradeoffs": _strategy_tradeoffs(
                walk, terminal_comparison, figures_dir
            ),
            "risk_path": _risk_path(walk, figures_dir, annualization_days),
            "sensitivity": _sensitivity(robustness, figures_dir),
            "tradeoff_map": _tradeoff_map(walk, robustness, figures_dir),
            "annual": _annual(robustness, figures_dir),
            "bootstrap": _bootstrap(robustness, figures_dir),
        }
