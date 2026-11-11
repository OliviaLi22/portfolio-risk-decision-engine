"""An illustrated reading of the observed strategy and sensitivity results."""

from __future__ import annotations

from pathlib import Path


def write_robustness_report(
    payload: dict, figures: dict[str, str], results_dir: Path
) -> None:
    """Write the narrative beside the figures so links also render on GitHub."""
    meta = payload["meta"]
    evidence = payload["robustness"]
    summary = payload["walk_forward"]["summary"]
    terminal = {row["strategy"]: row for row in payload["terminal"]["comparison"]}
    conclusions = payload["conclusions"]
    bootstrap = evidence["bootstrap"]
    labels = {
        "no_rebalancing": "No rebalancing",
        "rules_based": "Rules based",
        "optimized": "Optimized",
        "benchmark": "SPY benchmark",
    }

    def image(key: str, alt: str, caption: str) -> str:
        return f"![{alt}](figures/{figures[key]})\n\n*{caption}*\n\n"

    report = (
        f"# Portfolio robustness assessment\n\n"
        f"**{meta['name']}** · {meta['source']}  \n"
        f"Price sample: {meta['price_start']}–{meta['price_end']}. "
        f"Walk-forward evaluation: **{meta['oos_start']}–{meta['oos_end']}**.\n\n"
        "## The observed decision trade-off\n\n"
        f"{conclusions[0]['detail']}\n\n"
        "| Strategy | Net cumulative return | Realized tracking error | Mean historical stress loss exposure |\n"
        "| --- | ---: | ---: | ---: |\n"
    )
    for row in summary:
        report += (
            f"| {labels.get(row['strategy'], row['strategy'])} "
            f"| {row['cumulative_net_return']:.2%} "
            f"| {row['realized_tracking_error']:.2%} "
            f"| {row['mean_decision_stress_loss']:.2%} |\n"
        )
    report += "\n" + image(
        "strategy_tradeoffs",
        "Out-of-sample strategy risk, return, and historical stress exposure",
        "The first three panels describe the walk-forward period; the concentration panel is a separate final-date diagnostic. "
        "Net return covers the entire evaluation period; tracking error is annualized. "
        "Stress exposure is the mean loss in the selected past windows, evaluated at each decision date. "
        "Each panel has its own scale.",
    )
    report += "## How the outcomes evolved\n\n"
    report += image(
        "risk_path",
        "Rolling realized tracking error and drawdown during the walk-forward period",
        "Rolling tracking error uses the trailing 63 observed daily active returns and 252-day "
        "annualization. Drawdown is measured from each strategy's own running net-wealth peak. "
        "These are realized diagnostics, not the optimizer's predicted tracking error.",
    )
    report += image(
        "annual",
        "Calendar-year net returns and realized tracking error by strategy",
        "Calendar-year slices expose variation hidden by the full-period averages. "
        "The strategy is not reset at year boundaries, and annual returns compound into the full-period result.",
    )
    report += (
        "## Sensitivity to modeling choices\n\n"
        f"{conclusions[2]['detail']}\n\n"
        "The grid changes one setting at a time. Positive values below mean an improvement "
        "relative to no rebalancing under the same dates and cost assumptions. "
        "An infeasible variant has no performance estimate.\n\n"
    )
    report += image(
        "sensitivity",
        "Sensitivity of tracking error, net return, and stress exposure to model assumptions",
        "Differences are in percentage points, with a separate scale for each metric. "
        "The grid is exploratory; the fraction of successful variants is not a probability of future success.",
    )
    report += image(
        "tradeoff_map",
        "Realized tracking error versus historical stress loss exposure across assumptions",
        "Moving left reduces realized tracking error; moving down reduces exposure to the past stress windows. "
        "The plotted configurations are observed comparisons, not an estimated efficient frontier. "
        "Overlapping points can represent settings that produced nearly identical decisions.",
    )
    for finding in evidence["findings"]:
        if finding["title"] in {
            "Tracking error across tested assumptions",
            "Historical stress exposure is a separate trade-off",
            "The configured risk budget may be slack",
            "Some assumptions cannot be implemented",
        }:
            report += f"**{finding['title']}.** {finding['detail']}\n\n"
    report += "## Sampling uncertainty\n\n"
    report += image(
        "bootstrap",
        "Conditional paired-block bootstrap intervals for return difference and tracking-error improvement",
        f"Dots are observed estimates; lines are 95% percentile intervals from "
        f"{bootstrap['replicates']:,} paired resamples of {bootstrap['block_days']}-observation blocks. "
        "The reference line marks no difference. Annual arithmetic return difference "
        "is distinct from the full-period compounded return difference above.",
    )
    for finding in evidence["findings"]:
        if finding["title"] in {
            "Return difference remains uncertain",
            "Return difference is conditional on the observed sample",
            "Conditional tracking-error difference",
        }:
            report += f"**{finding['title']}.** {finding['detail']}\n\n"
    report += f"**Scope of the intervals:** {bootstrap['caveat']}\n\n"
    report += (
        "## Final-date allocation and concentration\n\n"
        f"{conclusions[1]['detail']} "
        "This retrospective allocation uses information available at the final sample close; "
        "it is separate from the sequence of out-of-sample decisions above.\n\n"
        "![Portfolio weights before and after the final-date optimization](figures/before_after_weights.png)\n\n"
        "| Final-date diagnostic | Before | Optimized |\n"
        "| --- | ---: | ---: |\n"
    )
    before, after = terminal["no_rebalancing"], terminal["optimized"]
    for key, label, fmt in (
        ("predicted_tracking_error", "Predicted annual tracking error", ".2%"),
        (
            "worst_historical_stress_loss",
            "Worst historical stress loss exposure",
            ".2%",
        ),
        ("weight_hhi", "Weight Herfindahl index", ".3f"),
        ("largest_asset_weight", "Largest asset weight", ".2%"),
        ("one_way_turnover", "One-way turnover", ".2%"),
        ("estimated_transaction_cost", "Estimated trading cost", ".3%"),
    ):
        report += f"| {label} | {before[key]:{fmt}} | {after[key]:{fmt}} |\n"
    report += "\n## Interpretation limits\n\n"
    report += "\n".join(f"- {item}" for item in payload["limitations"]) + "\n\n"
    report += (
        "## Reproduce the report\n\n"
        "```bash\nuv run python -m portfolio_risk_decision_engine.run_analysis dashboard\n```\n\n"
        "The charts are regenerated from saved local observations. PNG and SVG copies of "
        "the robustness charts are saved under `results/figures/`. Detailed metrics are in "
        "`results/tables/robustness_sensitivity.csv`, `robustness_annual.csv`, and "
        "`robustness_bootstrap.json`. See the [methodology notes](../docs/robustness.md).\n"
    )
    (results_dir / "robustness_summary.md").write_text(report, encoding="utf-8")
