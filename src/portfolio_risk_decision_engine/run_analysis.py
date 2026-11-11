"""Command-line entry point for local Yahoo download and offline analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .market_data import download_prices, load_processed, save_processed


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Illustrative portfolio risk decision engine"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("download-data", "analyze", "dashboard"),
        default="analyze",
    )
    parser.add_argument(
        "--config", type=Path, default=root / "configs" / "portfolio.json"
    )
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "processed")
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    parser.add_argument(
        "--dashboard-dir", type=Path, default=root / "dashboard" / "dist"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "download-data":
        prices = download_prices(config)
        save_processed(prices, config, args.data_dir)
        print(f"Saved {len(prices)} adjusted price rows to {args.data_dir}")
    else:
        from .reporting import create_reports

        prices, returns = load_processed(config, args.data_dir)
        walk = create_reports(prices, returns, config, args.results_dir)
        if args.command == "dashboard":
            from .dashboard import export_dashboard

            export_dashboard(
                prices, returns, config, args.results_dir, args.dashboard_dir, walk
            )
            print(
                f"Saved dashboard snapshot and robustness assessment to {args.dashboard_dir}"
            )
        print(f"Saved figures, tables, and decision summary to {args.results_dir}")


if __name__ == "__main__":
    main()
