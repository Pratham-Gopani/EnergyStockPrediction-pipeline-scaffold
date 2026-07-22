"""EnergyStockPrediction CLI entrypoint.

Subcommands:
    run [--force-date YYYY-MM-DD]   Run the daily pipeline once.
    schedule                        Start the APScheduler daemon (blocking).
    monitor                         Run the post-inference monitoring pass once.

All heavy imports (pipeline/scheduler/monitor modules, which transitively lazy-load
torch/transformers/yfinance/gnews/newspaper3k/apscheduler) happen inside each
command function, not at module top level, so `python main.py --help` succeeds
with only the light dependency set installed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date


def _cmd_run(args: argparse.Namespace) -> int:
    from scheduler.tasks import run_pipeline

    force_date = date.fromisoformat(args.force_date) if args.force_date else None
    stats = run_pipeline(force_date=force_date)
    print(stats)
    return 0


def _cmd_schedule(args: argparse.Namespace) -> int:
    from scheduler.scheduler import start

    start()
    return 0


def _cmd_monitor(args: argparse.Namespace) -> int:
    from scheduler.monitor import run_monitoring

    results = run_monitoring()
    print(results)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="EnergyStockPrediction pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the daily pipeline once")
    run_parser.add_argument(
        "--force-date",
        dest="force_date",
        default=None,
        help="ISO date (YYYY-MM-DD) to run the pipeline for, instead of today",
    )
    run_parser.set_defaults(func=_cmd_run)

    schedule_parser = subparsers.add_parser("schedule", help="Start the APScheduler daemon")
    schedule_parser.set_defaults(func=_cmd_schedule)

    monitor_parser = subparsers.add_parser("monitor", help="Run the monitoring pass once")
    monitor_parser.set_defaults(func=_cmd_monitor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
