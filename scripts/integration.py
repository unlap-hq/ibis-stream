#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


def _run_pytest(dialect: str, pytest_args: list[str]) -> int:
    marker = "integration" if dialect == "all" else f"integration and {dialect}"
    test_path = "tests/integration" if dialect == "all" else f"tests/integration/{dialect}"
    command = ["pytest", "-q", "-m", marker, test_path, *pytest_args]
    completed = subprocess.run(command, check=False)
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ibis-stream integration tests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run integration tests.")
    run_parser.add_argument(
        "--dialect",
        choices=["all", "bigquery"],
        default="all",
        help="Dialect integration suite to run.",
    )
    run_parser.add_argument("pytest_args", nargs="*", help="Additional pytest arguments.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        return _run_pytest(args.dialect, args.pytest_args)
    print(f"Unsupported command: {args.command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
