#!/usr/bin/env python3
import argparse
from src.fsdb_analyzer import FsdbAnalyzer


def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced FSDB Signal Analyzer with Complex Conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with advanced configuration
  %(prog)s -c ifu_analysis_advanced.yaml

  # Debug mode: limit to first trigger match
  %(prog)s -c ifu_analysis_advanced.yaml --debug-num 1
        """,
    )

    parser.add_argument(
        "-c", "--config", required=True,
        help="YAML configuration file path"
    )

    parser.add_argument(
        "--deps-only",
        action="store_true",
        help="Only generate dependency graph and exit (skip FSDB analysis)",
    )

    parser.add_argument(
        "--debug-num",
        type=int,
        default=0,
        help="Limit number of trigger matches for debugging (0 = unlimited)",
    )

    args = parser.parse_args()

    analyzer = FsdbAnalyzer(args.config, deps_only=args.deps_only, debug_num=args.debug_num)
    analyzer.run()


if __name__ == "__main__":
    main()
