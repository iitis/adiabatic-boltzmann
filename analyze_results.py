#!/usr/bin/env python3
"""
Analyze benchmark results.

Loads and summarizes results from completed benchmarks.

Usage:
    python analyze_results.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import setup_logging
from experiments.analyze_results import ResultsLoader


def main():
    """Main entry point."""
    # Setup logging
    setup_logging()
    
    try:
        loader = ResultsLoader(results_dir="experiments/results")
        loader.print_summary()
        return 0
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
