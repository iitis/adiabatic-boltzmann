"""
Remove result JSON files where any energy value exceeds 100× the number of spins.

The number of spins depends on the model:
  - 1D: n_spins = size
  - 2D: n_spins = size × size  (e.g. size=8 → 64 spins)

Usage (dry-run, default):
    python scripts/purge_extreme_results.py

Actually delete:
    python scripts/purge_extreme_results.py --delete

Custom threshold multiplier (default 100):
    python scripts/purge_extreme_results.py --threshold 50 --delete

Custom results root:
    python scripts/purge_extreme_results.py --results-dir path/to/results --delete
"""

import argparse
import json
import sys
from pathlib import Path


def n_spins(config: dict) -> int:
    """Return the number of spin variables for this config."""
    model = config["model"]
    size = config["size"]
    if model == "2d":
        return size * size
    elif model == "1d":
        return size
    else:
        raise ValueError(f"Unknown model type: {model!r}")


def is_extreme(data: dict, threshold: float) -> tuple[bool, float, float]:
    """
    Return (extreme, worst_abs_energy, limit) where extreme is True if any
    energy in history or the final_energy exceeds threshold * n_spins.
    """
    spins = n_spins(data["config"])
    limit = threshold * spins

    energies = list(data["history"]["energy"])
    if "final_energy" in data:
        energies.append(data["final_energy"])

    worst = max(abs(e) for e in energies)
    return worst > limit, worst, limit


def main():
    parser = argparse.ArgumentParser(
        description="Purge result files with extreme energy values."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root directory containing result JSON files (default: results/)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="Flag files where max |energy| > threshold × system_size (default: 100)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete flagged files (omit for a dry run)",
    )
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    if not results_root.is_dir():
        print(f"ERROR: results directory not found: {results_root}", file=sys.stderr)
        sys.exit(1)

    json_files = sorted(results_root.rglob("*.json"))
    if not json_files:
        print("No JSON files found.")
        return

    flagged = []
    parse_errors = []

    for path in json_files:
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            parse_errors.append((path, exc))
            continue

        try:
            extreme, worst, limit = is_extreme(data, args.threshold)
        except (KeyError, ValueError, TypeError) as exc:
            parse_errors.append((path, exc))
            continue

        if extreme:
            flagged.append((path, worst, limit))

    # Report parse errors — never silently ignore
    if parse_errors:
        print(f"\n{'='*60}")
        print(f"PARSE / SCHEMA ERRORS ({len(parse_errors)} files) — skipped:")
        for path, exc in parse_errors:
            print(f"  {path}: {exc}")

    # Report flagged files
    print(f"\n{'='*60}")
    mode = "DELETING" if args.delete else "DRY RUN — would delete"
    print(
        f"{mode}: {len(flagged)} / {len(json_files)} files "
        f"(threshold = {args.threshold}× n_spins; 1D: size, 2D: size²)"
    )
    print(f"{'='*60}")

    for path, worst, limit in flagged:
        print(f"  {path}")
        print(f"    max |energy| = {worst:.4g}  >  limit = {limit:.4g}")
        if args.delete:
            path.unlink()
            print(f"    [DELETED]")

    if not flagged:
        print("  (none)")

    if not args.delete:
        print(
            "\nRe-run with --delete to remove these files."
        )


if __name__ == "__main__":
    main()
