#!/usr/bin/env python3
"""
One-time migration: set history["beta_x"] to all-1.0s for runs where
beta_x is meaningless (custom/metropolis and custom/gibbs).

Overwrites files in place. Safe to re-run (idempotent).

Usage:
    python scripts/fix_beta_x.py
    python scripts/fix_beta_x.py --results src/results/
    python scripts/fix_beta_x.py --dry-run
"""

import argparse
import json
from pathlib import Path

FIXED_METHODS = {"metropolis", "gibbs"}


def fix_file(path: Path, dry_run: bool) -> bool:
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  [skip] {path}: {e}")
        return False

    cfg = data.get("config", {})
    if cfg.get("sampler") != "custom":
        return False
    if cfg.get("sampling_method") not in FIXED_METHODS:
        return False

    hist = data.get("history", {})
    beta_x = hist.get("beta_x", [])
    if not beta_x:
        return False

    if all(v == 1.0 for v in beta_x):
        return False  # already correct

    n = len(beta_x)
    hist["beta_x"] = [1.0] * n
    data["history"] = hist

    if dry_run:
        print(f"  [dry-run] would fix {path.name}  ({n} iters, "
              f"old range [{min(beta_x):.3f}, {max(beta_x):.3f}])")
        return True

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  fixed: {path.name}  ({n} iters)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"),
                        help="Results directory (default: results/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be changed without writing")
    args = parser.parse_args()

    paths = list(args.results.rglob("result_*.json"))
    print(f"Scanning {len(paths)} files in {args.results} ...")

    fixed = sum(fix_file(p, args.dry_run) for p in paths)
    verb = "would fix" if args.dry_run else "fixed"
    print(f"\nDone. {verb} {fixed} file(s).")


if __name__ == "__main__":
    main()
