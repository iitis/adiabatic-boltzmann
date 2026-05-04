"""
Move jax_results files from the old layout
    jax_results/{size}/{sampler}/{method}/result_{model}_...
to the current layout
    jax_results/{model_subdir}/{size}/{sampler}/{method}/result_{model}_...

Dry-run by default; pass --execute to actually move files.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent / "jax_results"
MODEL_RE = re.compile(r"^result_([^_]+)_")

MODEL_SUBDIR = {
    "1d":  "tfim_1d",
    "2d":  "tfim_2d",
    "lr1d": "lr_tfim_1d",
}


def plan(root: Path) -> list[tuple[Path, Path]]:
    moves = []
    for f in root.rglob("*.json"):
        rel = f.relative_to(root)
        parts = rel.parts  # (size_or_subdir, sampler, method, filename)
        if not parts[0].isdigit():
            continue  # already in new layout
        size, *rest = parts
        m = MODEL_RE.match(parts[-1])
        if not m:
            print(f"  [skip] unrecognised filename: {f}", file=sys.stderr)
            continue
        model = m.group(1)
        subdir = MODEL_SUBDIR.get(model)
        if subdir is None:
            print(f"  [skip] unknown model '{model}': {f}", file=sys.stderr)
            continue
        dst = root / subdir / Path(*parts)
        moves.append((f, dst))
    return moves


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true",
                        help="Actually move files (default: dry-run)")
    cli = parser.parse_args()

    moves = plan(ROOT)
    print(f"{'DRY RUN — ' if not cli.execute else ''}Files to move: {len(moves)}")

    for src, dst in moves:
        print(f"  {src.relative_to(ROOT)}  →  {dst.relative_to(ROOT)}")
        if cli.execute:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)

    if cli.execute:
        # Remove empty dirs left behind
        for d in sorted(ROOT.rglob("*"), reverse=True):
            if d.is_dir() and d.name.isdigit() and not any(d.rglob("*")):
                d.rmdir()
        print("Done.")
    else:
        print("\nRe-run with --execute to apply.")


if __name__ == "__main__":
    main()
