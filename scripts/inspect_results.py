#!/usr/bin/env python3
"""
inspect_results.py — find result_*.json[.gz] files for a (model, N, h[, solver])
combo and let you actually read them (they're gzip-compressed, not plain text).

Scans results/{model}/**/*.json* AND its sibling campaign dirs
results/sweeps*/{model}/**/*.json* (same convention as scripts/viz/plot_ttc.py's
loader) and filters by config.size / config.h / config.sampler+sampling_method
read from each file itself, not by guessing directory names.

Usage:
    # list matching files with a one-line summary each
    python scripts/inspect_results.py --model tfim_1d --N 64 --h 0.5

    # restrict to one solver: bare sampler ("fpga") or "sampler/method" ("velox/simulated_annealing")
    python scripts/inspect_results.py --model tfim_1d --N 64 --h 0.5 --solver fpga

    # decompress every match to plain, indented .json files you can open/cat/edit
    python scripts/inspect_results.py --model tfim_1d --N 64 --h 0.5 --solver fpga --dump-dir /tmp/inspect

    # dump full decompressed content of every match straight to the terminal
    python scripts/inspect_results.py --model tfim_1d --N 64 --h 0.5 --solver fpga --print

    # decompress the one matching file (narrow with --seed) and open it in an
    # editor window — writes to a system temp file, nothing to clean up after
    python scripts/inspect_results.py --model tfim_1d --N 64 --h 0.5 --solver fpga --seed 0 --open
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from itertools import chain
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

from helpers import load_result_json


def find_results(results_dir: Path, model: str, N: int, h: float,
                  solver: str | None = None, seed: int | None = None) -> list[Path]:
    """Return every result_*.json[.gz] path matching (model, N, h[, solver, seed]).

    Searches results_dir/model/** plus sibling results_dir/sweeps*/model/**
    (FPGA/Velox campaign directories aren't nested under results_dir/model).
    """
    roots = [results_dir / model]
    roots += [d / model for d in sorted(results_dir.glob("sweeps*")) if (d / model).exists()]

    want_sampler, want_method = (None, None)
    if solver is not None:
        if "/" in solver:
            want_sampler, want_method = solver.split("/", 1)
        else:
            want_sampler = solver

    matches = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(chain(root.rglob("*.json"), root.rglob("*.json.gz"))):
            try:
                data = load_result_json(path)
            except Exception as e:
                print(f"  [skip] {path.name}: {e}", file=sys.stderr)
                continue
            cfg = data.get("config", {})
            if cfg.get("size") != N:
                continue
            cfg_h = cfg.get("h")
            if cfg_h is None or abs(float(cfg_h) - h) > 1e-9:
                continue
            if want_sampler is not None and cfg.get("sampler") != want_sampler:
                continue
            if want_method is not None and cfg.get("sampling_method") != want_method:
                continue
            if seed is not None and cfg.get("seed") != seed:
                continue
            matches.append(path)

    return matches


def open_in_editor(path: Path) -> None:
    """Launch path in whatever editor is available, in its own window/tab."""
    opener = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    for candidate in filter(None, [opener, "code", "xdg-open", "open"]):
        if shutil.which(candidate.split()[0]) is None:
            continue
        try:
            subprocess.Popen([candidate, str(path)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Opened with `{candidate}`: {path}")
            return
        except OSError:
            continue
    print(f"Could not find an editor to open it automatically. File is at: {path}")


def print_summary(paths: list[Path]) -> None:
    if not paths:
        print("No matching result files found.")
        return

    print(f"{len(paths)} matching files:\n")
    header = f"{'seed':>5}  {'sampler/method':<28}  {'final_energy':>14}  {'exact_energy':>14}  {'error':>10}  file"
    print(header)
    print("-" * len(header))
    for path in paths:
        data = load_result_json(path)
        cfg = data.get("config", {})
        final_e = data.get("final_energy")
        exact_e = data.get("exact_energy")
        err = None
        if final_e is not None and exact_e:
            err = abs(final_e - exact_e) / abs(exact_e)
        print(
            f"{cfg.get('seed', '?'):>5}  "
            f"{cfg.get('sampler', '?') + '/' + cfg.get('sampling_method', '?'):<28}  "
            f"{final_e if final_e is not None else float('nan'):>14.6f}  "
            f"{exact_e if exact_e is not None else float('nan'):>14.6f}  "
            f"{err if err is not None else float('nan'):>10.4%}  "
            f"{path}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="e.g. tfim_1d")
    parser.add_argument("--N", type=int, required=True, help="System size")
    parser.add_argument("--h", type=float, required=True, help="Transverse field value")
    parser.add_argument("--solver", default=None,
                         help="Filter to a sampler ('fpga') or 'sampler/method' ('velox/simulated_annealing')")
    parser.add_argument("--seed", type=int, default=None, help="Filter to one seed")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--dump-dir", type=Path, default=None,
                         help="Write a decompressed, pretty-printed .json copy of every match here")
    parser.add_argument("--print", dest="print_full", action="store_true",
                         help="Print full decompressed content of every match to stdout")
    parser.add_argument("--open", dest="open_editor", action="store_true",
                         help="Decompress the (single) match to a temp file and open it in an "
                              "editor window — nothing to clean up, no --dump-dir needed. "
                              "Requires exactly one match; narrow with --solver/--seed.")
    args = parser.parse_args()

    paths = find_results(args.results_dir, args.model, args.N, args.h, args.solver, args.seed)

    if args.open_editor:
        if len(paths) != 1:
            print(f"--open requires exactly one match, found {len(paths)}. "
                  "Narrow with --solver/--seed:", file=sys.stderr)
            print_summary(paths)
            sys.exit(1)
        data = load_result_json(paths[0])
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix=paths[0].stem + "_", delete=False
        )
        json.dump(data, tmp, indent=2)
        tmp.close()
        open_in_editor(Path(tmp.name))
    elif args.dump_dir is not None:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            data = load_result_json(path)
            out_name = path.name[:-3] if path.name.endswith(".gz") else path.name
            out_path = args.dump_dir / out_name
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"wrote {out_path}")
    elif args.print_full:
        for path in paths:
            print(f"===== {path} =====")
            print(json.dumps(load_result_json(path), indent=2))
    else:
        print_summary(paths)


if __name__ == "__main__":
    main()
