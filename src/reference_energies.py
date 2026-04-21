"""
Master store for exact ground state reference energies.

Values are computed on first access and cached in reference_energies.json.
This file is the single source of truth — exact_energy fields in result
.json files are informational only and must not be re-imported as references.

1D: exact free-fermion diagonalization (exact for any N, O(N))
2D: exact Lanczos diagonalization via NetKet (L ≤ 4 only; raises otherwise)
"""

import fcntl
import json
import threading
from pathlib import Path

REFERENCE_FILE = Path(__file__).parent / "reference_energies.json"
_LOCK_FILE = REFERENCE_FILE.with_suffix(".lock")
_write_lock = threading.Lock()


def _key(model: str, size: int, h: float) -> str:
    # size is N for 1d, L (linear dimension) for 2d
    return f"{model}_N{size}_h{h:.10g}"


def _load() -> dict:
    if not REFERENCE_FILE.exists():
        return {}
    with REFERENCE_FILE.open() as f:
        return json.load(f)


def _save(data: dict) -> None:
    tmp = REFERENCE_FILE.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.rename(REFERENCE_FILE)


def lookup(model: str, size: int, h: float) -> float | None:
    """Return cached reference energy without triggering computation, or None."""
    val = _load().get(_key(model, size, h))
    return float(val) if val is not None else None


def get_or_compute(model: str, size: int, h: float, compute_fn) -> float:
    """
    Return cached reference energy or compute, store, and return it.

    Thread-safe and multi-process-safe (fcntl exclusive lock on .lock file).
    compute_fn must raise NotImplementedError if no exact value can be obtained
    within the required accuracy — this propagates to the caller unchanged.
    """
    key = _key(model, size, h)

    data = _load()
    if key in data:
        return float(data[key])

    with _write_lock:
        _LOCK_FILE.touch(exist_ok=True)
        with _LOCK_FILE.open("r") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                data = _load()
                if key not in data:
                    data[key] = compute_fn()
                    _save(data)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    return float(data[key])
