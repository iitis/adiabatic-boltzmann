"""
GPU energy metering for the "watt-hours to solution" (WHS) metric.

Named gpu_energy_* (never bare "energy") to avoid collision with the
Hamiltonian energy (E, E_loc, E_exact, error_per_spin) used everywhere
else in this codebase.

Measures energy by polling `nvidia-smi --query-gpu=power.draw` on a
background thread and integrating power over time (trapezoidal rule).
This is a coarse estimate (subprocess-poll resolution, not an on-device
energy counter) but requires no extra dependency and works on any
machine with the NVIDIA driver installed.
"""

import shutil
import subprocess
import threading
import time


def gpu_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _read_power_draw_w(gpu_index: int) -> float:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu_index),
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    ).stdout.strip()
    return float(out.splitlines()[0])


class GPUEnergyMeter:
    """Context manager measuring GPU energy consumption (joules) over its block.

    Usage:
        with GPUEnergyMeter() as meter:
            ...  # GPU work
        joules = meter.energy_j

    Raises RuntimeError on exit if fewer than 2 power samples were collected
    (e.g. the block finished faster than one poll interval, or nvidia-smi
    failed mid-run) — callers must not treat that as "zero energy consumed".
    """

    def __init__(self, gpu_index: int = 0, poll_interval_s: float = 0.5):
        self._gpu_index = gpu_index
        self._poll_interval_s = poll_interval_s
        self._samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0

    def _poll_loop(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                watts = _read_power_draw_w(self._gpu_index)
                self._samples.append((time.perf_counter() - self._start_time, watts))
            except (subprocess.SubprocessError, ValueError, OSError):
                pass
            elapsed = time.perf_counter() - t0
            self._stop.wait(max(0.0, self._poll_interval_s - elapsed))

    def __enter__(self):
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join(timeout=self._poll_interval_s + 5)

    @property
    def energy_j(self) -> float:
        if len(self._samples) < 2:
            raise RuntimeError(
                f"GPUEnergyMeter collected only {len(self._samples)} power "
                "sample(s) — cannot integrate energy. The measured block may "
                "be shorter than poll_interval_s, or nvidia-smi may be "
                "failing silently."
            )
        joules = 0.0
        for (t0, p0), (t1, p1) in zip(self._samples, self._samples[1:]):
            joules += 0.5 * (p0 + p1) * (t1 - t0)
        return joules
