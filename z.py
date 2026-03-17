#!/usr/bin/env python3
"""
What this script does
---------------------
1. Re-implements the paper's hybrid loop:
   RBM ansatz + variational Monte Carlo + stochastic reconfiguration,
   with D-Wave sampling of the RBM joint distribution.
2. Replaces the paper's Chimera embedding with Pegasus/Zephyr biclique
   embeddings for the logical K_{N,N} RBM graph.

The script generates per-topology analogs of the paper's QPU figures:
- 1D TFIM, L=64, h=0.5, sweeping anneal time.
- 2D TFIM, 8x8, sweeping transverse field.
- Adaptive beta_x traces (paper's Fig. 5 analogue).

Quick start
-----------
    export DWAVE_API_TOKEN=...your token...
    python reproduce_gardas_rbm_pegasus_zephyr_mp.py \
        --topologies pegasus zephyr \
        --processes 2 \
        --output-dir qpu_reproduction

A smaller smoke test:
    python reproduce_gardas_rbm_pegasus_zephyr_mp.py \
        --topologies pegasus \
        --iterations 25 \
        --target-samples 2000 \
        --anneal-times-1d 2 20 \
        --fields-2d 0.5 3.044 \
        --output-dir smoke_test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


try:
    import dimod
    import dwave_networkx as dnx
    from dwave.preprocessing import SpinReversalTransformComposite
    from dwave.samplers import SimulatedAnnealingSampler
    from dwave.system import DWaveSampler, FixedEmbeddingComposite
    from minorminer import find_embedding
    from minorminer.busclique import busgraph_cache
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires the D-Wave Ocean SDK. Install it with:\n"
        "    pip install 'dwave-ocean-sdk>=9.0' numpy matplotlib\n"
        f"Original import error: {exc}"
    )


Array = np.ndarray


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QPUAccessConfig:
    topology: str
    backend: str = "qpu"  # qpu | sim
    solver_name: Optional[str] = None
    token: Optional[str] = None
    endpoint: Optional[str] = None
    region: Optional[str] = None
    max_runtime_fraction: float = 0.85
    programming_thermalization: Optional[float] = None
    readout_thermalization: Optional[float] = None
    num_spin_reversal_transforms: int = 4
    chain_strength_prefactor: float = 1.5
    sim_graph_size: Optional[int] = None


@dataclass(frozen=True)
class RunConfig:
    name: str
    dim: int
    Lx: int
    Ly: int
    field: float
    annealing_time: float
    alpha: float = 1.0
    periodic: bool = True
    iterations: int = 300
    target_samples: int = 10_000
    requested_reads_per_child_call: Optional[int] = None
    learning_rate: float = 0.2
    diag_shift: float = 1.0e-3
    cg_tol: float = 1.0e-8
    cg_maxiter: int = 200
    param_init_scale: float = 1.0e-2
    param_clip: float = 3.0
    beta_x_init: float = 2.0
    beta_adapt: float = 0.05
    beta_min: float = 0.05
    beta_max: float = 20.0
    seed: int = 1234
    reference_energy_per_spin: Optional[float] = None


@dataclass
class RBMParameters:
    a: Array  # (N,)
    b: Array  # (M,)
    W: Array  # (M, N)

    def copy(self) -> "RBMParameters":
        return RBMParameters(self.a.copy(), self.b.copy(), self.W.copy())

    @property
    def n_visible(self) -> int:
        return int(self.a.shape[0])

    @property
    def n_hidden(self) -> int:
        return int(self.b.shape[0])

    def clip_(self, bound: float) -> None:
        np.clip(self.a, -bound, bound, out=self.a)
        np.clip(self.b, -bound, bound, out=self.b)
        np.clip(self.W, -bound, bound, out=self.W)


# ---------------------------------------------------------------------------
# Physics / RBM utilities
# ---------------------------------------------------------------------------


def log2cosh(x: Array) -> Array:
    """Stable computation of log(2 cosh(x))."""
    return np.logaddexp(x, -x)


def build_lattice(dim: int, Lx: int, Ly: int, periodic: bool) -> Tuple[int, Array, Array]:
    """
    Build the nearest-neighbor TFIM graph over visible spins.

    Returns
    -------
    N, edge_u, edge_v
        Number of visible sites and parallel arrays describing the undirected
        nearest-neighbor edges.
    """
    if dim == 1:
        N = Lx
        edges: List[Tuple[int, int]] = [(i, i + 1) for i in range(Lx - 1)]
        if periodic and Lx > 2:
            edges.append((Lx - 1, 0))
        edge_u = np.asarray([u for u, _ in edges], dtype=np.int64)
        edge_v = np.asarray([v for _, v in edges], dtype=np.int64)
        return N, edge_u, edge_v

    if dim == 2:
        N = Lx * Ly
        edges_set = set()

        def idx(x: int, y: int) -> int:
            return y * Lx + x

        for y in range(Ly):
            for x in range(Lx):
                here = idx(x, y)
                if x + 1 < Lx:
                    right = idx(x + 1, y)
                    edges_set.add(tuple(sorted((here, right))))
                elif periodic and Lx > 2:
                    right = idx(0, y)
                    edges_set.add(tuple(sorted((here, right))))

                if y + 1 < Ly:
                    up = idx(x, y + 1)
                    edges_set.add(tuple(sorted((here, up))))
                elif periodic and Ly > 2:
                    up = idx(x, 0)
                    edges_set.add(tuple(sorted((here, up))))

        edges = sorted(edges_set)
        edge_u = np.asarray([u for u, _ in edges], dtype=np.int64)
        edge_v = np.asarray([v for _, v in edges], dtype=np.int64)
        return N, edge_u, edge_v

    raise ValueError(f"Unsupported dimension: {dim}")


def exact_tfim_1d_energy_per_spin(L: int, field: float) -> float:
    """
    Exact finite-size ground-state energy density for the periodic 1D TFIM.

    For even L and positive field, the even-parity sector is obtained from the
    antiperiodic Jordan-Wigner momenta k_m = (2m+1) pi / L.
    """
    if L <= 0:
        raise ValueError("L must be positive")
    ks = (2 * np.arange(L) + 1) * math.pi / L
    eps = 2.0 * np.sqrt(1.0 + field * field - 2.0 * field * np.cos(ks))
    return float(-np.sum(eps) / (2.0 * L))


def init_rbm_parameters(N: int, M: int, scale: float, rng: np.random.Generator) -> RBMParameters:
    return RBMParameters(
        a=scale * rng.standard_normal(N),
        b=scale * rng.standard_normal(M),
        W=scale * rng.standard_normal((M, N)),
    )


def theta_batch(V: Array, params: RBMParameters) -> Array:
    return params.b[None, :] + V @ params.W.T


def gradient_hidden_batch(theta: Array) -> Array:
    return np.tanh(theta)


def local_energy_batch(
    V: Array,
    params: RBMParameters,
    field: float,
    edge_u: Array,
    edge_v: Array,
) -> Array:
    """
    Local energy for the positive-amplitude RBM ansatz used in the paper.

    E_loc(v) = -h sum_i Psi(v^i_flip)/Psi(v) - sum_<ij> v_i v_j
    """
    ns, N = V.shape
    theta = theta_batch(V, params)
    base = log2cosh(theta)

    transverse_sum = np.zeros(ns, dtype=np.float64)
    for i in range(N):
        delta = -2.0 * np.outer(V[:, i], params.W[:, i])
        log_ratio = -params.a[i] * V[:, i] + 0.5 * np.sum(log2cosh(theta + delta) - base, axis=1)
        transverse_sum += np.exp(log_ratio)

    diagonal_term = -np.sum(V[:, edge_u] * V[:, edge_v], axis=1)
    return -field * transverse_sum + diagonal_term


class SRLinearSystem:
    """
    Matrix-free stochastic reconfiguration system S x = F.

    Uses the exact RBM feature map corresponding to Eq. (15) in the paper:
        D_a_i = 0.5 v_i
        D_b_j = 0.5 tanh(theta_j)
        D_W_ji = 0.5 v_i tanh(theta_j)
    """

    def __init__(self, V: Array, H: Array, E: Array, diag_shift: float):
        self.V = np.asarray(V, dtype=np.float64)
        self.H = np.asarray(H, dtype=np.float64)
        self.E = np.asarray(E, dtype=np.float64)
        self.ns, self.N = self.V.shape
        self.M = self.H.shape[1]
        self.diag_shift = float(diag_shift)

        self.mu_a = 0.5 * self.V.mean(axis=0)
        self.mu_b = 0.5 * self.H.mean(axis=0)
        self.mu_W = 0.5 * (self.H.T @ self.V) / self.ns

        centered_E = self.E - self.E.mean()
        self.F_a = 0.5 * (centered_E @ self.V) / self.ns
        self.F_b = 0.5 * (centered_E @ self.H) / self.ns
        self.F_W = 0.5 * (self.H.T @ (centered_E[:, None] * self.V)) / self.ns

    def pack(self, a: Array, b: Array, W: Array) -> Array:
        return np.concatenate([a.ravel(), b.ravel(), W.ravel()])

    def unpack(self, x: Array) -> Tuple[Array, Array, Array]:
        a = x[: self.N]
        b = x[self.N : self.N + self.M]
        W = x[self.N + self.M :].reshape(self.M, self.N)
        return a, b, W

    @property
    def force(self) -> Array:
        return self.pack(self.F_a, self.F_b, self.F_W)

    def matvec(self, x: Array) -> Array:
        xa, xb, xW = self.unpack(x)
        z = 0.5 * (self.V @ xa + self.H @ xb + np.einsum("sm,mn,sn->s", self.H, xW, self.V))
        z -= float(self.mu_a @ xa + self.mu_b @ xb + np.sum(self.mu_W * xW))

        out_a = 0.5 * (z @ self.V) / self.ns + self.diag_shift * xa
        out_b = 0.5 * (z @ self.H) / self.ns + self.diag_shift * xb
        out_W = 0.5 * (self.H.T @ (z[:, None] * self.V)) / self.ns + self.diag_shift * xW
        return self.pack(out_a, out_b, out_W)


def conjugate_gradient(matvec, b: Array, tol: float = 1.0e-8, maxiter: int = 200) -> Tuple[Array, Dict[str, float]]:
    x = np.zeros_like(b)
    r = b - matvec(x)
    p = r.copy()
    rs_old = float(r @ r)
    info = {"iterations": 0, "residual_norm": math.sqrt(rs_old)}

    if rs_old <= tol * tol:
        return x, info

    for it in range(1, maxiter + 1):
        Ap = matvec(p)
        denom = float(p @ Ap)
        if abs(denom) < 1.0e-30:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)
        info = {"iterations": it, "residual_norm": math.sqrt(rs_new)}
        if rs_new <= tol * tol:
            return x, info
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    return x, info


# ---------------------------------------------------------------------------
# QPU sampler / embedding / runtime-safe batching
# ---------------------------------------------------------------------------


class TopologySampler:
    """
    Runtime-safe logical RBM sampler for one topology.

    The logical problem is always the complete bipartite RBM graph K_{N,M}. We
    prefer busclique biclique embeddings on Pegasus/Zephyr and fall back to
    generic minorminer if needed.

    The key runtime-safety feature is `safe_child_reads()`: it estimates QPU
    access time from the selected solver, the fixed embedding size, and the
    requested anneal time, then uses binary search to find the largest
    `num_reads` that remains safely below the solver's runtime limit.
    """

    def __init__(self, access: QPUAccessConfig):
        self.access = access
        self.base_sampler = self._make_base_sampler()
        self.target_graph = self._to_networkx_graph()
        self.embedding_cache: Dict[Tuple[int, int], Dict[str, Tuple[int, ...]]] = {}
        self.composite_cache: Dict[Tuple[int, int], FixedEmbeddingComposite] = {}
        self.runtime_cache: Dict[Tuple[int, int, float], Tuple[int, float]] = {}

    def _make_base_sampler(self):
        if self.access.backend == "qpu":
            solver_selector: object
            if self.access.solver_name:
                solver_selector = self.access.solver_name
            else:
                solver_selector = dict(topology__type=self.access.topology)

            kwargs: Dict[str, object] = {"solver": solver_selector}
            if self.access.token:
                kwargs["token"] = self.access.token
            if self.access.endpoint:
                kwargs["endpoint"] = self.access.endpoint
            if self.access.region:
                kwargs["region"] = self.access.region
            return DWaveSampler(**kwargs)

        # Local structured fallback for development/testing.
        if self.access.topology == "pegasus":
            m = self.access.sim_graph_size or 16
            graph = dnx.pegasus_graph(m)
        else:
            m = self.access.sim_graph_size or 8
            graph = dnx.zephyr_graph(m)
        return dimod.StructureComposite(SimulatedAnnealingSampler(), graph.nodes, graph.edges)

    def _to_networkx_graph(self):
        if self.access.backend == "qpu":
            return self.base_sampler.to_networkx_graph()
        if self.access.topology == "pegasus":
            return dnx.pegasus_graph(self.access.sim_graph_size or 16)
        return dnx.zephyr_graph(self.access.sim_graph_size or 8)

    @staticmethod
    def _merge_biclique_embedding(obj: object) -> Dict[str, Tuple[int, ...]]:
        """
        Normalize whatever the biclique helper returns into a single dict.

        Ocean versions have historically exposed biclique embeddings either as a
        single mapping or as a pair of mappings, one for each shore of the
        biclique. This helper accepts both forms.
        """
        if isinstance(obj, dict):
            return {str(k): tuple(v) for k, v in obj.items()}
        if isinstance(obj, tuple) and len(obj) == 2 and all(isinstance(x, dict) for x in obj):
            merged: Dict[str, Tuple[int, ...]] = {}
            merged.update({str(k): tuple(v) for k, v in obj[0].items()})
            merged.update({str(k): tuple(v) for k, v in obj[1].items()})
            return merged
        raise TypeError(f"Unsupported embedding object type: {type(obj)!r}")

    def get_labels(self, n_visible: int, n_hidden: int) -> Tuple[List[str], List[str]]:
        return [f"v{i}" for i in range(n_visible)], [f"h{j}" for j in range(n_hidden)]

    def get_embedding(self, n_visible: int, n_hidden: int) -> Dict[str, Tuple[int, ...]]:
        key = (n_visible, n_hidden)
        if key in self.embedding_cache:
            return self.embedding_cache[key]

        visible_labels, hidden_labels = self.get_labels(n_visible, n_hidden)

        embedding: Dict[str, Tuple[int, ...]] = {}
        try:
            cache = busgraph_cache(self.target_graph)
            raw = cache.find_biclique_embedding(visible_labels, hidden_labels)
            embedding = self._merge_biclique_embedding(raw)
        except Exception:
            embedding = {}

        if not embedding:
            source_edges = [(v, h) for v in visible_labels for h in hidden_labels]
            raw = find_embedding(source_edges, list(self.target_graph.edges), random_seed=17)
            embedding = {str(k): tuple(v) for k, v in raw.items()}

        expected = set(visible_labels) | set(hidden_labels)
        if not embedding or set(embedding.keys()) != expected:
            raise RuntimeError(
                f"Failed to embed K_{{{n_visible},{n_hidden}}} on topology {self.access.topology}. "
                "Try a smaller system or a different solver."
            )

        self.embedding_cache[key] = embedding
        return embedding

    def get_composite(self, n_visible: int, n_hidden: int):
        key = (n_visible, n_hidden)
        if key not in self.composite_cache:
            embedding = self.get_embedding(n_visible, n_hidden)
            composite = FixedEmbeddingComposite(
                SpinReversalTransformComposite(self.base_sampler),
                embedding,
            )
            self.composite_cache[key] = composite
        return self.composite_cache[key]

    def physical_qubit_count(self, n_visible: int, n_hidden: int) -> int:
        embedding = self.get_embedding(n_visible, n_hidden)
        return int(sum(len(chain) for chain in embedding.values()))

    def metadata(self) -> Dict[str, object]:
        meta: Dict[str, object] = {
            "backend": self.access.backend,
            "topology_requested": self.access.topology,
        }
        if self.access.backend == "qpu":
            props = getattr(self.base_sampler, "properties", {})
            meta.update(
                {
                    "solver_name": getattr(getattr(self.base_sampler, "solver", None), "name", None),
                    "chip_id": props.get("chip_id"),
                    "topology": props.get("topology"),
                    "num_qubits": props.get("num_qubits"),
                    "problem_run_duration_range": props.get("problem_run_duration_range"),
                    "num_reads_range": props.get("num_reads_range"),
                    "annealing_time_range": props.get("annealing_time_range"),
                }
            )
        return meta

    def _estimate_runtime_us(self, n_physical: int, num_reads: int, annealing_time: float) -> float:
        if self.access.backend != "qpu":
            return 0.0
        kwargs: Dict[str, object] = {"num_reads": int(num_reads), "annealing_time": float(annealing_time)}
        if self.access.programming_thermalization is not None:
            kwargs["programming_thermalization"] = float(self.access.programming_thermalization)
        if self.access.readout_thermalization is not None:
            kwargs["readout_thermalization"] = float(self.access.readout_thermalization)
        return float(self.base_sampler.solver.estimate_qpu_access_time(n_physical, **kwargs))

    def safe_child_reads(self, n_visible: int, n_hidden: int, annealing_time: float) -> Tuple[int, float]:
        """
        Largest safe `num_reads` for a single child QPU submission.

        Note: spin-reversal transforms are handled by the composite as multiple
        *independent* child submissions. Therefore the runtime limit is checked
        per child submission, not after multiplying by the number of transforms.
        """
        key = (n_visible, n_hidden, float(annealing_time))
        if key in self.runtime_cache:
            return self.runtime_cache[key]

        if self.access.backend != "qpu":
            self.runtime_cache[key] = (10_000, 0.0)
            return self.runtime_cache[key]

        props = getattr(self.base_sampler, "properties", {})
        runtime_limit = float(props.get("problem_run_duration_range", [0.0, 1_000_000.0])[1])
        read_hi = int(props.get("num_reads_range", [1, 10_000])[1])
        target_limit = self.access.max_runtime_fraction * runtime_limit
        n_physical = self.physical_qubit_count(n_visible, n_hidden)

        est1 = self._estimate_runtime_us(n_physical, 1, annealing_time)
        if est1 > target_limit:
            raise RuntimeError(
                "Even a single read would exceed the chosen safe runtime threshold. "
                f"estimated={est1:.1f}us, threshold={target_limit:.1f}us. "
                "Reduce annealing time or thermalization parameters."
            )

        lo, hi = 1, read_hi
        while lo < hi:
            mid = (lo + hi + 1) // 2
            est = self._estimate_runtime_us(n_physical, mid, annealing_time)
            if est <= target_limit:
                lo = mid
            else:
                hi = mid - 1

        safe_reads = int(lo)
        safe_est = float(self._estimate_runtime_us(n_physical, safe_reads, annealing_time))
        self.runtime_cache[key] = (safe_reads, safe_est)
        return self.runtime_cache[key]

    @staticmethod
    def _heuristic_chain_strength(bqm: dimod.BinaryQuadraticModel, prefactor: float) -> float:
        max_linear = max((abs(v) for v in bqm.linear.values()), default=0.0)
        max_quadratic = max((abs(v) for v in bqm.quadratic.values()), default=0.0)
        scale = max(1.0e-3, max_linear, max_quadratic)
        return float(prefactor * scale)

    @staticmethod
    def _visible_samples_from_sampleset(sampleset, n_visible: int) -> Array:
        variables = list(sampleset.variables)
        cols = [variables.index(f"v{i}") for i in range(n_visible)]
        raw = np.asarray(sampleset.record.sample[:, cols], dtype=np.int8)
        counts = np.asarray(sampleset.record.num_occurrences, dtype=np.int64)
        expanded = np.repeat(raw, counts, axis=0)
        return expanded.astype(np.int8, copy=False)

    def collect_visible_samples(
        self,
        params: RBMParameters,
        beta_x: float,
        annealing_time: float,
        target_samples: int,
        requested_reads_per_child_call: Optional[int] = None,
    ) -> Tuple[Array, Dict[str, object]]:
        n_visible = params.n_visible
        n_hidden = params.n_hidden
        visible_labels, hidden_labels = self.get_labels(n_visible, n_hidden)
        composite = self.get_composite(n_visible, n_hidden)

        # To sample p(v,h) ∝ exp(a·v + b·h + hWv), submit the negated Ising biases
        # because Ocean minimizes E = sum h_i s_i + sum J_ij s_i s_j.
        h: Dict[str, float] = {}
        J: Dict[Tuple[str, str], float] = {}
        for i, label in enumerate(visible_labels):
            h[label] = -float(params.a[i] / beta_x)
        for j, label in enumerate(hidden_labels):
            h[label] = -float(params.b[j] / beta_x)
        for j, h_label in enumerate(hidden_labels):
            for i, v_label in enumerate(visible_labels):
                J[(h_label, v_label)] = -float(params.W[j, i] / beta_x)

        bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
        chain_strength = self._heuristic_chain_strength(bqm, self.access.chain_strength_prefactor)

        safe_reads, safe_est = self.safe_child_reads(n_visible, n_hidden, annealing_time)
        child_reads = safe_reads
        if requested_reads_per_child_call is not None:
            child_reads = min(child_reads, int(requested_reads_per_child_call))
        child_reads = max(1, int(child_reads))

        srt = max(1, int(self.access.num_spin_reversal_transforms))
        effective_samples_per_batch = child_reads * srt
        n_batches = max(1, math.ceil(target_samples / effective_samples_per_batch))

        batch_arrays: List[Array] = []
        qpu_access_times: List[float] = []
        chain_break_fracs: List[float] = []

        for _ in range(n_batches):
            kwargs: Dict[str, object] = {
                "chain_strength": chain_strength,
                "num_reads": child_reads,
                "num_spin_reversal_transforms": srt,
            }
            if self.access.backend == "qpu":
                kwargs["annealing_time"] = float(annealing_time)
                kwargs["auto_scale"] = True
                if self.access.programming_thermalization is not None:
                    kwargs["programming_thermalization"] = float(self.access.programming_thermalization)
                if self.access.readout_thermalization is not None:
                    kwargs["readout_thermalization"] = float(self.access.readout_thermalization)

            sampleset = composite.sample(bqm, **kwargs)
            V = self._visible_samples_from_sampleset(sampleset, n_visible)
            batch_arrays.append(V)

            timing = {}
            if isinstance(getattr(sampleset, "info", None), Mapping):
                timing = dict(sampleset.info.get("timing", {}))
            if "qpu_access_time" in timing:
                qpu_access_times.append(float(timing["qpu_access_time"]))

            names = set(sampleset.record.dtype.names)
            if "chain_break_fraction" in names:
                c = np.asarray(sampleset.record.chain_break_fraction, dtype=np.float64)
                occ = np.asarray(sampleset.record.num_occurrences, dtype=np.int64)
                c_expanded = np.repeat(c, occ)
                if c_expanded.size:
                    chain_break_fracs.append(float(np.mean(c_expanded)))

        visible = np.concatenate(batch_arrays, axis=0)[:target_samples]
        props = getattr(self.base_sampler, "properties", {}) if self.access.backend == "qpu" else {}

        meta = {
            "beta_x": float(beta_x),
            "physical_qubits": int(self.physical_qubit_count(n_visible, n_hidden)),
            "chain_strength": float(chain_strength),
            "safe_reads_per_child_call": int(safe_reads),
            "reads_per_child_call": int(child_reads),
            "num_spin_reversal_transforms": int(srt),
            "requested_target_samples": int(target_samples),
            "effective_samples_per_batch": int(effective_samples_per_batch),
            "n_batches": int(n_batches),
            "estimated_runtime_us_per_child": float(safe_est),
            "runtime_limit_us": float(props.get("problem_run_duration_range", [0.0, 1_000_000.0])[1]) if props else 0.0,
            "avg_qpu_access_time_us": float(np.mean(qpu_access_times)) if qpu_access_times else 0.0,
            "avg_chain_break_fraction": float(np.mean(chain_break_fracs)) if chain_break_fracs else 0.0,
        }
        return visible, meta

    def close(self) -> None:
        sampler = self.base_sampler
        try:
            if hasattr(sampler, "close"):
                sampler.close()  # type: ignore[attr-defined]
                return
        except Exception:
            pass
        try:
            client = getattr(sampler, "client", None)
            if client is not None and hasattr(client, "close"):
                client.close()  # type: ignore[attr-defined]
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Training, saving, plotting
# ---------------------------------------------------------------------------


def save_json(obj: object, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


def save_history_csv(history: Sequence[Mapping[str, object]], path: Path) -> None:
    if not history:
        return
    keys = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def run_single_training(run: RunConfig, qpu: TopologySampler, run_dir: Path) -> Dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(run.seed)

    N, edge_u, edge_v = build_lattice(run.dim, run.Lx, run.Ly, run.periodic)
    M = int(round(run.alpha * N))
    params = init_rbm_parameters(N, M, run.param_init_scale, rng)

    history: List[Dict[str, object]] = []
    best_params = params.copy()
    best_energy_total = math.inf
    beta_x = float(run.beta_x_init)
    prev_energy_total: Optional[float] = None

    t0 = time.time()
    for iteration in range(1, run.iterations + 1):
        iter_start = time.time()
        V, sample_meta = qpu.collect_visible_samples(
            params=params,
            beta_x=beta_x,
            annealing_time=run.annealing_time,
            target_samples=run.target_samples,
            requested_reads_per_child_call=run.requested_reads_per_child_call,
        )
        theta = theta_batch(V, params)
        Hfeat = gradient_hidden_batch(theta)
        Eloc = local_energy_batch(V, params, run.field, edge_u, edge_v)

        sr = SRLinearSystem(V, Hfeat, Eloc, run.diag_shift)
        step, cg_info = conjugate_gradient(sr.matvec, sr.force, tol=run.cg_tol, maxiter=run.cg_maxiter)
        da, db, dW = sr.unpack(step)

        params.a -= run.learning_rate * da
        params.b -= run.learning_rate * db
        params.W -= run.learning_rate * dW
        params.clip_(run.param_clip)

        energy_total = float(np.mean(Eloc))
        energy_per_spin = energy_total / N

        if prev_energy_total is not None and energy_total > prev_energy_total:
            factor = 1.0 + run.beta_adapt if rng.random() < 0.5 else 1.0 - run.beta_adapt
            beta_x = float(np.clip(beta_x * factor, run.beta_min, run.beta_max))
        prev_energy_total = energy_total

        if energy_total < best_energy_total:
            best_energy_total = energy_total
            best_params = params.copy()

        row: Dict[str, object] = {
            "iteration": iteration,
            "energy_total": energy_total,
            "energy_per_spin": energy_per_spin,
            "best_energy_per_spin": best_energy_total / N,
            "beta_x": beta_x,
            "cg_iterations": int(cg_info["iterations"]),
            "cg_residual_norm": float(cg_info["residual_norm"]),
            "num_samples": int(V.shape[0]),
            "iteration_seconds": float(time.time() - iter_start),
            "chain_strength": sample_meta["chain_strength"],
            "reads_per_child_call": sample_meta["reads_per_child_call"],
            "safe_reads_per_child_call": sample_meta["safe_reads_per_child_call"],
            "num_spin_reversal_transforms": sample_meta["num_spin_reversal_transforms"],
            "n_batches": sample_meta["n_batches"],
            "estimated_runtime_us_per_child": sample_meta["estimated_runtime_us_per_child"],
            "runtime_limit_us": sample_meta["runtime_limit_us"],
            "avg_qpu_access_time_us": sample_meta["avg_qpu_access_time_us"],
            "avg_chain_break_fraction": sample_meta["avg_chain_break_fraction"],
        }
        if run.reference_energy_per_spin is not None:
            relerr = abs((energy_per_spin - run.reference_energy_per_spin) / run.reference_energy_per_spin)
            row["reference_energy_per_spin"] = float(run.reference_energy_per_spin)
            row["relative_error"] = float(relerr)
        history.append(row)

        if iteration == 1 or iteration % 10 == 0 or iteration == run.iterations:
            msg = (
                f"[{run.name}] {iteration:4d}/{run.iterations} "
                f"E/N={energy_per_spin:+.8f} beta_x={beta_x:.4f} "
                f"reads={sample_meta['reads_per_child_call']}x{sample_meta['num_spin_reversal_transforms']} "
                f"batches={sample_meta['n_batches']}"
            )
            print(msg, flush=True)

    elapsed = time.time() - t0

    save_history_csv(history, run_dir / "history.csv")
    np.savez_compressed(run_dir / "best_params.npz", a=best_params.a, b=best_params.b, W=best_params.W)
    np.savez_compressed(run_dir / "final_params.npz", a=params.a, b=params.b, W=params.W)

    summary = {
        "run": asdict(run),
        "n_visible": int(N),
        "n_hidden": int(M),
        "interaction_edges": int(edge_u.size),
        "best_energy_per_spin": float(best_energy_total / N),
        "runtime_seconds": float(elapsed),
        "sampler_metadata": qpu.metadata(),
    }
    save_json(summary, run_dir / "summary.json")

    return {
        "name": run.name,
        "history": history,
        "summary": summary,
        "output_dir": str(run_dir),
    }


def plot_1d_family(histories: Mapping[str, Sequence[Mapping[str, object]]], ref_energy: float, outpath: Path, title: str) -> None:
    taus = sorted(histories.keys(), key=lambda x: float(x))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    for tau in taus:
        hist = histories[tau]
        x = np.asarray([int(row["iteration"]) for row in hist], dtype=int)
        e = np.asarray([float(row["energy_per_spin"]) for row in hist], dtype=float)
        axes[0].plot(x, e, label=rf"$\tau={float(tau):g}$")

    axes[0].axhline(ref_energy, linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("energy per spin")
    axes[0].set_title(title)
    axes[0].legend(fontsize=9)

    for tau in taus:
        hist = histories[tau]
        x = np.asarray([int(row["iteration"]) for row in hist], dtype=int)
        rel = np.asarray([float(row.get("relative_error", np.nan)) for row in hist], dtype=float)
        rel = np.maximum(rel, 1.0e-12)
        axes[1].semilogy(x, rel, label=rf"$\tau={float(tau):g}$")

    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("relative error")
    axes[1].set_title("1D TFIM relative error")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_2d_family(
    histories: Mapping[str, Sequence[Mapping[str, object]]],
    outpath: Path,
    title: str,
    reference_energies: Optional[Mapping[str, float]] = None,
) -> None:
    fields = sorted(histories.keys(), key=lambda x: float(x))
    have_refs = reference_energies is not None and all(f in reference_energies for f in fields)

    if have_refs:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        ax_energy, ax_err = axes
    else:
        fig, ax_energy = plt.subplots(1, 1, figsize=(6.2, 4.2))
        ax_err = None

    for field in fields:
        hist = histories[field]
        x = np.asarray([int(row["iteration"]) for row in hist], dtype=int)
        e = np.asarray([float(row["energy_per_spin"]) for row in hist], dtype=float)
        ax_energy.plot(x, e, label=rf"$h={float(field):g}$")
        if reference_energies and field in reference_energies:
            ax_energy.axhline(float(reference_energies[field]), linestyle="--", linewidth=1.0)

        if ax_err is not None and reference_energies and field in reference_energies:
            ref = float(reference_energies[field])
            rel = np.maximum(np.abs((e - ref) / ref), 1.0e-12)
            ax_err.semilogy(x, rel, label=rf"$h={float(field):g}$")

    ax_energy.set_xlabel("iteration")
    ax_energy.set_ylabel("energy per spin")
    ax_energy.set_title(title)
    ax_energy.legend(fontsize=9)

    if ax_err is not None:
        ax_err.set_xlabel("iteration")
        ax_err.set_ylabel("relative error")
        ax_err.set_title("2D TFIM relative error")
        ax_err.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_beta_family(histories: Mapping[str, Sequence[Mapping[str, object]]], outpath: Path, title: str) -> None:
    taus = sorted(histories.keys(), key=lambda x: float(x))
    plt.figure(figsize=(6.6, 4.2))
    for tau in taus:
        hist = histories[tau]
        x = np.asarray([int(row["iteration"]) for row in hist], dtype=int)
        beta = np.asarray([float(row["beta_x"]) for row in hist], dtype=float)
        plt.plot(x, beta, label=rf"$\tau={float(tau):g}$")
    plt.xlabel("iteration")
    plt.ylabel(r"$\beta_x$")
    plt.title(title)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


# ---------------------------------------------------------------------------
# Suite orchestration
# ---------------------------------------------------------------------------


def load_reference_2d(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    out: Dict[str, float] = {}
    for key, value in data.items():
        out[str(key)] = float(value)
    return out


def topology_seed(base_seed: int, topology: str, offset: int) -> int:
    topo_shift = 0 if topology == "pegasus" else 100_000
    return int(base_seed + topo_shift + offset)


def run_topology_suite(worker_payload: Mapping[str, object]) -> Dict[str, object]:
    topology = str(worker_payload["topology"])
    root_dir = Path(str(worker_payload["output_dir"])) / topology
    root_dir.mkdir(parents=True, exist_ok=True)

    access = QPUAccessConfig(
        topology=topology,
        backend=str(worker_payload["backend"]),
        solver_name=worker_payload.get(f"solver_{topology}") or worker_payload.get("solver_name"),
        token=worker_payload.get("token"),
        endpoint=worker_payload.get("endpoint"),
        region=worker_payload.get("region"),
        max_runtime_fraction=float(worker_payload["max_runtime_fraction"]),
        programming_thermalization=worker_payload.get("programming_thermalization"),
        readout_thermalization=worker_payload.get("readout_thermalization"),
        num_spin_reversal_transforms=int(worker_payload["num_spin_reversal_transforms"]),
        chain_strength_prefactor=float(worker_payload["chain_strength_prefactor"]),
        sim_graph_size=worker_payload.get("sim_graph_size"),
    )

    qpu = TopologySampler(access)
    ref2d = load_reference_2d(worker_payload.get("reference_2d_json"))

    results_1d: Dict[str, Sequence[Mapping[str, object]]] = {}
    results_2d: Dict[str, Sequence[Mapping[str, object]]] = {}
    failures: List[Dict[str, str]] = []

    try:
        L1d = int(worker_payload["L1d"])
        Lx2d = int(worker_payload["Lx2d"])
        Ly2d = int(worker_payload["Ly2d"])
        field_1d = float(worker_payload["field_1d"])
        anneal_times_1d = [float(x) for x in worker_payload["anneal_times_1d"]]
        fields_2d = [float(x) for x in worker_payload["fields_2d"]]

        common = dict(
            alpha=float(worker_payload["alpha"]),
            periodic=not bool(worker_payload["open_boundary"]),
            iterations=int(worker_payload["iterations"]),
            target_samples=int(worker_payload["target_samples"]),
            requested_reads_per_child_call=(
                None if worker_payload.get("requested_reads_per_child_call") is None
                else int(worker_payload["requested_reads_per_child_call"])
            ),
            learning_rate=float(worker_payload["learning_rate"]),
            diag_shift=float(worker_payload["diag_shift"]),
            cg_tol=float(worker_payload["cg_tol"]),
            cg_maxiter=int(worker_payload["cg_maxiter"]),
            param_init_scale=float(worker_payload["param_init_scale"]),
            param_clip=float(worker_payload["param_clip"]),
            beta_x_init=float(worker_payload["beta_x_init"]),
            beta_adapt=float(worker_payload["beta_adapt"]),
            beta_min=float(worker_payload["beta_min"]),
            beta_max=float(worker_payload["beta_max"]),
        )

        # 1D sweep over anneal time.
        ref1d = exact_tfim_1d_energy_per_spin(L1d, field_1d)
        for idx, tau in enumerate(anneal_times_1d):
            name = f"1d_tau_{tau:g}"
            cfg = RunConfig(
                name=name,
                dim=1,
                Lx=L1d,
                Ly=1,
                field=field_1d,
                annealing_time=tau,
                seed=topology_seed(int(worker_payload["seed"]), topology, 10 * idx + 1),
                reference_energy_per_spin=ref1d,
                **common,
            )
            run_dir = root_dir / name
            try:
                outcome = run_single_training(cfg, qpu, run_dir)
                results_1d[str(tau)] = outcome["history"]
            except Exception as exc:
                failures.append({name: str(exc)})

        # 2D sweep over field.
        tau2d = float(worker_payload["anneal_time_2d"])
        for idx, field in enumerate(fields_2d):
            name = f"2d_h_{field:g}"
            ref = ref2d.get(str(field)) if ref2d else None
            cfg = RunConfig(
                name=name,
                dim=2,
                Lx=Lx2d,
                Ly=Ly2d,
                field=field,
                annealing_time=tau2d,
                seed=topology_seed(int(worker_payload["seed"]), topology, 10 * idx + 1001),
                reference_energy_per_spin=ref,
                **common,
            )
            run_dir = root_dir / name
            try:
                outcome = run_single_training(cfg, qpu, run_dir)
                results_2d[str(field)] = outcome["history"]
            except Exception as exc:
                failures.append({name: str(exc)})

        if results_1d:
            plot_1d_family(
                results_1d,
                ref_energy=ref1d,
                outpath=root_dir / "figure_1d_energy_error.png",
                title=f"{topology.capitalize()} 1D TFIM, L={L1d}, h={field_1d}",
            )
            plot_beta_family(
                results_1d,
                outpath=root_dir / "figure_beta_x.png",
                title=f"{topology.capitalize()} adaptive $\\beta_x$",
            )

        if results_2d:
            plot_2d_family(
                results_2d,
                outpath=root_dir / "figure_2d_energy.png",
                title=f"{topology.capitalize()} 2D TFIM, {Lx2d}x{Ly2d}, $\\tau={tau2d:g}$",
                reference_energies=ref2d if ref2d else None,
            )

        summary = {
            "topology": topology,
            "sampler_metadata": qpu.metadata(),
            "completed_1d_runs": sorted(results_1d.keys(), key=float),
            "completed_2d_runs": sorted(results_2d.keys(), key=float),
            "failures": failures,
            "output_dir": str(root_dir),
        }
        save_json(summary, root_dir / "topology_summary.json")
        return summary
    finally:
        qpu.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the Gardas-Rams-Dziarmaga QPU plots on Pegasus/Zephyr with runtime-safe batching"
    )
    parser.add_argument("--topologies", nargs="+", choices=["pegasus", "zephyr"], default=["pegasus", "zephyr"])
    parser.add_argument("--processes", type=int, default=2, help="Worker processes. For Pegasus+Zephyr, 2 is the natural choice.")
    parser.add_argument("--backend", choices=["qpu", "sim"], default="qpu")
    parser.add_argument("--solver-name", type=str, default=None, help="Optional single solver name override.")
    parser.add_argument("--solver-pegasus", type=str, default=None, help="Optional explicit Pegasus solver name.")
    parser.add_argument("--solver-zephyr", type=str, default=None, help="Optional explicit Zephyr solver name.")
    parser.add_argument("--token", type=str, default=os.getenv("DWAVE_API_TOKEN"))
    parser.add_argument("--endpoint", type=str, default=os.getenv("DWAVE_API_ENDPOINT"))
    parser.add_argument("--region", type=str, default=os.getenv("DWAVE_API_REGION"))
    parser.add_argument("--output-dir", type=str, default="qpu_reproduction")

    # Suite definitions, chosen to mirror the QPU figures in the paper.
    parser.add_argument("--L1d", type=int, default=64)
    parser.add_argument("--field-1d", type=float, default=0.5)
    parser.add_argument("--anneal-times-1d", nargs="+", type=float, default=[2.0, 20.0, 200.0, 2000.0])
    parser.add_argument("--Lx2d", type=int, default=8)
    parser.add_argument("--Ly2d", type=int, default=8)
    parser.add_argument("--fields-2d", nargs="+", type=float, default=[0.5, 1.0, 3.044])
    parser.add_argument("--anneal-time-2d", type=float, default=20.0)
    parser.add_argument("--reference-2d-json", type=str, default=None, help="Optional JSON mapping of 2D field -> reference energy per spin.")

    # Training hyperparameters.
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--open-boundary", action="store_true")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--target-samples", type=int, default=10_000)
    parser.add_argument("--requested-reads-per-child-call", type=int, default=None,
                        help="Optional upper bound. The script still caps this to the safe value from estimate_qpu_access_time().")
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--diag-shift", type=float, default=1.0e-3)
    parser.add_argument("--cg-tol", type=float, default=1.0e-8)
    parser.add_argument("--cg-maxiter", type=int, default=200)
    parser.add_argument("--param-init-scale", type=float, default=1.0e-2)
    parser.add_argument("--param-clip", type=float, default=3.0)
    parser.add_argument("--beta-x-init", type=float, default=2.0)
    parser.add_argument("--beta-adapt", type=float, default=0.05)
    parser.add_argument("--beta-min", type=float, default=0.05)
    parser.add_argument("--beta-max", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1234)

    # QPU runtime / submission control.
    parser.add_argument("--num-spin-reversal-transforms", type=int, default=4)
    parser.add_argument("--chain-strength-prefactor", type=float, default=1.5)
    parser.add_argument("--max-runtime-fraction", type=float, default=0.85,
                        help="Use at most this fraction of the solver's problem_run_duration_range upper bound per child submission.")
    parser.add_argument("--programming-thermalization", type=float, default=None)
    parser.add_argument("--readout-thermalization", type=float, default=None)
    parser.add_argument("--sim-graph-size", type=int, default=None)

    ns = parser.parse_args(argv)
    if ns.processes < 1:
        parser.error("--processes must be at least 1")
    return ns


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    root_dir = Path(args.output_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "backend": args.backend,
        "solver_name": args.solver_name,
        "solver_pegasus": args.solver_pegasus,
        "solver_zephyr": args.solver_zephyr,
        "token": args.token,
        "endpoint": args.endpoint,
        "region": args.region,
        "output_dir": str(root_dir),
        "L1d": args.L1d,
        "field_1d": args.field_1d,
        "anneal_times_1d": list(args.anneal_times_1d),
        "Lx2d": args.Lx2d,
        "Ly2d": args.Ly2d,
        "fields_2d": list(args.fields_2d),
        "anneal_time_2d": args.anneal_time_2d,
        "reference_2d_json": args.reference_2d_json,
        "alpha": args.alpha,
        "open_boundary": args.open_boundary,
        "iterations": args.iterations,
        "target_samples": args.target_samples,
        "requested_reads_per_child_call": args.requested_reads_per_child_call,
        "learning_rate": args.learning_rate,
        "diag_shift": args.diag_shift,
        "cg_tol": args.cg_tol,
        "cg_maxiter": args.cg_maxiter,
        "param_init_scale": args.param_init_scale,
        "param_clip": args.param_clip,
        "beta_x_init": args.beta_x_init,
        "beta_adapt": args.beta_adapt,
        "beta_min": args.beta_min,
        "beta_max": args.beta_max,
        "seed": args.seed,
        "num_spin_reversal_transforms": args.num_spin_reversal_transforms,
        "chain_strength_prefactor": args.chain_strength_prefactor,
        "max_runtime_fraction": args.max_runtime_fraction,
        "programming_thermalization": args.programming_thermalization,
        "readout_thermalization": args.readout_thermalization,
        "sim_graph_size": args.sim_graph_size,
    }

    worker_payloads = [{**payload, "topology": topo} for topo in args.topologies]

    if len(worker_payloads) == 1 or args.processes == 1:
        summaries = [run_topology_suite(worker_payloads[0])]
        for wp in worker_payloads[1:]:
            summaries.append(run_topology_suite(wp))
    else:
        nproc = min(args.processes, len(worker_payloads))
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=nproc) as pool:
            summaries = pool.map(run_topology_suite, worker_payloads)

    suite_summary = {
        "summaries": summaries,
        "arguments": vars(args),
    }
    save_json(suite_summary, root_dir / "suite_summary.json")

    print("\nFinished suite. Outputs:")
    for item in summaries:
        print(f"- {item['topology']}: {item['output_dir']}")
        if item.get("failures"):
            print(f"  failures: {item['failures']}")


if __name__ == "__main__":
    main()
