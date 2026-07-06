"""
Restricted Boltzmann Machine Implementation — JAX backend

Key reference: Gardas et al., Eq. 6-7 (ansatz) and Eq. 15 (gradients)

Wave function: Ψ(v) = e^(-a·v/2) ∏_j [2·cosh(b_j + W_j·v)]^(1/2)

JAX design notes
----------------
* RBMParams is a NamedTuple → automatically a JAX PyTree.  jax.jit, jax.grad,
  and optax optimisers all work on it with no extra registration.
* The RBM class holds *metadata only* (sizes, mask).  Live parameter arrays
  live in self.params (RBMParams) and are updated functionally via set_weights().
* Properties W / a / b delegate to self.params for backward compatibility.
  Direct assignment (rbm.W = ...) is supported via property setters so that
  checkpoint-restore code continues to work unchanged.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import NamedTuple
from abc import ABC, abstractmethod
from helpers import get_solver_name


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------


class RBMParams(NamedTuple):
    """
    Immutable RBM parameter PyTree.

    Being a NamedTuple it is automatically registered as a JAX PyTree, so it
    can flow through jax.jit / jax.grad / optax without any extra work.
    """

    a: jax.Array  # (n_visible,)        visible biases
    b: jax.Array  # (n_hidden,)         hidden biases
    W: jax.Array  # (n_visible, n_hidden) weight matrix


class FBMParams(NamedTuple):
    """
    Immutable Full Boltzmann Machine parameter PyTree.

    Extends RBMParams with a visible-visible coupling matrix J
    (symmetric, zero diagonal).
    """

    a: jax.Array  # (n_visible,)
    b: jax.Array  # (n_hidden,)
    W: jax.Array  # (n_visible, n_hidden)
    J: jax.Array  # (n_visible, n_visible) symmetric, zero diagonal


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class RBM(ABC):
    """Abstract RBM base.

    Subclasses must implement get_connectivity_mask().
    """

    def __init__(self, n_visible: int, n_hidden: int, key: jax.Array):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.scale = 0.01
        self.params: RBMParams = self.init_params(key)

    def init_params(self, key: jax.Array) -> RBMParams:
        """Create initial RBMParams: a=0, b=0, W ~ N(0, scale) * mask."""
        mask = jnp.asarray(self.get_connectivity_mask(), dtype=jnp.float64)
        W = jax.random.normal(key, (self.n_visible, self.n_hidden), dtype=jnp.float64)
        W = W * self.scale * mask
        return RBMParams(
            a=jnp.zeros(self.n_visible, dtype=jnp.float64),
            b=jnp.zeros(self.n_hidden, dtype=jnp.float64),
            W=W,
        )

    # ── Convenience properties (backward compat) ─────────────────────────

    @property
    def W(self) -> jax.Array:
        return self.params.W

    @property
    def a(self) -> jax.Array:
        return self.params.a

    @property
    def b(self) -> jax.Array:
        return self.params.b

    @W.setter
    def W(self, v):
        self.params = RBMParams(
            a=self.params.a, b=self.params.b, W=jnp.asarray(v, dtype=jnp.float64)
        )

    @a.setter
    def a(self, v):
        self.params = RBMParams(
            a=jnp.asarray(v, dtype=jnp.float64), b=self.params.b, W=self.params.W
        )

    @b.setter
    def b(self, v):
        self.params = RBMParams(
            a=self.params.a, b=jnp.asarray(v, dtype=jnp.float64), W=self.params.W
        )

    # ── Core maths ────────────────────────────────────────────────────────

    def logcosh(self, x):
        """Numerically stable log(cosh(x)) = logaddexp(x, -x)."""
        return jnp.logaddexp(x, -x)

    @abstractmethod
    def get_connectivity_mask(self) -> np.ndarray:
        """Return (n_visible, n_hidden) binary mask (1 = connected).

        Must return a plain NumPy array; it is converted to JAX inside
        init_params() / set_weights().
        """
        pass

    def log_psi(self, v: jax.Array) -> jax.Array:
        """
        log Ψ(v) = -a·v/2 + (1/2) Σ_j log[2·cosh(b_j + W_j·v)]
        """
        p = self.params
        theta = p.b + p.W.T @ v
        return -p.a @ v / 2 + 0.5 * jnp.sum(jnp.log(2) + self.logcosh(theta))

    def psi(self, v: jax.Array) -> jax.Array:
        """Ψ(v) — wave function amplitude."""
        return jnp.exp(self.log_psi(v))

    def psi_ratio(self, v: jax.Array, flip_idx: int) -> jax.Array:
        """Ψ(v_flip_i) / Ψ(v) computed efficiently in log space."""
        p = self.params
        vi = v[flip_idx]
        theta = p.b + p.W.T @ v
        theta_flipped = theta - 2 * vi * p.W[flip_idx, :]
        log_ratio = p.a[flip_idx] * vi + 0.5 * jnp.sum(
            self.logcosh(theta_flipped) - self.logcosh(theta)
        )
        return jnp.exp(log_ratio)

    def psi_ratio_pair(self, v: jax.Array, flip_i: int, flip_j: int) -> jax.Array:
        """Ψ(v with spins i and j simultaneously flipped) / Ψ(v), in log space."""
        p = self.params
        vi, vj = v[flip_i], v[flip_j]
        theta = p.b + p.W.T @ v
        theta_flipped = theta - 2 * vi * p.W[flip_i, :] - 2 * vj * p.W[flip_j, :]
        log_ratio = (
            p.a[flip_i] * vi
            + p.a[flip_j] * vj
            + 0.5 * jnp.sum(self.logcosh(theta_flipped) - self.logcosh(theta))
        )
        return jnp.exp(log_ratio)

    def gradient_log_psi(self, v: jax.Array) -> dict:
        """
        ∂log Ψ/∂p for all parameters.

        ∂log Ψ/∂a_i  = -v_i / 2
        ∂log Ψ/∂b_j  =  tanh(θ_j) / 2
        ∂log Ψ/∂W_ij =  v_i · tanh(θ_j) / 2
        """
        p = self.params
        theta = p.b + p.W.T @ v
        tanh_theta = jnp.tanh(theta)
        return {
            "a": -0.5 * v,
            "b": 0.5 * tanh_theta,
            "W": 0.5 * jnp.outer(v, tanh_theta),
        }

    # ── Weight serialisation ──────────────────────────────────────────────

    def get_weights(self) -> jax.Array:
        """Flatten params → 1-D JAX array  [a, b, W.ravel()]."""
        p = self.params
        return jnp.concatenate([p.a.ravel(), p.b.ravel(), p.W.ravel()])

    def set_weights(self, w: jax.Array) -> RBMParams:
        """
        Unpack flat vector into RBMParams, re-apply connectivity mask so that
        forbidden connections can never drift from zero due to SR numerical
        noise, update self.params, and return the new params.
        """
        N, M = self.n_visible, self.n_hidden
        mask = jnp.asarray(self.get_connectivity_mask(), dtype=jnp.float64)
        a = w[:N]
        b = w[N : N + M]
        W = w[N + M :].reshape(N, M) * mask
        self.params = RBMParams(a=a, b=b, W=W)
        return self.params

    # ── Diagnostics ───────────────────────────────────────────────────────

    def n_parameters(self) -> int:
        """Total number of free (non-zero) parameters."""
        mask = self.get_connectivity_mask()
        return self.n_visible + self.n_hidden + int(np.sum(mask))

    def sparsity(self) -> float:
        """Fraction of W entries that are zero (0 = dense, 1 = empty)."""
        mask = self.get_connectivity_mask()
        return 1.0 - float(np.sum(mask)) / mask.size

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_visible={self.n_visible}, n_hidden={self.n_hidden})"
        )


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class FullyConnectedRBM(RBM):
    """Dense RBM — all visible-hidden connections active."""

    def get_connectivity_mask(self) -> np.ndarray:
        return np.ones((self.n_visible, self.n_hidden))


class SRBM(RBM):
    """RBM with the diagonal of W zeroed out after initialisation."""

    def __init__(self, n_visible: int, n_hidden: int, key: jax.Array):
        super().__init__(n_visible, n_hidden, key)
        # Zero diagonal entries (min dimension to handle non-square W)
        diag = jnp.arange(min(n_visible, n_hidden))
        W_new = self.params.W.at[diag, diag].set(0.0)
        self.params = RBMParams(a=self.params.a, b=self.params.b, W=W_new)

    def get_connectivity_mask(self) -> np.ndarray:
        return np.ones((self.n_visible, self.n_hidden))


class DWaveTopologyRBM(RBM):
    """
    RBM whose visible-hidden connectivity is constrained to match a subgraph
    of a D-Wave QPU, enabling chain-free (trivial) embedding.

    The visible/hidden split follows the QPU's native bipartite structure:
    every Pegasus/Zephyr qubit has a "shore" bit (the first element of its
    ``pegasus_index`` / ``zephyr_index``), and edges connecting same-shore
    qubits ("odd"/external couplers) are the reason the full hardware graph
    is not bipartite. Restricting to cross-shore edges yields an exactly
    bipartite subgraph; we grow a dense, connected cluster inside that
    subgraph with exactly ``n_visible`` qubits on one shore and ``n_hidden``
    on the other, so every accepted hidden qubit is guaranteed at least one
    visible neighbour (no dead, zero-degree hidden units) and every kept
    edge is a real visible-hidden connection.

    Parameters
    ----------
    n_visible : int
    n_hidden  : int
    key       : jax.Array  PRNG key for weight initialisation
    solver    : str
        Topology name, ``"pegasus"`` or ``"zephyr"``.
    seed      : int
        Random seed for subgraph selection (default: 42).
    live      : bool
        If True (default), fetch the real hardware graph from a live D-Wave
        QPU via ``dwave.system.DWaveSampler`` (requires Ocean SDK
        credentials) — the mask then reflects that specific chip's actual
        yield (dead/missing qubits and couplers).
        If False, use ``dwave_networkx``'s idealized, defect-free fabric
        graph for the topology instead — no cloud access or credentials
        needed, at the cost of not reflecting any particular chip's real
        yield. The resulting RBM is **not** eligible for real QPU sampling
        (``_qubit_mapping`` is only meaningful against the graph it was
        built from); attempting to submit it to a live sampler raises.
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        key: jax.Array,
        solver: str = "zephyr",
        seed: int = 42,
        live: bool = True,
    ):
        if solver not in ("pegasus", "zephyr"):
            raise ValueError(
                f"Unknown D-Wave topology '{solver}' — expected 'pegasus' or 'zephyr'"
            )
        self._topology = solver
        self._solver = get_solver_name(solver)
        self._live = live
        self._qubit_mapping = None

        visible_qubits, hidden_qubits, edges = self._select_bipartite_subgraph(
            self._solver, self._topology, n_visible, n_hidden, seed, live
        )
        sorted_visible = sorted(visible_qubits)
        sorted_hidden = sorted(hidden_qubits)
        self._qubit_mapping = {
            **{phys: idx for idx, phys in enumerate(sorted_visible)},
            **{phys: n_visible + idx for idx, phys in enumerate(sorted_hidden)},
        }
        self._mask = self._mask_from_qubit_sets(
            sorted_visible, sorted_hidden, edges
        )

        super().__init__(n_visible, n_hidden, key)

    # ------------------------------------------------------------------
    # Mask construction
    # ------------------------------------------------------------------

    # First element of pegasus_index / zephyr_index is the qubit's "shore":
    # cross-shore couplers are exactly the bipartite subset of the hardware
    # graph (same-shore "odd"/external couplers are what make the full
    # Pegasus/Zephyr graph non-bipartite).
    _INDEX_KEY = {"pegasus": "pegasus_index", "zephyr": "zephyr_index"}

    # Shape of the idealized, defect-free dwave_networkx fabric matching the
    # generation each solver family is built on: P16 for Advantage_system6.x,
    # Z12 (t=4) for Advantage2_system1.x — confirmed against a live sampler's
    # sampler.properties["topology"]["shape"]. Only used when live=False —
    # no cloud access, no real-yield defects.
    _IDEAL_SHAPE = {"pegasus": (16,), "zephyr": (12, 4)}

    @staticmethod
    def _cache_path(solver: str, n_visible: int, n_hidden: int, seed: int, live: bool):
        from pathlib import Path

        cache_dir = Path(__file__).parent.parent / "embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_solver = solver.replace("/", "_").replace(".", "_")
        tag = "live" if live else "ideal"
        return cache_dir / f"{safe_solver}_{n_visible}v_{n_hidden}h_seed{seed}_{tag}.json"

    # In-memory cache of fetched hardware graphs, keyed by (solver, live).
    # The graph is the same for every (n_visible, n_hidden) combination on a
    # given solver, so without this a sweep over many alpha/N values would
    # reconnect to the D-Wave API and re-download the full ~40k-edge graph
    # once per combination instead of once per solver.
    _hw_graph_cache: dict = {}

    @staticmethod
    def _hw_graph_disk_cache_path(solver: str, live: bool):
        from pathlib import Path

        cache_dir = Path(__file__).parent.parent / "embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_solver = solver.replace("/", "_").replace(".", "_")
        tag = "live" if live else "ideal"
        return cache_dir / f"_hwgraph_{safe_solver}_{tag}.json"

    @classmethod
    def _fetch_hw_graph(cls, solver: str, topology: str, live: bool):
        """Hardware (or idealized fabric) graph, annotated with each qubit's
        shore index. live=False needs no cloud access or credentials.
        Cached in-memory for the process and on-disk across runs, so the
        (slow, and for live=True, API-hitting) fetch happens at most once
        per (solver, live) regardless of how many (n_visible, n_hidden)
        combinations are requested afterwards."""
        import json
        import networkx as nx

        mem_key = (solver, live)
        if mem_key in cls._hw_graph_cache:
            return cls._hw_graph_cache[mem_key]

        index_key = cls._INDEX_KEY[topology]
        disk_path = cls._hw_graph_disk_cache_path(solver, live)
        if disk_path.exists():
            print(f"[DWaveTopologyRBM] Loading cached hardware graph from {disk_path}")
            with open(disk_path) as f:
                data = json.load(f)
            g = nx.Graph()
            g.add_nodes_from((n, {index_key: tuple(idx)}) for n, idx in data["nodes"])
            g.add_edges_from(data["edges"])
            cls._hw_graph_cache[mem_key] = g
            return g

        import dwave_networkx as dnx

        gen = dnx.pegasus_graph if topology == "pegasus" else dnx.zephyr_graph
        if not live:
            shape = cls._IDEAL_SHAPE[topology]
            g = gen(*shape, data=True)
        else:
            from dwave.system import DWaveSampler

            print(f"[DWaveTopologyRBM] Fetching live hardware graph for solver '{solver}'...")
            sampler = DWaveSampler(solver=solver)
            shape = sampler.properties["topology"]["shape"]
            g = gen(*shape, node_list=sampler.nodelist, edge_list=sampler.edgelist, data=True)

        data = {
            "nodes": [[n, list(g.nodes[n][index_key])] for n in g.nodes()],
            "edges": [list(e) for e in g.edges()],
        }
        with open(disk_path, "w") as f:
            json.dump(data, f)
        print(f"[DWaveTopologyRBM] Saved hardware graph to {disk_path}")

        cls._hw_graph_cache[mem_key] = g
        return g

    @staticmethod
    def _bipartite_edges(hw_graph, index_key: str):
        """Cross-shore couplers only — an exactly bipartite edge set."""
        return [
            (u, v)
            for u, v in hw_graph.edges()
            if hw_graph.nodes[u][index_key][0] != hw_graph.nodes[v][index_key][0]
        ]

    @staticmethod
    def _grow_shore_balanced_subgraph(
        hw_graph, bip_edges, index_key: str, n_visible: int, n_hidden: int, seed: int
    ):
        """
        Select n_visible shore-0 qubits and n_hidden shore-1 qubits from the
        cross-shore edge set such that every selected qubit has >=1 real
        neighbour on the other shore (no dead, zero-degree units).

        The visible set is built with a max-coverage greedy: repeatedly add
        the shore-0 qubit that introduces the most shore-1 qubits not yet
        reachable by any already-selected visible qubit. This favours a
        *spread* of visible qubits over a tight, maximally-overlapping
        cluster, which matters whenever n_hidden is a large multiple of
        n_visible (alpha > 1) — a small tightly-clustered set of visible
        qubits simply doesn't neighbour enough distinct shore-1 qubits to
        supply that many hidden units without leaving some disconnected.
        The n_hidden qubits are then the highest-degree (into the visible
        set) shore-1 qubits reachable from it, which maximises kept edges
        subject to that constraint.
        """
        import random as _rng

        rng = _rng.Random(seed)
        shore = {n: hw_graph.nodes[n][index_key][0] for n in hw_graph.nodes()}
        adjacency: dict = {}
        for u, v in bip_edges:
            adjacency.setdefault(u, set()).add(v)
            adjacency.setdefault(v, set()).add(u)

        shore0_nodes = [n for n, s in shore.items() if s == 0 and n in adjacency]
        shore1_nodes = {n for n, s in shore.items() if s == 1 and n in adjacency}
        if len(shore0_nodes) < n_visible:
            raise RuntimeError(
                f"Solver exposes only {len(shore0_nodes)} shore-0 qubits with "
                f"cross-shore edges but n_visible={n_visible} are required."
            )

        order0 = shore0_nodes[:]
        rng.shuffle(order0)
        rank0 = {n: i for i, n in enumerate(order0)}

        # Pick visible qubits with a pacing greedy: maximise overlap with the
        # qubits already selected (i.e. prefer a dense local cluster, which
        # is what makes the RBM expressive) as long as doing so still leaves
        # enough remaining picks to reach n_hidden distinct shore-1 neighbours
        # overall; only when the running pace would fall short does it switch
        # to maximising *new* coverage instead. Pure density-maximisation
        # alone gets stuck for alpha > ~2 (a tight cluster's neighbourhood
        # plateaus); pure coverage-maximisation alone spreads visible qubits
        # so thin that most end up with a single hidden neighbour each.
        visible: set = set()
        covered: set = set()
        pool0 = set(order0)
        while len(visible) < n_visible:
            remaining = n_visible - len(visible)
            deficit = n_hidden - len(covered)
            if deficit <= 0:
                best = max(pool0, key=lambda n: (len(adjacency[n] & covered), -rank0[n]))
            else:
                pace = deficit / remaining
                eligible = [n for n in pool0 if len(adjacency[n] - covered) >= pace]
                if eligible:
                    best = max(eligible, key=lambda n: (len(adjacency[n] & covered), -rank0[n]))
                else:
                    best = max(pool0, key=lambda n: (len(adjacency[n] - covered), -rank0[n]))
            visible.add(best)
            covered |= adjacency[best]
            pool0.discard(best)

        if len(covered) < n_hidden:
            raise RuntimeError(
                f"The {n_visible} selected visible qubits only neighbour "
                f"{len(covered)} distinct shore-1 qubits, fewer than the "
                f"n_hidden={n_hidden} required — alpha is too large for this "
                f"hardware graph's local connectivity. Try a different seed "
                f"or a smaller n_hidden."
            )

        order1 = sorted(covered)
        rng.shuffle(order1)
        rank1 = {n: i for i, n in enumerate(order1)}

        # First guarantee every visible qubit keeps >=1 hidden neighbour: a
        # min-set-cover greedy over the visible set (picking purely by
        # highest-degree-into-visible below can otherwise starve a visible
        # qubit whose only neighbours are all low-degree and get trimmed).
        uncovered_visible = set(visible)
        pool1 = set(order1)
        hidden: set = set()
        while uncovered_visible:
            best = max(
                pool1,
                key=lambda n: (len(adjacency[n] & uncovered_visible), -rank1[n]),
            )
            hidden.add(best)
            uncovered_visible -= adjacency[best]
            pool1.discard(best)

        if len(hidden) > n_hidden:
            raise RuntimeError(
                f"Covering all {n_visible} visible qubits needs at least "
                f"{len(hidden)} hidden qubits, more than n_hidden={n_hidden} "
                f"— alpha is too small to keep every visible qubit connected "
                f"on this hardware graph. Try a different seed or a larger "
                f"n_hidden."
            )

        # Fill remaining hidden slots with whatever's left, ranked by degree
        # into the visible set, to maximise total kept edges.
        extra_needed = n_hidden - len(hidden)
        extra = sorted(pool1, key=lambda n: (-len(adjacency[n] & visible), rank1[n]))
        hidden |= set(extra[:extra_needed])

        return visible, hidden

    @classmethod
    def _select_bipartite_subgraph(
        cls, solver: str, topology: str, n_visible: int, n_hidden: int, seed: int, live: bool
    ):
        import json

        cache_path = cls._cache_path(solver, n_visible, n_hidden, seed, live)
        if cache_path.exists():
            print(f"[DWaveTopologyRBM] Loading cached embedding from {cache_path}")
            with open(cache_path) as f:
                data = json.load(f)
            return (
                set(data["visible_qubits"]),
                set(data["hidden_qubits"]),
                [tuple(e) for e in data["edges"]],
            )

        index_key = cls._INDEX_KEY[topology]
        hw_graph = cls._fetch_hw_graph(solver, topology, live)
        bip_edges = cls._bipartite_edges(hw_graph, index_key)
        visible_qubits, hidden_qubits = cls._grow_shore_balanced_subgraph(
            hw_graph, bip_edges, index_key, n_visible, n_hidden, seed
        )
        nodes = visible_qubits | hidden_qubits
        edges = [(u, v) for u, v in bip_edges if u in nodes and v in nodes]

        data = {
            "solver": solver,
            "topology": topology,
            "n_visible": n_visible,
            "n_hidden": n_hidden,
            "seed": seed,
            "live": live,
            "visible_qubits": sorted(visible_qubits),
            "hidden_qubits": sorted(hidden_qubits),
            "edges": [list(e) for e in edges],
        }
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[DWaveTopologyRBM] Saved embedding to {cache_path}")

        return visible_qubits, hidden_qubits, edges

    @staticmethod
    def _mask_from_qubit_sets(sorted_visible, sorted_hidden, edges) -> np.ndarray:
        """Build the (n_visible, n_hidden) mask directly from the shore-partitioned
        qubit sets — every edge here is by construction a real visible-hidden
        connection, so nothing is silently dropped."""
        v_index = {q: i for i, q in enumerate(sorted_visible)}
        h_index = {q: i for i, q in enumerate(sorted_hidden)}
        mask = np.zeros((len(sorted_visible), len(sorted_hidden)), dtype=np.float64)

        for u, v in edges:
            if u in v_index and v in h_index:
                mask[v_index[u], h_index[v]] = 1.0
            elif v in v_index and u in h_index:
                mask[v_index[v], h_index[u]] = 1.0

        if mask.sum() == 0:
            raise ValueError("Bipartite subgraph produced an empty visible-hidden mask.")
        return mask

    # ------------------------------------------------------------------
    # RBM interface
    # ------------------------------------------------------------------

    def get_connectivity_mask(self) -> np.ndarray:
        return self._mask

    def gradient_log_psi(self, v: jax.Array) -> dict:
        """Gradients with forbidden connections zeroed out."""
        gradients = super().gradient_log_psi(v)
        gradients["W"] = gradients["W"] * jnp.asarray(self._mask)
        return gradients

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def connectivity_summary(self) -> dict:
        mask = self._mask
        degrees_visible = mask.sum(axis=1)
        degrees_hidden = mask.sum(axis=0)
        return {
            "n_visible": self.n_visible,
            "n_hidden": self.n_hidden,
            "n_connections": int(mask.sum()),
            "max_connections": self.n_visible * self.n_hidden,
            "sparsity": self.sparsity(),
            "n_parameters": self.n_parameters(),
            "deg_visible_mean": float(degrees_visible.mean()),
            "deg_visible_min": float(degrees_visible.min()),
            "deg_visible_max": float(degrees_visible.max()),
            "deg_hidden_mean": float(degrees_hidden.mean()),
            "deg_hidden_min": float(degrees_hidden.min()),
            "deg_hidden_max": float(degrees_hidden.max()),
        }

    def __repr__(self) -> str:
        s = self.connectivity_summary()
        return (
            f"DWaveTopologyRBM("
            f"n_visible={self.n_visible}, "
            f"n_hidden={self.n_hidden}, "
            f"connections={s['n_connections']}/{s['max_connections']}, "
            f"sparsity={s['sparsity']:.2%})"
        )


# ---------------------------------------------------------------------------
# Full Boltzmann Machine
# ---------------------------------------------------------------------------


class FullBoltzmannMachine:
    """
    Full Boltzmann Machine VMC ansatz.

    Extends the RBM with visible-visible couplings J (symmetric, zero diagonal).
    Tracing out the hidden units remains analytically tractable:

        log Ψ(v) = -a·v/2 + ¼vᵀJv + ½ Σⱼ log[2·cosh(bⱼ + Wⱼ·v)]

    J maps directly to vis-vis QUBO edges — chain-free on D-Wave.
    """

    def __init__(self, n_visible: int, n_hidden: int, key: jax.Array):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.scale = 0.01
        # Precompute upper-triangle index arrays for J serialisation.
        self._triu_k, self._triu_l = np.triu_indices(n_visible, k=1)
        W = jax.random.normal(key, (n_visible, n_hidden), dtype=jnp.float64) * self.scale
        self.params = FBMParams(
            a=jnp.zeros(n_visible, dtype=jnp.float64),
            b=jnp.zeros(n_hidden, dtype=jnp.float64),
            W=W,
            J=jnp.zeros((n_visible, n_visible), dtype=jnp.float64),
        )

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def a(self) -> jax.Array:
        return self.params.a

    @a.setter
    def a(self, v):
        self.params = FBMParams(
            a=jnp.asarray(v, dtype=jnp.float64),
            b=self.params.b, W=self.params.W, J=self.params.J,
        )

    @property
    def b(self) -> jax.Array:
        return self.params.b

    @b.setter
    def b(self, v):
        self.params = FBMParams(
            a=self.params.a,
            b=jnp.asarray(v, dtype=jnp.float64),
            W=self.params.W, J=self.params.J,
        )

    @property
    def W(self) -> jax.Array:
        return self.params.W

    @W.setter
    def W(self, v):
        self.params = FBMParams(
            a=self.params.a, b=self.params.b,
            W=jnp.asarray(v, dtype=jnp.float64),
            J=self.params.J,
        )

    @property
    def J(self) -> jax.Array:
        return self.params.J

    @J.setter
    def J(self, v):
        self.params = FBMParams(
            a=self.params.a, b=self.params.b, W=self.params.W,
            J=jnp.asarray(v, dtype=jnp.float64),
        )

    # ── Core maths ────────────────────────────────────────────────────────

    def logcosh(self, x):
        return jnp.logaddexp(x, -x)

    def log_psi(self, v: jax.Array) -> jax.Array:
        """log Ψ(v) = -a·v/2 + ¼vᵀJv + ½ Σⱼ log[2·cosh(bⱼ + Wⱼ·v)]"""
        p = self.params
        theta = p.b + p.W.T @ v
        jvv = 0.25 * v @ p.J @ v
        return -p.a @ v / 2 + jvv + 0.5 * jnp.sum(jnp.log(2) + self.logcosh(theta))

    def psi(self, v: jax.Array) -> jax.Array:
        return jnp.exp(self.log_psi(v))

    def psi_ratio(self, v: jax.Array, flip_idx: int) -> jax.Array:
        """Ψ(v with spin flip_idx flipped) / Ψ(v), in log space."""
        p = self.params
        vi = v[flip_idx]
        theta = p.b + p.W.T @ v
        theta_flipped = theta - 2 * vi * p.W[flip_idx, :]
        log_ratio = (
            p.a[flip_idx] * vi
            - vi * (p.J[flip_idx, :] @ v)
            + 0.5 * jnp.sum(self.logcosh(theta_flipped) - self.logcosh(theta))
        )
        return jnp.exp(log_ratio)

    def psi_ratio_pair(self, v: jax.Array, flip_i: int, flip_j: int) -> jax.Array:
        """Ψ(v with spins i and j simultaneously flipped) / Ψ(v)."""
        p = self.params
        vi, vj = v[flip_i], v[flip_j]
        theta = p.b + p.W.T @ v
        theta_flipped = theta - 2 * vi * p.W[flip_i, :] - 2 * vj * p.W[flip_j, :]
        log_ratio = (
            p.a[flip_i] * vi + p.a[flip_j] * vj
            - vi * (p.J[flip_i, :] @ v)
            - vj * (p.J[flip_j, :] @ v)
            + 2 * vi * vj * p.J[flip_i, flip_j]
            + 0.5 * jnp.sum(self.logcosh(theta_flipped) - self.logcosh(theta))
        )
        return jnp.exp(log_ratio)

    def gradient_log_psi(self, v: jax.Array) -> dict:
        """
        ∂log Ψ/∂p for all parameters.

        ∂log Ψ/∂a_i  = -v_i / 2
        ∂log Ψ/∂b_j  =  tanh(θ_j) / 2
        ∂log Ψ/∂W_ij =  v_i · tanh(θ_j) / 2
        ∂log Ψ/∂J_ij =  v_i · v_j / 4   (i≠j, returned as full symmetric matrix)
        """
        p = self.params
        theta = p.b + p.W.T @ v
        tanh_theta = jnp.tanh(theta)
        N = self.n_visible
        diag_idx = jnp.arange(N)
        J_grad = (0.25 * jnp.outer(v, v)).at[diag_idx, diag_idx].set(0.0)
        return {
            "a": -0.5 * v,
            "b": 0.5 * tanh_theta,
            "W": 0.5 * jnp.outer(v, tanh_theta),
            "J": J_grad,
        }

    # ── Weight serialisation ──────────────────────────────────────────────

    def get_weights(self) -> jax.Array:
        """Flatten params → 1-D JAX array  [a, b, W.ravel(), J_upper_tri]."""
        p = self.params
        J_flat = p.J[self._triu_k, self._triu_l]
        return jnp.concatenate([p.a.ravel(), p.b.ravel(), p.W.ravel(), J_flat])

    def set_weights(self, w: jax.Array) -> FBMParams:
        """Unpack flat vector [a, b, W.ravel(), J_upper_tri] into FBMParams."""
        N, M = self.n_visible, self.n_hidden
        a = w[:N]
        b = w[N : N + M]
        W = w[N + M : N + M + N * M].reshape(N, M)
        J_flat = w[N + M + N * M :]
        J = jnp.zeros((N, N), dtype=jnp.float64)
        J = J.at[self._triu_k, self._triu_l].set(J_flat)
        J = J + J.T  # symmetrise
        self.params = FBMParams(a=a, b=b, W=W, J=J)
        return self.params

    # ── Diagnostics ───────────────────────────────────────────────────────

    def n_parameters(self) -> int:
        N, M = self.n_visible, self.n_hidden
        return N + M + N * M + N * (N - 1) // 2

    def get_connectivity_mask(self) -> np.ndarray:
        return np.ones((self.n_visible, self.n_hidden))

    def sparsity(self) -> float:
        return 0.0

    def __repr__(self) -> str:
        n_J = self.n_visible * (self.n_visible - 1) // 2
        return (
            f"FullBoltzmannMachine("
            f"n_visible={self.n_visible}, n_hidden={self.n_hidden}, "
            f"n_params={self.n_parameters()}, n_J={n_J})"
        )
