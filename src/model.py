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
from typing import NamedTuple, Optional
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

    Construction modes (checked in order):

    1. ``solver`` (str) — connect to a live D-Wave QPU by solver name, fetch
       its hardware graph, greedily select a dense subgraph of ``n_nodes``
       qubits (default: n_visible + n_hidden), and derive the mask from it.
       Requires dwave-system and networkx.

    2. ``graph`` (nx.Graph) — pre-built graph with integer node labels already
       in [0, n_visible + n_hidden).

    3. ``qpu_subgraph`` (nx.Graph) — raw QPU subgraph with physical qubit IDs;
       remapped to integers internally.

    4. No arguments — Chimera-inspired block-diagonal fallback, works without
       Ocean or networkx installed.

    Parameters
    ----------
    n_visible    : int
    n_hidden     : int
    key          : jax.Array  PRNG key for weight initialisation
    solver       : str, optional
        D-Wave solver name, e.g. ``"Advantage_system6.4"``.
    n_nodes      : int, optional
        Hardware qubits to select when using ``solver``.
        Defaults to n_visible + n_hidden.
    seed         : int
        Random seed for subgraph selection (default: 42).
    graph        : nx.Graph, optional
    qpu_subgraph : nx.Graph, optional
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        key: jax.Array,
        solver: str = "zephyr",
        n_nodes: Optional[int] = None,
        seed: int = 42,
    ):
        self._solver = get_solver_name(solver)
        self._qubit_mapping = None
        solver = self._solver
        subgraph = self._subgraph_from_solver(
            solver, n_visible + n_hidden if n_nodes is None else n_nodes, seed
        )
        sorted_nodes = sorted(subgraph.nodes())
        self._qubit_mapping = {phys: idx for idx, phys in enumerate(sorted_nodes)}
        import networkx as nx

        mapped = nx.relabel_nodes(subgraph, self._qubit_mapping)
        self._mask = self._mask_from_graph(mapped, n_visible, n_hidden)

        super().__init__(n_visible, n_hidden, key)

    # ------------------------------------------------------------------
    # Mask construction
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_path(solver: str, n_nodes: int, seed: int):
        from pathlib import Path

        cache_dir = Path(__file__).parent.parent / "embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_solver = solver.replace("/", "_").replace(".", "_")
        return cache_dir / f"{safe_solver}_{n_nodes}_seed{seed}.json"

    @staticmethod
    def _subgraph_from_solver(solver: str, n_nodes: int, seed: int):
        import json
        import networkx as nx

        cache_path = DWaveTopologyRBM._cache_path(solver, n_nodes, seed)

        if cache_path.exists():
            print(f"[DWaveTopologyRBM] Loading cached embedding from {cache_path}")
            with open(cache_path) as f:
                data = json.load(f)
            g = nx.Graph()
            g.add_nodes_from(data["nodes"])
            g.add_edges_from(data["edges"])
            return g

        from dwave.system import DWaveSampler

        sampler = DWaveSampler(solver=solver)
        hw_graph = sampler.to_networkx_graph()

        if hw_graph.number_of_nodes() < n_nodes:
            raise RuntimeError(
                f"Solver '{solver}' exposes {hw_graph.number_of_nodes()} qubits "
                f"but {n_nodes} are required."
            )

        selected = _dense_subgraph(hw_graph, n_nodes, seed)
        subgraph = hw_graph.subgraph(selected)

        data = {
            "solver": solver,
            "n_nodes": n_nodes,
            "seed": seed,
            "nodes": sorted(subgraph.nodes()),
            "edges": [list(e) for e in subgraph.edges()],
        }
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[DWaveTopologyRBM] Saved embedding to {cache_path}")

        return subgraph

    @staticmethod
    def _remap_graph(qpu_subgraph):
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx is required for QPU graph remapping. "
                "Install with: pip install networkx"
            )
        sorted_nodes = sorted(qpu_subgraph.nodes())
        relabel = {phys: idx for idx, phys in enumerate(sorted_nodes)}
        return nx.relabel_nodes(qpu_subgraph, relabel)

    @staticmethod
    def _mask_from_graph(graph, n_visible: int, n_hidden: int) -> np.ndarray:
        """
        Extract visible-hidden bipartite adjacency from a remapped graph.

        Convention: nodes 0..n_visible-1 → visible,
                    nodes n_visible..n_visible+n_hidden-1 → hidden.
        """
        n_total = n_visible + n_hidden
        mask = np.zeros((n_visible, n_hidden), dtype=np.float64)

        for u, v in graph.edges():
            u, v = int(u), int(v)
            if u >= n_total or v >= n_total:
                continue
            if u < n_visible and v >= n_visible:
                vis_idx, hid_idx = u, v - n_visible
            elif v < n_visible and u >= n_visible:
                vis_idx, hid_idx = v, u - n_visible
            else:
                continue
            if 0 <= vis_idx < n_visible and 0 <= hid_idx < n_hidden:
                mask[vis_idx, hid_idx] = 1.0

        if mask.sum() == 0:
            raise ValueError(
                "The provided graph produced an empty visible-hidden mask. "
                "Check that node indices follow the convention: "
                "visible nodes in [0, n_visible) and hidden nodes in "
                "[n_visible, n_visible+n_hidden)."
            )
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
# Convenience factory
# ---------------------------------------------------------------------------


def _dense_subgraph(full_graph, n_nodes: int, seed: int) -> set:
    """
    Grow a connected subgraph of `n_nodes` nodes by always annexing the
    unvisited neighbour with the highest overlap with the current set.
    """
    import random as _rng

    rng = _rng.Random(seed)
    all_nodes = list(full_graph.nodes())
    active = {rng.choice(all_nodes)}

    max_degree = max(len(list(full_graph.neighbors(n))) for n in all_nodes)

    while len(active) < n_nodes:
        best_score = -1
        best_candidate = None
        frontier = list(active)
        rng.shuffle(frontier)
        target = min(max_degree, len(active))
        found = False

        for node in frontier:
            candidates = [nb for nb in full_graph.neighbors(node) if nb not in active]
            rng.shuffle(candidates)
            for nb in candidates:
                overlap = len(set(full_graph.neighbors(nb)) & active)
                if overlap >= target:
                    active.add(nb)
                    found = True
                    break
                if overlap > best_score:
                    best_score = overlap
                    best_candidate = nb
            if found:
                break

        if not found:
            if best_candidate is None:
                raise RuntimeError(
                    f"Could not grow subgraph to {n_nodes} nodes — "
                    "the QPU graph may be disconnected or too small."
                )
            active.add(best_candidate)

    return active
