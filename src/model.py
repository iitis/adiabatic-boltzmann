"""
Restricted Boltzmann Machine Implementation

Key reference: Gardas et al., Eq. 6-7 (ansatz) and Eq. 15 (gradients)

Wave function: Ψ(v) = e^(-a·v/2) ∏_j [2·cosh(b_j + W_j·v)]^(1/2)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from helpers import get_solver_name


class RBM(ABC):
    """Abstract RBM base class."""

    def __init__(self, n_visible: int, n_hidden: int):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        scale = 0.01

        self.a = np.random.normal(0, scale, n_visible)
        self.b = np.random.normal(0, scale, n_hidden)
        self.W = np.random.normal(0, scale, (n_visible, n_hidden))

        # Apply mask immediately so W is sparse from init
        mask = self.get_connectivity_mask()
        self.W *= mask

    def logcosh(self, x):
        return np.logaddexp(x, -x)

    @abstractmethod
    def get_connectivity_mask(self) -> np.ndarray:
        """
        Return (n_visible, n_hidden) binary mask where mask[i,j] = 1
        means visible unit i is connected to hidden unit j.
        """
        pass

    def psi(self, v: np.ndarray) -> float:
        """Compute Ψ(v) - wave function value for configuration v.

        From paper: Ψ(v) = e^(-a·v/2) ∏_j [2·cosh(b_j + W_j·v)]^(1/2)
        """
        first_term = np.exp(-(self.a @ v) / 2)
        second_term = np.prod(
            [2 * np.cosh(self.b[i] + self.W.T[i] @ v) for i in range(self.n_hidden)]
        )
        return float(first_term * np.sqrt(second_term))

    def log_psi(self, v: np.ndarray) -> float:
        """
        Compute log(Ψ(v)) - log of the wave function.

        v: spin configuration, shape (n_visible,), values in {-1, +1}

        Returns: scalar log(Ψ(v))

        From Gardas et al., Eq. 6-7:
        Ψ(v) = e^(-a·v/2) ∏_j [2·cosh(b_j + W_j·v)]^(1/2)
        log(Ψ) = -a·v/2 + (1/2) ∑_j log[2·cosh(b_j + W_j·v)]
        """
        first_term = -(self.a @ v) / 2
        theta = self.b + self.W.T @ v
        second_term = 0.5 * np.sum(np.log(2) + self.logcosh(theta))
        return first_term + second_term

    def psi_ratio_old(self, v: np.ndarray, flip_idx: int) -> float:
        v_flipped = np.copy(v)
        v_flipped[flip_idx] *= -1
        return self.psi(v_flipped) / self.psi(v)

    def psi_ratio(self, v: np.ndarray, flip_idx: int) -> float:
        vi = v[flip_idx]
        log_ratio_a = self.a[flip_idx] * vi
        theta = self.b + self.W.T @ v
        theta_flipped = theta - 2 * vi * self.W[flip_idx, :]
        log_ratio_cosh = 0.5 * np.sum(self.logcosh(theta_flipped) - self.logcosh(theta))
        return np.exp(log_ratio_a + log_ratio_cosh)

    def gradient_log_psi(self, v: np.ndarray) -> dict:
        """
        Compute ∂log(Ψ)/∂p for all parameters p.

        ∂log(Ψ)/∂a_i = -v_i/2
        ∂log(Ψ)/∂b_j = (1/2) tanh(θ_j)
        ∂log(Ψ)/∂W_ij = (1/2) v_i tanh(θ_j)

        Returns dict with keys 'a', 'b', 'W' containing gradients of same shape as weights.
        """
        theta = self.b + self.W.T @ v
        grad_a = -0.5 * np.copy(v)
        grad_b = 0.5 * np.tanh(theta)
        grad_W = 0.5 * np.outer(v, np.tanh(theta))
        return {"a": grad_a, "b": grad_b, "W": grad_W}

    def get_weights(self) -> np.ndarray:
        """Flatten all weights into a 1D vector for SR algorithm."""
        return np.concatenate([self.a.flatten(), self.b.flatten(), self.W.flatten()])

    def set_weights(self, w: np.ndarray):
        """
        Unflatten 1D vector back to weights and re-apply the connectivity
        mask so that sparse entries can never drift away from zero due to
        numerical noise in the SR update.
        """
        idx = 0
        n_a = self.n_visible
        n_b = self.n_hidden
        n_w = self.n_visible * self.n_hidden

        self.a = w[idx : idx + n_a].copy()
        idx += n_a
        self.b = w[idx : idx + n_b].copy()
        idx += n_b
        self.W = w[idx : idx + n_w].reshape(self.n_visible, self.n_hidden).copy()

        # Keep forbidden connections at exactly zero
        self.W *= self.get_connectivity_mask()

    def n_parameters(self) -> int:
        """Total number of free (non-zero) parameters."""
        mask = self.get_connectivity_mask()
        return self.n_visible + self.n_hidden + int(mask.sum())

    def sparsity(self) -> float:
        """Fraction of W entries that are zero (0 = dense, 1 = empty)."""
        mask = self.get_connectivity_mask()
        return 1.0 - float(mask.sum()) / mask.size


class FullyConnectedRBM(RBM):
    """RBM with all visible-hidden connections (no topology constraint)."""

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
        solver: str = "zephyr",
        n_nodes: Optional[int] = None,
        seed: int = 42,
    ):
        self._solver = get_solver_name(solver)  # store for sampler to use
        self._qubit_mapping = None  # physical -> logical, set below
        solver = self._solver
        subgraph = self._subgraph_from_solver(
            solver, n_visible + n_hidden if n_nodes is None else n_nodes, seed
        )
        # Build physical->logical mapping sorted for reproducibility
        sorted_nodes = sorted(subgraph.nodes())
        self._qubit_mapping = {phys: idx for idx, phys in enumerate(sorted_nodes)}
        import networkx as nx

        mapped = nx.relabel_nodes(subgraph, self._qubit_mapping)
        self._mask = self._mask_from_graph(mapped, n_visible, n_hidden)

        super().__init__(n_visible, n_hidden)

    # ------------------------------------------------------------------
    # Mask construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_mask(
        n_visible: int,
        n_hidden: int,
        solver: str,
        n_nodes: Optional[int],
        seed: int,
    ) -> np.ndarray:
        """Build (n_visible, n_hidden) binary connectivity mask."""

        # Mode 1 — live QPU via solver name
        subgraph = DWaveTopologyRBM._subgraph_from_solver(
            solver,
            n_visible + n_hidden if n_nodes is None else n_nodes,
            seed,
        )
        mapped = DWaveTopologyRBM._remap_graph(subgraph)
        return DWaveTopologyRBM._mask_from_graph(mapped, n_visible, n_hidden)

    @staticmethod
    def _subgraph_from_solver(solver: str, n_nodes: int, seed: int):
        from dwave.system import DWaveSampler

        sampler = DWaveSampler(solver=solver)
        hw_graph = sampler.to_networkx_graph()

        if hw_graph.number_of_nodes() < n_nodes:
            raise RuntimeError(
                f"Solver '{solver}' exposes {hw_graph.number_of_nodes()} qubits "
                f"but {n_nodes} are required."
            )

        selected = _dense_subgraph(hw_graph, n_nodes, seed)
        return hw_graph.subgraph(selected)

    @staticmethod
    def _remap_graph(qpu_subgraph):
        """
        Normalise physical qubit IDs to a contiguous integer range 0..N-1
        so that node indices can be used directly as visible/hidden unit indices.
        """
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
        Extract the visible-hidden bipartite adjacency from a graph.

        Convention:
            nodes  0 .. n_visible-1          → visible units
            nodes  n_visible .. n_visible+n_hidden-1  → hidden units

        An edge (u, v) contributes to W[i, j] where i < n_visible and
        j = v - n_visible (or symmetrically).
        """
        n_total = n_visible + n_hidden
        mask = np.zeros((n_visible, n_hidden), dtype=np.float64)

        for u, v in graph.edges():
            u, v = int(u), int(v)
            if u >= n_total or v >= n_total:
                continue

            # Determine which node is visible and which is hidden
            if u < n_visible and v >= n_visible:
                vis_idx = u
                hid_idx = v - n_visible
            elif v < n_visible and u >= n_visible:
                vis_idx = v
                hid_idx = u - n_visible
            else:
                # Both visible or both hidden — not a V-H edge, skip
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
        """
        Return the (n_visible, n_hidden) binary connectivity mask.
        Cached at construction time — calling this is O(1).
        """
        return self._mask

    def gradient_log_psi(self, v: np.ndarray) -> dict:
        """
        Compute gradients and zero out entries not in the connectivity mask.
        This ensures the SR update never activates forbidden connections.
        """
        gradients = super().gradient_log_psi(v)
        gradients["W"] *= self._mask
        return gradients

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def connectivity_summary(self) -> dict:
        """
        Return a summary of the connectivity pattern for logging/debugging.
        """
        mask = self._mask
        degrees_visible = mask.sum(axis=1)  # connections per visible unit
        degrees_hidden = mask.sum(axis=0)  # connections per hidden unit
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


def _dense_subgraph(
    full_graph,
    n_nodes: int,
    seed: int,
) -> set:
    """
    Grow a connected subgraph of `n_nodes` nodes by always annexing the
    unvisited neighbour with the highest overlap with the current set.
    Ties are broken randomly.  Returns a set of node identifiers.
    """
    import random as _rng

    rng = _rng.Random(seed)
    all_nodes = list(full_graph.nodes())
    active = {rng.choice(all_nodes)}

    # Pre-compute degree for a fast upper-bound on achievable overlap
    max_degree = max(len(list(full_graph.neighbors(n))) for n in all_nodes)

    while len(active) < n_nodes:
        best_score = -1
        best_candidate = None

        # Shuffle to break ties randomly without a separate sort
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
