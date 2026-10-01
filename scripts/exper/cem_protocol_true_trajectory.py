"""Independent true energy <H>_Psi along the training trajectory of protocol QPU runs (CPU only).

Uses the per-draw parameters saved in raw/{tag}_N{N}_seed{seed}.npz by cem_zephyr_protocol_qpu.py
and re-evaluates the network at selected training steps with fresh Metropolis chains (same
estimator as the final-network score). This checks whether energies estimated from the device's
own samples (and therefore time-to-tolerance) are biased early in training.

Output (append-only, finished seeds skipped): {DIR}/true_traj_{tag}_N{N}.jsonl
Usage (from repo root): python scripts/exper/cem_protocol_true_trajectory.py DIR TAG N [K:I]
  K:I (optional) processes only seeds with seed % K == I, for parallel workers.
  e.g. python scripts/exper/cem_protocol_true_trajectory.py results/cem_pegasus_protocol_qpu P4cemPL 32
"""
import json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from model import FullyConnectedRBM, RBMParams
from ising import TransverseFieldIsing1D
from cem_zephyr_trap_gpu import score

STEPS = [0, 5, 10, 15, 20, 25, 30, 40, 50, 70, 99]  # training steps (after calibration)

d, tag, N = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
K, I = map(int, sys.argv[4].split(":")) if len(sys.argv) > 4 else (1, 0)
out = d / f"true_traj_{tag}_N{N}.jsonl"
done = {json.loads(l)["seed"] for l in open(out)} if out.exists() else set()
ising = TransverseFieldIsing1D(N, 0.5)
rbm = FullyConnectedRBM(N, N, jax.random.PRNGKey(0))
for line in open(d / f"{tag}_N{N}.jsonl"):
    r = json.loads(line)
    if r["seed"] in done or r["seed"] % K != I:
        continue
    raw = np.load(REPO / r["raw_file"])
    c = r["calib"]
    t0 = time.time()
    rows = []
    for t in STEPS:
        k = c + t  # draw index; parameters the step-t samples were drawn from
        rbm.params = RBMParams(a=jnp.asarray(raw["a"][k]), b=jnp.asarray(raw["b"][k]), W=jnp.asarray(raw["W"][k]))
        sc = score(rbm, ising)
        rows.append(dict(step=t, draw=k, E_own=r["E"][k], **sc))
    rec = dict(tag=tag, N=N, seed=r["seed"], exact=r["exact"], steps=rows,
               cum_time_s=[float(np.cumsum(r["sampling_time_s"])[c + t]) for t in STEPS])
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"N={N} seed={r['seed']} done in {time.time() - t0:.0f}s", flush=True)
