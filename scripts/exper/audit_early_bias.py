"""Device samples vs faithful samples of pi_theta at the same parameters, early in training (Pegasus raw).

For calls t of the P4 runs: signed magnetization m, |m| and domain walls of the 200 device samples, and the
same for 600 samples of pi_theta at the parameters of that call (Metropolis with global flip, 1000 sweeps).
    python scripts/exper/audit_early_bias.py N SEEDS   # writes results/audit_early_bias_N{N}.jsonl
"""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from model import FullyConnectedRBM, RBMParams
from sampler import ClassicalSampler

N, seeds = int(sys.argv[1]), sys.argv[2]
lo, hi = map(int, seeds.split("-"))
out = REPO / f"results/audit_early_bias_N{N}.jsonl"
stats = lambda V: dict(m=float(V.mean()), absm=float(np.abs(V.mean(1)).mean()), dw=float((V != np.roll(V, -1, 1)).sum(1).mean()),
                       frac_plus=float((V > 0).mean()))
with out.open("w") as f:
    for seed in range(lo, hi + 1):
        d = np.load(REPO / f"results/cem_pegasus_protocol_qpu/raw/P4cemPL_N{N}_seed{seed}.npz")
        for t in (0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 60, 102):
            rbm = FullyConnectedRBM(N, N, jax.random.PRNGKey(0))
            rbm.params = RBMParams(a=jnp.asarray(d["a"][t]), b=jnp.asarray(d["b"][t]), W=jnp.asarray(d["W"][t]))
            smp = ClassicalSampler("metropolis"); smp._key = jax.random.PRNGKey(1000 * seed + t)
            Vf = np.asarray(smp.sample(rbm, 600, {"global_flip": True, "n_warmup": 1000}))
            Vd = d["V"][t].astype(float)
            rec = dict(seed=seed, call=t, beta_x=float(d["beta_x"][t]), device=stats(Vd), faithful=stats(Vf),
                       Hplus=float((d["H"][t] > 0).mean()))
            f.write(json.dumps(rec) + "\n"); f.flush()
        print("seed", seed, flush=True)
