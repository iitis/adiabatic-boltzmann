"""Trainer-level check of the Zephyr protocol (cem_calib_iters=3, beta_feedback='pl') on the
synthetic hidden-temperature annealer, N=32, headline hyperparameters.
Usage: python cem_zephyr_trainer_p4_synth.py OUT.jsonl SEEDS [beta_hw]"""
import contextlib, io, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from model import FullyConnectedRBM
from ising import TransverseFieldIsing1D
from encoder import Trainer
from cem_zephyr_trap_gpu import SyntheticAnnealer, score, parse_seeds

out, seeds = Path(sys.argv[1]), parse_seeds(sys.argv[2])
beta_hw = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
for sd in seeds:
    key = jax.random.PRNGKey(sd); key, mk = jax.random.split(key)
    ising = TransverseFieldIsing1D(32, 0.5); rbm = FullyConnectedRBM(32, 32, mk)
    cfg = dict(learning_rate=0.08, n_iterations=100, n_samples=200, regularization=0.05, use_cem=True,
               cem_interval=5, seed=sd, n_parallel=1, beta_x_init=1.0, cem_calib_iters=3, beta_feedback="pl")
    tr = Trainer(rbm, ising, SyntheticAnnealer(beta_hw, seed=sd + 1000), cfg, args=None)
    with contextlib.redirect_stdout(io.StringIO()):
        h = tr.train()
    ex = ising.exact_ground_energy()
    r = dict(seed=sd, beta_hw=beta_hw, exact=ex, E=h["energy"], beta_x=h["beta_x"], calib_beta_x=h["calib_beta_x"],
             s_pl=h["beta_eff_pl"], **score(rbm, ising))
    with open(out, "a") as f:
        f.write(json.dumps(r) + "\n")
    print(f"seed={sd}: true err={(r['E_true']-ex)/abs(ex)*100:+.2f}% beta_x={h['beta_x'][-1]:.2f}", flush=True)
