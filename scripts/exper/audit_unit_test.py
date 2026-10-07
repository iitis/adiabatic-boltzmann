"""Single-temperature test on frozen QPU samples, decoded with unbiased tie-breaking.

Physical samples (results/audit_frozen_qpu/*.npz) are re-decoded by majority vote with ties broken
at random (Ocean's majority_vote maps ties to +1). Under a joint Boltzmann distribution of the
programmed RBM energy at one beta, every logical spin k obeys E[s_k | rest] = tanh(beta f_k), with
f_k its local field (hidden: b_j + sum_i W_ij v_i; visible: -a_i + sum_j W_ij u_j). We fit beta_k
per spin by conditional maximum likelihood and test homogeneity with
    chi2 = sum_k (beta_k - beta_bar)^2 / var_k,  var_k = 1 / sum_r f^2 sech^2(beta_k f)   (Fisher),
beta_bar the inverse-variance mean. Also T = log(beta_u|v / beta_v|u) from the pooled fits.
Null: the same statistics on 200 sets of exact (N<=16) or block-Gibbs (N>=32) draws at the pooled beta.

    python scripts/exper/audit_unit_test.py     # writes results/audit_unit_test.jsonl
"""
import glob, json
from pathlib import Path
import numpy as np
from audit_two_conditional import cmle, null_draws, RES

rng = np.random.default_rng(7)


def decode(phys, phys_vars, emb, N):
    idx = {int(q): i for i, q in enumerate(phys_vars)}
    out = np.empty((len(phys), 2 * N))
    for k in range(2 * N):
        s = phys[:, [idx[q] for q in emb[str(k)]]].sum(1)
        out[:, k] = np.where(s > 0, 1, np.where(s < 0, -1, rng.choice([-1, 1], len(s))))
    return out[:, :N], out[:, N:]


def stats(v, u, a, b, W):
    F = np.hstack([-a + u @ W.T, v @ W + b]); S = np.hstack([v, u])
    bk = np.array([cmle(S[:, k], F[:, k]) for k in range(S.shape[1])])
    var = 1 / np.array([np.sum(F[:, k] ** 2 / np.cosh(bk[k] * F[:, k]) ** 2) for k in range(S.shape[1])])
    bbar = np.sum(bk / var) / np.sum(1 / var)
    N = len(a)
    T = np.log(cmle(u, F[:, N:]) / cmle(v, F[:, :N]))
    return float(np.sum((bk - bbar) ** 2 / var)), float(T), bk


def main():
    out = RES / "audit_unit_test.jsonl"
    done = {json.loads(l)["file"] for l in open(out)} if out.exists() else set()
    for fn in sorted(glob.glob(str(RES / "audit_frozen_qpu/cs_*_cs1.0_bx1.0.npz"))):
        if Path(fn).stem in done:
            continue
        d = np.load(fn); job = json.loads(str(d["job"])); N = job["N"]
        emb = json.loads(str(d["embedding"]))
        v, u = decode(d["phys"], d["phys_vars"], emb, N)
        a, b, W = d["a"], d["b"], d["W"]
        chi2, T, bk = stats(v, u, a, b, W)
        beta = cmle(np.hstack([v, u]), np.hstack([-a + u @ W.T, v @ W + b]))
        null = np.array([stats(vv, uu, a, b, W)[:2] for vv, uu in null_draws(a, b, W, beta, len(v), 200)])
        chain_len = [len(emb[str(k)]) for k in range(2 * N)]
        rec = dict(file=Path(fn).stem, device=job["device"], N=N, seed=job["seed"], n_reads=len(v), beta=beta,
                   chi2=chi2, df=2 * N - 1, chi2_null_mean=float(null[:, 0].mean()),
                   p_chi2=float((1 + np.sum(null[:, 0] >= chi2)) / 201),
                   T=T, T_null_sd=float(null[:, 1].std()), z_T=float((T - null[:, 1].mean()) / null[:, 1].std()),
                   beta_units=bk.tolist(), chain_len=chain_len, cbf=float(d["cbf"].mean()))
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rec["file"], f"chi2={chi2:.0f} (null {rec['chi2_null_mean']:.0f}, p={rec['p_chi2']:.3f}) T={T:+.3f} z={rec['z_T']:+.1f}", flush=True)


if __name__ == "__main__":
    main()
