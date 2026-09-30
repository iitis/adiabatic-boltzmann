"""Compare Zephyr calibration-protocol QPU runs with the headline Zephyr/Pegasus +CEM runs."""
import gzip, glob, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact

REPO = Path(__file__).resolve().parent.parent.parent


def headline(dev, N):
    e = []
    for f in sorted(glob.glob(str(REPO / f"results/tfim_1d/{N}/dimod/{dev}/*h0.5*cem1*.gz"))):
        r = json.load(gzip.open(f))
        e.append((r["final_energy"] - r["exact_energy"]) / abs(r["exact_energy"]) * 100)
    return np.array(e)


def fmt(e):
    return f"{(e<1).sum():2d}/{len(e):2d} good, median {np.median(e):5.2f}%" if len(e) else "      -"


tags = sys.argv[1:] or ["P2calib3"]
ledger = REPO / "results/cem_zephyr_protocol_qpu/ledger.jsonl"
used = sum(json.loads(l)["qpu_s"] for l in open(ledger)) if ledger.exists() else 0
print(f"QPU ledger: {used:.1f}s used by protocol runs\n")
for tag in tags:
    for f in sorted(glob.glob(str(REPO / f"results/cem_zephyr_protocol_qpu/{tag}_N*.jsonl"))):
        rs = [json.loads(l) for l in open(f)]
        N = rs[0]["N"]
        e = np.array([(r["E_true"] - r["exact"]) / abs(r["exact"]) * 100 for r in rs])
        old = headline("zephyr", N)
        k = (e < 1).sum(), (old < 1).sum()
        p = fisher_exact([[k[0], len(e) - k[0]], [k[1], len(old) - k[1]]])[1] if len(old) else float("nan")
        c = rs[0]["calib"]
        bx0 = np.array([r["beta_x"][c] for r in rs]); bxf = np.array([r["beta_x"][-1] for r in rs])
        sdw = np.array([np.mean(r["sample_dw"][-5:]) for r in rs]); pdw = np.array([r["dw"] for r in rs])
        print(f"N={N:2d} {tag}: {fmt(e)} | old Zephyr {fmt(old)} (Fisher p={p:.3g}) | Pegasus {fmt(headline('pegasus', N))}")
        print(f"      beta_x after calib {np.median(bx0):.2f} [{bx0.min():.2f}-{bx0.max():.2f}], final {np.median(bxf):.2f};"
              f" domain walls: QPU samples {np.median(sdw):.2f} vs |Psi|^2 {np.median(pdw):.2f};"
              f" qpu/run {np.mean([r['qpu_s'] for r in rs]):.2f}s")
        print("      sorted true err %:", " ".join(f"{x:.1f}" for x in sorted(e)))
