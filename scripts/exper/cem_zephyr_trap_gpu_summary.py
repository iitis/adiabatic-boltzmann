"""Summarise results/cem_zephyr_trap_gpu/*.jsonl (chunks merged by config name)."""
import json, glob, re, collections, sys
import numpy as np
g = collections.defaultdict(list)
for f in glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results/cem_zephyr_trap_gpu/*.jsonl"):
    name = re.sub(r"_s\d+$", "", f.split("/")[-1][:-6])
    for l in open(f):
        r = json.loads(l); g[name].append(((r["E_true"] - r["exact"]) / abs(r["exact"]) * 100, r["dw"]))
for k in sorted(g):
    e = np.array([x[0] for x in g[k]])
    print(f"{k:26s} n={len(e):3d} good<1%={(e<1).sum():3d} ({(e<1).mean()*100:3.0f}%)  median={np.median(e):5.2f}%")
