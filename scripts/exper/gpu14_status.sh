#!/bin/bash
# Progress of the gpu14 run: queue logs, running jobs, latest checkpoints, DMRG references.
cd "$(dirname "$(readlink -f "$0")")/../.."
L=results/vmc_gpu/logs
echo "== $(date '+%F %T')  load $(cut -d' ' -f1-3 /proc/loadavg)  GPU $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
for q in gpu cpu; do
  f=$L/queue_gpu14_${q}_jobs.log
  echo "== $q queue: $(grep -c ' OK: ' $f) ok, $(grep -c 'FAILED' $f) failed, $(grep -c 'exit [1-9]' $f) retries; of $(grep -cv '^\s*\(#\|$\)' scripts/exper/gpu14_${q}_jobs.txt) jobs$(grep -q 'all jobs finished' $f && echo ' — FINISHED')"
  grep FAILED $f | cut -c1-200
done
echo "== running:"; ps -eo etime,args | grep -E "^ *[0-9:-]+ /home/phanussek/miniconda3/envs/test/bin/python scripts/(exper/vmc_gpu|validation/dmrg_graph)" | sed 's#/home/phanussek/miniconda3/envs/test/bin/python scripts/##' | cut -c1-150
echo "== VMC checkpoints: $(cat results/vmc_gpu/*.jsonl 2>/dev/null | wc -l); latest:"
tail -q -n1 results/vmc_gpu/*.jsonl 2>/dev/null | python3 -c "
import json,sys
for l in sys.stdin:
    r=json.loads(l); ref=r['E0'] if r['E0'] is not None else None
    print(f\"  {r['run']} s{r['seed']} step {r['step']}: E/N {r['E']/r['N']:.6f}\" + (f\"  err/N {(r['E']-ref)/r['N']:.2e}\" if ref is not None else f\"  (MF {r['E_mf']/r['N']:.6f})\"))"
echo "== DMRG (results/dmrg_baseline/graph.jsonl), best per (size, h):"
python3 -c "
import json
best={}
for l in open('results/dmrg_baseline/graph.jsonl'):
    r=json.loads(l); k=(r['size'],r['h'])
    if k not in best or r['chi']>best[k]['chi']: best[k]=r
for k,r in sorted(best.items()): print(f\"  {k[0]} h={k[1]:g}: chi {r['chi']} E/N {r['E_per_spin']:.6f} trunc {r['max_trunc_err']:.1e} t {r['t_dmrg']:.0f}s\")" 2>/dev/null
