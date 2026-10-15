#!/bin/bash
# Rebuild notebooks/mf_vs_h.{ipynb,html} once the cylinder VMC queue has finished (and again when DMRG finishes).
# Detaches itself (setsid + nohup).
SELF=$(readlink -f "$0"); cd "$(dirname "$SELF")/../.."
if [ -z "$RB_DETACHED" ]; then RB_DETACHED=1 setsid nohup "$SELF" >> results/vmc_gpu/logs/rebuild_notebook.log 2>&1 < /dev/null & echo "detached: pid $!"; exit 0; fi
P=/home/phanussek/miniconda3/envs/test/bin
rebuild() {
  echo "$(date '+%F %T') rebuilding ($1)"
  PATH=$P:$PATH JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 nice $P/python scripts/viz/make_mf_vs_h_notebook.py --execute \
    && PATH=$P:$PATH $P/jupyter nbconvert --to html notebooks/mf_vs_h.ipynb && echo "$(date '+%F %T') ok"
}
for q in tricyl_vmc tricyl_dmrg; do
  until grep -q "all jobs finished" results/vmc_gpu/logs/queue_${q}_jobs.log 2>/dev/null; do sleep 300; done
  rebuild "$q queue finished"
done
