#!/bin/bash
# All convergence benchmarks for the paper: 1000 PSFs per family at three
# tolerances, 200 PSFs per family with 3/5/7/9 gaussians, and 100-PSF traces.
set -euo pipefail
mkdir -p logs data/bench data/trace
for f in superbit hst roman; do
    sbatch --array=0-19 scripts/slurm_bench.sh ng5 "$f" 20
    sbatch --array=0-9  scripts/slurm_bench.sh ngall "$f" 10
    sbatch --array=0    scripts/slurm_bench.sh trace "$f" 1
done
# when all have finished:
#   python bench/merge.py data/bench/*_ng5_*.npy   --out data/psf_bench.npy
#   python bench/merge.py data/bench/*_ngall_*.npy --out data/psf_bench_ngauss.npy
#   make figures
