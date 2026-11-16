#!/bin/bash
# All convergence benchmarks for the paper, 9 jobs per PSF source (27 in
# all, to stay under per-user job limits):
#   ng5     1000 PSFs, plain EM and SQUAREM at three tolerances   4 jobs
#   ngall    200 PSFs, 3/5/7/9 Gaussians at tol 1e-6               2 jobs
#   moffat   200 PSFs, Moffat mixtures vs em5/em7 (appendix C)     2 jobs
#   trace    100-PSF convergence traces                            1 job
# Pass sbatch options (e.g. --partition=long) as arguments.
set -euo pipefail
mkdir -p logs data/bench data/trace
for f in superbit hst roman; do
    sbatch --array=0-3 "$@" scripts/slurm_bench.sh ng5 "$f" 4
    sbatch --array=0-1 "$@" scripts/slurm_bench.sh ngall "$f" 2
    sbatch --array=0-1 "$@" scripts/slurm_bench.sh moffat "$f" 2
    sbatch --array=0   "$@" scripts/slurm_bench.sh trace "$f" 1
done
# when all have finished:
#   python bench/merge.py data/bench/*_ng5_*.npy    --out data/psf_bench.npy
#   python bench/merge.py data/bench/*_ngall_*.npy  --out data/psf_bench_ngauss.npy
#   python bench/merge.py data/bench/*_moffat_*.npy --out data/psf_bench_moffat.npy
#   make figures
