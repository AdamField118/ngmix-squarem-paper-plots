#!/bin/bash
# Convergence benchmarks on a cluster.  Submit with scripts/submit_bench.sh;
# merge with `make merge-bench` style commands shown there.
#SBATCH --job-name=squarem-bench
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=bench:plots:${PYTHONPATH:-}
kind=$1      # ng5 | ngall | trace
family=$2    # superbit | hst | roman
nchunks=$3
c=${SLURM_ARRAY_TASK_ID}
tag=$(printf %04d "$c")
case $kind in
  ng5)   python bench/run_psf_bench.py --family "$family" --npool 1000 \
             --chunk "$c" --nchunks "$nchunks" \
             --out data/bench/${family}_ng5_${tag}.npy ;;
  ngall) python bench/run_psf_bench.py --family "$family" --npool 200 \
             --ngauss 3 5 7 9 --tols 1e-6 --chunk "$c" --nchunks "$nchunks" \
             --out data/bench/${family}_ngall_${tag}.npy ;;
  trace) python bench/run_trace.py --family "$family" --npsf 100 \
             --out data/trace/${family}.npz ;;
esac
