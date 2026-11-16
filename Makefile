# Rebuild every figure, table and quoted number in the paper.
#
#   make figures    redraw from data/ (seconds)
#   make data       rerun all benchmarks serially (hours; use the SLURM
#                   scripts in scripts/ on a cluster instead)
#
# SuperBIT PSFEx files are found with $SUPERBIT_PSFEX (a glob), the COSMOS
# sample with $COSMOS_DIR (optional).  Shear summaries are copied from
# ngmix-squarem-shear into data/shear/<source>/summary.json.
PY ?= python
export PYTHONPATH := bench:plots:$(PYTHONPATH)
FAMS := superbit hst roman
SHEAR := $(wildcard data/shear/*/summary.json)

figures: figures/convergence_em.pdf figures/iterations_cdf.pdf \
         figures/ngauss.pdf figures/numbers.tex figures/shear_bias.pdf \
         figures/fits2d_superbit.pdf

figures/convergence_em.pdf: $(wildcard data/trace/*.npz) plots/fig_convergence.py
	$(PY) plots/fig_convergence.py data/trace figures
figures/iterations_cdf.pdf: data/psf_bench.npy plots/fig_iterations.py
	$(PY) plots/fig_iterations.py $< figures
figures/ngauss.pdf: data/psf_bench_ngauss.npy plots/fig_ngauss.py
	$(PY) plots/fig_ngauss.py $< figures
# appendix D: one PSF per source and three models of it (sources whose
# data are missing are skipped)
figures/fits2d_superbit.pdf: plots/fig_fits2d.py
	$(PY) plots/fig_fits2d.py figures --families $(FAMS) --ipsf 0
# the same PSFs as 3D surfaces; not used in the paper
figures3d:
	$(PY) plots/fig_3d.py figures --families $(FAMS) --ipsf 0
figures/shear_bias.pdf: $(SHEAR) plots/fig_shear.py
	$(PY) plots/fig_shear.py figures $(SHEAR)
figures/numbers.tex: data/psf_bench.npy plots/paper_numbers.py $(SHEAR) \
                     $(wildcard data/psf_bench_ngauss.npy) \
                     $(wildcard data/psf_bench_moffat.npy)
	$(PY) plots/paper_numbers.py --bench data/psf_bench.npy \
	    --ngauss data/psf_bench_ngauss.npy --moffat data/psf_bench_moffat.npy \
	    --shear data/shear --outdir figures

data:
	mkdir -p data/trace data/bench
	for f in $(FAMS); do \
	  $(PY) bench/run_trace.py --family $$f --npsf 100 --out data/trace/$$f.npz; \
	  $(PY) bench/run_psf_bench.py --family $$f --npool 1000 \
	      --out data/bench/$${f}_ng5_0000.npy; \
	  $(PY) bench/run_psf_bench.py --family $$f --npool 200 --ngauss 3 5 7 9 \
	      --tols 1e-6 --out data/bench/$${f}_ngall_0000.npy; \
	  $(PY) bench/run_moffat_bench.py --family $$f --npool 200 \
	      --out data/bench/$${f}_moffat_0000.npy; \
	done
	$(PY) bench/merge.py data/bench/*_ng5_*.npy --out data/psf_bench.npy
	$(PY) bench/merge.py data/bench/*_ngall_*.npy --out data/psf_bench_ngauss.npy
	$(PY) bench/merge.py data/bench/*_moffat_*.npy --out data/psf_bench_moffat.npy

.PHONY: figures figures3d data
