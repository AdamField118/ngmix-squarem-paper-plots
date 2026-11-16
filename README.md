# ngmix-squarem-paper-plots

This repo makes every figure, table and quoted number in *Accelerating
Weak-Lensing Shape Measurement with SQUAREM-Optimized
Expectation–Maximization* (Field, Sanguinet & Saha).

The paper probes SQUAREM on three PSF sources:
- **SuperBIT PSFEx**: 50 exposures, 20 positions each = 1000 PSFs.
- **HST/ACS**: the PSFs distributed with the GalSim COSMOS sample, 1000 PSFs.
- **Roman H158**: GalSim design PSFs, 1000 positions.

The PSF models come from
[ngmix-squarem-shear](https://github.com/AdamField118/ngmix-squarem-shear),
so the convergence benchmarks here and the shear runs there use identical
PSFs.

| paper | script | input |
|---|---|---|
| Fig. 1, EM convergence, SuperBIT PSFs | `plots/fig_convergence.py` | `data/trace/*.npz` |
| Fig. 3, EM vs SQUAREM traces | `plots/fig_convergence.py` | `data/trace/*.npz` |
| Table 1, iterations and speedups | `plots/paper_numbers.py` | `data/psf_bench.npy` |
| Figs. 4–5, speedup and iteration CDFs | `plots/fig_iterations.py` | `data/psf_bench.npy` |
| Fig. 6, Table 2, shear calibration | `plots/fig_shear.py`, `plots/paper_numbers.py` | `data/shear/*/summary.json` |
| Appendix B, more gaussians | `plots/fig_ngauss.py` | `data/psf_bench_ngauss.npy` |
| Appendix C, Moffat mixtures (table) | `plots/paper_numbers.py` | `data/psf_bench_moffat.npy` |
| Appendix D, PSFs and three models of them | `plots/fig_fits2d.py` | the PSF models themselves |
| (not in the paper) the same as 3D surfaces, `make figures3d` | `plots/fig_3d.py` | the PSF models themselves |
| every number in the text | `plots/paper_numbers.py` | all of the above |

Every number in the paper's text comes from data. `paper_numbers.py` writes
`figures/numbers.tex` with one `\valdef{key}{value}` per statistic, and the
paper prints each with `\val{key}`. A key that has not been generated
appears in the PDF as a red **??key**, so a missing number cannot slip
through as a stale one. `figures/numbers.txt` lists every key and its full
value for review.

## Reproduce

```bash
conda env create -f environment.yml && conda activate squarem-plots
pip install -e ../ngmix-squarem-shear
# SuperBIT PSFs default to ../ngmix-squarem-shear/psf_data/...; override with
# export SUPERBIT_PSFEX='/path/to/*.psf'
export COSMOS_DIR=/path/to/COSMOS_25.2_training_sample   # if not in GalSim's share dir
```

On a cluster:

```bash
scripts/submit_bench.sh          # 9 SLURM arrays, see the file
python bench/merge.py data/bench/*_ng5_*.npy   --out data/psf_bench.npy
python bench/merge.py data/bench/*_ngall_*.npy --out data/psf_bench_ngauss.npy
```

Alternatively, run `make data` serially.

The shear summaries come from that repo's `scripts/collect_all.sh`. Copy
them into place:

```bash
for s in superbit hst roman; do
  mkdir -p data/shear/$s
  cp ../ngmix-squarem-shear/results/$s/summary.json data/shear/$s/
done
make figures
cp figures/*.pdf figures/*.tex ../paper/figures/
```

### Benchmark sizes

Each family is benchmarked four ways:

| benchmark | PSFs per family | fits per PSF |
|---|---|---|
| speed and quality | 1000 | 2 runners × 3 tolerances, same starting mixture |
| richer mixtures (appendix) | 200 | 2 runners × 4 mixture sizes at tol 1e-6 |
| convergence traces | 100 | 2 runners × 40 iteration caps |
| shear calibration (other repo) | 30,000 galaxies | 4 arms × 4 images |

Every statistic is reported as a median, a mean and a standard deviation
over PSFs or objects, not as a single example.

## Conventions

- **Colours:** EM is blue `#2a78d6`, solid. SQUAREM is orange `#eb6834`, dashed. The palette was checked for colour-vision deficiency, and every series also has its own line style or marker.
- **Shaded bands** are the 16th–84th percentiles over PSFs.
- **Convergence metrics:**
  - ngmix's own statistic: |ΔlogL/logL| between successive plain EM steps.
  - The scale-free **fixed-point residual**: the largest change in the mixture's size T (fractional) or ellipticity under one more plain EM step.
- **Same optimum:** two fits count as reaching the same optimum when their sizes agree to 10⁻³ (fractionally) and their ellipticities to 10⁻³. When they differ, the paper reports which one has the higher likelihood.
