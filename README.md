# rclh-joint-laws

Code and data for the article

> E. Ahmed, M. A. El-Beltagy, A. Etman, S. M. Elnady,
> *Joint laws from one coefficient solve: a residual-covariance Lawson–Hermite solver
> for nonlinear Wick–Itô–Skorohod equations driven by fractional Brownian motion*
> (submitted to Communications in Nonlinear Science and Numerical Simulation).

The residual-covariance Lawson–Hermite (RCLH) solver computes deterministic Hermite
coefficients for scalar Wick–Itô–Skorohod equations with noise coefficient βX. Its
reconstruction is an explicit function of a few Gaussian variables, so one coefficient
evolution returns the joint law of the solution at all output times. This repository
contains the solver and the study that verifies this law against the exact finite-factor
solution of the fishery application, and the scripts that draw the article's figures.

## Quick start

```bash
python -m pip install -r requirements.txt
cd law_study
python -B law_study.py run        # 18 fishery cells, about 12 minutes on one core
python -B law_study.py numbers    # results/law_summary.json
python -B ledger_split.py         # residual share of the yield variance
python -B make_law_numbers.py     # output/tex: macros and the two law tables
python -B fig_cell_samples.py     # samples of the cell shown in Figure 1 (about 1 minute)
cd ../figures
python -B build_figures.py        # output/figures: all figures (PDF and PNG)
```

`law_study.py run` resumes from `results/law_cells.jsonl`; the per-cell results used in the
article are included, so the later steps can be run without repeating the study. Delete
that file to recompute everything.

## What reproduces what

| Article item | Script | Output |
|---|---|---|
| Section 4.6, Table 7, supplement Table S10, abstract and highlight numbers | `law_study/law_study.py`, `law_study/make_law_numbers.py` | `output/tex/macros/v14_numbers.tex`, `output/tex/tables/data/*.tex` |
| Residual share of the yield variance (Proposition 3.15, Section 4.6) | `law_study/ledger_split.py` | `law_study/results/ledger_split.json` |
| Figure 1 | `law_study/fig_cell_samples.py`, `figures/build_figures.py` | `output/figures/figlaw_*` |
| Figures 3–8 | `figures/build_figures.py` from `figures/source_data/` | `output/figures/fig2_*` … `fig7_*` |

Figure 2 (the workflow diagram) is drawn in TikZ in the manuscript source. The
experiments of the earlier sections (Landau, logistic, cost and rank studies, historical
fishery) are reproduced by the reproducibility archive in the article's supplementary
material; `figures/source_data/` holds the rows those figures plot.

## Layout

- `solver/rclh/rclh_solve.py` — the `rclh_solve` interface used in the article.
- `solver/vendor/` — the numerical engine; every file is pinned in `MANIFEST_SHA256.json`
  and checked at import, so results come from exactly this code.
- `law_study/` — study plan (written before the run), scripts and per-cell results.
- `figures/` — figure script and plotted source data.

## Method of the law study

For each cell the solver is run once. Its coherent reconstruction and the explicit
full finite-factor (Bernoulli) solution are evaluated on the **same** 262,144 Gaussian
vectors (eight scrambled Sobol replicates of 2^15 points), so differences are pathwise.
A Gaussian law with the exact mean and variance is the moments-only comparator.
Intervals are 95% Student-t intervals over the replicates; differences are paired.

## Tested with

Python 3.12.4, NumPy 2.2.6, SciPy 1.14.1, Matplotlib 3.9.2, Pillow, on Windows 10
(single BLAS thread for timings).

## Citation

See `CITATION.cff`.
