# Residual-covariance Lawson–Hermite solver

E. Ahmed, M. A. El-Beltagy, A. Etman and S. M. Elnady,
*Residual-covariance Lawson–Hermite approximation of joint laws for nonlinear fractional Wick–Itô–Skorohod equations*.

## Current journal edition — 15 November 2026

The manuscript is formatted for Numerical Algorithms and prepared for author review.

- [Journal manuscript, 43 pages](paper/numerical_algorithms/Manuscript.pdf)
- [Online Resource 1, 19 pages](paper/numerical_algorithms/ESM_1.pdf)
- [Flat manuscript LaTeX sources](paper/numerical_algorithms/Manuscript_sources.zip)
- [Supplement LaTeX sources](paper/numerical_algorithms/ESM_1_sources.zip)
- [Online Resource 2: complete code and data](https://github.com/EslamAhmedGaber/rclh-joint-laws/releases/download/v17-2026-11-15/ESM_2.zip)

This edition uses the verified v17 numerical results. It has seven scientific figures,
compact captions and explanation in the main text. Both manuscript editions now start their figure sequence with the driver study. The journal edition and the saved original are separately identified.

## Original v17 with the schematic removed

- [Original one-column manuscript, 27 pages](paper/v17/Manuscript.pdf)
- [Original two-column manuscript, 21 pages](paper/v17/Manuscript_twocolumn.pdf)
- [Original supplement, 18 pages](paper/v17/Supplementary_material.pdf)
- [Original LaTeX sources](paper/v17/Manuscript_sources.zip)

The original numerical code and data are retained in Online Resource 2; its
README_v17_analysis.md gives the original reproduction instructions.

## Method and scope

Unresolved variance enters the nonlinear drift, while residual covariance couples
one reconstruction across output dates. The method returns approximate moments
and joint samples after one deterministic coefficient evolution.

The comparisons include published translated time-steppers, Monte Carlo, multilevel
Monte Carlo, randomized quasi-Monte Carlo and sparse-grid collocation, including
regimes where competitors are faster. The paired correction is unbiased for the
sampled numerical-scheme target; the manuscript states the remaining time bias.

## Reproduction

Extract Online Resource 2 and follow README.md. It preserves all numerical files
from the independently replayed v17 archive, including frozen Figure 6 sample arrays
in the analysis edition. All 652 generated values, generated tables and numerical
figure data were checked. Journal sources were compiled from fresh ZIP extractions
and compared against every page's words and coordinates.

The code uses NumPy, SciPy, Numba and Matplotlib. Journal artwork additionally uses
Pillow and PyMuPDF. LaTeX builds require latexmk, pdflatex and BibTeX.

## Earlier study

The solver/, law_study/ and figures/ directories preserve the earlier v14 finite-factor
law study. The release archive contains the continuous-driver implementation and
comparisons used by v17.
