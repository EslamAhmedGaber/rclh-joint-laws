# Law study plan (written 2026-11-07, before the full run)


## 1. Question
Does the coherent reconstruction X_coh(t; xi, eta) of RCLH, sampled from one coefficient solve,
reproduce the **law** of the full finite-factor solution (paths on common noise, quantiles,
low-stock and harvest-shortfall probabilities), not only its moments?
Pilot (2026-11-06, 3 cells, sigma = F = 0.5): yield correlation >= 0.9988, P(X_T < 0.2) equal to
3 digits, moment-matched Gaussian up to 5x off.

## 2. Design (fixed)
- Synthetic fishery of Section 4.4, all 18 cells: F in {0.2, 0.5, 0.8}, sigma in {0.25, 0.50},
  H in {0.10, 0.50, 0.80}; r = kappa = 1, X0 = 0.7, T = 2, N = 32.
- RCLH exactly as in the paper: endpoint factor, (M, P) = (3, 5), 8 nodes per coordinate, 4 RK4 substeps.
- Reference: the explicit Bernoulli solution of the full finite-factor equation (Proposition A.1),
  all 32 coordinates, 8 Gauss-Legendre nodes per cell (paper: 4, checked with 8).
- Common noise: 8 independent scrambled Sobol replicates of 2^15 points (262,144 vectors);
  RCLH and reference evaluated on the **same** vectors (xi = first M coordinates, eta = the rest).
- Naive alternative: Gaussian with the **reference** mean and variance (the best a moments-only
  method can do).

## 3. Quantities
Cumulative yield Y = F * trapezoid integral of X: quantiles 1%, 5%, 50%, 95%, 99%.
Terminal stock X_T: quantiles 1%, 5%, 99%.
Events (x* = 1 - F, deterministic equilibrium): P(X_T < x*/2), P(min_grid X < x*/2) (path event),
P(Y < 0.8 E Y) (harvest shortfall).
Pathwise: relative L2 discrepancy of X_T and of Y on common noise.
Ledger check (residual share of the yield variance): sum w w (1 - e^{-v}) K from the solver vs sampled
E|Y - E[Y | xi]|^2.

## 4. Uncertainty
Each quantity is estimated per replicate; the reported value is the replicate mean; 95% interval
half-width hw = t(0.975, 7) * sd / sqrt(8). Differences RCLH - reference are **paired** per replicate
and get their own half-width.

## 5. Definitions fixed in advance
- Quantile "agrees": |RCLH - ref| / |ref| <= 1% or the paired 95% interval contains 0.
- Probability "agrees": |RCLH - ref| <= 0.005 absolute, or <= 5% relative, or the paired interval contains 0.

## 6. Decision cases
| Case | Text | Table / figure | Abstract |
|---|---|---|---|
| A. All 18 cells agree on all quantities | Law capability stated as a result "in all 18 cells", max discrepancies given | New Fig. (law), main-text table for sigma = F = 0.5, full table in supplement | Leads the abstract |
| B. Agree except some far-tail entries | State where it holds and where not (regime sentence) | Same, failures listed | Leads, with "for quantiles down to x%" |
| C. Disagreement > 5% in several cells | Law capability reported as a diagnostic in Section 4.4 only | Supplement table only | Not repositioned; moments story kept |

## 7. Protocol
Seeds fixed (2026 + replicate). Results saved per cell (JSON lines) so an interruption loses nothing.
Script: `law_study.py`; outputs `results/law_cells.jsonl`, `results/law_summary.json`; `make_law_numbers.py` writes the macros and tables.
