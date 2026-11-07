"""Store the samples of the figure cell (H = 0.1, sigma = F = 0.5) for the law figure:
paths of replicate 0 (first 4096) and terminal stock / yield of all 8 replicates (262,144 vectors).
Same solver, reference, seeds and Gaussian vectors as law_study.py."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
import law_study as ls

H, sigma, F = ls.FIG_CELL
sol = ls.api.rclh_solve(x0=.7, beta=sigma, hurst=H, linear_rate=1 - F, coefficients={2: -1.0}, T=2.0, N=32,
                        modes=3, degree=5, quadrature_order=8, substeps=4)
t = sol.times; full = ls.endpoint_conditioned_factor(t[1:], H, modes=32).loadings
ref = ls.Reference(.7, 1 - F, 1.0, full, sigma, t); w = ls.trap(t, F)
XrT, XcT, Yr, Yc = [], [], [], []
for r in range(ls.NREP):
    z = ndtri(qmc.Sobol(32, scramble=True, seed=ls.SEED + r).random_base2(ls.POWER))
    Xr = ref.paths(z); Xc = ls.coherent_paths(sol, full, 3, sigma, z)
    if r == 0: keep = (Xr[:4096], Xc[:4096])
    XrT.append(Xr[:, -1]); XcT.append(Xc[:, -1]); Yr.append(Xr @ w); Yc.append(Xc @ w)
np.savez_compressed(HERE / 'results/fig_cell.npz', t=t, Xr=keep[0], Xc=keep[1], XrT=np.concatenate(XrT),
                    XcT=np.concatenate(XcT), Yr=np.concatenate(Yr), Yc=np.concatenate(Yc))
print('saved', len(np.concatenate(XrT)))
