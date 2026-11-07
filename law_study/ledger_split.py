"""Split the residual share of the yield variance (Prop. coherent-ledger) into its diagonal part
(residual variances) and off-diagonal part (residual cross-time covariances), all 18 fishery cells."""
import sys, json
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'solver/rclh'))
import numpy as np
import rclh_solve as api
from wis_chaos.fbm import endpoint_conditioned_factor

out = []
for H in (0.1, 0.5, 0.8):
    for s in (0.25, 0.5):
        for F in (0.2, 0.5, 0.8):
            sol = api.rclh_solve(x0=.7, beta=s, hurst=H, linear_rate=1 - F, coefficients={2: -1.0}, T=2.0, N=32,
                                 modes=3, degree=5, quadrature_order=8, substeps=4)
            t = sol.times; full = endpoint_conditioned_factor(t[1:], H, modes=32).loadings
            L = s * full[:, 3:]; v = np.zeros((33, 33)); v[1:, 1:] = L @ L.T
            w = np.full(33, t[1] - t[0]); w[[0, -1]] *= .5; w *= F
            C = (1 - np.exp(-v)) * np.asarray(sol.cross_moments); var = float(w @ np.asarray(sol.covariance) @ w)
            diag = float(w @ np.diag(np.diag(C)) @ w); total = float(w @ C @ w)
            out.append(dict(H=H, sigma=s, F=F, diag_share=diag / var, off_share=(total - diag) / var, total_share=total / var))
(HERE / 'results/ledger_split.json').write_text(json.dumps(out, indent=1))
for r in out: print(r)
