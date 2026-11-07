"""Law study of the fishery application (plan: LAW_STUDY_PLAN.md).

The coherent reconstruction of RCLH is sampled on the same Gaussian vectors as the explicit
full finite-factor (Bernoulli) solution of the fishery equation, in all 18 cells of Section 4.4.
Run:  python -B law_study.py run   (writes results/law_cells.jsonl, results/fig_cell.npz)
      python -B law_study.py numbers (writes results/law_summary.json; then run make_law_numbers.py)
"""
import sys, json, time
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "solver/rclh"))
import numpy as np
from math import factorial
from numpy.polynomial.hermite_e import hermeval
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtri
from scipy.stats import qmc, norm, t as student_t
import rclh_solve as api
from wis_chaos.fbm import endpoint_conditioned_factor

OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
CELLS = [(H, s, F) for H in (0.1, 0.5, 0.8) for s in (0.25, 0.5) for F in (0.2, 0.5, 0.8)]
FIG_CELL = (0.1, 0.5, 0.5)
NREP, POWER, SEED = 8, 15, 2026
YQ = (0.01, 0.05, 0.5, 0.95, 0.99); XQ = (0.01, 0.05, 0.99)


def herm_table(u, P):
    """Normalized Hermite He_n(u)/sqrt(n!) for n = 0..P."""
    out = [np.ones_like(u)]
    for n in range(1, P + 1):
        c = np.zeros(n + 1); c[n] = 1.0
        out.append(hermeval(u, c) / np.sqrt(factorial(n)))
    return out


def coherent_paths(sol, full, M, beta, z, drop_residual=False):
    """X_coh(t_i; xi, eta), eq. (coherent-reconstruction); index 0 is t = 0."""
    xi, eta = z[:, :M], z[:, M:]
    G = beta * full[:, :M]; L = beta * full[:, M:]
    idx = sol.basis_indices; Y = sol.lawson_coefficients; P = int(idx.sum(1).max())
    out = np.empty((len(z), Y.shape[0]))
    for i in range(Y.shape[0]):
        a = G[i - 1] if i else np.zeros(M)
        at = L[i - 1] if i else np.zeros(L.shape[1])
        tabs = [herm_table(xi[:, d] - a[d], P) for d in range(M)]
        val = np.zeros(len(z))
        for k, nu in enumerate(idx):
            term = Y[i, k] * np.ones(len(z))
            for d, n in enumerate(nu):
                if n: term = term * tabs[d][n]
            val += term
        res = 1.0 if drop_residual else np.exp(eta @ at - at @ at / 2)
        out[:, i] = np.exp(xi @ a - a @ a / 2) * res * val
    return out


class Reference:
    """Explicit full finite-factor solution (Bernoulli form, Prop. A.1), 8 Gauss-Legendre nodes per cell."""
    def __init__(self, x0, g, kappa, full, beta, t, order=8):
        self.x0, self.g, self.kappa, self.t, self.order = x0, g, kappa, t, order
        self.h = beta * np.vstack([np.zeros((1, full.shape[1])), full])
        nodes, w = leggauss(order); f = (nodes + 1) / 2
        self.L = (self.h[:-1, None, :] * (1 - f[None, :, None]) + self.h[1:, None, :] * f[None, :, None]).reshape(-1, full.shape[1])
        s = (t[:-1, None] + np.diff(t)[:, None] * f).ravel()
        self.w = (np.diff(t)[:, None] * w / 2).ravel()
        self.const = g * s + .5 * np.sum(self.L ** 2, axis=1); self.cross = self.L @ self.h.T

    def paths(self, z):
        v = z @ self.L.T; out = np.full((len(z), len(self.t)), self.x0)
        for k in range(1, len(self.t)):
            j = k * self.order
            integ = np.exp(v[:, :j] + self.const[:j] - self.cross[:j, k]) @ self.w[:j]
            out[:, k] = np.exp(self.g * self.t[k] + z @ self.h[k] - .5 * self.h[k] @ self.h[k]) / (1 / self.x0 + self.kappa * integ)
        return out


def trap(t, F):
    w = np.full(len(t), t[1] - t[0]); w[[0, -1]] *= .5; return F * w


def stats(X, Y, xstar, Yref_mean):
    d = dict()
    for a in YQ: d[f"Yq{a}"] = float(np.quantile(Y, a))
    for a in XQ: d[f"Xq{a}"] = float(np.quantile(X[:, -1], a))
    d["P_low_T"] = float(np.mean(X[:, -1] < xstar / 2))
    d["P_low_path"] = float(np.mean(X.min(1) < xstar / 2))
    d["P_short"] = float(np.mean(Y < 0.8 * Yref_mean))
    d["Ymean"], d["Yvar"] = float(Y.mean()), float(Y.var())
    d["Xmean"], d["Xvar"] = float(X[:, -1].mean()), float(X[:, -1].var())
    return d


def run_cell(H, sigma, F, keep=False):
    tic = time.perf_counter()
    sol = api.rclh_solve(x0=.7, beta=sigma, hurst=H, linear_rate=1 - F, coefficients={2: -1.0}, T=2.0, N=32,
                         modes=3, degree=5, quadrature_order=8, substeps=4)
    solve_s = time.perf_counter() - tic
    t = sol.times; M = 3
    full = endpoint_conditioned_factor(t[1:], H, modes=32).loadings
    assert np.allclose(full[:, :M], api.build_factor("endpoint", hurst=H, T=2.0, N=32, modes=M).loadings)
    ref = Reference(.7, 1 - F, 1.0, full, sigma, t); w = trap(t, F); xstar = 1 - F
    # ledger: E|J - E[J|xi]|^2 = sum w w (1 - e^{-v}) K   (K = second-moment matrix of X_coh)
    Lr = sigma * full[:, M:]; v = np.zeros((33, 33)); v[1:, 1:] = Lr @ Lr.T
    K = np.asarray(sol.cross_moments); ledger = float(w @ ((1 - np.exp(-v)) * K) @ w)
    reps = []; keepers = None
    for r in range(NREP):
        z = ndtri(qmc.Sobol(32, scramble=True, seed=SEED + r).random_base2(POWER))
        Xr = ref.paths(z); Xc = coherent_paths(sol, full, M, sigma, z)
        Yr, Yc = Xr @ w, Xc @ w
        Ya = coherent_paths(sol, full, M, sigma, z, drop_residual=True) @ w
        mref = float(Yr.mean())
        mom_g = dict()
        for a in YQ: mom_g[f"Yq{a}"] = float(Yr.mean() + Yr.std() * norm.ppf(a))
        for a in XQ: mom_g[f"Xq{a}"] = float(Xr[:, -1].mean() + Xr[:, -1].std() * norm.ppf(a))
        mom_g["P_low_T"] = float(norm.cdf((xstar / 2 - Xr[:, -1].mean()) / Xr[:, -1].std()))
        mom_g["P_short"] = float(norm.cdf((0.8 * mref - mref) / Yr.std()))
        reps.append(dict(ref=stats(Xr, Yr, xstar, mref), rclh=stats(Xc, Yc, xstar, mref), gauss=mom_g,
                         pathL2_T=float(np.sqrt(np.mean((Xc[:, -1] - Xr[:, -1]) ** 2) / np.mean(Xr[:, -1] ** 2))),
                         pathL2_Y=float(np.sqrt(np.mean((Yc - Yr) ** 2) / np.mean(Yr ** 2))),
                         corr_Y=float(np.corrcoef(Yc, Yr)[0, 1]),
                         ledger_mc=float(np.mean((Yc - Ya) ** 2)),
                         neg=float(np.mean(Xc < 0))))
        if keep and r == 0:
            keepers = dict(t=t, Xr=Xr[:4096], Xc=Xc[:4096], Yr=Yr, Yc=Yc)
    return dict(H=H, sigma=sigma, F=F, solve_seconds=solve_s, ledger=ledger,
                yield_var_closed=float(w @ np.asarray(sol.covariance) @ w),
                diagnostics={k: sol.diagnostics[k] for k in ("min_nodal_value", "min_covariance_eigenvalue", "residual_trace_fraction")},
                reps=reps), keepers


def summarize(row):
    reps = row["reps"]; n = len(reps); tq = student_t.ppf(.975, n - 1)
    def ci(vals):
        vals = np.asarray(vals, float); return float(vals.mean()), float(tq * vals.std(ddof=1) / np.sqrt(n))
    s = dict(H=row["H"], sigma=row["sigma"], F=row["F"], ledger=row["ledger"], solve_seconds=row["solve_seconds"])
    for key in reps[0]["ref"]:
        ref = ci([r["ref"][key] for r in reps]); rc = ci([r["rclh"][key] for r in reps])
        diff = ci([r["rclh"][key] - r["ref"][key] for r in reps])
        e = dict(ref=ref, rclh=rc, diff=diff)
        if key in reps[0]["gauss"]:
            e["gauss"] = ci([r["gauss"][key] for r in reps])
        if key.startswith(("Yq", "Xq")):
            e["agree"] = bool(abs(diff[0]) / abs(ref[0]) <= .01 or abs(diff[0]) <= diff[1])
        elif key.startswith("P_"):
            e["agree"] = bool(abs(diff[0]) <= .005 or abs(diff[0]) <= .05 * ref[0] or abs(diff[0]) <= diff[1])
        s[key] = e
    for key in ("pathL2_T", "pathL2_Y", "corr_Y", "ledger_mc", "neg"):
        s[key] = ci([r[key] for r in reps])
    return s


def main():
    mode = sys.argv[1]
    path = OUT / "law_cells.jsonl"
    if mode == "run":
        done = set()
        if path.exists():
            for line in path.read_text().splitlines():
                r = json.loads(line); done.add((r["H"], r["sigma"], r["F"]))
        for cell in CELLS:
            if cell in done: continue
            tic = time.perf_counter()
            row, keepers = run_cell(*cell, keep=(cell == FIG_CELL))
            row["wall_seconds"] = time.perf_counter() - tic
            with path.open("a") as f: f.write(json.dumps(row) + "\n")
            if keepers: np.savez_compressed(OUT / "fig_cell.npz", **keepers)
            print(cell, f"{row['wall_seconds']:.0f}s", flush=True)
    elif mode == "numbers":
        rows = [summarize(json.loads(l)) for l in path.read_text().splitlines()]
        (OUT / "law_summary.json").write_text(json.dumps(rows, indent=1))
        print(json.dumps(rows[0], indent=1)[:3000])


if __name__ == "__main__":
    main()
