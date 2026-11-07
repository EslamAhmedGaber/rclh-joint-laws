"""Figures of the article, with one fixed visual language (no model recomputation).

Visual language (one meaning per encoding, used in every figure):
  RCLH (proposed)                   navy, solid line, filled markers
  reference / exact                 grey open circles, no line (grey band = its interval)
  active-only drift (rho = 0)       orange, dashed, open squares
  no residual cross-time cov.       purple, dash-dot, open triangles
  moments-only / Gaussian / marginal black, dotted
  RQMC                              teal, dashed, diamonds;  IID Monte Carlo  brown, dash-dot, down triangles
  tolerance / limit                 light grey dashed line, labelled in place
  a parameter (H, refinement)       lightness of the method's colour (dark = rough / fine), direct labels
File names: figlaw = Figure 1; fig2..fig7 = Figures 3..8; Figure 2 is the TikZ workflow.
Law figure (Fig. 1) from ../law_study/results/fig_cell.npz (run law_study/fig_cell_samples.py first).
"""
from pathlib import Path
import csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter
from scipy.stats import norm, gaussian_kde

F = Path(__file__).resolve().parent
STUDY = F.parent / 'law_study' / 'results'
OUT = F.parent / 'output'
(OUT / 'figures').mkdir(parents=True, exist_ok=True); (OUT / 'qa').mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 9,
    'axes.labelsize': 9.5, 'axes.titlesize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'lines.linewidth': 1.3, 'lines.markersize': 4,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'rclh-joint-laws',
    'legend.frameon': False, 'legend.handlelength': 2.0})
NAVY, GREY, ORANGE, PURPLE, BLACK = '#1F3A68', '#808080', '#D55E00', '#8E5BA6', '#222222'
TEAL, BROWN, LIMIT = '#009E73', '#A6761D', '#B5B5B5'
SHADE = {.1: '#0B1F44', .25: '#2C4F86', .4: '#5677AE', .5: '#5677AE', .75: '#97AFD6', .8: '#97AFD6', .2: '#1A3561', .3: '#3D63A0'}
MK = {.1: 'o', .2: 'v', .25: 's', .3: '^', .4: 'D', .5: '^', .75: 'D', .8: 's'}
STY = dict(rclh=dict(color=NAVY, ls='-', marker='o', mfc=NAVY),
           ref=dict(color=GREY, ls='', marker='o', mfc='white'),
           active=dict(color=ORANGE, ls='--', marker='s', mfc='white'),
           nov=dict(color=PURPLE, ls='-.', marker='^', mfc='white'),
           naive=dict(color=BLACK, ls=':', marker='', mfc=BLACK),
           rqmc=dict(color=TEAL, ls='--', marker='D', mfc=TEAL),
           mc=dict(color=BROWN, ls='-.', marker='v', mfc=BROWN))


def rows(stem):
    with (F / 'source_data' / stem).open(newline='') as f:
        rs = list(csv.DictReader(f))
    for r in rs:
        for k, v in r.items():
            try: r[k] = float(v)
            except ValueError: pass
    return rs


def h(kind=None, lab='', **over):
    s = dict(STY[kind]) if kind else dict(color=BLACK, ls='-', marker='', mfc=BLACK)
    s.update(over)
    return Line2D([], [], color=s['color'], ls=s['ls'], marker=s['marker'], mfc=s['mfc'], ms=4.2, lw=1.3, label=lab)


def key(fig, handles, x, y, ncol=1):
    return fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(x, y), ncol=ncol, borderaxespad=0,
                      columnspacing=1.1, handletextpad=.45, labelspacing=.28)


def base(w, hh): return plt.figure(figsize=(w / 25.4, hh / 25.4))


def panel(fig, bounds, title):
    ax = fig.add_axes(bounds); ax.set_title(title, loc='left', pad=6)
    ax.grid(axis='y', color='#e3e3e3', lw=.45); ax.set_axisbelow(True); ax.tick_params(length=3, pad=3)
    return ax


def limit(ax, y, text, x=.98, va='bottom'):
    ax.axhline(y, color=LIMIT, ls='--', lw=.9, zorder=0)
    ax.text(x, y, text, transform=ax.get_yaxis_transform(), ha='right', va=va, fontsize=8, color='#666666')


metrics = []


def save(fig, n, edition):
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); bad = []
    texts = list(fig.texts) + [t for leg in fig.legends for t in leg.get_texts()]
    for ax in fig.axes:
        texts += list(ax.texts) + [ax.title, ax._left_title, ax.xaxis.label, ax.yaxis.label]
    for t in texts:
        if not t.get_visible() or not t.get_text(): continue
        bb = t.get_window_extent(rend)
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.width + 1 or bb.y1 > fig.bbox.height + 1: bad.append(t.get_text())
    assert not bad, (n, edition, 'text outside canvas', bad)
    for leg in fig.legends:
        for ax in fig.axes:
            assert not leg.get_window_extent(rend).overlaps(ax.get_window_extent(rend)), (n, edition, 'key overlaps data')
            assert not leg.get_window_extent(rend).overlaps(ax._left_title.get_window_extent(rend)), (n, edition, 'key overlaps title')
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / 'figures' / f'fig{n}_{edition}.{ext}', dpi=300, facecolor='white')
    fig.savefig(OUT / 'qa' / f'fig{n}_{edition}_gray.png', dpi=140, facecolor='white')
    from PIL import Image
    p = OUT / 'qa' / f'fig{n}_{edition}_gray.png'; Image.open(p).convert('L').save(p)
    metrics.append(dict(figure=n, edition=edition, width_mm=round(fig.get_figwidth() * 25.4, 1), height_mm=round(fig.get_figheight() * 25.4, 1)))
    plt.close(fig)


# ------------------------------------------------------------------ new Fig. 1: the law from one solve
def law(w, ed):
    d = np.load(STUDY / 'fig_cell.npz'); t = d['t']; Xr, Xc, Yr, Yc = d['Xr'], d['Xc'], d['Yr'], d['Yc']
    f = base(w, 80)
    a, b, c = [panel(f, [x, .17, .235, .56], ti) for x, ti in
               zip([.095, .42, .745], ['(a) Paths on common noise', '(b) Law of the cumulative yield', '(c) Low-stock probability'])]
    for j, i in enumerate([11, 40]):
        a.plot(t, Xc[i], color=NAVY, lw=1.2)
        a.plot(t, Xr[i], ls='', marker='o', mfc='white', color=GREY, ms=3.2)
    a.set(xlim=(0, 2.04), ylim=(0, None), xlabel='Time $t$', ylabel='Stock $X_t$')
    grid = np.linspace(min(Yr.min(), Yc.min()), np.quantile(Yr, .9995), 300)
    mu, sd = Yr.mean(), Yr.std()
    b.fill_between(grid, gaussian_kde(Yr)(grid), color='#d9d9d9', lw=0)
    b.plot(grid, gaussian_kde(Yc)(grid), color=NAVY)
    b.plot(grid, norm.pdf(grid, mu, sd), color=BLACK, ls=':')
    b.set(xlabel='Cumulative yield', ylabel='Density', ylim=(0, None), xlim=(grid[0], grid[-1]))
    lev = np.linspace(.125, .45, 14)  # lowest level 0.125: 78 of 262,144 exact samples lie below it
    # use the full-replicate terminal values stored in Yr/Yc companions when available
    XrT, XcT = d['XrT'] if 'XrT' in d else Xr[:, -1], d['XcT'] if 'XcT' in d else Xc[:, -1]
    pr = np.array([np.mean(XrT < l) for l in lev]); pc = np.array([np.mean(XcT < l) for l in lev])
    pg = norm.cdf((lev - XrT.mean()) / XrT.std())
    c.plot(lev, pc, color=NAVY); c.plot(lev, pr, ls='', marker='o', mfc='white', color=GREY, ms=3.6)
    c.plot(lev, pg, color=BLACK, ls=':')
    c.set(yscale='log', xlabel='Stock level at $t=T$', ylabel='Probability of a lower stock', xlim=(.1, .47))
    c.yaxis.set_minor_formatter(NullFormatter())
    key(f, [h('rclh', 'RCLH (one coefficient solve)', marker=''), h('ref', 'Exact finite-factor solution'),
            h('naive', 'Gaussian with exact mean and variance')], .5, .995, 3)
    save(f, 'law', ed)


# ------------------------------------------------------------------ Fig. 3 (driver), restyled
def driver(w, ed):
    stacked = ed == '5p'; f = base(w, 126 if stacked else 79)
    boxes = ([.19, .57, .77, .25], [.19, .10, .77, .27]) if stacked else ([.105, .17, .365, .56], [.605, .17, .365, .56])
    a, b = [panel(f, x, ti) for x, ti in zip(boxes, ['(a) Unresolved driver variance', '(b) Missed second chaos'])]
    ft = rows('fig2a_trace_fractions.csv'); ob = rows('fig2b_wick_euler_second_chaos.csv')
    for H in [.1, .25, .4]:
        rr = [r for r in ft if r['H'] == H]; xx = [r['M'] for r in rr]
        for k, ls, m, fill in [('u_endpoint_conditioned', '-', 'o', SHADE[H]), ('u_spectral', '--', '', SHADE[H]), ('u_dyadic_hierarchy', ':', 's', 'white')]:
            a.plot(xx, [r[k] for r in rr], color=SHADE[H], ls=ls, marker=m, ms=3.4, mfc=fill)
        a.text(xx[-1] * 1.1, rr[-1]['u_endpoint_conditioned'], f'$H={H:.2f}$', color=BLACK, fontsize=8.5, va='center')
    a.set(xscale='log', yscale='log', xlim=(3.4, 145), xlabel='Retained coordinates $M$', ylabel='Unresolved trace fraction')
    a.set_xticks([4, 8, 16, 32, 64], ['4', '8', '16', '32', '64']); a.xaxis.set_minor_formatter(NullFormatter())
    ka = [h(None, 'Endpoint-conditioned', color=NAVY, marker='o', mfc=NAVY), h(None, 'Spectral', color=NAVY, ls='--'),
          h(None, 'Dyadic', color=NAVY, marker='s', ls=':', mfc='white')]
    key(f, ka, .57 if stacked else .295, .99, ncol=1 if stacked else 2)
    for H in [.1, .2, .25, .3, .4]:
        rr = [r for r in ob if r['H'] == H]; x = [r['N'] for r in rr]
        b.plot(x, [r['asymptotic_beta1'] for r in rr], color=SHADE[H])
        b.plot(x, [r['E2_sq_beta1'] for r in rr], ls='', color=SHADE[H], marker=MK[H], ms=3.2, mfc=SHADE[H])
        b.text(x[-1] * 1.3, rr[-1]['E2_sq_beta1'], f'$H={H:.2f}$', color=BLACK, fontsize=8.5, va='center')
    b.set(xscale='log', yscale='log', xlim=(3, 110000), xlabel='Time cells $N$ ($T=1$)', ylabel=r'$\|\mathcal{E}_2\|_2^2/\beta^4$')
    b.set_xticks([4, 32, 256, 2048, 16384], ['4', '32', '256', '2048', '16384']); b.xaxis.set_minor_formatter(NullFormatter())
    key(f, [h(None, 'Exact', color=NAVY, marker='o', ls='', mfc=NAVY), h(None, 'Asymptote', color=NAVY)], .57 if stacked else .79, .47 if stacked else .99, ncol=2)
    save(f, 2, ed)


# ------------------------------------------------------------------ Fig. 4 Landau
def landau(w, ed):
    f = base(w, 72); axs = [panel(f, [x, .19, .235, .60], ti) for x, ti in zip([.087, .414, .741], ['(a) Drift ablation', '(b) Nominal error bound', '(c) Joint refinement'])]
    a, b, c = axs; vs = rows('fig3a_landau_ablation.csv'); lb = rows('fig3bc_landau_campaign.csv'); hs = [.1, .25, .5]
    for M, m in [(2, 'o'), (3, 's')]:
        for mode, kind in [('residual_aware', 'rclh'), ('active_only_rho0', 'active')]:
            s = STY[kind]; rr = [next(r for r in vs if r['H'] == hh and r['M'] == M and r['P'] == 5 and r['mode'] == mode) for hh in hs]
            a.plot(hs, [r['max_second_error'] for r in rr], color=s['color'], ls=s['ls'], marker=m,
                   mfc=s['color'] if kind == 'rclh' else 'white', alpha=1 if M == 3 else .55)
    a.set(yscale='log', ylim=(2e-4, .1), ylabel='Maximum second-moment error')
    a.text(.5, 1.3e-2, 'active-only', color=ORANGE, fontsize=8, ha='right'); a.text(.5, 4.5e-4, 'RCLH', color=NAVY, fontsize=8, ha='right')
    shades = ['#97AFD6', '#5677AE', '#0B1F44']; mks = ['o', 's', 'D']
    for q, col, m in zip([216, 2401, 32768], shades, mks):
        rr = [r for r in lb if r['active_nodes'] == q]; b.plot([r['hurst'] for r in rr], [r['maximum_combined_upper_bound'] for r in rr], color=col, marker=m, mfc=col)
    for q, col, m in zip([216, 2401], shades, mks):
        rr = [r for r in lb if r['active_nodes'] == q]; c.plot([r['hurst'] for r in rr], [r['joint_refinement_shift'] for r in rr], color=col, marker=m, mfc=col)
    limit(b, .005, 'tolerance'); limit(c, .001, 'limit')
    b.set(yscale='log', ylim=(3e-4, .01), ylabel=r'Nominal bound $\varepsilon$'); c.set(yscale='log', ylim=(3e-5, .002), ylabel='Normalized refinement shift')
    for ax in axs: ax.set_xticks(hs, ['0.10', '0.25', '0.50']); ax.set_xlim(.06, .54); ax.set_xlabel('Hurst index $H$')
    key(f, [h(None, '$M=2$', color=GREY, marker='o', ls=''), h(None, '$M=3$', color=GREY, marker='s', ls='')], .205, .985, 2)
    key(f, [h(None, lab, color=col, marker=m, mfc=col) for col, m, lab in zip(shades, mks, ['coarse', 'medium', 'fine'])], .70, .985, 3)
    save(f, 3, ed)


# ------------------------------------------------------------------ Fig. 5 logistic
def logistic(w, ed):
    f = base(w, 80); a, b, c = [panel(f, [x, .17, .235, .56], ti) for x, ti in zip([.087, .414, .741], ['(a) Nominal error bound', '(b) Variance trajectories', '(c) Standardized discrepancy'])]
    lb = rows('fig4a_logistic_bounds.csv'); cur = rows('fig4b_logistic_curves.csv'); zs = rows('fig4c_logistic_z.csv'); hs = [.1, .25, .5, .75]
    for j, (q, m) in enumerate(zip([16, 64, 256], ['o', 's', 'D'])):
        for i, H in enumerate(hs):
            r = next(r for r in lb if r['hurst'] == H and r['active_nodes'] == q)
            a.plot(i + (j - 1) * .19, r['maximum_combined_upper_bound'] * 1000, m, color=SHADE[H], mfc=SHADE[H] if j == 0 else 'white')
    limit(a, 5, 'tolerance', va='top'); a.set(ylim=(0, 5.6), xlim=(-.5, 3.5), xlabel='Hurst index $H$', ylabel=r'Nominal bound ($\times10^{-3}$)'); a.set_xticks(range(4), ['0.10', '0.25', '0.50', '0.75'])
    for H in hs:
        det = sorted([r for r in cur if r['hurst'] == H and r['series'] == 'deterministic' and r['quantity'] == 'variance'], key=lambda r: r['time'])
        ref = sorted([r for r in cur if r['hurst'] == H and r['series'] == 'finite reference' and r['quantity'] == 'variance'], key=lambda r: r['time'])
        t = [r['time'] for r in ref]; y = np.array([r['value'] for r in ref]); hw = np.array([r['half_width'] for r in ref])
        b.fill_between(t, y - hw, y + hw, color='#d9d9d9', lw=0)
        b.plot([r['time'] for r in det], [r['value'] for r in det], color=SHADE[H])
        b.plot(t, y, ls='', marker='o', mfc='white', color=GREY, ms=3)
        rr = sorted([r for r in zs if r['hurst'] == H], key=lambda r: r['time'])
        c.plot([r['time'] for r in rr], [r['z'] for r in rr], color=SHADE[H], marker=MK[H], ms=3.3, mfc=SHADE[H])
    b.text(.3, .068, '$H=0.10$', fontsize=8, color=SHADE[.1]); b.text(.62, .012, '$H=0.75$', fontsize=8, color=SHADE[.75])
    b.set(ylabel='Variance of $Z_t$', ylim=(0, .095)); c.axhspan(-1, 1, color='#efefef'); c.axhline(0, color='#888888', lw=.7); c.set(ylabel='Discrepancy (band units)', ylim=(-1.6, 3.2))
    for ax in [b, c]: ax.set(xlim=(0, 1.02), xlabel='Time $t$ ($T=1$)'); ax.set_xticks([0, .5, 1])
    key(f, [h(None, f'$Q_A={q}$', color=GREY, marker=m, ls='', mfc=GREY if q == 16 else 'white') for q, m in zip([16, 64, 256], ['o', 's', 'D'])], .205, .995, 3)
    key(f, [h('rclh', 'RCLH', marker=''), h('ref', 'Reference'), Patch(facecolor='#d9d9d9', label='95% band')] +
        [h(None, f'$H={H:.2f}$', color=SHADE[H], marker=MK[H], mfc=SHADE[H]) for H in hs], .70, .995, 4)
    save(f, 4, ed)


# ------------------------------------------------------------------ Fig. 6 fishery yield
def fishery(w, ed):
    f = base(w, 86); a = panel(f, [.09, .20, .31, .55], '(a) Cumulative yield and risk'); b = panel(f, [.515, .20, .465, .55], '(b) Residual ablations')
    rs = rows('fig5a_fishery_yield.csv'); abl = rows('fig5b_fishery_ablation.csv')
    for H in [.1, .5, .8]:
        for beta, ls in [(.25, '--'), (.5, '-')]:
            rr = sorted([r for r in rs if r['H'] == H and r['beta'] == beta], key=lambda r: r['harvest'])
            a.plot(np.sqrt([r['solver_yield_variance'] for r in rr]), [r['solver_yield_mean'] for r in rr], color=SHADE[H], ls=ls)
            a.plot(np.sqrt([r['reference_yield_variance'] for r in rr]), [r['reference_yield_mean'] for r in rr], ls='', marker='o', mfc='white', color=GREY, ms=3.6)
    for r in [r for r in rs if r['H'] == .8 and r['beta'] == .5]:
        a.annotate(f"$F={r['harvest']:.1f}$", (r['solver_yield_variance'] ** .5, r['solver_yield_mean']), xytext=(5, 0), textcoords='offset points', fontsize=8.5, va='center')
    a.set(xlim=(0, .335), ylim=(.22, .95), xlabel='Yield standard deviation', ylabel='Expected cumulative yield')
    key(f, [h(None, f'$H={hh:g}$', color=SHADE[hh]) for hh in [.1, .5, .8]] + [h(None, r'$\sigma=0.25$', color=GREY, ls='--'), h('ref', 'Reference')], .25, .995, 3)
    series = [('rel_err_full', 'rclh', 'RCLH'), ('rel_err_offdiag_dropped', 'nov', 'No residual cross-time covariance'), ('active_only_rel_err', 'active', 'Active-only drift')]
    for k, kind, lab in series:
        s = STY[kind]; x = []; y = []
        for gi, H in enumerate([.1, .5, .8]):
            for si, sig in enumerate([.25, .5]):
                for fi, harv in enumerate([.2, .5, .8]):
                    r = next(r for r in abl if r['H'] == H and r['sigma'] == sig and r['F'] == harv); x.append(gi * 7 + si * 3 + fi); y.append(100 * r[k])
        b.plot(x, y, ls='', marker=s['marker'], color=s['color'], mfc=s['mfc'], ms=4.2)
    b.set(yscale='log', ylim=(.003, 30), xlim=(-.8, 20.8), ylabel='Yield-variance relative error (%)'); b.set_yticks([.01, .1, 1, 10], ['0.01', '0.1', '1', '10'])
    b.set_xticks([1, 4, 8, 11, 15, 18], ['0.25', '0.50'] * 3); b.tick_params(axis='x', length=0)
    for gi, H in enumerate([.1, .5, .8]):
        b.text(gi * 7 + 2.5, -.19, f'$H={H:.2f}$', transform=b.get_xaxis_transform(), ha='center', fontsize=9)
        if gi < 2: b.axvline(gi * 7 + 6, color='#cccccc', lw=.6)
    key(f, [h(kind, lab, ls='') for k, kind, lab in series], .745, .995, 1)
    f.text(.745, .027, r'Within each $H$: two $\sigma$ levels; $F=0.2,0.5,0.8$ in each group.', ha='center', fontsize=8.5)
    save(f, 5, ed)


# ------------------------------------------------------------------ Fig. 7 cost
def cost(w, ed):
    f = base(w, 118); axs = [panel(f, [x, y, .38, .30], ti) for x, y, ti in [(.10, .55, '(a) Logistic, $H=0.1$'), (.605, .55, '(b) Logistic, $H=0.5$'), (.10, .12, '(c) Fishery, $H=0.1$'), (.605, .12, '(d) Fishery, $H=0.5$')]]
    data = rows('fig6_cost.csv')
    for i, (ax, case) in enumerate(zip(axs, ['logistic_H0.1', 'logistic_H0.5', 'fishery_H0.1', 'fishery_H0.5'])):
        for method in ['mc', 'rqmc', 'rclh']:
            s = STY[method]; rr = sorted([r for r in data if r['case'] == case and r['method'] == method], key=lambda r: r['median_ms'])
            if method != 'rclh': ax.plot([r['median_ms'] for r in rr], [r['bundle_error_percent'] for r in rr], color=s['color'], ls=s['ls'], lw=1.0)
            for r in rr: ax.plot(r['median_ms'], r['bundle_error_percent'], color=s['color'], marker=s['marker'], mfc='white' if r['origin'] == 'exploratory' else s['mfc'], ms=4.4, ls='')
        ax.set(xscale='log', yscale='log', xlim=(1.8, 4000), ylim=(.055, 55)); ax.set_xticks([10, 100, 1000], ['10', '100', '1000']); ax.set_yticks([.1, 1, 10], ['0.1', '1', '10']); ax.grid(color='#e3e3e3', lw=.45)
        ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
        if i >= 2: ax.set_xlabel('Median wall time (ms)')
        if i % 2 == 0: ax.set_ylabel('Full-output RMS error (%)')
    key(f, [h('mc', 'IID Monte Carlo'), h('rqmc', 'Scrambled Sobol RQMC'), h('rclh', 'RCLH configurations', ls=''), h('mc', 'Larger IID extension', ls='', mfc='white')], .54, .995, 2)
    save(f, 6, ed)


# ------------------------------------------------------------------ Fig. 8 historical fishery
def history(w, ed):
    f = base(w, 84); a = panel(f, [.087, .17, .385, .55], '(a) Effort-conditioned holdout'); b = panel(f, [.603, .17, .382, .55], '(b) Conditional harvest risk')
    rs = rows('fig7a_history.csv'); risk = rows('fig7b_risk.csv'); ys = [r['year'] for r in rs]
    a.axvspan(1949.5, 1955.5, color='#f0f0f0', zorder=-1); a.plot(ys, [r['observed'] for r in rs], 'o', color=BLACK, ms=3.6)
    for k, col, ls in [('schaefer', '#444444', '-'), ('persistence', '#999999', ':'), ('trend', BROWN, '--')]:
        rr = [r for r in rs if r[k] != '']; a.plot([r['year'] for r in rr], [r[k] for r in rr], color=col, ls=ls)
    a.set(xlim=(1933.5, 1955.5), ylim=(3, 13), xlabel='Year', ylabel='CPUE (thousand lb/day)'); a.set_xticks([1935, 1940, 1945, 1950, 1955]); a.text(1952.4, 12.25, 'Held out', ha='center', fontsize=8.5)
    key(f, [h(None, 'Observed CPUE', color=BLACK, marker='o', ls=''), h(None, 'Schaefer fit', color='#444444'), h(None, 'Persistence', color='#999999', ls=':'), h(None, 'Log-linear trend', color=BROWN, ls='--')], .29, .99, 2)
    for H in [.1, .5, .8]:
        for beta, ls in [(.15, '-'), (.35, '--')]:
            rr = sorted([r for r in risk if r['H'] == H and r['beta'] == beta], key=lambda r: r['F_fraction'])
            b.plot(np.sqrt([r['solver_variance'] for r in rr]), [r['solver_mean'] for r in rr], color=SHADE[H], ls=ls)
            b.plot(np.sqrt([r['reference_variance'] for r in rr]), [r['reference_mean'] for r in rr], ls='', marker='o', mfc='white', color=GREY, ms=3.6)
    b.set(xlabel='Harvest SD (million lb)', ylabel='Mean harvest (million lb)')
    key(f, [h(None, f'$H={hh:g}$', color=SHADE[hh]) for hh in [.1, .5, .8]] + [h(None, r'$\beta=0.35$', color=GREY, ls='--'), h('ref', 'Reference')], .80, .99, 3)
    save(f, 7, ed)


if __name__ == '__main__':
    import sys
    only = set(sys.argv[1:])
    for ed, w in [('5p', 522 / 72.27 * 25.4), ('3p', 468 / 72.27 * 25.4)]:
        if not only or 'law' in only: law(w, ed)
        if not only or 'old' in only:
            driver(252 / 72.27 * 25.4 if ed == '5p' else w, ed)
            for fn in [landau, logistic, fishery, cost, history]: fn(w, ed)
    (OUT / 'qa' / 'render_geometry.json').write_text(json.dumps(metrics, indent=2))
    print('rendered', len(metrics))
