from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ['evaluate_localized', 'compare_discount_factors', 'Region',
           'LocalizedResult']

_EPS = 1e-12
RAW = 'Raw pool'


# =========================================================================== #
#  0.  SMALL NUMERICAL UTILITIES
# =========================================================================== #
def _trapz(v, x):
    return float(np.trapezoid(v, x)) if hasattr(np, 'trapezoid') \
        else float(np.trapz(v, x))


def _mono(F):
    """Force a CDF slab to be monotone and inside [0, 1]."""
    F = np.clip(np.asarray(F, float), 0.0, 1.0)
    return np.maximum.accumulate(F, axis=-1)


def _pdf(F, x):
    """Density from a CDF on the shared grid."""
    return np.maximum(np.gradient(F, x, axis=-1), 0.0)


def _quantiles(F, x, p):
    """Quantiles by inverting one CDF row; F must already be monotone."""
    Fj = F + np.linspace(0, 1e-10, F.size)      # break ties for np.interp
    return np.interp(p, Fj, x)


def _hac(d, lag=None):
    """Mean of d and its Newey-West t statistic (Giacomini-White / DM)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 5:
        return (float(np.mean(d)) if n else np.nan), np.nan
    m = float(np.mean(d))
    e = d - m
    if lag is None:
        lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lag = max(0, min(int(lag), n - 2))
    v = float(np.dot(e, e) / n)
    for k in range(1, lag + 1):
        g = float(np.dot(e[k:], e[:-k]) / n)
        v += 2.0 * (1.0 - k / (lag + 1.0)) * g
    if v <= 0:
        return m, np.nan
    return m, m / np.sqrt(v / n)


def _stars(t, ok=True):
    if not ok or t is None or not np.isfinite(t):
        return ''
    a = abs(float(t))
    return '***' if a > 2.576 else '**' if a > 1.96 else '*' if a > 1.645 else ''


def _ks(u):
    u = np.asarray(u, float)
    u = u[np.isfinite(u)]
    if u.size < 3:
        return np.nan, np.nan
    r = stats.kstest(np.clip(u, 1e-9, 1 - 1e-9), 'uniform')
    return float(r.statistic), float(r.pvalue)


def _get(rep, *names, default=None, required=False):
    """Fetch the first key that exists, so the module tolerates small naming
    differences between the four recalibrators."""
    for nm in names:
        if nm in rep and rep[nm] is not None:
            return rep[nm]
    if required:
        raise KeyError(f"report has none of {names}; keys are "
                       f"{sorted(rep.keys())}")
    return default


# =========================================================================== #
#  1.  REGIONS  =  weight function 1_A(y)  +  its censoring pivots
# =========================================================================== #
@dataclass
class Region:
    """A time-varying region A_t together with the pivotal points that carry
    the censored mass Fbar (LSPS Sec. 3.2-3.3).

    kind
      'left'        A = (-inf, b_t)      pivot b_t
      'right'       A = (a_t, +inf)      pivot a_t
      'interval'    A = [a_t, b_t)       pivots a_t, b_t, split (gamma, 1-gamma)
      'complement'  A = R \\ [a_t, b_t]  single pivot at `centre`

    Half-open endpoints are what make `block_regions` a genuine PARTITION: with
    both ends open a realisation landing exactly on a cut belongs to no block.
    """
    name: str
    kind: str
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    centre: float = 0.0
    gamma: float = 1.0
    lo_closed: bool = False
    hi_closed: bool = False
    group: str = 'other'                 # 'tail' | 'block' | 'band'
    meta: dict = field(default_factory=dict)

    # -- membership ------------------------------------------------------- #
    def contains(self, t, z):
        z = np.asarray(z, float)
        if self.kind == 'left':
            return ((z <= self.hi[t]) if self.hi_closed
                    else (z < self.hi[t])).astype(float)
        if self.kind == 'right':
            return ((z >= self.lo[t]) if self.lo_closed
                    else (z > self.lo[t])).astype(float)
        up = (z >= self.lo[t]) if self.lo_closed else (z > self.lo[t])
        dn = (z <= self.hi[t]) if self.hi_closed else (z < self.hi[t])
        ins = up & dn
        return ins.astype(float) if self.kind == 'interval' \
            else (~ins).astype(float)

    def slice(self, sl):
        """Same region with thresholds cut to the evaluation window, so region
        time indexing and law time indexing agree."""
        return Region(self.name, self.kind,
                      None if self.lo is None else np.asarray(self.lo)[sl],
                      None if self.hi is None else np.asarray(self.hi)[sl],
                      self.centre, self.gamma, self.lo_closed, self.hi_closed,
                      self.group, dict(self.meta))

    def bounds(self, t):
        lo = -np.inf if self.lo is None else float(self.lo[t])
        hi = np.inf if self.hi is None else float(self.hi[t])
        return lo, hi

    def pivots(self, t):
        lo, hi = self.bounds(t)
        if self.kind == 'left':
            return np.array([hi]), np.array([1.0])
        if self.kind == 'right':
            return np.array([lo]), np.array([1.0])
        if self.kind == 'interval':
            return np.array([lo, hi]), np.array([self.gamma, 1.0 - self.gamma])
        return np.array([self.centre]), np.array([1.0])

    def mass(self, t, F, x):
        """P_F(A_t), read off the interpolated CDF (no quadrature round trip)."""
        lo, hi = self.bounds(t)
        Flo = 0.0 if not np.isfinite(lo) else float(np.interp(lo, x, F))
        Fhi = 1.0 if not np.isfinite(hi) else float(np.interp(hi, x, F))
        if self.kind == 'left':
            return Fhi
        if self.kind == 'right':
            return 1.0 - Flo
        inner = max(Fhi - Flo, 0.0)
        return inner if self.kind == 'interval' else 1.0 - inner


# ---- threshold builders --------------------------------------------------- #
def _causal_quantile(y, q, start, window=None, min_hist=12):
    """r_t = q-quantile of the past of y, strictly before t.

    `window=None` gives the expanding climatology.  An INTEGER gives a rolling
    window, which is what LSPS use (the estimation window) and what keeps the
    region meaningful when the series has a level shift: with an expanding
    climatology and US inflation over 2021-24, the top block of an
    "equal-probability" partition can end up holding no observations at all.
    """
    y = np.asarray(y, float)
    T = y.size
    r = np.full(T, np.nan)
    for t in range(T):
        lo = 0 if window is None else max(0, t - int(window))
        past = y[lo:t]
        if past.size >= min_hist:
            r[t] = np.quantile(past, q)
    if np.isfinite(r).any():                    # back-fill the burn-in
        first = int(np.argmax(np.isfinite(r)))
        r[:first] = r[first]
    if not np.isfinite(r[start:]).all():
        raise ValueError(f"threshold undefined inside the window (q={q}); "
                         f"raise eval_start or lower min_hist")
    return r


def _gamma_presample(y, lo, hi, start):
    """Share of the censored mass placed at the LOWER pivot of an interval.

    LSPS tune this with the fraction of observations below the lower pivot.
    Estimated from y STRICTLY BEFORE the evaluation window, so it is a constant
    on the scored sample and identical across candidates -- otherwise the rule
    stops being localizing (LSPS Example D.3).
    """
    past = np.asarray(y, float)[:start]
    if past.size < 10:
        return 0.5
    below = float(np.mean(past <= lo))
    out = float(np.mean((past <= lo) | (past >= hi)))
    return float(np.clip(below / max(out, 1e-9), 0.0, 1.0))


def _flag_duplicates(regs, tol=1e-12):
    """Mark regions that describe the SAME SET as one already listed.

    They are kept -- dropping one would break the partition -- but the `note`
    column says so, because two identical rows of numbers read as two pieces of
    evidence when they are one.
    """
    for i, r in enumerate(regs):
        for k in regs[:i]:
            if (k.kind, k.lo_closed, k.hi_closed) != \
                    (r.kind, r.lo_closed, r.hi_closed):
                continue
            same = True
            for a, b in ((k.lo, r.lo), (k.hi, r.hi)):
                if (a is None) != (b is None):
                    same = False
                elif a is not None and not np.allclose(a, b, atol=tol,
                                                       equal_nan=True):
                    same = False
            if same:
                r.meta['same_as'] = k.name
                break
    return regs


def _count_in(reg, y, sl):
    """Realised number of observations inside A_t over the scored window."""
    r = reg.slice(sl)
    ys = np.asarray(y, float)[sl]
    return int(sum(r.contains(t, np.array([ys[t]]))[0] for t in range(ys.size)))


def build_regions(y, eval_start, min_events=20, n_blocks=4,
                  tail_levels=(0.10, 0.15, 0.20, 0.25, 0.33),
                  threshold_window='auto', target=None, radii=(1.0,),
                  verbose=True):
    """Region set in which EVERY member is large enough to be estimated.

    tails   the smallest level q in `tail_levels` whose REALISED counts clear
            `min_events` on both sides, plus the next level up so the tail
            claim can be shown to be robust to the cut.
    blocks  an equal-probability PARTITION, with the number of blocks reduced
            from `n_blocks` until every block's REALISED count clears
            `min_events`.
    band    optional LSPS Sec. 4.2 inflation-target region: (target-r, target+r)
            and its complement.  Pass `target=2.0` for CPI inflation; leave it
            None for FX, where no exogenous centre exists.

    `threshold_window`: 'auto' uses a rolling window of max(48, eval_start)
    observations, None uses the expanding climatology, an int is used as is.
    """
    y = np.asarray(y, float)
    T, sl = y.size, slice(int(eval_start), None)
    n = T - int(eval_start)
    if threshold_window == 'auto':
        threshold_window = int(max(48, eval_start))
    win = None if threshold_window is None else int(threshold_window)

    def _mk_tails(q):
        return [Region(f"left tail  q={q:g}", 'left',
                       hi=_causal_quantile(y, q, eval_start, win),
                       group='tail', meta={'q': q, 'side': 'L'}),
                Region(f"right tail q={1 - q:g}", 'right',
                       lo=_causal_quantile(y, 1 - q, eval_start, win),
                       group='tail', meta={'q': 1 - q, 'side': 'R'})]

    # ---- blocks: reduce until every block is estimable ------------------ #
    def _mk_blocks(nb):
        cuts = [_causal_quantile(y, k / nb, eval_start, win)
                for k in range(1, nb)]
        out = []
        for k in range(nb):
            lo = cuts[k - 1] if k > 0 else None
            hi = cuts[k] if k < nb - 1 else None
            nm = f"block {k + 1}/{nb}"
            if lo is None:
                out.append(Region(nm, 'left', hi=hi, group='block',
                                  meta={'share': 1 / nb}))
            elif hi is None:
                out.append(Region(nm, 'right', lo=lo, lo_closed=True,
                                  group='block', meta={'share': 1 / nb}))
            else:
                g = _gamma_presample(y, float(np.nanmedian(lo[sl])),
                                     float(np.nanmedian(hi[sl])), eval_start)
                out.append(Region(nm, 'interval', lo=lo, hi=hi, gamma=g,
                                  lo_closed=True, group='block',
                                  meta={'share': 1 / nb}))
        return out

    blocks, nb_used = [], None
    for nb in range(int(n_blocks), 1, -1):
        cand = _mk_blocks(nb)
        if min(_count_in(r, y, sl) for r in cand) >= min_events:
            blocks, nb_used = cand, nb
            break
    if not blocks:
        blocks, nb_used = _mk_blocks(2), 2

    # ---- tails: tightest feasible level that is not a block cut --------- #
    # A tail at q = k/nb is the SAME SET as the outer block, so reporting both
    # prints one region twice and reads as corroboration.  Levels colliding
    # with a cut are skipped unless nothing else is feasible.
    cuts_q = {round(k / nb_used, 6) for k in range(1, nb_used)}
    prefer = [q for q in sorted(tail_levels) if round(q, 6) not in cuts_q]
    ladder = prefer or sorted(tail_levels)
    tails, q_used = [], None
    for q in ladder:
        cand = _mk_tails(q)
        if min(_count_in(r, y, sl) for r in cand) >= min_events:
            q_used, tails = q, cand
            nxt = [v for v in ladder if v > q]
            if nxt:
                extra = _mk_tails(nxt[0])
                if min(_count_in(r, y, sl) for r in extra) >= min_events:
                    tails += extra
            break
    if not tails:                                   # nothing clears the bar
        q_used = max(ladder)
        tails = _mk_tails(q_used)

    # ---- optional exogenous band (LSPS Sec. 4.2) ------------------------ #
    band = []
    if target is not None:
        for r in radii:
            lo = np.full(T, float(target) - r)
            hi = np.full(T, float(target) + r)
            g = _gamma_presample(y, target - r, target + r, eval_start)
            band.append(Region(f"band {target:g}+-{r:g}", 'interval',
                               lo=lo, hi=hi, gamma=g, lo_closed=True,
                               group='band'))
            band.append(Region(f"off band {target:g}+-{r:g}", 'complement',
                               lo=lo, hi=hi, centre=float(target),
                               group='band'))

    regs = tails + blocks + band
    _flag_duplicates(regs)
    if verbose:
        cnt = ', '.join(f"{r.name}:{_count_in(r, y, sl)}" for r in regs)
        wtxt = 'expanding' if win is None else f'rolling {win}'
        print(f"  regions ({wtxt} thresholds, n={n}, min_events={min_events}): "
              f"tails q={q_used:g}, {nb_used} blocks")
        print(f"    realised counts -> {cnt}")
    return regs


# =========================================================================== #
#  2.  PREDICTIVE LAWS
#      After this section nothing downstream knows which method produced a law.
# =========================================================================== #
@dataclass
class PredictiveLaw:
    name: str
    F: np.ndarray            # (n, M) CDF on the shared grid
    f: np.ndarray            # (n, M) density on the shared grid
    log_f_y: np.ndarray      # (n,)   log predictive density at the observation
    pit: np.ndarray          # (n,)
    log_jac: np.ndarray      # (n,)   log tau'(u_t); zeros for the raw pool
    mass: float              # grid mass, should be ~1

    @staticmethod
    def _finish(name, F, x, lfy, pit, jac):
        F = _mono(F)
        return PredictiveLaw(name, F, _pdf(F, x), np.asarray(lfy, float),
                             np.asarray(pit, float), np.asarray(jac, float),
                             float(np.mean(F[:, -1] - F[:, 0])))

    @classmethod
    def raw(cls, x, cdf_slab, y, sl, pit_stored=None):
        """The base pool.  The PIT is taken from `pit_data` when supplied
        rather than re-derived by interpolating y on the grid: the stored PIT
        is the one the recalibrators were actually fitted to."""
        F = _mono(np.asarray(cdf_slab, float)[sl])
        f = _pdf(F, x)
        ys = np.asarray(y, float)[sl]
        lfy = np.log(np.maximum(
            [np.interp(ys[i], x, f[i]) for i in range(ys.size)], 1e-300))
        pit = (np.array([np.interp(ys[i], x, F[i]) for i in range(ys.size)])
               if pit_stored is None
               else np.asarray(pit_stored, float)[sl])
        return cls._finish(RAW, F, x, lfy, pit, np.zeros(ys.size))

    @classmethod
    def recalibrated(cls, name, x, cdf_slab, y, sl, tau_grid, u_grid,
                     log_jac, pit_after):
        """G_t(z) = tau_t(F_t(z)),  g_t(z) = f_t(z) tau'_t(F_t(z)).

        log g(y) = log f(y) + log tau'(u) uses the recalibrator's own analytic
        Jacobian for the second term, so the unweighted log-score gain here
        reproduces the number the recalibrator already printed, to the digit.
        The grid copy of G is what supplies Fbar, the quantiles and the
        intervals.
        """
        F = _mono(np.asarray(cdf_slab, float)[sl])
        f = _pdf(F, x)
        ys = np.asarray(y, float)[sl]
        tau = np.asarray(tau_grid, float)[sl]
        ug = np.concatenate(([0.0], np.asarray(u_grid, float), [1.0]))
        G = np.empty_like(F)
        for i in range(F.shape[0]):
            tp = np.concatenate(([0.0], np.clip(tau[i], 0.0, 1.0), [1.0]))
            G[i] = np.interp(F[i], ug, np.maximum.accumulate(tp))
        jac = np.asarray(log_jac, float)[sl]
        lfy = np.log(np.maximum(
            [np.interp(ys[i], x, f[i]) for i in range(ys.size)], 1e-300)) + jac
        return cls._finish(name, G, x, lfy,
                           np.asarray(pit_after, float)[sl], jac)


# =========================================================================== #
#  3.  CENSORED (LOCALIZED) SCORING RULES -- LSPS Table 1, positively oriented
# =========================================================================== #
def _censored_scores(law, reg, t, y_t, x, do_power=False, alpha=2.0):
    """Every localized score for ONE law, ONE region, ONE period.

    Flat function on purpose: the raw pool and every recalibrator traverse
    literally the same code path, which is what makes the differences mean
    something.
    """
    F, f = law.F[t], law.f[t]
    w = float(reg.contains(t, np.array([y_t]))[0])
    lo, hi = reg.bounds(t)
    pA = float(np.clip(reg.mass(t, F, x), 1e-12, 1 - 1e-12))
    Fbar = 1.0 - pA
    lfy = float(law.log_f_y[t])
    out = {}

    # -- logarithmic family (LSPS Table 1, semi-local) ------------------- #
    out['CL'] = w * (lfy - np.log(pA))                       # LogS_sharp
    out['Bern'] = w * np.log(pA) + (1 - w) * np.log(Fbar)
    out['CSL'] = out['CL'] + out['Bern']                     # LogS_flat
    out['w'] = w
    out['pA'] = pA

    # -- censored CRPS (= twCRPS for one-sided indicators) --------------- #
    piv, gam = reg.pivots(t)
    extra = [y_t]
    for b in list(piv) + [lo, hi]:                # resolve every kink exactly:
        if np.isfinite(b):                        # an indicator STEPS at the
            extra += [b - 1e-9, b, b + 1e-9]      # bound, and a trapezoid
    xg = np.unique(np.concatenate([x, np.asarray(extra, float)]))
    xg = xg[(xg >= x[0]) & (xg <= x[-1])]         # straddling it halves the
    Fg = np.interp(xg, x, F)                      # boundary cell -- which is
    Flo = 0.0 if not np.isfinite(lo) else float(np.interp(lo, x, F))
    Fhi = 1.0 if not np.isfinite(hi) else float(np.interp(hi, x, F))

    if reg.kind == 'left':
        cumA = np.minimum(Fg, Fhi)
    elif reg.kind == 'right':
        cumA = np.maximum(Fg - Flo, 0.0)
    elif reg.kind == 'interval':
        cumA = np.clip(np.minimum(Fg, Fhi) - Flo, 0.0, None)
    else:
        cumA = np.minimum(Fg, Flo) + np.maximum(Fg - Fhi, 0.0)

    jump = np.zeros_like(xg)
    for p, g in zip(piv, gam):
        jump += g * Fbar * (xg >= p)
    Fflat = np.maximum.accumulate(np.clip(cumA + jump, 0.0, 1.0))

    def crps_at(z):
        return _trapz((Fflat - (xg >= z)) ** 2, xg)

    # off A the observation is represented by the SAME fixed pivot mixture that
    # carries the censored forecast mass -- that is what keeps the rule 'flat',
    # hence strictly locally proper: the representation must not depend on F
    out['CRPSf'] = crps_at(y_t) if w > 0 else \
        float(sum(g * crps_at(p) for p, g in zip(piv, gam)))

    # -- power / pseudospherical (optional; truncated by the finite grid) -- #
    if do_power:
        fa = f ** alpha
        if reg.kind in ('left', 'right', 'interval'):
            a, b = max(lo, x[0]), min(hi, x[-1])
            m = (x >= a) & (x <= b)
            na = _trapz(fa[m], x[m]) if m.sum() > 1 else 0.0
        else:
            m1, m2 = x <= max(lo, x[0]), x >= min(hi, x[-1])
            na = ((_trapz(fa[m1], x[m1]) if m1.sum() > 1 else 0.0)
                  + (_trapz(fa[m2], x[m2]) if m2.sum() > 1 else 0.0))
        na = max(na, 1e-300)
        fy = np.exp(min(lfy, 700.0))
        den = na + Fbar ** alpha
        out['QSf'] = (w * alpha * fy ** (alpha - 1)
                      + (1 - w) * alpha * Fbar ** (alpha - 1)
                      - (alpha - 1) * den)
        out['SphSf'] = ((w * fy ** (alpha - 1) + (1 - w) * Fbar ** (alpha - 1))
                        / den ** ((alpha - 1) / alpha))
    return out


# =========================================================================== #
#  4.  INTERVALS, PINBALL, TAIL CALIBRATION
# =========================================================================== #
def _interval_stats(law, x, ys, levels):
    """Coverage, mean width and the interval score of Gneiting-Raftery (2007),
    with the Bracher et al. (2021) split into WIDTH and MISS.  Both are
    reported because a monotone recalibration map moves probability along the
    axis: it can widen an interval and move a miss across the boundary at the
    same time, and those two effects cancel in the total."""
    n = ys.size
    Q = {}
    probs = sorted({p for a in levels for p in ((1 - a) / 2, (1 + a) / 2)}
                   | {0.5})
    qs = np.stack([_quantiles(law.F[i], x, probs) for i in range(n)])
    for j, p in enumerate(probs):
        Q[round(p, 6)] = qs[:, j]
    out = {}
    for a in levels:
        lo, hi = Q[round((1 - a) / 2, 6)], Q[round((1 + a) / 2, 6)]
        cov = (ys >= lo) & (ys <= hi)
        width = hi - lo
        miss = (2.0 / (1 - a)) * (np.maximum(lo - ys, 0)
                                  + np.maximum(ys - hi, 0))
        out[a] = dict(cov=float(np.mean(cov)), width=float(np.mean(width)),
                      IS=width + miss, miss=float(np.mean(miss)))
    out['median'] = Q[0.5]
    return out


def _pinball(law, x, ys, probs=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)):
    n = ys.size
    qs = np.stack([_quantiles(law.F[i], x, probs) for i in range(n)])
    L = np.zeros(n)
    for j, p in enumerate(probs):
        e = ys - qs[:, j]
        L += np.maximum(p * e, (p - 1) * e)
    return L / len(probs)


def _crps_unweighted(law, x, ys):
    n = ys.size
    out = np.empty(n)
    for i in range(n):
        out[i] = _trapz((law.F[i] - (x >= ys[i])) ** 2, x)
    return out


def _tail_calibration(law, x, ys, reg):
    """Allen, Koh, Segers & Ziegel (2025) diagnostics for one tail region.

    occurrence ratio  sum 1{y_i > r_i} / sum (1 - F_i(r_i))   -> 1 if the
                      forecast gets the PROBABILITY of the tail event right
    severity          uniformity of the excess PITs
                          z_i = (F_i(y_i) - F_i(r_i)) / (1 - F_i(r_i))
                      -> did it get the SHAPE inside the tail right
    combined          KS of the combined ratio
                          R(u) = #{z_i <= u} / sum (1 - F_i(r_i))
                      against the diagonal

    This is the same occurrence/severity split as CSL = Bern + CL, arrived at
    from the calibration side instead of the scoring side -- which is why the
    two panels of the output should tell the same story.
    """
    n = ys.size
    side = reg.meta.get('side')
    if side is None:
        return {}
    r = np.array([reg.bounds(t)[1 if side == 'L' else 0] for t in range(n)])
    Fr = np.array([np.interp(r[i], x, law.F[i]) for i in range(n)])
    Fy = np.array([np.interp(ys[i], x, law.F[i]) for i in range(n)])
    if side == 'L':
        hit, pex, z = ys < r, Fr, Fy / np.maximum(Fr, _EPS)
    else:
        hit, pex, z = ys > r, 1.0 - Fr, ((Fy - Fr)
                                         / np.maximum(1.0 - Fr, _EPS))
    denom = float(np.sum(pex))
    occ = float(np.sum(hit)) / denom if denom > 0 else np.nan
    zz = np.clip(z[hit], 0.0, 1.0)
    sev = _ks(zz)[0] if zz.size >= 3 else np.nan
    if zz.size >= 3 and denom > 0:
        u = np.linspace(0, 1, 101)
        R = np.array([np.sum(zz <= v) for v in u]) / denom
        comb = float(np.max(np.abs(R - u)))
    else:
        comb = np.nan
    return {'occurrence ratio': occ, 'severity KS': sev, 'combined KS': comb,
            'n exceed': int(np.sum(hit))}


# =========================================================================== #
#  5.  INPUT ALIGNMENT
#      Fbar, every quantile and every interval are read off cdf_data.  If
#      cdf_data is not the CDF whose PIT sits in pit_data, every LEVEL is
#      silently wrong while the unweighted GAINS stay exactly right, because
#      log f(y) is common to F and to tau(F) and cancels.  That combination is
#      the most dangerous one available, so it is checked rather than assumed.
# =========================================================================== #
def _pit_from_grid(x, slab, y):
    return np.array([np.interp(y[i], x, _mono(slab[i]))
                     for i in range(min(len(y), len(slab)))])


def _locate_pool(pit_data, u_ref, tol=1e-6):
    u = np.clip(np.asarray(u_ref, float), 1e-6, 1 - 1e-6)
    for j, p in enumerate(pit_data):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        if p.shape == u.shape and np.max(np.abs(p - u)) < tol:
            return j
    raise ValueError("this report's raw PITs match no row of pit_data -- the "
                     "reports and the pool arrays are not from the same run")


def _find_offset(x, y, pit_data, cdf_data, tol=5e-3):
    """Where y sits inside pool arrays built on the longer estimation sample."""
    T, Tp = len(y), pit_data.shape[1]
    if T == Tp:
        return 0
    best, berr = None, np.inf
    for off in range(0, Tp - T + 1):
        for p in range(cdf_data.shape[0]):
            u = _pit_from_grid(x, cdf_data[p, off:off + T], y)
            e = float(np.max(np.abs(u - pit_data[p, off:off + T])))
            if e < berr:
                best, berr = off, e
            if e <= tol:
                return off
    return best if best is not None else Tp - T


# =========================================================================== #
#  6.  RESULT CONTAINER
# =========================================================================== #
@dataclass
class LocalizedResult:
    discount_factor: float
    window: tuple
    n: int
    methods: list
    frames: dict
    notes: list

    def show(self, floatfmt='{:.4f}'):
        titles = {
            'master': 'MASTER TABLE -- levels  (delta = gain over the raw '
                      'pool, POSITIVE ALWAYS MEANS BETTER; HAC t-stat stars)',
            'by_region': 'BY REGION -- censored (LSPS) scores, gain over the '
                         'raw pool, and where the recalibration map acts',
            'diagnostics': 'INPUT CHECKS',
        }
        for k in ('diagnostics', 'master', 'by_region'):
            df = self.frames.get(k)
            if df is None or not len(df):
                continue
            print(f"\n{'=' * 118}\n{titles.get(k, k)}"
                  f"\n[discount factor {self.discount_factor:g}   window "
                  f"[{self.window[0]},{self.window[1]})   n={self.n}]"
                  f"\n{'=' * 118}")
            with pd.option_context('display.width', 300,
                                   'display.max_columns', 80,
                                   'display.max_rows', 400,
                                   'display.float_format',
                                   lambda v: floatfmt.format(v)):
                print(df.to_string(na_rep=''))
        self.verdict()
        if self.notes:
            print("\nNOTES")
            for nt in self.notes:
                print(f"  * {nt}")

    # -- the plain-English answer --------------------------------------- #
    def verdict(self):
        m = self.frames.get('master_levels')
        if m is None or not len(m):
            return
        cand = [c for c in m.columns if c != RAW]
        print(f"\n{'=' * 118}\nWHAT RECALIBRATION ACTUALLY IMPROVES  "
              f"(discount factor {self.discount_factor:g})\n{'=' * 118}")

        def g(row, col):
            try:
                return float(m.loc[row, col])
            except Exception:
                return np.nan

        for c in cand:
            gain = g('LogS  (unweighted)', c) - g('LogS  (unweighted)', RAW)
            tl = g('CSL  left tail', c) - g('CSL  left tail', RAW)
            tr = g('CSL  right tail', c) - g('CSL  right tail', RAW)
            print(f"\n  {c}")
            print(f"    unweighted LogS gain      {gain:+.4f}")
            best = np.nanmax([tl, tr])
            if np.isfinite(best) and best > 0 and abs(gain) > 1e-9:
                print(f"    tail CSL gain  L {tl:+.4f}   R {tr:+.4f}"
                      f"   -> best tail is {best / abs(gain):.1f}x the "
                      f"unweighted gain")
            else:
                print(f"    tail CSL gain  L {tl:+.4f}   R {tr:+.4f}"
                      f"   -> NO tail improvement at this cut")
            rb = g('gain-vs-level corr across blocks (neg = rebalancing)', c)
            if np.isfinite(rb):
                print(f"    balance                   corr(level, gain) across "
                      f"blocks = {rb:+.2f}"
                      f"   -> {'redistributes toward the weak blocks' if rb < -0.2 else 'no clear redistribution'}")
            cv0, cv1 = (g('mean |coverage - nominal|', RAW),
                        g('mean |coverage - nominal|', c))
            is0, is1 = (g('interval score 90% (lower better)', RAW),
                        g('interval score 90% (lower better)', c))
            if np.isfinite(cv1):
                print(f"    intervals                 mean |cov-nominal| "
                      f"{cv0:.3f} -> {cv1:.3f};  IS90 {is0:.4f} -> {is1:.4f}"
                      f"   -> {'better' if (cv1 < cv0 and is1 < is0) else 'mixed' if (cv1 < cv0 or is1 < is0) else 'worse'}")
        print("\n  Mechanism: in the by-region table, \"mean log tau'(u) | y in "
              "A\" is the average log-Jacobian\n  of the recalibration map over "
              "the periods whose outcome landed in A.  The unweighted LogS gain\n"
              "  is that same quantity averaged over ALL periods, so a tail "
              "value several times larger is\n  direct evidence that the map is "
              "doing its work in the tail -- which is the ESOBE claim.")

    def to_excel(self, path):
        with pd.ExcelWriter(path) as xl:
            for k, df in self.frames.items():
                if df is not None and len(df):
                    df.to_excel(xl, sheet_name=k[:31])
        print(f"written: {path}")


# =========================================================================== #
#  7.  THE FUNCTION YOU CALL
# =========================================================================== #
def static_reports(gas_reports, family, K=None):
    """Reports for the static expanding-window map of eq. (9), built from the GAS reports
    that already carry it, in the form evaluate_localized expects. Pass the result as one
    more entry of `methods`, e.g.

        methods = {'GAS spline': rg, 'CONV spline': rc, 'GAS beta': bg, 'CONV beta': bc,
                   'Static spline': static_reports(rg, 'rqs', K=6),
                   'Static beta':   static_reports(bg, 'beta')}

    family  'rqs'  : uses 'theta_static' (needs K, the number of spline bins)
            'beta' : uses 'logits_static'
    Each report gets: delta, u_original, eval_start, warmup, tau_grid (the static map on
    u_grid), u_grid, log_likelihood (= ll_static, its log-Jacobian) and v_transformed
    (= v_static). The map is piecewise constant between refits, so it is evaluated once
    per distinct parameter vector."""
    out = []
    for r in gas_reports:
        ug = np.asarray(_get(r, 'u_grid', required=True), float)
        if family == 'rqs':
            if K is None:
                raise ValueError("family='rqs' needs K")
            from recal_dma_new import _spline_on_grid, _UN
            if ug.shape != _UN.shape or not np.allclose(ug, _UN):
                raise ValueError("u_grid is not the recalibrator's quadrature grid")
            P = np.asarray(_get(r, 'theta_static', required=True), float)
            tau_of = lambda th: _spline_on_grid(th, K)[0]
        elif family == 'beta':
            from recal_beta_dma_new import _ab
            from scipy.special import betainc
            P = np.asarray(_get(r, 'logits_static', required=True), float)

            def tau_of(L):
                a, b = _ab(np.array([L[0]]), np.array([L[1]]))
                return betainc(a[0], b[0], np.clip(ug, 1e-12, 1 - 1e-12))
        else:
            raise ValueError("family must be 'rqs' or 'beta'")
        tau = np.empty((P.shape[0], ug.size))
        cache = {}
        for t in range(P.shape[0]):
            key = P[t].tobytes()
            if key not in cache:
                cache[key] = tau_of(P[t])
            tau[t] = cache[key]
        out.append({'delta': _get(r, 'delta', 'discount_factor', 'df', required=True),
                    'u_original': _get(r, 'u_original', 'u_raw', 'pit', required=True),
                    'eval_start': r.get('eval_start'), 'warmup': r.get('warmup'),
                    'tau_grid': tau, 'u_grid': ug,
                    'log_likelihood': np.asarray(_get(r, 'll_static', required=True), float),
                    'v_transformed': np.asarray(_get(r, 'v_static', required=True), float)})
    return out


def evaluate_localized(x, y, pit_data, cdf_data, methods, discount_factor,
                       eval_start=None, min_events=20, n_blocks=4,
                       threshold_window='auto', target=None, radii=(1.0,),
                       interval_levels=(0.50, 0.80, 0.90, 0.95),
                       hac_lag=None, do_power=False, alpha=2.0,
                       regions=None, verbose=True):
    """Localized comparison of the raw pool against every recalibrator, at ONE
    discount factor.

    Parameters
    ----------
    x, y             shared support grid and realisations, exactly as passed to
                     the recalibrators.  `x` strictly increasing.
    pit_data         (P, T) raw PITs, one row per base pool.
                     inflation: `pit_values`      FX: the TLP / linear-pool PITs
    cdf_data         (P, T, M) raw CDFs on `x`, one slab per base pool.
                     inflation: `cdfs`            FX: the matching CDF slabs
    methods          {'label': list_of_reports}, e.g.
                     {'GAS spline': rg_dma, 'CONV spline': rc_dma,
                      'GAS beta': bg_dma,  'CONV beta': bc_dma}
    discount_factor  THE FILTER.  Only the report whose 'delta' equals this is
                     taken from each method, so every table compares like with
                     like against a single raw pool.  Nothing else appears.
    min_events       minimum REALISED observations inside a region.  Regions are
                     coarsened until every one of them clears it.
    threshold_window 'auto' (rolling, max(48, eval_start)), None (expanding),
                     or an int.  Rolling is the right default on a series with
                     a level shift.
    target           exogenous centre for the LSPS Sec. 4.2 band region.
                     2.0 for CPI inflation; leave None for FX.
    do_power         also report the censored power / pseudospherical rules.
                     Off by default: they need alpha-norms and are truncated by
                     the finite x grid.

    Returns
    -------
    LocalizedResult -- `.show()` prints the tables, `.frames` holds them,
    `.to_excel(path)` writes them out.
    """
    x = np.asarray(x, float).ravel()
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing")
    y = np.asarray(y, float).ravel()
    T = y.size
    notes = []

    pit_data = np.atleast_2d(np.asarray(pit_data, float))
    cdf_data = np.asarray(cdf_data, float)
    if cdf_data.ndim == 2:
        cdf_data = cdf_data[None]
    if cdf_data.shape[-1] != x.size:
        raise ValueError(f"cdf_data's last axis is {cdf_data.shape[-1]} but x "
                         f"has {x.size} points -- transpose cdf_data to "
                         f"(P, T, M)")

    # ---- one report per method, at this discount factor ---------------- #
    picked = {}
    for nm, reps in methods.items():
        reps = list(reps)
        hits = [r for r in reps
                if abs(float(_get(r, 'delta', 'discount_factor', 'df',
                                  required=True)) - float(discount_factor))
                <= 1e-9]
        if not hits:
            have = sorted({float(_get(r, 'delta', 'discount_factor', 'df'))
                           for r in reps})
            raise ValueError(f"{nm}: no report at discount factor "
                             f"{discount_factor}; available {have}")
        if len(hits) > 1:
            raise ValueError(f"{nm}: {len(hits)} reports at discount factor "
                             f"{discount_factor}")
        picked[nm] = dict(hits[0])
    if not picked:
        raise ValueError("no methods given")

    # ---- they must share a base pool ----------------------------------- #
    pools = {nm: _locate_pool(pit_data, _get(r, 'u_original', 'u_raw', 'pit',
                                             required=True))
             for nm, r in picked.items()}
    if len(set(pools.values())) != 1:
        raise ValueError("the selected reports do not share a base pool ("
                         + ", ".join(f"{k}->row {v}" for k, v in pools.items())
                         + "); their raw scores are not comparable")
    pool = next(iter(pools.values()))

    # ---- y shorter than the pools: locate and trim --------------------- #
    Tp = pit_data.shape[1]
    off = 0
    if Tp != T:
        off = _find_offset(x, y, pit_data, cdf_data)
        cut = slice(off, off + T)
        pit_data, cdf_data = pit_data[:, cut], cdf_data[:, cut]
        for nm, r in picked.items():
            for k in ('tau_grid', 'log_likelihood', 'v_transformed',
                      'v_full_composition', 'u_original'):
                a = r.get(k)
                if a is not None and np.asarray(a).shape[0] == Tp:
                    r[k] = np.asarray(a)[cut]
            for k in ('eval_start', 'warmup'):
                if r.get(k) is not None:
                    r[k] = max(0, int(r[k]) - off)
        notes.append(f"y has {T} periods and the pools {Tp}: y matched at "
                     f"offset {off}; pools and every report array were cut to "
                     f"pool[{off}:{off + T}].")

    # ---- common evaluation window -------------------------------------- #
    starts = {nm: int(_get(r, 'eval_start', 'warmup', default=0))
              for nm, r in picked.items()}
    es = int(eval_start) if eval_start is not None else max(starts.values())
    if eval_start is None and len(set(starts.values())) > 1:
        notes.append(f"methods report eval_start {starts}; all scored on the "
                     f"common window [{es},{T}) so the raw-pool score vectors "
                     f"align across the candidate set")
    sl = slice(es, T)
    n = T - es
    if n < 20:
        raise ValueError(f"only {n} scored periods; raise the sample or lower "
                         f"eval_start")

    # ---- alignment check ------------------------------------------------ #
    slab = np.asarray(cdf_data[pool], float)
    u_ref = pit_data[pool]
    u_grid = _pit_from_grid(x, slab, y)
    err = float(np.max(np.abs(u_grid[sl] - u_ref[sl])))
    ks_stored, ks_grid = _ks(u_ref[sl])[0], _ks(u_grid[sl])[0]
    if err > 5e-3:
        notes.append(
            f"WARNING: the PIT recomputed from cdf_data[{pool}] differs from "
            f"pit_data[{pool}] by up to {err:.4f} on the evaluation window "
            f"(KS {ks_grid:.4f} from the grid vs {ks_stored:.4f} stored).  "
            f"Fbar, the quantiles and the intervals are all read off "
            f"cdf_data, so every LEVEL below is about a slightly different "
            f"object than the PIT diagnostics.  The GAINS are unaffected "
            f"(log f(y) cancels).  Check that cdf_data and pit_data come from "
            f"the same run and are indexed by the same period.")

    if verbose:
        print(f"\n{'=' * 100}\nLOCALIZED EVALUATION   base pool = pit_data row "
              f"{pool}   discount factor {discount_factor:g}"
              f"\n  window [{es},{T})   n={n}   offset {off}   "
              f"max |PIT_grid - PIT_stored| = {err:.2e}\n{'=' * 100}")

    # ---- build the laws -------------------------------------------------- #
    laws = [PredictiveLaw.raw(x, slab, y, sl, pit_stored=u_ref)]
    for nm, r in picked.items():
        v = _get(r, 'v_transformed', 'v_full_composition', 'pit_after',
                 required=True)
        laws.append(PredictiveLaw.recalibrated(
            nm, x, slab, y, sl,
            _get(r, 'tau_grid', required=True),
            _get(r, 'u_grid', required=True),
            _get(r, 'log_likelihood', 'log_jacobian', required=True), v))
    names = [lw.name for lw in laws]
    ys = y[sl]

    gm = laws[0].mass
    if gm < 0.999:
        notes.append(f"the x grid holds only {gm:.4f} of the raw predictive "
                     f"mass; CRPS levels are truncated by that much")
    outside = float(np.mean((ys < x[0]) | (ys > x[-1])))
    if outside > 0:
        notes.append(f"{outside:.1%} of observations fall outside the x grid")

    # ---- regions --------------------------------------------------------- #
    if regions is None:
        regions = build_regions(y, es, min_events=min_events,
                                n_blocks=n_blocks,
                                threshold_window=threshold_window,
                                target=target, radii=radii, verbose=verbose)
    regions = [r.slice(sl) for r in regions]
    counts = {}
    for r in regions:
        counts[r.name] = int(sum(r.contains(t, np.array([ys[t]]))[0]
                                 for t in range(n)))

    # ---- score every (law, region, period) ------------------------------- #
    S = {}
    for lw in laws:
        for r in regions:
            acc = {}
            for t in range(n):
                c = _censored_scores(lw, r, t, ys[t], x, do_power, alpha)
                for k, v in c.items():
                    acc.setdefault(k, np.empty(n))[t] = v
            for k, v in acc.items():
                S[(lw.name, r.name, k)] = v

    # ---- unweighted, interval and tail-calibration blocks ---------------- #
    U, IV, TC = {}, {}, {}
    for lw in laws:
        U[lw.name] = {
            'LogS': lw.log_f_y,
            'CRPS': _crps_unweighted(lw, x, ys),
            'Pinball': _pinball(lw, x, ys),
        }
        IV[lw.name] = _interval_stats(lw, x, ys, interval_levels)
        for r in regions:
            if r.group == 'tail':
                TC[(lw.name, r.name)] = _tail_calibration(lw, x, ys, r)

    # =================== MASTER TABLE ==================================== #
    rows, order = {}, []          # rows[label] = dict(method -> level)
    orient = {}                   # +1 higher better, -1 lower better

    def put(label, fn, up):
        rows[label] = {lw.name: fn(lw) for lw in laws}
        orient[label] = up
        order.append(label)

    def sect(title):
        rows[title] = {lw.name: np.nan for lw in laws}
        orient[title] = 0
        order.append(title)

    # calibration ------------------------------------------------------- #
    sect('--- CALIBRATION ---')
    put('PIT KS statistic', lambda lw: _ks(lw.pit)[0], -1)
    put('PIT KS p-value', lambda lw: _ks(lw.pit)[1], +1)
    put('PIT mean  (target 0.5)', lambda lw: float(np.mean(lw.pit)), 0)
    put('PIT var   (target 0.0833)', lambda lw: float(np.var(lw.pit)), 0)

    # unweighted -------------------------------------------------------- #
    sect('--- UNWEIGHTED (whole outcome space) ---')
    put('LogS  (unweighted)', lambda lw: float(np.mean(U[lw.name]['LogS'])), +1)
    put('CRPS  (lower better)', lambda lw: float(np.mean(U[lw.name]['CRPS'])), -1)
    put('Pinball (lower better)',
        lambda lw: float(np.mean(U[lw.name]['Pinball'])), -1)
    put('MAE of median forecast',
        lambda lw: float(np.mean(np.abs(ys - IV[lw.name]['median']))), -1)

    # localized: tails --------------------------------------------------- #
    tails = [r for r in regions if r.group == 'tail']
    tl = [r for r in tails if r.kind == 'left']
    tr = [r for r in tails if r.kind == 'right']
    if tl or tr:
        sect('--- LOCALIZED: TAILS  (LSPS censored rules) ---')
    for lab, grp in (('left', tl), ('right', tr)):
        if not grp:
            continue
        r0 = grp[0]
        put(f'CSL  {lab} tail',
            lambda lw, r=r0: float(np.mean(S[(lw.name, r.name, 'CSL')])), +1)
        put(f'  CL   {lab}  (shape in A)',
            lambda lw, r=r0: float(np.mean(S[(lw.name, r.name, 'CL')])), +1)
        put(f'  Bern {lab}  (P(A) itself)',
            lambda lw, r=r0: float(np.mean(S[(lw.name, r.name, 'Bern')])), +1)
        put(f'CRPSflat {lab} tail',
            lambda lw, r=r0: float(np.mean(S[(lw.name, r.name, 'CRPSf')])), +1)
        if len(grp) > 1:
            r1 = grp[1]
            put(f'CSL  {lab} tail (wider cut)',
                lambda lw, r=r1: float(np.mean(S[(lw.name, r.name, 'CSL')])),
                +1)

    # localized: band ---------------------------------------------------- #
    band = [r for r in regions if r.group == 'band']
    if band:
        sect('--- LOCALIZED: TARGET BAND ---')
        for r in band:
            put(f'CSL  {r.name}',
                lambda lw, rr=r: float(np.mean(S[(lw.name, rr.name, 'CSL')])),
                +1)

    # balance ------------------------------------------------------------ #
    blocks = [r for r in regions if r.group == 'block']
    if blocks:
        sect('--- BALANCE across an equal-probability partition ---')
        for r in blocks:
            put(f'CSL  {r.name}  (n={counts[r.name]})',
                lambda lw, rr=r: float(np.mean(S[(lw.name, rr.name, 'CSL')])),
                +1)
        put('CSL sd across blocks (lower=more even)',
            lambda lw: float(np.std([np.mean(S[(lw.name, r.name, 'CSL')])
                                     for r in blocks])), -1)
        put('CSL worst block (higher better)',
            lambda lw: float(min(np.mean(S[(lw.name, r.name, 'CSL')])
                                 for r in blocks)), +1)

        # Does the gain go to the blocks the raw pool handled WORST?  That is
        # the direct test of "recalibration redistributes rather than raises".
        # Negative correlation = the worst blocks improve most = rebalancing.
        def _rebal(lw):
            if lw.name == RAW or len(blocks) < 3:
                return np.nan
            base = np.array([np.mean(S[(RAW, r.name, 'CSL')]) for r in blocks])
            gain = np.array([np.mean(S[(lw.name, r.name, 'CSL')])
                             for r in blocks]) - base
            if np.std(base) < 1e-12 or np.std(gain) < 1e-12:
                return np.nan
            return float(np.corrcoef(base, gain)[0, 1])
        put('gain-vs-level corr across blocks (neg = rebalancing)',
            _rebal, 0)

    # intervals ----------------------------------------------------------- #
    sect('--- PREDICTION INTERVALS ---')
    for a in interval_levels:
        put(f'coverage {a:.0%}  (target {a:.2f})',
            lambda lw, aa=a: IV[lw.name][aa]['cov'], 0)
    put('mean |coverage - nominal|',
        lambda lw: float(np.mean([abs(IV[lw.name][a]['cov'] - a)
                                  for a in interval_levels])), -1)
    for a in interval_levels:
        put(f'interval score {a:.0%} (lower better)',
            lambda lw, aa=a: float(np.mean(IV[lw.name][aa]['IS'])), -1)
    put('mean width 90%', lambda lw: IV[lw.name][0.90]['width']
        if 0.90 in IV[lw.name] else np.nan, 0)

    # tail calibration ----------------------------------------------------- #
    if TC:
        sect('--- TAIL CALIBRATION (Allen et al. 2025) ---')
        for lab, grp in (('left', tl), ('right', tr)):
            if not grp:
                continue
            r0 = grp[0]
            put(f'occurrence ratio {lab} (target 1)',
                lambda lw, r=r0: TC[(lw.name, r.name)].get('occurrence ratio',
                                                           np.nan), 0)
            put(f'severity KS {lab} (lower better)',
                lambda lw, r=r0: TC[(lw.name, r.name)].get('severity KS',
                                                           np.nan), -1)
            put(f'combined KS {lab} (lower better)',
                lambda lw, r=r0: TC[(lw.name, r.name)].get('combined KS',
                                                           np.nan), -1)

    # map diagnostic -------------------------------------------------------- #
    sect("--- WHERE THE MAP ACTS: mean log tau'(u) ---")
    put("overall", lambda lw: float(np.mean(lw.log_jac)), 0)
    for lab, grp in (('left tail', tl), ('right tail', tr)):
        if not grp:
            continue
        r0 = grp[0]
        w0 = S[(RAW, r0.name, 'w')] > 0

        def _cond(lw, mask=w0):
            return float(np.mean(lw.log_jac[mask])) if mask.any() else np.nan
        put(f"conditional on y in {lab}", _cond, 0)

    master_levels = pd.DataFrame(rows).T.loc[order, names].astype(float)
    master_levels.index.name = 'metric'

    # ---- the gain table (positive = better, always) ---------------------- #
    gains = {}
    for lab in order:
        if orient[lab] == 0:
            gains[lab] = {c: np.nan for c in names[1:]}
            continue
        row = {}
        for lw in laws[1:]:
            d = _series_for(lab, lw.name, RAW, S, U, IV, regions)
            if d is None:
                a, b = master_levels.loc[lab, lw.name], master_levels.loc[lab, RAW]
                v = (a - b) * orient[lab]
                row[lw.name] = f"{v:+.4f}" if np.isfinite(v) else ''
            else:
                mean, t = _hac(d * orient[lab], hac_lag)
                row[lw.name] = f"{mean:+.4f}{_stars(t)}"
        gains[lab] = row
    gain_df = pd.DataFrame(gains).T.loc[order, names[1:]]

    # display frame: two-level header, section rows blank, everything a string
    lev_txt = master_levels.map(
        lambda v: '' if not np.isfinite(v) else f"{v:.4f}")
    lev_txt.columns = pd.MultiIndex.from_product([['level'], lev_txt.columns])
    gain_df = gain_df.fillna('')
    gain_df.columns = pd.MultiIndex.from_product(
        [['delta vs raw pool  (+ = better)'], gain_df.columns])
    master_out = pd.concat([lev_txt, gain_df], axis=1)
    for lab in order:
        if orient[lab] == 0 and lab.startswith('---'):
            master_out.loc[lab, :] = ''
    master_out.index.name = 'metric'

    # =================== BY-REGION TABLE ================================= #
    br = []
    for r in regions:
        cnt = counts[r.name]
        ok = cnt >= max(5, min_events // 4)
        note = ('SMALL' if cnt < min_events else '')
        if r.meta.get('same_as'):
            note = (note + ' ' if note else '') + f"= {r.meta['same_as']}"
        base = {('', 'group'): r.group, ('', 'n in A'): cnt, ('', 'note'): note}
        for rule, key in (('CSL', 'CSL'), ('CL', 'CL'), ('Bern', 'Bern'),
                          ('CRPSflat', 'CRPSf')):
            if rule in ('CSL', 'CRPSflat'):     # CSL_raw = CL_raw + Bern_raw,
                base[(rule, 'raw')] = round(    # so two raw columns suffice
                    float(np.mean(S[(RAW, r.name, key)])), 4)
            for lw in laws[1:]:
                d = S[(lw.name, r.name, key)] - S[(RAW, r.name, key)]
                mean, t = _hac(d, hac_lag)
                base[(rule, lw.name)] = (f"{mean:+.4f}{_stars(t, ok)}"
                                         if cnt > 0 else '  --  ')
        w = S[(RAW, r.name, 'w')] > 0
        for lw in laws[1:]:
            base[("mean log tau'(u) | y in A", lw.name)] = (
                round(float(np.mean(lw.log_jac[w])), 4) if w.any() else np.nan)
        br.append(base)
    by_region = pd.DataFrame(br, index=[r.name for r in regions])
    by_region.columns = pd.MultiIndex.from_tuples(by_region.columns)
    by_region.index.name = 'region'

    # =================== DIAGNOSTICS ===================================== #
    diag = pd.DataFrame([{
        'pool row': pool, 'offset': off, 'window': f"[{es},{T})", 'n': n,
        'max |PIT_grid - PIT_stored|': err,
        'KS PIT stored': ks_stored, 'KS PIT grid': ks_grid,
        'grid mass': gm, 'obs outside grid': outside,
    }]).T.rename(columns={0: 'value'})

    res = LocalizedResult(float(discount_factor), (es, T), n, names,
                          {'master': master_out, 'by_region': by_region,
                           'diagnostics': diag,
                           'master_levels': master_levels, 'counts':
                               pd.Series(counts, name='n in A').to_frame()},
                          notes)
    if verbose:
        res.show()
    return res


def _series_for(label, name, base, S, U, IV, regions):
    """Per-period difference vector for the labels where a HAC t-stat is
    meaningful.  Returns None for summary statistics (KS, coverage, sd across
    blocks), which get a plain difference instead."""
    reg = {r.name: r for r in regions}

    def find(pred):
        for r in regions:
            if pred(r):
                return r
        return None

    if label == 'LogS  (unweighted)':
        return U[name]['LogS'] - U[base]['LogS']
    if label == 'CRPS  (lower better)':
        return U[name]['CRPS'] - U[base]['CRPS']
    if label == 'Pinball (lower better)':
        return U[name]['Pinball'] - U[base]['Pinball']
    if label.startswith('interval score'):
        a = float(label.split()[2].rstrip('%')) / 100
        return IV[name][a]['IS'] - IV[base][a]['IS']
    for pre, key in (('CSL  ', 'CSL'), ('  CL   ', 'CL'), ('  Bern ', 'Bern'),
                     ('CRPSflat ', 'CRPSf')):
        if label.startswith(pre):
            tag = label[len(pre):].strip()
            wider = 'wider' in tag
            if tag.startswith('block'):
                nm = tag.split('  (')[0].strip()
                r = reg.get(nm)
            elif tag.startswith('left') or tag.startswith('right'):
                side = 'left' if tag.startswith('left') else 'right'
                kind = 'left' if side == 'left' else 'right'
                cands = [r for r in regions
                         if r.group == 'tail' and r.kind == kind]
                r = (cands[1] if wider and len(cands) > 1 else
                     cands[0] if cands else None)
            else:
                r = reg.get(tag) or find(lambda q: q.name == tag)
            if r is not None:
                return S[(name, r.name, key)] - S[(base, r.name, key)]
    return None


# =========================================================================== #
#  8.  CONVENIENCE: the same master table at several discount factors
# =========================================================================== #
def compare_discount_factors(x, y, pit_data, cdf_data, methods,
                             discount_factors, metric='CSL  left tail',
                             **kw):
    """Run `evaluate_localized` at each discount factor and stack ONE metric,
    so the choice of delta can be inspected without printing four full sets of
    tables.  Everything else is identical to `evaluate_localized`."""
    kw.setdefault('verbose', False)
    out = {}
    for d in discount_factors:
        r = evaluate_localized(x, y, pit_data, cdf_data, methods, d, **kw)
        lv = r.frames['master_levels']
        if metric in lv.index:
            out[d] = lv.loc[metric]
    df = pd.DataFrame(out).T
    df.index.name = 'discount factor'
    return df