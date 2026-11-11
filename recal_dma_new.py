import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from scipy.special import logsumexp

from rational_quadratic import rational_quadratic_spline
from metrics import compute_crps_at_time, compute_ece   # noqa: F401

_EPS = 1e-6
_ECE_BINS = 10 

# ---- shared quadrature (used by BOTH methods for E[Y] and CRPS) ----------- #
_GL_N = 256
_gl_x, _gl_w = np.polynomial.legendre.leggauss(_GL_N)
_UN = 0.5 * (_gl_x + 1.0)
_UW = 0.5 * _gl_w
# central interval reported alongside the mean (90% by default)
_IVL_PROBS = (0.05, 0.95)

# ---- convolution internals: deliberately not exposed --------------------- #
_GAS_EXPERTS = [                       # (name, a_slow, a_fast, b, gamma)
    ('identity', 0.000, 0.000, 0.00, 0.93),
    ('glacial',  0.005, 0.000, 0.90, 0.93),
    ('slow',     0.030, 0.000, 0.90, 0.93),
    ('medium',   0.050, 0.150, 0.85, 0.93),
    ('rapid',    0.080, 0.400, 0.60, 0.93),
]
_GAS_ALPHA = 0.98                      # DMA discount factor: memory ~50 obs
_GAS_FLOOR = 1e-3                      # keeps every filter recoverable
_MIN_TRAIN = 30
_HEAD_SCALE = 0.01
_WEIGHT_DECAY = 1e-3
_LAMBDA_ID = 1e-2
_PATIENCE = 40
_VAL_FRAC = 0.25
_PROBS = np.array([0.05, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 0.95])
FEATURE_NAMES = ['median_loc', 'log_iqr', 'log_spread90', 'bowley_skew',
                 'moors_kurt', 'left_tail', 'right_tail', 'tail_asym']


def _softplus(z): return np.logaddexp(0.0, z)
def _sigmoid(z):  return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# =========================================================================== #
#  1.  SHARED PREPROCESSING AND METRICS
#      Everything in this block is called identically by both methods.
# =========================================================================== #
def _prep_pit(pit):
    """The single entry point for PITs. Both methods must use this or
    ks_before will differ between them on identical data."""
    return np.clip(np.asarray(pit, float), _EPS, 1 - _EPS)


def _pit_moments(cdf_row, x, y_t, tau, dtau):
    """E[Y] and CRPS of the recalibrated law, in PIT space.

        E    = int Q(u) tau'(u) du
        CRPS = 2 int ( 1{y < Q(u)} - tau(u) ) ( Q(u) - y ) tau'(u) du

    with Q the RAW quantile function. tau = identity, tau' = 1 recovers the raw
    mean and raw CRPS from the same quadrature, so the recalibrated value and
    its baseline are commensurable by construction. Never evaluates the
    recalibrated CDF at a grid edge.
    """
    F = np.maximum.accumulate(np.clip(np.asarray(cdf_row, float), 0.0, 1.0))
    F = F + np.linspace(0.0, 1e-9, len(F))
    Q = np.interp(_UN, F, np.asarray(x, float))
    w = dtau * _UW
    w = w / max(w.sum(), 1e-12)
    mean = float(np.sum(Q * w))
    crps = float(2.0 * np.sum((np.asarray(Q > y_t, float) - tau) * (Q - y_t) * w))
    return mean, crps


def _pit_quantiles(cdf_row, x, tau, probs=(0.05, 0.95)):
    """Quantiles of the RECALIBRATED law G = tau o F, on the raw y scale.

        G(y) = p  <=>  tau(F(y)) = p  <=>  F(y) = tau^{-1}(p)
        =>  y_p = Q( tau^{-1}(p) ),   Q = raw quantile function.

    tau is the map evaluated on the _UN grid, so tau^{-1} is one monotone
    interpolation. tau = identity recovers the raw quantiles from the same
    quadrature, so calibrated and raw interval widths are commensurable.
    """
    F = np.maximum.accumulate(np.clip(np.asarray(cdf_row, float), 0.0, 1.0))
    F = F + np.linspace(0.0, 1e-9, len(F))
    Q = np.interp(_UN, F, np.asarray(x, float))          # raw quantile on _UN
    tau = np.maximum.accumulate(np.asarray(tau, float))  # guard monotonicity
    u_star = np.interp(np.asarray(probs, float), tau, _UN)
    return np.interp(u_star, _UN, Q)


def dm_test(d, hac_lag=0):
    """Giacomini-White / Diebold-Mariano on a per-observation score
    differential. The benchmark is the raw forecast at exactly zero, so this is
    applied to {ell_t} directly. Two-sided: read the SIGN of t, not just p."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return (float('nan'),) * 4
    dbar = d.mean()
    e = d - dbar
    var = float(e @ e) / n
    for L in range(1, int(hac_lag) + 1):
        var += 2.0 * (1.0 - L / (hac_lag + 1.0)) * float(e[L:] @ e[:-L]) / n
    se = np.sqrt(max(var, 1e-24) / n)
    return float(dbar), float(se), float(dbar / se), \
        float(2 * (1 - stats.norm.cdf(abs(dbar / se))))


def oracle_ceiling(u):
    """Largest per-obs log-score gain ANY map could achieve on this sample:
    the in-sample KL of a fitted Beta PIT density from uniform. In-sample, so a
    generous upper bound -- and very loose when the PITs pile at an endpoint.
    Report the realised gain as a fraction of it."""
    s = np.clip(np.asarray(u, float), 1e-6, 1 - 1e-6)
    try:
        a, b, _, _ = stats.beta.fit(s, floc=0, fscale=1)
    except Exception:
        return float('nan')
    g = np.linspace(1e-6, 1 - 1e-6, 5001)
    d = stats.beta.pdf(g, a, b)
    return float(np.trapezoid(d * np.log(np.maximum(d, 1e-300)), g))


def per_third_var(u):
    """Window-free view of whether dispersion miscalibration is stationary
    WITHIN the evaluation sample. Uniform target is 1/12 = 0.0833."""
    u = np.asarray(u, float)
    k = len(u) // 3
    return tuple(float(np.var(s)) for s in (u[:k], u[k:2 * k], u[2 * k:])) \
        if len(u) >= 6 else (float('nan'),) * 3


def _report(u, v, ll, y, ymean, ymean_raw, crps, crps_raw, sl, ece_bins):
    """The ONLY place scalar metrics are computed, for either method.

    Given identical (u, y, cdf) and an identical slice, every '*_before' and
    '*_raw' entry below is bit-for-bit identical across methods -- that is the
    point of this function existing.
    """
    uu, vv = u[sl], v[sl]
    n = len(uu)
    db, se, ts, pv = dm_test(ll[sl],hac_lag=int(np.floor(4 * (n / 100) ** (2 / 9)))) #### new change from db, se, ts, pv = dm_test(ll[sl])
    ceil = oracle_ceiling(uu)
    return {
        'n': int(n),
        # --- log score (benchmark = raw forecast = exactly 0) ---
        'logscore_total': float(ll[sl].sum()),
        'logscore_mean': float(ll[sl].mean()),
        'logscore_se': float(ll[sl].std(ddof=1) / np.sqrt(max(n, 1))),
        'dm_vs_raw': {'mean_gain': db, 'se': se, 't': ts, 'p': pv},
        'oracle_ceiling': ceil,
        'oracle_captured_pct': (db / ceil * 100.0) if (ceil and ceil > 1e-6) else float('nan'),
        # --- calibration ---
        'ks_before': float(stats.kstest(uu, 'uniform').statistic),
        'ks_after': float(stats.kstest(vv, 'uniform').statistic),
        'ks_null_mean': float(0.8687 / np.sqrt(n)),
        'ece_before': compute_ece(uu, n_bins=ece_bins),
        'ece_after': compute_ece(vv, n_bins=ece_bins),
        'pit_var_thirds_before': per_third_var(uu),
        'pit_var_thirds_after': per_third_var(vv),
        # --- sharpness / point forecast, same quadrature both sides ---
        'crps_raw_mean': float(crps_raw[sl].mean()),
        'crps_mean': float(crps[sl].mean()),
        'mean_bias_raw': float(np.mean(ymean_raw[sl] - y[sl])),
        'mean_bias': float(np.mean(ymean[sl] - y[sl])),
        # --- arrays ---
        'raw_pits': uu.astype(np.float32),
        'calibrated_pits': vv.astype(np.float32),
        'logscore_per_obs': ll[sl],
    }


def _print_report(label, o, extra_lines=()):
    """Identical print layout for both methods, so the two console blocks can
    be read side by side."""
    d = o['dm_vs_raw']
    pb, pa = o['pit_var_thirds_before'], o['pit_var_thirds_after']
    print(f"  {label}  n={o['n']}")
    print(f"     log-score total {o['logscore_total']:+9.3f}   per obs "
          f"{o['logscore_mean']:+.4f} (SE {o['logscore_se']:.4f})   "
          f"t={d['t']:+.2f}  p={d['p']:.3f}")
    print(f"     oracle ceiling {o['oracle_ceiling']:+.4f}/obs  -> captured "
          f"{o['oracle_captured_pct']:.0f}%")
    for ln in extra_lines:
        print(f"     {ln}")
    print(f"     KS   {o['ks_before']:.4f} -> {o['ks_after']:.4f}   "
          f"(null mean {o['ks_null_mean']:.4f})")
    print(f"     ECE  {o['ece_before']:.4f} -> {o['ece_after']:.4f}   "
          f"CRPS {o['crps_raw_mean']:.4f} -> {o['crps_mean']:.4f}")
    print(f"     mean bias {o['mean_bias_raw']:+.4f} -> {o['mean_bias']:+.4f}")
    print(f"     PIT var by third  {pb[0]:.4f} {pb[1]:.4f} {pb[2]:.4f}  ->  "
          f"{pa[0]:.4f} {pa[1]:.4f} {pa[2]:.4f}   (target 0.0833)")


# =========================================================================== #
#  2.  SPLINE HELPERS
# =========================================================================== #
def _spline_on_grid(theta_row, K):
    """tau(u) and tau'(u) at the shared quadrature nodes, for one theta."""
    th = torch.tensor(np.asarray(theta_row, np.float32)[None, :]).expand(_GL_N, -1)
    with torch.no_grad():
        v, ld, _ = rational_quadratic_spline(
            inputs=torch.tensor(_UN, dtype=torch.float32),
            unnormalized_widths=th[:, :K], unnormalized_heights=th[:, K:2 * K],
            unnormalized_derivatives=th[:, 2 * K:],
            inverse=False, enable_identity_init=True)
    return v.numpy().astype(float), np.exp(ld.numpy().astype(float))


# =========================================================================== #
#  3.  SHARED FILTER BANK
#      Used by BOTH methods: it IS method 1, and it is the outer map of
#      method 2. Nothing here is estimated -- theta_t depends only on the
#      input sequence before t, so every path is one-step-ahead.
# =========================================================================== #
def _logdet_grad_bank(val, PW, PH, PD):
    pw = torch.tensor(PW, dtype=torch.float32, requires_grad=True)
    ph = torch.tensor(PH, dtype=torch.float32, requires_grad=True)
    pd = torch.tensor(PD, dtype=torch.float32, requires_grad=True)
    inp = torch.full((PW.shape[0],), float(np.clip(val, _EPS, 1 - _EPS)),
                     dtype=torch.float32)
    _, ld, _ = rational_quadratic_spline(inp, pw, ph, pd, inverse=False,
                                         enable_identity_init=True)
    ld.sum().backward()

    def g(p, sh):
        return np.zeros(sh) if p.grad is None else p.grad.numpy().astype(np.float64)
    return (ld.detach().numpy().astype(np.float64),
            g(pw, PW.shape), g(ph, PH.shape), g(pd, PD.shape))


_STEP_NORMS = ('coord', 'global', 'global_sqrtP')


class GasBank:
    """E fixed two-timescale filters advanced one observation at a time.

    Nothing is estimated: the theta returned by step(val) is computed BEFORE
    val is used, so every path is one-step-ahead. The stepwise form lets the
    composition run its outer bank online, on the inner map's deployed output.

    step_norm     how the score is scaled before the update (eq. 18):
                  'coord'        per-coordinate RMS (RMSprop-style). Each
                                 coordinate moves at the nominal kappa
                                 whatever the dimension, so a K-bin spline
                                 (3K+1 params) is not slowed relative to a
                                 2-param map. Default.
                  'global'       RMS of the whole gradient norm -- the original
                                 rule; reproduces earlier results exactly.
                  'global_sqrtP' the original rule times sqrt(P), so the
                                 expected per-coordinate step is roughly
                                 dimension-free while directions are kept.
    slow_shrink   rho_s in S_{t+1} = rho_s S_t + kappa_s r_t g_t. 1.0 is the
                  original random walk; values just below 1 (e.g. 0.999) add
                  weak mean reversion towards the identity so the slow
                  component cannot drift indefinitely when nothing is wrong.
    """

    def __init__(self, K: int, warmup: int, step_norm: str = 'coord',
                 slow_shrink: float = 1.0) -> None:
        if step_norm not in _STEP_NORMS:
            raise ValueError(f"step_norm must be one of {_STEP_NORMS}")
        if not 0.0 < slow_shrink <= 1.0:
            raise ValueError("slow_shrink must lie in (0, 1]")
        self.K, self.warmup = int(K), int(warmup)
        self.step_norm, self.slow_shrink = step_norm, float(slow_shrink)
        self.a_s = np.array([e[1] for e in _GAS_EXPERTS])[:, None]
        self.a_f = np.array([e[2] for e in _GAS_EXPERTS])[:, None]
        self.bb = np.array([e[3] for e in _GAS_EXPERTS])[:, None]
        self.gg = np.array([e[4] for e in _GAS_EXPERTS])[:, None]
        self.E, self.P = len(_GAS_EXPERTS), 3 * self.K + 1
        self.S = np.zeros((self.E, self.P))
        self.Fq = np.zeros((self.E, self.P))
        gn_cols = self.P if step_norm == 'coord' else 1
        # per-coordinate start keeps the first step the same size as before
        self.gn = np.full((self.E, gn_cols), 1e-4 / (self.P if step_norm == 'coord' else 1))
        self.t = 0

    def step(self, val: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (log tau'_e(val) for each filter, theta used), then update."""
        K = self.K
        th = np.zeros((self.E, self.P)) if self.t < self.warmup else (self.S + self.Fq)
        l, gw, gh, gd = _logdet_grad_bank(val, th[:, :K], th[:, K:2 * K], th[:, 2 * K:])
        g = np.concatenate([gw, gh, gd], axis=1)
        bad = ~(np.isfinite(l) & np.isfinite(g).all(axis=1))
        if bad.any():
            l = np.where(bad, -50.0, l)
            g = np.where(bad[:, None], 0.0, g)
        if self.step_norm == 'coord':
            self.gn = self.gg * self.gn + (1 - self.gg) * g * g
            r = 1.0 / (np.sqrt(self.gn) + 1e-6)
        else:
            self.gn = self.gg * self.gn + (1 - self.gg) * (g * g).sum(1, keepdims=True)
            r = 1.0 / (np.sqrt(self.gn) + 1e-6)
            if self.step_norm == 'global_sqrtP':
                r = r * np.sqrt(self.P)
        self.S = self.slow_shrink * self.S + self.a_s * r * g
        self.Fq = self.bb * self.Fq + self.a_f * r * g
        self.t += 1
        return l, th


def run_gas_bank(seq, K, warmup, step_norm='coord', slow_shrink=1.0):
    """E fixed two-timescale filters in lockstep over a whole sequence.
    step_norm='global', slow_shrink=1 reproduces the original filter."""
    bank = GasBank(K, warmup, step_norm, slow_shrink)
    T = len(seq)
    theta = np.zeros((bank.E, T, bank.P)); ll = np.zeros((bank.E, T))
    for t in range(T):
        ll[:, t], theta[:, t, :] = bank.step(seq[t])
    return ll, theta


def dma_weights(loglik, alpha, start, floor=0.0):
    
    loglik = np.asarray(loglik, float)
    E, T = loglik.shape
    a = float(np.clip(alpha, 1e-6, 1.0))
    fl = float(floor)
    if not 0.0 <= fl < 1.0 / E:
        raise ValueError(f"floor must lie in [0, 1/E) = [0, {1.0/E:.3f})")
    logw = np.full(E, -np.log(E))            # pi_{1|0}: uniform
    W = np.zeros((E, T))
    for t in range(T):
        W[:, t] = np.exp(logw)               # pi_{t|t-1}, F_{t-1}-measurable
        if t < start:
            continue
        # UPDATE with this period's predictive likelihood
        lp = logw + np.clip(loglik[:, t], -50, 50)
        logw = lp - logsumexp(lp)
        # FORGET: temper toward uniform by the discount factor
        lp = a * logw
        logw = lp - logsumexp(lp)
        if fl > 0:                           # keep every filter recoverable
            w = (1.0 - fl * E) * np.exp(logw) + fl
            logw = np.log(w)
    return W


def _bank_eval(theta_col, uu, K, weights):
    """Mixture map sum_k w_k tau_k and its derivative. A convex mixture of
    monotone maps is a monotone map and differentiation is linear, so the
    mixture's log-score is a valid recalibrator's log-score."""
    E, P = theta_col.shape
    n = len(uu)
    th = torch.tensor(np.repeat(theta_col[:, None, :], n, 1).reshape(E * n, P),
                      dtype=torch.float32)
    inp = torch.tensor(np.tile(np.clip(uu, _EPS, 1 - _EPS), E), dtype=torch.float32)
    with torch.no_grad():
        vk, lk, _ = rational_quadratic_spline(
            inputs=inp, unnormalized_widths=th[:, :K],
            unnormalized_heights=th[:, K:2 * K],
            unnormalized_derivatives=th[:, 2 * K:],
            inverse=False, enable_identity_init=True)
    w = np.asarray(weights)[:, None]
    return ((w * vk.numpy().reshape(E, n)).sum(0),
            (w * np.exp(lk.numpy().reshape(E, n))).sum(0))


def _spline_rows(theta_rows: np.ndarray, uu: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """tau and log tau' for one theta per input (row t of theta_rows at uu[t])."""
    th = torch.tensor(np.asarray(theta_rows, np.float32))
    with torch.no_grad():
        v, ld, _ = rational_quadratic_spline(
            inputs=torch.tensor(np.clip(uu, _EPS, 1 - _EPS), dtype=torch.float32),
            unnormalized_widths=th[:, :K], unnormalized_heights=th[:, K:2 * K],
            unnormalized_derivatives=th[:, 2 * K:],
            inverse=False, enable_identity_init=True)
    return v.numpy().astype(float), ld.numpy().astype(float)


# =========================================================================== #
#  3b. BASELINES AND SHARPNESS (shared by both methods)
# =========================================================================== #
def run_static_expanding(u: np.ndarray, K: int, warmup: int,
                         refit_every: int = 25, min_train: int = _MIN_TRAIN,
                         n_steps: int = 150, lr: float = 0.05,
                         ridge: float = _LAMBDA_ID, window: int | None = None,
                         solver: str = 'adam', max_iter: int = 500, tol: float = 1e-7
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Static recalibration benchmark: eq. (9) on an expanding window
    (window=None) or a rolling window of the last `window` PITs.

    At each refit date t, theta maximises mean_{s<t} log tau'_theta(u_s) minus
    a ridge towards the identity, and is then held fixed until the next refit.
    Warm-started from the previous fit. Strictly one-step-ahead: theta_t uses
    u[:t] only. If the time-varying methods cannot beat this, their gain comes
    from recalibration as such, not from time variation.

    solver='adam' (default; reproduces all earlier results): n_steps Adam steps
        at learning rate lr per refit. This is penalised maximum likelihood with
        early stopping: the fixed step budget acts as extra regularisation.
    solver='lbfgs': L-BFGS with a strong-Wolfe line search in double precision,
        up to max_iter iterations or tolerance tol. Note that for PIT densities
        with unbounded edges (e.g. an under-dispersed pool) the penalised
        likelihood keeps rising as the spline's edge slopes grow, so at a weak
        ridge there is no well-defined optimum and "convergence" is a stall;
        use a stronger ridge with this solver.
    The ridge, not the optimiser, controls the bias-variance trade-off: a strong
    ridge helps on short, nearly calibrated series and hurts where the map must
    be flexible (multimodal or U-shaped PITs).
    """
    if solver not in ('adam', 'lbfgs'):
        raise ValueError("solver must be 'adam' or 'lbfgs'")
    u = _prep_pit(u)
    T, P = len(u), 3 * K + 1
    dtype = torch.float64 if solver == 'lbfgs' else torch.float32
    theta = torch.zeros(P, dtype=dtype, requires_grad=True)
    path = np.zeros((T, P))
    n_fail = 0
    for t in range(T):
        if t >= warmup + min_train and (t - warmup) % refit_every == 0:
            lo = 0 if window is None else max(0, t - int(window))
            ut = torch.tensor(u[lo:t], dtype=dtype)
            n = t - lo

            def objective():
                th = theta.unsqueeze(0).expand(n, -1)
                _, ld, _ = rational_quadratic_spline(
                    inputs=ut, unnormalized_widths=th[:, :K],
                    unnormalized_heights=th[:, K:2 * K],
                    unnormalized_derivatives=th[:, 2 * K:],
                    inverse=False, enable_identity_init=True)
                return -ld.mean() + ridge * (theta ** 2).mean()

            if solver == 'adam':
                opt = optim.Adam([theta], lr=lr)
                for _ in range(n_steps):
                    opt.zero_grad()
                    loss = objective()
                    loss.backward()
                    opt.step()
            else:
                prev = theta.detach().clone()
                opt = optim.LBFGS([theta], lr=1.0, max_iter=max_iter, history_size=20,
                                  tolerance_grad=tol, tolerance_change=tol * 1e-2,
                                  line_search_fn='strong_wolfe')

                def closure():
                    opt.zero_grad()
                    loss = objective()
                    loss.backward()
                    return loss

                try:
                    loss = opt.step(closure)
                    ok = bool(torch.isfinite(loss)) and bool(torch.isfinite(theta).all())
                except RuntimeError:
                    ok = False
                if not ok:
                    n_fail += 1
                    with torch.no_grad():
                        theta.copy_(prev)
        path[t] = theta.detach().numpy()
    if n_fail:
        import warnings
        warnings.warn(f"run_static_expanding: {n_fail} refit(s) failed and kept the previous parameters")
    v, ll = _spline_rows(path, u, K)
    return ll, path, np.clip(v, _EPS, 1 - _EPS)


def _interval_metrics(q: np.ndarray, y: np.ndarray, sl: slice,
                      probs: tuple = _IVL_PROBS) -> dict:
    """Mean central-interval width (sharpness) and interval score (Gneiting &
    Raftery 2007) over the evaluation slice. Lower is better for both; width
    alone is sharpness, the score rewards sharpness subject to coverage."""
    a = 1.0 - (probs[1] - probs[0])                 # e.g. 0.10 for a 90% interval
    lo, hi, yy = q[sl, 0], q[sl, 1], y[sl]
    width = hi - lo
    score = width + (2 / a) * (lo - yy) * (yy < lo) + (2 / a) * (yy - hi) * (yy > hi)
    return {'width': float(np.mean(width)), 'score': float(np.mean(score)),
            'coverage': float(np.mean((yy >= lo) & (yy <= hi))),
            'per_obs_score': score}


def _hac_lag(n: int) -> int:
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def _sharpness_block(q_cal, q_raw, y, sl) -> tuple[dict, list[str]]:
    """Sharpness/interval entries for the report, plus a DM test on the
    interval score (positive t = recalibrated interval better)."""
    cal, raw = _interval_metrics(q_cal, y, sl), _interval_metrics(q_raw, y, sl)
    d = raw['per_obs_score'] - cal['per_obs_score']
    db, se, ts, pv = dm_test(d, hac_lag=_hac_lag(len(d)))
    pct = int(round(100 * (_IVL_PROBS[1] - _IVL_PROBS[0])))
    out = {'ivl_width_raw': raw['width'], 'ivl_width_cal': cal['width'],
           'ivl_score_raw': raw['score'], 'ivl_score_cal': cal['score'],
           'ivl_coverage_raw': raw['coverage'], 'ivl_coverage_cal': cal['coverage'],
           'dm_ivl_score_vs_raw': {'mean_gain': db, 'se': se, 't': ts, 'p': pv}}
    line = (f"{pct}% interval  width {raw['width']:.4f} -> {cal['width']:.4f}   "
            f"coverage {raw['coverage']:.3f} -> {cal['coverage']:.3f}   "
            f"IS {raw['score']:.4f} -> {cal['score']:.4f} (t={ts:+.2f})")
    return out, [line]


# =========================================================================== #
#  4.  METHOD 1 --- two-timescale score-driven recalibration (aggregated)
# =========================================================================== #
def _effective_n(W):
    """exp(entropy) of a weight vector: 1 = fully concentrated, E = uniform.
    Read this before reading the weights themselves. Near-uniform weights are
    not a bug -- they are the aggregate saying the per-step score differences
    between adaptation rates are small relative to what sigma remembers, which
    is exactly the case on a series whose dispersion is stable."""
    W = np.clip(np.asarray(W, float), 1e-30, None)
    return np.exp(-(W * np.log(W)).sum(axis=0))


def tvgas_pit_dma_new(K, x, y, pit_data, cdf_data, discount_factors,
              warmup=15, eval_start=None, alpha=_GAS_ALPHA,
              floor=_GAS_FLOOR, step_norm='coord', slow_shrink=1.0,
              static_baseline=True, static_refit_every=25, verbose=True):
    """v_t = tau_GAS(u_t): DMA mixture of fixed-rate score-driven filters.

    step_norm, slow_shrink   see GasBank. step_norm='global', slow_shrink=1
                             reproduces the original filter exactly.
    static_baseline          also fit the static expanding-window map of
                             eq. (9) (run_static_expanding) and report a
                             HAC DM test of GAS against it ('dm_vs_static').
                             This is the test of whether TIME VARIATION
                             helps, as opposed to recalibration as such.
    Sharpness is reported directly: mean width, coverage and interval score
    of the central interval _IVL_PROBS, raw vs recalibrated.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    names = [e[0] for e in _GAS_EXPERTS]
    E = len(_GAS_EXPERTS)
    is_fast = np.array([e[2] > 0 for e in _GAS_EXPERTS])
    results = []

    for i, delta in enumerate(discount_factors):
        if verbose:
            print(f"\n{'=' * 72}\nTwo-timescale GAS (aggregated)  v = tau_GAS(u)  |  base: {delta}\n{'=' * 72}")
        u = _prep_pit(pit_data[i])                       # SHARED preprocessing
        cdf_seq = np.asarray(cdf_data[i], float)
        T = len(u)
        es = warmup if eval_start is None else int(eval_start)
        if not (warmup <= es < T):
            raise ValueError(f"eval_start={es} outside [warmup={warmup}, T={T})")

        # ---- the bank, on the raw PITs. Nothing estimated. --------------- #
        ll_bank, theta_bank = run_gas_bank(u, K, warmup, step_norm, slow_shrink)
        W = dma_weights(ll_bank, alpha, warmup, floor)
        ll = logsumexp(np.log(W + 1e-30) + ll_bank, axis=0)

        # ---- apply the mixture map --------------------------------------- #
        v = np.zeros(T)
        for t in range(T):
            v[t] = _bank_eval(theta_bank[:, t, :], np.array([u[t]]), K, W[:, t])[0][0]
        v = np.clip(v, _EPS, 1 - _EPS)

        # ---- moments, SHARED quadrature ---------------------------------- #
        crps = np.zeros(T); crps_raw = np.zeros(T)
        ymean = np.zeros(T); ymean_raw = np.zeros(T)
        tau_grid = np.zeros((T, _GL_N))   # the map itself, for localized eval
        q_cal = np.full((T, 2), np.nan); q_raw = np.full((T, 2), np.nan)
        ones = np.ones_like(_UN)
        for t in range(T):
            tau, dtau = _bank_eval(theta_bank[:, t, :], _UN, K, W[:, t])
            tau_grid[t] = tau
            ymean[t], crps[t] = _pit_moments(cdf_seq[t], x, y[t], tau, dtau)
            ymean_raw[t], crps_raw[t] = _pit_moments(cdf_seq[t], x, y[t], _UN, ones)
            q_cal[t] = _pit_quantiles(cdf_seq[t], x, tau, _IVL_PROBS)
            q_raw[t] = _pit_quantiles(cdf_seq[t], x, _UN, _IVL_PROBS)

        # ---- report, SHARED ---------------------------------------------- #
        sl = slice(es, T)
        out = _report(u, v, ll, y, ymean, ymean_raw, crps, crps_raw, sl, _ECE_BINS)

        # aggregation diagnostics: the analogue of the old dynamic-gain test
        ex_mean = ll_bank[:, sl].mean(axis=1)
        w_mean = W[:, sl].mean(axis=1)
        kbest = int(np.argmax(ll_bank[:, sl].sum(axis=1)))
        out['expert_names'] = names
        out['expert_logscore_mean'] = {n: float(m) for n, m in zip(names, ex_mean)}
        out['expert_weight_mean'] = {n: float(w) for n, w in zip(names, w_mean)}
        out['best_expert'] = names[kbest]
        out['best_expert_logscore_mean'] = float(ex_mean[kbest])
        out['mixture_minus_best_expert'] = float(out['logscore_mean'] - ex_mean[kbest])
        out['aggregation_floor'] = float(-np.log(E) / out['n'])   # holds at alpha=1
        out['eff_n_experts'] = float(_effective_n(W)[sl].mean())
        out['weight_on_fast'] = float(w_mean[is_fast].sum())
        out['alpha'] = float(alpha)
        out['floor'] = float(floor)
        out['dma_effective_memory'] = (float('inf') if alpha >= 1.0
                                       else 1.0 / (1.0 - alpha))
        out['step_norm'] = step_norm
        out['slow_shrink'] = float(slow_shrink)

        # ---- sharpness, measured not assumed ----------------------------- #
        sharp, sharp_lines = _sharpness_block(q_cal, q_raw, y, sl)
        out.update(sharp)

        # ---- static benchmark: does time variation help? ------------------ #
        static_lines = []
        ll_static = v_static = theta_static = None
        if static_baseline:
            ll_static, theta_static, v_static = run_static_expanding(
                u, K, warmup, refit_every=static_refit_every)
            d = ll[sl] - ll_static[sl]
            db, se, ts, pv = dm_test(d, hac_lag=_hac_lag(len(d)))
            out['logscore_static_mean'] = float(ll_static[sl].mean())
            out['ks_after_static'] = float(stats.kstest(v_static[sl], 'uniform').statistic)
            out['dm_vs_static'] = {'mean_gain': db, 'se': se, 't': ts, 'p': pv}
            static_lines = [f"static expanding map {out['logscore_static_mean']:+.4f}/obs "
                            f"(KS {out['ks_after_static']:.4f});  GAS - static "
                            f"{db:+.4f}  t={ts:+.2f}  p={pv:.3f}"]

        if verbose:
            top = np.argsort(-w_mean)[:3]
            print("  weights: " + "  ".join(f"{names[k]}={w_mean[k]:.2f}" for k in top)
                  + f"   (eff. {out['eff_n_experts']:.2f} of {E};"
                    f" mass on fast filters {out['weight_on_fast']:.2f};"
                    f" alpha={alpha:g}, memory~"
                    + ("inf" if alpha >= 1 else f"{1/(1-alpha):.0f}") + ")")
            _print_report(
                f"EVALUATION [{es},{T})", out,
                extra_lines=[
                    f"best single filter '{out['best_expert']}' "
                    f"{out['best_expert_logscore_mean']:+.4f}/obs; mixture "
                    f"{out['mixture_minus_best_expert']:+.4f} vs it "
                    f"(no-harm floor, valid at alpha=1: {out['aggregation_floor']:+.4f})",
                    *static_lines, *sharp_lines])

        res = {'delta': delta, 'head': 'tvgas_aggregated', 'K': K,
               'warmup': warmup, 'eval_start': es, 'oos_start': es,
               'u_original': u.astype(np.float32),
               'v_transformed': v,
               'log_likelihood': ll,
               'theta_all': np.einsum('kt,ktp->tp', W, theta_bank),
               'theta_experts': theta_bank, 'expert_logscore': ll_bank,
               'gas_expert_names': names, 'gas_weights': W,
               'crps_history': crps, 'crps_raw_history': crps_raw,
               'y_calibrated_mean': ymean, 'y_raw_mean': ymean_raw,
               'y_calibrated_q05': q_cal[:, 0], 'y_calibrated_q95': q_cal[:, 1],
               'y_raw_q05': q_raw[:, 0], 'y_raw_q95': q_raw[:, 1],
               'interval_probs': np.asarray(_IVL_PROBS, float),
               'tau_grid': tau_grid, 'u_grid': _UN.copy(),
               'll_static': ll_static, 'v_static': v_static,
               'theta_static': theta_static}
        res.update(out)
        results.append(res)

    return results


# =========================================================================== #
#  5.  METHOD 2 --- the composition recalibrator
# =========================================================================== #
def build_cdf_features(cdf_seq, x):
    """Eight scale-free shape descriptors of the predictive CDF.

    NOTE the quantile grid _PROBS and the unpacked names must agree. With the
    default grid these ARE the textbook quantities (quartiles for the IQR and
    Bowley skewness, octiles for Moors kurtosis). If you change _PROBS to
    deciles the descriptors remain valid shape statistics but the names stop
    meaning what they say -- rename them before they go in a paper.
    """
    x = np.asarray(x, float)
    lo, rng = x[0], (x[-1] - x[0]) + _EPS
    T = len(cdf_seq)
    out = np.zeros((T, len(FEATURE_NAMES)))
    for t in range(T):
        F = np.maximum.accumulate(np.clip(np.asarray(cdf_seq[t], float), 0, 1))
        F = F + np.linspace(0, 1e-9, len(F))
        q05, q125, q25, q375, q50, q625, q75, q875, q95 = np.interp(_PROBS, F, x)
        iqr = (q75 - q25) + _EPS
        out[t] = [(q50 - lo) / rng, np.log(iqr), np.log((q95 - q05) + _EPS),
                  (q75 + q25 - 2 * q50) / iqr,
                  ((q875 - q625) + (q375 - q125)) / iqr,
                  (q50 - q05) / iqr, (q95 - q50) / iqr,
                  ((q95 - q50) - (q50 - q05)) / ((q95 - q50) + (q50 - q05) + _EPS)]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def causal_rank_norm(X, min_history=20, clip=3.5):
    """z_t = Phi^{-1}(rank of X_t among its OWN past). Strictly causal, and
    invariant to monotone drift in a feature's level, so a trending spread
    cannot masquerade as a novel covariate value late in the sample."""
    X = np.asarray(X, float)
    T, J = X.shape
    Z = np.zeros((T, J))
    for t in range(min_history, T):
        r = ((X[:t] < X[t][None, :]).sum(0)
             + 0.5 * (X[:t] == X[t][None, :]).sum(0)) / t
        Z[t] = np.clip(stats.norm.ppf(np.clip((r * t + 0.5) / (t + 1), 1e-6, 1 - 1e-6)),
                       -clip, clip)
    return np.nan_to_num(Z, nan=0.0).astype(np.float32)


def _build_mlp(in_dim, hidden, out_dim, seed):
    """hidden=0 gives a linear head (the linear-in-features ablation)."""
    torch.manual_seed(seed)
    if hidden == 0:
        net = nn.Sequential(nn.Linear(in_dim, out_dim)).float()
    else:
        net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                            nn.Linear(hidden, out_dim)).float()
    with torch.no_grad():
        net[-1].weight.mul_(_HEAD_SCALE)
        net[-1].bias.zero_()
    return net


def _train_head(net, X_t, u_t, theta_bank_t, logW_t, K, lo, hi, num_epochs, lr):
    """Fit the MLP on [lo, hi) only. The bank path is frozen, so nothing here
    sees data at or beyond hi. The ridge is on theta_NN, i.e. a Gaussian prior
    centred on the identity MAP, not merely on the weights.

    theta_bank_t / logW_t must be the OUTER bank as actually deployed on
    w_s = tau_NN(u_s) for s < hi (not the bank filtered on raw u). Pass None
    for both to train the network-only ablation (objective = log tau_NN')."""
    n = hi - lo
    n_val = max(5, int(_VAL_FRAC * n))
    fit_hi = hi - n_val
    if fit_hi <= lo + 2:
        return net
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=_WEIGHT_DECAY)
    best, best_state, since = float('inf'), None, 0
    net.train()
    for _ in range(num_epochs):
        opt.zero_grad()
        th = net(X_t[lo:hi])
        w, ll_nn, _ = rational_quadratic_spline(
            inputs=u_t[lo:hi], unnormalized_widths=th[:, :K],
            unnormalized_heights=th[:, K:2 * K],
            unnormalized_derivatives=th[:, 2 * K:],
            inverse=False, enable_identity_init=True)
        w = torch.clamp(w, _EPS, 1 - _EPS)
        if theta_bank_t is None:
            obj = ll_nn
        else:
            E = theta_bank_t.shape[0]
            m = hi - lo
            tb = theta_bank_t[:, lo:hi, :].reshape(E * m, -1)
            _, lk, _ = rational_quadratic_spline(
                inputs=w.unsqueeze(0).expand(E, m).reshape(-1),
                unnormalized_widths=tb[:, :K], unnormalized_heights=tb[:, K:2 * K],
                unnormalized_derivatives=tb[:, 2 * K:],
                inverse=False, enable_identity_init=True)
            ll_gas = torch.logsumexp(logW_t[:, lo:hi] + lk.view(E, m), dim=0)
            obj = ll_nn + ll_gas
        k = fit_hi - lo
        loss = -obj[:k].mean() + _LAMBDA_ID * (th[:k] ** 2).mean()
        with torch.no_grad():
            val = -obj[k:].mean().item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if np.isfinite(val) and val < best:
            best, since = val, 0
            best_state = {kk: vv.detach().clone() for kk, vv in net.state_dict().items()}
        else:
            since += 1
            if since >= _PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def convolution_recal_pit_dma_new(K, x, y, pit_data, cdf_data, discount_factors,
                             warmup=15, eval_start=None,
                             alpha=_GAS_ALPHA, floor=_GAS_FLOOR,
                             refit_every=10, hidden_size=16, num_epochs=300,
                             learning_rate=1e-3, seed=42,
                             step_norm='coord', slow_shrink=1.0,
                             outer='bank', features=None, verbose=True):
    """v_t = tau_GAS(tau_NN(u_t)), scored prequentially over [eval_start, T).

    This is a COMPOSITION of two monotone maps (not a convolution).

    Every theta_NN_t comes from a head trained only on data before t, and the
    outer bank is advanced online on the DEPLOYED inner output w_s, so the
    whole window is one-step-ahead.

    What changed from the previous version
    ---------------------------------------
    * Training target. The head used to be fitted against the bank filtered on
      the RAW PITs (pass 1), but deployed in front of a bank re-filtered on w
      (pass 3) -- a mismatch between training and use. The outer bank now runs
      online inside the loop, and each refit trains against that deployed outer
      path theta_outer[:, s], W_outer[:, s], s < t. (The outer states are held
      at their deployed values while W changes; re-unrolling the filter inside
      the optimiser would be exact but costs O(t) filter steps per epoch.)
    * Ablations. outer='none' trains and scores the network-only map
      tau_NN (no filter bank); hidden_size=0 gives a linear head; features
      selects a subset of FEATURE_NAMES (e.g. drop 'median_loc' to check the
      gain is shape-driven rather than a location-keyed bias correction).
    * Reporting. A HAC DM test of the composition against the bank-on-raw
      alone ('dm_vs_gas_only') replaces the order-dependent NN/GAS split as the
      measure of what the network adds; the split is still reported. Training
      sizes and the first refit date are reported. Sharpness is measured.

    eval_start   first period scored; default warmup. The network is inactive
                 until warmup + _MIN_TRAIN, so pass warmup + _MIN_TRAIN to BOTH
                 methods to compare them where this one is fully active.
    alpha, floor, step_norm, slow_shrink
                 KEEP THESE EQUAL TO THE VALUES YOU PASS tvgas_pit_dma_new.
                 Pass 1 is that function's bank on the same raw PITs, so with
                 matching settings 'll_gas_only_raw' reproduces its output and
                 'dm_vs_gas_only' isolates the network's contribution.
    """
    if outer not in ('bank', 'none'):
        raise ValueError("outer must be 'bank' or 'none'")
    feat_idx = (list(range(len(FEATURE_NAMES))) if features is None
                else [FEATURE_NAMES.index(f) for f in features])
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    names = [e[0] for e in _GAS_EXPERTS]
    E = len(_GAS_EXPERTS)
    use_bank = outer == 'bank'
    results = []

    for i, delta in enumerate(discount_factors):
        if verbose:
            print(f"\n{'=' * 72}\nComposition  v = "
                  + ("tau_GAS(tau_NN(u))" if use_bank else "tau_NN(u)  [network only]")
                  + f"  |  base: {delta}\n{'=' * 72}")
        u = _prep_pit(pit_data[i])                       # SHARED preprocessing
        cdf_seq = np.asarray(cdf_data[i], float)
        T = len(u)
        P = 3 * K + 1
        es = warmup if eval_start is None else int(eval_start)
        if not (warmup <= es < T):
            raise ValueError(f"eval_start={es} outside [warmup={warmup}, T={T})")

        Z = causal_rank_norm(build_cdf_features(cdf_seq, x))[:, feat_idx]
        X_t = torch.tensor(Z, dtype=torch.float32)
        u_t = torch.tensor(u, dtype=torch.float32)

        # ---- pass 1: bank on the raw PITs = GAS alone (nothing estimated) - #
        ll_raw_bank, theta_raw = run_gas_bank(u, K, warmup, step_norm, slow_shrink)
        W_raw = dma_weights(ll_raw_bank, alpha, warmup, floor)
        ll_gas_raw = logsumexp(np.log(W_raw + 1e-30) + ll_raw_bank, axis=0)

        # ---- pass 2: inner head + outer bank, advanced together online ---- #
        bank = GasBank(K, warmup, step_norm, slow_shrink) if use_bank else None
        theta_bank = np.zeros((E, T, P)); ll_bank = np.zeros((E, T))
        theta_nn = np.zeros((T, P)); w_np = u.copy(); ll_nn = np.zeros(T)
        head = None
        refit_dates, train_sizes = [], []
        for t in range(T):
            k_since = t - warmup
            if k_since >= _MIN_TRAIN and k_since % refit_every == 0:
                if use_bank:
                    W_past = dma_weights(ll_bank[:, :t], alpha, warmup, floor)
                    tb = torch.tensor(theta_bank[:, :t, :], dtype=torch.float32)
                    lw = torch.tensor(np.log(W_past + 1e-30), dtype=torch.float32)
                else:
                    tb = lw = None
                head = _build_mlp(Z.shape[1], hidden_size, P, seed)
                head = _train_head(head, X_t, u_t, tb, lw, K, warmup, t,
                                   num_epochs, learning_rate)
                refit_dates.append(t); train_sizes.append(t - warmup)
            if head is not None:
                with torch.no_grad():
                    theta_nn[t] = head(X_t[t:t + 1]).numpy()[0]
                wv, lv = _spline_rows(theta_nn[t:t + 1], u[t:t + 1], K)
                w_np[t] = float(np.clip(wv[0], _EPS, 1 - _EPS))
                ll_nn[t] = float(lv[0])
            if use_bank:                      # theta used at t is pre-update
                ll_bank[:, t], theta_bank[:, t, :] = bank.step(w_np[t])

        if use_bank:
            W_gas = dma_weights(ll_bank, alpha, warmup, floor)
            ll_gas = logsumexp(np.log(W_gas + 1e-30) + ll_bank, axis=0)
        else:
            W_gas = None
            ll_gas = np.zeros(T)

        # the full composition IS the recalibrated PIT; no blend, no floor
        v_full = np.zeros(T); v_gas_only = np.zeros(T)
        for t in range(T):
            v_full[t] = (_bank_eval(theta_bank[:, t, :], np.array([w_np[t]]), K,
                                    W_gas[:, t])[0][0] if use_bank else w_np[t])
            v_gas_only[t] = _bank_eval(theta_raw[:, t, :], np.array([u[t]]), K, W_raw[:, t])[0][0]
        v_full = np.clip(v_full, _EPS, 1 - _EPS)
        ll_total = ll_nn + ll_gas

        # ---- moments, SHARED quadrature ---------------------------------- #
        crps = np.zeros(T); crps_raw = np.zeros(T)
        ymean = np.zeros(T); ymean_raw = np.zeros(T)
        tau_grid = np.zeros((T, _GL_N))   # the composed map, for localized eval
        q_cal = np.full((T, 2), np.nan); q_raw = np.full((T, 2), np.nan)
        ones = np.ones_like(_UN)
        for t in range(T):
            wg, dwg = _spline_on_grid(theta_nn[t], K)
            wg = np.clip(wg, _EPS, 1 - _EPS)
            if use_bank:
                vg, dvg = _bank_eval(theta_bank[:, t, :], wg, K, W_gas[:, t])
            else:
                vg, dvg = wg, np.ones_like(wg)
            tau_grid[t] = vg
            ymean[t], crps[t] = _pit_moments(cdf_seq[t], x, y[t], vg, dvg * dwg)
            ymean_raw[t], crps_raw[t] = _pit_moments(cdf_seq[t], x, y[t], _UN, ones)
            q_cal[t] = _pit_quantiles(cdf_seq[t], x, vg, _IVL_PROBS)
            q_raw[t] = _pit_quantiles(cdf_seq[t], x, _UN, _IVL_PROBS)

        # ---- report, SHARED ---------------------------------------------- #
        sl = slice(es, T)
        out = _report(u, v_full, ll_total, y, ymean, ymean_raw, crps, crps_raw,
                      sl, _ECE_BINS)
        out['logscore_nn'] = float(ll_nn[sl].mean())
        out['logscore_gas'] = float(ll_gas[sl].mean())
        out['logscore_gas_only_raw'] = float(ll_gas_raw[sl].mean())
        d = ll_total[sl] - ll_gas_raw[sl]
        db, se, ts, pv = dm_test(d, hac_lag=_hac_lag(len(d)))
        out['dm_vs_gas_only'] = {'mean_gain': db, 'se': se, 't': ts, 'p': pv}
        out['n_refits'] = len(refit_dates)
        out['first_refit_t'] = refit_dates[0] if refit_dates else None
        out['train_sizes'] = train_sizes
        out['n_nn_active'] = int(np.sum(np.abs(theta_nn[sl]).sum(1) > 0))
        out['outer'] = outer
        out['features'] = [FEATURE_NAMES[j] for j in feat_idx]
        out['hidden_size'] = int(hidden_size)
        out['alpha'] = float(alpha)
        out['floor'] = float(floor)
        out['step_norm'] = step_norm
        out['slow_shrink'] = float(slow_shrink)
        out['dma_effective_memory'] = (float('inf') if alpha >= 1.0
                                       else 1.0 / (1.0 - alpha))
        out['eff_n_experts_outer'] = (float(_effective_n(W_gas)[sl].mean())
                                      if use_bank else float('nan'))
        out['eff_n_experts_raw'] = float(_effective_n(W_raw)[sl].mean())
        sharp, sharp_lines = _sharpness_block(q_cal, q_raw, y, sl)
        out.update(sharp)

        if verbose:
            print(f"  alpha={alpha:g} (memory~"
                  + ("inf" if alpha >= 1 else f"{1/(1-alpha):.0f}")
                  + f"), floor={floor:g}, step_norm={step_norm}   eff. experts: raw bank "
                  f"{out['eff_n_experts_raw']:.2f}, outer bank "
                  f"{out['eff_n_experts_outer']:.2f} of {E}")
            print(f"  head: {'linear' if hidden_size == 0 else f'MLP H={hidden_size}'} on "
                  f"{len(feat_idx)} features; refits {out['n_refits']} (every {refit_every}, "
                  f"first at t={out['first_refit_t']}, train n "
                  + (f"{train_sizes[0]}..{train_sizes[-1]}" if train_sizes else "-")
                  + f");  active on {out['n_nn_active']}/{out['n']} scored periods")
            if use_bank:
                wg2 = W_gas[:, sl].mean(1)
                print("  outer GAS weights: " + "  ".join(
                    f"{names[k]}={wg2[k]:.2f}" for k in np.argsort(-wg2)[:3]))
            _print_report(
                f"EVALUATION [{es},{T})", out,
                extra_lines=[f"vs GAS alone (bank on raw) {db:+.4f}/obs  t={ts:+.2f}  "
                             f"p={pv:.3f}",
                             f"split (order-dependent)  NN {out['logscore_nn']:+.4f}  +  "
                             f"outer {out['logscore_gas']:+.4f}   | GAS alone "
                             f"{out['logscore_gas_only_raw']:+.4f}",
                             *sharp_lines])

        res = {'delta': delta, 'head': 'composition_v4', 'K': K,
               'warmup': warmup, 'eval_start': es, 'oos_start': es,
               'u_original': u.astype(np.float32),
               'w_transformed': w_np,
               'v_full_composition': v_full,
               'v_gas_only': np.clip(v_gas_only, _EPS, 1 - _EPS),
               'log_likelihood': ll_total, 'll_nn': ll_nn, 'll_gas': ll_gas,
               'll_gas_only_raw': ll_gas_raw,
               'theta_nn': theta_nn,
               'theta_gas': (np.einsum('kt,ktp->tp', W_gas, theta_bank)
                             if use_bank else None),
               'gas_expert_names': names, 'gas_weights': W_gas,
               'crps_history': crps, 'crps_raw_history': crps_raw,
               'y_calibrated_mean': ymean, 'y_raw_mean': ymean_raw,
               'y_calibrated_q05': q_cal[:, 0], 'y_calibrated_q95': q_cal[:, 1],
               'y_raw_q05': q_raw[:, 0], 'y_raw_q95': q_raw[:, 1],
               'interval_probs': np.asarray(_IVL_PROBS, float),
               'tau_grid': tau_grid, 'u_grid': _UN.copy(),
               'features_norm': Z, 'refit_dates': refit_dates}
        res.update(out)
        results.append(res)

    return results


# =========================================================================== #
#  6.  COMPARABILITY CHECK
# =========================================================================== #
def check_comparability(res_a, res_b, labels=('tvgas', 'convolution'), tol=0.0):
    """Assert that the baseline quantities really are identical.

    Every entry below depends only on (u, y, cdf) and the slice, never on the
    method, so with the same inputs and the same eval_start they must match
    exactly. Run this before you put the two in a table.
    """
    keys = ['n', 'ks_before', 'ece_before', 'crps_raw_mean', 'mean_bias_raw',
            'oracle_ceiling', 'ks_null_mean']
    print(f"{'metric':<18}{labels[0]:>16}{labels[1]:>16}   match")
    ok = True
    for k in keys:
        a, b = res_a[k], res_b[k]
        m = abs(float(a) - float(b)) <= tol
        ok &= m
        print(f"{k:<18}{float(a):>16.8f}{float(b):>16.8f}   {'yes' if m else 'NO'}")
    pa = res_a['pit_var_thirds_before']; pb = res_b['pit_var_thirds_before']
    m = all(abs(i - j) <= tol for i, j in zip(pa, pb))
    ok &= m
    print(f"{'pit_var_thirds':<18}{str([round(i,5) for i in pa]):>16}"
          f"{str([round(i,5) for i in pb]):>16}   {'yes' if m else 'NO'}")
    print("\nAll baseline metrics identical." if ok else
          "\nMISMATCH -- do not tabulate these side by side.")
    return ok