import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from scipy.optimize import minimize
from scipy.special import betainc, betaln, digamma, logsumexp

# ---- SHARED with the spline module: never reimplement these --------------- #
from recal_dma_new import (
    _EPS, _UN, _UW, _GL_N,
    _GAS_EXPERTS, _GAS_ALPHA, _GAS_FLOOR, _ECE_BINS,
    _IVL_PROBS, _pit_quantiles,
    _MIN_TRAIN, _HEAD_SCALE, _WEIGHT_DECAY, _LAMBDA_ID, _PATIENCE, _VAL_FRAC,
    _prep_pit, _pit_moments, _report, _print_report,
    dma_weights, _effective_n,
    dm_test, oracle_ceiling, per_third_var, check_comparability,   # noqa: F401
    build_cdf_features, causal_rank_norm, FEATURE_NAMES,           # noqa: F401
    _sharpness_block, _hac_lag, _STEP_NORMS,
    _sigmoid,
)

# ---- Beta link ------------------------------------------------------------ #
_FLOOR = 0.05          # floor on (a,b): bounds the boundary blow-up
_PITCLIP = 1e-4        # applied ONLY inside the Beta density, never to stored u
_SHIFT = float(np.log(np.expm1(1.0 - _FLOOR)))   # softplus(_SHIFT)+_FLOOR == 1


def _ab(la, lb):
    """logits -> (a, b). Vectorised. theta = 0 gives exactly a = b = 1."""
    return (np.logaddexp(0.0, la + _SHIFT) + _FLOOR,
            np.logaddexp(0.0, lb + _SHIFT) + _FLOOR)


def _ab_torch(l0, l1):
    return (torch.nn.functional.softplus(l0 + _SHIFT) + _FLOOR,
            torch.nn.functional.softplus(l1 + _SHIFT) + _FLOOR)


def _log_beta_pdf(u, a, b):
    uc = np.clip(u, _PITCLIP, 1 - _PITCLIP)
    return (a - 1) * np.log(uc) + (b - 1) * np.log1p(-uc) - betaln(a, b)


def _log_beta_pdf_torch(u, a, b):
    uc = torch.clamp(u, _PITCLIP, 1 - _PITCLIP)
    return ((a - 1) * torch.log(uc) + (b - 1) * torch.log1p(-uc)
            - (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)))


def _assert_identity():
    a, b = _ab(np.array([0.0]), np.array([0.0]))
    ld = _log_beta_pdf(np.array([0.01, 0.5, 0.99]), a, b)
    assert abs(a[0] - 1.0) < 1e-12 and abs(b[0] - 1.0) < 1e-12, (a, b)
    assert np.abs(ld).max() < 1e-12, ld
_assert_identity()


class _BetaincParam(torch.autograd.Function):
    """w = I_u(a,b) with gradients in (a,b), so the network can respond to the
    downstream GAS term. da/db by central difference; du analytically."""

    @staticmethod
    def forward(ctx, a, b, u):
        an, bn, un = (t.detach().cpu().numpy().astype(np.float64) for t in (a, b, u))
        ctx.save_for_backward(a, b, u)
        return torch.as_tensor(betainc(an, bn, np.clip(un, 0.0, 1.0)), dtype=a.dtype)

    @staticmethod
    def backward(ctx, go):
        a, b, u = ctx.saved_tensors
        an, bn, un = (t.detach().cpu().numpy().astype(np.float64) for t in (a, b, u))
        un = np.clip(un, 0.0, 1.0)
        h = 1e-4
        a_lo = np.maximum(an - h, _FLOOR)
        b_lo = np.maximum(bn - h, _FLOOR)
        dwda = (betainc(an + h, bn, un) - betainc(a_lo, bn, un)) / (an + h - a_lo)
        dwdb = (betainc(an, bn + h, un) - betainc(an, b_lo, un)) / (bn + h - b_lo)
        uc = np.clip(un, _PITCLIP, 1 - _PITCLIP)
        dwdu = np.exp((an - 1) * np.log(uc) + (bn - 1) * np.log1p(-uc) - betaln(an, bn))
        g = go.detach().cpu().numpy()
        return (torch.as_tensor(g * dwda, dtype=a.dtype),
                torch.as_tensor(g * dwdb, dtype=b.dtype),
                torch.as_tensor(g * dwdu, dtype=u.dtype))


# =========================================================================== #
#  SHARED FILTER BANK  (Beta)
#  Used by BOTH methods: it IS method 1, and it is the outer map of method 2.
#  Nothing is estimated -- logits_t depends only on seq[:t].
# =========================================================================== #
class GasBankBeta:
    """E fixed two-timescale score-driven Beta filters, one observation at a time.

        slow_{t+1} = rho_s slow_t + a_s r_t g_t   (rho_s = 1: unit root)
        fast_{t+1} = b fast_t     + a_f r_t g_t   mean-reverting

    The score is analytic -- no autograd per period:

        d/da log tau' = log u     - (psi(a) - psi(a+b))
        d/db log tau' = log(1-u)  - (psi(b) - psi(a+b))

    chained through the softplus link. step(val) returns the logits used at
    this period (computed before val is seen) and then updates.

    step_norm, slow_shrink have exactly the meaning they have in the spline
    module's GasBank. Use the SAME values for both families: the point of
    'coord' is that Beta (2 params) and spline (3K+1) then move at the same
    nominal rate per parameter, so a family comparison is not a step-size
    comparison. step_norm='global', slow_shrink=1 reproduces the original bank.
    """

    def __init__(self, warmup: int, step_norm: str = 'coord',
                 slow_shrink: float = 1.0) -> None:
        if step_norm not in _STEP_NORMS:
            raise ValueError(f"step_norm must be one of {_STEP_NORMS}")
        if not 0.0 < slow_shrink <= 1.0:
            raise ValueError("slow_shrink must lie in (0, 1]")
        self.warmup, self.step_norm, self.slow_shrink = int(warmup), step_norm, float(slow_shrink)
        self.a_s = np.array([e[1] for e in _GAS_EXPERTS])[:, None]
        self.a_f = np.array([e[2] for e in _GAS_EXPERTS])[:, None]
        self.br = np.array([e[3] for e in _GAS_EXPERTS])[:, None]
        self.gg = np.array([e[4] for e in _GAS_EXPERTS])[:, None]
        self.E, self.P = len(_GAS_EXPERTS), 2
        self.S = np.zeros((self.E, 2)); self.Fq = np.zeros((self.E, 2))
        cols = 2 if step_norm == 'coord' else 1
        self.gn = np.full((self.E, cols), 1e-4 / cols)
        self.t = 0

    def step(self, val: float) -> tuple[np.ndarray, np.ndarray]:
        L = np.zeros((self.E, 2)) if self.t < self.warmup else (self.S + self.Fq)
        a, b = _ab(L[:, 0], L[:, 1])
        pt = float(np.clip(val, _PITCLIP, 1 - _PITCLIP))
        lu, l1u = np.log(pt), np.log1p(-pt)
        l = (a - 1) * lu + (b - 1) * l1u - betaln(a, b)
        g = np.stack([(lu - (digamma(a) - digamma(a + b))) * _sigmoid(L[:, 0] + _SHIFT),
                      (l1u - (digamma(b) - digamma(a + b))) * _sigmoid(L[:, 1] + _SHIFT)],
                     axis=1)
        bad = ~(np.isfinite(l) & np.isfinite(g).all(axis=1))
        if bad.any():
            l = np.where(bad, -50.0, l)      # penalise, never reward, a blow-up
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
        self.Fq = self.br * self.Fq + self.a_f * r * g
        self.t += 1
        return l, L


def run_gas_bank_beta(seq, warmup, step_norm='coord', slow_shrink=1.0):
    """Whole-sequence wrapper. Returns loglik (E,T), logits (E,T,2)."""
    bank = GasBankBeta(warmup, step_norm, slow_shrink)
    T = len(seq)
    logits = np.zeros((bank.E, T, 2)); ll = np.zeros((bank.E, T))
    for t in range(T):
        ll[:, t], logits[:, t, :] = bank.step(seq[t])
    return ll, logits


def run_static_expanding_beta(u: np.ndarray, warmup: int, refit_every: int = 25,
                              min_train: int = _MIN_TRAIN, ridge: float = _LAMBDA_ID,
                              window: int | None = None
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Static Beta recalibration benchmark: eq. (9) on an expanding window
    (window=None) or a rolling window of the last `window` PITs.

    At each refit date t the logits maximise mean_{s<t} log Beta-pdf(u_s; a, b)
    (same link, floor and _PITCLIP as the filters, plus the same small ridge to
    the identity as the network) and are held until the next refit. This is
    the BLP of Gneiting & Ranjan (2013) refitted recursively. Closed-form score,
    so L-BFGS converges in a few iterations. Strictly one-step-ahead.
    Returns (log tau'_t(u_t), logits path (T,2), recalibrated PITs v).
    """
    u = _prep_pit(u)
    T = len(u)
    uc = np.clip(u, _PITCLIP, 1 - _PITCLIP)
    lu_all, l1u_all = np.log(uc), np.log1p(-uc)
    L = np.zeros(2)
    path = np.zeros((T, 2))

    def nll(Lv, m1, m2):
        a, b = _ab(Lv[0:1], Lv[1:2])
        a, b = float(a[0]), float(b[0])
        f = -((a - 1) * m1 + (b - 1) * m2 - betaln(a, b)) + ridge * np.mean(Lv ** 2)
        dab = digamma(a + b)
        g = np.array([-(m1 - digamma(a) + dab) * _sigmoid(Lv[0] + _SHIFT),
                      -(m2 - digamma(b) + dab) * _sigmoid(Lv[1] + _SHIFT)]) + ridge * Lv
        return f, g

    for t in range(T):
        if t >= warmup + min_train and (t - warmup) % refit_every == 0:
            lo = 0 if window is None else max(0, t - int(window))
            m1, m2 = lu_all[lo:t].mean(), l1u_all[lo:t].mean()
            L = minimize(nll, L, args=(m1, m2), jac=True, method='L-BFGS-B').x
        path[t] = L
    a, b = _ab(path[:, 0], path[:, 1])
    ll = _log_beta_pdf(u, a, b)
    v = np.clip(betainc(a, b, uc), _EPS, 1 - _EPS)
    return ll, path, v


def _bank_eval_beta(logits_col, uu, weights):
    """Mixture map sum_k w_k I_.(a_k,b_k) and its derivative. A convex mixture
    of monotone maps is a monotone map and differentiation is linear, so the
    mixture's log-score is a valid recalibrator's log-score."""
    a, b = _ab(logits_col[:, 0], logits_col[:, 1])
    uc = np.clip(np.asarray(uu, float), _PITCLIP, 1 - _PITCLIP)
    V = betainc(a[:, None], b[:, None], uc[None, :])
    D = np.exp(_log_beta_pdf(uc[None, :], a[:, None], b[:, None]))
    w = np.asarray(weights)[:, None]
    return (w * V).sum(0), (w * D).sum(0)


def _n_extreme(ll, sl):
    return int(np.sum(np.abs(ll[sl]) > 3.0))


# =========================================================================== #
#  METHOD 1 --- two-timescale score-driven recalibration (aggregated), Beta
# =========================================================================== #
def tvgas_pit_beta_dma_new(x, y, pit_data, cdf_data, discount_factors,
                   warmup=15, eval_start=None, alpha=_GAS_ALPHA,
                   floor=_GAS_FLOOR, step_norm='coord', slow_shrink=1.0,
                   static_baseline=True, static_refit_every=25, verbose=True):
    """v_t = tau_GAS_t(u_t) with Beta maps and psi replaced by aggregation.

    As in the spline version there is no optimiser: E = 5 fixed filters run in
    parallel and are combined by exponentially weighted fixed share on the
    realised log score, so nothing is estimated on the scored window and
    pi_{k,t} uses only scores strictly before t.

    This function is exactly the "bank on the raw PITs" stage of
    convolution_recal_pit_v3_beta_neww -- with the same warmup, ALPHA, FLOOR
    and eval_start it reproduces that function's 'll_gas_only_raw' and
    'v_gas_only' to machine precision, so the two are nested and "what does
    the network add" is answerable by subtraction. Give the two functions
    different discount factors and that nesting silently breaks: the
    subtraction then mixes a change of architecture with a change of
    forgetting rate, and neither effect is identified.

    alpha  DMA discount factor, effective memory 1/(1-alpha) observations.
           alpha = 1 is BMA over the bank and the log(E)/n no-harm bound
           against the identity filter holds; alpha < 1 trades that static
           bound for the ability to track a change in WHICH adaptation rate
           is right. Default 0.98 (~50 obs).
    floor  minimum weight per filter, so a decayed filter stays recoverable.
           Multiplicative forgetting has no such guarantee on its own.

    step_norm, slow_shrink   see GasBankBeta; keep equal to the spline run.
    static_baseline          also fit the recursive static BLP
                             (run_static_expanding_beta) and report a HAC DM
                             test of GAS against it ('dm_vs_static'): the test
                             of whether TIME VARIATION helps.
    Sharpness (interval width, coverage, interval score) is reported directly.

    No K: a Beta map has two parameters.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    names = [e[0] for e in _GAS_EXPERTS]
    E = len(_GAS_EXPERTS)
    is_fast = np.array([e[2] > 0 for e in _GAS_EXPERTS])
    results = []

    for i, delta in enumerate(discount_factors):
        if verbose:
            print(f"\n{'=' * 72}\nTwo-timescale GAS BETA (aggregated)  v = I_u(a_t,b_t)  |  base: {delta}\n{'=' * 72}")
        u = _prep_pit(pit_data[i])                       # SHARED preprocessing
        cdf_seq = np.asarray(cdf_data[i], float)
        T = len(u)
        es = warmup if eval_start is None else int(eval_start)
        if not (warmup <= es < T):
            raise ValueError(f"eval_start={es} outside [warmup={warmup}, T={T})")

        ll_bank, logits = run_gas_bank_beta(u, warmup, step_norm, slow_shrink)
        W = dma_weights(ll_bank, alpha, warmup, floor)
        ll = logsumexp(np.log(W + 1e-30) + ll_bank, axis=0)

        v = np.zeros(T)
        for t in range(T):
            v[t] = _bank_eval_beta(logits[:, t, :], np.array([u[t]]), W[:, t])[0][0]
        v = np.clip(v, _EPS, 1 - _EPS)

        # ---- moments, SHARED quadrature ---------------------------------- #
        crps = np.zeros(T); crps_raw = np.zeros(T)
        ymean = np.zeros(T); ymean_raw = np.zeros(T)
        tau_grid = np.zeros((T, _GL_N))      # the map itself, for localized eval
        q_cal = np.full((T, 2), np.nan); q_raw = np.full((T, 2), np.nan)
        ones = np.ones_like(_UN)
        for t in range(T):
            tau, dtau = _bank_eval_beta(logits[:, t, :], _UN, W[:, t])
            tau_grid[t] = tau
            ymean[t], crps[t] = _pit_moments(cdf_seq[t], x, y[t], tau, dtau)
            ymean_raw[t], crps_raw[t] = _pit_moments(cdf_seq[t], x, y[t], _UN, ones)
            q_cal[t] = _pit_quantiles(cdf_seq[t], x, tau, _IVL_PROBS)
            q_raw[t] = _pit_quantiles(cdf_seq[t], x, _UN, _IVL_PROBS)

        # ---- report, SHARED ---------------------------------------------- #
        sl = slice(es, T)
        out = _report(u, v, ll, y, ymean, ymean_raw, crps, crps_raw, sl, _ECE_BINS)

        a_path, b_path = _ab(np.einsum('kt,kt->t', W, logits[:, :, 0]),
                             np.einsum('kt,kt->t', W, logits[:, :, 1]))
        ex_mean = ll_bank[:, sl].mean(axis=1)
        w_mean = W[:, sl].mean(axis=1)
        kbest = int(np.argmax(ll_bank[:, sl].sum(axis=1)))
        out['expert_names'] = names
        out['expert_logscore_mean'] = {n: float(m) for n, m in zip(names, ex_mean)}
        out['expert_weight_mean'] = {n: float(w) for n, w in zip(names, w_mean)}
        out['best_expert'] = names[kbest]
        out['best_expert_logscore_mean'] = float(ex_mean[kbest])
        out['mixture_minus_best_expert'] = float(out['logscore_mean'] - ex_mean[kbest])
        out['aggregation_floor'] = float(-np.log(E) / out['n'])
        out['eff_n_experts'] = float(_effective_n(W)[sl].mean())
        out['weight_on_fast'] = float(w_mean[is_fast].sum())
        out['alpha'] = float(alpha)
        out['floor'] = float(floor)
        out['dma_effective_memory'] = (float('inf') if alpha >= 1.0
                                       else 1.0 / (1.0 - alpha))
        out['n_extreme'] = _n_extreme(ll, sl)
        out['a_range'] = (float(a_path[sl].min()), float(a_path[sl].max()))
        out['b_range'] = (float(b_path[sl].min()), float(b_path[sl].max()))
        out['step_norm'] = step_norm
        out['slow_shrink'] = float(slow_shrink)

        sharp, sharp_lines = _sharpness_block(q_cal, q_raw, y, sl)
        out.update(sharp)

        static_lines = []
        ll_static = v_static = logits_static = None
        if static_baseline:
            ll_static, logits_static, v_static = run_static_expanding_beta(
                u, warmup, refit_every=static_refit_every)
            d = ll[sl] - ll_static[sl]
            db, se, ts, pv = dm_test(d, hac_lag=_hac_lag(len(d)))
            out['logscore_static_mean'] = float(ll_static[sl].mean())
            out['ks_after_static'] = float(stats.kstest(v_static[sl], 'uniform').statistic)
            out['dm_vs_static'] = {'mean_gain': db, 'se': se, 't': ts, 'p': pv}
            static_lines = [f"static expanding BLP {out['logscore_static_mean']:+.4f}/obs "
                            f"(KS {out['ks_after_static']:.4f});  GAS - static "
                            f"{db:+.4f}  t={ts:+.2f}  p={pv:.3f}"]

        if verbose:
            top = np.argsort(-w_mean)[:3]
            print("  weights: " + "  ".join(f"{names[k]}={w_mean[k]:.2f}" for k in top)
                  + f"   (eff. {out['eff_n_experts']:.2f} of {E};"
                    f" mass on fast filters {out['weight_on_fast']:.2f})")
            _print_report(
                f"EVALUATION [{es},{T})", out,
                extra_lines=[
                    f"best single filter '{out['best_expert']}' "
                    f"{out['best_expert_logscore_mean']:+.4f}/obs; mixture "
                    f"{out['mixture_minus_best_expert']:+.4f} vs it "
                    f"(no-harm floor, valid at alpha=1: {out['aggregation_floor']:+.4f})",
                    f"a in [{out['a_range'][0]:.2f},{out['a_range'][1]:.2f}]  "
                    f"b in [{out['b_range'][0]:.2f},{out['b_range'][1]:.2f}]   "
                    f"|ell|>3 on {out['n_extreme']}/{out['n']} periods",
                    *static_lines, *sharp_lines])
            if out['n_extreme'] > max(2, out['n'] // 50):
                print("     [warn] the Beta boundary is contributing materially to "
                      "the log score; read n_extreme before quoting it")

        res = {'delta': delta, 'head': 'tvgas_beta_aggregated',
               'warmup': warmup, 'eval_start': es, 'oos_start': es,
               'u_original': u.astype(np.float32),
               'v_transformed': v,
               'log_likelihood': ll,
               'a_path': a_path, 'b_path': b_path,
               'logits_experts': logits, 'expert_logscore': ll_bank,
               'gas_expert_names': names, 'gas_weights': W,
               'crps_history': crps, 'crps_raw_history': crps_raw,
               'y_calibrated_mean': ymean, 'y_raw_mean': ymean_raw,
               'y_calibrated_q05': q_cal[:, 0], 'y_calibrated_q95': q_cal[:, 1],
               'y_raw_q05': q_raw[:, 0], 'y_raw_q95': q_raw[:, 1],
               'interval_probs': np.asarray(_IVL_PROBS, float),
               'tau_grid': tau_grid, 'u_grid': _UN.copy(),
               'll_static': ll_static, 'v_static': v_static,
               'logits_static': logits_static}
        res.update(out)
        results.append(res)

    return results


# =========================================================================== #
#  METHOD 2 --- the composition recalibrator, Beta
# =========================================================================== #
def _build_mlp_beta(in_dim, hidden, seed):
    """hidden=0 gives a linear head (the linear-in-features ablation)."""
    torch.manual_seed(seed)
    if hidden == 0:
        net = nn.Sequential(nn.Linear(in_dim, 2)).float()
    else:
        net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                            nn.Linear(hidden, 2)).float()
    with torch.no_grad():
        net[-1].weight.mul_(_HEAD_SCALE)
        net[-1].bias.zero_()
    return net


def _train_head_beta(net, X_t, u_t, gas_logits_t, logW_t, lo, hi, num_epochs, lr):
    """Fit the network on [lo, hi) only. The bank path is frozen, so nothing
    here sees data at or beyond hi. The ridge is on the network's OUTPUT, i.e.
    a Gaussian prior centred on the identity map (a = b = 1), not merely on
    the weights.

    gas_logits_t / logW_t must be the OUTER bank as actually deployed on
    w_s = tau_NN(u_s) for s < hi (not the bank filtered on raw u). Pass None
    for both to train the network-only ablation (objective = log tau_NN')."""
    n = hi - lo
    n_val = max(5, int(_VAL_FRAC * n))
    fit_hi = hi - n_val
    if fit_hi <= lo + 2:
        return net
    use_bank = gas_logits_t is not None
    m = hi - lo
    if use_bank:
        E = gas_logits_t.shape[0]
        ag, bg = _ab_torch(gas_logits_t[:, lo:hi, 0], gas_logits_t[:, lo:hi, 1])
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=_WEIGHT_DECAY)
    best, best_state, since = float('inf'), None, 0
    net.train()
    for _ in range(num_epochs):
        opt.zero_grad()
        th = net(X_t[lo:hi])
        a_nn, b_nn = _ab_torch(th[:, 0], th[:, 1])
        ll_nn = _log_beta_pdf_torch(u_t[lo:hi], a_nn, b_nn)
        if use_bank:
            w = torch.clamp(_BetaincParam.apply(a_nn, b_nn, u_t[lo:hi]),
                            _PITCLIP, 1 - _PITCLIP)
            ll_k = _log_beta_pdf_torch(w.unsqueeze(0).expand(E, m), ag, bg)
            ll_gas = torch.logsumexp(logW_t[:, lo:hi] + ll_k, dim=0)
            obj = ll_nn + ll_gas
        else:
            obj = ll_nn
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


def convolution_recal_pit_beta_dma_new(x, y, pit_data, cdf_data, discount_factors,
                                  warmup=15, eval_start=None,
                                  alpha=_GAS_ALPHA, floor=_GAS_FLOOR,
                                  refit_every=10, hidden_size=16, num_epochs=300,
                                  learning_rate=1e-3, seed=42,
                                  step_norm='coord', slow_shrink=1.0,
                                  outer='bank', features=None, verbose=True):
    """v_t = tau_GAS(tau_NN(u_t)) with Beta maps, scored prequentially over
    [eval_start, T). A COMPOSITION of two Beta maps (not a convolution).

    Mirrors convolution_recal_pit_dma_new in the spline module:
    * The outer bank runs online on the DEPLOYED inner output w_s, and each
      refit trains the head against that deployed outer path (previously it
      trained against the bank filtered on raw u, then deployed in front of a
      different bank). Outer states are held at their deployed values while
      the head changes.
    * Ablations: outer='none' (network-only map), hidden_size=0 (linear head),
      features=[...] (subset of FEATURE_NAMES, e.g. drop 'median_loc').
    * 'dm_vs_gas_only': HAC DM test of the composition against the bank on raw
      PITs -- what the network adds. The NN/GAS split is still reported but
      depends on ordering.
    * Refit dates, training sizes and sharpness are reported.

    eval_start   first period scored; default warmup. The network is inactive
                 until warmup + _MIN_TRAIN; pass that to BOTH methods to compare
                 where this one is fully active.
    alpha, floor, step_norm, slow_shrink
                 keep equal to what you pass tvgas_pit_beta_dma_new, or the
                 nesting ('ll_gas_only_raw' == its log score) breaks.
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
            print(f"\n{'=' * 72}\nComposition BETA  v = "
                  + ("I(a_G,b_G) o I(a_N,b_N)" if use_bank else "I(a_N,b_N)  [network only]")
                  + f"  |  base: {delta}\n{'=' * 72}")
        u = _prep_pit(pit_data[i])                       # SHARED preprocessing
        cdf_seq = np.asarray(cdf_data[i], float)
        T = len(u)
        es = warmup if eval_start is None else int(eval_start)
        if not (warmup <= es < T):
            raise ValueError(f"eval_start={es} outside [warmup={warmup}, T={T})")

        Z = causal_rank_norm(build_cdf_features(cdf_seq, x))[:, feat_idx]
        X_t = torch.tensor(Z, dtype=torch.float32)
        u_t = torch.tensor(u, dtype=torch.float32)
        uc = np.clip(u, _PITCLIP, 1 - _PITCLIP)

        # ---- pass 1: bank on the raw PITs = GAS alone (nothing estimated) - #
        ll_raw_bank, logits_raw = run_gas_bank_beta(u, warmup, step_norm, slow_shrink)
        W_raw = dma_weights(ll_raw_bank, alpha, warmup, floor)
        ll_gas_raw = logsumexp(np.log(W_raw + 1e-30) + ll_raw_bank, axis=0)

        # ---- pass 2: inner head + outer bank, advanced together online ---- #
        bank = GasBankBeta(warmup, step_norm, slow_shrink) if use_bank else None
        logits_gas = np.zeros((E, T, 2)); ll_bank = np.zeros((E, T))
        logits_nn = np.zeros((T, 2)); w_np = uc.copy(); ll_nn = np.zeros(T)
        head = None
        refit_dates, train_sizes = [], []
        for t in range(T):
            k_since = t - warmup
            if k_since >= _MIN_TRAIN and k_since % refit_every == 0:
                if use_bank:
                    W_past = dma_weights(ll_bank[:, :t], alpha, warmup, floor)
                    gl = torch.tensor(logits_gas[:, :t, :], dtype=torch.float32)
                    lw = torch.tensor(np.log(W_past + 1e-30), dtype=torch.float32)
                else:
                    gl = lw = None
                head = _build_mlp_beta(Z.shape[1], hidden_size, seed)
                head = _train_head_beta(head, X_t, u_t, gl, lw, warmup, t,
                                        num_epochs, learning_rate)
                refit_dates.append(t); train_sizes.append(t - warmup)
            if head is not None:
                with torch.no_grad():
                    logits_nn[t] = head(X_t[t:t + 1]).numpy()[0]
                a1, b1 = _ab(logits_nn[t, 0], logits_nn[t, 1])
                w_np[t] = float(np.clip(betainc(a1, b1, uc[t]), _PITCLIP, 1 - _PITCLIP))
                ll_nn[t] = float(_log_beta_pdf(u[t], a1, b1))
            if use_bank:                      # logits used at t are pre-update
                ll_bank[:, t], logits_gas[:, t, :] = bank.step(w_np[t])

        a_nn, b_nn = _ab(logits_nn[:, 0], logits_nn[:, 1])
        if use_bank:
            W_gas = dma_weights(ll_bank, alpha, warmup, floor)
            ll_gas = logsumexp(np.log(W_gas + 1e-30) + ll_bank, axis=0)
        else:
            W_gas = None
            ll_gas = np.zeros(T)

        # the full composition IS the recalibrated PIT; no blend, no floor
        v_full = np.zeros(T); v_gas_only = np.zeros(T)
        for t in range(T):
            v_full[t] = (_bank_eval_beta(logits_gas[:, t, :], np.array([w_np[t]]),
                                         W_gas[:, t])[0][0] if use_bank else w_np[t])
            v_gas_only[t] = _bank_eval_beta(logits_raw[:, t, :], np.array([u[t]]), W_raw[:, t])[0][0]
        v_full = np.clip(v_full, _EPS, 1 - _EPS)
        v_gas_only = np.clip(v_gas_only, _EPS, 1 - _EPS)
        ll_total = ll_nn + ll_gas

        # ---- moments, SHARED quadrature ---------------------------------- #
        crps = np.zeros(T); crps_raw = np.zeros(T)
        ymean = np.zeros(T); ymean_raw = np.zeros(T)
        tau_grid = np.zeros((T, _GL_N))   # the composed map, for localized eval
        q_cal = np.full((T, 2), np.nan); q_raw = np.full((T, 2), np.nan)
        ones = np.ones_like(_UN)
        unc = np.clip(_UN, _PITCLIP, 1 - _PITCLIP)
        for t in range(T):
            wg = np.clip(betainc(a_nn[t], b_nn[t], unc), _PITCLIP, 1 - _PITCLIP)
            dwg = np.exp(_log_beta_pdf(unc, a_nn[t], b_nn[t]))
            if use_bank:
                vg, dvg = _bank_eval_beta(logits_gas[:, t, :], wg, W_gas[:, t])
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
        if use_bank:
            a_gas, b_gas = _ab(np.einsum('kt,kt->t', W_gas, logits_gas[:, :, 0]),
                               np.einsum('kt,kt->t', W_gas, logits_gas[:, :, 1]))
        else:
            a_gas = b_gas = np.ones(T)
        out['logscore_nn'] = float(ll_nn[sl].mean())
        out['logscore_gas'] = float(ll_gas[sl].mean())
        out['logscore_gas_only_raw'] = float(ll_gas_raw[sl].mean())
        d = ll_total[sl] - ll_gas_raw[sl]
        db, se, ts, pv = dm_test(d, hac_lag=_hac_lag(len(d)))
        out['dm_vs_gas_only'] = {'mean_gain': db, 'se': se, 't': ts, 'p': pv}
        out['n_refits'] = len(refit_dates)
        out['first_refit_t'] = refit_dates[0] if refit_dates else None
        out['train_sizes'] = train_sizes
        out['n_nn_active'] = int(np.sum(np.abs(logits_nn[sl]).sum(1) > 0))
        out['n_extreme'] = _n_extreme(ll_total, sl)
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
        out['a_nn_range'] = (float(a_nn[sl].min()), float(a_nn[sl].max()))
        out['a_gas_range'] = (float(a_gas[sl].min()), float(a_gas[sl].max()))
        sharp, sharp_lines = _sharpness_block(q_cal, q_raw, y, sl)
        out.update(sharp)

        if verbose:
            print(f"  alpha={alpha:g}, floor={floor:g}, step_norm={step_norm}   "
                  f"eff. experts: raw bank {out['eff_n_experts_raw']:.2f}, outer bank "
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
                extra_lines=[
                    f"vs GAS alone (bank on raw) {db:+.4f}/obs  t={ts:+.2f}  p={pv:.3f}",
                    f"split (order-dependent)  NN {out['logscore_nn']:+.4f}  +  outer "
                    f"{out['logscore_gas']:+.4f}   | GAS alone {out['logscore_gas_only_raw']:+.4f}",
                    f"a_NN in [{out['a_nn_range'][0]:.2f},{out['a_nn_range'][1]:.2f}]  "
                    f"a_GAS in [{out['a_gas_range'][0]:.2f},{out['a_gas_range'][1]:.2f}]"
                    f"   |ell|>3 on {out['n_extreme']}/{out['n']} periods",
                    *sharp_lines])
            if out['n_extreme'] > max(2, out['n'] // 50):
                print("     [warn] the Beta boundary is contributing materially to "
                      "the log score; read n_extreme before quoting it")

        res = {'delta': delta, 'head': 'composition_v4_beta',
               'warmup': warmup, 'eval_start': es, 'oos_start': es,
               'u_original': u.astype(np.float32),
               'w_transformed': w_np,
               'v_full_composition': v_full,
               'v_gas_only': v_gas_only,
               'log_likelihood': ll_total, 'll_nn': ll_nn, 'll_gas': ll_gas,
               'll_gas_only_raw': ll_gas_raw,
               'a_nn': a_nn, 'b_nn': b_nn, 'a_gas': a_gas, 'b_gas': b_gas,
               'logits_nn': logits_nn,
               'logits_gas_experts': logits_gas if use_bank else None,
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