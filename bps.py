"""
bps.py -- Python port of McAlinn & West's MATLAB code for Dynamic Bayesian Predictive Synthesis
(McAlinn & West, 2019, Journal of Econometrics 210: 155-169).

    BPS(...)            <- BPS.m            one MCMC run on data y_{1:t}
    bps_predictive(...) <- the loop body of the driver script: the 1-step predictive of y_{t+1}
    run_bps(...)        <- the driver script: expanding window t = 50, ..., T-1

Synthesis model:  y_t = x_t' theta_t + nu_t,  theta_t = theta_{t-1} + omega_t,
                  nu_t ~ N(0, v_t),  omega_t ~ N(0, v_t W_t),
                  latent agent states x_{j,t} ~ t(a_{j,t}, A_{j,t}, n_{j,t}).

PORTING NOTES
* MATLAB is 1-based; here everything is 0-based. The driver's `for t = 50:T-1` fits on y(1:t)
  and forecasts y(t+1); here `start=50` fits on y[:s] and forecasts y[s] for s = 50..T-1.
* `delta = [0.95 0.99]` in the MATLAB is [state discount d, volatility discount beta].
* Defaults are the MATLAB settings: m_0 = (0, 1/J, ..., 1/J), C_0 = I, n_0 = 1/(1-0.99) = 100,
  s_0 = 0.01, burn_in = 3000, mcmc_iter = 5000.
* exact_port=True (default) reproduces three details of the MATLAB exactly, so results match the
  authors' code. exact_port=False replaces them with the textbook versions:
    (i)   initial latent scales: MATLAB draws phi ~ IG(beta*n/2, beta*n/2) (uses the volatility
          discount beta); textbook: IG(n/2, n/2).
    (ii)  latent-scale update: MATLAB `phi = 0.5*(n+1)/randg((n+d)/2)` is IG((n+d)/2, (n+1)/2);
          the conditional posterior of a t scale mixture is IG((n+1)/2, (n+d)/2).
    (iii) predictive draw of x_{t+1}: MATLAB uses ONE scale lambda shared by all agents, drawn with
          the first agent's dof times beta (`n(t+1)` is a linear index = n(t+1,1)); textbook:
          independent t draws per agent with each agent's own dof.
  The differences are small when the agents' dof are moderate, but they exist.

OUTPUT FORMAT (same as the LDF pool, so the recalibrators take it directly)
    cdfs (1, Tf, M), pdfs (1, Tf, M), pits (1, Tf), log_pd (1, Tf)  over the forecast window
    index (Tf,)  0-based period of each forecast (start .. T-1)
Pass y[index] (and the matching dates) to the recalibration functions together with these.

SPEED
The sampler is inherently sequential (one chain; forward filter and backward sampler are loops
over time). The per-time-step algebra is done on 5x5 arrays with numpy. Forecast origins are
independent, so run_bps runs them in parallel processes (n_jobs). Use smaller burn_in /
mcmc_iter for quick tests.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["BPS", "bps_predictive", "run_bps", "matlab_prior"]


def matlab_prior(J, delta=(0.95, 0.99)):
    """Prior of the MATLAB driver: m_0 = [0, 1/J...], C_0 = I, n_0 = 1/(1-delta(2)), s_0 = 0.01."""
    return dict(m_0=np.r_[0.0, np.full(J, 1.0 / J)], C_0=np.eye(J + 1),
                n_0=1.0 / (1.0 - delta[1]), s_0=0.01)


def _sym(x):
    return 0.5 * (x + x.T)


def _chol(x):
    x = _sym(x)
    try:
        return np.linalg.cholesky(x)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(x)
        return V * np.sqrt(np.clip(w, 0.0, None))     # a valid square root of the PSD part


def _ig(rng, shape, scale):
    """Inverse gamma draw as written in the MATLAB: scale / randg(shape)."""
    return scale / rng.gamma(shape)


# =========================================================================== #
#  BPS.m
# =========================================================================== #
def BPS(y, a_j, A_j, n_j, delta, m_0, C_0, n_0, s_0, burn_in, mcmc_iter, rng=None,
        exact_port=True):
    """Port of BPS.m. y (T,), a_j / A_j / n_j (T, J) agent t means, variances, dof.
    Returns a_k (mcmc_iter, p), R_k (mcmc_iter, p, p), v_k (mcmc_iter,), n_k (scalar),
    theta_post (mcmc_iter, T, p), X_post (mcmc_iter, T, J) -- post burn-in."""
    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(y, float); a_j = np.asarray(a_j, float)
    A_j = np.asarray(A_j, float); n_j = np.asarray(n_j, float)
    total = burn_in + mcmc_iter
    T, px = a_j.shape
    p = px + 1
    d, beta = float(delta[0]), float(delta[1])

    m_t = np.zeros((T + 1, p)); C_t = np.zeros((T + 1, p, p))
    n_t = np.zeros(T + 1); s_t = np.zeros(T + 1)
    a_t = np.zeros((T, p)); R_t = np.zeros((T, p, p))
    m_t[0], C_t[0], n_t[0] = m_0, C_0, n_0

    a_k = np.zeros((total, p)); R_k = np.zeros((total, p, p)); v_k = np.zeros(total)
    theta_post = np.zeros((mcmc_iter, T, p)); X_post = np.zeros((mcmc_iter, T, px))

    # ---- initial latent agent states ------------------------------------------------- #
    if exact_port:
        phi = 0.5 * beta * n_j[:T] / rng.gamma(beta * n_j[:T] / 2.0)
    else:
        phi = 0.5 * n_j[:T] / rng.gamma(n_j[:T] / 2.0)
    X = a_j[:T] + rng.standard_normal((T, px)) * np.sqrt(phi * A_j[:T])

    theta = np.zeros((T, p)); v = np.zeros(T)
    for i in range(total):
        s_t[0] = s_0
        # ---- forward filter ------------------------------------------------------------ #
        for t in range(T):
            F = np.r_[1.0, X[t]]
            a_t[t] = m_t[t]
            R_t[t] = C_t[t] / d
            f = F @ a_t[t]
            RF = R_t[t] @ F
            q = F @ RF + s_t[t]
            e = y[t] - f
            A = RF / q
            n_t[t + 1] = beta * n_t[t] + 1.0
            r = (beta * n_t[t] + e * e / q) / n_t[t + 1]
            s_t[t + 1] = r * s_t[t]
            m_t[t + 1] = a_t[t] + A * e
            C_t[t + 1] = _sym(r * (R_t[t] - q * np.outer(A, A)))
        # ---- sample theta at T ------------------------------------------------------------ #
        v[T - 1] = 1.0 / rng.gamma(n_t[T] / 2.0, 2.0 / (n_t[T] * s_t[T]))
        theta[T - 1] = m_t[T] + _chol(C_t[T] * (v[T - 1] / s_t[T])) @ rng.standard_normal(p)
        # ---- theta at T+1 ----------------------------------------------------------------- #
        n_k = beta * n_t[T] + 1.0
        v_k[i] = 1.0 / rng.gamma(beta * n_t[T] / 2.0, 2.0 / (beta * n_t[T] * s_t[T]))
        a_k[i] = m_t[T]
        R_k[i] = (C_t[T] / d) * (v_k[i] / s_t[T])
        # ---- backward sampler ------------------------------------------------------------- #
        for t in range(T - 2, -1, -1):
            v[t] = 1.0 / (beta / v[t + 1]
                          + rng.gamma((1.0 - beta) * n_t[t + 1] / 2.0, 2.0 / (n_t[t + 1] * s_t[t + 1])))
            m_star = m_t[t + 1] + d * (theta[t + 1] - a_t[t + 1])
            C_star = C_t[t + 1] * (1.0 - d) * (v[t] / s_t[t + 1])
            theta[t] = m_star + _chol(C_star) @ rng.standard_normal(p)
        # ---- sample X_t (independent over t given theta, v: vectorised) ---------------------- #
        Ad = phi * A_j[:T]                                    # diag of A_st, (T, px)
        th1, thp = theta[:, 0], theta[:, 1:]
        g = v + np.einsum("tj,tj->t", thp * thp, Ad)
        sigma = thp * Ad / g[:, None]                         # (T, px)
        a_star = a_j[:T] + sigma * (y - (th1 + np.einsum("tj,tj->t", thp, a_j[:T])))[:, None]
        # A_star = diag(Ad) - diag(Ad) thp' sigma  (rank one); draw with a per-t Cholesky
        A_star = (np.einsum("tj,jk->tjk", Ad, np.eye(px))
                  - np.einsum("tj,tk->tjk", Ad * thp, sigma))
        A_star = 0.5 * (A_star + np.transpose(A_star, (0, 2, 1)))
        try:
            L = np.linalg.cholesky(A_star)
        except np.linalg.LinAlgError:
            L = np.stack([_chol(A_star[t]) for t in range(T)])
        # MATLAB: a_star + randn(1,J) * chol(A_star) with chol upper-triangular U, U'U = A_star;
        # z U has covariance U'U = A_star, the same as L z with L = U'.
        X = a_star + np.einsum("tjk,tk->tj", L, rng.standard_normal((T, px)))
        dd = (X - a_j[:T]) ** 2 / A_j[:T]
        if exact_port:
            phi = 0.5 * (n_j[:T] + 1.0) / rng.gamma((n_j[:T] + dd) / 2.0)
        else:
            phi = 0.5 * (n_j[:T] + dd) / rng.gamma((n_j[:T] + 1.0) / 2.0)
        if i >= burn_in:
            theta_post[i - burn_in] = theta
            X_post[i - burn_in] = X

    return (a_k[burn_in:], R_k[burn_in:], v_k[burn_in:], float(n_k), theta_post, X_post)


# =========================================================================== #
#  driver loop body: one-step predictive of y_{t+1}
# =========================================================================== #
def bps_predictive(a_k, R_k, v_k, n_k, a_next, A_next, n_next, delta, x_grid, y_next=None,
                   rng=None, exact_port=True):
    """For each MCMC draw i: draw x_{t+1} from the agents, E_i = x a_k_i, V_i = x R_k_i x' + v_k_i;
    the predictive is the mixture over i of t_{n_k}(E_i, V_i) (V_i the squared scale, as in
    the driver's `mlike`). Returns cdf / pdf on x_grid, and pit / log_pd / mean at y_next."""
    rng = np.random.default_rng() if rng is None else rng
    a_next = np.asarray(a_next, float); A_next = np.asarray(A_next, float)
    n_next = np.asarray(n_next, float)
    S, p = a_k.shape
    J = p - 1
    beta = float(delta[1])
    z = rng.standard_normal((S, J))
    if exact_port:
        lam = np.sqrt(0.5 * beta * n_next[0] / rng.gamma(beta * n_next[0] / 2.0, size=S))[:, None]
    else:
        lam = np.sqrt(0.5 * n_next / rng.gamma(n_next / 2.0, size=(S, J)))
    x = np.concatenate([np.ones((S, 1)), a_next + lam * z * np.sqrt(A_next)], axis=1)
    E = np.einsum("si,si->s", x, a_k)
    V = np.einsum("si,sij,sj->s", x, R_k, x) + v_k
    sd = np.sqrt(V)
    xg = np.asarray(x_grid, float)
    zz = (xg[None, :] - E[:, None]) / sd[:, None]
    out = dict(cdf=stats.t.cdf(zz, n_k).mean(0), pdf=(stats.t.pdf(zz, n_k) / sd[:, None]).mean(0),
               mean=float(E.mean()), var=float(V.mean() * n_k / (n_k - 2) + E.var()) if n_k > 2 else np.nan)
    if y_next is not None:
        zy = (float(y_next) - E) / sd
        out["pit"] = float(stats.t.cdf(zy, n_k).mean())
        out["log_pd"] = float(np.log(np.mean(stats.t.pdf(zy, n_k) / sd)))   # = log mean(mlike)
        out["error"] = float(y_next - E.mean())
    return out


def _one_origin(s, y, a, A, n, x_grid, delta, prior, burn_in, mcmc_iter, seed, exact_port):
    rng = np.random.default_rng(seed)
    a_k, R_k, v_k, n_k, theta_post, _ = BPS(y[:s], a[:s], A[:s], n[:s], delta, prior["m_0"],
                                          prior["C_0"], prior["n_0"], prior["s_0"],
                                          burn_in, mcmc_iter, rng=rng, exact_port=exact_port)
    pred = bps_predictive(a_k, R_k, v_k, n_k, a[s], A[s], n[s], delta, x_grid, y[s], rng=rng,
                          exact_port=exact_port)
    pred.update(s=s, a_k_mean=a_k.mean(0), n_k=n_k, theta_last=theta_post[:, -1].mean(0))
    return pred


# =========================================================================== #
#  driver script
# =========================================================================== #
def run_bps(y, agent_mean, agent_var, agent_dof, x_grid, start=50, delta=(0.95, 0.99),
            prior=None, burn_in=3000, mcmc_iter=5000, seed=42, n_jobs=1, verbose=10,
            exact_port=True, label=None):
    """Driver script: for s = start..T-1 run BPS on y[:s] and form the predictive of y[s].

    y (T,); agent_mean / agent_var / agent_dof (T, J) -- the 'Agent Mean', 'Agent Var',
    'Agent dof' sheets. start=50 is the MATLAB `for t=50:T-1`.

    Returns dict in the LDF-pool format over the forecast window (Tf = T - start periods):
        cdfs (1,Tf,M), pdfs (1,Tf,M), pits (1,Tf), log_pd (1,Tf), mean (1,Tf), error (1,Tf),
        a_k (Tf, J+1)  posterior mean of the synthesis coefficients for the forecast,
        index (Tf,)    0-based period of each forecast, labels [label], x_grid, settings."""
    y = np.asarray(y, float).ravel()
    a = np.asarray(agent_mean, float); A = np.asarray(agent_var, float)
    n = np.asarray(agent_dof, float)
    T, J = a.shape
    prior = matlab_prior(J, delta) if prior is None else prior
    x_grid = np.asarray(x_grid, float)
    origins = list(range(int(start), T))
    if not origins or origins[0] < 2:
        raise ValueError(f"start={start}: BPS needs at least 2 periods of data before its first forecast "
                         f"(T={T}); the MATLAB driver uses start=50")
    args = dict(y=y, a=a, A=A, n=n, x_grid=x_grid, delta=delta, prior=prior, burn_in=burn_in,
                mcmc_iter=mcmc_iter, exact_port=exact_port)
    if n_jobs == 1:
        res = [_one_origin(s, seed=seed + s, **args) for s in origins]
    else:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_one_origin)(s, seed=seed + s, **args) for s in origins)
    res.sort(key=lambda r: r["s"])
    Tf, M = len(res), x_grid.size
    out = dict(cdfs=np.zeros((1, Tf, M)), pdfs=np.zeros((1, Tf, M)), pits=np.zeros((1, Tf)),
               log_pd=np.zeros((1, Tf)), mean=np.zeros((1, Tf)), error=np.zeros((1, Tf)),
               a_k=np.zeros((Tf, J + 1)), index=np.array(origins),
               labels=[float(delta[0]) if label is None else float(label)],
               x_grid=x_grid, delta=tuple(delta), prior=prior, start=start,
               burn_in=burn_in, mcmc_iter=mcmc_iter, exact_port=exact_port)
    for k, r in enumerate(res):
        out["cdfs"][0, k] = np.maximum.accumulate(np.clip(r["cdf"], 0.0, 1.0))
        out["pdfs"][0, k] = r["pdf"]
        out["pits"][0, k] = r["pit"]; out["log_pd"][0, k] = r["log_pd"]
        out["mean"][0, k] = r["mean"]; out["error"][0, k] = r["error"]
        out["a_k"][k] = r["a_k_mean"]
    mass = out["cdfs"][0, :, -1] - out["cdfs"][0, :, 0]
    out["grid_mass"] = mass
    if mass.min() < 0.995:
        print(f"[bps] {np.sum(mass < 0.995)} forecast(s) hold < 99.5% of their mass on x_grid "
              f"(min {mass.min():.3f}); widen x_grid")
    return out