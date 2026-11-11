import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox

_EPS = 1e-10


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _ad_uniform_statistic(u):
    """Anderson-Darling A^2 against a fully specified U(0,1), vectorised over rows.

    No small-sample modification is applied: the (1 + 0.75/n + 2.25/n^2)
    factor is Stephens' correction for testing normality with *estimated*
    mean and variance, and does not apply to a fully specified null.
    """
    u = np.clip(np.sort(np.atleast_2d(u), axis=-1), _EPS, 1 - _EPS)
    n = u.shape[-1]
    i = np.arange(1, n + 1)
    s = np.sum((2 * i - 1) * (np.log(u) + np.log1p(-u[:, ::-1])), axis=-1)
    return -n - s / n


def _ad_uniform_null(T, n_sim, rng, chunk=2000):
    """Sorted Monte Carlo null distribution of A^2 for i.i.d. U(0,1) of size T
    (the paper uses simulated critical values for its sample size)."""
    out = np.empty(n_sim)
    for start in range(0, n_sim, chunk):
        m = min(chunk, n_sim - start)
        out[start:start + m] = _ad_uniform_statistic(rng.uniform(size=(m, T)))
    return np.sort(out)


def _mc_pvalue(stat, null_sorted):
    """Upper-tail Monte Carlo p-value, (1 + #{null >= stat}) / (n_sim + 1)."""
    n_ge = null_sorted.size - np.searchsorted(null_sorted, stat, side="left")
    return (1 + n_ge) / (null_sorted.size + 1)


def _doornik_hansen(z):
    """Univariate Doornik-Hansen (2008) omnibus normality test; ~ chi2(2)."""
    z = np.asarray(z, dtype=float)
    n = z.size
    d = z - z.mean()
    m2, m3, m4 = np.mean(d**2), np.mean(d**3), np.mean(d**4)
    sqrt_b1 = m3 / m2**1.5
    b1 = sqrt_b1**2
    b2 = m4 / m2**2

    # Skewness: D'Agostino (1970) transformation
    beta = 3 * (n**2 + 27 * n - 70) * (n + 1) * (n + 3) / (
        (n - 2) * (n + 5) * (n + 7) * (n + 9))
    w2 = -1 + np.sqrt(2 * (beta - 1))
    delta = 1 / np.sqrt(np.log(np.sqrt(w2)))
    y = sqrt_b1 * np.sqrt((w2 - 1) * (n + 1) * (n + 3) / (12 * (n - 2)))
    z1 = delta * np.arcsinh(y)

    # Kurtosis: gamma approximation + Wilson-Hilferty cube-root transform
    dk = (n - 3) * (n + 1) * (n**2 + 15 * n - 4)
    a = (n - 2) * (n + 5) * (n + 7) * (n**2 + 27 * n - 70) / (6 * dk)
    c = (n - 7) * (n + 5) * (n + 7) * (n**2 + 2 * n - 5) / (6 * dk)
    k = (n + 5) * (n + 7) * (n**3 + 37 * n**2 + 11 * n - 313) / (12 * dk)
    alpha = a + b1 * c
    chi = 2 * k * (b2 - 1 - b1)          # >= 0 by Pearson's inequality
    z2 = (np.cbrt(chi / (2 * alpha)) - 1 + 1 / (9 * alpha)) * np.sqrt(9 * alpha)

    stat = z1**2 + z2**2
    return float(stat), float(stats.chi2.sf(stat, df=2))


def _ar1_exact_loglik(z, mu, sigma2, rho):
    """Exact Gaussian log-likelihood of z_t - mu = rho (z_{t-1} - mu) + e_t,
    e_t ~ N(0, sigma2): equation (2) of the paper / Berkowitz (2001)."""
    rho = np.clip(rho, -1 + 1e-8, 1 - 1e-8)
    v1 = sigma2 / (1 - rho**2)
    e = (z[1:] - mu) - rho * (z[:-1] - mu)
    return (-0.5 * np.log(2 * np.pi * v1) - 0.5 * (z[0] - mu) ** 2 / v1
            - 0.5 * (z.size - 1) * np.log(2 * np.pi * sigma2)
            - 0.5 * np.sum(e**2) / sigma2)


def _berkowitz(z):
    """Berkowitz (2001) LR tests on z_t = Phi^{-1}(p_t).

    LR3 : H0 (mu, sigma^2, rho) = (0, 1, 0), 3 df   -> joint fit + independence
    LR1 : H0 rho = 0,                        1 df   -> independence alone
    Also returns KLIC_hat = (logL_hat - logL_0) / T = LR3 / (2T), the
    Mitchell-Hall / Bao et al. KLIC estimate with h(.) the Gaussian AR(1).
    """
    T = z.size
    ll_0 = np.sum(norm.logpdf(z))                          # N(0,1), i.i.d.
    mu_r, s_r = z.mean(), z.std()                          # MLE under rho = 0
    ll_r = np.sum(norm.logpdf(z, loc=mu_r, scale=s_r))

    # Exact MLE of the unrestricted model; two starts (OLS and rho = 0)
    X = np.column_stack([np.ones(T - 1), z[:-1]])
    beta = np.linalg.lstsq(X, z[1:], rcond=None)[0]
    rho_ols = float(np.clip(beta[1], -0.99, 0.99))
    s2_ols = float(np.mean((z[1:] - X @ beta) ** 2))
    starts = [
        [beta[0] / (1 - rho_ols), 0.5 * np.log(max(s2_ols, 1e-8)), np.arctanh(rho_ols)],
        [mu_r, np.log(max(s_r, 1e-8)), 0.0],
    ]

    def nll(th):
        return -_ar1_exact_loglik(z, th[0], np.exp(2 * th[1]), np.tanh(th[2]))

    best = min((minimize(nll, x0, method="BFGS") for x0 in starts), key=lambda r: r.fun)
    ll_u = max(-best.fun, ll_r)        # guard against optimiser shortfall

    lr3 = 2 * (ll_u - ll_0)
    lr1 = 2 * (ll_u - ll_r)
    return {
        "mu": float(best.x[0]),
        "sigma2": float(np.exp(2 * best.x[1])),
        "rho": float(np.tanh(best.x[2])),
        "LR3": float(lr3), "LR3_pvalue": float(stats.chi2.sf(lr3, df=3)),
        "LR1": float(lr1), "LR1_pvalue": float(stats.chi2.sf(lr1, df=1)),
        "KLIC": float((ll_u - ll_0) / T),
    }


def _efficiency_lr(z, W):
    """Density forecast efficiency test (paper, Sec. 2.3 and Table VI).

    Regression (2) in intercept form, z_t = c + rho z_{t-1} + W_t' gamma + e_t,
    Gaussian LR test of gamma = 0 (k df). Row t of W must be information
    available when the forecast of x_t was made. Returns NaNs ("n.a." in
    Table VI) when W_t is perfectly collinear with (1, z_{t-1}).
    """
    W = np.asarray(W, dtype=float)
    if W.ndim == 1:
        W = W[:, None]
    y, zlag, Wt = z[1:], z[:-1], W[1:]
    ok = np.isfinite(Wt).all(axis=1)
    y, zlag, Wt = y[ok], zlag[ok], Wt[ok]
    n, k = Wt.shape
    Xr = np.column_stack([np.ones(n), zlag])
    Xu = np.column_stack([Xr, Wt])
    if n <= Xu.shape[1] + 1:
        return np.nan, np.nan, int(k)
    Xs = Xu / np.linalg.norm(Xu, axis=0)
    sv = np.linalg.svd(Xs, compute_uv=False)
    if sv.min() / sv.max() < 1e-8:
        return np.nan, np.nan, int(k)

    def rss(X):
        return np.sum((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2)

    lr = n * np.log(rss(Xr) / rss(Xu))
    return float(lr), float(stats.chi2.sf(lr, df=k)), int(k)


# ----------------------------------------------------------------------------
# Main function
# ----------------------------------------------------------------------------
def uniformity_test(
    pit_values: np.ndarray,
    discount_factors,
    model_names=None,
    alpha: float = 0.05,
    bonferroni: bool = False,
    lb_lags: int = 4,
    info_set= None,
    n_sim: int = 10_000,
    seed=None,
):
    """
    Parameters
    ----------
    pit_values : (P, T) array of PITs, one row per model.
    discount_factors : used only to build default model names.
    model_names : optional list of length P.
    alpha : nominal size of each test.
    bonferroni : if True, each test is run at alpha / n_tests.
    lb_lags : Ljung-Box maximum lag (paper uses 4).
    info_set : optional W for the efficiency test; shape (T,), (T, k) shared by
        all models, or (P, T, k) model-specific. Row t = information available
        when forecasting x_t (e.g. [y_{t-1}, y_{t-2}]); NaN rows are dropped.
    n_sim : Monte Carlo replications for the AD null distribution.
    seed : RNG seed for the AD simulation.
    """
    pit_values = np.atleast_2d(np.asarray(pit_values, dtype=float))
    P, T = pit_values.shape
    if model_names is None:
        model_names = [f"δ = {i}" for i in discount_factors]
    if len(model_names) != P:
        raise ValueError("Length of model_names must match number of rows in pit_values (P).")
    if not np.all(np.isfinite(pit_values)):
        raise ValueError("pit_values contains NaN or inf.")
    if np.any((pit_values < 0) | (pit_values > 1)):
        raise ValueError("pit_values must lie in [0, 1].")
    if T < 10:
        raise ValueError("Need at least T = 10 observations per model.")

    if info_set is not None:
        info_set = np.asarray(info_set, dtype=float)
        if info_set.shape[0 if info_set.ndim < 3 else 1] != T:
            raise ValueError("info_set must have T rows aligned with pit_values columns.")
        if info_set.ndim == 3 and info_set.shape[0] != P:
            raise ValueError("3-D info_set must have shape (P, T, k).")

    # Tests counted toward the pass/fail tally:
    # KS, AD, Chi2, DH, LB, Berkowitz, Mean, Variance, Cov5, Cov95 (+ Efficiency)
    n_tests = 10 + (info_set is not None)
    eff_alpha = alpha / n_tests if bonferroni else alpha

    # Quantities that depend only on T: computed once, outside the loop
    rng = np.random.default_rng(seed)
    ad_null = _ad_uniform_null(T, n_sim, rng)
    n_bins = 10 #max(5, min(30, T // 5))        # expected count >= 5 per bin when T >= 25
    lb_lag = int(max(1, min(lb_lags, T // 2)))

    results = []
    interpretation = []

    for i in range(P):
        pit_p = pit_values[i, :]
        z = norm.ppf(np.clip(pit_p, _EPS, 1 - _EPS))   # inverse normal transform

        # ------------------------------------------------------------------
        # 1. KS test against U(0,1) (scipy uses the exact finite-T null)
        # ------------------------------------------------------------------
        ks_stat, ks_pvalue = stats.kstest(pit_p, "uniform")

        # ------------------------------------------------------------------
        # 2. Anderson-Darling against a fully specified U(0,1),
        #    p-value from the simulated finite-T null distribution
        # ------------------------------------------------------------------
        ad_stat = float(_ad_uniform_statistic(pit_p)[0])
        ad_pvalue = _mc_pvalue(ad_stat, ad_null)

        # ------------------------------------------------------------------
        # 3. Pearson chi-square on the PIT histogram, df = n_bins - 1
        # ------------------------------------------------------------------
        observed_freq, _ = np.histogram(pit_p, bins=n_bins, range=(0, 1))
        chi2_stat, chi2_pvalue = stats.chisquare(observed_freq)

        # ------------------------------------------------------------------
        # 4. Doornik-Hansen normality test on z_t
        # ------------------------------------------------------------------
        dh_stat, dh_pvalue = _doornik_hansen(z)

        # ------------------------------------------------------------------
        # 5. Ljung-Box on the PITs (independence), lag 4 as in the paper
        # ------------------------------------------------------------------
        lb_result = acorr_ljungbox(pit_p, lags=[lb_lag])
        lb_stat = float(lb_result["lb_stat"].iloc[-1])
        lb_pvalue = float(lb_result["lb_pvalue"].iloc[-1])

        # ------------------------------------------------------------------
        # 6. Berkowitz LR tests on z_t (+ KLIC estimate)
        # ------------------------------------------------------------------
        bk = _berkowitz(z)

        # ------------------------------------------------------------------
        # 7. Moments of the PITs vs U(0,1): E = 1/2, Var = 1/12,
        #    asymptotic Var(mean) = 1/(12T), Var(var) = (mu4 - sigma^4)/T = 1/(180T)
        # ------------------------------------------------------------------
        mean_pit = float(np.mean(pit_p))
        var_pit = float(np.var(pit_p))
        mean_z = (mean_pit - 0.5) / np.sqrt(1 / (12 * T))
        var_z = (var_pit - 1 / 12) / np.sqrt(1 / (180 * T))
        mean_pvalue = float(2 * norm.sf(abs(mean_z)))
        var_pvalue = float(2 * norm.sf(abs(var_z)))

        # ------------------------------------------------------------------
        # 8. Tail coverage, exact two-sided binomial tests
        # ------------------------------------------------------------------
        n05, n95 = int(np.sum(pit_p <= 0.05)), int(np.sum(pit_p <= 0.95))
        coverage_05, coverage_95 = n05 / T, n95 / T
        coverage_10 = float(np.mean(pit_p <= 0.10))
        coverage_90 = float(np.mean(pit_p <= 0.90))
        cov05_pvalue = float(stats.binomtest(n05, T, 0.05).pvalue)
        cov95_pvalue = float(stats.binomtest(n95, T, 0.95).pvalue)

        # ------------------------------------------------------------------
        # 9. Efficiency test (optional)
        # ------------------------------------------------------------------
        if info_set is not None:
            W_i = info_set[i] if info_set.ndim == 3 else info_set
            eff_stat, eff_pvalue, eff_df = _efficiency_lr(z, W_i)
        else:
            eff_stat, eff_pvalue, eff_df = np.nan, np.nan, 0

        results.append({
            "Model": model_names[i],
            "Count": int(T),
            "KS_statistic": float(ks_stat),
            "KS_pvalue": float(ks_pvalue),
            "AD_statistic": ad_stat,
            "AD_pvalue": float(ad_pvalue),
            "Chi2_statistic": float(chi2_stat),
            "Chi2_pvalue": float(chi2_pvalue),
            "DH_statistic": dh_stat,
            "DH_pvalue": dh_pvalue,
            "LB_statistic": lb_stat,
            "LB_pvalue": lb_pvalue,
            "Bk_LR3_statistic": bk["LR3"],
            "Bk_LR3_pvalue": bk["LR3_pvalue"],
            "Bk_rho": bk["rho"],
            "Bk_LR1_statistic": bk["LR1"],
            "Bk_LR1_pvalue": bk["LR1_pvalue"],
            "Bk_mu": bk["mu"],
            "Bk_sigma2": bk["sigma2"],
            "KLIC": bk["KLIC"],
            "Eff_statistic": eff_stat,
            "Eff_df": eff_df,
            "Eff_pvalue": eff_pvalue,
            "Mean": mean_pit,
            "Mean_pvalue": mean_pvalue,
            "Variance": var_pit,
            "Variance_pvalue": var_pvalue,
            "Skewness": float(stats.skew(pit_p)),
            "Kurtosis": float(stats.kurtosis(pit_p)),
            "Coverage_5%": float(coverage_05),
            "Coverage_5%_pvalue": cov05_pvalue,
            "Coverage_10%": coverage_10,
            "Coverage_90%": coverage_90,
            "Coverage_95%": float(coverage_95),
            "Coverage_95%_pvalue": cov95_pvalue,
        })

        # ------------------------------------------------------------------
        # Pass/fail (all at eff_alpha)
        # ------------------------------------------------------------------
        ks_good = ks_pvalue > eff_alpha
        ad_good = ad_pvalue > eff_alpha
        chi2_good = chi2_pvalue > eff_alpha
        dh_good = dh_pvalue > eff_alpha
        lb_good = lb_pvalue > eff_alpha
        bk_good = bk["LR3_pvalue"] > eff_alpha
        mean_good = mean_pvalue > eff_alpha
        var_good = var_pvalue > eff_alpha
        coverage_5_good = cov05_pvalue > eff_alpha
        coverage_95_good = cov95_pvalue > eff_alpha

        flags = [ks_good, ad_good, chi2_good, dh_good, lb_good, bk_good,
                 mean_good, var_good, coverage_5_good, coverage_95_good]
        eff_applicable = info_set is not None and np.isfinite(eff_pvalue)
        if eff_applicable:
            eff_good = eff_pvalue > eff_alpha
            flags.append(eff_good)
        n_applicable = n_tests - (info_set is not None and not eff_applicable)

        uniform_ok = ks_good and ad_good and chi2_good and dh_good
        indep_ok = lb_good and bk_good

        def pf(flag):
            return "PASS" if flag else "FAIL"

        interpretation.append({
            "Model": model_names[i],
            "Tests_Passed": f"{sum(flags)}/{n_applicable}",
            "KS_Test": pf(ks_good),
            "AD_Test": pf(ad_good),
            "Chi2_Test": pf(chi2_good),
            "DH_Test": pf(dh_good),
            "LB_Test": pf(lb_good),
            "Berkowitz_Test": pf(bk_good),
            "Mean_Test": pf(mean_good),
            "Variance_Test": pf(var_good),
            "Coverage_5%_Test": pf(coverage_5_good),
            "Coverage_95%_Test": pf(coverage_95_good),
            "Efficiency_Test": (pf(eff_good) if eff_applicable else "N/A"),
            # Paper's terminology: probabilistic calibration = uniform PITs;
            # complete calibration = uniform AND independent PITs.
            "Probabilistic_Calibration": "Not rejected" if uniform_ok else "Rejected",
            "Complete_Calibration": "Not rejected" if (uniform_ok and indep_ok) else "Rejected",
        })

    tests_df = pd.DataFrame(results)
    interpretation_df = pd.DataFrame(interpretation)

    return tests_df, interpretation_df