import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.optim as optim
from rational_quadratic import rational_quadratic_spline
from nn import ResidualNet
from torch.nn import functional as F
from scipy.stats import kstest, anderson, jarque_bera, gaussian_kde, norm, chi2
import math
from statsmodels.stats.diagnostic import acorr_ljungbox


#####################################
# PIT function
#####################################

def pit_func(pit_vals: np.ndarray, x: np.ndarray,y: np.ndarray):
    P, T, M = pit_vals.shape

    pit_values = np.zeros((P, T))
    for p in range(P):
        for t in range(T):
            interp_func = interpolate.interp1d(x, pit_vals[p, t, :],
                                             kind='linear',
                                             bounds_error=False,
                                             fill_value=1e-10)
            pit_values[p, t] = interp_func(y[t])

    return pit_values


#####################################
# exchange rate linear pool pdf function
#####################################


def ex_tlp_pdf(
        mean: np.ndarray,          # Shape (K, T) - forecast means
        var: np.ndarray,           # Shape (K, T) - forecast variances
        x: np.ndarray,             # Grid points
        weights: np.ndarray,       # Shape (P, T, K) - combination weights
        y: np.ndarray              # Observed values (T,)
) -> np.ndarray:

    K, T = mean.shape
    P, T_w, K_w = weights.shape
    M = x.shape[0]

    assert (T, K) == (T_w, K_w), 'Shape mismatch'
    assert mean.shape == var.shape

    pdf_values = np.zeros((K, T, M), dtype=float)

    for k in range(K):
        for t in range(T):
            # PDF instead of CDF
            pdf_values[k, t, :] = norm.pdf(x, loc= mean[k, t], scale=np.sqrt(var[k, t]) )

    # Linear pool: weighted combination of PDFs
    TL = np.einsum('ktm,ptk -> ptm', pdf_values, weights)

    pdf = pit_func(TL, x, y)  # Shape (T,)

    return pdf


#####################################
# exchange rate calibrated pdf function
#####################################


def ex_calibrated_ll(
        log_original_pdf: np.ndarray,
        results: list,
        discount_factors: np.ndarray,
):
    calibrated_ll = np.zeros_like(log_original_pdf)
    for i in range(len(discount_factors)):
        absdet_i = results[i]['log_likelihood']
        calibrated_ll[i] = log_original_pdf[i,:] + absdet_i
    return calibrated_ll




#####################################
# Plot PIT histograms
#####################################


def plot_pit_histograms(TLP: np.ndarray, model_names=None, show_smooth: bool = False):
 
    P, T = TLP.shape
 
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(P)]
 
    # Create 4x3 grid (12 subplots total)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12),dpi=300)
 
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
 
    for p in range(P):
        ax = axes_flat[p]
        pits = TLP[p, :]
 
        # Create histogram
        ax.hist(pits, bins=20, density=True, alpha=0.7)
 
        # Smooth density curve (KDE) on top of the bins
        if show_smooth:
            kde = gaussian_kde(pits)
            x_grid = np.linspace(0, 1, 200)
            ax.plot(x_grid, kde(x_grid), color='C1', linewidth=2, label='KDE')
 
        # Add uniform reference line
        ax.axhline(1.0, color='k', linestyle='--', linewidth=1, label='Uniform', alpha=0.8)
 
        # Set labels and title
        ax.set_xlim(0, 1)
        ax.set_xlabel('PIT Values', fontsize=18)
        ax.set_ylabel('Density', fontsize=18)
        ax.tick_params(labelsize=16)  # This sets font size for both x and y axes
        ax.set_title(f'δ = {model_names[p]}', fontsize=18, pad=10)
        ax.grid(True, alpha=0.7)
 
    plt.tight_layout()
    plt.show()

######################################################

def _ad_uniform_pvalue(ad_stat_corrected: float) -> float:
    """
    Approximate p-value for the Anderson-Darling statistic against a fully
    specified Uniform(0,1) (Stephens 1974 'Case 0' critical values -- no
    parameters estimated from the data, which is the correct table for PIT
    uniformity testing). Instead of snapping to the 5 tabulated levels,
    log-linearly interpolate/extrapolate between them so the result is a
    continuous, monotonic approximation rather than a 5-value step function.
    Still an approximation (the exact asymptotic CDF of A^2 is an infinite
    series -- see Marsaglia & Marsaglia 2004), but far more usable for
    ranking/comparing series than the discrete bucket version.
    """
    table_alpha = np.array([0.10, 0.05, 0.025, 0.01])
    table_cv = np.array([1.933, 2.492, 3.070, 3.857])
    log_a = np.log(table_alpha)

    if ad_stat_corrected <= table_cv[0]:
        slope = (log_a[1] - log_a[0]) / (table_cv[1] - table_cv[0])
        log_p = log_a[0] + slope * (ad_stat_corrected - table_cv[0])
    elif ad_stat_corrected >= table_cv[-1]:
        slope = (log_a[-1] - log_a[-2]) / (table_cv[-1] - table_cv[-2])
        log_p = log_a[-1] + slope * (ad_stat_corrected - table_cv[-1])
    else:
        log_p = np.interp(ad_stat_corrected, table_cv, log_a)

    return float(np.clip(np.exp(log_p), 1e-6, 1.0))


def _kl_uniform_critical_value(T: int, n_bins: int, alpha: float, n_sims: int = 2000, seed: int = 0) -> float:
    """
    Data-driven critical value for the (clipped) KL-divergence-from-uniform
    statistic, simulated under the null (T iid Uniform(0,1) draws binned into
    n_bins), rather than a fixed magic-number threshold. Replaces the old
    hardcoded `kl_divergence < 0.02` rule, so the test is calibrated to the
    actual sample size and bin count instead of being arbitrary.
    """
    rng = np.random.default_rng(seed)
    eps = 1e-10
    uniform_prob = np.ones(n_bins) / n_bins
    kls = np.empty(n_sims)
    for s in range(n_sims):
        u = rng.uniform(size=T)
        freq, _ = np.histogram(u, bins=n_bins, range=(0, 1))
        p = np.maximum(freq / T, eps)
        kls[s] = np.sum(p * np.log(p / uniform_prob))
    return float(np.quantile(kls, 1 - alpha))


def test_and_interpret_pit(
    pit_values: np.ndarray,
    discount_factors,
    model_names=None,
    alpha: float = 0.05,
    bonferroni: bool = False,
):
    P, T = pit_values.shape
    if model_names is None:
        model_names = [f"δ = {i}" for i in discount_factors]
    if len(model_names) != P:
        raise ValueError("Length of model_names must match number of rows in pit_values (P).")

    # Effective alpha for the individual pass/fail flags. Running 9 tests at
    # the nominal alpha means a well-calibrated model will still "FAIL" one
    # test roughly 1 - 0.95^9 ≈ 37% of the time by chance. Bonferroni gives a
    # (conservative) per-test alpha that keeps the family-wise error rate
    # near the nominal level; off by default to match the original behaviour.
    n_tests = 9
    eff_alpha = alpha / n_tests if bonferroni else alpha
    zcrit = norm.ppf(1 - eff_alpha / 2)

    # KL critical value depends only on (T, n_bins), which are the same for
    # every row here, so simulate it once outside the loop rather than P times.
    n_bins_kl = max(5, min(30, T // 3))
    kl_crit = _kl_uniform_critical_value(T, n_bins_kl, eff_alpha)

    results = []
    interpretation = []

    for i in range(P):
        pit_p = pit_values[i, :]

        # ------------------------------------------------------------------
        # 1. KS test against Uniform(0,1)
        # ------------------------------------------------------------------
        ks_stat, ks_pvalue = stats.kstest(pit_p, 'uniform')

        # ------------------------------------------------------------------
        # 2. Anderson-Darling test directly against Uniform(0,1)
        #    FIX: clip to (eps, 1-eps) before taking logs -- PIT values that
        #    land exactly at 0 or 1 (e.g. from an interpolator's boundary
        #    fill value) previously sent this statistic to -inf/NaN.
        #    FIX: continuous p-value via _ad_uniform_pvalue instead of a
        #    5-level step function.
        # ------------------------------------------------------------------
        eps = 1e-10
        u_sorted = np.clip(np.sort(pit_p), eps, 1 - eps)
        n = len(u_sorted)
        idx = np.arange(1, n + 1)
        ad_stat = (
            -n
            - (1.0 / n) * np.sum(
                (2 * idx - 1) * (np.log(u_sorted) + np.log(1 - u_sorted[::-1]))
            )
        )
        ad_stat_corrected = ad_stat * (1 + 0.75 / n + 2.25 / n**2)
        ad_pvalue = _ad_uniform_pvalue(ad_stat_corrected)

        # ------------------------------------------------------------------
        # 3. Chi-square goodness-of-fit against Uniform(0,1)
        #    FIX: floor n_bins so this can't crash / silently degrade for
        #    small T.
        # ------------------------------------------------------------------
        n_bins = max(5, min(30, T // 5))
        observed_freq, _ = np.histogram(pit_p, bins=n_bins, range=(0, 1))
        expected_freq = T / n_bins
        chi2_stat = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)
        chi2_pvalue = 1 - stats.chi2.cdf(chi2_stat, df=n_bins - 1)

        # ------------------------------------------------------------------
        # 4. Ljung-Box test on PITs for serial dependence
        #    FIX: floor max_lag at 1 to avoid a crash for small T.
        # ------------------------------------------------------------------
        max_lag = max(1, min(10, T // 5))
        lb_result = acorr_ljungbox(pit_p, lags=max_lag)
        lb_stat = float(lb_result['lb_stat'].iloc[-1])
        lb_pvalue = float(lb_result['lb_pvalue'].iloc[-1])

        # ------------------------------------------------------------------
        # 5. Descriptive statistics
        # ------------------------------------------------------------------
        mean_pit = np.mean(pit_p)
        var_pit = np.var(pit_p)
        skew_pit = stats.skew(pit_p)
        kurt_pit = stats.kurtosis(pit_p)

        # ------------------------------------------------------------------
        # 6. Coverage at tail quantiles
        # ------------------------------------------------------------------
        coverage_05 = np.mean(pit_p <= 0.05)
        coverage_10 = np.mean(pit_p <= 0.10)
        coverage_90 = np.mean(pit_p <= 0.90)
        coverage_95 = np.mean(pit_p <= 0.95)

        # ------------------------------------------------------------------
        # 7. KL divergence from Uniform(0,1)
        #    FIX: n_bins floored (see n_bins_kl above); pass/fail now uses a
        #    simulated critical value (kl_crit) instead of a fixed 0.02.
        # ------------------------------------------------------------------
        observed_freq_kl, _ = np.histogram(pit_p, bins=n_bins_kl, range=(0, 1))
        observed_prob = np.maximum(observed_freq_kl / T, 1e-10)
        uniform_prob = np.ones(n_bins_kl) / n_bins_kl
        kl_divergence = np.sum(observed_prob * np.log(observed_prob / uniform_prob))

        results.append({
            "Model": model_names[i],
            "Count": int(T),
            "KS_statistic": float(ks_stat),
            "KS_pvalue": float(ks_pvalue),
            "AD_statistic": float(ad_stat_corrected),
            "AD_pvalue": float(ad_pvalue),
            "Chi2_statistic": float(chi2_stat),
            "Chi2_pvalue": float(chi2_pvalue),
            "LB_statistic": lb_stat,
            "LB_pvalue": lb_pvalue,
            "KL_divergence": float(kl_divergence),
            "KL_critical_value": kl_crit,
            "Mean": float(mean_pit),
            "Variance": float(var_pit),
            "Skewness": float(skew_pit),
            "Kurtosis": float(kurt_pit),
            "Coverage_5%": float(coverage_05),
            "Coverage_10%": float(coverage_10),
            "Coverage_90%": float(coverage_90),
            "Coverage_95%": float(coverage_95),
        })

        # ------------------------------------------------------------------
        # Pass/fail thresholds (all now driven by eff_alpha, optionally
        # Bonferroni-corrected)
        # ------------------------------------------------------------------
        ks_good = ks_pvalue > eff_alpha
        ad_good = ad_pvalue > eff_alpha
        chi2_good = chi2_pvalue > eff_alpha
        lb_good = lb_pvalue > eff_alpha

        mean_good = abs(mean_pit - 0.5) <= zcrit * np.sqrt(1 / (12 * T))
        var_good = abs(var_pit - 1 / 12) <= zcrit * np.sqrt(1 / (180 * T))

        se = np.sqrt(0.05 * 0.95 / T)
        coverage_5_good = abs(coverage_05 - 0.05) <= zcrit * se
        coverage_95_good = abs(coverage_95 - 0.95) <= zcrit * se

        kl_good = kl_divergence < kl_crit

        good_tests = sum([
            ks_good, ad_good, chi2_good, lb_good,
            mean_good, var_good,
            coverage_5_good, coverage_95_good,
            kl_good,
        ])

        interpretation.append({
            "Model": model_names[i],
            "Tests_Passed": f"{good_tests}/{n_tests}",
            "KS_Test": "PASS" if ks_good else "FAIL",
            "AD_Test": "PASS" if ad_good else "FAIL",
            "Chi2_Test": "PASS" if chi2_good else "FAIL",
            "LB_Test": "PASS" if lb_good else "FAIL",
            "Mean_Test": "PASS" if mean_good else "FAIL",
            "Variance_Test": "PASS" if var_good else "FAIL",
            "KL_Test": "PASS" if kl_good else "FAIL",
            "Coverage_5%_Test": "PASS" if coverage_5_good else "FAIL",
            "Coverage_95%_Test": "PASS" if coverage_95_good else "FAIL",
        })

    tests_df = pd.DataFrame(results)
    interpretation_df = pd.DataFrame(interpretation)

    return tests_df, interpretation_df

#################################################

def plot_ecdf(ax, data, title, color='blue'):
    """Helper to plot ECDF."""
    sorted_vals = np.sort(data)
    n = len(sorted_vals)
    ecdf = np.arange(1, n+1) / n
    
    ax.plot(sorted_vals, ecdf, color=color, lw=2)
    ax.plot([0, 1], [0, 1], 'r--', lw=1, label="45° line")
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("PIT Value", fontsize=10)
    ax.set_ylabel("ECDF", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')


#################################################


def split_equal(Raw_pit):
    num_cols = Raw_pit.shape[1]

    # if odd, drop last column
    if num_cols % 2 == 1:
        Raw_pit = Raw_pit[:, :-1]

    # now guaranteed even
    d = Raw_pit.shape[1] // 2

    first_half  = Raw_pit[:, :d]
    second_half = Raw_pit[:, d:]

    return first_half, second_half, d


def nsf_pit(
        K,
        num_epochs,
        pit_data,
        discount_factors,
        seed=42
):
    
    torch.manual_seed(seed)

    first_half, second_half, d = split_equal(pit_data)

    spline_params_size = 3 * K + 1

    results = []

    for i, delta in enumerate(discount_factors):
        print(f"\n{'='*60}")
        print(f"Training for δ = {delta}")
        print(f"{'='*60}")
        
        # Prepare data
        first_half_data = first_half[i].copy()
        second_half_data = second_half[i].copy()
        
        first_half_tensor = torch.tensor(first_half_data, dtype=torch.float32).unsqueeze(0)
        second_half_tensor = torch.tensor(second_half_data, dtype=torch.float32)
        first_half_tensor.requires_grad = False
        second_half_tensor.requires_grad = False
        
        # Create model
        residual_net = ResidualNet(
            in_features=first_half_tensor.shape[1],
            out_features=spline_params_size,
            hidden_features=64,
            context_features=None,
            num_blocks=2,
            activation=F.relu,
            dropout_probability=0.1,
            use_batch_norm=False
        )

        optimizer = optim.Adam(residual_net.parameters(), lr=1e-3)
    
        # Training loop
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            theta = residual_net(first_half_tensor)
            theta_w = theta[:, :K]
            theta_h = theta[:, K:2*K]
            theta_d = theta[:, 2*K:]
            
            batch_size = second_half_tensor.shape[0]
            theta_w_expanded = theta_w.expand(batch_size, -1)
            theta_h_expanded = theta_h.expand(batch_size, -1)
            theta_d_expanded = theta_d.expand(batch_size, -1)
            
            v_i, logabsdet_i, derivative_i = rational_quadratic_spline(
                inputs=second_half_tensor,
                unnormalized_widths=theta_w_expanded,
                unnormalized_heights=theta_h_expanded,
                unnormalized_derivatives=theta_d_expanded,
                inverse=False
            )
            
            loss = -logabsdet_i.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            if epoch % 200 == 0:
                print(f"  Epoch {epoch:4d} | Loss: {loss.item():8.4f}")
        
        # Evaluate
        residual_net.eval()
        with torch.no_grad():
            theta = residual_net(first_half_tensor)
            theta_w = theta[:, :K].expand(batch_size, -1)
            theta_h = theta[:, K:2*K].expand(batch_size, -1)
            theta_d = theta[:, 2*K:].expand(batch_size, -1)
            
            v_transformed, log_likelihood, derivative = rational_quadratic_spline(
                inputs=second_half_tensor,
                unnormalized_widths=theta_w,
                unnormalized_heights=theta_h,
                unnormalized_derivatives=theta_d,
                inverse=False
            )
            
            u_original = second_half_tensor.numpy()
            v_transformed = v_transformed.numpy()
            log_likelihood = log_likelihood.numpy()
            derivative = derivative.numpy()
        
        # Calculate statistics
        ks_before = stats.kstest(u_original, 'uniform')
        ks_after = stats.kstest(v_transformed, 'uniform')
        
        # Store results
        results.append({
            'delta': delta,
            'u_original': u_original,
            'v_transformed': v_transformed,
            'log_likelihood': log_likelihood,
            'derivative': derivative,
            'ks_before': ks_before.statistic,
            'ks_after': ks_after.statistic,
            'mean_before': u_original.mean(),
            'mean_after': v_transformed.mean()
        })
    
    return results

        