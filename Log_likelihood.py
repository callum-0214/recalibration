import numpy as np
from scipy.stats import norm, beta
import pandas as pd

from pit import pit_func



##############################
# Original PDF function for TLP
##############################

def tlp_pdf(
        forecast: np.ndarray,
        x: np.ndarray,
        weights: np.ndarray,
        y: np.ndarray,
        sigma_y: float,
) -> np.ndarray:

    K,T = forecast.shape
    P,T_w, K_w = weights.shape
    M = x.shape[0]

    assert (T, K) == (T_w, K_w), 'Shape mismatch'

    pdf_values = np.zeros((K, T, M), dtype=float)

    for k in range(K):
        for t in range(T):
            # Vectorized over all grid points at once
            pdf_values[k, t, :] = norm.pdf(x, loc=(forecast.to_numpy())[k, t], scale=sigma_y)

    TL = np.einsum('ktm,ptk -> ptm', pdf_values, weights)

    pdf = pit_func(TL, x, y)

    return pdf



def log_score_comparison(cdfs, a, b, TLP, TLP_pdf, before_log_score,discount_factors):
    """
    Compare mean log-scores before and after calibration.
    """

    P,T = cdfs.shape
    P,T = a.shape
    P,T = b.shape

    log_scores = np.zeros((P, T), dtype=float)

    for p in range(P):
        for t in range(T):
            # raw cdf
            F_lp = TLP[p,t] 

            # raw pdf
            f_lp = TLP_pdf[p,t]
            
            # 2) Beta‐PDF at the PIT value u = new_cdf[p,t]
            a_t = a[p, t]   # shape used *before* seeing y_t
            b_t = b[p, t]
            f_beta = beta.pdf(F_lp, a_t, b_t) * f_lp
            
            log_scores[p, t] = np.log(f_beta + 1e-12)  # jitter to avoid log(0)

    after_mean_log_score = log_scores.mean(axis=1)
    before_mean_log_score = before_log_score.mean(axis=1)

    after_cumulative_log_score = log_scores.sum(axis=1)
    before_cumulative_log_score = before_log_score.sum(axis=1)
    
    results_df = pd.DataFrame({
    'discount_factor': discount_factors,
    'Before Calibration': before_mean_log_score,
    'After Calibration': after_mean_log_score,
    'Before Cumulative Log Score': before_cumulative_log_score,
    'After Cumulative Log Score': after_cumulative_log_score})

    # Add comparison column
    results_df['Comparison'] = results_df['After Calibration'] > results_df['Before Calibration']
    results_df['Comparison'] = results_df['Comparison'].map({True: 'Higher Mean Log Score', False: 'Lower Mean Log Score'})
    return results_df, log_scores




import numpy as np
import matplotlib.pyplot as plt


def _log_score_original(cdf_series, x_grid, y_series):
    """
    Original forecast log score log f_t(y_t), computed from the discretised CDF.
    Interpolates the density (finite differences of the CDF) at y_t.
    """
    x_mid = 0.5 * (x_grid[:-1] + x_grid[1:])
    dx    = np.diff(x_grid)
    log_f = np.empty(len(y_series))
    for t in range(len(y_series)):
        pdf = np.maximum(np.diff(cdf_series[t]) / dx, 0.0)
        s   = np.trapezoid(pdf, x_mid)
        if s > 1e-10:
            pdf = pdf / s
        f_at_y = float(np.interp(y_series[t], x_mid, pdf))
        log_f[t] = np.log(max(f_at_y, 1e-30))
    return log_f


def plot_log_score_comparison(
        results,
        cdf_data,
        y,
        x_grid,
        train_frac=0.7,
        which='test',        # 'test' | 'train' | 'both'
        figsize=(15, 10),
):
    """
    Compare the log score before vs after recalibration.

    For each δ in `results` this produces three panels:
      1. Cumulative log score over time  (before vs after, on the chosen slice).
      2. Per-observation log score  (before vs after, thin lines + moving average).
      3. Bar summary of mean log score across δ, plus the mean gain per δ.

    The identity from the paper:
        log f̃_t(y_t)  =  log f_t(y_t)  +  log τ'_{θ_t}(u_t)
    is used to construct 'after' from 'before' + the stored log-Jacobian.
    """
    P       = len(results)
    T       = len(y)
    T_train = int(round(train_frac * T))
    x_grid  = np.asarray(x_grid)
    y_arr   = np.asarray(y)

    # ---- per-δ log-score series --------------------------------------------
    log_before_full = []   # log f_t(y_t) for the whole sequence
    log_after_full  = []   # log f̃_t(y_t) for the whole sequence
    deltas          = []

    for i, res in enumerate(results):
        deltas.append(res['delta'])
        lb = _log_score_original(cdf_data[i], x_grid, y_arr)          # length T
        la = lb + np.asarray(res['log_likelihood'], dtype=float)      # length T
        log_before_full.append(lb)
        log_after_full.append(la)

    log_before_full = np.stack(log_before_full)   # (P, T)
    log_after_full  = np.stack(log_after_full)    # (P, T)

    # ---- slice selection ----------------------------------------------------
    def _slice(a):
        if which == 'train': return a[:, :T_train]
        if which == 'test' : return a[:,  T_train:]
        return a                                     # both

    LB, LA = _slice(log_before_full), _slice(log_after_full)
    t_axis = np.arange(LB.shape[1])
    if which == 'test':
        t_axis = t_axis + T_train
    slice_name = which.upper() + f"  (n={LB.shape[1]})"

    # ---- figure -------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(2, 2, height_ratios=[2, 1.2], hspace=0.35, wspace=0.25)

    # Panel A: cumulative log score across time, one line per δ, before vs after.
    axA = fig.add_subplot(gs[0, :])
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, P))
    for i in range(P):
        axA.plot(t_axis, LB[i].cumsum(), color=colors[i], alpha=0.35, linewidth=1.0,
                 linestyle='--',
                 label=f"before δ={deltas[i]:g}" if i in (0, P-1) else None)
        axA.plot(t_axis, LA[i].cumsum(), color=colors[i], alpha=0.95, linewidth=1.3,
                 label=f"after  δ={deltas[i]:g}" if i in (0, P-1) else None)
    axA.set_title(f"Cumulative log score on {slice_name}  "
                  f"(dashed = before recalibration, solid = after)")
    axA.set_xlabel("t"); axA.set_ylabel(r"$\sum_{s\leq t} \log \tilde f_s(y_s)$")
    axA.grid(alpha=0.3); axA.legend(frameon=False, loc='best', ncol=2, fontsize=8)

    # Panel B: mean log score per δ, before vs after (bars).
    axB = fig.add_subplot(gs[1, 0])
    mean_before = LB.mean(axis=1); mean_after = LA.mean(axis=1)
    xb = np.arange(P); w = 0.4
    axB.bar(xb - w/2, mean_before, w, label='before', color='lightgrey', edgecolor='k')
    axB.bar(xb + w/2, mean_after,  w, label='after',  color='steelblue', edgecolor='k')
    axB.set_xticks(xb); axB.set_xticklabels([f"{d:g}" for d in deltas], rotation=45)
    axB.set_ylabel(r"mean $\log\tilde f$")
    axB.set_title(f"Mean log score by δ  ({slice_name})")
    axB.grid(alpha=0.3, axis='y'); axB.legend(frameon=False)

    # Panel C: gain per δ (after − before) with 95 % bootstrap CI.
    axC = fig.add_subplot(gs[1, 1])
    gains, los, his = [], [], []
    rng = np.random.default_rng(0)
    n   = LB.shape[1]
    for i in range(P):
        diff = LA[i] - LB[i]
        gains.append(diff.mean())
        # paired bootstrap
        boot = rng.choice(diff, size=(1000, n), replace=True).mean(axis=1)
        los.append(np.quantile(boot, 0.025)); his.append(np.quantile(boot, 0.975))
    gains = np.array(gains); los = np.array(los); his = np.array(his)
    yerr  = np.vstack([gains - los, his - gains])
    bar_c = np.where(gains > 0, 'seagreen', 'crimson')
    axC.bar(xb, gains, yerr=yerr, color=bar_c, edgecolor='k', capsize=3)
    axC.axhline(0, color='k', linewidth=0.8)
    axC.set_xticks(xb); axC.set_xticklabels([f"{d:g}" for d in deltas], rotation=45)
    axC.set_ylabel(r"gain in mean $\log\tilde f$ per obs.")
    axC.set_title(f"Recalibration gain by δ  ({slice_name},  95% bootstrap CI)")
    axC.grid(alpha=0.3, axis='y')

    fig.suptitle("Log-score comparison: original pool vs recalibrated forecast exchange rate",
                 y=1.00, fontsize=13)
    #plt.tight_layout()
    plt.show()

    # ---- tidy summary table -------------------------------------------------
    import pandas as pd
    return pd.DataFrame({
        "delta":         deltas,
        "mean_before":   mean_before,
        "mean_after":    mean_after,
        "gain_per_obs":  gains,
        "gain_ci_lo":    los,
        "gain_ci_hi":    his,
        "total_gain":    gains * n,
        "n":             n,
    })