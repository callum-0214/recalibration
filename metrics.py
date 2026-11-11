import numpy as np
import matplotlib.pyplot as plt

######################
#Expected Calibration Error (ECE) for PIT values
######################

def compute_ece(pit_values, n_bins):
    """
    Computes ECE for PIT values using manual binning logic.
    Adapted from _ECELoss to compare Observed vs Expected Frequency.
    """
    # Define boundaries exactly as in the snippet
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Logic based on snippet: in_bin mask
        in_bin = (pit_values > bin_lower) & (pit_values <= bin_upper)
        
        # prop_in_bin: The Observed Frequency (equivalent to 'accuracy' in classif.)
        prop_in_bin = np.mean(in_bin)
        
        # expected_prop: The Ideal Frequency (equivalent to 'confidence' if perfect)
        # For Uniform PITs, this is just the bin width.
        expected_prop = bin_upper - bin_lower
        
        # ECE calculation: Sum of absolute differences
        ece += np.abs(prop_in_bin - expected_prop)
        
    return ece


######################
#Continuous Ranked Probability Score (CRPS) for PIT values
######################


def compute_crps_at_time(cdf_values, x_grid, y_obs):
        """
        Compute CRPS at a single time step.
        CRPS = ∫(F(x) - 1{x ≥ y})² dx
        """
        indicator = (x_grid >= y_obs).astype(float)
        integrand = (cdf_values - indicator)**2
        crps = np.trapezoid(integrand, x_grid)
        return crps
    
    
    
def metrics_comparison(x,y,original_pit,original_cdf, beta_results, nsf_results, beta_log, nsf_log,figsize=(18, 4)):

    P = len(nsf_results)
    T = original_pit.shape[1]
    discount_factors = [nsf_results[p]['delta'] for p in range(P)]
    fig, axes = plt.subplots(1,4,figsize=figsize)

    ax1 = axes[0]
    n_bins=30
    nsf_ece = [nsf_results[p]['ece_score'] for p in range(P)]
    beta_ece = [compute_ece(beta_results['all_calibrated_cdfs'][p],n_bins=n_bins) for p in range(P)]
    original_ece = [compute_ece(original_pit[p],n_bins=n_bins) for p in range(P)]

    ax1.plot(discount_factors,original_ece,color='red',label='original',linestyle='',marker='o')
    ax1.plot(discount_factors,beta_ece,color='blue',label='beta',linestyle='',marker='o')
    ax1.plot(discount_factors,nsf_ece,color='green',label='nsf',linestyle='',marker='o')
    ax1.set_xlabel('Discount Factors')
    ax1.set_ylabel('ECE')
    ax1.set_title(f'ECE comparison')
    ax1.legend(frameon=False,loc='best')
    ax1.grid(alpha=0.3)

    ax2 = axes[1]

    original_kl = []

    for p in range(P):
        hist_v, _ = np.histogram(original_pit[p], bins=n_bins, density=True)
        hist_v = hist_v / (hist_v.sum() + 1e-10)
        uniform_pmf = np.ones_like(hist_v) / len(hist_v)

        epsilon = 1e-10
        kl_divergence = np.sum(hist_v * np.log((hist_v + epsilon) / (uniform_pmf + epsilon)))
        original_kl.append(kl_divergence)

    nsf_kl = [nsf_results[p]['kl_divergence'] for p in range(P)]
    beta_kl = [beta_results['kl_divergence'][p] for p in range(P)]
    
    ax2.plot(discount_factors,original_kl,color='red',label='original',linestyle='',marker='o')
    ax2.plot(discount_factors,beta_kl,color='blue',label='beta',linestyle='',marker='o')
    ax2.plot(discount_factors,nsf_kl,color='green',label='nsf',linestyle='',marker='o')
    ax2.set_xlabel('Discount Factors')
    ax2.set_ylabel('KL Divergence')
    ax2.set_title(f'KL Divergence comparison')
    ax2.legend(frameon=False,loc='best')
    ax2.grid(alpha=0.3)

    ax3 =axes[2]

    original_crps = np.zeros((P,T))
    original_crps_mean= []

    for p in range(P):
        for t in range(T):
            original_crps[p,t] = compute_crps_at_time(original_cdf[p,t,:],x,y[t])

    original_crps_mean.append(original_crps.mean(axis=1))
    original_crps_mean = np.array(original_crps_mean[0]) 

    nsf_crps_mean = [nsf_results[p]['crps_history'].mean() for p in range(P)]
    beta_crps_mean = [beta_results['crps_history'][p].mean() for p in range(P)]

    ax3.plot(discount_factors,original_crps_mean,color='red',label='orignal',linestyle='',marker='o')
    ax3.plot(discount_factors,beta_crps_mean,color='blue',label='beta',linestyle='',marker='o')
    ax3.plot(discount_factors,nsf_crps_mean,color='green',label='nsf',linestyle='',marker='o')
    ax3.set_xlabel('Discount Factors')
    ax3.set_ylabel('Mean CRPS')
    ax3.set_title(f'CRPS comparison')
    ax3.legend(frameon=False,loc='best')
    ax3.grid(alpha=0.3)

    ax4 = axes[3]
    beta_log_score = beta_log['After Calibration']
    nsf_log_score = nsf_log['After Calibration']
    original_log_score = nsf_log['Before Calibration']

    ax4.plot(discount_factors,original_log_score,color='red',label='original',linestyle='',marker='o')
    ax4.plot(discount_factors,beta_log_score,color='blue',label='beta',linestyle='',marker='o')
    ax4.plot(discount_factors,nsf_log_score,color='green',label='nsf',linestyle='',marker='o')
    ax4.set_xlabel('Discount Factors')
    ax4.set_ylabel('Log likelihood')
    ax4.set_title(f'Log likelihood comparison')
    ax4.legend(frameon=False,loc='best')
    ax4.grid(alpha=0.3)


    plt.tight_layout()
    plt.show()