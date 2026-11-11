
import numpy as np
from scipy.stats import norm


# ============================================================================
# Algorithm 1: raw CDF over grid points
# ============================================================================
def tlp(
        forecast: np.ndarray,
        x: np.ndarray,
        weights: np.ndarray,
        sigma_y: float,
) -> np.ndarray:

    K,T = forecast.shape
    P,T_w, K_w = weights.shape
    M = x.shape[0]

    assert (T, K) == (T_w, K_w), 'Shape mismatch'

    cdf_values = np.zeros((K, T, M), dtype=float)

    for k in range(K):
        for t in range(T):
            # Vectorized over all grid points at once
            cdf_values[k, t, :] = norm.cdf(x, loc=(forecast.to_numpy())[k, t], scale=sigma_y)

    TL = np.einsum('ktm,ptk -> ptm', cdf_values, weights)

    return TL

# Fixed tlp function
def ex_tlp(forecast, var, x, weights):
    K, T = forecast.shape
    P, T_w, K_w = weights.shape
    M = x.shape[0]
    
    assert (T, K) == (T_w, K_w), f'Shape mismatch: mean {(K,T)}, weights {(T_w,K_w)}'
    
    cdf_values = np.zeros((K, T, M))
    
    for k in range(K):
        for t in range(T):
            cdf_values[k, t, :] = norm.cdf(x, loc= forecast[k, t], scale=np.sqrt(var[k, t]))
    
    # Linear pool
    linear_pool = np.einsum('ktm,ptk -> ptm', cdf_values, weights)

    return linear_pool


# ============================================================================
# Algorithm 2: regurlarized hypergeometric function 3F2
# ============================================================================

def geom_3f2(z, a_params, b_params, m=10):
    """
    Compute generalized hypergeometric function 3F2 using series expansion.
    """
    if np.abs(z) < 1e-10:
        return 1.0

    result = 1.0
    term = 1.0

    for k in range(1, m+1):
        # Pochhammer symbols
        poch_a = np.prod([a + k - 1 for a in a_params])
        poch_b = np.prod([b + k - 1 for b in b_params])

        term *= (poch_a / poch_b) * z / k
        result += term

        if np.abs(term) < 1e-12:
            break

    return result