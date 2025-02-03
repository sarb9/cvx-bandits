import numpy as np
from scipy.stats import truncnorm

##############################
#   UTILITY FUNCTIONS        #
##############################


def compute_minimizer(w, ts):
    """
    Given a weight vector w (defining f(x) = sum_i w_i |x - ts_i|)
    and the corresponding basis centers ts (assumed sorted),
    return the minimizer (the weighted median).
    """
    total_weight = np.sum(w)
    cum_weights = np.cumsum(w)
    idx = np.searchsorted(cum_weights, total_weight / 2)
    # For a little smoothness, we could interpolate between ts[idx-1] and ts[idx].
    return ts[min(idx, len(ts) - 1)]


##############################
#   HIT-AND-RUN SAMPLING     #
##############################


def conditional_params(i, current_w, current_mean, current_cov):
    """
    Compute the conditional distribution parameters for coordinate i,
    given the other coordinates in current_w for a Gaussian with mean current_mean
    and covariance current_cov.
    Returns (m_cond, var_cond).
    """
    n = len(current_w)
    other_idx = [j for j in range(n) if j != i]
    Sigma_ii = current_cov[i, i]
    Sigma_i_other = current_cov[i, other_idx]
    Sigma_other_other = current_cov[np.ix_(other_idx, other_idx)]
    mean_i = current_mean[i]
    mean_other = current_mean[other_idx]
    inv_Sigma_other_other = np.linalg.inv(Sigma_other_other)
    m_cond = mean_i + Sigma_i_other @ inv_Sigma_other_other @ (
        current_w[other_idx] - mean_other
    )
    var_cond = Sigma_ii - Sigma_i_other @ inv_Sigma_other_other @ Sigma_i_other.T
    return m_cond, var_cond


def sample_conditional(i, current_w, current_mean, current_cov, B):
    """
    Sample a new value for coordinate i from its conditional distribution,
    truncated to [0, B - sum_{j != i} current_w[j]].
    """
    m_cond, var_cond = conditional_params(i, current_w, current_mean, current_cov)
    sd_cond = np.sqrt(var_cond)
    sum_others = np.sum(np.delete(current_w, i))
    lower = 0.0
    upper = B - sum_others
    if upper < lower:  # safeguard
        print(f"This should not happen: upper < lower: {upper} < {lower}")
        print(f"{sd_cond}")
        upper = lower
    a, b = (lower - m_cond) / sd_cond, (upper - m_cond) / sd_cond
    new_val = truncnorm.rvs(a, b, loc=m_cond, scale=sd_cond)
    return new_val


def coordinate_hit_and_run(
    current_mean,
    current_cov,
    B,
    num_samples=1000,
    burn_in=500,
    init=None,
    sample_at_least_once=False,
):
    """
    Coordinate-wise hit-and-run (Gibbs) sampler for a Gaussian (truncated to the region
    {w: w_i >= 0 for all i, sum(w) <= B}).
    """
    n = len(current_mean)
    if init is None:
        init = np.copy(current_mean)
        init[init <= 0] = 1e-3
        if np.sum(init) > B:
            init = init * (B / np.sum(init))
    current_w = np.copy(init)
    samples = []
    total_iters = burn_in + num_samples
    for it in range(total_iters):
        i = np.random.randint(n)
        current_w[i] = sample_conditional(i, current_w, current_mean, current_cov, B)
        if it >= burn_in:
            samples.append(np.copy(current_w))

        if sample_at_least_once and it == total_iters - 1 and len(samples) == 0:
            # If we want at least one sample, and we haven't got any yet, resample.
            it = total_iters - 2
    return np.array(samples)


def sample_posterior(
    post_mean,
    post_cov,
    B,
    num_samples=1,
    burn_in=500,
    init=None,
    sampler_type="coordinate",
    sample_at_least_once=False,
):
    """
    Sample from the posterior distribution of a Gaussian with mean post_mean and covariance post_cov,
    truncated to the region {w: w_i >= 0 for all i, sum(w) <= B}.
    It's possible to add more sampler types in the future.
    """
    return coordinate_hit_and_run(
        post_mean, post_cov, B, num_samples, burn_in, init, sample_at_least_once
    )
