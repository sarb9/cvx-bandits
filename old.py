def compute_line_interval(w, d, B, margin=1e-3):
    """
    Given a point w and direction d, compute the interval [t_lower, t_upper] such that
    w_new = w + t*d satisfies:
       (i)  for each coordinate: w[i] + t*d[i] >= 0, and
      (ii)  sum(w) + t * sum(d) <= B.
    """
    #make sure w is inside the box
    assert np.all(w >= 0), f"Invalid w, it's negative: {np.min(w)}"
    assert np.sum(w) <= B, f"Invalid w, sum(w) > B: {np.sum(w)}"
    lower_candidates = []
    upper_candidates = []
    n = len(w)
    for i in range(n):
        if d[i] == 0:
            continue
        if d[i] > 0:
            lower_candidates.append(min(margin -w[i], 0) / d[i])
            # upper_candidates.append((B- margin - w[i]) / d[i])
        elif d[i] < 0:
            upper_candidates.append(min(margin -w[i], 0) / d[i])
            # lower_candidates.append((B - margin - w[i]) / d[i])

    S = np.sum(w)
    D = np.sum(d)
    assert B - S >= 0, f"Invalid B: {B}, sum(w): {S}"
    if D > 0:
        upper_candidates.append(((B - margin) - S) / D)
    elif D < 0:
        lower_candidates.append(((B - margin) - S) / D)
    t_lower = max(lower_candidates) if lower_candidates else -np.inf
    t_upper = min(upper_candidates) if upper_candidates else np.inf

    max_w = w + t_upper * d
    min_w = w + t_lower * d
    if not np.all(max_w >= 0) or not np.all(min_w >= 0):
        #where is this happening?
        print()
        print("Problem with t_upper or t_lower")
        index = np.argmin(min_w)
        print(f"min_w: {min_w[index]}")
        print(f"max_w: {max_w[index]}")
        print(f"w: {w[index]}")
        print(f"d: {d[index]}")
        print(f"t_upper: {t_upper}")
        print(f"t_lower: {t_lower}")
        print(f"({margin -w[index]}) / {d[index]}")
        print()
        print()
        import pickle
        with open('data.pkl', 'wb') as f:
            pickle.dump((w, d, B, margin), f)


    assert np.all(max_w >= 0), f"Problem with t_upper, it's negative: {np.min(max_w)}"
    assert np.sum(max_w) <= B, f"Problem with t_upper, it's large: {np.sum(max_w)}"
    assert np.all(min_w >= 0), f"Problem with t_lower, it's negative: {np.min(min_w)}"
    assert np.sum(min_w) <= B, f"Problem with t_lower, it's large: {np.sum(min_w)}"
    assert t_lower <= t_upper, f"Invalid interval: {t_lower} > {t_upper}"
    return t_lower, t_upper

def directional_hit_and_run(post_mean, post_cov, B, num_samples=1000, burn_in=500, init=None, sample_at_least_once=False):
    """
    Directional hit-and-run sampler for a Gaussian (truncated to {w: w_i >= 0, sum(w) <= B}).
    At each iteration, a random direction is chosen; the line segment in that direction is computed,
    and a new point is sampled along that direction according to the one-dimensional Gaussian marginal.
    """
    n = len(post_mean)
    post_cov_inv = np.linalg.inv(post_cov)
    if init is None:
        init = np.copy(post_mean)
        init[init <= 0] = 1e-3
        if np.sum(init) > B:
            init = init * (B / np.sum(init))
    current_w = np.copy(init)
    samples = []
    total_iters = burn_in + num_samples
    it = 0
    while it <= total_iters:
        assert np.all(current_w >= 0), f"Invalid current_w, it's negative: {np.min(current_w)}"
        assert np.sum(current_w) <= B, f"Invalid current_w, sum(w) > B: {np.sum(current_w)}"
        d = np.random.randn(n)
        d = d / np.linalg.norm(d)
        t_lower, t_upper = compute_line_interval(current_w, d, B)
        if t_lower > t_upper:
            it += 1
            continue
        q_val = d.T @ post_cov_inv @ d
        v = current_w - post_mean
        c_val = d.T @ post_cov_inv @ v
        t_mean = - c_val / q_val
        t_std = np.sqrt(1.0 / q_val)
        a, b = (t_lower - t_mean) / t_std, (t_upper - t_mean) / t_std
        if np.isclose(a, b):
            if it >= burn_in:
                samples.append(np.copy(current_w))
            it += 1
            continue
        t_sample = truncnorm.rvs(a, b, loc=t_mean, scale=t_std)
        current_w = current_w + t_sample * d
        if it >= burn_in:
            samples.append(np.copy(current_w))
        if sample_at_least_once and len(samples) == 0:
            # If we want at least one sample, and we haven't got any yet, resample.
            it = it - 1
        it += 1
    return np.array(samples)
