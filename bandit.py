import numpy as np
from scipy.stats import truncnorm

from sampling import sample_posterior, compute_minimizer


class BanditProblem:
    def __init__(self, w_true, ts, sigma, f_star=None, B=5, interval=[-1, 1]):
        """
        w_true: true weight vector defining f*(x) = sum_i w_true[i]*|x - ts[i]|
        ts: array of basis centers (assumed sorted)
        sigma: standard deviation of Gaussian noise.
        """
        self.w_true = w_true
        self.ts = ts
        self.sigma = sigma
        self.B = B
        self.interval = interval

        # f_star overrides the true function if provided.
        self.f_star = True
        self.f = f_star if f_star else self._f

        self.optimal_x, self.optimal_value = self.compute_optimum()

    def _f(self, x):
        """
        Evaluate the true function f*(x) in a vectorized fashion.
        x can be a scalar or an array.
        """
        x = np.atleast_1d(x)
        # f(x) = sum_i w_true[i] * |x - ts[i]|
        return np.sum(
            np.abs(x[:, None] - self.ts[None, :]) * self.w_true[None, :], axis=1
        )

    def query(self, x):
        """
        Given a query point x (scalar), return a noisy observation of f*(x).
        """
        true_val = self.f(np.array([x]))[0]
        interval = [0, self.B]
        # Truncate the normal distribution to the interval [0, B] with mean true_val and variance sigma.
        a, b = (interval[0] - true_val) / self.sigma, (
            interval[1] - true_val
        ) / self.sigma
        loss = truncnorm.rvs(a, b, loc=true_val, scale=self.sigma)
        return loss

    def compute_optimum(self):
        """
        Compute the minimizer and minimum value of f* on [0,1].
        (Here we use the weighted-median as an approximation.)
        """
        if self.f_star:
            # Find the minimizer of f* by grid search.
            xs = np.linspace(self.interval[0], self.interval[1], 1000)

            fs = [self.f(x) for x in xs]
            x_star = xs[np.argmin(fs)]
            f_star = np.min(fs)
            return x_star, f_star

        x_star = compute_minimizer(self.w_true, self.ts)
        f_star = self.f(np.array([x_star]))[0]
        return x_star, f_star


class BanditFactory:
    def __init__(self, prior_mu, prior_Sigma, interval, B, n_basis, sigma):
        """
        prior_mu, prior_Sigma: parameters for the Gaussian prior over weights.
        B: budget (constraint: sum(w) <= B, w_i >= 0)
        n: number of basis functions.
        sigma: noise standard deviation.
        """
        self.mu = prior_mu
        self.Sigma = prior_Sigma
        self.B = B
        self.n_basis = n_basis
        self.sigma = sigma
        self.interval = interval

        self.ts = np.linspace(self.interval[0], self.interval[1], self.n_basis)

    def create_problem(self):
        """
        Create a new BanditProblem instance by sampling a true weight vector from the prior.
        """
        # Use hit-and-run to sample one point from the prior (truncated).
        prior_sample = sample_posterior(
            self.mu, self.Sigma, self.B, num_samples=1, burn_in=500
        )
        w_true = prior_sample[0]
        return BanditProblem(w_true, self.ts, self.sigma)
