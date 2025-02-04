import numpy as np

from algorithms.learner import Learner
from sampling import sample_posterior, compute_minimizer


class ThompsonSamplingLearner(Learner):
    def __init__(self, prior_mu, prior_Sigma, B, n, noise_sigma, burn_in):
        """
        Thompson Sampling learner which uses hit-and-run to sample a function from the posterior.
        """
        super().__init__(prior_mu, prior_Sigma, B, n, noise_sigma)
        self.ts = np.linspace(1 / n, 1, n)
        self.name = f"ThompsonSampling"
        self.burn_in = burn_in

        self.ws = (
            []
        )  # list of weight vectors sampled from the posterior just for visualization

    def select_action(self):
        """
        At time t, compute the posterior, sample a function from it, and return its minimizer.
        """
        post_mean, post_cov = self.get_posterior()
        # Sample one weight vector from the posterior.
        sampled_ws = sample_posterior(
            post_mean,
            post_cov,
            self.B,
            num_samples=1,
            burn_in=self.burn_in,
        )
        sampled_w = None
        sampled_w = sampled_ws[0]

        self.ws.append(sampled_w)  # for visualization
        # Compute the minimizer of the sampled function.
        x_t = compute_minimizer(sampled_w, self.ts)
        return x_t

    def get_posterior(self):
        """
        Compute the posterior distribution on the weight vector given the history,
        using the standard Bayesian linear regression formulas.
        """
        if len(self.history_x) == 0:
            return self.prior_mu, self.prior_Sigma
        # Construct design matrix: each row is phi(x) = [|x - t| for t in ts].
        Phi = np.array([np.abs(x - self.ts) for x in self.history_x])
        Sigma_inv = np.linalg.inv(self.prior_Sigma)
        post_cov_inv = Sigma_inv + (Phi.T @ Phi) / (self.noise_sigma**2)
        post_cov = np.linalg.inv(post_cov_inv)
        post_mean = post_cov @ (
            Sigma_inv @ self.prior_mu
            + (Phi.T @ np.array(self.history_y)) / (self.noise_sigma**2)
        )
        return post_mean, post_cov
