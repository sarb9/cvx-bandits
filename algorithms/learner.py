import numpy as np


class Learner:
    def __init__(self, prior_mu, prior_Sigma, B, n, sigma):
        """
        Base learner class. Stores prior information and history of actions/observations.
        """
        self.prior_mu = prior_mu
        self.prior_Sigma = prior_Sigma
        self.B = B
        self.n = n
        self.sigma = sigma
        self.ts = np.linspace(1 / n, 1, n)
        self.history_x = []
        self.history_y = []
        self.cum_regret = 0.0  # cumulative regret

    def select_action(self):
        """
        To be implemented by subclasses.
        Should return an action x in [0,1].
        """
        raise NotImplementedError

    def update(self, x, y):
        """
        Record the observation.
        """
        self.history_x.append(x)
        self.history_y.append(y)

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
        post_cov_inv = Sigma_inv + (Phi.T @ Phi) / (self.sigma**2)
        post_cov = np.linalg.inv(post_cov_inv)
        post_mean = post_cov @ (
            Sigma_inv @ self.prior_mu
            + (Phi.T @ np.array(self.history_y)) / (self.sigma**2)
        )
        return post_mean, post_cov
