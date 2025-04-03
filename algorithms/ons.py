import numpy as np
import math

from algorithms.learner import Learner


class ONS(Learner):

    def __init__(
        self,
        interval,
        prior_mu,
        prior_Sigma,
        B,
        n_basis,
        noise_sigma,
        horizon,
        name,
        sigma,
        lambda_,
        eta,
        epsilon,
        delta,
        C,
    ):
        super().__init__(
            interval, prior_mu, prior_Sigma, B, n_basis, noise_sigma, horizon, name
        )

        self.sigma = sigma
        self.lambda_ = lambda_
        self.eta = eta
        self.epsilon = epsilon
        self.delta = delta

        L = C * np.log(1 / delta)

        self.mu = 0
        self.gs = 0
        self.X = None

    def minkowski_functional(self, x):
        """
        Minkowski projection wrt convex body [-1, 1].
        """
        return np.abs(x)

    def minkowski_functional_pi(self, x):
        return max(1, self.minkowski_functional(x) / (1 - self.epsilon))

    def select_action(self):
        """
        Select the action.
        """
        # Sample X from the normal distribution with mean mu and variance variance.
        self.X = np.random.normal(self.mu, self.sigma)
        action = self.X / self.minkowski_functional_pi(self.X)

        assert self.interval[0] <= action <= self.interval[1]

        return action

    def v(self, x):
        return self.minkowski_functional_pi(x) - 1

    def density(self, x):
        """
        Probability density function of the normal distribution with mean mu and variance variance.
        """
        return np.exp(-0.5 * (x - self.mu) ** 2 / self.sigma) / np.sqrt(
            2 * np.pi * self.sigma
        )

    def update(self, x, y):
        """
        Record the observation.
        """
        super().update(x, y)

        Y = (
            self.minkowski_functional_pi(self.X) * y
            + (2 * self.v(self.X)) / self.epsilon
        )

        r = (self.X - self.lambda_ * self.mu) / (1 - self.lambda_)
        R = self.density(r) / ((1 - self.lambda_) * self.density(self.X))

        g = (R * Y * (1 / self.sigma) * (self.X - self.mu)) / ((1 - self.lambda_) ** 2)

        H_1 = (self.lambda_ * R * Y) / (2 * (1 - self.lambda_) ** 2)
        H_2 = (self.X - self.mu) ** 2 / (
            (1 - self.lambda_) ** 2 * self.sigma**2
        ) - 1 / self.sigma
        H = H_1 * H_2

        self.sigma = 1 / (1 / self.sigma + self.eta * H)

        target = self.mu - self.eta * g * self.sigma
        # set mu to the projection of target onto the interval * (1 - epsilon)
        self.mu = min(
            max(target, self.interval[0] * (1 - self.epsilon)),
            self.interval[1] * (1 - self.epsilon),
        )
