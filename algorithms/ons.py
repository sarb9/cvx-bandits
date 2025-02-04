import numpy as np
import math

from algorithms.learner import Learner


class ONS(Learner):

    def __init__(self, horizon, interval, delta, M=1, C=1):
        self.history_x = []
        self.history_y = []
        self.interval = interval
        self.cum_regret = 0

        self.delta = delta
        L = C * np.log(1 / delta)

        self.horizon = horizon
        self.variance = 0.1  # / (10 * M * np.sqrt(L))
        self.lamda = 0.1  # / (40 * M * (L**3.5))
        self.eta = 0.01  # 4 * M #/ (3 * np.sqrt(horizon * L)) * 100
        self.epsilon = (
            1 / 100
        )  # min(600 * M * (L**5.5) / np.sqrt(horizon), 1 / np.sqrt(horizon))

        assert interval[0] <= 0 <= interval[1]
        assert 0 < self.epsilon
        assert self.epsilon < 0.5

        self.mu = 0
        self.gs = 0

    def minkowski_functional(self, x):
        """
        Minkowski functional for convex body [0, 1].
        """
        return np.abs(x)

    def minkowski_functional_pi(self, x):
        return max(1, self.minkowski_functional(x) / (1 - self.epsilon))

    def select_action(self):
        """
        Select the action.
        """
        # Sample X from the normal distribution with mean mu and variance variance.
        X = np.random.normal(self.mu, self.variance)
        action = X / self.minkowski_functional_pi(X)

        assert self.interval[0] <= action <= self.interval[1]

        return action

    def v(self, x):
        return self.minkowski_functional_pi(x) - 1

    def density(self, x):
        """
        Probability density function of the normal distribution with mean mu and variance variance.
        """
        return np.exp(-0.5 * (x - self.mu) ** 2 / self.variance) / np.sqrt(
            2 * np.pi * self.variance
        )

    def update(self, x, y):
        """
        Record the observation.
        """
        super().update(x, y)

        X = x
        Y = self.minkowski_functional_pi(X) * y + (2 * self.v(x)) / self.epsilon

        r = (X - self.lamda * self.mu) / (1 - self.lamda)
        R = self.density(r) / ((1 - self.lamda) * self.density(X))

        g = (R * Y * (1 / self.variance) * (X - self.mu)) / ((1 - self.lamda) ** 2)

        H_1 = (self.lamda * R * Y) / (2 * (1 - self.lamda) ** 2)
        H_2 = (X - self.mu) ** 2 / (
            (1 - self.lamda) ** 2 * self.variance**2
        ) - 1 / self.variance
        H = H_1 * H_2

        self.variance = 1 / (1 / self.variance + self.eta * H)

        target = self.mu - self.eta * g * self.variance
        # set mu to the projection of target onto the interval * (1 - epsilon)
        self.mu = min(
            max(target, self.interval[0] * (1 - self.epsilon)),
            self.interval[1] * (1 - self.epsilon),
        )
