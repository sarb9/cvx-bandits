import numpy as np
import math

from algorithms.learner import Learner


class ExpWeights(Learner):

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
    ):
        super().__init__(
            interval,
            prior_mu,
            prior_Sigma,
            B,
            n_basis,
            noise_sigma,
            horizon,
            name,
        )

        self.Xs = []
        self.epsilon = 1 / np.sqrt(horizon)
        self.learning_rate = 1 / np.sqrt(horizon)

        # Discritize the interval with a grid of size 1/epsilon.
        self.l = np.arange(0, 1 + self.epsilon, self.epsilon)

        # Initialize the weights.
        self.q = np.zeros(len(self.l)) / len(self.l)
        self.p = np.zeros(len(self.l)) / len(self.l)
        self.T = np.zeros(shape=(len(self.l), len(self.l))) / len(self.l)
        self.s_hats = np.zeros((horizon, len(self.l)))

    def smaller_index(self, x: float) -> int:
        """ """
        return math.floor(x / self.epsilon)

    def select_action(self):
        """
        Select the action.
        """

        for x in range(len(self.l)):
            s = 0
            for t in range(len(self.Xs)):
                s += self.s_hats[t, x]
            self.q[x] = np.exp(-self.learning_rate * s)
        self.q = self.q / np.sum(self.q)

        mu = np.dot(self.q, self.l)
        pi_mu = self.smaller_index(mu)

        for y in range(len(self.l)):
            mx, mn = max(y, pi_mu), min(y, pi_mu)
            for x in range(len(self.l)):
                self.T[x, y] = 0 if x < mn or x > mx else 1 / (mx - mn + 1)

        self.p = self.T @ self.q

        assert np.isclose(np.sum(self.q), 1)
        assert np.isclose(np.sum(self.p), 1)
        for y in range(len(self.l)):
            assert np.isclose(np.sum(self.T[:, y]), 1)

        idx = np.random.choice(range(len(self.l)), p=self.p)
        action = self.l[idx]
        self.Xs.append(idx)

        return action

    def update(self, x, y):
        """
        Record the observation.
        """
        super().update(x, y)

        # Update s_hats
        time_step = len(self.Xs) - 1
        for j in range(len(self.l)):
            if self.p[self.Xs[-1]] == 0:
                self.s_hats[time_step, j] = 0
                continue
            self.s_hats[time_step, j] = self.T[self.Xs[-1], j] * y / self.p[self.Xs[-1]]
