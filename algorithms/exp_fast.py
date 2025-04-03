import numpy as np
import math

from algorithms.learner import Learner


class ExpWeightsFast(Learner):

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
            interval, prior_mu, prior_Sigma, B, n_basis, noise_sigma, horizon, name
        )
        self.epsilon = 1 / np.sqrt(horizon)
        self.learning_rate = 1 / np.sqrt(horizon)

        # Discritize the interval with a grid of size 1/epsilon.
        self.l = np.arange(0, 1 + self.epsilon, self.epsilon)

        # Initialize the weights.
        self.s_hats = np.zeros(len(self.l))
        self.pi_mu = None
        self.X = None
        self.pX_given_y = None
        self.q = None

    def smaller_index(self, x: float) -> int:
        return math.floor(x / self.epsilon)

    def select_action(self):
        q = np.ones(len(self.l))
        for x in range(len(self.l)):
            q[x] *= np.exp(-self.learning_rate * self.s_hats[x])
        q = q / np.sum(q)

        # sample y from q
        idx_y = np.random.choice(range(len(self.l)), p=q)

        mu = np.dot(q, self.l)
        pi_mu = self.smaller_index(mu)

        # Just compute T[:, y]
        mn = min(idx_y, pi_mu)
        mx = max(idx_y, pi_mu)
        idx_x = np.random.choice(range(mn, mx + 1))
        self.pX_given_y = 1 / (mx - mn + 1)
        self.pi_mu = pi_mu
        self.X = idx_x
        self.q = q

        action = self.l[idx_x]

        return action

    def update(self, x, y):
        """
        Record the observation.
        """
        super().update(x, y)
        print(f"u_hat: {self.pi_mu * self.epsilon}")

        # Update s_hats
        p = 0
        for j in range(len(self.l)):
            mn, mx = min(j, self.pi_mu), max(j, self.pi_mu)
            t = 0 if self.X < mn or self.X > mx else 1 / (mx - mn + 1)
            p += t * self.q[j]

            self.s_hats[j] = t * y
        self.s_hats /= p
