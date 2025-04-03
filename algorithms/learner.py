import numpy as np


class Learner:

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
        """
        Base learner class. Stores prior information and history of actions/observations.
        """
        self.interval = interval
        self.prior_mu = prior_mu
        self.prior_Sigma = prior_Sigma
        self.B = B
        self.n_basis = n_basis
        self.noise_sigma = noise_sigma
        self.horizon = horizon
        self.name = name

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
