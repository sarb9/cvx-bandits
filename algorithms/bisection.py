import numpy as np
import math

from algorithms.learner import Learner


class BisectionLearner(Learner):

    def __init__(self, initial_interval, horizon, delta, sigma, confidence_const=24):
        """
        BisectionLearner implements the noisy bisection method (Algorithms 4.2 and 4.3) for
        one-dimensional convex optimization.

        Parameters:
          initial_interval : [x, y] defining the initial search interval.
          n                : total query budget (n) for the bisection method.
          delta            : confidence parameter.
          sigma            : noise standard deviation.
          confidence_const : constant in the confidence bound (default: 24).

        Note:
          This learner does not use a Bayesian prior (and hence the usual prior_mu, prior_Sigma, B, n)
          so we call the parent Learner constructor with dummy values for consistency.
        """
        # TODO: Implement the constructor.
        # super().__init__(prior_mu=None, prior_Sigma=None, B=None, n=None, sigma=sigma)
        self.history_x = []
        self.history_y = []
        self.cum_regret = 0.0  # cumulative regret

        self.sigma = sigma
        self.name = "BisectionLearner"
        self.initial_interval = initial_interval
        self.delta = delta
        self.confidence_const = confidence_const
        self.horizon = horizon

        self.current_time_step = 1  # Current "REAL" time step.
        self.remaining_rounds = None  # Remaining rounds for the current interval.
        self.action_number = None

        self.K = np.copy(initial_interval)
        self.reset_bisection()

        self.k_max = 1 + math.ceil(math.log(self.horizon) / math.log(4 / 3))

    def reset_bisection(self):
        """
        Reset the bisection method.
        """
        self.x0 = self.K[0] * 0.75 + self.K[1] * 0.25
        self.x1 = self.K[0] * 0.5 + self.K[1] * 0.5
        self.x2 = self.K[0] * 0.25 + self.K[1] * 0.75
        self.bisect_round = 0  # Total rounds used so far.
        self.action_values = [[], [], []]
        self.action_values = [[], [], []]
        self.remaining_rounds = self.horizon - self.current_time_step

    def select_action(self):
        """
        Select an action using the bisection method.
        """
        action_number = self.bisect_round % 3
        self.action_number = action_number

        if action_number == 0:
            return self.x0
        elif action_number == 1:
            return self.x1
        else:
            return self.x2

    def update(self, x, y):
        """
        Record the observation.
        """
        super().update(x, y)

        # Update the action values.
        self.action_values[self.action_number].append(y)
        self.current_time_step += 1
        self.bisect_round += 1

        if self.bisect_round % 3 == 0 and self.bisect_round > 0:
            # Form sample averages.
            assert (
                len(self.action_values[0])
                == len(self.action_values[1])
                == len(self.action_values[2])
            )
            f_hat_0 = np.mean(self.action_values[0])
            f_hat_1 = np.mean(self.action_values[1])
            f_hat_2 = np.mean(self.action_values[2])

            # Confidence bound.
            c = self.sigma * np.sqrt(
                (self.confidence_const / self.bisect_round)
                * np.log((4 * self.remaining_rounds) / (3 * self.delta))
            )

            if (f_hat_2 - f_hat_1) >= c:
                self.K = [self.K[0], self.x2]
                self.reset_bisection()
            elif (f_hat_0 - f_hat_1) >= c:
                self.K = [self.x0, self.K[1]]
                self.reset_bisection()
