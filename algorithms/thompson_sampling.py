import numpy as np

from algorithms.learner import Learner
from sampling import sample_posterior, compute_minimizer

class ThompsonSamplingLearner(Learner):
    def __init__(self, prior_mu, prior_Sigma, B, n, sigma, sampler_type='coordinate', num_samples=1, sample_at_least_once=False):
        """
        Thompson Sampling learner which uses hit-and-run to sample a function from the posterior.
        """
        super().__init__(prior_mu, prior_Sigma, B, n, sigma)
        self.sampler_type = sampler_type
        self.name = f"ThompsonSampling ({sampler_type})"
        self.num_samples = num_samples
        self.sample_at_least_once = sample_at_least_once

    def select_action(self):
        """
        At time t, compute the posterior, sample a function from it, and return its minimizer.
        """
        post_mean, post_cov = self.get_posterior()
        # Sample one weight vector from the posterior.
        sampled_ws = sample_posterior(post_mean, post_cov, self.B, num_samples=self.num_samples, burn_in=500,
                                      init=post_mean, sampler_type=self.sampler_type, sample_at_least_once=self.sample_at_least_once)
        sampled_w = None
        if len(sampled_ws) == 0:
            print("Warning: no samples obtained.", self.sample_at_least_once)
            print(f"sampler_type: {self.sampler_type}")
            print("man ke daram migam behetoon")
        try:
            sampled_w = sampled_ws[0]
        except:
            raise ValueError("No samples obtained.")
        # Compute the minimizer of the sampled function.
        x_t = compute_minimizer(sampled_w, self.ts)
        return x_t