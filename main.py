import numpy as np
import seaborn as sns

from bandit import BanditFactory
from runner import Runner
from algorithms.thompson_sampling import ThompsonSamplingLearner
from algorithms.bisection import BisectionLearner
from algorithms.exp_weights import ExpWeights
from algorithms.exp_fast import ExpWeightsFast
from algorithms.ons import ONS


sns.set(style="whitegrid", context="talk", font_scale=1.1)

if __name__ == "__main__":
    # Define model parameters.
    n = 100  # number of basis functions
    B = 5.0  # budget on sum of weights
    sigma = 0.1  # noise standard deviation
    horizon = 100  # number of time steps per trial
    num_trials = 8  # number of independent trials

    # Prior for weights: choose a random mean (normalized) and identity covariance.
    np.random.seed(42)
    prior_mu = np.random.uniform(0, 1, n)
    B_mu = np.random.uniform(0, 1) * B
    prior_mu = prior_mu / np.sum(prior_mu) * B_mu
    prior_Sigma = np.eye(n) * 1.0

    # Create a BanditFactory.
    factory = BanditFactory(prior_mu, prior_Sigma, B, n, sigma)

    # Define the learner(s) to test.
    learner_configs = [
        (
            "ThompsonSampling",
            ThompsonSamplingLearner,
            {
                "prior_mu": prior_mu,
                "prior_Sigma": prior_Sigma,
                "B": B,
                "n": n,
                "noise_sigma": sigma,
                "burn_in": 1000,
            },
        ),
        (
            "BisectionLearner",
            BisectionLearner,
            {
                "initial_interval": [0, 1],
                "horizon": horizon,
                "delta": 1 / np.sqrt(horizon),
                "sigma": sigma,
                "confidence_const": 24,
            },
        ),
        (
            "ExpWeights",
            ExpWeights,
            {"horizon": horizon},
        ),
        (
            "ExpWeightsFast",
            ExpWeightsFast,
            {"horizon": horizon},
        ),
        (
            "ONS",
            ONS,
            {
                "horizon": horizon,
                "interval": [-1, 1],
                "delta": 1 / np.sqrt(horizon),
                "M": 1,
                "C": 1,
            },
        ),
    ]

    # Create and run the Runner.
    runner = Runner(factory, learner_configs, horizon, num_trials)
    regrets = runner.run()
    runner.plot_regret(individual=False)
