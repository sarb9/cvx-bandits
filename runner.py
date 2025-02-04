import concurrent.futures

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##############################
#          RUNNER            #
##############################


class Runner:
    def __init__(
        self, bandit_factory, learner_configs, horizon, num_trials=10, problem=None
    ):
        """
        bandit_factory: an instance of BanditFactory.
        learner_configs: a list of tuples, each of the form
                         (learner_name, learner_class, learner_kwargs)
                         where learner_kwargs is a dict of parameters.
        horizon: number of time steps per trial.
        num_trials: number of independent trials to average over.
        """
        self.bandit_factory = bandit_factory
        self.learner_configs = learner_configs
        self.horizon = horizon
        self.num_trials = num_trials
        self.results = {}  # to store cumulative regret curves for each learner
        self.problem = problem

        self.learners = []  # Just for visualization

    def run_trial(self, trial_id):
        """
        Runs a single bandit experiment trial.
        Returns the cumulative regret curve for this trial.
        """
        # Create a new problem instance from the factory
        problem = (
            self.bandit_factory.create_problem()
            if self.problem is None
            else self.problem
        )

        # Instantiate learners for this trial
        learners = {}
        for name, learner_class, learner_kwargs in self.learner_configs:
            learners[name] = learner_class(**learner_kwargs)

        print("Optimal action and value: ", problem.compute_optimum())

        # Store cumulative regret per learner for this trial
        trial_regrets = {name: np.zeros(self.horizon) for name in learners}

        # Run the trial for the specified horizon.
        for t in range(self.horizon):
            for name, learner in learners.items():
                x_t = learner.select_action()
                y_t = problem.query(x_t)
                learner.update(x_t, y_t)
                f_val = problem.f(np.array([x_t]))[0]
                inst_regret = (
                    f_val - problem.optimal_value
                )  # Compute expected regret instead of realized random regret
                learner.cum_regret += inst_regret
                trial_regrets[name][t] = learner.cum_regret

                if t % 10 == 0:
                    print(
                        f"Trial {trial_id + 1}/{self.num_trials}, Time step {t}/{self.horizon}, Learner {name}, Regret: {learner.cum_regret}"
                    )
        return trial_regrets, learners

    def run(self):
        # Initialize regret curves (averaged over trials)
        regrets = {name: [] for (name, _, _) in self.learner_configs}

        # for trial in range(self.num_trials):

        #     # Use ProcessPoolExecutor to parallelize the trials
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Prepare a list of futures, one for each trial
            futures = [
                executor.submit(self.run_trial, trial_id)
                for trial_id in range(self.num_trials)
            ]

            # Collect the results as they complete
            for future in concurrent.futures.as_completed(futures):
                trial_regrets, learners = future.result()
                for name, regret_curve in trial_regrets.items():
                    regrets[name].append(regret_curve)
                self.learners.append(learners)

                print(f"{trial_regrets.keys()} completed.")

        print("All trials completed.")
        # Save regrets in self.results
        self.results = regrets
        return regrets

    def plot_regret(self, individual=False):

        sns.set_theme(style="ticks", context="paper", font_scale=1)
        plt.figure(figsize=(10, 6))

        x_values = np.arange(1, self.horizon + 1)
        for name, curves in self.results.items():
            # curves is a list (or array) of cumulative regret arrays (shape: (horizon,))
            if individual:
                for trial_curve in curves:
                    plt.plot(x_values, trial_curve, alpha=0.3, label=name)
            else:
                avg_reg = np.mean(curves, axis=0)
                std_reg = np.std(curves, axis=0)
                plt.plot(x_values, avg_reg, label=name, linewidth=2)
                plt.fill_between(
                    x_values, avg_reg - std_reg, avg_reg + std_reg, alpha=0.2
                )

        plt.xlabel("Time Step")
        plt.ylabel("Cumulative Regret")
        plt.title("Cumulative Regret vs Time (Averaged over Trials with Std)")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
