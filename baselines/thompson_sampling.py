
import numpy as np
from scipy.stats import invgamma

class ThompsonSampling:
    def __init__(self, n_arms: int, budget: float, M: int, c: np.ndarray, **kwargs):
        self.n_arms = n_arms
        self.budget = budget
        self.M = M
        self.c = c

        self.prior_mu = np.zeros(n_arms)
        self.prior_nu = np.ones(n_arms) * 0.1
        self.prior_alpha = np.ones(n_arms) * 1
        self.prior_beta = np.ones(n_arms) * 1

        self.pulls = np.zeros(n_arms)
        self.sum_x = np.zeros(n_arms)
        self.sum_x_sq = np.zeros(n_arms)

    def select_arm(self, t: int, remaining_budget: float) -> int:
        sampled_means = np.zeros(self.n_arms)

        for i in range(self.n_arms):
            
            n = self.pulls[i]
            if n == 0:
                post_mu = self.prior_mu[i]
                post_nu = self.prior_nu[i]
                post_alpha = self.prior_alpha[i]
                post_beta = self.prior_beta[i]
            else:
                mean_x = self.sum_x[i] / n
                post_nu = self.prior_nu[i] + n
                post_mu = (self.prior_nu[i] * self.prior_mu[i] + n * mean_x) / post_nu
                post_alpha = self.prior_alpha[i] + n / 2
                sum_sq_diff = self.sum_x_sq[i] - n * mean_x**2
                post_beta = self.prior_beta[i] + 0.5 * (sum_sq_diff + (n * self.prior_nu[i] / post_nu) * (mean_x - self.prior_mu[i])**2)

            
            sampled_var = invgamma.rvs(a=post_alpha, scale=post_beta)
            if sampled_var <= 0: sampled_var = 1e-9

            sampled_mean = np.random.normal(loc=post_mu, scale=np.sqrt(sampled_var / post_nu))
            sampled_means[i] = sampled_mean

        costs_per_pull = self.M * self.c
        eligible_arms_mask = costs_per_pull <= remaining_budget
        
        if not np.any(eligible_arms_mask):
            return -1 

        ratios = np.divide(sampled_means, self.c, out=np.full_like(self.c, -np.inf), where=self.c > 0)
        final_ratios = np.where(eligible_arms_mask, ratios, -np.inf)
        
        return np.argmax(final_ratios)

    def update(self, arm_index: int, reward: float, cost: float):
        self.pulls[arm_index] += 1
        self.sum_x[arm_index] += reward
        self.sum_x_sq[arm_index] += reward**2