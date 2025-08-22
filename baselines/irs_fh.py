
import numpy as np
from scipy.stats import invgamma

class IRSFH:
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
        future_expected_rewards = np.zeros(self.n_arms)

        for i in range(self.n_arms):
            
            n = self.pulls[i]
            if n == 0:
                post_mu, post_nu, post_alpha, post_beta = self.prior_mu[i], self.prior_nu[i], self.prior_alpha[i], self.prior_beta[i]
            else:
                mean_x = self.sum_x[i] / n
                post_nu = self.prior_nu[i] + n
                post_mu = (self.prior_nu[i] * self.prior_mu[i] + n * mean_x) / post_nu
                post_alpha = self.prior_alpha[i] + n / 2
                sum_sq_diff = self.sum_x_sq[i] - n * mean_x**2
                post_beta = self.prior_beta[i] + 0.5 * (sum_sq_diff + (n * self.prior_nu[i] / post_nu) * (mean_x - self.prior_mu[i])**2)
            
            
            sampled_var = invgamma.rvs(a=post_alpha, scale=post_beta)
            sampled_mean = np.random.normal(loc=post_mu, scale=np.sqrt(sampled_var / post_nu))

            
            cost_per_pull = self.M * self.c[i]
            if cost_per_pull <= 0:
                future_expected_rewards[i] = -np.inf
                continue
            
            n_future_pulls = int(np.floor(remaining_budget / cost_per_pull))

            if n_future_pulls <= 0:
                future_expected_rewards[i] = sampled_mean 
                continue

            
            simulated_rewards = np.random.normal(loc=sampled_mean, scale=np.sqrt(sampled_var), size=n_future_pulls)

            future_pulls = n + n_future_pulls
            future_sum_x = self.sum_x[i] + np.sum(simulated_rewards)
            future_expected_rewards[i] = future_sum_x / future_pulls

        
        eligible_arms_mask = (self.M * self.c) <= remaining_budget
        if not np.any(eligible_arms_mask):
            return -1

        ratios = np.divide(future_expected_rewards, self.c, out=np.full_like(self.c, -np.inf), where=self.c > 0)
        final_ratios = np.where(eligible_arms_mask, ratios, -np.inf)
        
        return np.argmax(final_ratios)

    def update(self, arm_index: int, reward: float, cost: float):
        self.pulls[arm_index] += 1
        self.sum_x[arm_index] += reward
        self.sum_x_sq[arm_index] += reward**2