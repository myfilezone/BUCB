
import numpy as np
from scipy.stats import invgamma

class IRSVZeroLog:
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

    def _solve_knapsack_dp_heuristic(self, B_t, arm_rewards_sequences):
        best_val = -np.inf
        best_arm = -1
        costs_per_pull = self.M * self.c
        budget_float = float(B_t)

        for i in range(self.n_arms):
             cost_i = costs_per_pull[i]
             if cost_i > 0 and len(arm_rewards_sequences[i]) > 0:
                 num_pulls = int(np.floor(budget_float / cost_i))
                 num_pulls = min(len(arm_rewards_sequences[i]), num_pulls)
                 
                 if num_pulls > 0:
                     
                     future_mean_reward_seq = np.array(arm_rewards_sequences[i][:num_pulls])
                     
                     
                     safe_rewards = np.maximum(-1/self.M + 1e-9, future_mean_reward_seq)
                    
                     log_utilities = np.log1p(safe_rewards)
                     total_log_utility = np.sum(log_utilities)
                     
                     current_val = total_log_utility / (num_pulls * cost_i)

                     if current_val > best_val:
                         best_val = current_val
                         best_arm = i
        return best_arm

    def select_arm(self, t: int, remaining_budget: float) -> int:
        arm_rewards_sequences = {}
        costs_per_pull = self.M * self.c

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
            
            
            cost_per_pull = costs_per_pull[i]
            max_pulls = int(np.floor(remaining_budget / cost_per_pull)) if cost_per_pull > 0 else 0
            
            future_rewards_samples = []
            if max_pulls > 0:
                future_rewards_samples = np.random.normal(loc=sampled_mean, scale=np.sqrt(sampled_var), size=max_pulls)

            
            rewards_seq = []
            current_pulls = self.pulls[i]
            current_sum_x = self.sum_x[i]
            for j in range(max_pulls):
                pred_mean_reward = current_sum_x / current_pulls if current_pulls > 0 else self.prior_mu[i]
                rewards_seq.append(pred_mean_reward)
                
                if j < len(future_rewards_samples):
                    current_pulls += 1
                    current_sum_x += future_rewards_samples[j]

            arm_rewards_sequences[i] = rewards_seq

        
        eligible_arms_mask = costs_per_pull <= remaining_budget
        if not np.any(eligible_arms_mask):
            return -1

        best_arm = self._solve_knapsack_dp_heuristic(remaining_budget, arm_rewards_sequences)

        if best_arm == -1:
             eligible_arms = np.where(eligible_arms_mask)[0]
             return np.random.choice(eligible_arms) if len(eligible_arms) > 0 else -1

        return best_arm

    def update(self, arm_index: int, reward: float, cost: float):
        self.pulls[arm_index] += 1
        self.sum_x[arm_index] += reward
        self.sum_x_sq[arm_index] += reward**2