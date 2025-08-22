
import numpy as np

class LogUCB:
    def __init__(self, n_arms: int, budget: float, M: int, c: np.ndarray, **kwargs):
        self.n_arms = n_arms
        self.budget = budget
        self.M = M
        self.c = c

        self.t = 1
        self.T = np.zeros(n_arms)
        
        self.u_hat = np.zeros(n_arms)
        
        self.y_hat = np.zeros(n_arms)

    def select_arm(self, t: int, remaining_budget: float) -> int:
        
        if self.t <= self.n_arms:
            arm_to_pull = self.t - 1
            cost_per_pull = self.M * self.c[arm_to_pull]
            if remaining_budget < cost_per_pull:
                eligible_arms_mask_init = (self.M * self.c) <= remaining_budget
                eligible_arms = np.where(eligible_arms_mask_init)[0]
                if len(eligible_arms) == 0: return -1
                return eligible_arms[np.argmin(self.c[eligible_arms])]
            return arm_to_pull

        
        T_safe = np.where(self.T == 0, 1, self.T)
        
        empirical_mean_utility = self.y_hat

        utility_range = np.log(2)
        exploration_bonus = utility_range * np.sqrt(2 * np.log(self.t) / T_safe)
        
        
        utility_ucb = empirical_mean_utility + exploration_bonus
        
        
        indices = np.divide(utility_ucb, self.c, out=np.full_like(self.c, -np.inf), where=self.c!=0)

        
        eligible_arms_mask = (self.M * self.c) <= remaining_budget
        if not np.any(eligible_arms_mask):
            return -1
            
        final_indices = np.where(eligible_arms_mask, indices, -np.inf)
        
        return np.argmax(final_indices)

    def update(self, arm_index: int, reward: float, cost: float):
        
        current_utility = np.log(reward) if reward > 0 else 0
        
        
        self.y_hat[arm_index] = (self.T[arm_index] * self.y_hat[arm_index] + current_utility) / (self.T[arm_index] + 1)
        
        
        self.u_hat[arm_index] = (self.T[arm_index] * self.u_hat[arm_index] + (reward / self.M)) / (self.T[arm_index] + 1)
        
        self.T[arm_index] += 1
        self.t += 1