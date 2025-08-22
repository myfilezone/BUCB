
import numpy as np

class EpsilonGreedy:
    def __init__(self, n_arms: int, budget: float, M: int, c: np.ndarray, epsilon: float = 0.1, **kwargs):
        self.n_arms = n_arms
        self.budget = budget
        self.M = M
        self.c = c
        self.epsilon = epsilon
        self.adaptive = (epsilon == 0)

        
        self.t = 1
        self.T = np.zeros(n_arms)
        self.u_hat = np.zeros(n_arms)

    def select_arm(self, t: int, remaining_budget: float) -> int:
        
        if self.t <= self.n_arms:
            arm_to_pull = self.t - 1
            if remaining_budget < self.M * self.c[arm_to_pull]:
                eligible_arms = np.where(self.M * self.c <= remaining_budget)[0]
                if len(eligible_arms) == 0: return -1
                return eligible_arms[np.argmin(self.c[eligible_arms])]
            return arm_to_pull

        
        rate = 1.0 / self.t if self.adaptive else self.epsilon

        
        if np.random.random() < rate:
            
            eligible_arms = np.where(self.M * self.c <= remaining_budget)[0]
            if len(eligible_arms) == 0:
                return -1
            return np.random.choice(eligible_arms)
        else:
            
            eligible_arms_mask = (self.M * self.c) <= remaining_budget
            if not np.any(eligible_arms_mask):
                return -1
            
            
            u_hat_safe = np.where(self.u_hat <= 0, 1e-9, self.u_hat)
            
            log_utility = np.log(u_hat_safe)
            ratios = np.where(eligible_arms_mask, log_utility / self.c, -np.inf)
            return np.argmax(ratios)

    def update(self, arm_index: int, reward: float, cost: float):
        self.u_hat[arm_index] = (self.T[arm_index] * self.M * self.u_hat[arm_index] + reward) / ((self.T[arm_index] + 1) * self.M)
        self.T[arm_index] += 1
        self.t += 1