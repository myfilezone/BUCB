
import numpy as np

class UCBBV1:
    def __init__(self, n_arms: int, budget: float, M: int, c: np.ndarray, **kwargs):
        self.n_arms = n_arms
        self.budget = budget
        self.M = M
        self.c = c

        self.lambda_ = np.min(c) if len(c) > 0 else 1e-6

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

        T_safe = np.where(self.T == 0, 1, self.T)

        empirical_ratio = self.u_hat / self.c

        confidence_term = np.sqrt(np.log(self.t - 1) / T_safe)
        
        denominator = self.lambda_ - confidence_term
        safe_denominator = np.where(denominator <= 0, 1e-9, denominator)
        
        exploration_term = ((1 + 1 / self.lambda_) * confidence_term) / safe_denominator
        
        indices = empirical_ratio + exploration_term
        
        
        eligible_arms_mask = (self.M * self.c) <= remaining_budget
        if not np.any(eligible_arms_mask):
            return -1
            
        final_indices = np.where(eligible_arms_mask, indices, -np.inf)
        
        return np.argmax(final_indices)

    def update(self, arm_index: int, reward: float, cost: float):
        
        self.u_hat[arm_index] = (self.T[arm_index] * self.u_hat[arm_index] + (reward / self.M)) / (self.T[arm_index] + 1)

        self.T[arm_index] += 1
        self.t += 1