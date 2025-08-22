
import numpy as np

class ETC:
    def __init__(self, n_arms: int, budget: float, M: int, c: np.ndarray, explore_rate: float = 0.1, **kwargs):
        self.n_arms = n_arms
        self.budget = budget
        self.M = M
        self.c = c
        self.explore_budget = budget * explore_rate

        
        self.t = 1
        self.T = np.zeros(n_arms)
        self.u_hat = np.zeros(n_arms)
        self.total_expense = 0
        self.best_arm_commit = -1 

    def select_arm(self, t: int, remaining_budget: float) -> int:
        
        if self.total_expense < self.explore_budget:
            arm_to_pull = (self.t - 1) % self.n_arms
            if remaining_budget < self.M * self.c[arm_to_pull]:
                
                eligible_arms = np.where(self.M * self.c <= remaining_budget)[0]
                if len(eligible_arms) == 0: return -1
                return np.random.choice(eligible_arms)
            return arm_to_pull
        
        
        else:
            
            if self.best_arm_commit == -1:
                u_hat_safe = np.where(self.u_hat <= 0, 1e-9, self.u_hat)
                
                log_utility = np.log(u_hat_safe)
                ratios = log_utility / self.c
                self.best_arm_commit = np.argmax(ratios)
            
            
            if remaining_budget < self.M * self.c[self.best_arm_commit]:
                return -1 
            
            return self.best_arm_commit


    def update(self, arm_index: int, reward: float, cost: float):
        self.total_expense += cost
        self.u_hat[arm_index] = (self.T[arm_index] * self.M * self.u_hat[arm_index] + reward) / ((self.T[arm_index] + 1) * self.M)
        self.T[arm_index] += 1
        self.t += 1