
import numpy as np

class Exp3MB:
    def __init__(self, n_arms: int, budget: float, M: int, c: np.ndarray, gamma: float = 0.1, is_simulation: bool = True, **kwargs):
        self.n_arms = n_arms
        self.budget = budget
        self.M = M
        self.c = c
        self.gamma = gamma

        
        self.is_simulation = is_simulation
        self.weights = np.ones(n_arms)
        self.chosen_prob = 0.0 

    def select_arm(self, t: int, remaining_budget: float) -> int:
        
        total_weight = np.sum(self.weights)
        probabilities = (1 - self.gamma) * (self.weights / total_weight) + (self.gamma / self.n_arms)
        
        
        eligible_arms_mask = (self.M * self.c) <= remaining_budget
        if not np.any(eligible_arms_mask):
            return -1

        
        eligible_probs = probabilities * eligible_arms_mask
        
        
        if np.sum(eligible_probs) == 0:
            eligible_indices = np.where(eligible_arms_mask)[0]
            if len(eligible_indices) == 0:
                return -1
            return np.random.choice(eligible_indices)

        
        normalized_probs = eligible_probs / np.sum(eligible_probs)
        
        chosen_arm = np.random.choice(self.n_arms, p=normalized_probs)
        self.chosen_prob = probabilities[chosen_arm] 
        
        return chosen_arm

    def update(self, arm_index: int, reward: float, cost: float):
        min_payoff = self.M * (1 - np.max(self.c))
        max_payoff = self.M * (2 - np.min(self.c))
        payoff = reward - cost
        
        if max_payoff - min_payoff > 0:
            normalized_payoff = (payoff - min_payoff) / (max_payoff - min_payoff)
        else:
            normalized_payoff = 0.5 
        
        normalized_payoff = np.clip(normalized_payoff, 0, 1)

        
        estimated_payoff = normalized_payoff / self.chosen_prob
        
        
        
        self.weights[arm_index] *= np.exp(self.gamma * estimated_payoff / self.n_arms)