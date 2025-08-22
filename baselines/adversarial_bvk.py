
import numpy as np

class _LagrangeBwK_Game:
    def __init__(self, n_arms, T0, budget, M, c, gamma_exp3=0.1):
        self.n_arms = n_arms
        self.T0 = T0
        self.budget = budget
        self.M = M
        self.c = c 
        self.gamma_exp3 = gamma_exp3
        self.weights = np.ones(n_arms)

    def choose_arm(self):
        total_weight = np.sum(self.weights)
        probabilities = (1 - self.gamma_exp3) * (self.weights / total_weight) + (self.gamma_exp3 / self.n_arms)
        chosen_arm = np.random.choice(self.n_arms, p=probabilities)
        return chosen_arm, probabilities[chosen_arm]

    def update_weights(self, arm_index, prob, reward, cost):
        norm_reward = (reward / self.M - 1) if self.M > 0 else 0
        
        payoff = norm_reward + 1 - (self.T0 / self.budget) * cost
        
        min_payoff = 0 + 1 - (self.T0 / self.budget) * (self.M * np.max(self.c))
        max_payoff = 1 + 1 - (self.T0 / self.budget) * (self.M * np.min(self.c))

        if max_payoff - min_payoff > 0:
            exp3_reward = (payoff - min_payoff) / (max_payoff - min_payoff)
        else:
            exp3_reward = 0.5
        exp3_reward = np.clip(exp3_reward, 0, 1)

        estimated_reward = exp3_reward / prob
        self.weights[arm_index] *= np.exp(self.gamma_exp3 * estimated_reward / self.n_arms)


class AdversarialBwK:
    """
    An implementation of the adversarial bandits with knapsacks algorithm
    from Immorlica et al. (2023).
    """
    def __init__(self, n_arms: int, budget: float, M: int, c: np.ndarray, **kwargs):
        self.n_arms = n_arms
        self.budget = budget
        self.M = M
        self.c = c
        self.t = 1
        self.chosen_prob = 0.0

        g_min = np.sqrt(budget) if budget > 1 else 1.0
        g_max = budget
        kappa = 2.0
        
        if g_max > g_min:
            u_max = np.log(g_max / g_min) / np.log(kappa)
            u = np.random.uniform(0, u_max)
            g_hat = g_min * (kappa ** u)
        else:
            g_hat = g_min
            
        T0 = g_hat / 2.0
        
        self.game = _LagrangeBwK_Game(n_arms, T0, budget, M, c)

    def select_arm(self, t: int, remaining_budget: float) -> int:
        arm_index, prob = self.game.choose_arm()
        self.chosen_prob = prob
        
        cost_per_pull = self.M * self.c[arm_index]

        if remaining_budget < cost_per_pull:
            return -1

        return arm_index

    def update(self, arm_index: int, reward: float, cost: float):
        self.game.update_weights(arm_index, self.chosen_prob, reward, cost)
        self.t += 1