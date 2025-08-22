

import numpy as np

import environments.simulation_env
import environments.real_data_env


from environments.simulation_env import (
    SimulationEnvironment, 
    HeavytailEnv, 
    NonStationaryHeavytailEnv,
    MultimodalEnv,
    NonStationaryMultimodalEnv
)

from environments.real_data_env import RealDataEnvironment


class Oracle:
    def __init__(self, n_arms: int, budget: float, M: int, c: np.ndarray, env, **kwargs):
        self.n_arms = n_arms
        self.budget = budget
        self.M = M
        self.c = c 
        self.env = env
        
    def _get_true_utility_cost_ratios_at_t(self, t: int):
        ratios = np.zeros(self.n_arms)

        
        

        if isinstance(self.env, (NonStationaryHeavytailEnv, NonStationaryMultimodalEnv)):
            
            
            true_utility = np.array([self.env.get_true_expected_utility(i, t) for i in range(self.n_arms)])
            ratios = np.divide(true_utility, self.c, out=np.full_like(self.c, -np.inf), where=self.c!=0)

        elif isinstance(self.env, (HeavytailEnv, MultimodalEnv)):
            
            true_utility = np.array([self.env.get_true_expected_utility(i) for i in range(self.n_arms)])
            ratios = np.divide(true_utility, self.c, out=np.full_like(self.c, -np.inf), where=self.c!=0)

        elif isinstance(self.env, RealDataEnvironment):
            
            for i, arm_name in enumerate(self.env.arms):
                pool = self.env.session_data_pools.get(arm_name, [])
                if pool:
                    mean_utility = np.mean([d['utility'] for d in pool])
                    mean_cost = self.c[i]
                    if mean_cost > 0:
                        ratios[i] = mean_utility / mean_cost
                    else:
                        ratios[i] = -np.inf
                else:
                    ratios[i] = -np.inf
        
        elif isinstance(self.env, SimulationEnvironment):
            
            true_mu = np.array([dist.mean() for dist in self.env.dist_dic.values()])
            
            
            expected_total_reward = self.M * true_mu
            
            safe_rewards = np.where(expected_total_reward > 0, expected_total_reward, -np.inf)
            true_utility = np.log(safe_rewards) 
            
            ratios = np.divide(true_utility, self.c, out=np.full_like(self.c, -np.inf), where=self.c!=0)
            
        else:
            raise TypeError(f"Oracle cannot identify the current env type: {type(self.env)}")
        
        return ratios

    def select_arm(self, t: int, remaining_budget: float) -> int:
        current_ratios = self._get_true_utility_cost_ratios_at_t(t)
        if not np.any(np.isfinite(current_ratios)):
            return -1

        best_arm_at_t = np.argmax(current_ratios)
        
        
        cost_multiplier = self.M
        
        cost_of_best_arm = cost_multiplier * self.c[best_arm_at_t]
        if remaining_budget >= cost_of_best_arm:
            return best_arm_at_t
        else:
            
            costs_per_pull = cost_multiplier * self.c
            eligible_arms_mask = (costs_per_pull <= remaining_budget)
            
            if not np.any(eligible_arms_mask):
                return -1

            eligible_ratios = np.where(eligible_arms_mask, current_ratios, -np.inf)
            if not np.any(np.isfinite(eligible_ratios)):
                return -1

            return np.argmax(eligible_ratios)

    def update(self, arm_index: int, reward: float, cost: float):
        pass