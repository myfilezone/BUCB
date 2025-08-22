

import numpy as np
from scipy.stats import norm, truncnorm, lognorm
import math

class BaseEnvironment:
    def __init__(self, n_arms: int, costs: np.ndarray, M: int):
        self.n_arms = n_arms
        self.c = costs  
        self.M = M      

    def get_arm_costs(self) -> np.ndarray:
        return self.c
        
    def get_reward(self, arm_index: int, t: int = 0) -> tuple[float, float]:
        raise NotImplementedError

    def get_true_expected_utility(self, arm_index: int) -> float:
        raise NotImplementedError

class SimulationEnvironment(BaseEnvironment):
    def __init__(self, n_arms: int, M: int, c_low: float = 1.0, c_high: float = 2.0, reward_low: float = 1.0, reward_high: float = 2.0, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        costs = np.random.uniform(low=c_low, high=c_high, size=n_arms)
        
        super().__init__(n_arms, costs, M)
        self.reward_low = reward_low
        self.reward_high = reward_high
        self.dist_dic = self._create_distributions(seed)

    def _create_distributions(self, seed):
        if seed is not None:
            np.random.seed(seed)
        mu = np.random.uniform(low=self.reward_low, high=self.reward_high, size=self.n_arms)
        if self.reward_low == 0:
            num_digits = 1
        else:
            num_digits = int(math.log10(abs(self.reward_low))) + 1

        sigma = np.full(self.n_arms, 5.0*num_digits)
        
        dist_dic = {}
        for i in range(self.n_arms):
            a, b = (self.reward_low - mu[i]) / sigma[i], (self.reward_high - mu[i]) / sigma[i]
            dist_dic[i] = truncnorm(a, b, loc=mu[i], scale=sigma[i])
        return dist_dic

    def get_reward(self, arm_index: int) -> tuple[float, float]:
        samples_sum = np.sum(self.dist_dic[arm_index].rvs(self.M))
        
        
        if samples_sum <= 0:
            return 0.0, -np.inf 
            
        utility = np.log(samples_sum)
        
        return samples_sum, utility


class HeavytailEnv(BaseEnvironment):
    def __init__(self, n_arms: int, M: int, c_low: float = 1.0, c_high: float = 2.0, reward_low: float = 1000, reward_high: float = 3000, seed=None, **kwargs):
        if n_arms < 5:
            raise ValueError("Need at leat 5 arms.")
        if seed is not None:
            np.random.seed(seed)
        
        self.reward_low = reward_low
        self.reward_high = reward_high
        self.dist_dic = self._create_arm_distributions(n_arms)

        reward_means = np.array([self.dist_dic[i].mean() for i in range(n_arms)])

        desc_reward_indices = np.argsort(reward_means)[::-1]

        costs = np.random.uniform(c_low, c_high, n_arms)
        sorted_costs = np.sort(costs)[::-1]

        costs = np.zeros(n_arms)
        
        costs[desc_reward_indices] = sorted_costs
        
        super().__init__(n_arms, costs, M)

        print("RealisticStationaryEnvironment initiated.")
        self._print_arm_summary()

    def _utility_function(self, total_reward: float) -> float:
        return np.where(total_reward <= 0, -np.inf, np.log1p(total_reward))
        
    def _create_arm_distributions(self, n_arms: int):
        dist_dic = {}
        for i in range(n_arms):
  
            mu = np.random.uniform(7.0, 8.0)  
            sigma = np.random.uniform(0.5, 1.2) 

            
            dist_dic[i] = lognorm(s=sigma, scale=np.exp(mu))
            
        return dist_dic

    def get_reward(self, arm_index: int, t: int = 0) -> tuple[float, float]:
        if arm_index not in self.dist_dic:
            raise ValueError(f"Arm {arm_index} not exist.")

        samples = self.dist_dic[arm_index].rvs(size=self.M)
        
        samples = np.clip(samples, self.reward_low, self.reward_high)

        mean_reward = np.mean(samples)

        total_reward = np.sum(samples)
        utility = self._utility_function(total_reward)
        
        return mean_reward, utility
    

    def get_true_expected_utility(self, arm_index: int, n_samples=100000) -> float:
            if arm_index not in self.dist_dic:
                raise ValueError(f"Arm {arm_index} not exist.")

            
            dist = self.dist_dic[arm_index]

            all_samples = dist.rvs(size=(n_samples, self.M))
            all_samples = np.clip(all_samples, self.reward_low, self.reward_high)

            total_rewards = np.sum(all_samples, axis=1) 

            utilities = self._utility_function(total_rewards)

            true_expected_utility = np.mean(utilities)
            
            return true_expected_utility

    def _print_arm_summary(self):
        print("\n" + "="*50)
        print("Summary of environmental arm theoretical properties:")
        print("="*50)
        print(f"{'Arm':<5} | {'Type':<18} | {'Cost':>6} | {'Reward Mean':>13} | {'Reward Std':>12} | {'Utility Approx':>16} | {'Efficiency (U/C)':>18}")
        print("-"*100)
        
        efficiencies = []
        for i in range(self.n_arms):
            mean = self.dist_dic[i].mean()
            std = self.dist_dic[i].std()
            cost = self.c[i]
            utility = self.get_true_expected_utility(i)
            efficiency = utility / cost if cost > 0 else np.inf
            efficiencies.append(efficiency)
            
        best_arm_by_efficiency = np.argmax(efficiencies)
        print("-"*100)
        print(f"Theoretically most efficient arm (Utility/Cost): Arm {best_arm_by_efficiency}")
        print("="*50 + "\n")


class NonStationaryHeavytailEnv(HeavytailEnv):
    def __init__(self, n_arms: int, M: int, c_low: float = 1.0, c_high: float = 2.0, reward_low: float = 1000, reward_high: float = 3000, seed=None, num_phases=3, total_budget=6000, **kwargs):
        if seed is not None:
            np.random.seed(seed)

        costs = np.random.uniform(low=c_low, high=c_high, size=n_arms)

        super(HeavytailEnv, self).__init__(n_arms, costs, M)

        self.reward_low = reward_low
        self.reward_high = reward_high

        total_time_horizon = 1500 
        self.num_phases = num_phases
        phase_duration = total_time_horizon // num_phases
        
        self.phase_starts = [i * phase_duration for i in range(num_phases)]
        
        self.phase_starts.append(np.inf)

        self.dist_phases = []
        for i in range(self.num_phases):
            phase_seed = seed + i if seed is not None else None
            if phase_seed is not None:
                np.random.seed(phase_seed)

            self.dist_phases.append(self._create_arm_distributions(n_arms))

        self._print_phase_summary() 

    def _determine_phase(self, t: int) -> int:
        phase = np.searchsorted(self.phase_starts, t, side='right') - 1
        return max(0, phase) 

    def get_reward(self, arm_index: int, t: int) -> tuple[float, float]:
        current_phase_index = self._determine_phase(t)
        current_dists = self.dist_phases[current_phase_index]

        if arm_index not in current_dists:
            raise ValueError(f"Arm {arm_index} does not exist at phase {current_phase_index}.")

        dist = current_dists[arm_index]
        samples = dist.rvs(size=self.M)
        samples = np.clip(samples, self.reward_low, self.reward_high)

        mean_reward = np.mean(samples)
        total_reward = np.sum(samples)
        utility = self._utility_function(total_reward) 

        return mean_reward, utility

    def get_true_expected_utility(self, arm_index: int, t: int = 0, n_samples=100000) -> float:
        current_phase_index = self._determine_phase(t)
        current_dists = self.dist_phases[current_phase_index]
        
        if arm_index not in current_dists:
            raise ValueError(f"Arm {arm_index} does not exist at phase {current_phase_index}.")

        
        dist = current_dists[arm_index]
        
        
        all_samples = dist.rvs(size=(n_samples, self.M))
        all_samples = np.clip(all_samples, self.reward_low, self.reward_high)
        
        total_rewards = np.sum(all_samples, axis=1)
        utilities = self._utility_function(total_rewards)
        
        
        true_expected_utility = np.mean(utilities)
        return true_expected_utility

    def _print_phase_summary(self):
        for i, dist_dic in enumerate(self.dist_phases):
            start_t = self.phase_starts[i]
            end_t = self.phase_starts[i+1] if self.phase_starts[i+1] != np.inf else "..."
            efficiencies = []
            for arm_idx in range(self.n_arms):
                mean = dist_dic[arm_idx].mean()
                cost = self.c[arm_idx]
                utility = self._utility_function(self.M * mean)
                efficiency = utility / cost if cost > 0 else np.inf
                efficiencies.append(efficiency)
                print(f"{arm_idx:<5} | {cost:8.2f} | {mean:13.2f} | {utility:16.2f} | {efficiency:18.2f}")


class MultimodalEnv(BaseEnvironment):
    def __init__(self, n_arms: int, M: int, c_low: float = 1.0, c_high: float = 2.0, reward_low: float = 1000, reward_high: float = 3000, seed=None, **kwargs):
        if n_arms <= 1:
            raise ValueError("Need at least 2 arms.")
        if seed is not None:
            np.random.seed(seed)
        
        self.reward_low = reward_low
        self.reward_high = reward_high
        
        self.dist_params_dic = self._create_arm_distributions(n_arms)

        reward_means = np.array([
            self._get_true_expected_reward(i) for i in range(n_arms)
        ])
        desc_reward_indices = np.argsort(reward_means)[::-1]
        
        costs = np.random.uniform(c_low, c_high, n_arms)
        sorted_costs = np.sort(costs)[::-1]

        final_costs = np.zeros(n_arms)
        final_costs[desc_reward_indices] = sorted_costs
        
        super().__init__(n_arms, final_costs, M)
        
        self._print_arm_summary()


    def _create_arm_distributions(self, n_arms: int):
        params_dic = {}
        for i in range(n_arms):
            num_peaks = 3   
            weights = np.random.dirichlet(np.ones(num_peaks), size=1)[0]
            means = np.random.uniform(self.reward_low, self.reward_high, size=num_peaks)
            stds = np.random.uniform(0.05*self.reward_low, 0.25*self.reward_low, size=num_peaks)
            
            params_dic[i] = {
                'weights': weights,
                'means': means,
                'stds': stds
            }
        return params_dic

    def _utility_function(self, total_reward: float) -> float:
        return np.where(total_reward <= 0, -np.inf, np.log(total_reward))

    def get_reward(self, arm_index: int, t: int = 0) -> tuple[float, float]:
        if arm_index not in self.dist_params_dic:
            raise ValueError(f"Arm {arm_index} not exist.")
        
        params = self.dist_params_dic[arm_index]
        component_idx = np.random.choice(len(params['weights']), p=params['weights'])
        
        mean = params['means'][component_idx]
        std = params['stds'][component_idx]
        
        a, b = (self.reward_low - mean) / std, (self.reward_high - mean) / std
        dist = truncnorm(a, b, loc=mean, scale=std)
        
        samples = dist.rvs(size=self.M)

        mean_reward = np.mean(samples)
        total_reward = np.sum(samples)
        utility = self._utility_function(total_reward)
        
        return mean_reward, utility

    def _get_true_expected_reward(self, arm_index: int) -> float:
        params = self.dist_params_dic[arm_index]
        total_expected_reward = 0
        for i in range(len(params['weights'])):
            mean, std = params['means'][i], params['stds'][i]
            a, b = (self.reward_low - mean) / std, (self.reward_high - mean) / std
            true_mean = truncnorm(a, b, loc=mean, scale=std).mean()
            total_expected_reward += params['weights'][i] * true_mean
        return total_expected_reward

    def get_true_expected_utility(self, arm_index: int, n_samples=100000) -> float:
        params = self.dist_params_dic[arm_index]
        
        
        total_rewards = np.zeros(n_samples)
        
        for i in range(n_samples):
            
            component_idx = np.random.choice(len(params['weights']), p=params['weights'])
            mean, std = params['means'][component_idx], params['stds'][component_idx]
            
            
            a, b = (self.reward_low - mean) / std, (self.reward_high - mean) / std
            dist = truncnorm(a, b, loc=mean, scale=std)
            samples = dist.rvs(size=self.M)
            total_rewards[i] = np.sum(samples)

        utilities = self._utility_function(total_rewards) 

        true_expected_utility = np.mean(utilities)
        
        return true_expected_utility

    def _print_arm_summary(self):
        efficiencies = []
        reward_means = []
        for i in range(self.n_arms):
            mean = self._get_true_expected_reward(i)
            reward_means.append(mean)
            cost = self.c[i]
            utility = self.get_true_expected_utility(i)
            efficiency = utility / cost if cost > 0 else np.inf
            efficiencies.append(efficiency)
            print(f"{i:<5} | {cost:6.2f} | {mean:13.2f} | {utility:16.2f} | {efficiency:18.2f}")


class NonStationaryMultimodalEnv(MultimodalEnv):
    def __init__(self, n_arms: int, M: int, c_low: float = 1.0, c_high: float = 2.0, reward_low: float = 1000, reward_high: float = 3000, seed=None, num_phases=3, total_budget=6000, **kwargs):
        self.reward_low = reward_low
        self.reward_high = reward_high
 
        if seed is not None:
            np.random.seed(seed)

        temp_params = self._create_arm_distributions(n_arms)
        temp_reward_means = np.array([
            self._get_true_expected_reward_from_params(p) for p in temp_params.values()
        ])
        desc_reward_indices = np.argsort(temp_reward_means)[::-1]
        
        costs = np.random.uniform(c_low, c_high, n_arms)
        sorted_costs = np.sort(costs)[::-1]
        final_costs = np.zeros(n_arms)
        final_costs[desc_reward_indices] = sorted_costs
        
        super(MultimodalEnv, self).__init__(n_arms, final_costs, M)

        total_time_horizon = 1500 
        self.num_phases = num_phases
        phase_duration = total_time_horizon // num_phases
        self.phase_starts = [i * phase_duration for i in range(num_phases)]
        self.phase_starts.append(np.inf)

        
        self.dist_phases = []
        for i in range(self.num_phases):
            phase_seed = seed + i if seed is not None else None
            if phase_seed is not None:
                np.random.seed(phase_seed)
            self.dist_phases.append(self._create_arm_distributions(n_arms))

        self._print_phase_summary() 

    def _determine_phase(self, t: int) -> int:
        phase = np.searchsorted(self.phase_starts, t, side='right') - 1
        return max(0, phase)

    def get_reward(self, arm_index: int, t: int) -> tuple[float, float]:
        current_phase_index = self._determine_phase(t)
        current_params_dic = self.dist_phases[current_phase_index]

        if arm_index not in current_params_dic:
            raise ValueError(f"Arm {arm_index} does not exist at phase {current_phase_index}.")
        
        params = current_params_dic[arm_index]
        component_idx = np.random.choice(len(params['weights']), p=params['weights'])
        
        mean = params['means'][component_idx]
        std = params['stds'][component_idx]
        
        a, b = (self.reward_low - mean) / std, (self.reward_high - mean) / std
        dist = truncnorm(a, b, loc=mean, scale=std)
        
        samples = dist.rvs(size=self.M)
        
        mean_reward = np.mean(samples)
        total_reward = np.sum(samples)
        utility = self._utility_function(total_reward)
        
        return mean_reward, utility
    
    
    def _get_true_expected_reward_from_params(self, params: dict) -> float:
        total_expected_reward = 0
        for i in range(len(params['weights'])):
            mean, std = params['means'][i], params['stds'][i]
            a, b = (self.reward_low - mean) / std, (self.reward_high - mean) / std
            true_mean = truncnorm(a, b, loc=mean, scale=std).mean()
            total_expected_reward += params['weights'][i] * true_mean
        return total_expected_reward

    def get_true_expected_utility(self, arm_index: int, t: int = 0, n_samples=100000) -> float:
        current_phase_index = self._determine_phase(t)
        current_params_dic = self.dist_phases[current_phase_index]

        if arm_index not in current_params_dic:
            raise ValueError(f"Arm {arm_index} does not exist at phase {current_phase_index}.")

        params = current_params_dic[arm_index]

        total_rewards = np.zeros(n_samples)
        for i in range(n_samples):
            component_idx = np.random.choice(len(params['weights']), p=params['weights'])
            mean, std = params['means'][component_idx], params['stds'][component_idx]
            
            a, b = (self.reward_low - mean) / std, (self.reward_high - mean) / std
            dist = truncnorm(a, b, loc=mean, scale=std)
            samples = dist.rvs(size=self.M)
            total_rewards[i] = np.sum(samples)

        utilities = self._utility_function(total_rewards)
        true_expected_utility = np.mean(utilities)
        
        return true_expected_utility

    def _print_phase_summary(self):
        for i, params_dic in enumerate(self.dist_phases):
            start_t = self.phase_starts[i]
            end_t = self.phase_starts[i+1] if self.phase_starts[i+1] != np.inf else "..."

            efficiencies = []
            for arm_idx in range(self.n_arms):
                params = params_dic[arm_idx]
                mean = self._get_true_expected_reward_from_params(params)
                cost = self.c[arm_idx]

                utility = self.get_true_expected_utility(arm_idx, t=start_t)
                efficiency = utility / cost if cost > 0 else np.inf
                efficiencies.append(efficiency)
                print(f"{arm_idx:<5} | {cost:8.2f} | {mean:13.2f} | {utility:16.2f} | {efficiency:18.2f}")
