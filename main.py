import os
import pandas as pd
import numpy as np
import datetime
import importlib
import inspect


def load_modules(package_name):
    modules = {}
    package_path = os.path.join(os.path.dirname(__file__), package_name)
    for filename in os.listdir(package_path):
        if filename.endswith('.py') and filename != '__init__.py':
            module_name = f"{package_name}.{filename[:-3]}"
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and obj.__module__ == module_name:
                    modules[name] = obj
    return modules

baselines = load_modules('baselines')
environments = load_modules('environments')

CONFIG = {
    'simulation': {
        'env_class': environments['SimulationEnvironment'],
        'env_params': {'M': 5, 'c_low': 1.0, 'c_high': 2.0, 'reward_low': 1.0, 'reward_high': 2.0},
        'n_arms': 30,
        'budgets': np.linspace(0, 10000, 21),
        'num_trials': 100,
        'algorithms': {
            'Oracle': {'class': baselines['Oracle'], 'requires_env': True},
            'BUCB': {'class': baselines['BUCB']},
            'BUCBLog': {'class': baselines['BUCBLog']},
            'BUCBLin': {'class': baselines['BUCBLin']},
            'UCBBV1': {'class': baselines['UCBBV1']},
            'UCBBV1Log': {'class': baselines['UCBBV1Log']},
            'UCBG': {'class': baselines['UCBG']},
            'UCBGLog': {'class': baselines['UCBGLog']},
            'ThompsonSampling': {'class': baselines['ThompsonSampling']},
            'ThompsonSamplingLog': {'class': baselines['ThompsonSamplingLog']},
            'IRSVZero': {'class': baselines['IRSVZero']},
            'IRSVZeroLog': {'class': baselines['IRSVZeroLog']},
            'IRSFH': {'class': baselines['IRSFH']},
            'IRSFHLog': {'class': baselines['IRSFHLog']},
            'AdversarialBwK': {'class': baselines['AdversarialBwK']},
            'AdversarialBwKLog': {'class': baselines['AdversarialBwKLog']},
            'Exp3MB': {'class': baselines['Exp3MB'], 'gamma': 0.1},
            'Exp3MBLog': {'class': baselines['Exp3MBLog'], 'gamma': 0.1},
        }
    },
    'heavytail_stationary': {
        'env_class': environments['HeavytailEnv'],
        'env_params': {'M': 5, 'c_low': 1.0, 'c_high': 2.0, 'reward_low': 1000, 'reward_high': 3000},
        'n_arms': 30,
        'budgets': np.linspace(0, 10000, 21),
        'num_trials': 100,
        'algorithms': {
            'Oracle': {'class': baselines['Oracle'], 'requires_env': True},
            'BUCB': {'class': baselines['BUCB']},
            'BUCBLog': {'class': baselines['BUCBLog']},
            'BUCBLin': {'class': baselines['BUCBLin']},
            'UCBBV1': {'class': baselines['UCBBV1']},
            'UCBBV1Log': {'class': baselines['UCBBV1Log']},
            'UCBG': {'class': baselines['UCBG']},
            'UCBGLog': {'class': baselines['UCBGLog']},
            'ThompsonSampling': {'class': baselines['ThompsonSampling']},
            'ThompsonSamplingLog': {'class': baselines['ThompsonSamplingLog']},
            'IRSVZero': {'class': baselines['IRSVZero']},
            'IRSVZeroLog': {'class': baselines['IRSVZeroLog']},
            'IRSFH': {'class': baselines['IRSFH']},
            'IRSFHLog': {'class': baselines['IRSFHLog']},
            'AdversarialBwK': {'class': baselines['AdversarialBwK']},
            'AdversarialBwKLog': {'class': baselines['AdversarialBwKLog']},
            'Exp3MB': {'class': baselines['Exp3MB'], 'gamma': 0.1},
            'Exp3MBLog': {'class': baselines['Exp3MBLog'], 'gamma': 0.1},
        }
    },
    'heavytail_nonstationary': {
        'env_class': environments['NonStationaryHeavytailEnv'],
        'env_params': {'M': 5, 'c_low': 1.0, 'c_high': 2.0, 'reward_low': 1000, 'reward_high': 3000},
        'n_arms': 30,
        'budgets': np.linspace(0, 10000, 21),
        'num_trials': 100,
        'algorithms': {
            'Oracle': {'class': baselines['Oracle'], 'requires_env': True},
            'BUCB': {'class': baselines['BUCB']},
            'BUCBLog': {'class': baselines['BUCBLog']},
            'BUCBLin': {'class': baselines['BUCBLin']},
            'UCBBV1': {'class': baselines['UCBBV1']},
            'UCBBV1Log': {'class': baselines['UCBBV1Log']},
            'UCBG': {'class': baselines['UCBG']},
            'UCBGLog': {'class': baselines['UCBGLog']},
            'ThompsonSampling': {'class': baselines['ThompsonSampling']},
            'ThompsonSamplingLog': {'class': baselines['ThompsonSamplingLog']},
            'IRSVZero': {'class': baselines['IRSVZero']},
            'IRSVZeroLog': {'class': baselines['IRSVZeroLog']},
            'IRSFH': {'class': baselines['IRSFH']},
            'IRSFHLog': {'class': baselines['IRSFHLog']},
            'AdversarialBwK': {'class': baselines['AdversarialBwK']},
            'AdversarialBwKLog': {'class': baselines['AdversarialBwKLog']},
            'Exp3MB': {'class': baselines['Exp3MB'], 'gamma': 0.1},
            'Exp3MBLog': {'class': baselines['Exp3MBLog'], 'gamma': 0.1},
        }
    },
    'multimodal_stationary': {
        'env_class': environments['MultimodalEnv'],
        'env_params': {'M': 5, 'c_low': 1.0, 'c_high': 2.0, 'reward_low': 1000, 'reward_high': 3000},
        'n_arms': 30,
        'budgets': np.linspace(0, 10000, 21),
        'num_trials': 100,
        'algorithms': {
            'Oracle': {'class': baselines['Oracle'], 'requires_env': True},
            'BUCB': {'class': baselines['BUCB']},
            'BUCBLog': {'class': baselines['BUCBLog']},
            'BUCBLin': {'class': baselines['BUCBLin']},
            'UCBBV1': {'class': baselines['UCBBV1']},
            'UCBBV1Log': {'class': baselines['UCBBV1Log']},
            'UCBG': {'class': baselines['UCBG']},
            'UCBGLog': {'class': baselines['UCBGLog']},
            'ThompsonSampling': {'class': baselines['ThompsonSampling']},
            'ThompsonSamplingLog': {'class': baselines['ThompsonSamplingLog']},
            'IRSVZero': {'class': baselines['IRSVZero']},
            'IRSVZeroLog': {'class': baselines['IRSVZeroLog']},
            'IRSFH': {'class': baselines['IRSFH']},
            'IRSFHLog': {'class': baselines['IRSFHLog']},
            'AdversarialBwK': {'class': baselines['AdversarialBwK']},
            'AdversarialBwKLog': {'class': baselines['AdversarialBwKLog']},
            'Exp3MB': {'class': baselines['Exp3MB'], 'gamma': 0.1},
            'Exp3MBLog': {'class': baselines['Exp3MBLog'], 'gamma': 0.1},
        }
    },
    'multimodal_nonstationary': {
        'env_class': environments['NonStationaryMultimodalEnv'],
        'env_params': {'M': 5, 'c_low': 1.0, 'c_high': 2.0, 'reward_low': 1000, 'reward_high': 3000},
        'n_arms': 30,
        'budgets': np.linspace(0, 10000, 21),
        'num_trials': 100,
        'algorithms': {
            'Oracle': {'class': baselines['Oracle'], 'requires_env': True},
            'BUCB': {'class': baselines['BUCB']},
            'BUCBLog': {'class': baselines['BUCBLog']},
            'BUCBLin': {'class': baselines['BUCBLin']},
            'UCBBV1': {'class': baselines['UCBBV1']},
            'UCBBV1Log': {'class': baselines['UCBBV1Log']},
            'UCBG': {'class': baselines['UCBG']},
            'UCBGLog': {'class': baselines['UCBGLog']},
            'ThompsonSampling': {'class': baselines['ThompsonSampling']},
            'ThompsonSamplingLog': {'class': baselines['ThompsonSamplingLog']},
            'IRSVZero': {'class': baselines['IRSVZero']},
            'IRSVZeroLog': {'class': baselines['IRSVZeroLog']},
            'IRSFH': {'class': baselines['IRSFH']},
            'IRSFHLog': {'class': baselines['IRSFHLog']},
            'AdversarialBwK': {'class': baselines['AdversarialBwK']},
            'AdversarialBwKLog': {'class': baselines['AdversarialBwKLog']},
            'Exp3MB': {'class': baselines['Exp3MB'], 'gamma': 0.1},
            'Exp3MBLog': {'class': baselines['Exp3MBLog'], 'gamma': 0.1},
        }
    },
    'real_data': {
        'env_class': environments['RealDataEnvironment'],
        'env_params': {'M': 5, 'dataset_path': 'datasets/WaterlooSQoE-IV', 'lambda_punishment': 1000},  
        'n_arms': 5,
        'budgets': np.linspace(0, 10000, 21), 
        'num_trials': 100,
        'algorithms': {
            'Oracle': {'class': baselines['Oracle'], 'requires_env': True},
            'BUCB': {'class': baselines['BUCB']},
            'BUCBLog': {'class': baselines['BUCBLog']},
            'BUCBLin': {'class': baselines['BUCBLin']},
            'UCBBV1': {'class': baselines['UCBBV1']},
            'UCBBV1Log': {'class': baselines['UCBBV1Log']},
            'UCBG': {'class': baselines['UCBG']},
            'UCBGLog': {'class': baselines['UCBGLog']},
            'ThompsonSampling': {'class': baselines['ThompsonSampling']},
            'ThompsonSamplingLog': {'class': baselines['ThompsonSamplingLog']},
            'IRSVZero': {'class': baselines['IRSVZero']},
            'IRSVZeroLog': {'class': baselines['IRSVZeroLog']},
            'IRSFH': {'class': baselines['IRSFH']},
            'IRSFHLog': {'class': baselines['IRSFHLog']},
            'AdversarialBwK': {'class': baselines['AdversarialBwK']},
            'AdversarialBwKLog': {'class': baselines['AdversarialBwKLog']},
            'Exp3MB': {'class': baselines['Exp3MB'], 'gamma': 0.1},
            'Exp3MBLog': {'class': baselines['Exp3MBLog'], 'gamma': 0.1},
            
        }
    }
}

def run_single_instance(algorithm_class, budget, env, algo_params):
    n_arms = env.n_arms
    costs = env.get_arm_costs()
    M = env.M

    init_params = {
        'n_arms': n_arms, 'budget': budget, 'M': M, 'c': costs, **algo_params
    }
    
    
    if algo_params.get('requires_env', False):
        init_params['env'] = env
    
    
    init_params.pop('requires_env', None)
    
    algo = algorithm_class(**init_params)
    
    remaining_budget = budget
    total_utility = 0.0
    t = 1
    
    while True:
        arm_index = algo.select_arm(t, remaining_budget)
        if arm_index == -1: break

        reward, utility = env.get_reward(arm_index, t)
        
        cost = M * costs[arm_index]

        if remaining_budget < cost: break
            
        remaining_budget -= cost
        total_utility += utility
        
        algo.update(arm_index, reward, cost)
        t += 1
        
    
    return total_utility, t - 1
def main(experiment_name='simulation'):
    if experiment_name not in CONFIG:
        return
        
    exp_config = CONFIG[experiment_name]
    all_results = []

    oracle_results = {} 
    oracle_config = {'Oracle': exp_config['algorithms']['Oracle']}

    for trial in range(exp_config['num_trials']):
        env = exp_config['env_class'](
            n_arms=exp_config['n_arms'], 
            seed=trial if experiment_name == 'simulation' else None, 
            **exp_config['env_params']
        )
        exp_config['n_arms'] = env.n_arms
        
        for budget in exp_config['budgets']:
            
            optimal_utility, optimal_rounds = run_single_instance(
                algorithm_class=oracle_config['Oracle']['class'],
                budget=budget, env=env, algo_params=oracle_config['Oracle']
            )
            
            oracle_results[(trial, budget)] = (optimal_utility, optimal_rounds)


    algorithms_to_run = {k: v for k, v in exp_config['algorithms'].items() if k != 'Oracle'}

    for trial in range(exp_config['num_trials']):
        env = exp_config['env_class'](
            n_arms=exp_config['n_arms'], 
            seed=trial if experiment_name == 'simulation' else None, 
            **exp_config['env_params']
        )
        exp_config['n_arms'] = env.n_arms

        for budget in exp_config['budgets']:
            for algo_name, params in algorithms_to_run.items():
                
                total_utility, round_num = run_single_instance(
                    algorithm_class=params['class'], budget=budget,
                    env=env, algo_params=params
                )
                
                optimal_utility, _ = oracle_results.get((trial, budget), (0.0, 0))
                
                regret = optimal_utility - total_utility
 
                avg_utility_per_round = total_utility / round_num if round_num > 0 else 0.0
                avg_regret_per_round = regret / round_num if round_num > 0 else 0.0

                
                all_results.append({
                    'trial': trial,
                    'budget': budget,
                    'algorithm': algo_name,
                    'total_utility': total_utility,
                    'regret': regret,
                    'round_num': round_num,                 
                    'avg_utility': avg_utility_per_round,   
                    'avg_regret': avg_regret_per_round      
                })

    df_results = pd.DataFrame(all_results)
    
    if not os.path.exists('data'): os.makedirs('data')
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/{experiment_name}_results_summary_{timestamp}.csv"
    df_results.to_csv(filename, index=False)
    
    summary = df_results.groupby(['algorithm', 'budget']).agg(
        avg_total_utility=('total_utility', 'mean'),
        avg_total_regret=('regret', 'mean'),
        std_total_regret=('regret', 'std'),
        avg_round_num=('round_num', 'mean'),
        mean_of_avg_utility=('avg_utility', 'mean'),
        mean_of_avg_regret=('avg_regret', 'mean')
    ).reset_index()

    
    summary.rename(columns={
        'avg_total_utility': 'avg_utility',
        'avg_total_regret': 'avg_regret',
        'std_total_regret': 'std_regret',
        'mean_of_avg_utility': 'avg_per_round_utility',
        'mean_of_avg_regret': 'avg_per_round_regret'
    }, inplace=True)
    
    print(f"\n--- {experiment_name} result summary ---")
    print(summary)

if __name__ == '__main__':
    main(experiment_name='heavytail_stationary')
    
    
    
    
    