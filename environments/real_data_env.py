
import os
import pandas as pd
import numpy as np
from .simulation_env import BaseEnvironment

class RealDataEnvironment(BaseEnvironment):
    def __init__(self, dataset_path: str, M: int, lambda_punishment: float = 1.0, **kwargs):
        self.dataset_path = dataset_path
        self.lambda_punishment = lambda_punishment
        
        self.arms, self.costs, self.session_data_pools = self._load_and_process_dataset()
        
        
        super().__init__(n_arms=len(self.arms), costs=self.costs, M=M)

    def _load_and_process_dataset(self):
        main_data_path = os.path.join(self.dataset_path, 'data.csv')
        logs_path = os.path.join(self.dataset_path, 'streaming_logs')
        
        if not os.path.exists(main_data_path):
            raise FileNotFoundError(f"Primary data file not found: {main_data_path}")
        if not os.path.exists(logs_path):
            raise FileNotFoundError(f"Log directory not found: {logs_path}")
            
        df_main = pd.read_csv(main_data_path)
        
        arms = sorted(df_main['abr'].unique())
        arm_map = {name: i for i, name in enumerate(arms)}

        session_data_pools = {arm: [] for arm in arms}

        for _, session_row in df_main.iterrows():
            arm_name = session_row['abr'].strip()
            log_file = session_row['streaming_log'].strip()
            log_file_path = os.path.join(logs_path, log_file)
            
            if not os.path.exists(log_file_path) or arm_name not in arm_map:
                continue
            
            df_log = pd.read_csv(log_file_path)
            
            
            avg_bitrate = df_log['video_bitrate'].mean()
            total_rebuffering = df_log['rebuffering_duration'].sum()
            calculated_reward = avg_bitrate 
            
            
            mos_utility = session_row['mos']
            
            
            cost = df_log['chunk_size'].sum() / (1024 ** 2) 

            
            session_data_pools[arm_name].append({
                'reward': calculated_reward,
                'utility': mos_utility,
                'cost': cost
            })

        
        avg_costs = np.zeros(len(arms))
        for arm_name, data_list in session_data_pools.items():
            if data_list:
                arm_idx = arm_map[arm_name]
                avg_costs[arm_idx] = np.mean([d['cost'] for d in data_list])
            else:
                avg_costs[arm_map[arm_name]] = np.inf

        return arms, avg_costs, session_data_pools

    def get_reward(self, arm_index: int, t=0) -> tuple[float, float]:
        arm_name = self.arms[arm_index]
        session_pool = self.session_data_pools[arm_name]
        
        if not session_pool:
            return 0.0, 0.0
        
        indices = np.random.choice(len(session_pool), size=self.M, replace=True)
        
        total_reward = 0.0
        total_utility = 0.0
        
        for idx in indices:
            session = session_pool[idx]
            total_reward += session['reward']
            total_utility += session['utility']
            
        return total_reward, total_utility