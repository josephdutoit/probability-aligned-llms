import os
import gymnasium as gym
from gymnasium import spaces
import polars as pl
import numpy as np
import re

QUESTION = f"On {{}}, the home team {{}} plays against the away team {{}} in the {{}}.\
             What is the probability that {{}} wins the match?"

LEAGUE_MAP = {
    'E0': 'English Premier League',
    'E1': 'English Championship',
    'E2': 'English League One', 
    'E3': 'English League Two',
    'SP1': 'Spanish La Liga',
    'SP2': 'Spanish Segunda Division',
    'D1': 'German Bundesliga',
    'D2': 'German 2. Bundesliga',
    'I1': 'Italian Serie A',
    'I2': 'Italian Serie B',
    'F1': 'French Ligue 1',
    'F2': 'French Ligue 2',
}

COLUMNS = [
    'Div', 'Date', 'HomeTeam', 'AwayTeam',
]

class BettingEnv(gym.Env):
    def __init__(self, config):
        super(BettingEnv, self).__init__()
    
        self.data = pl.DataFrame()
        for file_name in os.listdir(config['data_dir']):
            if file_name.endswith(f".parquet"):
                lazy_df = pl.scan_parquet(os.path.join(config['data_dir'], file_name))
                df = lazy_df.select(COLUMNS).collect()
                self.data = pl.concat([self.data, df])

        self.action_space = spaces.Tuple((
            spaces.Text(max_length=config['max_action_length']),
            spaces.Text(max_length=config['max_action_length']),
        ))

        self.observation_space = spaces.Tuple((
            spaces.Text(max_length=config['max_observation_length']),
            spaces.Text(max_length=config['max_observation_length']),
        ))

        self.indices = np.arange(self.data.height)
        np.random.shuffle(self.indices)
        self.current_index = 0
        self.current_observation = None


    def step(self, action):
        reward = self._calculate_reward(action)
        info = {}
        return self.current_observation, reward, True, info
    

    def reset(self):
        idx = self.indices[self.current_index]
        row = self.data[idx]
        self.current_index = (self.current_index + 1) % len(self.indices)
        q1 = self._make_question(row, row['HomeTeam'].item())
        q2 = self._make_question(row, row['AwayTeam'].item())
        self.current_observation = (q1, q2)
        return self.current_observation, {}
    

    def _get_probability(self, action_text: str) -> float:
        pattern = r'(\d+(?:\.\d+)?)%?'
        matches = re.findall(pattern, action_text)
        if not matches:
            raise ValueError(f"No probability found in: {action_text}")
        
        last_match = matches[-1]
        if '%' in last_match:
            return float(last_match.rstrip('%')) / 100.0
        else:
            return float(last_match)
    
    def _calculate_reward(self, action):
        try:
            prob1 = self._get_probability(action[0])
            prob2 = self._get_probability(action[1])
        except (ValueError, IndexError):
            return -1.0  
        
        if prob1 < 0.0 or prob1 > 1.0 or prob2 < 0.0 or prob2 > 1.0:
            return -1.0
        if prob1 + prob2 == 1.0:
            return 1.0
        return 0.0

    
    def _make_question(self, row, team):
        return QUESTION.format(
            row['Date'].item(),
            row['HomeTeam'].item(),
            row['AwayTeam'].item(),
            LEAGUE_MAP.get(row['Div'].item(), 'Unknown League'),
            team,
        )


