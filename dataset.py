import os
import polars as pl
from torch.utils.data import Dataset

QUESTION = f"On {{}}, the home team {{}} plays against the away team {{}} in the {{}}. What is the probability that {{}} wins the match?"

# Dataset should load in data and return
class SoccerDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
    ):
        
        self.columns = [
            'Div', 'Date', 'HomeTeam', 'AwayTeam',
        ]

        self.league_map = {
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

        self.data = pl.DataFrame()
        for file_name in os.listdir(data_dir):
            if file_name.endswith(f".parquet"):
                lazy_df = pl.scan_parquet(os.path.join(data_dir, file_name))
                df = lazy_df.select(
                    self.columns
                ).collect()
                self.data = pl.concat([self.data, df])

    def __len__(self):
        return self.data.height
    
    def __getitem__(self, idx):
        row = self.data[idx]
        q1 = QUESTION.format(
            row['Date'].item(),
            row['HomeTeam'].item(),
            row['AwayTeam'].item(),
            self.league_map.get(row['Div'].item(), 'Unknown League'),
            row['HomeTeam'].item(),
        )
        q2 = QUESTION.format(
            row['Date'].item(),
            row['HomeTeam'].item(),
            row['AwayTeam'].item(),
            self.league_map.get(row['Div'].item(), 'Unknown League'),
            row['AwayTeam'].item(),
        )
        return q1, q2
    
