import os
import numpy as np
import polars as pl

QUESTION = f"On {{}}, the home team {{}} plays against the away team {{}} in the {{}}. What is the probability that {{}} the match?"

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


class ProbabilityDataset:
    def __init__(self, data_dir: str):
        self.data = get_data(data_dir)

    def get_prompts(self):
        idx = np.random.randint(0, self.data.height, 1).item()
        row = self.data[idx]

        q1 = QUESTION.format(
            row['Date'].item(),
            row['HomeTeam'].item(),
            row['AwayTeam'].item(),
            LEAGUE_MAP.get(row['Div'].item(), 'domestic league'),
            row['HomeTeam'].item() + ' wins',
        )

        q2 = QUESTION.format(
            row['Date'].item(),    
            row['HomeTeam'].item(),
            row['AwayTeam'].item(),
            LEAGUE_MAP.get(row['Div'].item(), 'domestic league'),
            row['HomeTeam'].item() + ' does not win',
        )

        return [q1, q2]


def get_data(data_dir: str):
    data = pl.DataFrame()
    for file_name in os.listdir(data_dir):
        if file_name.endswith(f"epl_combined.parquet"):
            lazy_df = pl.scan_parquet(os.path.join(data_dir, file_name))
            df = lazy_df.select(
                COLUMNS
            ).collect()
            data = pl.concat([data, df])
    return data


    
