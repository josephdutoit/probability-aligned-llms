import os
import polars as pl

QUESTION = f"On {{}}, the home team {{}} plays against the away team {{}} in the {{}}. What is the probability that {{}} wins the match?"

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

def load_qs(data_dir: str):
    data = pl.DataFrame()
    for file_name in os.listdir(data_dir):
        if file_name.endswith(f".parquet"):
            lazy_df = pl.scan_parquet(os.path.join(data_dir, file_name))
            df = lazy_df.select(
                COLUMNS
            ).collect()
            data = pl.concat([data, df])
    qs = []
    for row in data.iter_rows():
        q1 = QUESTION.format(
            row[1],
            row[2],
            row[3],
            LEAGUE_MAP.get(row[0], 'Unknown League'),
            row[2],
        )
        q2 = QUESTION.format(
            row[1],
            row[2],
            row[3],
            LEAGUE_MAP.get(row[0], 'Unknown League'),
            row[3],
        )
        qs.append((q1, q2))
    return qs


    
