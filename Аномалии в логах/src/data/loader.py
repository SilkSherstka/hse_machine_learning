from pathlib import Path

import pandas as pd


def load_traffic_data(data_path: str) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df
