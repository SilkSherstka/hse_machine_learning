import pandas as pd


def add_rolling_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    out = df.copy()
    out["rolling_mean"] = out["TrafficCount"].rolling(window=window, min_periods=window).mean()
    out["rolling_std"] = out["TrafficCount"].rolling(window=window, min_periods=window).std()
    return out
