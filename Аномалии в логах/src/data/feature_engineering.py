import pandas as pd


def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["Timestamp"].dt.hour
    out["day_of_week"] = out["Timestamp"].dt.dayofweek
    out["traffic_lag_1"] = out["TrafficCount"].shift(1)
    out["traffic_pct_change"] = out["TrafficCount"].pct_change()
    return out


def drop_na_for_model(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.dropna(subset=columns).copy()
