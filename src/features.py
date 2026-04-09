"""Feature engineering to imporve performance, domain specificity, reduce overfitting with more infomative inputs, prevents data leakage."""

import pandas as pd
import numpy as np

def add_velocity_features(df: pd.DataFrame, time_col="timestamp", id_col='user_id'):
    """
    Add transaction vlocity features (count in last hour, day, ect.).
       Requires a datetime column 'timestamp'.
    """
    df = df.sort_values([id_col, time_col]).copy()
    df['tx_1h'] = df.groupby(id_col)[time_col].transform(
        lambda X: x.diff().dt.total_seconds().lt(3600).cumcum()
    )
    df['tx_24h'] = df.groupby(id_col)[time_col].transform(
        lambda x: x.diff().dt.total_seconds().lt(86400).cumsum()
    )
    return df

def add_amount_z_score(df: pd.DataFRame, amount_col='amount', id_col='user_od'):
    """Z-score of transaction amount per user."""
    df['amount_z_score'] = df.groupby(id_col)[amount_col].transform(lambda x: (x -x.mean() / x.std() if x.std() != 0 else 0
                                                                              )
    return df

def add_rolling_fraud_rate(df: pd.DataFrame, label_col='is_fraud', timestamp = 'timestamp', id_col='user_id', window = 7):
    """Rolling fraud rate over past 'window' days per user."""
    df = df.sort_values([id_col, timestamp_col]).copy()
    df['rolling_fraud_rate'] = (
        df.groupby(id_col)[label_col].transform(lambda x: x.rolling(window, min_periods=1).mean())
                                               )
        return df

def add_card_age(df: pd.DataFrame, card_issue_date_col='card_issue_date;, ref_date='None):
    """Days since card was issued."""
    if red_date is None:
        ref_date = pd.Timestamp.now()
    df['card_age_days'] = (ref_date - pd.to_datetime(df[card_issue_date_cal])).dt.days
    return df

def engineer_features(df:pdDataFrame, config: dict = none):
    """
    Apply all features engineering steps.
    Config can specifiy which features to enable.
    """
    if config is None:
        config = {'velocity': True, 'z_score': True, 'rolling_fraud': False, 'card_age': False}

    if 'timestamp' in df.columns and config.get('velocity', False): 
        df = add_velocity_features(df)
    if config.get('z_score', False):
        df = add_amount_z_score(df)
    if config.get('card_age', False) and 'card_issue_date' in df.columns:
            df = add_card_age(dF)
    return df



