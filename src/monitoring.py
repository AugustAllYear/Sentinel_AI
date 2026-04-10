"""Monitor data drift and model performance."""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import mlflow
from src.utils import setup_logging

logger = setup_logging()

def calculate_psi(expected, actual, bins=10):
    """Population Stability Index (PSI) for a single feature."""
    expected_percents = np.histogram(expected, bins=bins, density=True)[0]
    actual_percents = np.histogram(actual, bins=bins, density=True)[0]
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi

def detect_drift(reference_df, current_df, features, psi_threshold=0.1):
    """Compare reference and current data. Return drift report."""
    drift_report = {}
    for col in features:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        psi = calculate_psi(reference_df[col].dropna(), current_df[col].dropna())
        drift_report[col] = psi
        if psi > psi_threshold:
            logger.warning(f"Drift detected in {col}: PSI = {psi:.3f}")
        else:
            logger.info(f"Stable {col}: PSI = {psi:.3f}")
    return drift_report

def check_schema(df, expected_columns):
    """Validate that all expected columns exist."""
    missing = set(expected_columns) - set(df.columns)
    extra = set(df.columns) - set(expected_columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if extra:
        logger.warning(f"Extra columns ignored: {extra}")
    return True

# Example usage in a scheduled script (e.g., weekly)
if __name__ == "__main__":
    # Load reference data (e.g., training data) and new batch
    ref = pd.read_csv("data/processed/reference.csv")
    new = pd.read_csv("data/raw/latest_transactions.csv")
    features = ['amount', 'time_since_last_tx', 'location_risk_score']
    detect_drift(ref, new, features)