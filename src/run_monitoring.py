"""Run data drift monitoring and trigger retraining if needed.
This script should be called weekly (e.g., via GitHub Actions cron job).
"""

import pandas as pd
import os
import sys
import subprocess
from src.monitoring import detect_drift, check_schema
from src.utils import setup_logging

logger = setup_logging()

def main(reference_path="data/processed/reference.csv",
         current_path="data/raw/latest_transactions.csv",
         psi_threshold=0.1):
    """
    Compare current data against reference and retrain if drift exceeds threshold.
    """
    if not os.path.exists(reference_path):
        logger.error(f"Reference file not found: {reference_path}")
        logger.info("No reference data available. Running initial training.")
        subprocess.run(["python", "src/train.py", "--data_path", current_path])
        return

    if not os.path.exists(current_path):
        logger.warning(f"No new data found at {current_path}. Skipping drift detection.")
        return

    ref = pd.read_csv(reference_path)
    curr = pd.read_csv(current_path)

    # Define features to monitor (subset of all features)
    features = ['amount', 'time_since_last_tx', 'location_risk_score', 'num_transactions_30d']

    # Check schema
    try:
        check_schema(curr, ref.columns.tolist())
    except ValueError as e:
        logger.error(f"Schema mismatch: {e}")
        sys.exit(1)

    # Detect drift
    drift_report = detect_drift(ref, curr, features, psi_threshold=psi_threshold)

    # If any drift exceeds threshold, trigger retraining
    if any(psi > psi_threshold for psi in drift_report.values()):
        logger.warning("Drift detected. Triggering model retraining...")
        subprocess.run(["python", "src/train.py", "--data_path", current_path])
    else:
        logger.info("No significant drift detected. Model is still valid.")

if __name__ == "__main__":
    main()