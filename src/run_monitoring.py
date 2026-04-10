"""Run data drift monitoring and trigger retraining if needed."""

import pandas as pd
from src.monitoring import detect_drift, check_schema
from src.utils import setup_logging
import subprocess
import sys

logger = setup_logging()

def main(reference_path="data/processed/reference.csv", current_path="data/raw/latest_transactions.csv", psi_threshold=0.1):
    ref = pd.read_csv(reference_path)
    curr = pd.read_csv(current_path)
    
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