"""Train fraud detection model with MLflow."""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from src.data import generate_fraud_data, preprocess_data
import argparse

def load_real_data(file_path: str):
    """Load real transaction data from CSV."""
    df = pd.read_csv(file_path)
    # Ensure required columns exist; you may need to adapt column names
    required_cols = ['amount', 'time_since_last_tx', 'avg_transaction_amount_30d',
                     'num_transactions_30d', 'location_risk_score', 'card_type',
                     'is_foreign', 'hour_of_day', 'day_of_week', 'is_fraud']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df

def train_model(use_synthetic=True, data_path=None):
    """Train model with either synthetic or real data."""
    if data_path is None:
            raise ValueError("Must provide data_path when use_synthetic=False")
        df = load_real_data(data_path)
    else:
        if use_synthetic:
        df = generate_fraud_data()

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.log_model(preprocessor, "preprocessor")
        print(f"ROC-AUC: {auc:.4f}")

    return model, preprocessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--data_path", type=str, help="Path to real data CSV")
    args = parser.parse_args()

    if args.synthetic:
        train_model(use_synthetic=True)
    else:
        train_model(use_synthetic=False, data_path=args.data_path)