"""Train fraud detection model with MLflow."""

import mlflow
import mlflow.sklearn
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from src.data import generate_fraud_data, preprocess_data
from src.features import engineer_features
from src.utils import setup_logging, log_metrics_to_mlflow, plot_roc_curve, plot_confusion_matrix

logger = setup_logging()

def load_real_data(file_path: str):
    """Load real transaction data from CSV."""
    df = pd.read_csv(file_path)
    required_cols = ['amount', 'time_since_last_tx', 'avg_transaction_amount_30d',
                     'num_transactions_30d', 'location_risk_score', 'card_type',
                     'is_foreign', 'hour_of_day', 'day_of_week', 'is_fraud']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df

def train_model(data_path=None):
    """Train model with real data if path provided, else synthetic."""
    if data_path is not None:
        logger.info(f"Loading real data from {data_path}")
        df = load_real_data(data_path)
    else:
        logger.info("No data path provided. Generating synthetic data.")
        df = generate_fraud_data()

    # Apply feature engineering
    df = engineer_features(df, config={'velocity': True, 'z_score': True, 'rolling_fraud': False})

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'   # handle imbalanced fraud data
    )

    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)

        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("avg_precision", avg_precision)
        mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.log_model(preprocessor, "preprocessor")

        logger.info(f"ROC-AUC: {auc:.4f}, Average Precision: {avg_precision:.4f}")

        # Plot and log figures
        plot_roc_curve(y_test, y_pred_proba, save_path="roc_curve.png")
        plot_confusion_matrix(y_test, y_pred, save_path="confusion_matrix.png")
        mlflow.log_artifact("roc_curve.png")
        mlflow.log_artifact("confusion_matrix.png")

    return model, preprocessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to real data CSV (optional)")
    args = parser.parse_args()
    train_model(data_path=args.data_path)