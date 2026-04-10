"""Train fraud detection model (RandomForest or Autoencoder) with MLflow."""

import mlflow
import mlflow.sklearn
import pandas as pd
import argparse
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from src.data import generate_fraud_data, preprocess_data
from src.features import engineer_features
from src.utils import setup_logging, plot_roc_curve, plot_confusion_matrix
from src.deep_learning import train_autoencoder, predict_anomaly

logger = setup_logging()

def load_real_data(file_path: str):
    df = pd.read_csv(file_path)
    required_cols = ['amount', 'time_since_last_tx', 'avg_transaction_amount_30d',
                     'num_transactions_30d', 'location_risk_score', 'card_type',
                     'is_foreign', 'hour_of_day', 'day_of_week', 'is_fraud']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df

def train_model(data_path=None, use_autoencoder=False):
    if data_path is not None:
        logger.info(f"Loading real data from {data_path}")
        df = load_real_data(data_path)
    else:
        logger.info("No data path provided. Generating synthetic data.")
        df = generate_fraud_data()

    df = engineer_features(df, config={'velocity': False, 'z_score': True, 'rolling_fraud': False})

    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    with mlflow.start_run():
        if use_autoencoder:
            logger.info("Training autoencoder for anomaly detection")
            model, scaler = train_autoencoder(X_train, epochs=20)
            # Save scaler
            joblib.dump(scaler, "scaler.joblib")
            mlflow.log_artifact("scaler.joblib")
            
            # For autoencoder, we use reconstruction error as anomaly score
            _, mse_train = predict_anomaly(model, scaler, X_train)
            _, mse_test = predict_anomaly(model, scaler, X_test)
            auc = roc_auc_score(y_test, mse_test)
            ap = average_precision_score(y_test, mse_test)
            mlflow.log_metric("roc_auc", auc)
            mlflow.log_metric("avg_precision", ap)
            mlflow.log_artifact("autoencoder_model.pth")
            logger.info(f"Autoencoder ROC-AUC: {auc:.4f}, AP: {ap:.4f}")
        else:
            logger.info("Training RandomForest classifier")
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            mlflow.log_metric("roc_auc", auc)
            mlflow.log_metric("avg_precision", ap)
            mlflow.sklearn.log_model(model, "model")
            mlflow.sklearn.log_model(preprocessor, "preprocessor")
            logger.info(f"RandomForest ROC-AUC: {auc:.4f}, AP: {ap:.4f}")

            # Save reference data for drift monitoring
            os.makedirs("data/processed", exist_ok=True)
            df.to_csv("data/processed/reference.csv", index=False)
            logger.info("Saved reference dataset to data/processed/reference.csv")

            plot_roc_curve(y_test, y_proba, save_path="roc_curve.png")
            plot_confusion_matrix(y_test, model.predict(X_test), save_path="confusion_matrix.png")
            mlflow.log_artifact("roc_curve.png")
            mlflow.log_artifact("confusion_matrix.png")

    return model, preprocessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to real data CSV")
    parser.add_argument("--autoencoder", action="store_true", help="Use autoencoder instead of RandomForest")
    args = parser.parse_args()
    train_model(data_path=args.data_path, use_autoencoder=args.autoencoder)