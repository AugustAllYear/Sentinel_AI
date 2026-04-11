"""Evaluate fraud detection model."""

import mlflow
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve
from src.data import generate_fraud_data, preprocess_data
from src.features import engineer_features
from src.utils import setup_logging, plot_roc_curve, plot_confusion_matrix, load_model

logger = setup_logging()

def evaluate(model, preprocessor, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    logger.info("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")

    auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    logger.info(f"ROC-AUC: {auc:.4f}, Average Precision: {avg_precision:.4f}")

    plot_roc_curve(y_test, y_proba, save_path="eval_roc.png")
    plot_confusion_matrix(y_test, y_pred, save_path="eval_cm.png")

def find_optimal_threshold(y_true, y_proba, target_precision=0.8):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    for i, p in enumerate(precisions):
        if p >= target_precision:
            return thresholds[i]
    return 0.5

if __name__ == "__main__":
    # Generate synthetic data for evaluation (or load real)
    df = generate_fraud_data()
    # Use same feature config as training to avoid velocity issues
    df = engineer_features(df, config={'velocity': False, 'z_score': True, 'rolling_fraud': False})
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    # Check if model exists
    if not os.path.exists("models/model.joblib"):
        logger.error("Model not found. Please run src/train.py first.")
        sys.exit(1)

    model, preprocessor = load_model("models/model.joblib", "models/preprocessor.joblib")
    evaluate(model, preprocessor, X_test, y_test)

    # Example threshold calculation
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = find_optimal_threshold(y_test, y_proba)
    print(f"Recommended threshold: {threshold:.3f}")