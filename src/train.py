"""Train fraud detection model with MLflow."""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from src.data import generate_fraud_data, preprocess_data

def train_model():
    # Generate data (replace with actual data loading)
    df = generate_fraud_data()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    # Train with MLflow
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
    train_model()