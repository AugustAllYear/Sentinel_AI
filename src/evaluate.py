"""Evaluation script for fraud model."""

import mlflow
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.data import generate_fraud_data, preprocess_data
import joblib

def evaluate(model, preprocessor, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

if __name__ == "__main__":
    # Example: load best model from MLflow
    # (you would typically load the model from a registered run)
    df = generate_fraud_data()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    # For demo, train a new model
    from src.train import train_model
    model, preprocessor = train_model()
    evaluate(model, preprocessor, X_test, y_test)
                                                    
        
    