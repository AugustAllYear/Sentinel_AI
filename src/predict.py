"""Predict fraud probability for new transactions."""

import pandas as pd
import joblib

def load_model(model_path="models/best_model.joblib", preprocessor_path="models/preprocessor.joblib"):
    """Load saved model and preprocessor."""
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

def predict(new_data: pd.DataFrame, model, preprocessor):
    """Return fraud probability for each transaction."""
    X = new_data[preprocessor.feature_names_in_]  # assuming preprocessor stores feature names
    X_transformed = preprocessor.transform(X)
    proba = model.predict_proba(X_transformed)[:, 1]
    return proba

if __name__ == "__main__":
    # Example: generate new data
    from src.data import generate_fraud_data
    df = generate_fraud_data()
    model, preprocessor = load_model()
    probs = predict(df.head(), model, preprocessor)
    print("Fraud probabilities:", probs)