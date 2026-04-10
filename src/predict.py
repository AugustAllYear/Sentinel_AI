"""Predict fraud probability for new transactions."""

import pandas as pd
import joblib

def load_model(model_path="models/model.joblib", preprocessor_path="models/preprocessor.joblib"):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

def predict(new_data: pd.DataFrame, model, preprocessor):
    """Return fraud probabilities."""
    if hasattr(preprocessor, 'feature_names_in_'):
        feature_names = preprocessor.feature_names_in_
    else:
        feature_names = ['amount', 'time_since_last_tx', 'avg_transaction_amount_30d',
                         'num_transactions_30d', 'location_risk_score', 'card_type',
                         'is_foreign', 'hour_of_day', 'day_of_week']
    X = new_data[feature_names]
    X_transformed = preprocessor.transform(X)
    proba = model.predict_proba(X_transformed)[:, 1]
    return proba

def flag_transactions(probs, threshold=0.5, high_risk_threshold=0.8):
    """
    Tiered flagging:
    - high risk (proba > high_risk_threshold) → 2 (auto-block)
    - medium risk (threshold <= proba <= high_risk_threshold) → 1 (manual review)
    - low risk (proba < threshold) → 0 (approve)
    """
    flags = []
    for p in probs:
        if p >= high_risk_threshold:
            flags.append(2)
        elif p >= threshold:
            flags.append(1)
        else:
            flags.append(0)
    return flags

if __name__ == "__main__":
    from src.data import generate_fraud_data
    df = generate_fraud_data()
    model, preprocessor = load_model()
    probs = predict(df.head(), model, preprocessor)
    print("Fraud probabilities:", probs)