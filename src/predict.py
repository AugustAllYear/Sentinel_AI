"""Predict fraud probability for new transactions."""

import pandas as pd
import joblib

def load_model(model_path="models/model.joblib", preprocessor_path="models/preprocessor.joblib"):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

def predict(new_data: pd.DataFrame, model, preprocessor):
    # Get feature names from preprocessor if available
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

if __name__ == "__main__":
    from src.data import generate_fraud_data
    df = generate_fraud_data()
    model, preprocessor = load_model()
    probs = predict(df.head(), model, preprocessor)
    print("Fraud probabilities:", probs

def predict_with_shap(new_data, model, preprocessor, feature_names):
    probs = predict(new_data, model, preprocessor)
    shap_values = None
    if model.__class__.__name__ == 'RandomForestClassifier':
        import shap
        explainer = shap.TreeExplainer(model)
        X_transformed = preprocessor.transform(new_data)
        shap_values = explainer.shap_values(X_transformed)[1]
    return probs, shap_values
    