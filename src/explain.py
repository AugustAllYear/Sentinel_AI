"""SHAP explanations for model predictions."""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

def explain_prediction(model, preprocessor, X_sample, feature_names):
    """
    Generate SHAP explanations for a single prediction or a batch.
    """
    # Transform sample using preprocessor
    X_transformed = preprocessor.transform(X_sample)
    # Use a tree explainer for RandomForest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    # For binary classification, shap_values[1] is for class 1 (fraud)
    shap_values_fraud = shap_values[1]

    # Summary plot
    shap.summary_plot(shap_values_fraud, X_transformed, feature_names=feature_names, show=False)
    plt.savefig("shap_summary.png")
    plt.close()

    # Force plot for first sample (requires JS, better to use waterfall)
    shap.waterfall_plot(shap.Explanation(values=shap_values_fraud[0],
                                         base_values=explainer.expected_value[1],
                                         data=X_transformed[0],
                                         feature_names=feature_names))
    plt.savefig("shap_waterfall.png")
    plt.close()
    return shap_values_fraud

# Integration in predict.py or app.py