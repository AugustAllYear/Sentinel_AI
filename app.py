import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import shap
from src.data import generate_fraud_data
from src.predict import load_model, predict, flag_transactions

os.makedirs("images", exist_ok=True)

st.set_page_config(page_title="Sentinel_AI Dashboard", layout="wide")
st.title("Fraud Detection Dashboard")

@st.cache_resource
def get_model():
    return load_model("models/model.joblib", "models/preprocessor.joblib")

model, preprocessor = get_model()

option = st.sidebar.selectbox("Data Source", ["Synthetic Demo", "Upload CSV"])
if option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload transaction data (CSV)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a file")
        st.stop()
else:
    df = generate_fraud_data(100)

if st.button("Predict Fraud Probability"):
    probs = predict(df, model, preprocessor)
    df['fraud_prob'] = probs

    # Tiered flagging
    flags = flag_transactions(probs, threshold=0.5, high_risk_threshold=0.8)
    df['risk_level'] = flags

    st.subheader("Predictions")
    st.write(df.head(10))

    # Histogram
    fig, ax = plt.subplots()
    df['fraud_prob'].hist(bins=20, ax=ax)
    ax.set_xlabel("Fraud Probability")
    ax.set_title("Distribution of Fraud Probabilities")
    st.pyplot(fig)

    # SHAP explanations
    if model.__class__.__name__ == 'RandomForestClassifier':
        st.subheader("SHAP Explanations")
        sample = df.drop(columns=['fraud_prob', 'risk_level']).head(5)
        if hasattr(preprocessor, 'feature_names_in_'):
            feature_names = preprocessor.feature_names_in_
        else:
            feature_names = ['amount', 'time_since_last_tx', 'avg_transaction_amount_30d',
                             'num_transactions_30d', 'location_risk_score', 'card_type',
                             'is_foreign', 'hour_of_day', 'day_of_week']
        X_sample = sample[feature_names]
        X_transformed = preprocessor.transform(X_sample)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        shap.initjs()
        st.write("#### Force Plot for First Transaction")
        shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_transformed[0],
                        feature_names=feature_names, matplotlib=True, show=False)
        plt.savefig(os.path.join("images", "force_plot.png"), bbox_inches='tight')
        st.image("images/force_plot.png")

        st.write("#### Summary Plot")
        shap.summary_plot(shap_values[1], X_transformed, feature_names=feature_names, show=False)
        plt.savefig(os.path.join("images", "summary_plot.png"), bbox_inches='tight')
        st.image("images/summary_plot.png")