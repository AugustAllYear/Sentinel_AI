import streamlit as st
import pandas as pd
import joblib
from src.data import generate_fraud_data
from src.predict import predict
import matplotlib.pyplot as plt

st.title("Sentinel_AI: Fraud Detection Dashboard")

# Load model
model, preprocessor = joblib.load("models/model.joblib"), joblib.load("models/preprocessor.joblib")

# Sidebar for data source
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
    st.write(df.head())
    st.bar_chart(df['fraud_prob'].value_counts().sort_index())
    # Plot histogram
    fig, ax = plt.subplots()
    df['fraud_prob'].hist(bins=20, ax=ax)
    st.pyplot(fig)
