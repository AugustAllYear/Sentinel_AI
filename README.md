# Sential AI - Adaptive Fraud Scoring and Detection Engine

End‑to‑end machine learning pipeline for real‑time fraud detection, with CI/CD, automated retraining, and a monitoring dashboard.

## Features
- Synthetic data generator or load real transaction CSV.
- Feature engineering (velocity, amount z‑score, rolling fraud rate).
- Random Forest classifier with `class_weight='balanced'` for imbalanced data.
- Evaluation metrics: ROC‑AUC and Average Precision.
- MLflow tracking for experiments.
- Unit tests with pytest.
- GitHub Actions CI/CD (linting, testing, training).
- Scheduled weekly retraining (cron).
- Streamlit dashboard for predictions and monitoring.

### Recommended Features for Fraud Detection

| Feature | Description | Value |
|---------|-------------|-------|
| `transaction_velocity_1h` | Number of transactions in last hour | Catches rapid succession fraud |
| `amount_z_score` | (amount - avg_amount) / std_amount per user | Detects unusual transaction size |
| `days_since_last_fraud` | Days since last flagged fraud (if available) | Incorporates recency of risk |
| `card_age_days` | Days since card issuance | New cards may be riskier |
| `merchant_risk_score` | External merchant risk tier | Uses external data |
| `rolling_fraud_rate_7d` | Fraction of fraud in last 7 days per user | Temporal pattern |


## Setup

1. Clone the repository:
```bash
   git clone https://github.com/yourusername/sentinel_ai.git
   cd sentinel_ai
```
2. Create a virtual environment (Python 3.11 recommended):
```bash
    python -m venv venv
    source venv/bin/activate #mac
```
3. Install dependencies:
```bash
    pip install -r requirements.txt
```
4. (Optional) COnfigure MLflow tracking URI: set environment variable
   `MLFLOW_TRACKING_URI`

## Usage

### Configuration

All key parameters for data generation, feature engineering, model training, and monitoring are centralized in `config/config.yaml`. You can modify this file to adjust the pipeline's behavior without changing the code.

- **Data**: `synthetic_samples` and `random_state` control synthetic data generation.
- **Features**: Enable/disable feature groups like `velocity`, `z_score`, and `rolling_fraud`.
- **Model**: Choose between `random_forest` and `autoencoder`, and tune their hyperparameters.
- **Training**: Adjust the `test_size` for data splitting.
- **Monitoring**: Set the drift detection threshold (`psi_threshold`) and the features to monitor.
- **Paths**: Configure where models and data are saved.

### Train model with synthetic data
```bash
    python src/train.py --synthetic
```

### Train with real data
```bash
    python src/train.py --data_path data/raw/transactions.csv. #adjust as eneded
```

### Evaluate
```bash
    python src/evaluate.py
```

### Predict on new data
```
    from src.predict import load_model, predict
    import pandas as pd
    model, preprocessor = load_model()
    new_data = pd.read_csv("new_transactions.csv")
    probs = predict(new_data, model, preprocessor)
```
### Run dashboard
```
    streamlit run app.py
```
### Testing
```
    pytest tests/
```
### CI/CD

- GitHub Actions runs test and training on every push
- Weekly retraining via cron job (see `github/workflows/retrain.yml`).

## Logging & Monitoring

Warnings and errors are captured in multiple places:

- **Console**: Immediate feedback during local development.
- **GitHub Actions**: Logs available in the Actions tab for each CI/CD run.
- **MLflow**: Custom metrics and warnings can be logged as artifacts or metrics.
- **Log file**: Set up `logging.basicConfig(filename='sentinel.log')` to persist logs.

This multi‑layer approach ensures visibility across development, CI/CD, and production environments.

## Project Structure
```
sentinel_ai/
├── .github/
│   └── workflows/
│       └── ci.yml                # CI/CD workflow
│       └── retrain.yml           # Retraining if drift detected
│       └── scheduled_retrain.yml   # Set to every Sunday at midnight
├── -------config/
│   └── config.yaml               # (optional) configuration
├── data/
│   ├── raw/                      # raw input data (ignored by git)
│   └── processed/                # cleaned data
├── models/                       # saved model artifacts (MLflow)
├── src/
│   ├── __init__.py
│   ├── data.py                   # data loading & preprocessing
│   ├── features.py               # feature engineering
│   ├── train.py                  # model training
│   ├── evaluate.py               # evaluation & metrics
│   ├── predict.py                # prediction on new data
│   └── utils.py                  # helpers
├── tests/
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py                      # (optional) for packaging  
```

## Methodology for Projecting Business Outcomes

We estimate fraud prevention impact using:

- **Baseline fraud loss**: Historical loss without model.
- **Model performance**: Expected recall at a given precision threshold (e.g., at 80% precision, recall = 0.6).
- **Cost assumptions**:
  - Average fraud amount per transaction.
  - Cost of manual review per alert.

**Formula**:
Expected savings = (Total transaction value × Fraud rate × Recall) - (Alerts × Review cost)


**Example** (synthetic data):
- Monthly transactions: 1M, average amount $100 → $100M.
- Fraud rate: 1% → $1M fraud loss.
- Model recall: 0.6 → catches $600k fraud.
- Alert rate: 0.5% → 5,000 alerts × $5 review = $25k.
- Net savings = $600k - $25k = $575k per month.

We can adapt these numbers when real data is available.

**NOTE**:
In `train.py` `RandomForestClassifier(class_weight='balanced') 
- `balanced` automatically adjusts weights inversly proportional to class frequencies. For highly imblalnce fraud dataet (e.g. 1% fraud, 99% legitimate), the fraud class gets higher weight, making the modle penalize missclassificaiton fo fraud more heavily.
- Impact of fraud rate: The lower the fraud rate, the higher the weight assigned to fraud class.This helps the model not to simply predict "not fraud" for everything.
- Alternative: YOu can manually set `calss_weight={0.1, 1:10}` if you know the cost of missign a fraud is 10x that od a false alarm.
  
## Running the Streamlit Dashboard

1. Ensure you have trained a model and saved it in 'models/'.
2. Install streamlit if not alreadys: pip install streamlit`
3. Rum the dashboard:
```bash
    streamlit run app.py
```
4. Use the sidebar to upload a CSV file or generate synthetic data, then click "Predict Fraud Probability".

## Future Work

- **Real‑time API**: Deploy with FastAPI for sub‑100ms latency scoring.
- **Deep learning extensions**: Already implemented autoencoders; next: LSTM for sequence fraud.
- **Multi‑modal data**: Incorporate device fingerprint, IP geolocation.
- **Federated learning**: Train across institutions without sharing raw data.
- **Automated retraining triggers**: Use drift detection to trigger retraining outside weekly schedule.
- **Explainability dashboard**: Interactive SHAP visualizations in Streamlit.
- **Regulatory compliance**: Add audit trails and model cards.
-- Note:
Recommended CI/CD platforms: GitHub Actions, GitLab CI, Jenkins, Azure DevOps, CircleCI. For a data science project, GitHub Actions is popular because it integrates with code repositories and is free for public/private repos up to a limit.

If data files being used are below 14GB data can be processed using GitHub actions. Otherwise may need to upgrade to (S3) and trigger jobs

## Applications
E‑commerce: Flag suspicious transactions in real time.

Banking: Credit card fraud detection.

Insurance: Claim fraud detection.

Fintech: Payment gateway fraud prevention.

---

## Use case examples and outcomes

### Use Case 1: E‑commerce payment fraud
- **Input**: Transaction amount, user history, device fingerprint, time since last order.
- **Features**: Velocity (orders per hour), amount z‑score, rolling chargeback rate.
- **Outcome**: Model flags high‑risk transactions for 3D Secure challenge, reducing fraud losses by 25% while maintaining conversion.

### Use Case 2: Credit card fraud detection
- **Input**: Card transaction stream, merchant category, location.
- **Features**: Distance from previous transaction, card age, merchant risk score.
- **Outcome**: Real‑time scoring with <100ms latency; 15% increase in fraud detection compared to rule‑based system.

### Use Case 3: Insurance claim fraud
- **Input**: Claim amount, policyholder history, claim type.
- **Features**: Claim frequency in last year, anomaly in reported damage.
- **Outcome**: Prioritize high‑risk claims for manual review, reducing investigation costs by 40%.

These outcomes directly translate to **ROI**: lower fraud losses, reduced operational costs, and improved customer experience.

---

## Potential Outcomes
Reduce false positives by 30% through better feature engineering.

Increase fraud capture rate by 20% with ensemble models.

Automate review queue prioritization, saving analyst hours.

## License
MIT

## Contact
For questions or concerns please contact August Vollbrecht at augustvollbrecht@gmail.com

