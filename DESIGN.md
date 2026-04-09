# Sentinel_AI Design Document

## System Overview
Batch fraud detection pipeline using Random Forest with optional deep learning extension.

## Data Flow
1. Input: CSV of transactions (or synthetic generator).
2. Feature engineering (velocity, z-score, rolling fraud rate).
3. Preprocessing (scaling, one-hot encoding).
4. Model training (RandomForest with class_weight='balanced').
5. Evaluation (ROC-AUC, Average Precision).
6. Prediction API (planned) and dashboard.

## Components
- `src/data.py`: Data loading & preprocessing.
- `src/features.py`: Feature engineering.
- `src/train.py`: Training with MLflow.
- `src/evaluate.py`: Evaluation and metrics.
- `src/predict.py`: Scoring new data.
- `src/utils.py`: Logging, plotting, model I/O.
- `app.py`: Streamlit dashboard.

## Assumptions
- Imbalanced fraud labels (fraud rate < 5%).
- Weekly retraining sufficient to mitigate drift.

## Monitoring
- Data drift: PSI (Population Stability Index) on feature distributions.
- Model performance drift: Weekly ROC-AUC/AP comparison.
- Warnings/Errors: Logged to console, MLflow, and GitHub Actions.

## Dependencies
See `requirements.txt`.

## Future Enhancements
- Deep learning autoencoders for anomaly detection.
- SHAP explanations.
- Real-time API.