import pytest
from src.train import train_model
from sklearn.metrics import roc_auc_score

def test_train_model():
    model, preprocessor = train_model()  # uses synthetic data by default
    assert hasattr(model, 'predict_proba')