import pytest
from src.train import train_model
from sklearn.metrics import roc_auc_score

def test_train_model():
    model, preprocessor = train_model(use_synthetic=True)
    assert hasattr(model, 'predict_proba')