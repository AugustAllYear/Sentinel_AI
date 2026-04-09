import pytest
from src.data import generate_fraud_data, preprocess_data

def test_generate_fraud_data():
    df = generate_fraud_data(100)
    assert df.shape[0] == 100
    assert 'is_fraud' in df.colums

def test_process_data():
    df = generate_fraud_data()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    assert X_train.shape[1] == X_test.shape[1]
    assert len(y_train) + len(y_test) == len(df)