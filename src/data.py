"""Data loading and preprocessing for fraud detection."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from datetime import datetime, timedelta


def generate_fraud_data(n_samples=config['data']['synthetic_smaples'], random_state=config['data']['random_state']):
    np.random.seed(random_state)
    
    
    # Generate timestamps within the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = [start_date + timedelta(seconds=np.random.randint(0, 30*24*3600)) 
                  for _ in range(n_samples)]
    
    data = {
        'transaction_id': range(n_samples),
        'user_id': np.random.randint(1, 1001, n_samples),  # 1000 unique users
        'timestamp': timestamps,
        'amount': np.random.exponential(scale=500, size=n_samples).clip(1, 5000).astype(int),
        'time_since_last_tx': np.random.exponential(scale=30, size=n_samples).astype(int),
        'avg_transaction_amount_30d': np.random.normal(300, 100, n_samples).clip(10, 2000).astype(int),
        'num_transactions_30d': np.random.poisson(lam=5, size=n_samples),
        'location_risk_score': np.random.uniform(0, 1, n_samples),
        'card_type': np.random.choice(['credit', 'debit', 'prepaid'], n_samples),
        'is_foreign': np.random.choice([0,1], n_samples, p=[0.7,0.3]),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
    }
    df = pd.DataFrame(data)

    def generate_fraud(row):
        prob = 0.1
        if row['amount'] > 2000:
            prob += 0.05
        if row['location_risk_score'] > 0.8:
            prob += 0.1
        if row['is_foreign'] == 1:
            prob += 0.03
        if row['num_transactions_30d'] > 10:
            prob -= 0.02
        prob = max(0.0, min(prob, 0.8))
        return np.random.binomial(1, prob)

    df['is_fraud'] = df.apply(generate_fraud, axis=1)
    return df

def preprocess_data(df, fit_preprocessor=True):
    features = ['amount', 'time_since_last_tx', 'avg_transaction_amount_30d',
                'num_transactions_30d', 'location_risk_score', 'card_type',
                'is_foreign', 'hour_of_day', 'day_of_week']
    X = df[features]
    y = df['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = ['amount', 'time_since_last_tx', 'avg_transaction_amount_30d',
                        'num_transactions_30d', 'location_risk_score', 'hour_of_day', 'day_of_week']
    categorical_features = ['card_type', 'is_foreign']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    if fit_preprocessor:
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
    else:
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor