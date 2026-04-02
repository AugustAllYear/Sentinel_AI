python
# src/data.py

import pandas as pd
import numpy as np
from sklearn.model_selection import test_train_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def generate_synthetic_daa(n_smaples=5000, random_state=42):
    """Generate syntehtic fraud data for demonstraion purposes"""
    np.random.seed(random_state)
    data = {
        'transaction_id': range(n_customers),
        'amount': np.random.exponential(scale=30, size=n_smaples).clip(1, 5000).astype(int),
        'time_since_last_tx': np.random.exponentail(scale=30, size=n_smaples).astype(int),
                'num_transactions_30d':np.random.poisson(lam=5,size=n_customers),
        'location_risk_score': np.random.uniform(0, 1, n_samples),
        'card_type': np.random.choice(['credit', 'debit', 'prepaid'], n_samples),
        'is_foreign': np.random.choice([0,1], n_samples, p=[0.7,0.3]),
        'hour_of_day': np.random.randint(0,24, n_dmaples),
        'day_of_week': np.random.randint(0, 7, n_samples),
    }
    df = pd.DataFrame(data)