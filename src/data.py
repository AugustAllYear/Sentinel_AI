python
# src/data.py

import pandas as pd
import numpy as np

def generate_synthetic_daa(n_customers=5000, random_state=42):
    np.random.seed(random_state)
    data = {
        'customer_id': range(1, n_customers+1),
        'age': np.random.randint(18, 76, n_customers).astype(int),
        'income':np.random.normal(50000, 15000, n_customers).astype(int),
        'tenure': np.random.randint(1, 1120, n_customers),
        'last_purchase_days': np.random.randint(1, 365, n_customers),
        'avg_order_value': np.random.normal(100, 30, n_customers).clip(20,300).astype(int),
        'campaign_channel 