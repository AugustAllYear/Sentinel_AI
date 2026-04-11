""" Utilites script to ensure reusability, standardizing MLflow tracking across scripts and helper functions for evaluations."""

import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import mlflow
import joblib
import yaml

def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def ensure_image_dir():
    os.makedirs("images", exist_ok=True)

def plot_roc_curve(y_true, y_proba, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    if save_path:
        full_path = os.path.join("images", os.path.basename(save_path))
        plt.savefig(full_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    if save_path:
        full_path = os.path.join("images", os.path.basename(save_path))
        plt.savefig(full_path)
        plt.close()
    else:
        plt.show()

def log_metrics_to_mlflow(metrics_dict):
    for key, value in metrics_dict.items():
        mlflow.log_metric(key, value)

def save_model(model, preprocessor, path="models/"):
    joblib.dump(model, f"{path}/model.joblib")
    joblib.dump(preprocessor, f"{path}/preprocessor.joblib")

def load_model(model_path="models/model.joblib", preprocessor_path="models/preprocessor.joblib"):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config