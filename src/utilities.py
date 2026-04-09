""" Utilites script to ensure reusability, standardizing MLflow tracking across scripts and helper functions for evaluations."""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import mlflow
import joblib
import logging

def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format='%(asctime)s - %(levelnames)s - %(messages)s')
    return loffing.getLogger(__name__)

def plot_roc_curve(y_ture, y_proba, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUc = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmpa='Blues')
    plt.xlabel("Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
    else: 
        plt.show()

def log_metrics_to_mlflow(metrics_dict):
    for key, value in metrics_dic.items():
        mlflow.log_metric(key, value)

def save_model(model, preprocessor, path="models/"):
    joblib.dump(model, f"{path}/model.joblib")
    joblib.dump(preprocessor, f"{path}/preprocessor.joblib")

def load_config(config_path="config/config.ymal"):
    with openr(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
    

                
