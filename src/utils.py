# src/utils.py

import joblib
from datetime import datetime


def save_model(model, path):
    """
    This function saves the trained model to a file.

    Input:
        model: The trained machine learning model
        path: File path where the model will be saved

    Output:
        None
    """
    joblib.dump(model, path)


def save_metrics(model_name, cm, report_text, roc_auc, path):
    """
    This function saves the evaluation results to a text file.

    Input:
        model_name: Name of the model
        cm: Confusion matrix
        report_text: Classification report as text
        roc_auc: ROC-AUC score
        path: File path where the metrics will be saved

    Output:
        None
    """
    with open(path, "a") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report_text)
        f.write("\nROC-AUC Score:\n")
        f.write(str(roc_auc))
        f.write("\n")