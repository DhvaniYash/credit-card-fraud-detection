# src/utils.py

import joblib
from datetime import datetime


def save_model(model, path):
    joblib.dump(model, path)


def save_metrics(model_name, cm, report_text, roc_auc, path):
    with open(path, "a") as f:
        f.write("\n" + "="*60 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report_text)
        f.write("\nROC-AUC Score:\n")
        f.write(str(roc_auc))
        f.write("\n")