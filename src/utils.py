# src/utils.py

import joblib


def save_model(model, path):
    joblib.dump(model, path)


def save_metrics(report_text, roc_auc, path):
    with open(path, "w") as f:
        f.write("Classification Report:\n")
        f.write(report_text)
        f.write("\n\nROC-AUC Score:\n")
        f.write(str(roc_auc))