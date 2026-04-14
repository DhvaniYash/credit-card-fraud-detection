# src/evaluate.py

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate_model(model, X_test, y_test):
    """
    This function evaluates the trained model on the test data.

    Input:
        model: The trained machine learning model
        X_test: Testing feature data
        y_test: True labels for the test data

    Output:
        cm: Confusion matrix
        report: Classification report
        roc_auc: ROC-AUC score
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    return cm, report, roc_auc