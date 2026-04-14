# src/models.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from src.config import RANDOM_STATE


def get_logistic_model():
    """
    This function creates and returns the Logistic Regression model.

    Input:
        None

    Output:
        LogisticRegression: A Logistic Regression model with the chosen settings.
    """
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )


def get_random_forest_model():
    """
    This function creates and returns the Random Forest model.

    Input:
        None

    Output:
        RandomForestClassifier: A Random Forest model with the chosen settings.
    """
    return RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )


def get_mlp_model():
    """
    This function creates and returns the MLP model.

    Input:
        None

    Output:
        MLPClassifier: A Multi-Layer Perceptron model with the chosen settings.
    """
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=100,
        random_state=RANDOM_STATE
    )