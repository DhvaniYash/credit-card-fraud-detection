# src/models.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from src.config import RANDOM_STATE


def get_logistic_model():
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )


def get_random_forest_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )


def get_mlp_model():
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=100,
        random_state=RANDOM_STATE
    )