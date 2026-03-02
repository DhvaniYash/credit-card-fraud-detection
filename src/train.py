# src/train.py

from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.models import get_random_forest_model
from src.evaluate import evaluate_model
from src.utils import save_model, save_metrics
from src.config import MODEL_SAVE_PATH, METRICS_SAVE_PATH


def main():
    print("Loading data...")
    df = load_data()

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Initializing model...")
    model = get_random_forest_model()

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    cm, report, roc_auc = evaluate_model(model, X_test, y_test)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    print("\nROC-AUC Score:")
    print(roc_auc)

    print("\nSaving model and metrics...")
    save_model(model, MODEL_SAVE_PATH)
    save_metrics(report, roc_auc, METRICS_SAVE_PATH)

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()