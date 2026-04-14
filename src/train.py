# src/train.py

from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.models import get_mlp_model
from src.evaluate import evaluate_model
from src.utils import save_model, save_metrics
from src.config import MODEL_SAVE_PATH, METRICS_SAVE_PATH


def main():
    """
    This function runs the full machine learning pipeline for the project.

    Input:
        None

    Output:
        None directly, but it prints the results,
        saves the trained model, and saves the evaluation metrics.
    """

    # Load the dataset
    print("Loading data...")
    df = load_data()

    # Preprocess the dataset and split it into train and test sets
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Create the MLP model
    print("Initializing model...")
    model = get_mlp_model()

    # Train the model using the training data
    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate the trained model on the test data
    print("Evaluating model...")
    cm, report, roc_auc = evaluate_model(model, X_test, y_test)

    # Print the evaluation results
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    print("\nROC-AUC Score:")
    print(roc_auc)

    # Save the trained model and the metrics
    print("\nSaving model and metrics...")
    save_model(model, MODEL_SAVE_PATH)
    model_name = "MLP_SMOTE"

    save_metrics(
        model_name,
        cm,
        report,
        roc_auc,
        METRICS_SAVE_PATH
    )

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()