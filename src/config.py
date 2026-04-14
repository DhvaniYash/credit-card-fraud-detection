# src/config.py

# Path to the dataset file
DATA_PATH = "data/creditcard.csv"

# Test set size and random seed used in the project
TEST_SIZE = 0.2
RANDOM_STATE = 42

# File paths used to save the trained model and evaluation metrics
MODEL_SAVE_PATH = "results/models/mlp_model.pkl"
METRICS_SAVE_PATH = "results/metrics.txt"

# Example:
# If you want to save a different model, you can change the file name.
# Example: MODEL_SAVE_PATH = "results/models/random_forest_model.pkl"

# If you want to save metrics in a different file, you can change that too.
# Example: METRICS_SAVE_PATH = "results/random_forest_metrics.txt"