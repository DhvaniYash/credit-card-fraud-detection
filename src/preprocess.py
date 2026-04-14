# src/preprocess.py

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import TEST_SIZE, RANDOM_STATE


def preprocess_data(df):
    """
    This function prepares the dataset for model training.

    Input:
        df (DataFrame): The full credit card fraud dataset.

    Output:
        X_train: Scaled and SMOTE-resampled training features
        X_test: Scaled testing features
        y_train: Resampled training labels
        y_test: Testing labels
        scaler: Fitted StandardScaler object
    """

    # Separate input features and target column
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Scale the feature values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Apply SMOTE only on the training data
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, scaler