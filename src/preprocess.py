# src/preprocess.py

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import TEST_SIZE, RANDOM_STATE

def preprocess_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Scale features
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test