# src/data_loader.py

import pandas as pd
from src.config import DATA_PATH

def load_data():
    """
    This function loads the credit card fraud dataset from the file path.

    Input:
        None

    Output:
        df (DataFrame): The dataset loaded as a pandas DataFrame.
    """
    df = pd.read_csv(DATA_PATH)
    
    return df