# Credit Card Fraud Detection

This machine learning project focuses on detecting fraudulent credit card transactions under severe class imbalance. The goal is to build and compare models that can identify fraud more effectively than a simple baseline approach.

## Project Objective

The main objective of this project is to:
- detect fraudulent transactions from credit card data
- handle the class imbalance problem
- compare multiple machine learning models
- evaluate model performance using metrics that are important for fraud detection
- provide a simple Gradio app for fraud prediction on uploaded CSV files

## Dataset

The dataset used in this project is the credit card fraud detection dataset.

- File name: `creditcard.csv`
- Location: `data/creditcard.csv`

This dataset contains transaction features and a target column that shows whether a transaction is fraudulent or not.

## Project Structure

```text
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── exploration.ipynb
│
├── results/
│   ├── models/
│   │   ├── logistic_model.pkl
│   │   ├── mlp_model.pkl
│   │   ├── random_forest_model.pkl
│   │   └── scaler.pkl
│   ├── plots/
│   └── metrics.txt
│
├── src/
│   ├── old_baseline_model.py
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── models.py
│   ├── preprocess.py
│   ├── train.py
│   └── utils.py
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Models Used

The project includes the following models:
- Logistic Regression
- Random Forest
- Multi-Layer Perceptron (MLP)

## Evaluation Metrics

The models are evaluated using:
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

These metrics are important because fraud detection is an imbalanced classification problem, so accuracy alone is not enough.

## Current Progress

So far, this project includes:
- project folder structure
- data loading and preprocessing
- SMOTE for imbalance handling
- Logistic Regression model
- Random Forest model
- MLP model
- evaluation metrics and comparison summary
- saved model outputs
- result plots
- Gradio app for CSV-based fraud detection

## Model Comparison Summary

- RandomForest_SMOTE: precision 0.87, recall 0.83, f1-score 0.85, ROC-AUC 0.9685
- LogisticRegression_SMOTE: precision 0.06, recall 0.92, f1-score 0.11, ROC-AUC 0.9708
- MLP_SMOTE: precision 0.72, recall 0.81, f1-score 0.76, ROC-AUC 0.9635

Based on these results, RandomForest_SMOTE gave the best overall balance for fraud detection.

## Gradio App

This project includes a Gradio app that allows users to upload a CSV file and get fraud predictions for each transaction using the trained Random Forest model.

### App Features

- upload a CSV file
- predict whether each transaction is **Fraud** or **Not Fraud**
- show a clean **Prediction Summary**
- show a simple **What This Means** explanation
- display a clean results table with:
  - `Transaction_Number`
  - `Prediction_Label`
- download the prediction results as a CSV file

### Files Used by the App

The app uses:

- `app.py`
- `results/models/random_forest_model.pkl`
- `results/models/scaler.pkl`

### How to Run the Gradio App

```bash
python app.py
```
Then open the local link shown in the terminal, usually:

```text
http://127.0.0.1:7860
```

### App Output
After uploading a CSV file, the app shows:

- total number of transactions
- how many were predicted as fraud
- how many were predicted as not fraud
- row-by-row prediction labels
- a downloadable CSV file of the prediction results

## How to Run the Training Pipeline
1. Open the project in VS Code
2. Open a terminal in the project folder
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the training script:
```bash
python -m src.train
```

## Results

The generated outputs are saved in:
- results/models/ for saved model files
- results/metrics.txt for evaluation results
- results/plots/ for visualizations

## Future Improvements

- tune hyperparameters
- improve app styling
- add fraud count summary in the app
- add downloadable prediction results
- improve final report and discussion
- explore additional imbalance handling techniques


## Author
Dhvani Yash