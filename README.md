# Credit Card Fraud Detection

This machine learning project focuses on detecting fraudulent credit card transactions under severe class imbalance. The goal is to build and compare models that can identify fraud more effectively than a simple baseline approach.

## Project Objective

The main objective of this project is to:
- detect fraudulent transactions from credit card data
- handle the class imbalance problem
- compare multiple machine learning models
- evaluate model performance using metrics that are important for fraud detection

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
├── README.md
├── requirements.txt
└── .gitignore
```
## Models Used

The project includes the following models:

Logistic Regression

Random Forest

Multi-Layer Perceptron (MLP)

## Evaluation Metrics

The models are evaluated using:

Precision

Recall

F1-score

ROC-AUC

Confusion Matrix

These metrics are important because fraud detection is an imbalanced classification problem, so accuracy alone is not enough.

## Current Progress

So far, this project includes:

project folder structure

data loading and preprocessing

model training pipeline

evaluation functions

saved model outputs

metrics output file

## How to Run

Open the project in VS Code

Open a terminal in the project folder

Install dependencies using requirements.txt

Run the training script from the src folder

## Results

The generated outputs are saved in:

results/models/ for saved model files

results/metrics.txt for evaluation results

results/plots/ for visualizations

Current model comparison summary:

- RandomForest_SMOTE: precision 0.87, recall 0.83, f1-score 0.85, ROC-AUC 0.9685
- LogisticRegression_SMOTE: precision 0.06, recall 0.92, f1-score 0.11, ROC-AUC 0.9708
- MLP_SMOTE: precision 0.72, recall 0.81, f1-score 0.76, ROC-AUC 0.9635

Based on these results, RandomForest_SMOTE gave the best overall balance for fraud detection.

## Future Improvements

- tune hyperparameters
- add more result visualizations
- improve final report and discussion
- explore additional imbalance handling techniques


Author

Dhvani Yash