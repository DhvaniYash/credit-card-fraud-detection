# Credit Card Fraud Detection under Class Imbalance

This project focuses on detecting fraudulent credit card transactions using machine learning. Since fraud cases are very rare compared to normal transactions, this is a class imbalance problem. The main goal of this project is to compare different machine learning models and see which one performs best for fraud detection.

## Project Objective

The main objectives of this project are:
- detect fraudulent credit card transactions
- handle severe class imbalance in the dataset
- compare multiple machine learning models
- evaluate model performance using proper fraud detection metrics

## Dataset

The dataset used in this project is the credit card fraud detection dataset.

- File name: `creditcard.csv`
- Location: `data/creditcard.csv`

The target column is `Class`:
- `0` = normal transaction
- `1` = fraudulent transaction

This dataset is highly imbalanced, so metrics like precision, recall, F1-score, and ROC-AUC are more useful than accuracy alone.

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
│   ├── plot_results.py
│   ├── preprocess.py
│   ├── train.py
│   └── utils.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

### Models Used

The project includes the following models:

- Logistic Regression
- Random Forest
- Multi-Layer Perceptron (MLP)

### Preprocessing

The preprocessing steps used in this project include:

- loading the dataset
- splitting the data into training and testing sets
- scaling the feature values using StandardScaler
- applying SMOTE only on the training data to handle class imbalance

## Evaluation Metrics

The models are evaluated using:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

These metrics are important because fraud detection is an imbalanced classification problem, so accuracy alone is not enough.

### Current Progress

So far, this project includes:

- project folder structure
- data loading and preprocessing
- model training pipeline
- evaluation functions
- saved model outputs
- automatic metric plotting from saved results

### How to Run

Open the project in VS Code.

Open a terminal in the project folder.

Install dependencies using:

```bash
pip install -r requirements.txt
```

Run the training pipeline using:

```bash
python -m src.train
```

Generate the comparison plots using:

```bash
python -m src.plot_results
``` 

### Results

The generated outputs are saved in:

- `results/models/` for saved model files
- `results/metrics.txt` for evaluation results
- `results/plots/` for visualizations

### Current model comparison summary:

- RandomForest_SMOTE: precision 0.87, recall 0.83, f1-score 0.85, ROC-AUC 0.9685
- LogisticRegression_SMOTE: precision 0.06, recall 0.92, f1-score 0.11, ROC-AUC 0.9708
- MLP_SMOTE: precision 0.72, recall 0.81, f1-score 0.76, ROC-AUC 0.9635

Based on these results, RandomForest_SMOTE gave the best overall balance for fraud detection.

### Notes
- `old_baseline_model.py` was used as an earlier baseline experiment.
- `plot_results.py` now reads values automatically from `results/metrics.txt` and creates plots from the saved results.
- The code includes comments and documentation for readability and maintenance.

### Future Improvements
- tune hyperparameters more deeply
- add more result visualizations
- improve final report and discussion
- explore additional imbalance handling techniques

### Author
Dhvani Chaudhari