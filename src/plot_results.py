# src/plot_results.py

import os
import matplotlib.pyplot as plt

def main():
    models = ["RandomForest_SMOTE", "LogisticRegression_SMOTE", "MLP_SMOTE"]
    precision = [0.87, 0.06, 0.72]
    recall = [0.83, 0.92, 0.81]
    f1_score = [0.85, 0.11, 0.76]
    roc_auc = [0.9684509513569075, 0.9708434302252134, 0.9635151683070527]

    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: F1-score comparison
    plt.figure(figsize=(8, 5))
    plt.bar(models, f1_score)
    plt.title("F1-Score Comparison")
    plt.xlabel("Models")
    plt.ylabel("F1-score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_score_comparison.png"))
    plt.close()

    # Plot 2: ROC-AUC comparison
    plt.figure(figsize=(8, 5))
    plt.bar(models, roc_auc)
    plt.title("ROC-AUC Comparison")
    plt.xlabel("Models")
    plt.ylabel("ROC-AUC")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_auc_comparison.png"))
    plt.close()

    print("Plots saved successfully in results/plots")

if __name__ == "__main__":
    main()