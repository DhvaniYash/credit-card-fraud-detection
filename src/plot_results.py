# src/plot_results.py

import os
import re
import matplotlib.pyplot as plt


def extract_metrics(metrics_file):
    """
    This function reads the metrics.txt file and extracts the fraud-class
    precision, recall, f1-score, and ROC-AUC for each model.

    Input:
        metrics_file: Path to the metrics text file

    Output:
        results: A dictionary containing the metrics for each model
    """
    with open(metrics_file, "r") as f:
        content = f.read()

    sections = content.split("=" * 60)
    results = {}

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Get model name
        model_match = re.search(r"Model:\s*(.+)", section)
        if not model_match:
            continue
        model_name = model_match.group(1).strip()

        # Get fraud-class metrics (class 1 row)
        class_1_match = re.search(
            r"\n\s*1\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+",
            section
        )

        # Get ROC-AUC score
        roc_match = re.search(r"ROC-AUC Score:\s*([\d.]+)", section)

        if class_1_match and roc_match:
            precision = float(class_1_match.group(1))
            recall = float(class_1_match.group(2))
            f1_score = float(class_1_match.group(3))
            roc_auc = float(roc_match.group(1))

            results[model_name] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "roc_auc": roc_auc
            }

    return results


def plot_metric(models, values, title, ylabel, output_path):
    """
    This function makes a bar chart for one evaluation metric.

    Input:
        models: List of model names
        values: List of metric values
        title: Title of the plot
        ylabel: Label for the y-axis
        output_path: File path to save the plot

    Output:
        None
    """
    plt.figure(figsize=(8, 5))
    plt.bar(models, values)
    plt.title(title)
    plt.xlabel("Models")
    plt.ylabel(ylabel)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    """
    This function reads saved metrics automatically and creates comparison plots.

    Input:
        None

    Output:
        None directly, but it saves the plots in the results/plots folder.
    """
    metrics_file = "results/metrics.txt"
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)

    results = extract_metrics(metrics_file)

    models = list(results.keys())
    precision = [results[model]["precision"] for model in models]
    recall = [results[model]["recall"] for model in models]
    f1_score = [results[model]["f1_score"] for model in models]
    roc_auc = [results[model]["roc_auc"] for model in models]

    plot_metric(
        models,
        precision,
        "Precision Comparison",
        "Precision",
        os.path.join(output_dir, "precision_comparison.png")
    )

    plot_metric(
        models,
        recall,
        "Recall Comparison",
        "Recall",
        os.path.join(output_dir, "recall_comparison.png")
    )

    plot_metric(
        models,
        f1_score,
        "F1-Score Comparison",
        "F1-score",
        os.path.join(output_dir, "f1_score_comparison.png")
    )

    plot_metric(
        models,
        roc_auc,
        "ROC-AUC Comparison",
        "ROC-AUC",
        os.path.join(output_dir, "roc_auc_comparison.png")
    )

    print("Plots saved successfully in results/plots")


if __name__ == "__main__":
    main()