#app.py

import gradio as gr
import pandas as pd
import joblib
import os

MODEL_PATH = "results/models/random_forest_model.pkl"
SCALER_PATH = "results/models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_fraud(file):
    df = pd.read_csv(file.name)

    if "Class" in df.columns:
        df = df.drop("Class", axis=1)

    scaled_data = scaler.transform(df)
    predictions = model.predict(scaled_data)

    result_df = pd.DataFrame()
    result_df["Transaction_Number"] = range(1, len(predictions) + 1)
    result_df["Prediction_Label"] = ["Fraud" if p == 1 else "Not Fraud" for p in predictions]

    fraud_count = (result_df["Prediction_Label"] == "Fraud").sum()
    not_fraud_count = (result_df["Prediction_Label"] == "Not Fraud").sum()

    summary = (
        f"Total Transactions: {len(result_df)} | "
        f"Fraud: {fraud_count} | "
        f"Not Fraud: {not_fraud_count}"
    )

    output_file = "prediction_results.csv"
    result_df.to_csv(output_file, index=False)

    explanation = (
        "These predictions are based on patterns the Random Forest model learned from the training data. "
        "A row predicted as Fraud means it looks similar to transactions the model has seen as fraudulent before."
    )

    return summary, explanation, result_df, output_file

def toggle_submit_button(file):
    if file is None:
        return gr.update(interactive=False, variant="secondary")
    return gr.update(interactive=True, variant="primary")

with gr.Blocks(theme=gr.themes.Soft(), title="Credit Card Fraud Detection App") as demo:
        gr.Markdown("# Credit Card Fraud Detection App")
        gr.Markdown(
        "Upload a CSV file of transactions to predict whether each transaction is **Fraud** or **Not Fraud** using the trained Random Forest model."
        )

        file_input = gr.File(label="Upload CSV File")
        submit_button = gr.Button("Submit", interactive=False, variant="secondary")
        summary_text = gr.Textbox(label="Prediction Summary", interactive=False)
        explanation_text = gr.Textbox(label="What This Means", interactive=False)
        gr.Markdown("**Note:** Demo predictions depend on how close the uploaded transaction rows are to the real training data used by the model.")
        output_table = gr.Dataframe(label="Fraud Detection Results")
        download_file = gr.File(label="Download Prediction Results")
        file_input.change(fn=toggle_submit_button, inputs=file_input, outputs=submit_button)
        submit_button.click(fn=predict_fraud, inputs=file_input, outputs=[summary_text, explanation_text, output_table, download_file])
demo.launch()