#app.py

import gradio as gr
import pandas as pd
import joblib

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

    return result_df

def toggle_submit_button(file):
    if file is None:
        return gr.update(interactive=False, variant="secondary")
    return gr.update(interactive=True, variant="primary")

with gr.Blocks() as demo:
    gr.Markdown("# Credit Card Fraud Detection App")
    gr.Markdown("Upload a CSV file to detect fraudulent transactions using the trained Random Forest model.")

    file_input = gr.File(label="Upload CSV File")
    submit_button = gr.Button("Submit", interactive=False, variant="secondary")
    output_table = gr.Dataframe(label="Fraud Detection Results")

    file_input.change(fn=toggle_submit_button, inputs=file_input, outputs=submit_button)
    submit_button.click(fn=predict_fraud, inputs=file_input, outputs=output_table)

demo.launch()