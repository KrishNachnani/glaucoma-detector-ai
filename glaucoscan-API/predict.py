# predict.py

import pickle
import joblib
import numpy as np
import sys
import pandas as pd


print("Starting prediction...")

# ---- Load saved model and preprocessing steps ----
with open("models/glaucoma_mlp_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded.")

scaler = joblib.load("models/scaler.joblib")
selector = joblib.load("models/feature_selector.joblib")

with open("models/class_mapping.pkl", "rb") as f:
    class_mapping = pickle.load(f)


print("Preprocessing files loaded.")

# ---- Load input from CSV ----
input_csv_path = "example_input.csv"  
print(f"Loading input from {input_csv_path}...")

input_data = pd.read_csv("example_input.csv", header=None)
input_data.columns = [f"feature_{i}" for i in range(input_data.shape[1])]

print(f"Input shape: {input_data.shape}")

# ---- Preprocess ----
X_scaled = scaler.transform(input_data)
X_selected = selector.transform(X_scaled)


# ---- Predict ----
y_pred = model.predict(X_selected)[0]
label_map = class_mapping.get('label', {})
label = label_map.get(y_pred, f"Class {y_pred}")
print(f"Prediction: {label} (class {y_pred})")
