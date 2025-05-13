# app.py

import streamlit as st
import joblib
import pandas as pd

# Load model, scaler, and feature list
model = joblib.load("C:/Users/Gaurav/Started with ML/svm/svm_model.pkl")
scaler = joblib.load("C:/Users/Gaurav/Started with ML/svm/scaler.pkl")
features = joblib.load("C:/Users/Gaurav/Started with ML/svm/features.pkl")

st.title("🧬 Breast Cancer Detection (SVM)")
st.markdown("Predict whether a tumor is **Benign** or **Malignant**")

# Create input fields
input_data = {}
for feature in features:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

# Predict
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    result = "Malignant" if prediction == 0 else "Benign"
    st.success(f"Prediction: **{result}**")
    st.info(f"Confidence: Benign: {prob[1]*100:.2f}%, Malignant: {prob[0]*100:.2f}%")
