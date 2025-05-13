import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("C:/Users/Gaurav/Started with ML/random forest/heart_disease_model.pkl")
scaler = joblib.load("C:/Users/Gaurav/Started with ML/random forest/scaler.pkl")
feature_names = joblib.load("C:/Users/Gaurav/Started with ML/random forest/feature_names.pkl")

# Streamlit UI
st.title("❤️ Heart Disease Prediction App")
st.write("Enter the following medical details to predict the risk of heart disease:")

# Input fields
input_dict = {}
for feature in feature_names:
    input_dict[feature] = st.number_input(f"{feature}", value=0.0)

# Predict Button
if st.button("Predict"):
    input_df = pd.DataFrame([input_dict])

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]
    result = "High Risk of Heart Disease" if prediction == 1 else "Low Risk of Heart Disease"

    st.subheader("🩺 Prediction Result:")
    st.success(result)
