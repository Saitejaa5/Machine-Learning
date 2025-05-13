import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("C:/Users/Gaurav/Started with ML/KNN/wine_knn_model.pkl")
scaler = joblib.load("C:/Users/Gaurav/Started with ML/KNN/scaler.pkl")

# Page title
st.title("🍷 Wine Type Classifier (KNN)")
st.write("Enter the chemical features to classify the wine type:")

# Feature input sliders
alcohol = st.number_input("Alcohol", 10.0, 15.0, step=0.1)
malic_acid = st.number_input("Malic Acid", 0.5, 5.0, step=0.1)
ash = st.number_input("Ash", 1.0, 3.5, step=0.1)
alcalinity_of_ash = st.number_input("Alcalinity of Ash", 10.0, 30.0, step=0.5)
magnesium = st.number_input("Magnesium", 70, 160, step=1)
total_phenols = st.number_input("Total Phenols", 0.5, 4.0, step=0.1)
flavanoids = st.number_input("Flavanoids", 0.0, 5.0, step=0.1)
nonflavanoid_phenols = st.number_input("Nonflavanoid Phenols", 0.0, 1.0, step=0.1)
proanthocyanins = st.number_input("Proanthocyanins", 0.0, 4.0, step=0.1)
color_intensity = st.number_input("Color Intensity", 1.0, 15.0, step=0.5)
hue = st.number_input("Hue", 0.5, 2.0, step=0.1)
od280_od315 = st.number_input("OD280/OD315 of Diluted Wines", 1.0, 4.0, step=0.1)
proline = st.number_input("Proline", 300, 1700, step=10)

# Collect input
input_data = np.array([[alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
                        total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
                        color_intensity, hue, od280_od315, proline]])

# Predict
if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    wine_type = ["Class 0", "Class 1", "Class 2"]
    st.success(f"Predicted Wine Type: {wine_type[prediction[0]]}")
