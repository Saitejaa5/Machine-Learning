import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("C:/Users/Gaurav/Started with ML/regression/house price pridiction/model.pkl")
scaler = joblib.load("C:/Users/Gaurav/Started with ML/regression/house price pridiction/scaler.pkl")

st.title("🏡 California House Price Prediction App")

# Sample values to auto-fill
sample_values = {
    "median_income": 3.5,
    "total_rooms": 1500,
    "total_bedrooms": 300,
    "population": 800,
    "households": 250,
    "latitude": 34.2,
    "longitude": -118.4,
    "rooms_per_household": 6.0,
    "bedrooms_per_room": 0.2,
    "population_per_household": 3.2
}

st.markdown("Fill in the house features below:")

use_sample = st.checkbox("Use Sample Input")

# Collect inputs
median_income = st.number_input("Median Income", value=sample_values["median_income"] if use_sample else 0.0)
total_rooms = st.number_input("Total Rooms", value=sample_values["total_rooms"] if use_sample else 1.0)
total_bedrooms = st.number_input("Total Bedrooms", value=sample_values["total_bedrooms"] if use_sample else 1.0)
population = st.number_input("Population", value=sample_values["population"] if use_sample else 1.0)
households = st.number_input("Households", value=sample_values["households"] if use_sample else 1.0)
latitude = st.number_input("Latitude", format="%.2f", value=sample_values["latitude"] if use_sample else 0.0)
longitude = st.number_input("Longitude", format="%.2f", value=sample_values["longitude"] if use_sample else 0.0)
rooms_per_household = st.number_input("Rooms per Household", value=sample_values["rooms_per_household"] if use_sample else 0.0)
bedrooms_per_room = st.number_input("Bedrooms per Room", value=sample_values["bedrooms_per_room"] if use_sample else 0.0)
population_per_household = st.number_input("Population per Household", value=sample_values["population_per_household"] if use_sample else 0.0)

# Prepare input
input_data = pd.DataFrame([[
    median_income, total_rooms, total_bedrooms, population,
    households, latitude, longitude,
    rooms_per_household, bedrooms_per_room, population_per_household
]], columns=[
    "median_income", "total_rooms", "total_bedrooms", "population",
    "households", "latitude", "longitude",
    "rooms_per_household", "bedrooms_per_room", "population_per_household"
])

# Predict on button click
if st.button("Predict House Price"):
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    st.success(f"🏠 Estimated House Price: ${prediction[0]:,.2f}")
