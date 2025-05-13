import streamlit as st
import joblib
import numpy as np

model = joblib.load("C:/Users/Gaurav/Started with ML/regression/car prediction/car_price_model1.joblib")
scaler = joblib.load("C:/Users/Gaurav/Started with ML/regression/car prediction/scaler.joblib")
columns = joblib.load("C:/Users/Gaurav/Started with ML/regression/car prediction/model_columns1.joblib")

st.title("🚗 Car Price Prediction")

# Inputs
user_input = {}

# Manual numeric inputs
for field in ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 
              'curbweight', 'enginesize', 'boreratio', 'stroke',
              'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']:
    user_input[field] = st.number_input(f"{field}", step=1.0)

# Dropdowns
dropdowns = {
    'fueltype': ['gas', 'diesel'],
    'aspiration': ['std', 'turbo'],
    'doornumber': ['two', 'four'],
    'carbody': ['sedan', 'hatchback', 'wagon', 'hardtop', 'convertible'],
    'drivewheel': ['fwd', 'rwd', '4wd'],
    'enginelocation': ['front', 'rear'],
    'enginetype': ['ohc', 'ohcf', 'dohc', 'ohcv', 'rotor', 'l'],
    'cylindernumber': ['four', 'six', 'five', 'three', 'eight', 'two', 'twelve'],
    'fuelsystem': ['mpfi', '2bbl', '1bbl', 'spdi', '4bbl', 'idi', 'spfi'],
    'CarBrand': ['toyota', 'bmw', 'nissan', 'audi', 'mazda', 'chevrolet', 'volkswagen',
                 'mercedes-benz', 'subaru', 'mitsubishi', 'porsche', 'volvo',
                 'honda', 'peugeot', 'dodge', 'buick', 'renault', 'jaguar', 'saab']
}

# One-hot encode the user selection
for feature, options in dropdowns.items():
    selected = st.selectbox(f"{feature}", options)
    for opt in options:
        user_input[f"{feature}_{opt}"] = 1 if selected == opt else 0

# Final input
input_data = []
for col in columns:
    input_data.append(user_input.get(col, 0))

input_data_scaled = scaler.transform([input_data])

if st.button("Predict"):
    prediction = model.predict(input_data_scaled)[0]
    st.success(f"Estimated Car Price: ${prediction:,.2f}")
