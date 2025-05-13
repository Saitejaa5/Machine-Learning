import streamlit as st
import pickle
import numpy as np

# Load model and columns
model = pickle.load(open("C:/Users/Gaurav/Started with ML/regression/car prediction/car_price_model.pkl", "rb"))
model_columns = pickle.load(open("C:/Users/Gaurav/Started with ML/regression/car prediction/model_columns.pkl", "rb"))

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")
st.title("🚗 Car Price Prediction App")
st.markdown("Fill the details below to estimate car price.")

# --- User Inputs ---
symboling = st.slider("Symboling", -2, 3, 0)
wheelbase = st.number_input("Wheelbase")
carlength = st.number_input("Car Length")
carwidth = st.number_input("Car Width")
carheight = st.number_input("Car Height")
curbweight = st.number_input("Curb Weight")
enginesize = st.number_input("Engine Size")
boreratio = st.number_input("Bore Ratio")
stroke = st.number_input("Stroke")
compressionratio = st.number_input("Compression Ratio")
horsepower = st.number_input("Horsepower")
peakrpm = st.number_input("Peak RPM")
citympg = st.number_input("City MPG")
highwaympg = st.number_input("Highway MPG")

# Categorical Inputs
fueltype = st.selectbox("Fuel Type", ["gas", "diesel"])
aspiration = st.selectbox("Aspiration", ["std", "turbo"])
doornumber = st.selectbox("Number of Doors", ["two", "four"])
carbody = st.selectbox("Car Body", ["sedan", "hatchback", "wagon", "hardtop", "convertible"])
drivewheel = st.selectbox("Drive Wheel", ["fwd", "rwd", "4wd"])
enginelocation = st.selectbox("Engine Location", ["front", "rear"])
enginetype = st.selectbox("Engine Type", ['ohc', 'ohcf', 'dohc', 'ohcv', 'rotor', 'l'])
cylindernumber = st.selectbox("Number of Cylinders", ['four', 'six', 'five', 'three', 'eight', 'two', 'twelve'])
fuelsystem = st.selectbox("Fuel System", ['mpfi', '2bbl', '1bbl', 'spdi', '4bbl', 'idi', 'spfi'])
carbrand = st.selectbox("Car Brand", ['toyota', 'bmw', 'nissan', 'audi', 'mazda', 'chevrolet',
                                      'volkswagen', 'mercedes-benz', 'subaru', 'mitsubishi', 
                                      'porsche', 'volvo', 'honda', 'peugeot', 'dodge', 'buick', 'renault', 'jaguar', 'saab'])

# --- Convert to Input Vector ---
input_data = {
    'symboling': symboling,
    'wheelbase': wheelbase,
    'carlength': carlength,
    'carwidth': carwidth,
    'carheight': carheight,
    'curbweight': curbweight,
    'enginesize': enginesize,
    'boreratio': boreratio,
    'stroke': stroke,
    'compressionratio': compressionratio,
    'horsepower': horsepower,
    'peakrpm': peakrpm,
    'citympg': citympg,
    'highwaympg': highwaympg,
    f'fueltype_{fueltype}': 1,
    f'aspiration_{aspiration}': 1,
    f'doornumber_{doornumber}': 1,
    f'carbody_{carbody}': 1,
    f'drivewheel_{drivewheel}': 1,
    f'enginelocation_{enginelocation}': 1,
    f'enginetype_{enginetype}': 1,
    f'cylindernumber_{cylindernumber}': 1,
    f'fuelsystem_{fuelsystem}': 1,
    f'CarBrand_{carbrand}': 1,
}

# Fill missing with 0
final_input = []
for col in model_columns:
    final_input.append(input_data.get(col, 0))

# Predict
if st.button("Predict Car Price"):
    prediction = model.predict([final_input])[0]
    st.success(f"Estimated Selling Price: $ {prediction:,.2f}")
