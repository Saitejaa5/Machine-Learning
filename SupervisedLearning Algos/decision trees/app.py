import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model and feature names
model = pickle.load(open("C:/Users/Gaurav/Started with ML/decision tree/student_model.pkl", 'rb'))
with open("C:/Users/Gaurav/Started with ML/decision tree/feature_columns.pkl", 'rb') as f:
    feature_columns = pickle.load(f)

st.title("🎓 Student Performance Predictor")
st.write("Predict if a student will pass based on their habits and background.")

# Input form
sex = st.selectbox("Sex", ['M', 'F'])
age = st.slider("Age", 15, 22, 17)
studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4])
failures = st.slider("Number of Past Failures", 0, 4, 0)
internet = st.selectbox("Internet Access", ['yes', 'no'])
goout = st.slider("Going Out with Friends (1-5)", 1, 5, 3)
Dalc = st.slider("Workday Alcohol Consumption (1-5)", 1, 5, 1)
Walc = st.slider("Weekend Alcohol Consumption (1-5)", 1, 5, 1)
absences = st.slider("Number of School Absences", 0, 30, 4)

# Prepare input
input_dict = {
    'age': age,
    'studytime': studytime,
    'failures': failures,
    'goout': goout,
    'Dalc': Dalc,
    'Walc': Walc,
    'absences': absences,
    'sex_M': 1 if sex == 'M' else 0,
    'internet_yes': 1 if internet == 'yes' else 0,
    # Add other dummy vars with default 0 if not in input
}

# Convert to DataFrame and fill missing dummy vars
input_df = pd.DataFrame([input_dict])
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns
input_df = input_df[feature_columns]

# Predict
if st.button("Predict"):
    result = model.predict(input_df)[0]
    st.subheader("🎯 Prediction:")
    st.success("✅ Pass" if result == 1 else "❌ Fail")
