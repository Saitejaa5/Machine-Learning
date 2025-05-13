import streamlit as st
import joblib
import numpy as np

# Load the model and vectorizer
model = joblib.load("C:/Users/Gaurav/Started with ML/naive bayes/spam_classifier_model.pkl")
vectorizer = joblib.load("C:/Users/Gaurav/Started with ML/naive bayes/tfidf_vectorizer.pkl")

# App title
st.title("📩 Spam Email Classifier")
st.write("Enter a message below to check if it's spam or not.")

# User input
user_input = st.text_area("Enter your message:")

# Prediction
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)[0]
        label = "🔴 Spam" if prediction == 1 else "🟢 Not Spam"
        st.success(f"Prediction: {label}")
