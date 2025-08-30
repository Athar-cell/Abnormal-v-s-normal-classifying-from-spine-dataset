import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("classifier.pkl")

st.title("Normal vs Abnormal Classifier")

st.write("Enter patient data to predict whether it's Normal or Abnormal.")

# Example: suppose you have 3 features
feature1 = st.number_input("Enter Feature 1")
feature2 = st.number_input("Enter Feature 2")
feature3 = st.number_input("Enter Feature 3")

if st.button("Predict"):
    # Convert inputs into array
    input_data = np.array([[feature1, feature2, feature3]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]

    # Show result
    if prediction == 1:
        st.error("Abnormal ❌")
    else:
        st.success("Normal ✅")
