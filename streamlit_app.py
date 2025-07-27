import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("random_forest_model.pkl")

# Define the feature input order
feature_order = ['thal_2', 'thal_3', 'oldpeak', 'thalach', 'exang', 'ca', 'slope_2']

# Page title
st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")
st.title("💓 Heart Disease Risk Predictor")
st.markdown("Enter patient clinical data to predict the likelihood of heart disease.")

# User input form
with st.form("input_form"):
    thal_2 = st.selectbox("Thalassemia Type 2 (thal_2)", [0, 1])
    thal_3 = st.selectbox("Thalassemia Type 3 (thal_3)", [0, 1])
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=6.5, step=0.1)
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=50.0, max_value=210.0, step=1.0)
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    ca = st.number_input("Number of Vessels Colored (ca)", min_value=0, max_value=3, step=1)
    slope_2 = st.selectbox("ST Slope Level (slope_2)", [0, 1])
    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    input_data = pd.DataFrame([[thal_2, thal_3, oldpeak, thalach, exang, ca, slope_2]], columns=feature_order)
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][prediction]

    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High risk of heart disease detected with confidence: **{proba:.2f}**")
    else:
        st.success(f"✅ No heart disease detected with confidence: **{proba:.2f}**")