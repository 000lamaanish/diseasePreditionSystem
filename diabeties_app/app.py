
import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Diabetes Predictor", layout="centered")

st.title("🧪 Diabetes Risk Predictor")
st.write("Enter patient details:")

# Inputs
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 200, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 100, 30)

# Prediction
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

    prob = model.predict_proba(input_data)[0][1]

    st.subheader(f"Diabetes Probability: {prob:.2f}")

    # Risk levels (based on your threshold learning)
    if prob < 0.3:
        st.success("Low Risk ✅")
    elif prob < 0.6:
        st.warning("Medium Risk ⚠️")
    else:
        st.error("High Risk 🔴")

