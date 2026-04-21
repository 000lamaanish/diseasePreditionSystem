
import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import pandas as pd

# Page config
st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")

# Load model
model_path = "model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model not found. Run training first.")
    st.stop()

model = joblib.load(model_path)

# -----------------------
# HEADER
# -----------------------
st.title("🧪 Diabetes Risk Prediction System")
st.markdown("AI-powered early screening tool for diabetes risk assessment")

# -----------------------
# SIDEBAR INPUT
# -----------------------
st.sidebar.header("📋 Patient Details")

pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.number_input("Glucose Level", 0, 200, 120)
bp = st.sidebar.number_input("Blood Pressure", 0, 150, 70)
skin = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.number_input("Age", 1, 100, 30)

# -----------------------
# MAIN AREA
# -----------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Patient Summary")
    st.write(f"Age: {age}")
    st.write(f"Glucose: {glucose}")
    st.write(f"BMI: {bmi}")

with col2:
    st.subheader("🧠 Model Prediction")

# -----------------------
# PREDICTION BUTTON
# -----------------------
if st.button("🔍 Predict Risk"):

    data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    prob = model.predict_proba(data)[0][1]

    st.subheader(f"🧪 Diabetes Probability: {prob:.2f}")

    # -----------------------
    # 📊 PROBABILITY BAR
    # -----------------------
    fig, ax = plt.subplots()
    ax.bar(["No Diabetes", "Diabetes"], [1-prob, prob])
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    # Progress bar
    st.progress(int(prob * 100))

    # -----------------------
    # 🧠 FEATURE IMPORTANCE
    # -----------------------
    st.subheader("🧠 Feature Importance")

    # Get feature importance from model inside pipeline
    rf_model = model.named_steps["model"]

    features = [
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ]

    importance = rf_model.feature_importances_

    df_imp = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    fig2, ax2 = plt.subplots()
    ax2.barh(df_imp["Feature"], df_imp["Importance"])
    ax2.invert_yaxis()
    st.pyplot(fig2)

    # -----------------------
    # Risk classification
    # -----------------------
    if prob < 0.3:
        st.success("🟢 Low Risk")
        st.info("Maintain a healthy lifestyle and regular checkups.")
        
    elif prob < 0.6:
        st.warning("🟡 Medium Risk")
        st.info("Consider consulting a doctor and improving diet/exercise.")
        
    else:
        st.error("🔴 High Risk")
        st.info("Medical attention recommended. Please consult a professional.")

# -----------------------
# FOOTER
# -----------------------
st.markdown("---")
st.markdown("Built with ❤️ using Machine Learning & Streamlit")
