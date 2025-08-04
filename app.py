import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("loan_model.pkl")

# App title
st.set_page_config(page_title="Loan Default Prediction", page_icon="💰", layout="centered")
st.title("💼 Loan Default Prediction App")

st.sidebar.header("🔍 Enter Applicant Details")

# Sidebar inputs
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
loan_amount = st.sidebar.slider("Loan Amount (in 1000s)", min_value=50, max_value=700, step=10)
loan_term = st.sidebar.selectbox("Loan Term (Months)", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480])
credit_history = st.sidebar.radio("Credit History", [1.0, 0.0])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

if st.sidebar.button("Predict"):
    # Preprocess inputs (same encoding used in training)
    input_data = pd.DataFrame({
        'Gender': [1 if gender == 'Male' else 0],
        'Married': [1 if married == 'Yes' else 0],
        'Education': [0 if education == 'Graduate' else 1],
        'Self_Employed': [1 if self_employed == 'Yes' else 0],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_term],
        'Credit_History': [credit_history],
        'Property_Area': [0 if property_area == 'Urban' else 1 if property_area == 'Rural' else 2]
    })

    # Prediction
    pred = model.predict(input_data)[0]
    pred_proba = model.predict_proba(input_data)[0]

    # Result message
    if pred == 0:
        st.success("✅ You are unlikely to default on the loan!")
        st.markdown("😄 **Low Risk**")
    else:
        st.error("⚠️ High risk of loan default!")
        st.markdown("😟 **Consider reviewing your financials.**")

    # Show probability chart
    st.subheader("📊 Prediction Probability")
    fig, ax = plt.subplots()
    labels = ["Non-Default", "Default"]
    ax.bar(labels, pred_proba, color=["green", "red"])
    ax.set_ylabel("Probability")
    st.pyplot(fig)




