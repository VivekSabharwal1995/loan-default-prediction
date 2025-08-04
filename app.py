import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from fpdf import FPDF

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Set page config with dark theme
st.set_page_config(page_title="Loan Default Prediction", layout="centered")

# --- Title ---
st.markdown("<h1 style='color:#f63366;'>🏦 Loan Default Prediction App</h1>", unsafe_allow_html=True)
st.write("Predict the risk of a loan default using applicant details.")

# --- Sidebar Form for Input ---
st.sidebar.header("📋 Applicant Information")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount (in thousands)", min_value=0)
loan_amount_term = st.sidebar.number_input("Loan Amount Term (in months)", min_value=0)
credit_history = st.sidebar.selectbox("Credit History", ["Good (1)", "Bad (0)"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# --- Encode inputs ---
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
dependents = 3 if dependents == "3+" else int(dependents)
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1.0 if credit_history == "Good (1)" else 0.0
property_area_dict = {"Urban": 2, "Semiurban": 1, "Rural": 0}
property_area = property_area_dict[property_area]

# --- Prepare input ---
input_data = np.array([[gender, married, dependents, education, self_employed,
                        applicant_income, coapplicant_income, loan_amount,
                        loan_amount_term, credit_history, property_area]])
input_scaled = scaler.transform(input_data)

# --- Predict ---
if st.sidebar.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    result = "✅ Low Risk"

