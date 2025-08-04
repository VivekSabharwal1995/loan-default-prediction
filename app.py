import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
from fpdf import FPDF
import plotly.graph_objects as go
import base64

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Set page config
st.set_page_config(page_title="Loan Default Prediction", layout="wide", page_icon="💸")

# Background color and style
st.markdown("""
    <style>
    body {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .reportview-container {
        background: #1e1e1e;
    }
    .sidebar .sidebar-content {
        background: #111;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("💸 Loan Default Prediction App")

# Sidebar form
st.sidebar.header("🔍 Enter Applicant Details")
with st.sidebar.form("input_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
    loan_amount_term = st.number_input("Loan Term (in days)", min_value=0)
    credit_history = st.selectbox("Credit History", ["1.0", "0.0"])
    property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
    submit = st.form_submit_button("Predict")

if submit:
    # Encoding inputs
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    dependents = 3 if dependents == "3+" else int(dependents)
    education = 0 if education == "Graduate" else 1
    self_employed = 1 if self_employed == "Yes" else 0
    credit_history = float(credit_history)

    property_dict = {"Urban": 2, "Rural": 0, "Semiurban": 1}
    property_area = property_dict[property_area]

    input_data = pd.DataFrame([[
        gender, married, dependents, education, self_employed,
        applicant_income, coapplicant_income, loan_amount,
        loan_amount_term, credit_history, property_area
    ]], columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                 'Loan_Amount_Term', 'Credit_History', 'Property_Area'])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    st.subheader("🧾 Prediction Result:")
    if prediction == 1:
        st.success("✅ Loan is likely to be approved (No Default Expected).")
    else:
        st.error("❌ Loan is likely to be rejected or defaulted.")

    # --- Visuals ---
    st.markdown("### 📊 Visual Summary")
    col1, col2 = st.columns(2)

    # Gauge Chart
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=loan_amount,
        title={'text': "Loan Amount (in Thousands)", 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 700]},
            'bar': {'color': "deepskyblue"},
            'steps': [
                {'range': [0, 300], 'color': "#333"},
                {'range': [300, 700], 'color': "#666"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': loan_amount
            }
        }
    ))

    # Donut Chart
    donut_fig = go.Figure(data=[go.Pie(
        labels=['Applicant Income', 'Coapplicant Income'],
        values=[applicant_income, coapplicant_income],
        hole=0.5,
        marker=dict(colors=["#00BFFF", "#FF6347"])
    )])
    donut_fig.update_layout(title_text="Income Contribution Breakdown")

    col1.plotly_chart(gauge_fig, use_container_width=True)
    col2.plotly_chart(donut_fig, use_container_width=True)

    # --- PDF Download ---
    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Loan Default Prediction Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", '', 12)
        fields = {
            "Gender": "Male" if gender else "Female",
            "Married": "Yes" if married else "No",
            "Dependents": dependents,
            "Education": "Graduate" if education == 0 else "Not Graduate",
            "Self Employed": "Yes" if self_employed else "No",
            "Applicant Income": applicant_income,
            "Coapplicant Income": coapplicant_income,
            "Loan Amount": loan_amount,
            "Loan Term": loan_amount_term,
            "Credit History": credit_history,
            "Property Area": list(property_dict.keys())[list(property_dict.values()).index(property_area)],
            "Prediction": "Loan Approved" if prediction == 1 else "Loan Rejected"
        }

        for key, value in fields.items():
            pdf.cell(0, 10, f"{key}: {value}", ln=True)

        pdf_output = BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)
        return pdf_output

    pdf_btn = st.download_button(
        label="📄 Download Prediction Report (PDF)",
        data=generate_pdf(),
        file_name="loan_prediction_report.pdf",
        mime="application/pdf"
    )
