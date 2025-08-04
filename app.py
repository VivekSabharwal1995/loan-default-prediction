import streamlit as st
import pandas as pd
import numpy as np
import pickle
from fpdf import FPDF
from io import BytesIO
import plotly.graph_objects as go

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Loan Default Prediction", layout="centered")

st.markdown("<h1 style='text-align: center;'>🏦 Loan Default Prediction App</h1>", unsafe_allow_html=True)
st.markdown("Predict loan default risk using applicant details. This app uses a machine learning model trained on past data.")
st.markdown("---")

# Sidebar form input
with st.form("loan_form"):
    st.subheader("📋 Applicant Information")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
    loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0)
    credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])
    property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    submit = st.form_submit_button("Predict")

if submit:
    # Map inputs
    input_data = {
        "Gender": 1 if gender == "Male" else 0,
        "Married": 1 if married == "Yes" else 0,
        "Dependents": 3 if dependents == "3+" else int(dependents),
        "Education": 1 if education == "Graduate" else 0,
        "Self_Employed": 1 if self_employed == "Yes" else 0,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": 1.0 if credit_history == "Good (1)" else 0.0,
        "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area],
    }

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    result = "✅ Low Risk: Loan Approved." if prediction == 1 else "❌ High Risk: Loan Likely to Default."
    result_color = "green" if prediction == 1 else "red"

    st.markdown("---")
    st.subheader("📌 Prediction Result")
    st.markdown(f"<h3 style='color:{result_color}'>{result}</h3>", unsafe_allow_html=True)

    # Donut chart (risk visualization)
    risk_score = model.predict_proba(input_scaled)[0][0]
    gauge_fig = go.Figure(data=[go.Pie(
        labels=["High Risk", "Low Risk"],
        values=[risk_score, 1-risk_score],
        hole=0.6,
        marker=dict(colors=["red", "green"])
    )])
    gauge_fig.update_layout(width=400, height=300, showlegend=True, title="Loan Risk Breakdown")
    st.plotly_chart(gauge_fig)

    st.markdown("### 📊 User Input Summary")
    input_summary = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self Employed": self_employed,
        "Applicant Income": applicant_income,
        "Coapplicant Income": coapplicant_income,
        "Loan Amount": loan_amount,
        "Loan Term (months)": loan_amount_term,
        "Credit History": credit_history,
        "Property Area": property_area,
    }

    for k, v in input_summary.items():
        st.write(f"**{k}:** {v}")

    # PDF generation
    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Loan Default Prediction Report", ln=True, align='C')
        pdf.ln(10)

        for label, value in input_summary.items():
            pdf.cell(200, 10, txt=f"{label}: {value}", ln=True)

        pdf.ln(10)
        pdf.set_text_color(255, 0, 0) if prediction == 0 else pdf.set_text_color(0, 128, 0)
        pdf.cell(200, 10, txt=f"Prediction: {result}", ln=True)

        pdf_output = pdf.output(dest='S').encode('latin-1')
        return BytesIO(pdf_output)

    st.download_button("📄 Download Report as PDF", data=generate_pdf(), file_name="Loan_Prediction_Report.pdf", mime="application/pdf")
