import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from io import BytesIO
from fpdf import FPDF
import base64

# --- PAGE CONFIG ---
st.set_page_config(page_title="Loan Default Predictor", page_icon="🏦", layout="centered")

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
    body {
        background-color: #1e1e1e;
        color: #f5f5f5;
    }
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1565372912702-6c63320f7cbb');
        background-size: cover;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER WITH LOGO ---
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Logo_IT.svg/1280px-Logo_IT.svg.png", width=100)
st.title("🏦 Loan Default Prediction App")
st.markdown("Predict loan default risk using applicant details. This app uses a machine learning model trained on past data.")

# --- Load model and scaler ---
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# --- Input Section ---
with st.form("loan_form"):
    st.subheader("📋 Applicant Information")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
        loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
        loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0)
        credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submit = st.form_submit_button("Predict Loan Default")

# --- Data Preprocessing ---
if submit:
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    dependents = 3 if dependents == "3+" else int(dependents)
    education = 0 if education == "Graduate" else 1
    self_employed = 1 if self_employed == "Yes" else 0
    credit_history = 1 if credit_history == "Good (1)" else 0
    property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

    input_data = np.array([[gender, married, dependents, education,
                            self_employed, applicant_income, coapplicant_income,
                            loan_amount, loan_amount_term, credit_history, property_area]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    result_label = "❌ High Risk" if prediction == 1 else "✅ Low Risk"

    # --- Show Result ---
    st.subheader("📌 Prediction Result")
    if prediction == 1:
        st.error("❌ High Risk: Loan Likely to Default.")
    else:
        st.success("✅ Low Risk: Loan Likely to be Approved.")

    # --- User Input Bar Chart ---
    st.subheader("📊 User Input Summary")
    input_summary = {
        'Applicant Income': applicant_income,
        'Coapplicant Income': coapplicant_income,
        'Loan Amount': loan_amount,
        'Loan Term': loan_amount_term
    }
    input_df = pd.DataFrame.from_dict(input_summary, orient='index', columns=['Value'])
    fig = px.bar(input_df, x=input_df.index, y='Value', text='Value', title="📈 Feature Breakdown",
                 color_discrete_sequence=["#636EFA"])
    st.plotly_chart(fig)

    # --- Prediction Pie Chart ---
    pie_data = pd.DataFrame({
        'Result': [result_label, ''],
        'Value': [1, 0]
    })
    fig2 = px.pie(pie_data, names='Result', values='Value',
                  color='Result',
                  color_discrete_map={'❌ High Risk': 'red', '✅ Low Risk': 'green'},
                  title='Prediction Status')
    st.plotly_chart(fig2)

    # --- Generate PDF Report ---
    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Loan Default Prediction Report", ln=True, align='C')
        pdf.ln(10)
        for key, val in input_summary.items():
            pdf.cell(200, 10, txt=f"{key}: {val}", ln=True)
        pdf.cell(200, 10, txt=f"Prediction Result: {result_label}", ln=True)
        return pdf.output(dest="S").encode("latin1")

    pdf_data = generate_pdf()
    b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="loan_prediction_report.pdf">📄 Download Report (PDF)</a>'
    st.markdown(href, unsafe_allow_html=True)
