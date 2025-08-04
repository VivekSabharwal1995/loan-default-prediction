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
    result = "✅ Low Risk: Loan Likely to be Repaid." if prediction == 1 else "❌ High Risk: Loan Likely to Default."

    st.markdown("### 📌 Prediction Result")
    st.markdown(f"<h3 style='color:#00ffcc;'>{result}</h3>", unsafe_allow_html=True)

    # --- Show Summary Table ---
    input_dict = {
        "Gender": "Male" if gender else "Female",
        "Married": "Yes" if married else "No",
        "Dependents": dependents,
        "Education": "Graduate" if education == 0 else "Not Graduate",
        "Self Employed": "Yes" if self_employed else "No",
        "Applicant Income": applicant_income,
        "Coapplicant Income": coapplicant_income,
        "Loan Amount": loan_amount,
        "Loan Term": loan_amount_term,
        "Credit History": "Good" if credit_history == 1.0 else "Bad",
        "Property Area": list(property_area_dict.keys())[list(property_area_dict.values()).index(property_area)]
    }

    summary_df = pd.DataFrame.from_dict(input_dict, orient='index', columns=["Details"])
    st.markdown("### 📊 User Input Summary")
    st.table(summary_df)

    # --- Gauge chart (Income ratio) ---
    income_ratio = round(applicant_income / (loan_amount * 1000 + 1), 2)
    gauge_fig = px.indicator.Indicator(
        mode="gauge+number",
        value=income_ratio,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Income-to-Loan Ratio"},
        gauge={'axis': {'range': [0, 1]}}
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

    # --- Donut chart (Loan components) ---
    loan_parts = {
        "Applicant Income": applicant_income,
        "Coapplicant Income": coapplicant_income,
        "Loan Amount ×1000": loan_amount * 1000
    }
    donut_df = pd.DataFrame({
        "Type": list(loan_parts.keys()),
        "Value": list(loan_parts.values())
    })
    donut_fig = px.pie(donut_df, names="Type", values="Value", hole=0.5,
                       title="Income vs Loan Breakdown", color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(donut_fig, use_container_width=True)

    # --- Generate PDF ---
    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Loan Default Prediction Report", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        for key, value in input_dict.items():
            pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt=f"Prediction: {result}", ln=True)
        return pdf.output(dest="S").encode("latin1")

    pdf_data = generate_pdf()
    st.download_button("📄 Download Prediction Report", data=pdf_data, file_name="Loan_Prediction_Report.pdf")

# --- Footer ---
st.markdown("---")
st.markdown("<center>Made with ❤️ using Streamlit</center>", unsafe_allow_html=True)
