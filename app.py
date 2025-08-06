import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from io import BytesIO
from fpdf import FPDF

# Set page config
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Custom CSS for dark mode styling
st.markdown(
    """
    <style>
    body {
        background-color: #0e1117;
        color: #fafafa;
    }
    .main {
        background-color: #0e1117;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("User Input Features")

# Input fields
gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
married = st.sidebar.selectbox("Married", ("Yes", "No"))
dependents = st.sidebar.selectbox("Dependents", ("0", "1", "2", "3+"))
education = st.sidebar.selectbox("Education", ("Graduate", "Not Graduate"))
self_employed = st.sidebar.selectbox("Self Employed", ("Yes", "No"))
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_amount_term = st.sidebar.selectbox("Loan Amount Term (in days)", (360, 180, 120, 240, 300))
credit_history = st.sidebar.selectbox("Credit History", (1.0, 0.0))
property_area = st.sidebar.selectbox("Property Area", ("Urban", "Semiurban", "Rural"))

# Mapping inputs
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0
dependents = 3 if dependents == "3+" else int(dependents)
property_dict = {"Urban": 2, "Semiurban": 1, "Rural": 0}
property_area = property_dict[property_area]

# Input data for prediction
input_data = np.array([[gender, married, dependents, education, self_employed,
                        applicant_income, coapplicant_income, loan_amount,
                        loan_amount_term, credit_history, property_area]])
scaled_input = scaler.transform(input_data)

# Prediction
if st.sidebar.button("Predict"):
    prediction = model.predict(scaled_input)[0]

    st.subheader("🎯 Prediction Result")
    if prediction == 1:
        st.success("✅ Loan will be **Approved**.")
    else:
        st.error("❌ Loan will be **Rejected**.")

    # Visualization
    st.subheader("📊 Applicant Overview")

    # Bar chart
    bar_data = pd.DataFrame({
        "Feature": ["Applicant Income", "Coapplicant Income", "Loan Amount"],
        "Value": [applicant_income, coapplicant_income, loan_amount]
    })
    st.plotly_chart(px.bar(bar_data, x="Feature", y="Value", title="Income vs Loan"))

    # Donut chart
    donut_data = pd.DataFrame({
        "Category": ["Applicant Income", "Coapplicant Income"],
        "Amount": [applicant_income, coapplicant_income]
    })
    fig = px.pie(donut_data, names="Category", values="Amount", hole=0.5, title="Income Distribution")
    st.plotly_chart(fig)

    # PDF Report
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

        # 🛠 Fixed: convert PDF to BytesIO
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        return BytesIO(pdf_bytes)

    st.download_button(
        label="📄 Download Prediction Report (PDF)",
        data=generate_pdf(),
        file_name="loan_prediction_report.pdf",
        mime="application/pdf"
    )
