import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from fpdf import FPDF

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Apply custom background (light theme)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f9f9f9;
        color: #262730;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title
st.title("🏦 Loan Default Prediction App")
st.markdown("Use this app to predict whether a loan applicant is likely to **default or not**.")

# Tabs
tab1, tab2, tab3 = st.tabs(["📥 Input Form", "📊 Prediction", "📈 Visuals"])

with tab1:
    st.subheader("👤 Applicant Information")
    gender = st.selectbox("Gender", ("Male", "Female"))
    married = st.selectbox("Married", ("Yes", "No"))
    dependents = st.selectbox("Dependents", ("0", "1", "2", "3+"))
    education = st.selectbox("Education", ("Graduate", "Not Graduate"))
    self_employed = st.selectbox("Self Employed", ("No", "Yes"))
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_amount_term = st.selectbox("Loan Term (months)", (360, 180, 120, 240, 300, 60))
    credit_history = st.selectbox("Credit History", ("Good (1)", "Bad (0)"))
    property_area = st.selectbox("Property Area", ("Urban", "Semiurban", "Rural"))

    if st.button("🔮 Predict Loan Default"):
        # Encode inputs
        gender = 1 if gender == "Male" else 0
        married = 1 if married == "Yes" else 0
        education = 0 if education == "Graduate" else 1
        self_employed = 1 if self_employed == "Yes" else 0
        credit_history = 1 if credit_history == "Good (1)" else 0
        property_area_dict = {"Urban": 2, "Semiurban": 1, "Rural": 0}
        property_area = property_area_dict[property_area]
        dependents = 3 if dependents == "3+" else int(dependents)

        input_data = np.array([[gender, married, dependents, education, self_employed,
                                applicant_income, coapplicant_income, loan_amount,
                                loan_amount_term, credit_history, property_area]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        result_label = "High Risk" if prediction == 1 else "Low Risk"

        with tab2:
            st.subheader("📊 Prediction Result")
            st.success(f"The applicant is predicted to be **{result_label}**.")

            # Generate PDF report
            def generate_pdf(input_dict, prediction_result):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=14)
                pdf.cell(200, 10, txt="Loan Default Prediction Report", ln=True, align='C')
                pdf.ln(10)

                pdf.set_font("Arial", size=12)
                for key, value in input_dict.items():
                    pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

                pdf.ln(5)
                pdf.set_font("Arial", "B", 12)
                pdf.set_text_color(255, 0, 0) if prediction_result == "High Risk" else pdf.set_text_color(0, 128, 0)
                pdf.cell(200, 10, txt=f"Prediction: {prediction_result}", ln=True)

                file_path = "/tmp/loan_report.pdf"
                pdf.output(file_path)
                return file_path

            input_dict = {
                "Gender": "Male" if gender == 1 else "Female",
                "Married": "Yes" if married == 1 else "No",
                "Dependents": dependents,
                "Education": "Graduate" if education == 0 else "Not Graduate",
                "Self Employed": "Yes" if self_employed == 1 else "No",
                "Applicant Income": applicant_income,
                "Coapplicant Income": coapplicant_income,
                "Loan Amount": loan_amount,
                "Loan Term": loan_amount_term,
                "Credit History": "Good" if credit_history == 1 else "Bad",
                "Property Area": [ "Rural", "Semiurban", "Urban" ][property_area]
            }

            pdf_file = generate_pdf(input_dict, result_label)

            with open(pdf_file, "rb") as f:
                st.download_button("📄 Download PDF Report", f, file_name="Loan_Prediction_Report.pdf")

        with tab3:
            st.subheader("🔍 Applicant Data Visualization")

            df = pd.DataFrame([input_dict])
            fig = px.bar(df.T, labels={"index": "Feature", "value": "Value"}, title="User Input Summary")
            st.plotly_chart(fig)

            fig2 = px.pie(values=[1 if result_label == "High Risk" else 0, 1 if result_label == "Low Risk" else 0],
                          names=["High Risk", "Low Risk"],
                          title="Prediction Pie Chart")
            st.plotly_chart(fig2)
