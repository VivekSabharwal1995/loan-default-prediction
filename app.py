import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# App title
st.set_page_config(page_title="Loan Default Prediction", layout="centered")
st.title("🏦 Loan Default Prediction App")
st.write("Enter applicant details to predict loan default risk.")

# Input fields
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
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Additional inputs for charts
age = st.number_input("Applicant Age", min_value=18, max_value=100)
income = applicant_income + coapplicant_income
credit_score = st.slider("Credit Score (300–900)", min_value=300, max_value=900, value=700)

# Convert inputs to numeric codes
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
dependents = 3 if dependents == "3+" else int(dependents)
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1 if credit_history == "Good (1)" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# Create input array
input_data = np.array([[gender, married, dependents, education,
                        self_employed, applicant_income, coapplicant_income,
                        loan_amount, loan_amount_term, credit_history, property_area]])

# Scale numeric features
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("🔍 Predict Loan Default"):
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error("❌ High Risk: Loan Likely to Default.")
    else:
        st.success("✅ Low Risk: Loan Likely to be Approved.")

    # 🎯 User Input Bar Chart
    st.subheader("📊 User Input Summary")
    input_summary = {
        'Age': age,
        'Loan Amount': loan_amount,
        'Annual Income': income,
        'Credit Score': credit_score
    }
    input_df = pd.DataFrame.from_dict(input_summary, orient='index', columns=['Value'])
    fig = px.bar(input_df, x=input_df.index, y='Value', text='Value', title="User Feature Breakdown")
    st.plotly_chart(fig)

    # 🎯 Prediction Pie Chart
    st.subheader("📌 Prediction Result")
    result_label = "Default Risk" if prediction[0] == 1 else "Low Risk"
    pie_data = pd.DataFrame({
        'Result': [result_label, ''],
        'Value': [1, 0]
    })
    fig2 = px.pie(pie_data, names='Result', values='Value',
                  color='Result',
                  color_discrete_map={'Default Risk': 'red', 'Low Risk': 'green'},
                  title='Prediction Status')
    st.plotly_chart(fig2)
