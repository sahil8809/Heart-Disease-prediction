import streamlit as st
import joblib
import pandas as pd

# Load Model
model = joblib.load("heart_model.pkl")

st.title("❤️ Heart Disease Prediction")

# User Inputs
age = st.number_input("Age", min_value=20, max_value=100, value=50)

sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.selectbox(
    "Chest Pain Type",
    options=[0, 1, 2, 3]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=80,
    max_value=220,
    value=120
)

chol = st.number_input(
    "Cholesterol",
    min_value=100,
    max_value=600,
    value=200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    options=[0, 1]
)

restecg = st.selectbox(
    "Rest ECG",
    options=[0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=60,
    max_value=220,
    value=150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    options=[0, 1]
)

oldpeak = st.number_input(
    "Old Peak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope = st.selectbox(
    "Slope",
    options=[0, 1, 2]
)

ca = st.selectbox(
    "Number of Major Vessels",
    options=[0, 1, 2, 3]
)

thal = st.selectbox(
    "Thal",
    options=[0, 1, 2, 3]
)


if st.button("Predict"):

    input_data = pd.DataFrame([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]],
                              columns=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease Detected")