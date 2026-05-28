import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "models", "stacking_regressor.pkl")
)

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("House Price Prediction using Stacking Regressor")
st.write("Enter house details to predict median house price.")

crim = st.number_input("Crime Rate (crim)", value=0.1)
zn = st.number_input("Residential Land Zoned (%)", value=0.0)
indus = st.number_input("Industrial Proportion", value=8.0)
chas = st.selectbox("Charles River Bound", [0, 1])
nox = st.number_input("Nitric Oxide Concentration", value=0.5)
rm = st.number_input("Average Rooms", value=6.0)
age = st.number_input("Age of Property (%)", value=50.0)
dis = st.number_input("Distance to Employment Centers", value=4.0)
rad = st.number_input("Accessibility to Highways", value=5)
tax = st.number_input("Property Tax Rate", value=300)
ptratio = st.number_input("Pupil-Teacher Ratio", value=18.0)
b = st.number_input("Black Population Metric", value=390.0)
lstat = st.number_input("Lower Status Population %", value=10.0)

if st.button("Predict House Price"):

    input_data = np.array([[
        crim,
        zn,
        indus,
        chas,
        nox,
        rm,
        age,
        dis,
        rad,
        tax,
        ptratio,
        b,
        lstat
    ]])

    prediction = model.predict(input_data)

    st.success(f"Predicted Median House Price: ${prediction[0]:.2f}k")