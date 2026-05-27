import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Deployment-safe path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "dt_regressor.pkl"))

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("House Price Prediction using Decision Tree Regressor")
st.write("Enter house details to predict median house price.")

# Inputs
crim = st.number_input("Crime Rate (crim)", value=0.1)
zn = st.number_input("Residential Land Zoned (%) (zn)", value=0.0)
indus = st.number_input("Industrial Proportion (indus)", value=8.0)
chas = st.selectbox("Charles River Bound (chas)", [0, 1])
nox = st.number_input("Nitric Oxide Concentration (nox)", value=0.5)
rm = st.number_input("Average Rooms (rm)", value=6.0)
age = st.number_input("Age of Property (%)", value=50.0)
dis = st.number_input("Distance to Employment Centers (dis)", value=4.0)
rad = st.number_input("Accessibility to Highways (rad)", value=5)
tax = st.number_input("Property Tax Rate (tax)", value=300)
ptratio = st.number_input("Pupil-Teacher Ratio (ptratio)", value=18.0)
b = st.number_input("Black Population Metric (b)", value=390.0)
lstat = st.number_input("Lower Status Population % (lstat)", value=10.0)

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