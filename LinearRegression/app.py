import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("LinearRegression/models/laptop_price_model.pkl")

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")

st.title("💻 Laptop Price Predictor")
st.write("Choose key laptop specifications")

# User inputs
company = st.selectbox("Brand", [
    'Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo',
    'MSI', 'Microsoft', 'Samsung', 'LG', 'Huawei', 'Xiaomi'
])

typename = st.selectbox("Laptop Type", [
    'Ultrabook', 'Notebook', 'Gaming',
    '2 in 1 Convertible', 'Workstation'
])

ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64])

os = st.selectbox("Operating System", [
    'Windows 10', 'macOS', 'Linux', 'No OS', 'Chrome OS'
])

weight = st.number_input("Weight (kg)", 0.5, 5.0, 2.0)

screenw = st.selectbox("Screen Resolution Width", [
    1366, 1920, 2560, 2880, 3840
])

screenh = st.selectbox("Screen Resolution Height", [
    768, 1080, 1440, 1800, 2160
])

touchscreen = st.selectbox("Touchscreen", [0, 1])

cpu_company = st.selectbox("CPU Brand", [
    'Intel', 'AMD', 'Samsung'
])

cpu_freq = st.number_input("CPU Frequency (GHz)", 1.0, 5.0, 2.5)

primary_storage = st.selectbox("Storage (GB)", [
    128, 256, 512, 1024, 2048
])

gpu_company = st.selectbox("GPU Brand", [
    'Intel', 'AMD', 'Nvidia', 'ARM'
])

# Predict
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "Company": [company],
        "TypeName": [typename],
        "Inches": [15.6],
        "Ram": [ram],
        "OS": [os],
        "Weight": [weight],
        "Screen": ["Full HD"],
        "ScreenW": [screenw],
        "ScreenH": [screenh],
        "Touchscreen": [touchscreen],
        "IPSpanel": [1],
        "RetinaDisplay": [0],
        "CPU_company": [cpu_company],
        "CPU_freq": [cpu_freq],
        "PrimaryStorage": [primary_storage],
        "SecondaryStorage": [0],
        "PrimaryStorageType": ["SSD"],
        "SecondaryStorageType": ["No"],
        "GPU_company": [gpu_company]
    })

    prediction_log = model.predict(input_data)[0]
    prediction = np.exp(prediction_log)

    st.success(f"Estimated Laptop Price: €{prediction:.2f}")