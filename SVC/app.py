import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Deployment-safe paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "svr_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

# Load original dataset for column alignment
df = pd.read_csv(os.path.join(BASE_DIR, "data", "laptop_prices.csv"))

st.set_page_config(page_title="Laptop Price Prediction", layout="centered")

st.title("Laptop Price Prediction using Support Vector Regression")

st.write("Enter laptop specifications to predict price.")

# User Inputs
company = st.selectbox("Company", df["Company"].unique())
product = st.selectbox("Product", df["Product"].unique())
typename = st.selectbox("Type", df["TypeName"].unique())
inches = st.number_input("Screen Size (Inches)", value=15.6)
ram = st.number_input("RAM (GB)", value=8)
os_name = st.selectbox("Operating System", df["OS"].unique())
weight = st.number_input("Weight (kg)", value=2.0)
screen = st.selectbox("Screen Type", df["Screen"].unique())
screenw = st.number_input("Screen Width", value=1920)
screenh = st.number_input("Screen Height", value=1080)
touchscreen = st.selectbox("Touchscreen", df["Touchscreen"].unique())
ipspanel = st.selectbox("IPS Panel", df["IPSpanel"].unique())
retina = st.selectbox("Retina Display", df["RetinaDisplay"].unique())
cpu_company = st.selectbox("CPU Company", df["CPU_company"].unique())
cpu_freq = st.number_input("CPU Frequency (GHz)", value=2.5)
cpu_model = st.selectbox("CPU Model", df["CPU_model"].unique())
primary_storage = st.number_input("Primary Storage (GB)", value=256)
secondary_storage = st.number_input("Secondary Storage (GB)", value=0)
primary_storage_type = st.selectbox("Primary Storage Type", df["PrimaryStorageType"].unique())
secondary_storage_type = st.selectbox("Secondary Storage Type", df["SecondaryStorageType"].unique())
gpu_company = st.selectbox("GPU Company", df["GPU_company"].unique())
gpu_model = st.selectbox("GPU Model", df["GPU_model"].unique())

if st.button("Predict Price"):

    input_dict = {
        "Company": company,
        "Product": product,
        "TypeName": typename,
        "Inches": inches,
        "Ram": ram,
        "OS": os_name,
        "Weight": weight,
        "Screen": screen,
        "ScreenW": screenw,
        "ScreenH": screenh,
        "Touchscreen": touchscreen,
        "IPSpanel": ipspanel,
        "RetinaDisplay": retina,
        "CPU_company": cpu_company,
        "CPU_freq": cpu_freq,
        "CPU_model": cpu_model,
        "PrimaryStorage": primary_storage,
        "SecondaryStorage": secondary_storage,
        "PrimaryStorageType": primary_storage_type,
        "SecondaryStorageType": secondary_storage_type,
        "GPU_company": gpu_company,
        "GPU_model": gpu_model
    }

    input_df = pd.DataFrame([input_dict])

    X = df.drop("Price_euros", axis=1)
    X_encoded = pd.get_dummies(X, drop_first=True)

    input_encoded = pd.get_dummies(input_df, drop_first=True)

    input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    input_scaled = scaler.transform(input_encoded)

    prediction = model.predict(input_scaled)

    st.success(f"Predicted Laptop Price: € {prediction[0]:.2f}")