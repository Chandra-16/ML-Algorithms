import streamlit as st
import numpy as np
import joblib
import os

# Deployment-safe paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "knn_classifier.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

st.set_page_config(page_title="Wine Classification", layout="centered")

st.title("Wine Classification using KNN Classifier")
st.write("Enter wine chemical properties to classify the wine type.")

# User Inputs
alcohol = st.number_input("Alcohol", value=13.0)
malic_acid = st.number_input("Malic Acid", value=2.0)
ash = st.number_input("Ash", value=2.3)
alcalinity_of_ash = st.number_input("Alcalinity of Ash", value=19.0)
magnesium = st.number_input("Magnesium", value=100.0)
total_phenols = st.number_input("Total Phenols", value=2.5)
flavanoids = st.number_input("Flavanoids", value=2.0)
nonflavanoid_phenols = st.number_input("Nonflavanoid Phenols", value=0.3)
proanthocyanins = st.number_input("Proanthocyanins", value=1.5)
color_intensity = st.number_input("Color Intensity", value=5.0)
hue = st.number_input("Hue", value=1.0)
od280_od315 = st.number_input("OD280/OD315", value=3.0)
proline = st.number_input("Proline", value=750.0)

if st.button("Classify Wine"):

    input_data = np.array([[
        alcohol,
        malic_acid,
        ash,
        alcalinity_of_ash,
        magnesium,
        total_phenols,
        flavanoids,
        nonflavanoid_phenols,
        proanthocyanins,
        color_intensity,
        hue,
        od280_od315,
        proline
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)

    predicted_class = prediction[0]
    confidence = np.max(probabilities) * 100

    st.success(
        f"Predicted Wine Class: {predicted_class}\nConfidence: {confidence:.2f}%"
    )