import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "models", "adaboost_classifier.pkl")
)

st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

st.title("Titanic Survival Prediction using AdaBoost Classifier")
st.write("Enter passenger details to predict survival.")

# Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])

sex = st.selectbox("Sex", ["male", "female"])

age = st.number_input("Age", min_value=0, max_value=100, value=30)

sibsp = st.number_input("Number of Siblings/Spouses Aboard", value=0)

parch = st.number_input("Number of Parents/Children Aboard", value=0)

fare = st.number_input("Fare", value=30.0)

embarked = st.selectbox("Embarked", ["S", "C", "Q"])

if st.button("Predict Survival"):

    input_dict = {
        "Pclass": pclass,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Sex_male": 1 if sex == "male" else 0,
        "Embarked_Q": 1 if embarked == "Q" else 0,
        "Embarked_S": 1 if embarked == "S" else 0
    }

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    confidence = np.max(probability) * 100

    if prediction[0] == 1:
        st.success(f"Passenger likely survived.\nConfidence: {confidence:.2f}%")
    else:
        st.error(f"Passenger likely did not survive.\nConfidence: {confidence:.2f}%")