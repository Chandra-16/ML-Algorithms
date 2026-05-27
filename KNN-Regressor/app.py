import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Deployment-safe paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "knn_regressor.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

# Load original dataset
df = pd.read_csv(os.path.join(BASE_DIR, "data", "student.csv"))

# Fill missing values exactly like training
df["Teacher_Quality"] = df["Teacher_Quality"].fillna(df["Teacher_Quality"].mode()[0])
df["Parental_Education_Level"] = df["Parental_Education_Level"].fillna(
    df["Parental_Education_Level"].mode()[0]
)
df["Distance_from_Home"] = df["Distance_from_Home"].fillna(
    df["Distance_from_Home"].mode()[0]
)

st.set_page_config(page_title="Student Exam Score Prediction", layout="centered")

st.title("Student Exam Score Prediction using KNN Regressor")
st.write("Enter student details to predict exam score.")

# Inputs
hours_studied = st.number_input("Hours Studied", value=20)
attendance = st.number_input("Attendance (%)", value=80)
parental_involvement = st.selectbox("Parental Involvement", df["Parental_Involvement"].unique())
access_to_resources = st.selectbox("Access to Resources", df["Access_to_Resources"].unique())
extracurricular = st.selectbox("Extracurricular Activities", df["Extracurricular_Activities"].unique())
sleep_hours = st.number_input("Sleep Hours", value=7)
previous_scores = st.number_input("Previous Scores", value=70)
motivation = st.selectbox("Motivation Level", df["Motivation_Level"].unique())
internet = st.selectbox("Internet Access", df["Internet_Access"].unique())
tutoring = st.number_input("Tutoring Sessions", value=2)
family_income = st.selectbox("Family Income", df["Family_Income"].unique())
teacher_quality = st.selectbox("Teacher Quality", df["Teacher_Quality"].unique())
school_type = st.selectbox("School Type", df["School_Type"].unique())
peer_influence = st.selectbox("Peer Influence", df["Peer_Influence"].unique())
physical_activity = st.number_input("Physical Activity", value=3)
learning_disabilities = st.selectbox("Learning Disabilities", df["Learning_Disabilities"].unique())
parental_education = st.selectbox("Parental Education Level", df["Parental_Education_Level"].unique())
distance = st.selectbox("Distance from Home", df["Distance_from_Home"].unique())
gender = st.selectbox("Gender", df["Gender"].unique())

if st.button("Predict Exam Score"):

    input_dict = {
        "Hours_Studied": hours_studied,
        "Attendance": attendance,
        "Parental_Involvement": parental_involvement,
        "Access_to_Resources": access_to_resources,
        "Extracurricular_Activities": extracurricular,
        "Sleep_Hours": sleep_hours,
        "Previous_Scores": previous_scores,
        "Motivation_Level": motivation,
        "Internet_Access": internet,
        "Tutoring_Sessions": tutoring,
        "Family_Income": family_income,
        "Teacher_Quality": teacher_quality,
        "School_Type": school_type,
        "Peer_Influence": peer_influence,
        "Physical_Activity": physical_activity,
        "Learning_Disabilities": learning_disabilities,
        "Parental_Education_Level": parental_education,
        "Distance_from_Home": distance,
        "Gender": gender
    }

    input_df = pd.DataFrame([input_dict])

    X = df.drop("Exam_Score", axis=1)
    X_encoded = pd.get_dummies(X, drop_first=True)

    input_encoded = pd.get_dummies(input_df, drop_first=True)

    input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    input_scaled = scaler.transform(input_encoded)

    prediction = model.predict(input_scaled)

    st.success(f"Predicted Exam Score: {prediction[0]:.2f}")