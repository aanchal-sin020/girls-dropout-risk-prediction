import streamlit as st
import numpy as np
import joblib

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fb;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model and scaler
model = joblib.load("dropout_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Girls Dropout Risk AI", layout="centered")

st.title("AI System for Predicting Girls' Dropout Risk")

st.write("Enter student information to estimate dropout risk.")

age = st.slider("Age", 10, 18, 14)
attendance = st.slider("Attendance (%)",0,100,75)
household_income = st.number_input("Household Income",10000,500000,120000)
distance_to_school_km = st.slider("Distance to School (km)",0,20,5)
parents_education_level = st.slider("Parents Education Level",0,4,2)

rural_area = st.selectbox("Rural Area",[0,1])
receives_scholarship = st.selectbox("Receives Scholarship",[0,1])
has_toilet_at_school = st.selectbox("School has Toilet",[0,1])
has_menstrual_hygiene_access = st.selectbox("Menstrual Hygiene Access",[0,1])
helps_in_household_work = st.selectbox("Helps in Household Work",[0,1])
works_after_school = st.selectbox("Works After School",[0,1])
early_marriage_risk = st.selectbox("Early Marriage Risk",[0,1])

features = np.array([[

age,
attendance,
household_income,
parents_education_level,
rural_area,
distance_to_school_km,
has_toilet_at_school,
receives_scholarship,
has_menstrual_hygiene_access,
helps_in_household_work,
works_after_school,
early_marriage_risk

]])

features_scaled = scaler.transform(features)

if st.button("Predict Dropout Risk"):

    probability = model.predict_proba(features_scaled)[0][1]

    if probability > 0.7:
        category = "High Risk"
    elif probability > 0.4:
        category = "Medium Risk"
    else:
        category = "Low Risk"

    st.subheader("Prediction Result")

    # Progress bar
    st.progress(float(probability))

    st.write(f"Dropout Probability: {probability:.2f}")

    # Color coded warning
    if probability > 0.7:
        st.error("High Risk of School Dropout")

    elif probability > 0.4:
        st.warning("Moderate Risk of School Dropout")

    else:
        st.success("Low Risk of School Dropout")

    st.subheader("Suggested Intervention")

    if early_marriage_risk == 1:
        st.write("Community counseling and awareness programs to prevent early marriage.")

    elif household_income < 100000:
        st.write("Financial support or scholarship programs recommended.")

    elif distance_to_school_km > 10:
        st.write("Transport assistance or hostel facility recommended.")

    elif has_toilet_at_school == 0:
        st.write("Improve sanitation infrastructure at school.")

    elif works_after_school == 1:
        st.write("Provide economic assistance to reduce child labor pressure.")

    else:

        st.write("Regular monitoring recommended.")
