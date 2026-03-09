import streamlit as st
import pickle
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(page_title="CardioGuard AI", layout="wide")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl","rb"))

# ---------------- TITLE ----------------
st.title("CardioGuard AI")
st.subheader("Heart Attack Risk Assessment Tool")

st.write("Enter basic lifestyle and health details to estimate your cardiac risk.")

# ---------------- USER INPUTS ----------------

col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age", 18, 90, 35)

    gender = st.selectbox("Gender", ["Male", "Female"])
    sex = 1 if gender == "Male" else 0

    weight = st.slider("Weight (kg)", 40, 120, 70)

    height = st.slider("Height (cm)", 140, 200, 170)

    smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
    smoking = 1 if smoking == "Yes" else 0

    diabetes = st.selectbox("Do you have diabetes?", ["No", "Yes"])
    diabetes = 1 if diabetes == "Yes" else 0


with col2:

    bp_history = st.selectbox("History of high blood pressure?", ["No", "Yes"])
    bp_history = 1 if bp_history == "Yes" else 0

    family_history = st.selectbox("Family history of heart disease?", ["No", "Yes"])
    family_history = 1 if family_history == "Yes" else 0

    chest_pain = st.selectbox("Do you experience chest pain during physical activity?", ["No", "Yes"])
    chest_pain = 1 if chest_pain == "Yes" else 0

    exercise = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])

    heart_rate = st.slider("Resting Heart Rate", 50, 120, 75)

# ---------------- FEATURE MAPPING ----------------

# Convert BMI
bmi = weight / ((height/100)**2)

# Approximate mappings for model features

cp = 2 if chest_pain else 0
trestbps = 140 if bp_history else 120
chol = 260 if smoking else 200
fbs = 1 if diabetes else 0
restecg = 1
thalach = 200 - age
exang = chest_pain
oldpeak = 2 if chest_pain else 1
slope = 1
ca = 2 if family_history else 0
thal = 2

# Arrange input for model
data = np.array([[age, sex, cp, trestbps, chol, fbs,
                  restecg, thalach, exang, oldpeak,
                  slope, ca, thal]])

# ---------------- PREDICTION ----------------

if st.button("Analyze Heart Risk"):

    prob = model.predict_proba(data)[0][1]
    percent = round(prob * 100, 2)

    st.subheader("Prediction Result")

    if percent < 30:
        st.success(f"Low Risk ({percent}%)")

    elif percent < 70:
        st.warning(f"Moderate Risk ({percent}%)")

    else:
        st.error(f"High Risk ({percent}%)")

    st.progress(percent/100)

    st.write("⚠️ This system is an educational prototype and not a medical diagnosis tool.")
