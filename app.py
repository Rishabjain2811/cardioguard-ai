import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="CardioGuard AI", layout="wide")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl","rb"))

# ---------------- HEADER ----------------
st.title("CardioGuard AI")
st.subheader("Heart Attack Risk Screening Tool")

st.write(
"""
This tool estimates cardiac risk based on symptoms and lifestyle factors.  
It helps users decide whether medical consultation may be necessary.
"""
)

# ---------------- INPUT SECTION ----------------

col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age", 18, 90, 35)

    gender = st.selectbox("Gender", ["Male", "Female"])
    sex = 1 if gender == "Male" else 0

    weight = st.slider("Weight (kg)", 40, 120, 70)

    height = st.slider("Height (cm)", 140, 200, 170)

    smoking = st.selectbox("Smoking Habit", ["No", "Yes"])
    smoking = 1 if smoking == "Yes" else 0

    alcohol = st.selectbox("Alcohol Consumption", ["No", "Occasionally", "Regularly"])


with col2:

    bp_history = st.selectbox("History of High Blood Pressure", ["No", "Yes"])
    bp_history = 1 if bp_history == "Yes" else 0

    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    diabetes = 1 if diabetes == "Yes" else 0

    family_history = st.selectbox("Family History of Heart Disease", ["No", "Yes"])
    family_history = 1 if family_history == "Yes" else 0

    exercise = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])


# ---------------- SYMPTOMS ----------------

st.markdown("### Current Symptoms")

col3, col4 = st.columns(2)

with col3:

    chest_pain_level = st.slider(
        "Chest Pain Intensity",
        0,10,2,
        help="0 = no pain, 10 = severe pain"
    )

    chest_pain_type = st.selectbox(
        "Type of Chest Pain",
        [
            "No Pain",
            "Sharp / Sudden Pain",
            "Pressure or Tightness",
            "Burning / Indigestion-like Pain"
        ]
    )

with col4:

    pain_duration = st.selectbox(
        "Pain Duration",
        [
            "No pain",
            "Few seconds",
            "Few minutes",
            "More than 10 minutes"
        ]
    )

    breath = st.selectbox(
        "Shortness of Breath During Activity",
        ["No", "Yes"]
    )

    fatigue = st.selectbox(
        "Unusual Fatigue During Activity",
        ["No", "Yes"]
    )

# ---------------- FEATURE MAPPING ----------------

bmi = weight / ((height/100)**2)

cp = 2 if chest_pain_level > 4 else 0

trestbps = 140 if bp_history else 120

chol = 260 if smoking else 200

fbs = 1 if diabetes else 0

restecg = 1

thalach = 200 - age

exang = 1 if breath == "Yes" else 0

oldpeak = 2 if chest_pain_level > 5 else 1

slope = 1

ca = 2 if family_history else 0

thal = 2

data = np.array([[age, sex, cp, trestbps, chol, fbs,
                  restecg, thalach, exang, oldpeak,
                  slope, ca, thal]])

# ---------------- PREDICTION ----------------

if st.button("Analyze Cardiac Risk"):

    prob = model.predict_proba(data)[0][1]

    percent = round(prob * 100,2)

    st.markdown("## Risk Assessment Result")

    if percent < 30:

        st.success(f"Low Risk ({percent}%)")

        st.write(
        """
        Your current symptoms indicate low cardiac risk.

        Preventive tips:
        • Maintain regular exercise  
        • Follow a balanced diet  
        • Avoid smoking  
        • Monitor stress levels
        """
        )

    elif percent < 70:

        st.warning(f"Moderate Risk ({percent}%)")

        st.write(
        """
        Moderate cardiac risk detected.

        Recommended actions:
        • Monitor symptoms closely  
        • Reduce smoking or alcohol  
        • Improve physical activity  
        • Consider consulting a physician if symptoms persist
        """
        )

    else:

        st.error(f"High Risk ({percent}%)")

        st.write(
        """
        High cardiac risk detected.

        Recommended actions:
        • Seek medical consultation immediately  
        • Avoid strenuous activity  
        • Monitor symptoms carefully  
        • Consider emergency care if chest pain persists
        """
        )

    st.progress(percent/100)

    st.caption(
    "This system provides an educational risk estimate and does not replace professional medical diagnosis."
    )
