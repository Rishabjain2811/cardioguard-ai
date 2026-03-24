import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="CardioGuard AI", layout="centered")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# ---------------- HEADER ----------------
st.title("CardioGuard AI")
st.subheader("Smart Heart Attack Risk Screening")

st.write(
"""
Enter your symptoms and lifestyle details to estimate your cardiac risk.  
This tool helps you decide whether medical consultation is needed.
"""
)

# ---------------- INPUTS ----------------

st.markdown("### Basic Information")

age = st.slider("Age", 18, 90, 35)

gender = st.selectbox("Gender", ["Male","Female"])
sex = 1 if gender == "Male" else 0

weight = st.slider("Weight (kg)", 40, 120, 70)
height = st.slider("Height (cm)", 140, 200, 170)

# ---------------- LIFESTYLE ----------------

st.markdown("### Lifestyle Factors")

smoking = st.selectbox("Do you smoke?", ["No","Yes"])
smoking = 1 if smoking == "Yes" else 0

bp = st.selectbox("Do you have high blood pressure?", ["No","Yes"])
bp = 1 if bp == "Yes" else 0

diabetes = st.selectbox("Do you have diabetes?", ["No","Yes"])
diabetes = 1 if diabetes == "Yes" else 0

family = st.selectbox("Family history of heart disease?", ["No","Yes"])
family = 1 if family == "Yes" else 0

activity = st.selectbox("Physical activity level", ["Low","Moderate","High"])

# ---------------- SYMPTOMS ----------------

st.markdown("### Current Symptoms")

chest_pain_level = st.slider("Chest Pain Intensity (0-10)", 0, 10, 2)

chest_pain_type = st.selectbox(
    "Chest Pain Type",
    ["No Pain", "Sharp Pain", "Pressure/Tightness", "Burning Sensation"]
)

breath = st.selectbox("Shortness of breath?", ["No","Yes"])
breath = 1 if breath == "Yes" else 0

fatigue = st.selectbox("Unusual fatigue?", ["No","Yes"])
fatigue = 1 if fatigue == "Yes" else 0

# ---------------- FEATURE MAPPING ----------------

bmi = weight / ((height/100)**2)

cp = 2 if chest_pain_level > 4 else 0

trestbps = 140 if bp else 120

chol = 250 if smoking else 200

fbs = 1 if diabetes else 0

restecg = 1

thalach = 200 - age

exang = breath

oldpeak = 2 if chest_pain_level > 5 else 1

slope = 1

ca = 2 if family else 0

thal = 2

data = np.array([[age, sex, cp, trestbps, chol, fbs,
                  restecg, thalach, exang, oldpeak,
                  slope, ca, thal]])

# ---------------- SCALE ----------------
data = scaler.transform(data)

# ---------------- PREDICT ----------------

if st.button("Analyze Risk"):

    prob = model.predict_proba(data)[0][1]
    percent = round(prob * 100, 2)

    st.markdown("## Risk Result")

    # ---------------- RISK LEVEL ----------------
    if percent < 30:
        st.success(f"Low Risk ({percent}%)")

        st.write("You currently show low signs of cardiac risk.")

        st.markdown("### Maintain Good Health")
        st.write("• Continue regular exercise")
        st.write("• Maintain healthy diet")
        st.write("• Avoid smoking and stress")

    elif percent < 70:
        st.warning(f"Moderate Risk ({percent}%)")

        st.write("Some risk factors detected.")

        st.markdown("### Recommended Actions")
        st.write("• Improve lifestyle habits")
        st.write("• Reduce smoking or alcohol")
        st.write("• Monitor symptoms regularly")
        st.write("• Consider medical checkup")

    else:
        st.error(f"High Risk ({percent}%)")

        st.write("High probability of cardiac risk detected.")

        st.markdown("### Immediate Attention Required")
        st.write("• Consult a doctor immediately")
        st.write("• Avoid physical exertion")
        st.write("• Monitor chest pain carefully")

    st.progress(percent/100)

    # ---------------- KEY RISK FACTORS ----------------
    st.markdown("### Key Risk Contributors")

    risks = []

    if chest_pain_level > 5:
        risks.append("Severe chest pain")

    if breath == 1:
        risks.append("Shortness of breath")

    if smoking == 1:
        risks.append("Smoking habit")

    if bp == 1:
        risks.append("High blood pressure")

    if diabetes == 1:
        risks.append("Diabetes")

    if family == 1:
        risks.append("Family history")

    if risks:
        for r in risks:
            st.write("•", r)
    else:
        st.write("No major risk factors detected")

    # ---------------- DISCLAIMER ----------------
    st.caption(
        "This is an AI-based screening tool and not a medical diagnosis. Consult a healthcare professional for accurate evaluation."
    )
