import streamlit as st
import pickle
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(page_title="CardioGuard AI", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.title {
    text-align:center;
    color:#d4a373;
}
.subtitle {
    text-align:center;
    color:#ccc;
}
.block {
    background-color:#161b22;
    padding:20px;
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- HEADER ----------------
st.markdown("<h1 class='title'>CardioGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Smart Heart Attack Risk Screening</p>", unsafe_allow_html=True)

st.write("")

# ---------------- INPUT ----------------
st.markdown("## Basic Information")
c1, c2, c3 = st.columns(3)

with c1:
    age = st.slider("Age", 18, 80, 30)
    sex = st.selectbox("Gender", ["Male", "Female"])
    sex = 1 if sex == "Male" else 0

with c2:
    weight = st.slider("Weight (kg)", 40, 120, 70)
    height = st.slider("Height (cm)", 140, 200, 170)

with c3:
    bmi = weight / ((height/100) ** 2)

st.markdown("## Lifestyle Factors")
c1, c2 = st.columns(2)

with c1:
    smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
    high_bp = st.selectbox("Do you have high blood pressure?", ["No", "Yes"])
    diabetes = st.selectbox("Do you have diabetes?", ["No", "Yes"])

with c2:
    family_history = st.selectbox("Family history of heart disease?", ["No", "Yes"])
    activity = st.selectbox("Physical activity level", ["Low", "Moderate", "High"])

st.markdown("## Current Symptoms")
c1, c2 = st.columns(2)

with c1:
    chest_pain_intensity = st.slider("Chest Pain Intensity (0-10)", 0, 10, 2)
    chest_pain_type = st.selectbox("Chest Pain Type", ["None", "Mild", "Sharp Pain"])

with c2:
    breath_shortness = st.selectbox("Shortness of breath?", ["No", "Yes"])
    fatigue = st.selectbox("Unusual fatigue?", ["No", "Yes"])

st.write("")

# ---------------- PREDICT ----------------
if st.button("Analyze Risk", use_container_width=True):

    # ---------------- MAP TO MODEL FEATURES ----------------
    cp = 2 if chest_pain_type == "Sharp Pain" else 1 if chest_pain_type == "Mild" else 0
    trestbps = 150 if high_bp == "Yes" else 120
    chol = 260 if bmi > 27 else 200
    fbs = 1 if diabetes == "Yes" else 0
    restecg = 1
    thalach = 120 if activity == "Low" else 150
    exang = 1 if breath_shortness == "Yes" else 0
    oldpeak = chest_pain_intensity / 5
    slope = 1
    ca = 2 if smoking == "Yes" else 0
    thal = 2

    data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

    data = scaler.transform(data)

    prob = model.predict_proba(data)[0][1]
    percent = prob * 100

    # ---------------- HYBRID SYMPTOM SCORING ----------------
    symptom_score = 0

    if chest_pain_intensity > 7:
        symptom_score += 25
    if chest_pain_type == "Sharp Pain":
        symptom_score += 15
    if breath_shortness == "Yes":
        symptom_score += 15
    if fatigue == "Yes":
        symptom_score += 10
    if smoking == "Yes":
        symptom_score += 10
    if high_bp == "Yes":
        symptom_score += 10
    if diabetes == "Yes":
        symptom_score += 10
    if family_history == "Yes":
        symptom_score += 10

    final_percent = min(100, percent + symptom_score)

    # ---------------- OUTPUT ----------------
    st.markdown("## Risk Result")

    if final_percent < 30:
        st.success(f"Low Risk ({round(final_percent,1)}%)")
    elif final_percent < 70:
        st.warning(f"Moderate Risk ({round(final_percent,1)}%)")
    else:
        st.error(f"High Risk ({round(final_percent,1)}%)")

    st.progress(final_percent / 100)

    # ---------------- RECOMMENDATIONS ----------------
    st.markdown("### Recommended Actions")

    actions = []

    if smoking == "Yes":
        actions.append("Stop smoking immediately")
    if high_bp == "Yes":
        actions.append("Monitor and control blood pressure")
    if diabetes == "Yes":
        actions.append("Manage blood sugar levels")
    if activity == "Low":
        actions.append("Increase physical activity")
    if chest_pain_intensity > 6:
        actions.append("Seek medical consultation immediately")

    if not actions:
        actions.append("Maintain a healthy lifestyle")

    for a in actions:
        st.write("•", a)

    # ---------------- KEY CONTRIBUTORS ----------------
    st.markdown("### Key Risk Contributors")

    contributors = []

    if chest_pain_intensity > 7:
        contributors.append("Severe chest pain")
    if breath_shortness == "Yes":
        contributors.append("Shortness of breath")
    if smoking == "Yes":
        contributors.append("Smoking habit")
    if high_bp == "Yes":
        contributors.append("High blood pressure")
    if diabetes == "Yes":
        contributors.append("Diabetes")
    if family_history == "Yes":
        contributors.append("Family history")

    for c in contributors:
        st.write("•", c)

    st.caption("AI-based screening tool. Not a medical diagnosis.")
