import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="CardioGuard AI", layout="wide")

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}

.block-container {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 600;
    color: #f8fafc;
    letter-spacing: 1px;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #94a3b8;
    margin-bottom: 40px;
}

.card {
    background: #1e293b;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    transition: 0.3s ease-in-out;
}

.card:hover {
    transform: scale(1.01);
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg,#d4af37,#b08930);
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 16px;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(90deg,#b08930,#d4af37);
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    color: #f1f5f9;
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

# ---------------- HEADER ----------------
st.markdown("<div class='title'>CardioGuard AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Heart Attack Risk Prediction & Prevention Advisory System</div>", unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------
st.markdown("<div class='section-title'>Patient Clinical Parameters</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 20, 100, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    sex = 1 if sex == "Male" else 0
    resting_bp = st.slider("Resting Blood Pressure", 80, 200, 120)

with col2:
    cholesterol = st.slider("Serum Cholesterol", 100, 600, 220)
    chest_pain = st.slider("Chest Pain Type", 0, 3, 1)
    max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)

with col3:
    exercise_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    fasting_sugar = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
    fasting_sugar = 1 if fasting_sugar == "Yes" else 0
    st_depression = st.slider("ST Depression", 0.0, 6.0, 1.0)
    major_vessels = st.slider("Major Vessels (0-4)", 0, 4, 0)
    thalassemia = st.slider("Thalassemia Status", 0, 3, 2)
    slope = st.slider("Slope of ST Segment", 0, 2, 1)
    rest_ecg = st.slider("Rest ECG Result", 0, 2, 1)

st.write("")

# ---------------- PREDICTION ----------------
if st.button("Analyze Cardiac Risk"):

    data = np.array([[age, sex, chest_pain, resting_bp, cholesterol,
                      fasting_sugar, rest_ecg, max_hr, exercise_angina,
                      st_depression, slope, major_vessels, thalassemia]])

    probability = model.predict_proba(data)[0][1]
    risk_percent = round(probability * 100, 2)

    st.markdown("---")
    st.markdown("<div class='section-title'>Clinical Risk Summary</div>", unsafe_allow_html=True)

    # Risk Level
    if risk_percent < 30:
        st.success(f"Low Clinical Risk — {risk_percent}%")
    elif risk_percent < 70:
        st.warning(f"Moderate Clinical Risk — {risk_percent}%")
    else:
        st.error(f"High Clinical Risk — {risk_percent}%")

    st.progress(risk_percent / 100)

    st.info("Prediction is based on combined interaction of multiple clinical indicators.")

    # Show only key abnormal factors
    st.markdown("<div class='section-title'>Key Risk Indicators</div>", unsafe_allow_html=True)

    risk_factors = []

    if resting_bp > 140:
        risk_factors.append("Elevated Resting Blood Pressure")
    if cholesterol > 240:
        risk_factors.append("High Serum Cholesterol")
    if chest_pain > 1:
        risk_factors.append("High-Risk Chest Pain Pattern")
    if major_vessels > 0:
        risk_factors.append("Major Vessel Involvement")
    if st_depression > 1:
        risk_factors.append("Significant ST Depression")
    if exercise_angina == 1:
        risk_factors.append("Exercise-Induced Angina Present")

    if risk_factors:
        for factor in risk_factors:
            st.write("•", factor)
    else:
        st.write("No dominant abnormal indicators detected.")

    # Recommendation
    st.markdown("<div class='section-title'>Recommended Action</div>", unsafe_allow_html=True)

    if risk_percent > 70:
        st.error("Immediate cardiology consultation strongly recommended.")
    elif risk_percent > 40:
        st.warning("Medical evaluation advised in near term.")
    else:
        st.success("Maintain preventive lifestyle and routine monitoring.")

    st.caption("Academic AI-based decision support system. Not a substitute for professional medical diagnosis.")
