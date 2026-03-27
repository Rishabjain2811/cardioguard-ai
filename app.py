import streamlit as st
import pickle
import numpy as np
import time

# ---------------- CONFIG ----------------
st.set_page_config(page_title="CardioGuard AI", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
body { background-color: #0f1117; }

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 600;
    color: #f5f5f5;
}

.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 30px;
}

.card {
    background: #161a23;
    padding: 25px;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
}

.section-title {
    color: #d4af37;
    font-weight: 600;
    margin-bottom: 10px;
    font-size: 20px;
}

.stButton > button {
    background: linear-gradient(90deg, #d4af37, #b8962e);
    color: black;
    font-weight: 600;
    border-radius: 10px;
    height: 50px;
    border: none;
}

.stProgress > div > div > div {
    background-color: #d4af37;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# ---------------- HEADER ----------------
st.markdown("<div class='title'>CardioGuard AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI Powered Heart Attack Risk Screening</div>", unsafe_allow_html=True)

# ---------------- INPUT ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Basic Information</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    age = st.slider("Age", 18, 80, 30)
    sex = st.selectbox("Gender", ["Male", "Female"])
    sex = 1 if sex=="Male" else 0

with c2:
    weight = st.slider("Weight (kg)", 40, 120, 70)
    height = st.slider("Height (cm)", 140, 200, 170)

with c3:
    bmi = weight / ((height/100)**2)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- LIFESTYLE ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Lifestyle Factors</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    smoking = st.selectbox("Smoking", ["No","Yes"])
    high_bp = st.selectbox("High Blood Pressure", ["No","Yes"])
    diabetes = st.selectbox("Diabetes", ["No","Yes"])

with c2:
    family_history = st.selectbox("Family History", ["No","Yes"])
    activity = st.selectbox("Activity Level", ["Low","Moderate","High"])

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SYMPTOMS ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Current Symptoms</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    chest_pain_intensity = st.slider("Chest Pain Intensity", 0, 10, 2)
    chest_pain_type = st.selectbox("Chest Pain Type", ["None","Mild","Sharp Pain"])

with c2:
    breath_shortness = st.selectbox("Shortness of Breath", ["No","Yes"])
    fatigue = st.selectbox("Fatigue", ["No","Yes"])

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- BUTTON ----------------
if st.button("Analyze Risk", use_container_width=True):

    # 🔄 Loading animation
    with st.spinner("Analyzing patient data..."):
        time.sleep(1.5)

    # ---------------- MODEL INPUT ----------------
    cp = 2 if chest_pain_type=="Sharp Pain" else 1 if chest_pain_type=="Mild" else 0
    trestbps = 150 if high_bp=="Yes" else 120
    chol = 260 if bmi>27 else 200
    fbs = 1 if diabetes=="Yes" else 0
    restecg = 1
    thalach = 120 if activity=="Low" else 150
    exang = 1 if breath_shortness=="Yes" else 0
    oldpeak = chest_pain_intensity/5
    slope = 1
    ca = 2 if smoking=="Yes" else 0
    thal = 2

    data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                      thalach,exang,oldpeak,slope,ca,thal]])

    data = scaler.transform(data)
    prob = model.predict_proba(data)[0][1]
    percent = prob * 100

    # ---------------- HYBRID BOOST ----------------
    boost = 0
    if chest_pain_intensity > 7: boost += 25
    if chest_pain_type == "Sharp Pain": boost += 15
    if breath_shortness == "Yes": boost += 15
    if fatigue == "Yes": boost += 10
    if smoking == "Yes": boost += 10
    if high_bp == "Yes": boost += 10
    if diabetes == "Yes": boost += 10
    if family_history == "Yes": boost += 10

    final_percent = min(100, percent + boost)

    # ---------------- AI REPORT ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Clinical Report</div>', unsafe_allow_html=True)

    # Typing effect
    text = f"Predicted Heart Risk: {round(final_percent,1)}%"
    display = st.empty()
    for i in range(len(text)+1):
        display.markdown(f"### {text[:i]}")
        time.sleep(0.02)

    # Risk level
    if final_percent < 30:
        st.success("LOW RISK")
    elif final_percent < 70:
        st.warning("MODERATE RISK")
    else:
        st.error("HIGH RISK")

    st.progress(final_percent/100)

    # ---------------- CONTRIBUTORS ----------------
    st.markdown("### Key Risk Contributors")

    contributors = []
    if chest_pain_intensity > 7: contributors.append("Severe Chest Pain")
    if breath_shortness == "Yes": contributors.append("Breathlessness")
    if smoking == "Yes": contributors.append("Smoking")
    if high_bp == "Yes": contributors.append("High BP")
    if diabetes == "Yes": contributors.append("Diabetes")

    if contributors:
        for c in contributors:
            st.write("•", c)
    else:
        st.write("No major risk contributors detected")

    # ---------------- RECOMMENDATIONS ----------------
    st.markdown("### Recommended Actions")

    actions = []
    if smoking=="Yes": actions.append("Stop smoking immediately")
    if high_bp=="Yes": actions.append("Control blood pressure")
    if diabetes=="Yes": actions.append("Manage sugar levels")
    if activity=="Low": actions.append("Increase physical activity")
    if chest_pain_intensity>6: actions.append("Consult a doctor urgently")

    if not actions:
        actions.append("Maintain healthy lifestyle")

    for a in actions:
        st.write("•", a)

    st.caption("AI-based screening tool. Not a medical diagnosis.")
    st.markdown('</div>', unsafe_allow_html=True)
