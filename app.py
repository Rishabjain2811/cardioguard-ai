import streamlit as st
import pickle
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(page_title="CardioGuard AI", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {
    background-color: #faf7f2;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
}
.title {
    text-align:center;
    color:#5c4033;
}
.subtitle {
    text-align:center;
    color:#8b6f47;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
model = pickle.load(open("model.pkl","rb"))

# ---------------- HEADER ----------------
st.markdown("<h1 class='title'>CardioGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Heart Attack Risk Prediction & Prevention Advisor</p>", unsafe_allow_html=True)

st.write("")

# ---------------- INPUT ----------------
st.markdown("### Patient Clinical Parameters")

c1,c2,c3 = st.columns(3)

with c1:
    age = st.slider("Age",20,100,45)
    sex = st.selectbox("Sex",["Male","Female"])
    sex = 1 if sex=="Male" else 0
    trestbps = st.slider("Resting Blood Pressure",80,200,120)
    chol = st.slider("Cholesterol",100,600,220)

with c2:
    cp = st.slider("Chest Pain Type",0,3,1)
    thalach = st.slider("Max Heart Rate",60,220,150)
    exang = st.selectbox("Exercise Induced Angina",["No","Yes"])
    exang = 1 if exang=="Yes" else 0
    oldpeak = st.slider("ST Depression (Oldpeak)",0.0,6.0,1.0)

with c3:
    fbs = st.selectbox("Fasting Sugar >120",["No","Yes"])
    fbs = 1 if fbs=="Yes" else 0
    ca = st.slider("Major Vessels",0,4,0)
    slope = st.slider("Slope",0,2,1)
    thal = st.slider("Thal",0,3,2)
    restecg = st.slider("Rest ECG",0,2,1)

st.write("")

# ---------------- PREDICT ----------------
if st.button("Analyze Heart Risk", use_container_width=True):

    data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                      thalach,exang,oldpeak,slope,ca,thal]])

    prob = model.predict_proba(data)[0][1]
    percent = round(prob*100,2)

    st.markdown("## Clinical Risk Report")

    # ---------------- RISK LEVEL ----------------
    if percent < 30:
        level = "LOW"
        color_box = st.success
    elif percent < 70:
        level = "MODERATE"
        color_box = st.warning
    else:
        level = "HIGH"
        color_box = st.error

    color_box(f"Overall Cardiac Risk Level: {level}")
    st.metric("Predicted Probability", f"{percent}%")
    st.progress(percent/100)

    # ---------------- PARAMETER ANALYSIS ----------------
    st.markdown("### Parameter Status Analysis")

    status = []

    def check(name, val, low, high):
        if val < low:
            status.append((name, "Below Normal"))
        elif val > high:
            status.append((name, "Above Normal"))
        else:
            status.append((name, "Normal"))

    check("Blood Pressure", trestbps, 90, 140)
    check("Cholesterol", chol, 150, 240)
    check("Max Heart Rate", thalach, 120, 200)

    for s in status:
        st.write(f"• **{s[0]}** → {s[1]}")

    # ---------------- RISK DRIVERS ----------------
    st.markdown("### Primary Risk Drivers")

    drivers = []
    if chol > 240: drivers.append("Elevated cholesterol level")
    if trestbps > 140: drivers.append("High resting blood pressure")
    if fbs == 1: drivers.append("Impaired fasting glucose")
    if exang == 1: drivers.append("Exercise-induced angina")
    if oldpeak > 2: drivers.append("ST depression abnormality")

    if drivers:
        for d in drivers:
            st.write("🔴", d)
    else:
        st.write("No dominant high-risk indicators detected")

    # ---------------- LIFESTYLE SCORE ----------------
    score = 100
    if chol > 240: score -= 15
    if trestbps > 140: score -= 15
    if fbs == 1: score -= 10
    if thalach < 120: score -= 10
    if exang == 1: score -= 15

    st.markdown("### Lifestyle Risk Score")
    st.metric("Estimated Heart Health Score", f"{score}/100")

    # ---------------- ACTION PLAN ----------------
    st.markdown("### Recommended Preventive Actions")

    actions = []

    if chol > 240:
        actions.append("Shift to low-fat & high-fiber diet")
    if trestbps > 140:
        actions.append("Reduce sodium & monitor BP regularly")
    if thalach < 120:
        actions.append("Increase aerobic exercise gradually")
    if fbs == 1:
        actions.append("Control sugar & carbohydrate intake")
    if age > 50:
        actions.append("Schedule cardiology screening")

    if not actions:
        actions.append("Maintain current healthy lifestyle")

    for a in actions:
        st.write("✅", a)

    # ---------------- FOLLOW UP ----------------
    st.markdown("### Follow-Up Guidance")

    if level == "HIGH":
        st.error("Consult a cardiologist soon for detailed evaluation.")
    elif level == "MODERATE":
        st.warning("Medical checkup recommended within near term.")
    else:
        st.success("Continue preventive lifestyle and periodic monitoring.")

    st.caption("AI-based academic prototype — not a substitute for medical diagnosis.")
