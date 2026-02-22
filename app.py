import streamlit as st
import pickle
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(page_title="CardioGuard AI", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
.title {
    text-align:center;
    color:#d4af37;
}
.subtitle {
    text-align:center;
    color:#cbd5e1;
}
.section {
    margin-top:30px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))

feature_names = ["age","sex","cp","trestbps","chol","fbs","restecg",
                 "thalach","exang","oldpeak","slope","ca","thal"]

# ---------------- HEADER ----------------
st.markdown("<h1 class='title'>CardioGuard AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Heart Attack Risk Prediction & Prevention Advisor</p>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- INPUT SECTION ----------------
st.markdown("## Patient Clinical Parameters")

c1, c2, c3 = st.columns(3)

with c1:
    age = st.slider("Age", 20, 100, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    sex = 1 if sex == "Male" else 0
    trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 600, 220)

with c2:
    cp = st.slider("Chest Pain Type (0-3)", 0, 3, 1)
    thalach = st.slider("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    exang = 1 if exang == "Yes" else 0
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)

with c3:
    fbs = st.selectbox("Fasting Sugar >120", ["No", "Yes"])
    fbs = 1 if fbs == "Yes" else 0
    ca = st.slider("Major Vessels (0-4)", 0, 4, 0)
    slope = st.slider("Slope (0-2)", 0, 2, 1)
    thal = st.slider("Thal (0-3)", 0, 3, 2)
    restecg = st.slider("Rest ECG (0-2)", 0, 2, 1)

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("Analyze Heart Risk", use_container_width=True):

    data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                      thalach,exang,oldpeak,slope,ca,thal]])

    prob = model.predict_proba(data)[0][1]
    percent = round(prob * 100, 2)

    st.markdown("## Clinical Risk Report")

    # ---------------- RISK CATEGORY ----------------
    if percent < 30:
        level = "LOW"
        st.success(f"Overall Clinical Risk Category: {level}")
    elif percent < 70:
        level = "MODERATE"
        st.warning(f"Overall Clinical Risk Category: {level}")
    else:
        level = "HIGH"
        st.error(f"Overall Clinical Risk Category: {level}")

    st.metric("Predicted Probability of Heart Disease", f"{percent}%")
    st.progress(percent / 100)

    st.info("Risk estimation is based on multi-parameter interaction patterns, not individual thresholds alone.")

    # ---------------- FEATURE IMPORTANCE ----------------
    st.markdown("### Top Contributing Model Features")

    importances = model.feature_importances_
    sorted_indices = np.argsort(importances)[::-1][:3]

    for idx in sorted_indices:
        st.write(f"• {feature_names[idx]} (importance: {round(importances[idx],3)})")

    # ---------------- PARAMETER STATUS ----------------
    st.markdown("### Parameter Status Overview")

    def status_check(name, value, low, high):
        if value < low:
            st.write(f"• {name}: Below normal range")
        elif value > high:
            st.write(f"• {name}: Above normal range")
        else:
            st.write(f"• {name}: Within normal range")

    status_check("Blood Pressure", trestbps, 90, 140)
    status_check("Cholesterol", chol, 150, 240)
    status_check("Max Heart Rate", thalach, 120, 200)

    # ---------------- RISK DRIVERS ----------------
    st.markdown("### Primary Risk Indicators")

    drivers = []

    if cp > 1:
        drivers.append("Abnormal chest pain pattern")
    if ca > 0:
        drivers.append("Coronary vessel blockage indicator")
    if thal > 2:
        drivers.append("Thalassemia abnormality")
    if oldpeak > 1:
        drivers.append("ST depression irregularity")
    if chol > 240:
        drivers.append("Elevated cholesterol level")
    if trestbps > 140:
        drivers.append("High resting blood pressure")

    if drivers:
        for d in drivers:
            st.write("🔴", d)
    else:
        st.write("Risk influenced by complex multi-factor interaction.")

    # ---------------- LIFESTYLE SCORE ----------------
    st.markdown("### Heart Health Score")

    score = 100

    # probability-based adjustment
    if percent > 70:
        score -= 25
    elif percent > 40:
        score -= 15

    # clinical adjustments
    if chol > 240: score -= 10
    if trestbps > 140: score -= 10
    if fbs == 1: score -= 5
    if exang == 1: score -= 10

    score = max(score, 0)

    st.metric("Estimated Heart Health Score", f"{score}/100")

    # ---------------- ACTION PLAN ----------------
    st.markdown("### Recommended Preventive Actions")

    actions = []

    if chol > 240:
        actions.append("Adopt low-fat, high-fiber diet")
    if trestbps > 140:
        actions.append("Reduce sodium intake and monitor blood pressure")
    if thalach < 120:
        actions.append("Increase aerobic exercise gradually")
    if fbs == 1:
        actions.append("Control sugar intake and monitor glucose")
    if age > 50:
        actions.append("Schedule periodic cardiology screening")

    if not actions:
        actions.append("Maintain current healthy lifestyle with periodic check-ups")

    for a in actions:
        st.write("✅", a)

    # ---------------- FOLLOW UP ----------------
    st.markdown("### Follow-Up Recommendation")

    if level == "HIGH":
        st.error("Consult a cardiologist for detailed clinical evaluation.")
    elif level == "MODERATE":
        st.warning("Medical check-up recommended in the near term.")
    else:
        st.success("Continue preventive lifestyle and routine monitoring.")

    st.caption("Academic AI-based decision support tool — not a substitute for professional medical diagnosis.")
