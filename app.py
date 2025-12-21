import streamlit as st
import pandas as pd
import pickle

# ================= LOAD MODEL FILES ================= #

with open("model/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# ================= EXTRACT CATEGORIES FROM MODEL ================= #

wart_types = sorted({
    col.replace("Wart Type_", "")
    for col in feature_columns
    if col.startswith("Wart Type_")
})

treatment_methods = sorted({
    col.replace("Treatment Method_", "")
    for col in feature_columns
    if col.startswith("Treatment Method_")
})

side_effect_types = ["None", "Mild", "Severe"]

# ================= PAGE CONFIG ================= #

st.set_page_config(
    page_title="Wart Treatment Decision Support System",
    page_icon="🩺",
    layout="centered"
)

# ================= CSS ================= #

st.markdown("""
<style>

.stApp {
    background: linear-gradient(180deg, #e6f4ff, #f9fcff);
    font-family: 'Segoe UI', sans-serif;
    color: #003366;
}

.main-title {
    text-align: center;
    font-size: clamp(26px, 4vw, 38px);
    font-weight: 700;
    color: #007acc;
    margin-top: 20px;
}

.subtitle {
    text-align: center;
    font-size: clamp(14px, 2.5vw, 16px);
    color: #335f8a;
    margin-bottom: 30px;
}

.card {
    background: white;
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

label {
    color: #003366 !important;
    font-weight: 600;
}

div.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-size: 16px;
    font-weight: 700;
    border-radius: 30px;
    padding: 14px;
    border: none;
    width: 100%;
}

[data-testid="stMetricValue"] {
    font-size: 30px;
    font-weight: 800;
    color: #0072ff;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER ================= #

st.markdown(
    '<div class="main-title">🩺 Wart Treatment Decision Support System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning–based system to predict success and recommend optimal treatment</div>',
    unsafe_allow_html=True
)

# ================= INPUT CARD ================= #

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("👤 Patient & 🦠 Treatment Inputs")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    wart_type = st.selectbox("Wart Type", wart_types)

with col2:
    treatment_cost = st.number_input("Treatment Cost", min_value=0, value=100)
    treatment_method = st.selectbox("Treatment Method", treatment_methods)
    side_effects = st.selectbox("Side Effects", side_effect_types)

predict_btn = st.button("🚀 Predict & Recommend")
st.markdown('</div>', unsafe_allow_html=True)

# ================= PREDICTION & RECOMMENDATION ================= #

if predict_btn:
    # Base input template
    base_input = {feature: 0 for feature in feature_columns}

    base_input["Age"] = age
    base_input["Treatment Cost"] = treatment_cost
    base_input["Gender_Male"] = 1 if gender == "Male" else 0
    base_input[f"Wart Type_{wart_type}"] = 1

    if side_effects == "Mild":
        base_input["Side Effects_Mild"] = 1
    elif side_effects == "Severe":
        base_input["Side Effects_Severe"] = 1

    # ---------- Predict selected treatment ---------- #
    selected_input = base_input.copy()
    selected_input[f"Treatment Method_{treatment_method}"] = 1

    selected_df = pd.DataFrame([selected_input])
    selected_scaled = scaler.transform(selected_df)
    selected_prob = model.predict_proba(selected_scaled)[0][1]
    selected_rate = round(selected_prob * 100, 2)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    st.metric("Selected Treatment Success Rate", f"{selected_rate}%")
    st.progress(selected_prob)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Recommendation: simulate all treatments ---------- #
    results = {}

    for method in treatment_methods:
        temp_input = base_input.copy()

        for m in treatment_methods:
            temp_input[f"Treatment Method_{m}"] = 0

        temp_input[f"Treatment Method_{method}"] = 1

        temp_df = pd.DataFrame([temp_input])
        temp_scaled = scaler.transform(temp_df)
        prob = model.predict_proba(temp_scaled)[0][1]
        results[method] = round(prob * 100, 2)

    best_treatment = max(results, key=results.get)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🩺 Treatment Recommendation")

    st.success(
        f"✅ Recommended Treatment: **{best_treatment}** "
        f"(Predicted Success Rate: **{results[best_treatment]}%**)"
    )

    st.write("### 📊 Success Rate Comparison")
    for k, v in results.items():
        st.write(f"- **{k}** : {v}%")

    st.info(
        "ℹ️ This recommendation is generated by simulating all treatment options "
        "using the trained machine learning model. For educational use only."
    )

    st.markdown('</div>', unsafe_allow_html=True)
