import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ================= LOAD MODEL FILES ================= #
with open("model/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

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
st.markdown('<div class="main-title">🩺 Wart Treatment Decision Support System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">ML-based tool to estimate treatment success probability and recommend optimal therapy</div>',
    unsafe_allow_html=True
)

# ================= EXTRACT SUPPORTED CATEGORIES ================= #
def extract_categories(prefix: str):
    return sorted({c.replace(prefix, "") for c in feature_columns if c.startswith(prefix)})

wart_types = extract_categories("Wart Type_")
treatment_methods = extract_categories("Treatment Method_")
side_effect_options = ["None", "Mild", "Severe"]

# ================= COST TABLE (INR – Approx Market Values) ================= #
COST_TABLE = {
    "Topical": (300, 800),
    "Salicylic Acid": (200, 600),
    "Cryotherapy": (1500, 5000),
    "Immunotherapy": (500, 3000),
    "Electrosurgery": (10000, 30000),
    "Electrocautery": (10000, 30000),
    "Surgical": (15000, 40000),
    "Laser": (5000, 50000),
}

def estimate_cost_inr(method, wart_type, side_effect):
    base_min, base_max = COST_TABLE.get(method, (2000, 10000))
    base = (base_min + base_max) / 2

    wt_mult = 1.0
    if "Plantar" in wart_type:
        wt_mult = 1.15
    elif "Genital" in wart_type:
        wt_mult = 1.25
    elif "Mosaic" in wart_type:
        wt_mult = 1.20

    se_mult = 1.0
    if side_effect == "Mild":
        se_mult = 1.05
    elif side_effect == "Severe":
        se_mult = 1.12

    return int(round(base * wt_mult * se_mult))

# ================= BUILD INPUT ================= #
def build_input_df(age, gender, wart_type, method, side_effect, treatment_cost):
    x = {col: 0 for col in feature_columns}

    if "Age" in x:
        x["Age"] = age
    if "Treatment Cost" in x:
        x["Treatment Cost"] = treatment_cost
    if "Gender_Male" in x:
        x["Gender_Male"] = 1 if gender == "Male" else 0

    wt_col = f"Wart Type_{wart_type}"
    if wt_col in x:
        x[wt_col] = 1

    tm_col = f"Treatment Method_{method}"
    if tm_col in x:
        x[tm_col] = 1

    if "Side Effects_Mild" in x:
        x["Side Effects_Mild"] = 1 if side_effect == "Mild" else 0
    if "Side Effects_Severe" in x:
        x["Side Effects_Severe"] = 1 if side_effect == "Severe" else 0

    return pd.DataFrame([x], columns=feature_columns)

def predict_success_proba(df):
    X_scaled = scaler.transform(df.values)  # SAFE for Streamlit Cloud
    return float(model.predict_proba(X_scaled)[0][1])

# ================= UI INPUTS ================= #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("👤 Patient & 🦠 Treatment Inputs")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    wart_type = st.selectbox("Wart Type", wart_types or ["Common"])

with col2:
    side_effects = st.selectbox("Side Effects", side_effect_options)
    chosen_method = st.selectbox(
        "Treatment Method (for single prediction)",
        treatment_methods or ["Cryotherapy"]
    )

treatment_cost = estimate_cost_inr(chosen_method, wart_type, side_effects)

st.info(f"💰 Estimated Treatment Cost for **{chosen_method}**: **₹{treatment_cost:,}**")

show_reco = st.toggle("Show best treatment recommendation (rank all options)", value=True)
predict_btn = st.button("🚀 Predict", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ================= RESULTS ================= #
if predict_btn:
    selected_df = build_input_df(
        age=int(age),
        gender=gender,
        wart_type=wart_type,
        method=chosen_method,
        side_effect=side_effects,
        treatment_cost=int(treatment_cost)
    )

    proba = predict_success_proba(selected_df)
    success_rate = round(proba * 100, 2)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction (Selected Method)")
    st.metric("Predicted Success Rate", f"{success_rate}%")
    st.progress(proba)

    if proba >= 0.5:
        st.success("✅ High probability of successful treatment")
    else:
        st.error("❌ Low probability of treatment success")

    st.markdown('</div>', unsafe_allow_html=True)

    if show_reco and treatment_methods:
        rows = []
        for m in treatment_methods:
            auto_c = estimate_cost_inr(m, wart_type, side_effects)

            df_m = build_input_df(
                age=int(age),
                gender=gender,
                wart_type=wart_type,
                method=m,
                side_effect=side_effects,
                treatment_cost=int(auto_c)
            )

            p = predict_success_proba(df_m)

            rows.append({
                "Treatment Method": m,
                "Estimated Cost (INR)": auto_c,
                "Success Probability": round(p, 4),
                "Success Rate (%)": round(p * 100, 2)
            })

        rec_df = pd.DataFrame(rows).sort_values(
            by="Success Probability", ascending=False
        ).reset_index(drop=True)

        best = rec_df.iloc[0]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🏆 Best Treatment Recommendation")
        st.success(
            f"**{best['Treatment Method']}** → "
            f"Success ≈ **{best['Success Rate (%)']}%**, "
            f"Estimated Cost ≈ **₹{best['Estimated Cost (INR)']:,}**"
        )

        st.dataframe(rec_df, use_container_width=True)
        st.caption(
            "Note: Recommendation is model-based and cost-adjusted. "
            "Final clinical decisions should involve medical professionals."
        )
        st.markdown('</div>', unsafe_allow_html=True)
