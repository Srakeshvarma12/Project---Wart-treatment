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
label { color: #003366 !important; font-weight: 600; }
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
st.markdown('<div class="subtitle">ML-based tool to estimate treatment success probability + recommend best option</div>', unsafe_allow_html=True)

# ================= Extract ALL supported categories from trained features ================= #
def extract_categories(prefix: str) -> list[str]:
    vals = []
    for col in feature_columns:
        if col.startswith(prefix):
            vals.append(col.replace(prefix, "").strip())
    # unique + sorted
    return sorted(set(vals))

wart_types = extract_categories("Wart Type_")
treatment_methods = extract_categories("Treatment Method_")

# Side effects: keep explicit (stable UI), but only use model-supported columns safely
side_effect_options = ["None", "Mild", "Severe"]

# ================= Market-based cost estimates (INR) =================
# NOTE: These are approximate ranges; real cost varies by city, clinic, # lesions, sessions.
# Sources (India):
# - Laser: ₹5,000–₹50,000 (avg ~₹20,000)  (Yashoda Hospitals / HexaHealth)
# - Electrocautery: ₹10,000–₹30,000 (Pristyn Care / Lybrate) [genital examples]
# - Cryotherapy: varies widely; often per lesion/session (some sources mention ₹1,500–₹5,000 per session; genital packages higher)
# - Topical (salicylic/lactic acid): OTC ~₹400 range for solutions (1mg)
# - Immunotherapy (some intralesional vaccine vials reported ~₹450 in literature)
COST_TABLE = {
    "Topical": (300, 800),
    "Salicylic Acid": (200, 600),
    "Cryotherapy": (1500, 5000),         # per session/lesion typical clinic range (varies)
    "Immunotherapy": (500, 3000),        # depends on antigen/vaccine + sessions
    "Electrosurgery": (10000, 30000),
    "Electrocautery": (10000, 30000),
    "Surgical": (15000, 40000),
    "Laser": (5000, 50000),
}

def estimate_cost_inr(method: str, wart_type: str, side_effect: str) -> int:
    base_min, base_max = COST_TABLE.get(method, (2000, 10000))
    base = (base_min + base_max) / 2

    # small multipliers (simple, explainable heuristic)
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

    est = int(round(base * wt_mult * se_mult))
    return max(est, 0)

# ================= Helper: build model input safely =================
def build_input_df(
    age: int,
    gender: str,
    wart_type: str,
    method: str,
    side_effect: str,
    treatment_cost: int
) -> pd.DataFrame:
    # start with all zeros in correct order
    x = {col: 0 for col in feature_columns}

    # numeric
    if "Age" in x:
        x["Age"] = age
    if "Treatment Cost" in x:
        x["Treatment Cost"] = treatment_cost

    # gender (most common encoding)
    if "Gender_Male" in x:
        x["Gender_Male"] = 1 if gender == "Male" else 0

    # wart type one-hot
    wt_col = f"Wart Type_{wart_type}"
    if wt_col in x:
        x[wt_col] = 1

    # treatment method one-hot
    tm_col = f"Treatment Method_{method}"
    if tm_col in x:
        x[tm_col] = 1

    # side effects (support both styles)
    # If your model uses Mild/Severe columns:
    if "Side Effects_Mild" in x:
        x["Side Effects_Mild"] = 1 if side_effect == "Mild" else 0
    if "Side Effects_Severe" in x:
        x["Side Effects_Severe"] = 1 if side_effect == "Severe" else 0
    # If your model uses a one-hot style (rare)
    se_col = f"Side Effects_{side_effect}"
    if se_col in x:
        x[se_col] = 1

    return pd.DataFrame([x], columns=feature_columns)

def predict_success_proba(input_df: pd.DataFrame) -> float:
    """
    FIX for your Streamlit Cloud error:
    Some sklearn objects enforce feature-name checks.
    Passing numpy array avoids mismatch issues while keeping column order correct.
    """
    X = input_df.values  # exact order = feature_columns
    X_scaled = scaler.transform(X)
    return float(model.predict_proba(X_scaled)[0][1])

# ================= UI =================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("👤 Patient & 🦠 Treatment Inputs")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    wart_type = st.selectbox("Wart Type", wart_types if wart_types else ["Common", "Plantar", "Flat"])

with col2:
    side_effects = st.selectbox("Side Effects", side_effect_options)
    chosen_method = st.selectbox("Treatment Method (for single prediction)", treatment_methods if treatment_methods else ["Cryotherapy", "Immunotherapy", "Topical"])

# Auto-generate treatment cost (fixed, non-editable)
treatment_cost = estimate_cost_inr(chosen_method, wart_type, side_effects)

st.info(
    f"💰 Estimated Treatment Cost for **{chosen_method}**: "
    f"**₹{treatment_cost:,}**"
)

# Recommendation toggle
show_reco = st.toggle("Show best treatment recommendation (rank all options)", value=True)

predict_btn = st.button("🚀 Predict", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ================= RESULTS =================
if predict_btn:
    # Single selected method prediction
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

    # Recommendation: evaluate ALL supported methods
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
                "Estimated Cost (INR)": int(auto_c),
                "Success Probability": round(p, 4),
                "Success Rate (%)": round(p * 100, 2),
            })

        rec_df = pd.DataFrame(rows).sort_values(by="Success Probability", ascending=False).reset_index(drop=True)
        best = rec_df.iloc[0]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🏆 Best Treatment Recommendation (Model-Based)")
        st.success(f"Recommended: **{best['Treatment Method']}**  •  Success ≈ **{best['Success Rate (%)']}%**  •  Est. cost ≈ **₹{best['Estimated Cost (INR)']:,}**")

        st.write("All treatment options ranked:")
        st.dataframe(rec_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.caption(
            "Note: Recommendation is based on the trained ML model + estimated costs. "
            "Actual clinical decision depends on lesion count, location, clinician evaluation, and patient factors."
        )


