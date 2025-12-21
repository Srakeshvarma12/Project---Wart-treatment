import streamlit as st
import pandas as pd
import pickle

# =====================================================
# LOAD TRAINED MODEL FILES
# =====================================================
with open("model/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model/features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Wart Treatment Decision Support System",
    page_icon="🩺",
    layout="centered"
)

# =====================================================
# CSS (LIGHT SKY-BLUE THEME)
# =====================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #e6f4ff, #f9fcff);
    font-family: 'Segoe UI', sans-serif;
}
.card {
    background: white;
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}
.main-title {
    text-align: center;
    font-size: clamp(26px, 4vw, 38px);
    font-weight: 700;
    color: #007acc;
}
.subtitle {
    text-align: center;
    color: #335f8a;
    margin-bottom: 30px;
}
div.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-size: 16px;
    font-weight: 700;
    border-radius: 30px;
    padding: 14px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown('<div class="main-title">🩺 Wart Treatment Decision Support System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">ML-based tool to estimate treatment success probability and recommend age-appropriate options</div>',
    unsafe_allow_html=True
)

# =====================================================
# UTILITIES
# =====================================================
def extract_categories(prefix):
    return sorted({c.replace(prefix, "") for c in feature_columns if c.startswith(prefix)})

wart_types = extract_categories("Wart Type_")
treatment_methods = extract_categories("Treatment Method_")
side_effect_options = ["None", "Mild", "Severe"]

# =====================================================
# AGE GROUP LOGIC (CLINICAL SAFETY)
# =====================================================
def age_group(age):
    if age < 12:
        return "child"
    elif age < 18:
        return "adolescent"
    elif age < 60:
        return "adult"
    else:
        return "elderly"

AGE_RESTRICTIONS = {
    "child": ["Electrosurgery", "Laser", "Surgical"],
    "elderly": ["Surgical"]
}

# =====================================================
# COST ESTIMATION (INR)
# =====================================================
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

def estimate_cost(method, wart_type, side_effect):
    base = sum(COST_TABLE.get(method, (2000, 10000))) / 2
    if wart_type == "Genital":
        base *= 1.25
    if side_effect == "Mild":
        base *= 1.05
    elif side_effect == "Severe":
        base *= 1.12
    return int(base)

# =====================================================
# BUILD INPUT (SAFE FEATURE ALIGNMENT)
# =====================================================
def build_input_df(age, gender, wart_type, method, side_effect, cost):
    row = {col: 0 for col in feature_columns}
    row["Age"] = age
    row["Treatment Cost"] = cost
    row["Gender_Male"] = 1 if gender == "Male" else 0

    if f"Wart Type_{wart_type}" in row:
        row[f"Wart Type_{wart_type}"] = 1

    if f"Treatment Method_{method}" in row:
        row[f"Treatment Method_{method}"] = 1

    if "Side Effects_Mild" in row:
        row["Side Effects_Mild"] = 1 if side_effect == "Mild" else 0
    if "Side Effects_Severe" in row:
        row["Side Effects_Severe"] = 1 if side_effect == "Severe" else 0

    return pd.DataFrame([row], columns=feature_columns)

def predict_proba(df):
    X = scaler.transform(df.values)
    return model.predict_proba(X)[0][1]

# =====================================================
# INPUT UI
# =====================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("👤 Patient & 🦠 Treatment Inputs")

c1, c2 = st.columns(2)

with c1:
    age = st.number_input("Age", 1, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    wart_type = st.selectbox("Wart Type", wart_types)

with c2:
    side_effects = st.selectbox("Side Effects", side_effect_options)
    selected_method = st.selectbox("Treatment Method (single prediction)", treatment_methods)

cost = estimate_cost(selected_method, wart_type, side_effects)
st.info(f"💰 Estimated Treatment Cost for **{selected_method}**: **₹{cost:,}**")

show_reco = st.toggle("Show decision-support recommendation (rank options)", value=True)
predict_btn = st.button("🚀 Predict")
st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# RESULTS
# =====================================================
if predict_btn:
    df_sel = build_input_df(age, gender, wart_type, selected_method, side_effects, cost)
    p = predict_proba(df_sel)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction (Selected Method)")
    st.metric("Predicted Success Rate", f"{round(p*100,2)}%")
    st.progress(p)
    st.markdown('</div>', unsafe_allow_html=True)

    # =================================================
    # DECISION SUPPORT (AGE-AWARE)
    # =================================================
    if show_reco:
        rows = []
        group = age_group(age)
        restricted = AGE_RESTRICTIONS.get(group, [])

        for m in treatment_methods:
            auto_cost = estimate_cost(m, wart_type, side_effects)
            df_m = build_input_df(age, gender, wart_type, m, side_effects, auto_cost)
            prob = predict_proba(df_m)

            # Penalize unsafe methods
            if m in restricted:
                prob *= 0.6

            rows.append({
                "Treatment Method": m,
                "Estimated Cost (INR)": auto_cost,
                "Adjusted Success (%)": round(prob*100, 2),
                "Age Safe": "No" if m in restricted else "Yes"
            })

        rec_df = pd.DataFrame(rows).sort_values("Adjusted Success (%)", ascending=False)

        best = rec_df.iloc[0]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🏆 Model-Supported Recommendation")
        st.success(
            f"Recommended option: **{best['Treatment Method']}** "
            f"(≈ {best['Adjusted Success (%)']}% success, ₹{best['Estimated Cost (INR)']:,})"
        )
        st.dataframe(rec_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.caption(
            "⚠️ This is a machine-learning–based decision support system. "
            "It does not replace clinical judgment or provide medical prescriptions."
        )
