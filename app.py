# import streamlit as st
# import pandas as pd
# import pickle
# import numpy as np

# # Load saved model files
# with open("model/logistic_model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("model/scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# with open("model/features.pkl", "rb") as f:
#     feature_columns = pickle.load(f)

# st.set_page_config(page_title="Wart Treatment Recommendation", layout="centered")

# st.title("🩺 Wart Treatment Recommendation System")
# st.write("Enter patient and treatment details to predict treatment success.")

# # ---------------- User Inputs ---------------- #

# age = st.number_input("Age", min_value=1, max_value=100, value=30)
# treatment_cost = st.number_input("Treatment Cost", min_value=0, value=100)

# gender = st.selectbox("Gender", ["Male", "Female"])
# wart_type = st.selectbox("Wart Type", ["Common", "Plantar", "Flat"])
# treatment_method = st.selectbox(
#     "Treatment Method",
#     ["Cryotherapy", "Immunotherapy", "Topical"]
# )
# side_effects = st.selectbox(
#     "Side Effects",
#     ["None", "Mild", "Severe"]
# )

# # ---------------- Prediction ---------------- #

# if st.button("Predict Treatment Outcome"):

#     input_data = {
#         "Age": age,
#         "Treatment Cost": treatment_cost,
#         "Gender_Male": 1 if gender == "Male" else 0,
#         "Wart Type_Plantar": 1 if wart_type == "Plantar" else 0,
#         "Wart Type_Flat": 1 if wart_type == "Flat" else 0,
#         "Treatment Method_Immunotherapy": 1 if treatment_method == "Immunotherapy" else 0,
#         "Treatment Method_Topical": 1 if treatment_method == "Topical" else 0,
#         "Side Effects_Mild": 1 if side_effects == "Mild" else 0,
#         "Side Effects_Severe": 1 if side_effects == "Severe" else 0
#     }

#     # Convert input to DataFrame
#     input_df = pd.DataFrame([input_data])

#     # Align with training features
#     input_df = input_df.reindex(columns=feature_columns, fill_value=0)

#     # Scale input
#     input_scaled = scaler.transform(input_df)

#     # Predict
#     prediction = model.predict(input_scaled)[0]

#     if prediction == 1:
#         st.success("✅ High probability of successful treatment")
#     else:
#         st.error("❌ Low probability of treatment success")



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

# ================= PAGE CONFIG ================= #

st.set_page_config(
    page_title="Wart Treatment Decision Support System",
    page_icon="🩺",
    layout="centered"
)

# ================= CSS ================= #

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(180deg, #e6f4ff, #f9fcff);
    font-family: 'Segoe UI', sans-serif;
    color: #003366;
}

/* Main title */
.main-title {
    text-align: center;
    font-size: clamp(26px, 4vw, 38px);
    font-weight: 700;
    color: #007acc;
    margin-top: 20px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: clamp(14px, 2.5vw, 16px);
    color: #335f8a;
    margin-bottom: 30px;
}

/* Card */
.card {
    background: white;
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* Labels */
label {
    color: #003366 !important;
    font-weight: 600;
}

/* Button */
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

/* Metric */
[data-testid="stMetricValue"] {
    font-size: 30px;
    font-weight: 800;
    color: #0072ff;
}

/* Progress bar */
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
    '<div class="subtitle">Machine Learning–based clinical tool to estimate treatment success probability</div>',
    unsafe_allow_html=True
)

# ================= INPUT CARD ================= #

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("👤 Patient & 🦠 Treatment Inputs")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    wart_type = st.selectbox("Wart Type", ["Common", "Plantar", "Flat", "Genital", "Butchers", "Filiform", "Mosaic"])

with col2:
    treatment_cost = st.number_input("Treatment Cost", min_value=0, value=100)
    treatment_method = st.selectbox(
        "Treatment Method",
        ["Cryotherapy", "Immunotherapy", "Topical", "Electrosurgery", "Salicylic Acid", "Surgical", "Laser"]
    )

    # ✅ MILD FIX — explicit & model-safe
    side_effects = st.selectbox(
        "Side Effects",
        ["None", "Mild", "Severe"]
    )

predict_btn = st.button("🚀 Predict Treatment Success")
st.markdown('</div>', unsafe_allow_html=True)

# ================= PREDICTION ================= #

if predict_btn:
    input_data = {
        "Age": age,
        "Treatment Cost": treatment_cost,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Wart Type_Plantar": 1 if wart_type == "Plantar" else 0,
        "Wart Type_Flat": 1 if wart_type == "Flat" else 0,
        "Treatment Method_Immunotherapy": 1 if treatment_method == "Immunotherapy" else 0,
        "Treatment Method_Topical": 1 if treatment_method == "Topical" else 0,

        # ✅ Correct side-effect mapping (same as your earlier working code)
        "Side Effects_Mild": 1 if side_effects == "Mild" else 0,
        "Side Effects_Severe": 1 if side_effects == "Severe" else 0
    }

    # Align with trained feature set
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    input_scaled = scaler.transform(input_df)
    probability = model.predict_proba(input_scaled)[0][1]
    success_rate = round(probability * 100, 2)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    st.metric("Predicted Success Rate", f"{success_rate}%")
    st.progress(probability)

    if probability >= 0.5:
        st.success("✅ High probability of successful treatment")
    else:
        st.error("❌ Low probability of treatment success")

    st.markdown('</div>', unsafe_allow_html=True)
