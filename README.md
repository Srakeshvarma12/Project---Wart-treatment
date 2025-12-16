🩺 Wart Treatment Decision Support System

An AI-powered clinical decision support system that predicts the probability of treatment success for wart treatments based on patient and treatment characteristics.

🔗 Live App:
👉 (https://project---wart-treatment-gh7exzv3va8liynunuxemu.streamlit.app/)

📌 Project Overview

This project uses Machine Learning to assist healthcare decision-making by estimating treatment success outcomes for wart patients.
The system takes patient details and treatment parameters as input and provides a success probability prediction using a trained classification model.

⚙️ Tech Stack Used
Programming & Tools

Python

Streamlit (Web UI & Deployment)

GitHub (Version control)

Data & ML

Pandas – Data processing

NumPy – Numerical computation

Scikit-learn – Machine Learning (Logistic Regression)

StandardScaler – Feature scaling

Model Persistence

Pickle – Model & scaler serialization

🧠 Machine Learning Details

Model Used: Logistic Regression

Problem Type: Binary Classification

Target Variable: Treatment Success (High / Low probability)

Preprocessing:

One-hot encoding for categorical variables

Feature scaling using StandardScaler

Saved Artifacts:

logistic_model.pkl

scaler.pkl

features.pkl

🧾 Input Features

Age

Gender

Wart Type (Common, Plantar, Flat, etc.)

Treatment Method (Cryotherapy, Immunotherapy, Topical, Electrosurgery, etc.)

Treatment Cost

Side Effects (None, Mild, Severe)

📊 Output

Predicted Treatment Success Probability

Clear visual feedback:

✅ High probability of success

❌ Low probability of success

🖥️ Application Features

Clean and responsive UI built with Streamlit

Dropdowns dynamically aligned with training data

Real-time predictions

Deployable as a cloud-based prototype
