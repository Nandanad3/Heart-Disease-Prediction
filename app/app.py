import streamlit as st
import pickle
import pandas as pd

with open("app/style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# Page Config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# Load Model
with open("models/heart_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Header
st.markdown("""
# ❤️ Heart Disease Prediction System
### Machine Learning Based Health Risk Assessment
""")

st.divider()

# Layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=50
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=700,
        value=200
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )

with col2:

    oldpeak = st.number_input(
        "Old Peak",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        ["No", "Yes"]
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"]
    )

    ca = st.slider(
        "Major Vessels",
        0,
        4,
        0
    )

st.divider()

predict_button = st.button(
    "🔍 Predict Heart Disease Risk",
    use_container_width=True
)

if predict_button:

    input_df = pd.DataFrame(
        columns=model_columns
    )

    input_df.loc[0] = 0

    if "age" in input_df.columns:
        input_df["age"] = age

    if "trestbps" in input_df.columns:
        input_df["trestbps"] = trestbps

    if "chol" in input_df.columns:
        input_df["chol"] = chol

    if "thalch" in input_df.columns:
        input_df["thalch"] = thalach

    if "oldpeak" in input_df.columns:
        input_df["oldpeak"] = oldpeak

    if "ca" in input_df.columns:
        input_df["ca"] = ca

    if "fbs" in input_df.columns:
        input_df["fbs"] = 1 if fbs == "Yes" else 0

    if "exang" in input_df.columns:
        input_df["exang"] = 1 if exang == "Yes" else 0

    prediction = model.predict(input_df)

    st.divider()

    if prediction[0] == 1:

        st.error(
            "⚠️ High Risk of Heart Disease Detected"
        )

        st.markdown("""
### Recommendations
- Consult a cardiologist
- Maintain healthy diet
- Exercise regularly
- Monitor blood pressure
- Avoid smoking
""")

    else:

        st.success(
            "✅ No Significant Heart Disease Risk Detected"
        )

        st.markdown("""
### Healthy Habits
- Continue regular exercise
- Maintain balanced diet
- Routine health checkups
- Stay hydrated
""")