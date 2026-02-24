import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# ---------------- SAFE SHAP IMPORT ----------------
try:
    import shap
    shap_available = True
except:
    shap_available = False

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="FinSecure AI - Fraud Detection",
    layout="wide",
    page_icon="💳"
)

# ---------------- DARK THEME ----------------
st.markdown("""
    <style>
    body {background-color: #0E1117;}
    .main {background-color: #0E1117;}
    h1, h2, h3, h4 {color: white;}
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/xgboost_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ---------------- BRANDING ----------------
st.title("💳 FinSecure AI")
st.markdown("### Enterprise-Grade Fraud Detection & Risk Intelligence Platform")
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙ Risk Configuration")

threshold = st.sidebar.slider(
    "Fraud Decision Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

mode = st.sidebar.radio(
    "Input Mode",
    ["Demo Mode", "Upload CSV"]
)

# ---------------- INPUT SECTION ----------------
if mode == "Demo Mode":
    input_df = pd.DataFrame(
        np.random.normal(size=(1, len(model.feature_names_in_))),
        columns=model.feature_names_in_
    )
    st.subheader("📊 Generated Demo Transaction")
    st.dataframe(input_df)

else:
    uploaded_file = st.file_uploader("Upload Transaction CSV")
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        st.dataframe(input_df.head())
    else:
        st.stop()

# ---------------- FEATURE ALIGNMENT ----------------
if "Class" in input_df.columns:
    input_df = input_df.drop("Class", axis=1)

input_df = input_df[model.feature_names_in_]

if "Amount" in input_df.columns and "Time" in input_df.columns:
    input_df[['Amount', 'Time']] = scaler.transform(
        input_df[['Amount', 'Time']]
    )

# ---------------- PREDICTION ----------------
if st.button("🚀 Analyze Transaction"):

    probabilities = model.predict_proba(input_df)[:, 1]

    st.markdown("---")
    st.subheader("📈 Risk Assessment Results")

    for i, prob in enumerate(probabilities):

        decision = 1 if prob >= threshold else 0

        col1, col2, col3 = st.columns(3)

        col1.metric("Fraud Probability", f"{prob:.4f}")
        col2.metric("Decision Threshold", threshold)

        if decision == 1:
            col3.error("⚠ HIGH RISK")
        else:
            col3.success("✅ LOW RISK")

        st.progress(float(prob))

    # ---------------- HISTORY LOGGING ----------------
    history = input_df.copy()
    history["Fraud_Probability"] = probabilities
    history["Threshold"] = threshold
    history["Timestamp"] = datetime.datetime.now()

    history.to_csv(
        "outputs/prediction_history.csv",
        mode="a",
        header=False,
        index=False
    )

    # ---------------- DOWNLOAD REPORT ----------------
    csv = history.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Risk Report",
        csv,
        "risk_report.csv",
        "text/csv"
    )

    # ---------------- SHAP EXPLAINABILITY ----------------
    st.markdown("---")
    st.subheader("🧠 Model Explainability (SHAP)")

    if shap_available:

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)

            # Global SHAP Importance
            st.subheader("📊 Global Feature Impact")

            fig_summary = plt.figure()
            shap.summary_plot(
                shap_values,
                input_df,
                plot_type="bar",
                show=False
            )
            st.pyplot(fig_summary)

            # Local SHAP Explanation
            st.subheader("🔎 Individual Transaction Explanation")

            fig_force = plt.figure()
            shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                input_df.iloc[0],
                matplotlib=True,
                show=False
            )
            st.pyplot(fig_force)

        except Exception as e:
            st.warning("SHAP explanation temporarily unavailable in this environment.")

    else:
        st.info("Model explainability (SHAP) disabled in cloud deployment.")