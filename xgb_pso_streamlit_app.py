# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:42:06 2026

@author: 24550372
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="XGB-PSO Predictor for Pharmaceutical Degradation",
    page_icon="🧪",
    layout="centered",
)

# =========================
# Load model
# =========================
@st.cache_resource
def load_model():
    # Put the model file in the same folder as this app,
    # or replace the path below with your deployed path.
    return joblib.load("XGBPSOModel_success_seed1446.pkl")

model = load_model()

# =========================
# Title / description
# =========================
st.title("XGB-PSO Predictor for Pharmaceutical Degradation by Various Photocatalysts")
st.markdown(
    "This app predicts **Degradation (%)** of various pharmaceuticals by photocatalysts using trained **XGBoost + PSO** model."
)

with st.expander("Input format and notes", expanded=False):
    st.markdown(
        """
- Please enter all inputs using the same units applied during model training.
- Oxidant = 1 indicates the presence of oxidant, whereas Oxidant = 2 indicates no oxidant.
- Light source codes: 1 = UV, 2 = Visible, 3 = Simulated solar light.
- The application predicts **Degradation (%)** based on the trained XGB-PSO model.

        """
    )

# =========================
# Input form
# =========================
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        bet = st.number_input(
            "BET specific surface area (m² g⁻¹)",
            min_value=0.0,
            value=100.0,
            step=0.1,
            format="%.4f",
        )
        oxidant = st.number_input(
            "Oxidant (numeric code)",
            value=1.0,
            step=1.0,
            format="%.0f",
        )
        cox = st.number_input(
            "Oxidant concentration (mM)",
            min_value=0.0,
            value=1.0,
            step=0.1,
            format="%.4f",
        )
        mw = st.number_input(
            "Molecular Weight (g/mol)",
            min_value=0.0,
            value=100.0,
            step=0.1,
            format="%.4f",
        )
        hbdc = st.number_input(
            "HBDC",
            min_value=0.0,
            value=0.0,
            step=0.1,
            format="%.4f",
        )
        hbac = st.number_input(
            "HBAC",
            min_value=0.0,
            value=0.0,
            step=0.1,
            format="%.4f",
        )

    with col2:
        tpsa = st.number_input(
            "Topological Polar Surface Area (Å²)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            format="%.4f",
        )
        c0 = st.number_input(
            "Initial concentration of pollutant, C₀ (mg/L)",
            min_value=0.0,
            value=10.0,
            step=0.1,
            format="%.4f",
        )
        ph = st.number_input(
            "Solution pH",
            min_value=0.0,
            value=7.0,
            step=0.1,
            format="%.4f",
        )
        light = st.number_input(
            "Light source (numeric code)",
            value=1.0,
            step=1.0,
            format="%.0f",
        )
        cphotocat = st.number_input(
            "Catalyst dosage (mg/L)",
            min_value=0.0,
            value=100.0,
            step=0.1,
            format="%.4f",
        )
        tmin = st.number_input(
            "t (min)",
            min_value=0.0,
            value=60.0,
            step=1.0,
            format="%.4f",
        )

    submitted = st.form_submit_button("Predict Degradation (%)")

# =========================
# Prediction
# =========================
feature_columns = [
    "BET specific surface area (m² g⁻¹)",
    "Oxidant",
    "Oxidant concentration (mM)",
    "Molecular Weight (g/mol)",
    "HBDC",
    "HBAC",
    "Topological Polar Surface Area (Å²)",
    "Initial concentration of pollutant (mg/L)",
    "Solution pH",
    "Light source",
    "Catalyst dosage (mg/L)",
    "t (min)",
]

if submitted:
    input_data = pd.DataFrame(
        [[
            bet,
            oxidant,
            cox,
            mw,
            hbdc,
            hbac,
            tpsa,
            c0,
            ph,
            light,
            cphotocat,
            tmin,
        ]],
        columns=feature_columns,
    )

    prediction = float(model.predict(input_data)[0])
    prediction = max(0.0, min(100.0, prediction))

    st.success(f"Predicted Degradation: {prediction:.2f}%")

    with st.expander("Show input data used for prediction"):
        st.dataframe(input_data, use_container_width=True)

# # =========================
# # Sidebar info
# # =========================
# st.sidebar.header("Deployment")
# st.sidebar.code(
#     "pip install streamlit joblib xgboost pandas numpy\n"
#     "streamlit run xgb_pso_streamlit_app.py",
#     language="bash",
# )

st.sidebar.header("Model Information")

st.sidebar.write(
"""
**Model:** XGBoost optimized with Particle Swarm Optimization (PSO)

**Inputs:**
- BET specific surface area (m² g⁻¹)
- Oxidant
- Oxidant concentration (mM)
- Molecular Weight (g/mol)
- HBDC
- HBAC
- Topological Polar Surface Area (Å²)
- Initial pollutant concentration (mg/L)
- Solution pH
- Light source
- Catalyst dosage (mg/L)
- Reaction time (min)

**Output:**
- Predicted Degradation (%)

---

**Developer:** **Amir Dashti**  
School of Civil and Environmental Engineering  
**University of Technology Sydney**  
Email: amirdashti13681990@gmail.com
"""
)
