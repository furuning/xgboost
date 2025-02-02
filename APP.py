import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Load the fetal state model
model = joblib.load('XGBoost.pkl')

# Define feature names
feature_names = [
    "PDW", "MPV", "MCV", "E0S", "PLT", "MON", "RDW"
]

# Streamlit user interface
st.title("Henoch-Schönlein purpura Predictor")

# Input features
PDW = st.number_input("Platelet Distribution Width (PDW):", min_value=14.70, max_value=18.0, value=15.00)
MPV = st.number_input("Mean Platelet Volume (MPV):", min_value=6.50, max_value=14.40, value=10.00)
MCV = st.number_input("Mean Corpuscular Volume (MCV):", min_value=57.40, max_value=103.60, value=80.00)
EOS = st.number_input("Eosinophils Absolute Value (EOS):", min_value=0.00, max_value=1.80, value=0.15)
PLT = st.number_input("Platelet Count (PLT):", min_value=3, max_value=820, value=300)
MON = st.number_input("Monocytes Absolute Value (MON):", min_value=0.02, max_value=2.51, value=1.30)
RDW = st.number_input("Red Cell Distribution Width-Standard Deviation (RDW-SD):", min_value=11.00, max_value=22.20, value=15.00)

# Collect input values into a list
feature_values = [PDW,MPV,MCV,EOS,PLT,MON,RDW]

# Convert feature values into a DataFrame
features_df = pd.DataFrame([feature_values], columns=feature_names)

if st.button("Predict"):
    # Predict class and probabilities using DataFrame
    predicted_class = model.predict(features_df)[0]
    predicted_proba = model.predict_proba(features_df)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 0:
        advice = (
            f"According to our model, the child is in an normal. "
            f"The model predicts that the child has a {probability:.1f}% probability of being normal. "
            "It is recommended to continue monitoring the child's health regularly."
        )
    else:
        advice = (
            f"According to our model, the child is in an abnormal state. "
            f"The model predicts that the child has a {probability:.1f}% probability of being abnormal. "
            "It is strongly advised to seek immediate medical attention for further evaluation."
        )
   
    st.write(advice)

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values_Explanation = explainer(features_df)

    # Display SHAP waterfall plot only for the predicted class
    plt.figure(figsize=(10, 5), dpi=1200)
    shap.plots.waterfall(shap_values_Explanation[0], show=False, max_display=13)
    plt.savefig("shap_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_plot.png")
