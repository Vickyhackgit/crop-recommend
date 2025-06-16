

# app.py - Streamlit Deployment Code for Crop Residue Industry Prediction

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# === Load ML Model ===
@st.cache_resource
def load_model():
    data = joblib.load("crop_residue_model.joblib")
    return data['model'], data['encoders'], data['feature_names']

model, encoders, feature_names = load_model()

# === Reference Residue Info ===
CROP_RESIDUE_INFO = {
    'Wheat': {'residue_to_crop_ratio': 1.5, 'residue_distribution': {'Straw': 0.80, 'Husk': 0.20}},
    'Rice': {'residue_to_crop_ratio': 1.7, 'residue_distribution': {'Straw': 0.90, 'Chaff': 0.10}},
    'Maize': {'residue_to_crop_ratio': 1.2, 'residue_distribution': {'Stover': 0.50, 'Cobs': 0.30, 'Leaves': 0.20}},
    'Sugarcane': {'residue_to_crop_ratio': 0.4, 'residue_distribution': {'Bagasse': 0.60, 'Trash': 0.30, 'Tops': 0.10}},
    'Cotton': {'residue_to_crop_ratio': 3.0, 'residue_distribution': {'Stalks': 0.70, 'Boll Shells/Husks': 0.30}}
}

# === App Title ===
st.title("\U0001F33E Crop Residue to Industry Recommendation System")

# === Input Type ===
st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose input method:", ["Manual Entry", "Upload CSV/JSON"])

# === Manual Entry ===
if input_method == "Manual Entry":
    st.subheader("Enter Farm & Residue Details")
    farm_id = st.text_input("Farm ID", "F1001")
    crop_type = st.selectbox("Crop Type", list(CROP_RESIDUE_INFO.keys()))
    production = st.number_input("Crop Production (tons)", min_value=1.0, value=100.0)
    area = st.number_input("Area (hectares)", min_value=1.0, value=20.0)

    input_features = {
        'Farm_ID': farm_id,
        'Crop_Type': crop_type,
        'Moisture_pct': st.slider("Moisture %", 0.0, 100.0, 12.5),
        'Cellulose_pct': st.slider("Cellulose %", 0.0, 100.0, 38.0),
        'CN_Ratio': st.slider("C:N Ratio", 0.0, 150.0, 80.0),
        'Calorific_MJ_kg': st.slider("Calorific Value (MJ/kg)", 0.0, 50.0, 16.8),
        'Lignin_pct': st.slider("Lignin %", 0.0, 100.0, 15.2),
        'Nitrogen_pct': st.slider("Nitrogen %", 0.0, 100.0, 0.8),
        'Silica_pct': st.slider("Silica %", 0.0, 100.0, 6.0),
        'Ash_pct': st.slider("Ash %", 0.0, 100.0, 8.1),
        'Bulk_Density': st.slider("Bulk Density", 0.0, 2.0, 0.45),
        'Harvest_Season': st.selectbox("Harvest Season", ["Autumn", "Winter", "Summer"]),
        'Storage_Condition': st.selectbox("Storage Condition", ["Covered", "Open"]),
        'Transportation_Distance_km': st.slider("Transport Distance (km)", 0, 500, 30),
        'Local_Market_Price': st.slider("Local Market Price", 0, 5000, 125),
        'Residue_Age_days': st.slider("Residue Age (days)", 0, 365, 35)
    }

    if st.button("Predict Suitable Industries for Residues"):
        if crop_type in CROP_RESIDUE_INFO:
            st.subheader("Estimated Residue & Recommendations")
            ratio = CROP_RESIDUE_INFO[crop_type]['residue_to_crop_ratio']
            total_residue = production * ratio
            st.write(f"Residue-to-Crop Ratio: **{ratio}**, Total Residue: **{total_residue:.2f} tons**")

            residue_qty = {
                res_type: total_residue * pct
                for res_type, pct in CROP_RESIDUE_INFO[crop_type]['residue_distribution'].items()
            }
            st.bar_chart(pd.Series(residue_qty))

            for res_type, qty in residue_qty.items():
                sample = input_features.copy()
                sample['Residue_Type'] = res_type
                df = pd.DataFrame([sample])

                for col in ['Crop_Type', 'Residue_Type', 'Harvest_Season', 'Storage_Condition']:
                    df[col] = encoders[col].transform(df[col])
                for f in feature_names:
                    if f not in df.columns:
                        df[f] = 0
                df = df[feature_names]

                try:
                    probs = model.predict_proba(df)[0]
                    idx = np.argmax(probs)
                    industry = encoders['Industry'].classes_[idx]
                    confidence = probs[idx]

                    st.success(f"{qty:.2f} tons of **{res_type}** → **{industry}** ({confidence:.1%} confidence)")

                    prob_df = pd.DataFrame({
                        'Industry': encoders['Industry'].classes_,
                        'Probability': probs
                    })
                    st.bar_chart(prob_df.set_index("Industry"))

                except Exception as e:
                    st.error(f"Prediction error for {res_type}: {e}")
        else:
            st.warning("Residue mapping not available for selected crop.")

# === Upload Option (optional for future use) ===
elif input_method == "Upload CSV/JSON":
    uploaded_file = st.file_uploader("Upload a single-row CSV or JSON", type=["csv", "json"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_json(uploaded_file)
            st.success("File uploaded successfully!")
            st.dataframe(df_input)
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.warning("Please upload a file to continue.")
