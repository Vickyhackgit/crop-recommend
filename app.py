# -*- coding: utf-8 -*-
"""Crop Residue Industry Prediction System
Extended Version with Auto Residue Calculation and Recommendation per Type
"""

# === Imports ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === Load Trained Model ===
@st.cache_resource
def load_model():
    data = joblib.load("crop_residue_model.joblib")
    return data['model'], data['encoders'], data['feature_names']

model, encoders, feature_names = load_model()

# === Reference: Crop to Residue Info ===
CROP_RESIDUE_INFO = {
    'Wheat': {
        'residue_to_crop_ratio': 1.5,
        'residue_distribution': {
            'Straw': 0.80,
            'Husk': 0.20
        }
    },
    'Rice': {
        'residue_to_crop_ratio': 1.7,
        'residue_distribution': {
            'Straw': 0.90,
            'Chaff': 0.10
        }
    },
    'Maize': {
        'residue_to_crop_ratio': 1.2,
        'residue_distribution': {
            'Stover': 0.50,
            'Cobs': 0.30,
            'Leaves': 0.20
        }
    },
    'Sugarcane': {
        'residue_to_crop_ratio': 0.4,
        'residue_distribution': {
            'Bagasse': 0.60,
            'Trash': 0.30,
            'Tops': 0.10
        }
    },
    'Cotton': {
        'residue_to_crop_ratio': 3.0,
        'residue_distribution': {
            'Stalks': 0.70,
            'Boll Shells/Husks': 0.30
        }
    }
}

# === Streamlit Interface ===
st.title("\U0001F33E Crop Residue to Industry Recommendation System")

# Manual Input
st.sidebar.header("Enter Basic Farm Data")
crop_type = st.sidebar.selectbox("Crop Type", list(CROP_RESIDUE_INFO.keys()))
area = st.sidebar.number_input("Area (in hectares)", value=50.0)
production = st.sidebar.number_input("Crop Production (tons)", value=250.0)
farm_id = st.text_input("Farm ID", "F1001")

# Extra Parameters
input_data = {
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

# Calculate Yield and Residue
if crop_type in CROP_RESIDUE_INFO:
    ratio = CROP_RESIDUE_INFO[crop_type]['residue_to_crop_ratio']
    total_residue = production * ratio
    yield_ha = production / area
    input_data['Yield_TonsPerHa'] = yield_ha
    input_data['Residue-to-Crop Ratio'] = ratio
    input_data['Total_Residue_Generated'] = total_residue

    st.subheader("\U0001F4CA Estimated Residue Breakdown")
    st.write(f"Yield/ha: **{yield_ha:.2f} tons/ha**, Residue Ratio: **{ratio}**, Total: **{total_residue:.2f} tons**")

    residue_qty = {}
    for rtype, perc in CROP_RESIDUE_INFO[crop_type]['residue_distribution'].items():
        residue_qty[rtype] = total_residue * perc
    st.bar_chart(pd.Series(residue_qty))
else:
    st.warning("Residue info not found for selected crop.")

# Predict for Each Residue Type
if st.button("\U0001F52E Predict Industry Allocation"):
    st.subheader("\U0001F3ED Industry Recommendations")
    for res_type, qty in residue_qty.items():
        modified_input = input_data.copy()
        modified_input['Residue_Type'] = res_type
        df = pd.DataFrame([modified_input])

        for col in ['Crop_Type', 'Residue_Type', 'Harvest_Season', 'Storage_Condition']:
            df[col] = encoders[col].transform(df[col])
        for f in feature_names:
            if f not in df.columns:
                df[f] = 0
        df = df[feature_names]

        try:
            probs = model.predict_proba(df)[0]
            best_idx = np.argmax(probs)
            best_industry = encoders['Industry'].classes_[best_idx]
            conf = probs[best_idx]

            st.success(f"**{qty:.2f} tons {res_type}** → **{best_industry}** ({conf:.1%} confidence)")

            prob_df = pd.DataFrame({
                'Industry': encoders['Industry'].classes_,
                'Probability': probs
            }).sort_values(by='Probability', ascending=False)
            st.bar_chart(prob_df.set_index('Industry'))

        except Exception as e:
            st.error(f"Prediction error for {res_type}: {e}")
