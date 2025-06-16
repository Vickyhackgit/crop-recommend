# app.py - Crop Residue to Industry Prediction (Final with Encoder Check)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# === Load ML Model ===
@st.cache_resource
def load_model():
    data = joblib.load("crop_residue_model.joblib")
    return data['model'], data['encoders'], data['feature_names']

model, encoders, feature_names = load_model()

# === Crop-Residue Reference ===
CROP_RESIDUE_INFO = {
    'Wheat': {'residue_to_crop_ratio': 1.5, 'residue_distribution': {'Straw': 0.80, 'Husk': 0.20}},
    'Rice': {'residue_to_crop_ratio': 1.7, 'residue_distribution': {'Straw': 0.90, 'HUsk': 0.10}},
    'Maize': {'residue_to_crop_ratio': 1.2, 'residue_distribution': {'Stover': 0.50, 'Cobs': 0.30, 'Husk': 0.20}},
    'Sugarcane': {'residue_to_crop_ratio': 0.4, 'residue_distribution': {'Bagasse': 0.60, 'Straw': 0.40}},
    'Cotton': {'residue_to_crop_ratio': 3.0, 'residue_distribution': {'Straw': 0.70, 'Husks': 0.30}}
}

st.title("Crop Residue to Industry Recommendation System")

st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose input method:", ["Manual Entry", "Upload CSV/JSON"])

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
        'Harvest_Season': st.selectbox("Harvest Season", list(encoders['Harvest_Season'].classes_)),
        'Storage_Condition': st.selectbox("Storage Condition", list(encoders['Storage_Condition'].classes_)),
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

            # Pie Chart of Residue
            fig1, ax1 = plt.subplots()
            ax1.pie(residue_qty.values(), labels=[f"{k} ({v:.1f}t)" for k, v in residue_qty.items()], autopct='%1.1f%%',
                    startangle=90, colors=sns.color_palette("bright"))
            ax1.axis('equal')
            ax1.set_title("Residue Type Distribution")
            st.pyplot(fig1)

            # Prediction & Allocation
            industry_results = []
            for res_type, qty in residue_qty.items():
                entry = input_features.copy()
                entry['Residue_Type'] = res_type
                df = pd.DataFrame([entry])

                for col in ['Crop_Type', 'Residue_Type', 'Harvest_Season', 'Storage_Condition']:
                    if df.at[0, col] in encoders[col].classes_:
                        df[col] = encoders[col].transform(df[col])
                    else:
                        st.error(f"❌ Value '{df.at[0, col]}' not recognized for column '{col}'. Please check your input.")
                        st.stop()

                for f in feature_names:
                    if f not in df.columns:
                        df[f] = 0
                df = df[feature_names]

                probs = model.predict_proba(df)[0]
                idx = np.argmax(probs)
                industry = encoders['Industry'].classes_[idx]
                confidence = probs[idx]
                industry_results.append((res_type, industry, qty, confidence))

            # Final Allocation
            df_result = pd.DataFrame(industry_results, columns=['Residue', 'Industry', 'Quantity_tons', 'Confidence'])
            st.write("### Residue to Industry Mapping")
            st.dataframe(df_result)

            totals = df_result.groupby("Industry")["Quantity_tons"].sum().sort_values(ascending=False)
            st.subheader("Final Industry Allocation")
            st.bar_chart(totals)

        else:
            st.warning("Selected crop type not found in database.")

elif input_method == "Upload CSV/JSON":
    uploaded_file = st.file_uploader("Upload CSV/JSON with residue details", type=["csv", "json"])
    if uploaded_file:
        try:
            df_input = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_json(uploaded_file)
            st.success("File uploaded successfully.")
            st.dataframe(df_input)
        except Exception as e:
            st.error(f"Error reading file: {e}")
