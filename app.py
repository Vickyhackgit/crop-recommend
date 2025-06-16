# app.py - Crop Residue to Industry Prediction with Pie + Bar Chart Output

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

# === Reference Residue Info ===
CROP_RESIDUE_INFO = {
    'Wheat': {'residue_to_crop_ratio': 0.92, 'residue_distribution': {'Straw': 0.85, 'Husk': 0.10, 'Stalks': 0.05}},
    'Rice': {'residue_to_crop_ratio': 0.4572, 'residue_distribution': {'Straw': 0.90, 'Chaff/Stalks': 0.10}},
    'Sugarcane': {'residue_to_crop_ratio': 0.1425, 'residue_distribution': {'Bagasse': 0.60, 'Trash': 0.30, 'Tops': 0.10}},
    'Cotton': {'residue_to_crop_ratio': 0.5679, 'residue_distribution': {'Stalks': 0.70, 'Boll Shells/Husks': 0.30}},
    'Maize': {'residue_to_crop_ratio': 0.0846, 'residue_distribution': {'Stover': 0.50, 'Cobs': 0.30, 'Leaves': 0.20}}
}

# === Title ===
st.title("Crop Residue to Industry Recommendation System")

# === Input Method ===
st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose input method:", ["Manual Entry", "Upload CSV/JSON"])

if input_method == "Manual Entry":
    st.subheader("Enter Farm & Production Details")
    farm_id = st.text_input("Farm ID", "F1001")
    crop_type = st.selectbox("Crop Type", list(CROP_RESIDUE_INFO.keys()))
    production = st.number_input("Crop Production (tons)", min_value=1.0, value=250.0)
    area = st.number_input("Area (hectares)", min_value=1.0, value=50.0)

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
            st.subheader("Residue Estimation")
            ratio = CROP_RESIDUE_INFO[crop_type]['residue_to_crop_ratio']
            total_residue = production * ratio
            yield_per_ha = production / area
            st.write(f"**Yield per hectare:** {yield_per_ha:.2f} tons/ha")
            st.write(f"**Total Residue Generated:** {total_residue:.2f} tons (Ratio: {ratio})")

            residue_qty = {
                r_type: total_residue * r_pct
                for r_type, r_pct in CROP_RESIDUE_INFO[crop_type]['residue_distribution'].items()
            }

            # PIE CHART
            fig1, ax1 = plt.subplots()
            labels = [f"{k} ({v:.1f}t)" for k, v in residue_qty.items()]
            ax1.pie(residue_qty.values(), labels=labels, autopct='%1.1f%%', startangle=90,
                    colors=sns.color_palette("bright"))
            ax1.axis('equal')
            ax1.set_title("Residue Type Distribution")
            st.pyplot(fig1)

            # === Industry Recommendation ===
            industry_allocation = []

            for res_type, qty in residue_qty.items():
                sample = input_features.copy()
                sample['Residue_Type'] = res_type
                df = pd.DataFrame([sample])

                for col in ['Crop_Type', 'Residue_Type', 'Harvest_Season', 'Storage_Condition']:
                    if df.at[0, col] not in encoders[col].classes_:
                        st.error(f"Invalid value for {col}: {df.at[0, col]}")
                        st.stop()
                    df[col] = encoders[col].transform(df[col])

                for f in feature_names:
                    if f not in df.columns:
                        df[f] = 0
                df = df[feature_names]

                probs = model.predict_proba(df)[0]
                best_idx = np.argmax(probs)
                best_industry = encoders['Industry'].classes_[best_idx]
                industry_allocation.append((res_type, best_industry, qty))

            # Grouped Allocation Summary
            df_alloc = pd.DataFrame(industry_allocation, columns=['Residue', 'Industry', 'Quantity_tons'])
            industry_totals = df_alloc.groupby('Industry')['Quantity_tons'].sum().sort_values(ascending=False)

            st.subheader("Industry Allocation Summary")
            fig2, ax2 = plt.subplots()
            sns.barplot(x=industry_totals.values, y=industry_totals.index, ax=ax2, palette="crest")
            ax2.set_xlabel("Total Residue Allocated (tons)")
            ax2.set_title("Recommended Industry Allocation")
            for i, v in enumerate(industry_totals.values):
                ax2.text(v + 1, i, f"{v:.1f}t", va='center')
            st.pyplot(fig2)

        else:
            st.warning("⚠️ Selected crop not found in reference data.")

elif input_method == "Upload CSV/JSON":
    uploaded_file = st.file_uploader("Upload a CSV or JSON file", type=["csv", "json"])
    if uploaded_file:
        try:
            df_input = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_json(uploaded_file)
            st.success("File uploaded successfully.")
            st.dataframe(df_input)
        except Exception as e:
            st.error(f"Error reading file: {e}")
