
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the model
@st.cache_resource
def load_model():
    data = joblib.load("crop_residue_model.joblib")
    return data['model'], data['encoders'], data['feature_names']

# Load crop-residue mapping from training data
@st.cache_data
def load_crop_residue_mapping():
    df = pd.read_csv("train100.csv")  # Make sure this file is present and renamed without space!
    mapping = {}

    for crop in df['Crop_Type'].unique():
        crop_df = df[df['Crop_Type'] == crop]
        residues = []
        for val in crop_df['Residue_Type']:
            if pd.notna(val):
                # Split by comma and strip whitespace
                residues.extend([x.strip() for x in str(val).split(',')])
        mapping[crop] = sorted(set(residues))

    return mapping

# Load everything
model, encoders, feature_names = load_model()
crop_to_residues = load_crop_residue_mapping()

# Title
st.title("Crop Residue to Industry Recommendation System")

# Input method
st.sidebar.header("Input Method")
input_method = st.sidebar.radio("Choose input method:", ["Manual Entry", "Upload CSV/JSON"])

# === Manual Input ===
if input_method == "Manual Entry":
    st.subheader("Enter Residue Data")

    selected_crop = st.selectbox("Crop Type", list(crop_to_residues.keys()))
    selected_residue = st.selectbox("Residue Type", crop_to_residues[selected_crop])

    input_data = {
        'Farm_ID': st.text_input("Farm ID", "F1001"),
        'Crop_Type': selected_crop,
        'Residue_Type': selected_residue,
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
    df_input = pd.DataFrame([input_data])

# === File Upload Input ===
elif input_method == "Upload CSV/JSON":
    uploaded_file = st.file_uploader("Upload a single row of farm residue data (CSV or JSON)", type=["csv", "json"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_input = pd.read_csv(uploaded_file)
            else:
                df_input = pd.read_json(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.dataframe(df_input)
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.stop()
    else:
        st.warning(" Please upload a file to continue.")
        st.stop()

# === Encode & Predict ===
def preprocess_input(df_input):
    for col in ['Crop_Type', 'Residue_Type', 'Harvest_Season', 'Storage_Condition']:
        df_input[col] = encoders[col].transform(df_input[col])
    for f in feature_names:
        if f not in df_input.columns:
            df_input[f] = 0
    return df_input[feature_names]

if st.button("Predict suitable Industry"):
    try:
        X = preprocess_input(df_input)
        probs = model.predict_proba(X)[0]
        pred_idx = np.argmax(probs)
        industry = encoders['Industry'].classes_[pred_idx]
        confidence = probs[pred_idx]

        st.success(f"✅ Recommended Industry: **{industry}**")
        st.write(f"Confidence: **{confidence:.2%}**")

        st.subheader("📊 Prediction Probabilities")
        prob_df = pd.DataFrame({
            'Industry': encoders['Industry'].classes_,
            'Probability': probs
        })
        st.bar_chart(prob_df.set_index("Industry"))

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
