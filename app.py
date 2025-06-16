import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_resource
def load_model():
    data = joblib.load("crop_residue_model.joblib")
    return data["model"], data["encoders"], data["feature_names"]

model, encoders, feature_names = load_model()

# Crop reference: ratio + residue types
CROP_RESIDUE_INFO = {
    "Wheat": {"residue_to_crop_ratio": 0.92, "residue_distribution": {"Straw": 0.85, "Husk": 0.10, "Stalks": 0.05}},
    "Rice": {"residue_to_crop_ratio": 0.4572, "residue_distribution": {"Straw": 0.90, "Chaff/Stalks": 0.10}},
    "Sugarcane": {"residue_to_crop_ratio": 0.1425, "residue_distribution": {"Bagasse": 0.60, "Trash": 0.30, "Tops": 0.10}},
    "Cotton": {"residue_to_crop_ratio": 0.5679, "residue_distribution": {"Stalks": 0.70, "Boll Shells/Husks": 0.30}},
    "Maize": {"residue_to_crop_ratio": 0.0846, "residue_distribution": {"Stover": 0.50, "Cobs": 0.30, "Leaves": 0.20}},
}

st.title("🌾 Crop Residue to Industry Recommendation System")

st.sidebar.header("Input")
crop = st.sidebar.selectbox("Select Crop", list(CROP_RESIDUE_INFO.keys()))
prod = st.sidebar.number_input("Production (tons)", min_value=1.0, value=250.0)
area = st.sidebar.number_input("Area (ha)", min_value=1.0, value=50.0)

# Fixed values / placeholder for simplicity
base_features = {
    "Moisture_pct": 12.5,
    "Cellulose_pct": 38.0,
    "CN_Ratio": 80.0,
    "Calorific_MJ_kg": 16.8,
    "Lignin_pct": 15.2,
    "Nitrogen_pct": 0.8,
    "Silica_pct": 6.0,
    "Ash_pct": 8.1,
    "Bulk_Density": 0.45,
    "Harvest_Season": "Autumn",
    "Storage_Condition": "Covered",
    "Transportation_Distance_km": 30,
    "Local_Market_Price": 125,
    "Residue_Age_days": 35,
}

if st.button("Predict Industry Allocation"):
    ratio = CROP_RESIDUE_INFO[crop]["residue_to_crop_ratio"]
    residue_dist = CROP_RESIDUE_INFO[crop]["residue_distribution"]
    total_residue = prod * ratio
    st.write(f"Total Estimated Residue: **{total_residue:.2f} tons**")

    # Pie chart
    residue_breakdown = {k: total_residue * v for k, v in residue_dist.items()}
    fig1, ax1 = plt.subplots()
    ax1.pie(residue_breakdown.values(), labels=[f"{k} ({v:.1f}t)" for k, v in residue_breakdown.items()],
            autopct="%1.1f%%", colors=sns.color_palette("bright"), startangle=90)
    ax1.axis("equal")
    ax1.set_title("Residue Breakdown by Type")
    st.pyplot(fig1)

    # Predict best-fit industry for each residue
    industry_alloc = []

    for residue_type, qty in residue_breakdown.items():
        input_data = base_features.copy()
        input_data.update({
            "Crop_Type": crop,
            "Residue_Type": residue_type,
            "Farm_ID": "F001",
        })
        df = pd.DataFrame([input_data])
        for col in ["Crop_Type", "Residue_Type", "Harvest_Season", "Storage_Condition"]:
            df[col] = encoders[col].transform(df[col])
        for f in feature_names:
            if f not in df.columns:
                df[f] = 0
        df = df[feature_names]
        pred_proba = model.predict_proba(df)[0]
        pred_industry = encoders["Industry"].classes_[np.argmax(pred_proba)]
        industry_alloc.append((pred_industry, qty))

    # Aggregate allocations
    df_industry = pd.DataFrame(industry_alloc, columns=["Industry", "Tons"])
    df_summary = df_industry.groupby("Industry").sum().sort_values("Tons", ascending=False)

    # Show final bar chart
    st.subheader("📊 Residue Allocation to Industries")
    fig2, ax2 = plt.subplots()
    sns.barplot(x=df_summary["Tons"], y=df_summary.index, ax=ax2, palette="crest")
    for i, v in enumerate(df_summary["Tons"]):
        ax2.text(v + 0.5, i, f"{v:.1f}t", va="center")
    ax2.set_xlabel("Allocated Residue (tons)")
    ax2.set_ylabel("Industry")
    ax2.set_title("Recommended Industry Allocation (All Residues)")
    st.pyplot(fig2)
