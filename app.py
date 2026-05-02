import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data_processed.csv")
recon_df = pd.read_csv("reconstruction_error.csv")

# Page config
st.set_page_config(page_title="IoT Anomaly Detection", layout="wide")

# Title
st.title("Time Series Anomaly Detection — IoT Bearing Sensors")
st.markdown("Detecting industrial equipment failure **2,000+ time steps before breakdown** using unsupervised ML on NASA bearing sensor data.")

# Metrics
st.subheader("Results at a Glance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Readings", "2,156")
col2.metric("Isolation Forest Anomalies", "108")
col3.metric("Autoencoder Anomalies", "108")
col4.metric("Cross-Model Agreement", "51 points")

st.markdown("---")

# Plot 1 — Sensor Behavior
st.subheader("Bearing Sensor Degradation Over Time")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df["time_index"], df["std"], color="steelblue")
ax1.set_xlabel("Time")
ax1.set_ylabel("Standard Deviation")
ax1.set_title("Bearing Sensor Behavior Over Time")
st.pyplot(fig1)

# Plot 2 — Isolation Forest
st.subheader("Isolation Forest — Anomaly Detection")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(df["time_index"], df["std"], color="steelblue", label="STD")
ax2.scatter(
    df[df["anomaly_iforest"] == 1]["time_index"],
    df[df["anomaly_iforest"] == 1]["std"],
    color="red", label="Anomaly", zorder=5
)
ax2.set_xlabel("Time")
ax2.set_ylabel("STD")
ax2.legend()
st.pyplot(fig2)

# Plot 3 — Autoencoder
st.subheader("Autoencoder — Reconstruction Error")
fig3, ax3 = plt.subplots(figsize=(10, 4))
ax3.plot(recon_df["time_index"], recon_df["reconstruction_error"], color="steelblue", label="Reconstruction Error")
ax3.axhline(recon_df["threshold"].iloc[0], color="red", linestyle="--", label="Threshold")
ax3.set_xlabel("Time")
ax3.set_ylabel("Error")
ax3.legend()
st.pyplot(fig3)

st.markdown("---")

# Business Impact
st.subheader("Business Impact")
st.markdown("""
- First anomaly detected at time step **~180–230**
- Catastrophic failure occurs at time step **~2,156**
- **2,000+ time steps of early warning** — enough to schedule maintenance and prevent breakdown
- Enables **predictive maintenance** — replacing reactive fixes with data-driven intervention
""")

st.markdown("---")
st.caption("Built by Akilrahman | NASA Bearing Dataset | Isolation Forest + Autoencoder")