"""
KOBO DROUGHT EARLY WARNING SYSTEM
Streamlit dashboard (Phase 26)

Run locally:
    streamlit run dashboard.py

Expects a `data/` folder alongside this file containing everything saved
from the Colab notebook's Milestone 16 export cells:
    kobo_model.pkl
    kobo_spi_params.pkl
    kobo_climatology.pkl
    kobo_dashboard_history.csv
    kobo_grid_performance.geojson
    kobo_latest_status.json
"""

import json
import pickle
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(
    page_title="Kobo Drought Early Warning System",
    page_icon="🌾",
    layout="wide",
)

# ---------------------------------------------------------------
# Load everything (cached so it only runs once per session)
# ---------------------------------------------------------------
@st.cache_data
def load_status():
    with open(DATA_DIR / "kobo_latest_status.json") as f:
        return json.load(f)


@st.cache_data
def load_history():
    df = pd.read_csv(DATA_DIR / "kobo_dashboard_history.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_grid():
    return gpd.read_file(DATA_DIR / "kobo_grid_performance.geojson")


try:
    status = load_status()
    history = load_history()
    grid = load_grid()
    data_missing = False
except FileNotFoundError as e:
    data_missing = True
    missing_file = str(e)

if data_missing:
    st.error(
        "Dashboard data not found. Make sure the `data/` folder (exported from the "
        "Colab notebook's Milestone 16 cells) sits alongside this file.\n\n"
        f"Details: {missing_file}"
    )
    st.stop()

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.title("🌾 KOBO DROUGHT EARLY WARNING SYSTEM")
st.caption("Raya Kobo Woreda, North Wollo Zone, Amhara Region, Ethiopia")

# ---------------------------------------------------------------
# Phase 26.3: Main status
# ---------------------------------------------------------------
status_colors = {"Normal": "#2ecc71", "Watch": "#f39c12", "Warning": "#e74c3c"}
status_color = status_colors.get(status["status"], "#95a5a6")

col1, col2, col3 = st.columns([1.2, 1, 1])

with col1:
    st.markdown(
        f"""
        <div style="background-color:{status_color}22; border:2px solid {status_color};
                    border-radius:12px; padding:20px; text-align:center;">
            <div style="font-size:14px; color:#666;">CURRENT STATUS</div>
            <div style="font-size:36px; font-weight:bold; color:{status_color};">
                {status['status'].upper()}
            </div>
            <div style="font-size:14px; color:#666; margin-top:8px;">
                {status['current_class']} drought class — {status['current_confidence']*100:.0f}% model confidence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.metric(
        "15-day rainfall forecast",
        f"{status['forecast_pct_of_normal']:.0f}% of normal",
        help="ECMWF AIFS deterministic forecast, compared to climatological expectation for this period.",
    )

with col3:
    probs = status["class_probabilities"]
    st.write("**Class probabilities**")
    for cls in ["Normal", "Moderate", "Severe"]:
        st.progress(probs.get(cls, 0), text=f"{cls}: {probs.get(cls, 0)*100:.0f}%")

st.info(f"📋 {status['forecast_note']}")

st.divider()

# ---------------------------------------------------------------
# Phase 26.8: Data currency — keep the interface honest
# ---------------------------------------------------------------
with st.expander("ℹ️ Data currency (click to expand) — different sources update at different speeds"):
    c1, c2, c3 = st.columns(3)
    c1.metric("NDVI (vegetation)", status.get("ndvi_date", "unknown"))
    c2.metric("LST (temperature)", status.get("lst_date", "unknown"))
    c3.metric("Rainfall / soil moisture", status.get("rain_soil_date", "unknown"))
    st.caption(
        "Satellite and reanalysis products have real processing lags that vary by source. "
        "This status reflects the most recent available reading from each — not necessarily today."
    )

# ---------------------------------------------------------------
# Phase 26.4: Current indicators
# ---------------------------------------------------------------
st.subheader("Current indicators")
vec = status["current_vector"]
ind_cols = st.columns(5)
indicator_display = [
    ("NDVI anomaly", vec.get("ndvi_anomaly_z"), "σ"),
    ("VCI", vec.get("vci"), ""),
    ("Rainfall anomaly", vec.get("rain_anomaly_z"), "σ"),
    ("LST anomaly", vec.get("lst_anomaly_z"), "σ"),
    ("Soil moisture anomaly", vec.get("soil_anomaly_z"), "σ"),
]
for col, (label, val, unit) in zip(ind_cols, indicator_display):
    if val is not None:
        col.metric(label, f"{val:.2f}{unit}")
    else:
        col.metric(label, "N/A")

st.divider()

# ---------------------------------------------------------------
# Phase 26.5: Spatial risk map
# ---------------------------------------------------------------
st.subheader("Spatial risk pattern across Kobo")
st.caption(
    "Model accuracy by grid cell (validation period). Drought *frequency* is fairly uniform "
    "across Kobo by construction (SPI is normalized per cell) — what varies spatially is how "
    "reliably the model can predict it."
)

geojson = json.loads(grid.to_json())
fig_map = px.choropleth_mapbox(
    grid,
    geojson=geojson,
    locations=grid.index,
    color="accuracy",
    color_continuous_scale="RdYlGn",
    range_color=(0.7, 1.0),
    mapbox_style="carto-positron",
    zoom=8.5,
    center={"lat": 12.1, "lon": 39.65},
    opacity=0.75,
    labels={"accuracy": "Model accuracy"},
)
fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=450)
st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# ---------------------------------------------------------------
# Phase 26.6: Forecast panel
# ---------------------------------------------------------------
st.subheader("Forecast breakdown")
fw = status.get("forecast_windows", {})
if fw:
    fc1, fc2, fc3 = st.columns(3)
    for col, (label, key_mm, key_pct) in zip(
        [fc1, fc2, fc3],
        [
            ("1-3 days", "1_3_day_mm", "1_3_day_pct_normal"),
            ("4-7 days", "4_7_day_mm", "4_7_day_pct_normal"),
            ("8-15 days", "8_15_day_mm", "8_15_day_pct_normal"),
        ],
    ):
        pct = fw.get(key_pct, 0)
        delta_color = "inverse" if pct < 80 else "normal"
        col.metric(
            f"{label} forecast",
            f"{fw.get(key_mm, 0):.1f} mm",
            f"{pct:.0f}% of normal",
            delta_color=delta_color,
        )
    st.caption(
        "Source: ECMWF AIFS deterministic (aifs-single) open forecast, total precipitation (tp). "
        "This is a single deterministic run, not an ensemble — treat it as one plausible scenario, "
        "not a certainty."
    )
else:
    st.write("Forecast data not available in this export.")

st.divider()

# ---------------------------------------------------------------
# Phase 26.7: Explainability
# ---------------------------------------------------------------
st.subheader("What's driving this prediction")
drivers = status.get("top_drivers", [])
if drivers:
    drivers_df = pd.DataFrame(drivers)
    drivers_df["contribution"] = drivers_df["coefficient"] * drivers_df["value_z"]
    fig_drivers = go.Figure(
        go.Bar(
            x=drivers_df["contribution"],
            y=drivers_df["feature"],
            orientation="h",
            marker_color=["#e74c3c" if c < 0 else "#2ecc71" for c in drivers_df["contribution"]],
        )
    )
    fig_drivers.update_layout(
        xaxis_title=f"Contribution to '{status['current_class']}' class logit",
        height=300,
        margin={"t": 10},
    )
    st.plotly_chart(fig_drivers, use_container_width=True)
    st.caption(
        "⚠️ This is multinomial logistic regression — a feature's contribution to one class's score "
        "only tells part of the story, since it's the *relative* comparison across all three classes "
        "(Normal/Moderate/Severe) that determines the final prediction. Treat this as a rough guide "
        "to influence, not a simple one-directional causal explanation."
    )
else:
    st.write("Driver data not available in this export.")

st.divider()

# ---------------------------------------------------------------
# Historical time series
# ---------------------------------------------------------------
st.subheader("Historical record (2000-2024)")
fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(x=history["date"], y=history["ndvi_anomaly_z"], name="NDVI anomaly (z)", line=dict(color="green")))
fig_hist.add_trace(go.Scatter(x=history["date"], y=history["spi_6"], name="SPI-6", line=dict(color="steelblue")))

for _, row in history[history["drought_class"] == "Severe"].iterrows():
    fig_hist.add_vrect(x0=row["date"], x1=row["date"] + pd.DateOffset(months=1), fillcolor="red", opacity=0.12, line_width=0)
for _, row in history[history["drought_class"] == "Moderate"].iterrows():
    fig_hist.add_vrect(x0=row["date"], x1=row["date"] + pd.DateOffset(months=1), fillcolor="orange", opacity=0.08, line_width=0)

fig_hist.update_layout(height=350, margin={"t": 10}, legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig_hist, use_container_width=True)
st.caption("Shaded bands: classified drought months (red = Severe, orange = Moderate), from SPI-6.")

st.divider()
st.caption(
    "Kobo Drought Early Warning System — a portfolio project. Not an official or operational "
    "warning service. Built with MODIS NDVI/LST, CHIRPS, ERA5-Land, and ECMWF AIFS via Google Earth "
    "Engine, scikit-learn, and Streamlit."
)
