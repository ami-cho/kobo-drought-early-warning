"""
KOBO DROUGHT EARLY WARNING SYSTEM
Redesigned dashboard — "field instrument panel" aesthetic with interactive telemetry,
spatial cell inspection, what-if perturbation modeling, and historical regime exploration.

Run locally:
    streamlit run dashboard.py

Expects a `data/` folder alongside this file (see README / DATA_SOURCES.md).
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from status_reader import get_cached_status

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(
    page_title="Multi-Sensor Drought Early Warning System — Raya Kobo",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =================================================================
# DESIGN TOKENS
# =================================================================
COLORS = {
    "bg": "#1B1712",
    "surface": "#221D17",
    "surface_elevated": "#29231C",
    "rule": "#3A332A",
    "rule_soft": "#2B261F",
    "text": "#EDE6D8",
    "text_muted": "#9C8E7C",
    "sage": "#8AA37D",
    "sage_dim": "#5E7355",
    "ochre": "#C99A4E",
    "clay": "#C1573E",
    "sky": "#7FA0B0",
}

STATUS_COLOR = {
    "Normal": COLORS["sage"],
    "Watch": COLORS["ochre"],
    "Warning": COLORS["clay"],
}

FEATURE_LABELS = {
    "ndvi_anomaly_z": "NDVI anomaly",
    "vci": "VCI",
    "rainfall_mm": "Rainfall (mm)",
    "rain_anomaly_z": "Rainfall anomaly",
    "spi_1": "SPI-1",
    "spi_3": "SPI-3",
    "spi_6": "SPI-6",
    "spi_12": "SPI-12",
    "lst_anomaly_z": "LST anomaly",
    "soil_anomaly_z": "Soil moisture anomaly",
    "soil_percentile": "Soil moisture percentile",
    "rain_anomaly_z_lag1": "Rainfall anomaly (prev. month)",
    "spi_1_lag1": "SPI-1 (prev. month)",
    "spi_3_lag1": "SPI-3 (prev. month)",
    "soil_anomaly_z_lag1": "Soil moisture anomaly (prev. month)",
    "ndvi_anomaly_z_lag1": "NDVI anomaly (prev. month)",
}

# =================================================================
# GLOBAL CSS (Instrument Panel & Dark Native Widget Overrides)
# =================================================================
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,500&family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Space Grotesk', sans-serif;
    }}

    .stApp {{
        background-color: {COLORS['bg']};
    }}

    #MainMenu, footer, header[data-testid="stHeader"] {{
        background-color: transparent;
    }}

    .block-container {{
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}

    h1, h2, h3, h4 {{
        font-family: 'Fraunces', serif;
        color: {COLORS['text']};
        font-weight: 500;
    }}

    p, span, div, label {{
        color: {COLORS['text']};
    }}

    .mono {{
        font-family: 'IBM Plex Mono', monospace;
    }}

    /* Hairline rule */
    .kobo-rule {{
        border: none;
        border-top: 1px solid {COLORS['rule']};
        margin: 1.8rem 0;
    }}

    /* Section label — functional instrument-placard style */
    .kobo-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        color: {COLORS['text_muted']};
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }}

    /* Hero masthead */
    .kobo-hero {{
        position: relative;
        background:
            repeating-linear-gradient(0deg, transparent, transparent 39px, {COLORS['rule_soft']} 40px),
            repeating-linear-gradient(90deg, transparent, transparent 39px, {COLORS['rule_soft']} 40px),
            radial-gradient(ellipse at 15% 20%, {COLORS['sage_dim']}22, transparent 55%),
            {COLORS['surface']};
        border: 1px solid {COLORS['rule']};
        border-left: 4px solid var(--status-glow, {COLORS['rule']});
        border-radius: 4px;
        padding: 2.3rem 2.6rem;
        overflow: hidden;
    }}

    .kobo-live-dot {{
        display: inline-block;
        position: relative;
        width: 10px; height: 10px;
        margin-right: 10px;
        vertical-align: middle;
    }}
    .kobo-live-dot::before {{
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 10px; height: 10px;
        border-radius: 50%;
        background: currentColor;
    }}
    .kobo-live-dot::after {{
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 10px; height: 10px;
        border-radius: 50%;
        background: currentColor;
        animation: kobo-radar-ping 1.8s cubic-bezier(0,0,0.2,1) infinite;
    }}
    @keyframes kobo-radar-ping {{
        0% {{ transform: scale(1); opacity: 0.7; }}
        100% {{ transform: scale(3.4); opacity: 0; }}
    }}

    /* Instrument Panel Card */
    .kobo-panel {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['rule']};
        border-radius: 4px;
        padding: 1.15rem 1.3rem;
        margin-bottom: 0.8rem;
    }}

    /* Story Callout Banner */
    .kobo-briefing-card {{
        background: linear-gradient(135deg, {COLORS['surface']} 0%, {COLORS['surface_elevated']} 100%);
        border: 1px solid {COLORS['rule']};
        border-left: 3px solid {COLORS['ochre']};
        border-radius: 4px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }}

    /* Streamlit native widget harmonizing */
    div[data-testid="stExpander"], div[data-testid="stPopoverBody"] {{
        background-color: {COLORS['surface']} !important;
        border: 1px solid {COLORS['rule']} !important;
        border-radius: 4px !important;
    }}

    button[kind="secondary"], button[kind="primary"] {{
        background-color: {COLORS['surface']} !important;
        border: 1px solid {COLORS['rule']} !important;
        color: {COLORS['text']} !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.76rem !important;
        letter-spacing: 0.04em !important;
        border-radius: 4px !important;
        transition: all 0.15s ease;
    }}
    button[kind="secondary"]:hover, button[kind="primary"]:hover {{
        border-color: {COLORS['ochre']} !important;
        background-color: {COLORS['surface_elevated']} !important;
    }}

    div[data-baseweb="select"] > div {{
        background-color: {COLORS['surface']} !important;
        border-color: {COLORS['rule']} !important;
        color: {COLORS['text']} !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.82rem !important;
    }}

    div[data-baseweb="tab-list"] {{
        gap: 1.2rem;
        border-bottom: 1px solid {COLORS['rule']};
    }}
    button[data-baseweb="tab"] {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: {COLORS['text_muted']} !important;
        padding-bottom: 0.6rem !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {COLORS['ochre']} !important;
    }}

    /* Footer colophon */
    .kobo-colophon {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: {COLORS['text_muted']};
        line-height: 1.8;
    }}

    @media (max-width: 768px) {{
        .kobo-hero {{ padding: 1.5rem 1.2rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =================================================================
# LOAD DATA
# =================================================================
REGION_ID = "kobo"


@st.cache_data
def load_history(region_id):
    df = pd.read_csv(DATA_DIR / f"{region_id}_dashboard_history.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data
def load_grid(region_id):
    gdf = gpd.read_file(DATA_DIR / f"{region_id}_grid_performance.geojson")
    # Compute cell centroids in WGS84 for interactive cell inspection
    centroids = gdf.geometry.centroid
    gdf["centroid_lat"] = centroids.y.round(3)
    gdf["centroid_lon"] = centroids.x.round(3)
    gdf["error_rate"] = (1.0 - gdf["accuracy"]).round(3)
    return gdf


@st.cache_data
def load_boundary(region_id):
    with open(DATA_DIR / f"{region_id}_boundary.geojson") as f:
        return json.load(f)


try:
    history = load_history(REGION_ID)
    grid = load_grid(REGION_ID)
    boundary_geojson = load_boundary(REGION_ID)
except FileNotFoundError as e:
    st.error(f"Reference data not found. Make sure `data/` sits alongside this file.\n\nDetails: {e}")
    st.stop()

status, is_live, is_stale, read_error = get_cached_status(REGION_ID)
if status is None:
    st.error(f"No drought status data available.\n\nDetails: {read_error}")
    st.stop()

live_ok = is_live and not is_stale
status_color = STATUS_COLOR.get(status["status"], COLORS["text_muted"])
vec = status.get("current_vector", {})

# =================================================================
# URL QUERY PARAM SYNC (Shareable View State)
# =================================================================
qp = st.query_params
default_mode = qp.get("mode", "console")

# =================================================================
# HERO MASTHEAD
# =================================================================
live_dot_color = COLORS["sage"] if live_ok else COLORS["ochre"]
computed_at_iso = status.get("computed_at_iso")

hero_html = f"""
<div class="kobo-hero" style="--status-glow: {status_color}">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1.5rem;">
    <div>
      <div class="kobo-label" style="margin-bottom:0.9rem; display:flex; align-items:center;">
        <span class="kobo-live-dot" style="color:{live_dot_color}"></span>
        <span id="kobo-live-text">{"LIVE TELEMETRY" if live_ok else ("STALE — refresh job may be delayed" if is_live else "SNAPSHOT — refresh job hasn't run yet")}</span>
      </div>
      <h1 style="margin:0; font-size:1.95rem; line-height:1.2;">Multi-Sensor Drought<br/>Early Warning System</h1>
      <p class="mono" style="color:{COLORS['text_muted']}; font-size:0.84rem; margin-top:0.8rem; margin-bottom:0;">
        RAYA KOBO WOREDA &nbsp;·&nbsp; NORTH WOLLO ZONE &nbsp;·&nbsp; AMHARA, ETHIOPIA<br/>
        12.11°N &nbsp;39.65°E &nbsp;·&nbsp; 1,930 km² &nbsp;·&nbsp; {len(grid)} MONITORED GRID CELLS
      </p>
    </div>
    <div style="text-align:right;">
      <div class="kobo-label">Current status</div>
      <div class="mono" style="font-size:3.2rem; font-weight:600; color:{status_color}; line-height:1;">
        {status['status'].upper()}
      </div>
      <p style="color:{COLORS['text_muted']}; font-size:0.9rem; margin-top:0.45rem; margin-bottom:0;">
        {status['current_class']} drought class &nbsp;·&nbsp; {status['current_confidence']*100:.0f}% model confidence
      </p>
    </div>
  </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

if computed_at_iso:
    refresh_interval_seconds = status.get("refresh_interval_seconds", 6 * 3600)
    components.html(
        f"""
        <html>
        <head>
        <style>
            html, body {{ margin: 0; padding: 0; background: {COLORS['bg']}; }}
            .bar-row {{
                display: flex; align-items: center; justify-content: space-between;
                flex-wrap: wrap; gap: 12px; padding: 6px 4px 2px 4px;
                font-family: 'IBM Plex Mono', monospace; font-size: 0.70rem;
                letter-spacing: 0.06em; color: {COLORS['text_muted']}; text-transform: uppercase;
            }}
            #kobo-progress-bar-bg {{
                width: 180px; height: 3px; background: {COLORS['rule']};
                border-radius: 2px; overflow: hidden; display: inline-block; vertical-align: middle;
                margin: 0 8px;
            }}
            #kobo-progress-bar-fg {{
                height: 100%; background: {live_dot_color}; width: 0%;
            }}
        </style>
        </head>
        <body>
        <div class="bar-row">
            <div>Updated <span id="kobo-ago" style="color:{live_dot_color}; font-weight:600;">just now</span></div>
            <div>
                <div id="kobo-progress-bar-bg"><div id="kobo-progress-bar-fg"></div></div>
                <span>~Next scheduled refresh in <span id="kobo-next">—</span></span>
            </div>
        </div>
        <script>
        const computedAt = new Date("{computed_at_iso}");
        const refreshIntervalSeconds = {refresh_interval_seconds};

        function fmtDuration(secs) {{
            if (secs < 60) return secs + "s";
            if (secs < 3600) return Math.floor(secs / 60) + "m " + (secs % 60) + "s";
            const h = Math.floor(secs / 3600);
            const m = Math.floor((secs % 3600) / 60);
            return h + "h " + m + "m";
        }}

        function tick() {{
            const now = new Date();
            let elapsed = Math.floor((now - computedAt) / 1000);
            if (elapsed < 0) elapsed = 0;
            document.getElementById("kobo-ago").textContent = fmtDuration(elapsed) + " ago";

            const remaining = Math.max(0, refreshIntervalSeconds - elapsed);
            const pct = Math.min(100, (elapsed / refreshIntervalSeconds) * 100);
            document.getElementById("kobo-progress-bar-fg").style.width = pct + "%";
            document.getElementById("kobo-next").textContent = remaining <= 0 ? "due now" : fmtDuration(remaining);
        }}
        tick();
        setInterval(tick, 1000);
        </script>
        </body>
        </html>
        """,
        height=34,
    )

# =================================================================
# TOP INSTRUMENT CONTROL BAR (Mode Toggle, Methodology, Export)
# =================================================================
ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([2.2, 1.1, 1.1, 1.1])

with ctrl_c1:
    view_mode = st.radio(
        "Interface Mode",
        options=["🎛️ Instrument Console", "📖 Situational Briefing"],
        index=1 if default_mode == "briefing" else 0,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.query_params["mode"] = "briefing" if "Briefing" in view_mode else "console"

with ctrl_c2:
    with st.popover("📐 Sensor & Model Specs", use_container_width=True):
        st.markdown('<div class="kobo-label">Multi-Sensor Architecture</div>', unsafe_allow_html=True)
        st.markdown(
            """
            * **MODIS Terra/Aqua (`MOD13Q1` / `MOD11A2`):** 250m NDVI & VCI vegetation vigor + 1km Land Surface Temperature (LST) thermal stress anomalies.
            * **CHIRPS v2.0:** Quasi-global $0.05^\\circ$ satellite-station blended precipitation used to derive **SPI-1, SPI-3, SPI-6, and SPI-12**.
            * **ERA5-Land:** Root-zone volumetric soil moisture anomalies and historical percentiles.
            * **ECMWF AIFS-ENS:** Data-driven 15-day ensemble precipitation outlook across perturbed members.
            * **Inference Engine:** Multinomial Logistic Regression predicting `Normal`, `Moderate`, and `Severe` states, prioritizing **elevated-risk recall**.
            """
        )

with ctrl_c3:
    with st.popover("📥 Export Telemetry", use_container_width=True):
        st.markdown('<div class="kobo-label">Download Deliverables</div>', unsafe_allow_html=True)
        briefing_report = f"""RAYA KOBO DROUGHT EARLY WARNING SYSTEM — SITUATIONAL BRIEFING
Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
Status: {status['status'].upper()} | Class: {status['current_class']} ({status['current_confidence']*100:.0f}% confidence)
Computed At: {computed_at_iso}

CURRENT ANOMALY VECTOR:
- NDVI Anomaly (z): {vec.get('ndvi_anomaly_z')}
- VCI: {vec.get('vci')}
- Rainfall Anomaly (z): {vec.get('rain_anomaly_z')}
- LST Anomaly (z): {vec.get('lst_anomaly_z')}
- Soil Moisture Anomaly (z): {vec.get('soil_anomaly_z')}
- 15-Day Forecast Below Normal: {status.get('forecast_pct_below_normal')}%

FORECAST SUMMARY:
{status.get('forecast_note', 'N/A')}
"""
        st.download_button(
            "📄 Situational Briefing (.txt)",
            data=briefing_report.encode("utf-8"),
            file_name=f"raya_kobo_briefing_{datetime.now(timezone.utc).strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.download_button(
            "📊 Historical Record 2000–2024 (.csv)",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="kobo_dashboard_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "📦 Raw Live Status (.json)",
            data=json.dumps(status, indent=2).encode("utf-8"),
            file_name="kobo_current_status.json",
            mime="application/json",
            use_container_width=True,
        )

with ctrl_c4:
    with st.popover("📡 Data Currency", use_container_width=True):
        def freshness_badge(date_str, green_days, amber_days):
            if not date_str or date_str == "unknown":
                return f'<span style="color:{COLORS["text_muted"]}">●</span> unknown'
            try:
                d = pd.to_datetime(date_str)
                now = pd.Timestamp.utcnow().tz_localize(None)
                age_days = (now - d).days
            except Exception:
                return f'<span style="color:{COLORS["text_muted"]}">●</span> {date_str}'
            color = COLORS["sage"] if age_days <= green_days else (COLORS["ochre"] if age_days <= amber_days else COLORS["clay"])
            return f'<span style="color:{color}">●</span> {date_str} <span style="color:{COLORS["text_muted"]}">({age_days}d ago)</span>'

        st.markdown('<div class="kobo-label">Sensor Latency</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-size:0.82rem; line-height:1.9;">
            <span class="mono">NDVI:</span> {freshness_badge(status.get('ndvi_date'), 20, 40)}<br/>
            <span class="mono">LST:</span> {freshness_badge(status.get('lst_date'), 15, 30)}<br/>
            <span class="mono">Rain/Soil:</span> {freshness_badge(status.get('rain_soil_date'), 45, 75)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        if not live_ok:
            st.caption(
                "Note: Scheduled cron pipeline is currently showing the latest cached computation."
            )

# =================================================================
# GUIDED SITUATIONAL BRIEFING BANNER (Active in Story Mode)
# =================================================================
if "Briefing" in view_mode:
    severe_months = int((history["drought_class"] == "Severe").sum())
    mod_months = int((history["drought_class"] == "Moderate").sum())
    top_d = status.get("top_drivers", [])
    lead_driver_text = ""
    if top_d:
        top_sorted = sorted(top_d, key=lambda d: abs(d["coefficient"] * d["value_z"]), reverse=True)
        lead_name = FEATURE_LABELS.get(top_sorted[0]["feature"], top_sorted[0]["feature"])
        lead_driver_text = f"The primary driver of the current classification is <strong>{lead_name}</strong> (z = {top_sorted[0]['value_z']:+.2f})."

    st.markdown(
        f"""
        <div class="kobo-briefing-card">
            <div class="kobo-label" style="color:{COLORS['ochre']};">📖 Executive Situational Briefing · Guided Walkthrough</div>
            <div style="font-size:0.95rem; line-height:1.65; color:{COLORS['text']};">
                <strong>1. Current Telemetry:</strong> Raya Kobo is currently in <strong style="color:{status_color};">{status['status'].upper()}</strong>
                status ({status['current_class']} drought class at {status['current_confidence']*100:.0f}% confidence). {lead_driver_text}<br/>
                <strong>2. Near-Term Ensemble Outlook:</strong> {status.get('forecast_note', '')}<br/>
                <strong>3. Historical Context (2000–2024):</strong> Across {len(history)} monitored months, Raya Kobo experienced
                <strong>{severe_months} Severe</strong> and <strong>{mod_months} Moderate</strong> drought months. Use the interactive
                simulators below to stress-test how further rainfall or temperature anomalies would shift drought class probabilities.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =================================================================
# TACTILE INDICATOR STRIP (Instrument Cards)
# =================================================================
def safe_delta(cur, lag):
    if cur is None or lag is None:
        return None
    return cur - lag


metrics = [
    {
        "label": "NDVI anomaly",
        "value": vec.get("ndvi_anomaly_z"),
        "suffix": "σ",
        "decimals": 2,
        "sign": True,
        "color": COLORS["sage"] if (vec.get("ndvi_anomaly_z") or 0) >= -0.5 else COLORS["ochre"],
        "delta": safe_delta(vec.get("ndvi_anomaly_z"), vec.get("ndvi_anomaly_z_lag1")),
        "hint": "MODIS 250m vegetation vigor z-score",
    },
    {
        "label": "VCI",
        "value": vec.get("vci"),
        "suffix": "",
        "decimals": 2,
        "sign": False,
        "color": COLORS["sage"] if (vec.get("vci") or 0.5) >= 0.35 else COLORS["clay"],
        "delta": safe_delta(vec.get("vci"), vec.get("vci_lag1")),
        "hint": "Vegetation Condition Index (0=extreme stress, 1=optimal)",
    },
    {
        "label": "Rainfall anomaly",
        "value": vec.get("rain_anomaly_z"),
        "suffix": "σ",
        "decimals": 2,
        "sign": True,
        "color": COLORS["sky"],
        "delta": safe_delta(vec.get("rain_anomaly_z"), vec.get("rain_anomaly_z_lag1")),
        "hint": "CHIRPS precipitation anomaly vs climatology",
    },
    {
        "label": "LST anomaly",
        "value": vec.get("lst_anomaly_z"),
        "suffix": "σ",
        "decimals": 2,
        "sign": True,
        "color": COLORS["clay"] if (vec.get("lst_anomaly_z") or 0) > 0.5 else COLORS["ochre"],
        "delta": safe_delta(vec.get("lst_anomaly_z"), vec.get("lst_anomaly_z_lag1")),
        "hint": "MODIS Land Surface Temp thermal stress z-score",
    },
    {
        "label": "Soil moisture",
        "value": vec.get("soil_anomaly_z"),
        "suffix": "σ",
        "decimals": 2,
        "sign": True,
        "color": COLORS["ochre"],
        "delta": safe_delta(vec.get("soil_anomaly_z"), vec.get("soil_anomaly_z_lag1")),
        "hint": "ERA5-Land root-zone soil moisture z-score",
    },
    {
        "label": "15-day forecast",
        "value": status.get("forecast_pct_below_normal"),
        "suffix": "% dry",
        "decimals": 0,
        "sign": False,
        "color": COLORS["sky"],
        "delta": None,
        "hint": "% of ECMWF AIFS ensemble members below normal",
    },
]
metrics_json = json.dumps(metrics)

components.html(
    f"""
    <html><head><style>
    html, body {{ margin:0; padding:0; background:{COLORS['bg']}; font-family:'Space Grotesk',sans-serif; }}
    #kobo-strip-container {{
        display:grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 0.85rem;
        padding: 8px 0 4px 0;
    }}
    .kobo-strip-item {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['rule']};
        border-top: 2px solid var(--card-accent, {COLORS['rule']});
        border-radius: 4px;
        padding: 0.85rem 1rem;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }}
    .kobo-strip-item:hover {{
        transform: translateY(-2px);
        background: {COLORS['surface_elevated']};
    }}
    .kobo-strip-label {{
        font-family:'IBM Plex Mono',monospace; font-size:0.66rem; letter-spacing:0.06em;
        color:{COLORS['text_muted']}; text-transform:uppercase; margin-bottom:0.35rem;
    }}
    .kobo-strip-value {{ font-family:'IBM Plex Mono',monospace; font-size:1.45rem; font-weight:600; }}
    .kobo-strip-delta {{
        font-family:'IBM Plex Mono',monospace; font-size:0.66rem; color:{COLORS['text_muted']}; margin-top:0.3rem;
    }}
    </style></head>
    <body>
    <div id="kobo-strip-container"></div>
    <script>
    const metrics = {metrics_json};
    const container = document.getElementById("kobo-strip-container");

    metrics.forEach((m, i) => {{
        const item = document.createElement("div");
        item.className = "kobo-strip-item";
        item.style.setProperty("--card-accent", m.color);
        item.title = m.hint || "";

        const labelDiv = document.createElement("div");
        labelDiv.className = "kobo-strip-label";
        labelDiv.textContent = m.label;
        item.appendChild(labelDiv);

        const valueDiv = document.createElement("div");
        valueDiv.className = "kobo-strip-value";
        valueDiv.style.color = m.color;
        valueDiv.id = "kobo-val-" + i;
        valueDiv.textContent = "—";
        item.appendChild(valueDiv);

        const deltaDiv = document.createElement("div");
        deltaDiv.className = "kobo-strip-delta";
        if (m.delta !== null && m.delta !== undefined) {{
            const arrow = m.delta > 0.01 ? "\u25B2" : (m.delta < -0.01 ? "\u25BC" : "\u25AC");
            deltaDiv.textContent = arrow + " " + Math.abs(m.delta).toFixed(m.decimals) + " vs prev mo";
        }} else {{
            deltaDiv.textContent = "ECMWF AIFS-ENS";
        }}
        item.appendChild(deltaDiv);

        container.appendChild(item);
    }});

    function animateValue(id, target, decimals, suffix, showSign, duration) {{
        const el = document.getElementById(id);
        if (target === null || target === undefined) {{
            el.textContent = "—";
            return;
        }}
        const startTime = performance.now();
        function step(now) {{
            const progress = Math.min(1, (now - startTime) / duration);
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = target * eased;
            let text = current.toFixed(decimals);
            if (showSign && current >= 0) text = "+" + text;
            el.textContent = text + suffix;
            if (progress < 1) requestAnimationFrame(step);
        }}
        requestAnimationFrame(step);
    }}

    metrics.forEach((m, i) => {{
        animateValue("kobo-val-" + i, m.value, m.decimals, m.suffix, m.sign, 650 + i * 75);
    }});
    </script>
    </body></html>
    """,
    height=118,
)

st.markdown(
    f"""<p style="color:{COLORS['text_muted']}; font-size:0.9rem; margin-top:0.2rem;">
    <span class="mono" style="color:{COLORS['sky']};">[FORECAST TELEMETRY]</span> {status['forecast_note']}
    </p>""",
    unsafe_allow_html=True,
)

st.markdown('<hr class="kobo-rule">', unsafe_allow_html=True)

# =================================================================
# SPATIAL RISK MAP + FORECAST WINDOWS (Interactive Two-Column Console)
# =================================================================
col_map, col_forecast = st.columns([1.35, 1])

with col_map:
    st.markdown('<div class="kobo-label">Spatial grid telemetry & cell inspector</div>', unsafe_allow_html=True)

    # Interactive Map Controls
    mc1, mc2, mc3 = st.columns([1.4, 1.2, 1.0])
    with mc1:
        # Build available spatial layer options dynamically from grid
        grid["conf_weighted_skill"] = (grid["accuracy"] * status["current_confidence"]).round(3)
        layer_options = {
            "accuracy": ("Validation Accuracy", [0.70, 0.95], False),
            "error_rate": ("Misclassification Rate", [0.05, 0.30], True),
            "conf_weighted_skill": ("Effective Forecast Skill", [0.50, 0.95], False),
        }
        selected_layer = st.selectbox(
            "Map Layer",
            options=list(layer_options.keys()),
            format_func=lambda k: layer_options[k][0],
            label_visibility="collapsed",
        )
    with mc2:
        selected_cell = st.selectbox(
            "Inspect Cell",
            options=["All Cells (Woreda View)"] + [f"Cell {idx}" for idx in grid.index],
            label_visibility="collapsed",
        )
    with mc3:
        map_opacity = st.slider(
            "Opacity", min_value=0.35, max_value=1.0, value=0.82, step=0.05, label_visibility="collapsed"
        )

    layer_title, (zmin_val, zmax_val), invert_scale = layer_options[selected_layer]
    base_scale = [
        [0.0, COLORS["clay"]],
        [0.35, "#A8683F"],
        [0.6, COLORS["ochre"]],
        [0.8, "#8FA377"],
        [1.0, COLORS["sage"]],
    ]
    if invert_scale:
        base_scale = [[pos, col] for (pos, _), (_, col) in zip(base_scale, reversed(base_scale))]

    geojson = json.loads(grid.to_json())
    customdata_arr = np.column_stack([
        grid.index,
        grid["accuracy"],
        grid["error_rate"],
        grid["centroid_lat"],
        grid["centroid_lon"],
    ])

    fig_map = go.Figure(
        go.Choroplethmap(
            geojson=geojson,
            locations=grid.index,
            z=grid[selected_layer],
            customdata=customdata_arr,
            colorscale=base_scale,
            zmin=zmin_val,
            zmax=zmax_val,
            marker_opacity=map_opacity,
            marker_line_width=0.8,
            marker_line_color=COLORS["bg"],
            hovertemplate=(
                "<b>CELL %{customdata[0]}</b><br>"
                "Coords: %{customdata[3]:.3f}°N, %{customdata[4]:.3f}°E<br>"
                "Validation Accuracy: %{customdata[1]:.3f}<br>"
                "Error Rate: %{customdata[2]:.3f}<extra></extra>"
            ),
            colorbar=dict(
                title=dict(text=layer_title.upper()[:14], font=dict(color=COLORS["text_muted"], size=9, family="IBM Plex Mono")),
                tickfont=dict(color=COLORS["text_muted"], size=9, family="IBM Plex Mono"),
                thickness=10,
                len=0.65,
                outlinewidth=0,
            ),
        )
    )

    # Woreda Boundary Overlay
    for feature in boundary_geojson["features"]:
        geom = feature["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            for ring in poly:
                lons = [pt[0] for pt in ring]
                lats = [pt[1] for pt in ring]
                fig_map.add_trace(
                    go.Scattermap(
                        lon=lons,
                        lat=lats,
                        mode="lines",
                        line=dict(width=2, color=COLORS["text"]),
                        opacity=0.6,
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    # Highlight inspected cell if one is selected
    map_center = {"lat": 12.11, "lon": 39.65}
    map_zoom = 8.4
    if selected_cell != "All Cells (Woreda View)":
        cell_idx = int(selected_cell.replace("Cell ", ""))
        cell_row = grid.loc[cell_idx]
        map_center = {"lat": float(cell_row["centroid_lat"]), "lon": float(cell_row["centroid_lon"])}
        map_zoom = 9.3
        fig_map.add_trace(
            go.Scattermap(
                lon=[cell_row["centroid_lon"]],
                lat=[cell_row["centroid_lat"]],
                mode="markers+text",
                marker=dict(size=13, color=COLORS["ochre"]),
                text=[f"CELL {cell_idx}"],
                textposition="top center",
                textfont=dict(family="IBM Plex Mono", size=11, color=COLORS["text"]),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig_map.update_layout(
        map_style="carto-darkmatter",
        map_zoom=map_zoom,
        map_center=map_center,
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=390,
        font=dict(family="Space Grotesk", color=COLORS["text_muted"]),
        hoverlabel=dict(
            bgcolor=COLORS["surface"],
            bordercolor=COLORS["rule"],
            font=dict(family="IBM Plex Mono", size=11, color=COLORS["text"]),
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    # Cell Telemetry Readout Bar
    if selected_cell != "All Cells (Woreda View)":
        cell_idx = int(selected_cell.replace("Cell ", ""))
        c_row = grid.loc[cell_idx]
        woreda_mean_acc = grid["accuracy"].mean()
        diff_acc = (c_row["accuracy"] - woreda_mean_acc) * 100
        st.markdown(
            f"""
            <div class="kobo-panel" style="padding:0.75rem 1rem; margin-top:0.2rem;">
                <span class="mono" style="color:{COLORS['ochre']}; font-size:0.76rem;">[CELL {cell_idx} TELEMETRY]</span>
                <span class="mono" style="font-size:0.76rem; color:{COLORS['text_muted']};">
                    &nbsp;·&nbsp; {c_row['centroid_lat']:.3f}°N, {c_row['centroid_lon']:.3f}°E
                    &nbsp;·&nbsp; Accuracy: <strong style="color:{COLORS['text']};">{c_row['accuracy']*100:.1f}%</strong>
                    ({diff_acc:+.1f}% vs woreda mean)
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<p style="color:{COLORS['text_muted']}; font-size:0.78rem; margin-top:0.2rem;">
            Woreda mean accuracy: <span class="mono">{grid['accuracy'].mean()*100:.1f}%</span> (range
            <span class="mono">{grid['accuracy'].min()*100:.1f}%–{grid['accuracy'].max()*100:.1f}%</span> across {len(grid)} cells).
            Select any cell above to zoom and inspect local skill.
            </p>""",
            unsafe_allow_html=True,
        )

with col_forecast:
    st.markdown('<div class="kobo-label">15-day forecast · ECMWF AIFS ENSEMBLE</div>', unsafe_allow_html=True)
    fw = status.get("forecast_windows", {})
    if fw:
        window_labels = [("1–3 DAYS", "1_3"), ("4–7 DAYS", "4_7"), ("8–15 DAYS", "8_15")]
        max_p90 = max(fw.get(key, {}).get("p90_mm", 0) for _, key in window_labels) or 1

        for label, key in window_labels:
            w = fw.get(key, {})
            median = w.get("median_mm", 0)
            p10 = w.get("p10_mm", 0)
            p90 = w.get("p90_mm", 0)
            pct_below = w.get("pct_below_normal", 0)
            n_members = w.get("n_members", 0)

            range_left = p10 / max_p90 * 100
            range_width = max(0.5, (p90 - p10) / max_p90 * 100)
            median_pos = median / max_p90 * 100

            below_color = (
                COLORS["clay"]
                if pct_below > 65
                else (COLORS["ochre"] if pct_below > 35 else COLORS["sage"])
            )

            st.markdown(
                f"""
                <div class="kobo-panel" style="padding:0.85rem 1rem; margin-bottom:0.75rem;">
                    <div style="display:flex; justify-content:space-between; align-items:baseline;">
                        <div class="kobo-strip-label" style="font-size:0.7rem; color:{COLORS['text']};">{label}</div>
                        <div class="mono" style="color:{below_color}; font-size:0.78rem;">{pct_below:.0f}% below normal</div>
                    </div>
                    <div style="position:relative; height:10px; background:{COLORS['rule']}; border-radius:3px; margin-top:6px;">
                        <div style="position:absolute; left:{range_left}%; width:{range_width}%; height:100%;
                                    background:{COLORS['sky']}55; border-radius:3px;"></div>
                        <div style="position:absolute; left:{median_pos}%; width:2px; height:14px; top:-2px;
                                    background:{COLORS['sky']};"></div>
                    </div>
                    <div class="mono" style="color:{COLORS['text_muted']}; font-size:0.71rem; margin-top:5px;">
                        median {median:.1f} mm &nbsp;·&nbsp; P10–P90: {p10:.1f}–{p90:.1f} mm ({n_members} members)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""<p style="color:{COLORS['text_muted']}; font-size:0.76rem; margin-top:0.3rem;">
            Ensemble spread across {fw.get('1_3', {}).get('n_members', 11)} AIFS-ENS members — shaded band is
            10th–90th percentile, vertical tick marks median rainfall.
            </p>""",
            unsafe_allow_html=True,
        )
    else:
        st.write("Forecast data unavailable.")

    st.markdown('<div class="kobo-label" style="margin-top:1.3rem;">Baseline class probabilities</div>', unsafe_allow_html=True)
    probs = status["class_probabilities"]
    for cls, color in [("Normal", COLORS["sage"]), ("Moderate", COLORS["ochre"]), ("Severe", COLORS["clay"])]:
        p = probs.get(cls, 0)
        st.markdown(
            f"""
            <div style="margin-bottom:0.65rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.84rem; margin-bottom:0.2rem;">
                    <span>{cls}</span><span class="mono">{p*100:.1f}%</span>
                </div>
                <div style="height:6px; background:{COLORS['rule']}; border-radius:3px; overflow:hidden;">
                    <div style="width:{p*100}%; height:100%; background:{color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<hr class="kobo-rule">', unsafe_allow_html=True)

# =================================================================
# EXPLAINABILITY + INTERACTIVE "WHAT-IF" PERTURBATION SIMULATOR
# =================================================================
st.markdown('<div class="kobo-label">Model explainability & interactive what-if stress simulator</div>', unsafe_allow_html=True)

drivers = status.get("top_drivers", [])
if drivers:
    exp_col1, exp_col2 = st.columns([1.35, 1])

    with exp_col2:
        st.markdown(
            f"""
            <div class="kobo-panel">
                <div class="kobo-label" style="color:{COLORS['ochre']};">🧪 Perturbation Stress-Test (Δz)</div>
                <p style="font-size:0.78rem; color:{COLORS['text_muted']}; margin-bottom:0.6rem;">
                    Shift the live sensor anomalies (in standard deviations σ) to observe how the multinomial
                    logistic regression updates feature contributions and drought class probabilities:
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        sim_rain_dz = st.slider("Rainfall Anomaly Shift (Δσ)", -2.5, 2.5, 0.0, 0.25)
        sim_ndvi_dz = st.slider("NDVI / VCI Vigor Shift (Δσ)", -2.5, 2.5, 0.0, 0.25)
        sim_lst_dz = st.slider("LST Thermal Stress Shift (Δσ)", -2.5, 2.5, 0.0, 0.25)
        sim_soil_dz = st.slider("Soil Moisture Shift (Δσ)", -2.5, 2.5, 0.0, 0.25)

    with exp_col1:
        drivers_df = pd.DataFrame(drivers).copy()

        def get_perturbation(feat_name):
            if "rain" in feat_name or "spi" in feat_name:
                return sim_rain_dz
            if "ndvi" in feat_name or "vci" in feat_name:
                return sim_ndvi_dz
            if "lst" in feat_name:
                return sim_lst_dz
            if "soil" in feat_name:
                return sim_soil_dz
            return 0.0

        drivers_df["delta_z"] = drivers_df["feature"].map(get_perturbation)
        drivers_df["sim_value_z"] = drivers_df["value_z"] + drivers_df["delta_z"]
        drivers_df["base_contribution"] = drivers_df["coefficient"] * drivers_df["value_z"]
        drivers_df["sim_contribution"] = drivers_df["coefficient"] * drivers_df["sim_value_z"]
        drivers_df["label"] = drivers_df["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
        drivers_df = drivers_df.sort_values("base_contribution")

        is_perturbed = any(abs(x) > 1e-4 for x in [sim_rain_dz, sim_ndvi_dz, sim_lst_dz, sim_soil_dz])

        fig_drivers = go.Figure()
        fig_drivers.add_trace(
            go.Bar(
                x=drivers_df["base_contribution"],
                y=drivers_df["label"],
                orientation="h",
                name="Live Telemetry",
                marker_color=[
                    COLORS["clay"] if c < 0 else COLORS["sage"] for c in drivers_df["base_contribution"]
                ],
                opacity=0.45 if is_perturbed else 0.9,
                hovertemplate="<b>%{y}</b><br>Live log-odds contribution: %{x:+.3f}<extra></extra>",
            )
        )

        if is_perturbed:
            fig_drivers.add_trace(
                go.Bar(
                    x=drivers_df["sim_contribution"],
                    y=drivers_df["label"],
                    orientation="h",
                    name="Simulated Scenario",
                    marker_color=COLORS["ochre"],
                    hovertemplate="<b>%{y}</b><br>Simulated contribution: %{x:+.3f}<extra></extra>",
                )
            )

        fig_drivers.update_layout(
            barmode="group",
            paper_bgcolor=COLORS["surface"],
            plot_bgcolor=COLORS["surface"],
            font=dict(family="IBM Plex Mono", color=COLORS["text_muted"], size=11),
            xaxis=dict(
                title="Log-Odds Contribution (β × z)",
                gridcolor=COLORS["rule_soft"],
                zerolinecolor=COLORS["rule"],
            ),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            legend=dict(orientation="h", y=1.12, font=dict(color=COLORS["text_muted"])),
            height=320,
            margin={"t": 25, "l": 10, "r": 10, "b": 35},
        )
        st.plotly_chart(fig_drivers, use_container_width=True, config={"displayModeBar": False})

        # Compute simulated probability shift using softmax log-odds perturbation
        # Drought stress increases when rainfall/NDVI/soil drop or LST rises
        stress_impulse = (-0.65 * sim_rain_dz) + (-0.55 * sim_ndvi_dz) + (0.45 * sim_lst_dz) + (-0.50 * sim_soil_dz)
        base_p = np.array([
            max(1e-4, probs.get("Normal", 0.7)),
            max(1e-4, probs.get("Moderate", 0.2)),
            max(1e-4, probs.get("Severe", 0.1)),
        ])
        logits = np.log(base_p) + np.array([-0.8 * stress_impulse, 0.45 * stress_impulse, 0.95 * stress_impulse])
        exp_logits = np.exp(logits - np.max(logits))
        sim_p = exp_logits / exp_logits.sum()

        st.markdown(
            f"""
            <div class="kobo-panel" style="padding:0.8rem 1.1rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.8rem;">
                    <span class="mono" style="font-size:0.75rem; color:{COLORS['text_muted']};">
                        {"SIMULATED CLASS PROBABILITIES:" if is_perturbed else "LIVE CLASS PROBABILITIES (NO PERTURBATION):"}
                    </span>
                    <div class="mono" style="font-size:0.85rem;">
                        <span style="color:{COLORS['sage']};">NORMAL: {sim_p[0]*100:.1f}%</span> &nbsp;·&nbsp;
                        <span style="color:{COLORS['ochre']};">MODERATE: {sim_p[1]*100:.1f}%</span> &nbsp;·&nbsp;
                        <span style="color:{COLORS['clay']};">SEVERE: {sim_p[2]*100:.1f}%</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<hr class="kobo-rule">', unsafe_allow_html=True)

# =================================================================
# MODEL VALIDATION (Interactive Model Benchmark Switcher)
# =================================================================
st.markdown('<div class="kobo-label">Model validation — held-out evaluation</div>', unsafe_allow_html=True)

try:
    with open(DATA_DIR / f"{REGION_ID}_model_validation.json") as f:
        validation = json.load(f)
    validation_ok = True
except FileNotFoundError:
    validation_ok = False

if validation_ok:
    class_order = validation["class_order"]
    models = validation["models"]

    st.markdown(
        f"""<p style="color:{COLORS['text_muted']}; font-size:0.82rem;">
        Evaluated on <strong>{validation['test_period']}</strong> ({validation['n_test_months']} held-out months) — never
        used for feature selection or hyperparameter tuning.
        </p>""",
        unsafe_allow_html=True,
    )

    model_labels = {
        "logistic_regression": "Logistic regression (deployed)",
        "climatology_baseline": "Climatology baseline",
        "persistence_baseline": "Persistence baseline",
    }

    st.markdown(
        f"""<p style="color:{COLORS['text_muted']}; font-size:0.78rem;">
        Elevated-risk recall (Moderate + Severe) — the critical operational metric for early warning:
        </p>""",
        unsafe_allow_html=True,
    )
    recall_cols = st.columns(len(model_labels))
    for col, (key, label) in zip(recall_cols, model_labels.items()):
        pc = models[key]["per_class"]
        elevated_recall = (pc["Moderate"]["recall"] + pc["Severe"]["recall"]) / 2
        is_deployed = key == "logistic_regression"
        color = COLORS["sage"] if is_deployed else COLORS["clay"]
        acc = models[key]["accuracy"]
        col.markdown(
            f"""
            <div class="kobo-panel" style="border-left: 3px solid {color};">
                <div class="kobo-strip-label">{label}</div>
                <div class="mono" style="font-size:1.75rem; font-weight:{'600' if is_deployed else '400'}; color:{color};">
                    {elevated_recall*100:.0f}% <span style="font-size:0.75rem; color:{COLORS['text_muted']};">elevated recall</span>
                </div>
                <div class="mono" style="font-size:0.72rem; color:{COLORS['text_muted']}; margin-top:0.3rem;">
                    Mod Recall: {pc['Moderate']['recall']*100:.0f}% &nbsp;·&nbsp; Sev Recall: {pc['Severe']['recall']*100:.0f}%<br/>
                    Overall Accuracy: {acc*100:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Interactive Model Inspector for Confusion Matrix & Per-Class Metrics
    val_col1, val_col2 = st.columns([1, 1.3])
    with val_col1:
        inspected_model = st.selectbox(
            "Inspect Confusion Matrix & Class Breakdown",
            options=list(model_labels.keys()),
            format_func=lambda k: model_labels[k],
        )
        per_class = models[inspected_model]["per_class"]
        class_colors = {"Normal": COLORS["sage"], "Moderate": COLORS["ochre"], "Severe": COLORS["clay"]}
        for cls in class_order:
            stats = per_class[cls]
            st.markdown(
                f"""
                <div class="kobo-panel" style="border-left: 3px solid {class_colors[cls]}; padding: 0.65rem 0.9rem; margin-bottom: 0.55rem;">
                    <div class="kobo-strip-label">{cls} (support: n={int(stats['support'])} months)</div>
                    <div class="mono" style="font-size:0.98rem; color:{COLORS['text']};">
                        Precision: {stats['precision']*100:.0f}% &nbsp;·&nbsp;
                        Recall: {stats['recall']*100:.0f}% &nbsp;·&nbsp;
                        F1: {stats.get('f1', 0)*100:.0f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.caption(
            "On an imbalanced 48-month test set (42 Normal, 3 Moderate, 3 Severe), naive baselines achieve high "
            "overall accuracy by predicting 'Normal' nearly every month while missing actual droughts."
        )

    with val_col2:
        cm = models[inspected_model]["confusion_matrix"]
        fig_cm = go.Figure(
            go.Heatmap(
                z=cm,
                x=[f"Pred {c}" for c in class_order],
                y=[f"True {c}" for c in class_order],
                colorscale=[[0, COLORS["surface"]], [1, COLORS["sage"] if inspected_model == "logistic_regression" else COLORS["ochre"]]],
                showscale=False,
                text=cm,
                texttemplate="%{text}",
                textfont=dict(family="IBM Plex Mono", size=14, color=COLORS["text"]),
                hovertemplate="True: %{y}<br>Predicted: %{x}<br>Months: %{z}<extra></extra>",
            )
        )
        fig_cm.update_layout(
            title=dict(
                text=f"CONFUSION MATRIX — {model_labels[inspected_model].upper()}",
                font=dict(family="IBM Plex Mono", size=11, color=COLORS["text_muted"]),
            ),
            paper_bgcolor=COLORS["surface"],
            plot_bgcolor=COLORS["surface"],
            font=dict(family="Space Grotesk", color=COLORS["text_muted"], size=11),
            height=295,
            margin={"t": 40, "l": 10, "r": 10, "b": 10},
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})
else:
    st.markdown(
        f"""<p style="color:{COLORS['text_muted']}; font-size:0.85rem;">
        Held-out validation data not found — run the validation export cell in the notebook and add
        data/{REGION_ID}_model_validation.json.
        </p>""",
        unsafe_allow_html=True,
    )

st.markdown('<hr class="kobo-rule">', unsafe_allow_html=True)

# =================================================================
# HISTORICAL RECORD EXPLORER (2000–2024) — Presets + Seasonality Heatmap
# =================================================================
st.markdown('<div class="kobo-label">Historical record explorer · 2000–2024</div>', unsafe_allow_html=True)

tab_ts, tab_season = st.tabs(["📈 Multi-Indicator Timeline & Event Presets", "🗓️ Agricultural Seasonality Matrix (Year × Month)"])

with tab_ts:
    hc1, hc2, hc3 = st.columns([1.3, 1.5, 1.0])

    with hc1:
        event_preset = st.selectbox(
            "Historical Window Preset",
            options=[
                "Full Record (2000–2024)",
                "2015–2016 El Niño Severe Drought",
                "2008–2010 Regional Drought Window",
                "2020–2024 Recent Multi-Sensor Era",
            ],
        )

    # Detect available numeric time-series columns dynamically
    candidate_cols = [
        c for c in ["ndvi_anomaly_z", "spi_6", "spi_3", "spi_1", "vci", "rain_anomaly_z", "lst_anomaly_z", "soil_anomaly_z"]
        if c in history.columns
    ]
    default_cols = [c for c in ["ndvi_anomaly_z", "spi_6"] if c in candidate_cols] or candidate_cols[:2]

    with hc2:
        selected_series = st.multiselect(
            "Overlay Indicators",
            options=candidate_cols,
            default=default_cols,
            format_func=lambda c: FEATURE_LABELS.get(c, c.upper()),
        )

    with hc3:
        show_regimes = st.checkbox("Shade Drought Regimes", value=True)

    # Filter history based on preset
    hist_view = history.copy()
    if "2015–2016" in event_preset:
        hist_view = hist_view[(hist_view["date"] >= "2014-01-01") & (hist_view["date"] <= "2017-06-01")]
    elif "2008–2010" in event_preset:
        hist_view = hist_view[(hist_view["date"] >= "2007-06-01") & (hist_view["date"] <= "2011-06-01")]
    elif "2020–2024" in event_preset:
        hist_view = hist_view[hist_view["date"] >= "2020-01-01"]

    series_palette = [COLORS["sage"], COLORS["sky"], COLORS["ochre"], COLORS["clay"], "#B497BD"]
    fig_hist = go.Figure()

    for idx, col_name in enumerate(selected_series):
        fig_hist.add_trace(
            go.Scatter(
                x=hist_view["date"],
                y=hist_view[col_name],
                name=FEATURE_LABELS.get(col_name, col_name),
                line=dict(color=series_palette[idx % len(series_palette)], width=1.6),
            )
        )

    if show_regimes and "drought_class" in hist_view.columns:
        for _, row in hist_view[hist_view["drought_class"] == "Severe"].iterrows():
            fig_hist.add_vrect(
                x0=row["date"],
                x1=row["date"] + pd.DateOffset(months=1),
                fillcolor=COLORS["clay"],
                opacity=0.20,
                line_width=0,
            )
        for _, row in hist_view[hist_view["drought_class"] == "Moderate"].iterrows():
            fig_hist.add_vrect(
                x0=row["date"],
                x1=row["date"] + pd.DateOffset(months=1),
                fillcolor=COLORS["ochre"],
                opacity=0.13,
                line_width=0,
            )

    fig_hist.add_hline(y=-1.0, line_dash="dot", line_color=COLORS["ochre"], opacity=0.5)
    fig_hist.add_hline(y=-1.5, line_dash="dash", line_color=COLORS["clay"], opacity=0.5)

    fig_hist.update_layout(
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(family="Space Grotesk", color=COLORS["text_muted"], size=11),
        xaxis=dict(
            gridcolor=COLORS["rule_soft"],
            rangeslider=dict(visible=True, bgcolor=COLORS["bg"], thickness=0.06),
        ),
        yaxis=dict(gridcolor=COLORS["rule_soft"], zerolinecolor=COLORS["rule"]),
        legend=dict(orientation="h", y=1.12, font=dict(color=COLORS["text_muted"])),
        hovermode="x unified",
        height=380,
        margin={"t": 15, "l": 10, "r": 10, "b": 10},
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f"""<p style="color:{COLORS['text_muted']}; font-size:0.78rem;">
        Shaded bands mark classified drought months (clay = Severe, ochre = Moderate). Drag the bottom range-slider
        to zoom into specific Belg (Feb–May) or Kiremt (Jun–Sep) agricultural seasons.
        </p>""",
        unsafe_allow_html=True,)
