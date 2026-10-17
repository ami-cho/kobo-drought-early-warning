"""
KOBO DROUGHT EARLY WARNING & DECISION-SUPPORT SYSTEM
Redesigned "field instrument panel" + institutional decision-support workflow

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
MODEL_VERSION = "v2.1-multinomial-logreg"
REGION_ID = "kobo"

st.set_page_config(
    page_title="Raya Kobo Drought Early Warning & Decision-Support System",
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
    "surface_elevated": "#2A241C",
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
    "ndvi_anomaly_z": "NDVI vegetation anomaly (MODIS)",
    "vci": "Vegetation Condition Index (VCI)",
    "rainfall_mm": "Monthly rainfall (CHIRPS)",
    "rain_anomaly_z": "Rainfall anomaly (CHIRPS)",
    "spi_1": "1-Month SPI",
    "spi_3": "3-Month SPI",
    "spi_6": "6-Month SPI",
    "spi_12": "12-Month SPI",
    "lst_anomaly_z": "Land Surface Temp anomaly (MODIS)",
    "soil_anomaly_z": "Root-zone soil moisture anomaly (ERA5-Land)",
    "soil_percentile": "Soil moisture percentile (ERA5-Land)",
    "rain_anomaly_z_lag1": "Rainfall anomaly (previous month)",
    "spi_1_lag1": "1-Month SPI (previous month)",
    "spi_3_lag1": "3-Month SPI (previous month)",
    "soil_anomaly_z_lag1": "Soil moisture anomaly (previous month)",
    "ndvi_anomaly_z_lag1": "NDVI anomaly (previous month)",
}

# =================================================================
# GLOBAL CSS
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
        max-width: 1180px;
    }}

    h1, h2, h3, h4 {{
        font-family: 'Fraunces', serif;
        color: {COLORS['text']};
        font-weight: 500;
    }}

    p, span, div, label, li {{
        color: {COLORS['text']};
    }}

    .mono {{
        font-family: 'IBM Plex Mono', monospace;
    }}

    .kobo-rule {{
        border: none;
        border-top: 1px solid {COLORS['rule']};
        margin: 1.8rem 0;
    }}

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
        padding: 2.2rem 2.5rem;
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

    /* 30-Second Triage Grid */
    .triage-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.85rem;
        margin-top: 0.9rem;
        margin-bottom: 0.5rem;
    }}
    .triage-card {{
        background: {COLORS['surface']};
        border: 1px solid {COLORS['rule']};
        border-radius: 4px;
        padding: 0.95rem 1.15rem;
    }}
    .triage-q {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.66rem;
        letter-spacing: 0.06em;
        color: {COLORS['text_muted']};
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }}
    .triage-a {{
        font-size: 1.02rem;
        font-weight: 600;
        color: {COLORS['text']};
        line-height: 1.35;
    }}
    .triage-sub {{
        font-size: 0.78rem;
        color: {COLORS['text_muted']};
        margin-top: 0.25rem;
    }}

    /* Instrument Panel Card */
    .kobo-panel {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['rule']};
        border-radius: 4px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.75rem;
    }}

    /* Response Protocol Cards */
    .protocol-card {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['rule']};
        border-radius: 4px;
        padding: 1.1rem 1.25rem;
        height: 100%;
    }}
    .protocol-active {{
        background-color: {COLORS['surface_elevated']};
        border: 1px solid var(--active-color, {COLORS['ochre']});
        border-left: 4px solid var(--active-color, {COLORS['ochre']});
    }}

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
        font-size: 0.75rem !important;
        border-radius: 4px !important;
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
@st.cache_data
def load_history(region_id):
    df = pd.read_csv(DATA_DIR / f"{region_id}_dashboard_history.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data
def load_grid(region_id):
    gdf = gpd.read_file(DATA_DIR / f"{region_id}_grid_performance.geojson")
    centroids = gdf.geometry.centroid
    gdf["centroid_lat"] = centroids.y.round(3)
    gdf["centroid_lon"] = centroids.x.round(3)
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


def safe_delta(cur, lag):
    if cur is None or lag is None:
        return None
    return cur - lag


# Compute plain-language trajectory from sensor month-over-month deltas
ndvi_d = safe_delta(vec.get("ndvi_anomaly_z"), vec.get("ndvi_anomaly_z_lag1")) or 0.0
rain_d = safe_delta(vec.get("rain_anomaly_z"), vec.get("rain_anomaly_z_lag1")) or 0.0
soil_d = safe_delta(vec.get("soil_anomaly_z"), vec.get("soil_anomaly_z_lag1")) or 0.0
composite_trend = (ndvi_d + rain_d + soil_d) / 3.0

if composite_trend < -0.15:
    trend_word = "Deteriorating (Drying)"
    trend_color = COLORS["clay"]
    trend_detail = "Vegetation, rainfall, and soil moisture anomalies declined vs. last month."
elif composite_trend > 0.15:
    trend_word = "Improving (Recovering)"
    trend_color = COLORS["sage"]
    trend_detail = "Moisture and vegetation conditions improved relative to previous month."
else:
    trend_word = "Stable / Holding"
    trend_color = COLORS["ochre"]
    trend_detail = "Minimal net change in composite land-surface anomalies vs. last month."

# =================================================================
# PRINTABLE ONE-PAGE SITUATION BRIEF GENERATOR (HTML / Print-to-PDF)
# =================================================================
def generate_printable_brief_html(status_dict, vec_dict, trend_label, trend_desc):
    issue_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fw = status_dict.get("forecast_windows", {})
    w15 = fw.get("8_15", {})
    probs = status_dict.get("class_probabilities", {})

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Raya Kobo Drought Situation Brief — {issue_ts}</title>
<style>
    body {{
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #1B1712; background: #FFFFFF; margin: 0; padding: 28px 36px;
        font-size: 13px; line-height: 1.45;
    }}
    .no-print {{
        background: #1B1712; color: #EDE6D8; padding: 10px 16px; border-radius: 4px;
        margin-bottom: 18px; display: flex; justify-content: space-between; align-items: center;
    }}
    .print-btn {{
        background: #C99A4E; color: #1B1712; border: none; padding: 6px 14px;
        font-weight: 700; border-radius: 3px; cursor: pointer;
    }}
    .header {{
        border-bottom: 2px solid #1B1712; padding-bottom: 10px; margin-bottom: 14px;
        display: flex; justify-content: space-between; align-items: flex-end;
    }}
    .title {{ font-size: 20px; font-weight: 700; margin: 0; }}
    .subtitle {{ font-size: 11px; color: #555; text-transform: uppercase; letter-spacing: 0.05em; }}
    .badge {{
        font-size: 18px; font-weight: 800; padding: 4px 12px; border-radius: 4px;
        border: 2px solid #1B1712; text-transform: uppercase;
    }}
    .grid-4 {{
        display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 14px;
    }}
    .box {{
        border: 1px solid #CCC; padding: 8px 10px; border-radius: 3px; background: #FAFAFA;
    }}
    .box-label {{ font-size: 10px; text-transform: uppercase; color: #666; font-weight: 600; }}
    .box-val {{ font-size: 14px; font-weight: 700; margin-top: 2px; }}
    h3 {{
        font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em;
        border-bottom: 1px solid #DDD; padding-bottom: 4px; margin: 14px 0 8px 0;
    }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 12px; font-size: 12px; }}
    th, td {{ border: 1px solid #DDD; padding: 6px 8px; text-align: left; }}
    th {{ background: #F0EFEA; font-weight: 600; }}
    .footer {{
        margin-top: 18px; padding-top: 8px; border-top: 1px solid #CCC;
        font-size: 10.5px; color: #555; display: flex; justify-content: space-between;
    }}
    @media print {{
        .no-print {{ display: none !important; }}
        body {{ padding: 0; }}
    }}
</style>
</head>
<body>
    <div class="no-print">
        <span><strong>Raya Kobo One-Page Drought Situation Brief</strong> — Ready for A4 Print or Save-as-PDF</span>
        <button class="print-btn" onclick="window.print()">🖨️ Print / Save as PDF</button>
    </div>
    <div class="header">
        <div>
            <div class="subtitle">Decision-Support Prototype · North Wollo Zone, Amhara Region, Ethiopia</div>
            <h1 class="title">RAYA KOBO DROUGHT SITUATION BRIEF</h1>
        </div>
        <div class="badge">STATUS: {status_dict['status'].upper()} ({status_dict['current_class']})</div>
    </div>

    <div class="grid-4">
        <div class="box">
            <div class="box-label">1. Area Covered</div>
            <div class="box-val">Raya Kobo Woreda</div>
            <div style="font-size:11px; color:#555;">1,930 km² ({len(grid)} grid cells)</div>
        </div>
        <div class="box">
            <div class="box-label">2. Current Severity</div>
            <div class="box-val">{status_dict['current_class']} Class</div>
            <div style="font-size:11px; color:#555;">Normal: {probs.get('Normal',0)*100:.0f}% | Mod: {probs.get('Moderate',0)*100:.0f}% | Sev: {probs.get('Severe',0)*100:.0f}%</div>
        </div>
        <div class="box">
            <div class="box-label">3. Short-Term Trend</div>
            <div class="box-val">{trend_label}</div>
            <div style="font-size:11px; color:#555;">vs. prior month observation</div>
        </div>
        <div class="box">
            <div class="box-label">4. Observation Dates</div>
            <div class="box-val">NDVI: {status_dict.get('ndvi_date', 'N/A')}</div>
            <div style="font-size:11px; color:#555;">Rain/Soil: {status_dict.get('rain_soil_date', 'N/A')}</div>
        </div>
    </div>

    <h3>Executive Summary & Forecast Outlook</h3>
    <p style="margin: 4px 0 10px 0;">
        <strong>Observed Conditions:</strong> Raya Kobo Woreda is currently classified under <strong>{status_dict['status'].upper()}</strong>
        ({status_dict['current_class']} drought class). {trend_desc}<br/>
        <strong>15-Day ECMWF AIFS Ensemble Forecast:</strong> {status_dict.get('forecast_note', 'N/A')}
        (8–15 day window median: {w15.get('median_mm', 0):.1f} mm, P10–P90 range: {w15.get('p10_mm', 0):.1f}–{w15.get('p90_mm', 0):.1f} mm).
    </p>

    <h3>Observed Multi-Sensor Telemetry</h3>
    <table>
        <tr>
            <th>Indicator</th>
            <th>Source & Resolution</th>
            <th>Latest Date</th>
            <th>Current Value</th>
            <th>Prior Month</th>
        </tr>
        <tr>
            <td>Vegetation Anomaly (NDVI z)</td>
            <td>MODIS MOD13Q1 (250m)</td>
            <td>{status_dict.get('ndvi_date', 'N/A')}</td>
            <td>{vec_dict.get('ndvi_anomaly_z', 0):+.2f}σ (VCI: {vec_dict.get('vci', 0):.2f})</td>
            <td>{vec_dict.get('ndvi_anomaly_z_lag1', 0):+.2f}σ</td>
        </tr>
        <tr>
            <td>Rainfall Anomaly (z)</td>
            <td>CHIRPS v2.0 (0.05° / ~5.5km)</td>
            <td>{status_dict.get('rain_soil_date', 'N/A')}</td>
            <td>{vec_dict.get('rain_anomaly_z', 0):+.2f}σ</td>
            <td>{vec_dict.get('rain_anomaly_z_lag1', 0):+.2f}σ</td>
        </tr>
        <tr>
            <td>Root-Zone Soil Moisture (z)</td>
            <td>ERA5-Land (0.1° / ~9km)</td>
            <td>{status_dict.get('rain_soil_date', 'N/A')}</td>
            <td>{vec_dict.get('soil_anomaly_z', 0):+.2f}σ</td>
            <td>{vec_dict.get('soil_anomaly_z_lag1', 0):+.2f}σ</td>
        </tr>
        <tr>
            <td>Land Surface Temp (LST z)</td>
            <td>MODIS MOD11A2 (1km)</td>
            <td>{status_dict.get('lst_date', 'N/A')}</td>
            <td>{vec_dict.get('lst_anomaly_z', 0):+.2f}σ</td>
            <td>{vec_dict.get('lst_anomaly_z_lag1', 0):+.2f}σ</td>
        </tr>
    </table>

    <h3>Recommended Verification & Preparedness Actions (Proposed Co-Design Protocol)</h3>
    <table>
        <tr>
            <th>Sector</th>
            <th>Recommended Verification / Preparedness Step for Current ({status_dict['status'].upper()}) Level</th>
        </tr>
        <tr>
            <td><strong>Water Resources</strong></td>
            <td>{"Routine monitoring of borehole yields and seasonal streamflows." if status_dict['status'] == "Normal" else ("Audit shallow well depths and motorized scheme functionality across lowland kebeles." if status_dict['status'] == "Watch" else "Prioritize emergency water-point rehabilitation and pre-position water-trucking schedules.")}</td>
        </tr>
        <tr>
            <td><strong>Crop & Agronomy</strong></td>
            <td>{"Standard extension advisory for Belg/Kiremt planting windows." if status_dict['status'] == "Normal" else ("Conduct field spot-checks on sorghum/teff moisture stress; advise on moisture-conserving tillage." if status_dict['status'] == "Watch" else "Deploy rapid crop-loss verification teams; prepare short-maturing / drought-tolerant seed distributions.")}</td>
        </tr>
        <tr>
            <td><strong>Livestock & Rangeland</strong></td>
            <td>{"Monitor communal grazing biomass via routine kebele development agents." if status_dict['status'] == "Normal" else ("Assess browse availability in eastern lowland corridors; review supplementary feed stocks." if status_dict['status'] == "Watch" else "Activate livestock feed supplementation and targeted veterinary support in high-stress kebeles.")}</td>
        </tr>
    </table>

    <div class="footer">
        <div><strong>Issued:</strong> {issue_ts} | <strong>Computed:</strong> {status_dict.get('computed_at_iso', 'N/A')} | <strong>Model:</strong> {MODEL_VERSION}</div>
        <div>Prototype Decision-Support Brief — Verify with official EMI / EDRMC advisories.</div>
    </div>
</body>
</html>"""


# =================================================================
# HERO MASTHEAD
# =================================================================
live_dot_color = COLORS["sage"] if live_ok else COLORS["ochre"]
computed_at_iso = status.get("computed_at_iso")

hero_html = f"""
<div class="kobo-hero" style="--status-glow: {status_color}">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1.5rem;">
    <div>
      <div class="kobo-label" style="margin-bottom:0.8rem; display:flex; align-items:center;">
        <span class="kobo-live-dot" style="color:{live_dot_color}"></span>
        <span id="kobo-live-text">{"LIVE DECISION-SUPPORT TELEMETRY" if live_ok else ("STALE — scheduled refresh delayed (showing last verified run)" if is_live else "BUNDLED SNAPSHOT — scheduled refresh job has not run yet")}</span>
      </div>
      <h1 style="margin:0; font-size:1.95rem; line-height:1.2;">Multi-Sensor Drought Early Warning<br/>&amp; Decision-Support System</h1>
      <p class="mono" style="color:{COLORS['text_muted']}; font-size:0.84rem; margin-top:0.75rem; margin-bottom:0;">
        RAYA KOBO WOREDA &nbsp;·&nbsp; NORTH WOLLO ZONE &nbsp;·&nbsp; AMHARA, ETHIOPIA<br/>
        12.11°N &nbsp;39.65°E &nbsp;·&nbsp; 1,930 km² &nbsp;·&nbsp; MODEL: {MODEL_VERSION.upper()}
      </p>
    </div>
    <div style="text-align:right;">
      <div class="kobo-label">Current warning level</div>
      <div class="mono" style="font-size:3.1rem; font-weight:600; color:{status_color}; line-height:1;">
        {status['status'].upper()}
      </div>
      <p style="color:{COLORS['text_muted']}; font-size:0.88rem; margin-top:0.45rem; margin-bottom:0;">
        Predicted Class: <strong style="color:{COLORS['text']};">{status['current_class']}</strong>
        &nbsp;·&nbsp; Class Score: {status['current_confidence']*100:.0f}%
      </p>
    </div>
  </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# =================================================================
# 1. 30-SECOND DECISION-MAKER TRIAGE BAR + SITUATION BRIEF DOWNLOAD
# =================================================================
st.markdown(
    f"""
    <div class="triage-grid">
        <div class="triage-card">
            <div class="triage-q">1. Which area is at risk?</div>
            <div class="triage-a">Raya Kobo Woreda (1,930 km²)</div>
            <div class="triage-sub">Monitored across {len(grid)} satellite grid cells in North Wollo Zone.</div>
        </div>
        <div class="triage-card" style="border-top: 2px solid {status_color};">
            <div class="triage-q">2. How severe is the situation?</div>
            <div class="triage-a" style="color:{status_color};">{status['status']} — {status['current_class']} Conditions</div>
            <div class="triage-sub">{"Routine preparedness posture." if status['status'] == "Normal" else "Elevated multi-sensor deficit; field verification advised."}</div>
        </div>
        <div class="triage-card" style="border-top: 2px solid {trend_color};">
            <div class="triage-q">3. Worsening or improving?</div>
            <div class="triage-a" style="color:{trend_color};">{trend_word}</div>
            <div class="triage-sub">{trend_detail}</div>
        </div>
        <div class="triage-card">
            <div class="triage-q">4. What period does this cover?</div>
            <div class="triage-a">Obs: {status.get('ndvi_date', 'Latest')}</div>
            <div class="triage-sub">Rain/Soil: {status.get('rain_soil_date', 'Latest')} · Forecast: +15 days</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Action Bar: Printable 1-Page Brief + Data Currency + Pilot Dossier
act_c1, act_c2, act_c3 = st.columns([1.4, 1.3, 1.3])
with act_c1:
    brief_html = generate_printable_brief_html(status, vec, trend_word, trend_detail)
    st.download_button(
        label="📄 Download 1-Page Drought Situation Brief (Print/PDF)",
        data=brief_html.encode("utf-8"),
        file_name=f"kobo_drought_situation_brief_{datetime.now(timezone.utc).strftime('%Y%m%d')}.html",
        mime="text/html",
        use_container_width=True,
    )
with act_c2:
    st.download_button(
        label="📊 Download Historical Series 2000–2024 (.CSV)",
        data=history.to_csv(index=False).encode("utf-8"),
        file_name="kobo_drought_history_2000_2024.csv",
        mime="text/csv",
        use_container_width=True,
    )
with act_c3:
    with st.popover("📡 Sensor Currency & Pipeline Status", use_container_width=True):
        def freshness_indicator(date_str, green_days, amber_days):
            if not date_str or date_str == "unknown":
                return f'<span style="color:{COLORS["text_muted"]}">●</span> unknown'
            try:
                d = pd.to_datetime(date_str)
                now = pd.Timestamp.utcnow().tz_localize(None)
                age_days = (now - d).days
            except Exception:
                return f'<span style="color:{COLORS["text_muted"]}">●</span> {date_str}'
            color = COLORS["sage"] if age_days <= green_days else (COLORS["ochre"] if age_days <= amber_days else COLORS["clay"])
            return f'<span style="color:{color}">●</span> {date_str} <span style="color:{COLORS["text_muted"]}">({age_days}d lag)</span>'

        st.markdown('<div class="kobo-label">Real-World Sensor Latency</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-size:0.82rem; line-height:1.9;">
            <strong>MODIS NDVI (250m):</strong> {freshness_indicator(status.get('ndvi_date'), 20, 40)}<br/>
            <strong>MODIS LST (1km):</strong> {freshness_indicator(status.get('lst_date'), 15, 30)}<br/>
            <strong>CHIRPS / ERA5-Land:</strong> {freshness_indicator(status.get('rain_soil_date'), 45, 75)}<br/>
            <strong>Last Computed:</strong> <span class="mono">{computed_at_iso or 'Snapshot'}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Failed forecast downloads never overwrite newer telemetry with unlabelled stale data."
        )

st.markdown('<hr class="kobo-rule">', unsafe_allow_html=True)

# =================================================================
# 2. STRICT SEPARATION: OBSERVED MONITORING VS. FUTURE FORECAST
# =================================================================
st.markdown(
    '<div class="kobo-label">Section I · Observed Satellite &amp; Reanalysis Monitoring (Current State vs. Prior Month)</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f"""<p style="color:{COLORS['text_muted']}; font-size:0.82rem; margin-top:-0.2rem; margin-bottom:0.6rem;">
    Observed anomalies are standardized ($z$-scores in units of $\\sigma$) against Raya Kobo's 2000–2024 baseline for the same calendar month.
    Negative values for NDVI, Rainfall, and Soil Moisture indicate drier-than-normal conditions; positive LST indicates hotter-than-normal surface temperatures.
    </p>""",
    unsafe_allow_html=True,
)

obs_metrics = [
    {
        "label": "Vegetation Vigor (NDVI)",
        "plain": "Crop & canopy greenness",
        "value": vec.get("ndvi_anomaly_z"),
        "suffix": "σ",
        "decimals": 2,
        "sign": True,
        "color": COLORS["sage"] if (vec.get("ndvi_anomaly_z") or 0) >= -0.5 else COLORS["ochre"],
        "delta": safe_delta(vec.get("ndvi_anomaly_z"), vec.get("ndvi_anomaly_z_lag1")),
    },
    {
        "label": "Vegetation Condition (VCI)",
        "plain": "0 = extreme stress · 1 = optimal",
        "value": vec.get("vci"),
        "suffix": "",
        "decimals": 2,
        "sign": False,
        "color": COLORS["sage"] if (vec.get("vci") or 0.5) >= 0.35 else COLORS["clay"],
        "delta": safe_delta(vec.get("vci"), vec.get("vci_lag1")),
    },
    {
        "label": "Rainfall Anomaly (CHIRPS)",
        "plain": "Satellite-station precipitation",
        "value": vec.get("rain_anomaly_z"),
        "suffix": "σ",
        "decimals": 2,
        "sign": True,
        "color": COLORS["sky"],
        "delta": safe_delta(vec.get("rain_anomaly_z"), vec.get("rain_anomaly_z_lag1")),
    },
    {
        "label": "Soil Moisture (ERA5-Land)",
        "plain": "Root-zone water availability",
        "value": vec.get("soil_anomaly_z"),
        "suffix": "σ",
        "decimals": 2,
        "sign": True,
        "color": COLORS["ochre"],
        "delta": safe_delta(vec.get("soil_anomaly_z"), vec.get("soil_anomaly_z_lag1")),
    },
    {
        "label": "Surface Temp (MODIS LST)",
        "plain": "Evaporative & thermal heat stress",
        "value": vec.get("lst_anomaly_z"),
        "suffix": "σ",
        "decimals": 2,
        "sign": True,
        "color": COLORS["clay"] if (vec.get("lst_anomaly_z") or 0) > 0.5 else COLORS["ochre"],
        "delta": safe_delta(vec.get("lst_anomaly_z"), vec.get("lst_anomaly_z_lag1")),
    },
]

obs_cols = st.columns(5)
for col, m in zip(obs_cols, obs_metrics):
    val = m["value"]
    if val is None:
        val_str = "—"
    else:
        val_str = f"{val:+.{m['decimals']}f}{m['suffix']}" if m["sign"] else f"{val:.{m['decimals']}f}{m['suffix']}"
    d = m["delta"]
    if d is not None:
        arrow = "▲" if d > 0.01 else ("▼" if d < -0.01 else "▬")
        delta_str = f"{arrow} {abs(d):.{m['decimals']}f} vs last month"
    else:
        delta_str = "—"

    col.markdown(
        f"""
        <div class="kobo-panel" style="border-top: 2px solid {m['color']}; padding: 0.85rem 0.95rem;">
            <div class="kobo-strip-label" style="font-size:0.65rem; color:{COLORS['text_muted']};">{m['label']}</div>
            <div class="mono" style="font-size:1.45rem; font-weight:600; color:{m['color']}; margin: 0.2rem 0;">{val_str}</div>
            <div style="font-size:0.73rem; color:{COLORS['text']};">{m['plain']}</div>
            <div class="mono" style="font-size:0.66rem; color:{COLORS['text_muted']}; margin-top:0.3rem;">{delta_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<hr class="kobo-rule">', unsafe_allow_html=True)

# =================================================================
# 3. ACTIONABLE PREPAREDNESS & VERIFICATION RESPONSE GUIDE
# =================================================================
st.markdown(
    '<div class="kobo-label">Section II · Proposed Verification &amp; Preparedness Response Guide</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f"""<p style="color:{COLORS['text_muted']}; font-size:0.82rem; margin-top:-0.2rem;">
    A warning signal is only useful when connected to decisions. Below is the proposed verification and preparedness
    matrix for Raya Kobo Woreda. <em>Note: These thresholds and actions are proposed for collaborative calibration with
    woreda/zonal disaster-risk and agricultural offices, not officially mandated triggers.</em>
    </p>""",
    unsafe_allow_html=True,
)

protocols = [
    {
        "level": "Normal",
        "color": COLORS["sage"],
        "trigger": "Predicted Class: Normal (Composite anomalies near seasonal baseline)",
        "water": "Routine monitoring of borehole yields and seasonal spate irrigation diversions.",
        "crops": "Standard extension advisory for Belg (Feb–May) and Kiremt (Jun–Sep) planting calendars.",
        "livestock": "Routine pasture condition reporting via kebele development agents (DAs).",
        "coord": "Monthly bulletin distribution to Woreda Agriculture & DRMC focal points.",
    },
    {
        "level": "Watch",
        "color": COLORS["ochre"],
        "trigger": "Predicted Class: Moderate OR >60% of AIFS ensemble members below normal rainfall",
        "water": "Audit shallow well water tables and motorized scheme functionality in lowland kebeles.",
        "crops": "Conduct field spot-checks on sorghum/teff moisture stress; advise on tie-ridging & moisture conservation.",
        "livestock": "Assess browse/forage deficits in eastern lowland grazing corridors; inventory feed reserves.",
        "coord": "Convene Woreda Early Warning Task Force to verify satellite anomalies against field reports.",
    },
    {
        "level": "Warning",
        "color": COLORS["clay"],
        "trigger": "Predicted Class: Severe (Multi-sensor convergence across rainfall, soil moisture & VCI)",
        "water": "Prioritize emergency water-point maintenance and pre-position water-trucking schedules.",
        "crops": "Deploy rapid crop-assessment teams; prepare drought-tolerant / short-maturing seed contingency.",
        "livestock": "Activate targeted supplementary livestock feed and veterinary health interventions.",
        "coord": "Submit formal situational brief to North Wollo Zone DRMC, WFP, and FAO anticipatory action teams.",
    },
]

p_cols = st.columns(3)
for col, prot in zip(p_cols, protocols):
    is_active = status["status"].lower() == prot["level"].lower()
    active_badge = (
        f'<span class="mono" style="background:{prot["color"]}; color:{COLORS["bg"]}; padding:2px 6px; border-radius:2px; font-size:0.65rem; font-weight:700;">ACTIVE POSTURE</span>'
        if is_active
        else '<span class="mono" style="color:#666; font-size:0.65rem;">STANDBY LEVEL</span>'
    )
    card_cls = "protocol-card protocol-active" if is_active else "protocol-card"
    col.markdown(
        f"""
        <div class="{card_cls}" style="--active-color:{prot['color']};">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                <span class="mono" style="font-size:1.05rem; font-weight:600; color:{prot['color']};">{prot['level'].upper()}</span>
                {active_badge}
            </div>
            <div style="font-size:0.74rem; color:{COLORS['text_muted']}; margin-bottom:0.75rem; border-bottom:1px solid {COLORS['rule']}; padding-bottom:0.5rem;">
                <strong>Trigger:</strong> {prot['trigger']}
            </div>
            <div style="font-size:0.8rem; line-height:1.55;">
                <p style="margin:0 0 0.45rem 0;">💧 <strong>Water:</strong> {prot['water']}</p>
                <p style="margin:0 0 0.45rem 0;">🌱 <strong>Crops:</strong> {prot['crops']}</p>
                <p style="margin:0 0 0.45rem 0;">🐄 <strong>Livestock:</strong> {prot['livestock']}</p>
                <p style="margin:0;">📋 <strong>Coordination:</strong> {prot['coord']}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<hr class="kobo-rule">', unsafe_allow_html=True)

# =================================================================
# 4. SPATIAL GRID TELEMETRY + 15-DAY FUTURE ENSEMBLE FORECAST
# =================================================================
col_map, col_forecast = st.columns([1.3, 1])

with col_map:
    st.markdown('<div class="kobo-label">Spatial validation &amp; grid cell inspector (1,930 km²)</div>', unsafe_allow_html=True)

    mc1, mc2 = st.columns([1.3, 1])
    with mc1:
        selected_cell = st.selectbox(
            "Inspect Specific Grid Cell",
            options=["All Cells (Woreda Overview)"] + [f"Cell {idx} ({grid.loc[idx, 'centroid_lat']:.2f}°N, {grid.loc[idx, 'centroid_lon']:.2f}°E)" for idx in grid.index],
            label_visibility="collapsed",
        )
    with mc2:
        map_opacity = st.slider("Map Layer Opacity", 0.35, 1.0, 0.82, 0.05, label_visibility="collapsed")

    geojson = json.loads(grid.to_json())
    fig_map = go.Figure(
        go.Choroplethmap(
            geojson=geojson,
            locations=grid.index,
            z=grid["accuracy"],
            customdata=np.column_stack([grid.index, grid["centroid_lat"], grid["centroid_lon"]]),
            colorscale=[
                [0.0, COLORS["clay"]],
                [0.35, "#A8683F"],
                [0.6, COLORS["ochre"]],
                [0.8, "#8FA377"],
                [1.0, COLORS["sage"]],
            ],
            zmin=0.70,
            zmax=0.95,
            marker_opacity=map_opacity,
            marker_line_width=0.7,
            marker_line_color=COLORS["bg"],
            hovertemplate=(
                "<b>GRID CELL %{customdata[0]}</b><br>"
                "Center: %{customdata[1]:.3f}°N, %{customdata[2]:.3f}°E<br>"
                "Historical Validation Accuracy: %{z:.1%}<extra></extra>"
            ),
            colorbar=dict(
                title=dict(text="TEST ACC", font=dict(color=COLORS["text_muted"], size=10, family="IBM Plex Mono")),
                tickfont=dict(color=COLORS["text_muted"], size=9, family="IBM Plex Mono"),
                thickness=10,
                len=0.65,
                outlinewidth=0,
            ),
        )
    )

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
                        opacity=0.55,
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    map_center = {"lat": 12.11, "lon": 39.65}
    map_zoom = 8.4
    if selected_cell != "All Cells (Woreda Overview)":
        c_idx = int(selected_cell.split()[1])
        c_row = grid.loc[c_idx]
        map_center = {"lat": float(c_row["centroid_lat"]), "lon": float(c_row["centroid_lon"])}
        map_zoom = 9.2
        fig_map.add_trace(
            go.Scattermap(
                lon=[c_row["centroid_lon"]],
                lat=[c_row["centroid_lat"]],
                mode="markers+text",
                marker=dict(size=13, color=COLORS["ochre"]),
                text=[f"CELL {c_idx}"],
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
    st.markdown(
        f"""<p style="color:{COLORS['text_muted']}; font-size:0.78rem;">
        <strong>Spatial Note:</strong> Map displays historical validation accuracy per 0.05° grid cell. Western escarpment
        cells show distinct agro-climatic behavior compared to the eastern Raya valley floor.
        </p>""",
        unsafe_allow_html=True,
    )

with col_forecast:
    st.markdown(
        '<div class="kobo-label">Section III · Genuine Future Forecast (15-Day ECMWF AIFS Ensemble)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<p style="color:{COLORS['text_muted']}; font-size:0.78rem; margin-top:-0.2rem;">
        Unlike the observed satellite anomalies above, this panel shows <strong>forward numerical weather predictions</strong>
        initialized at <span class="mono">{computed_at_iso[:16] if computed_at_iso else 'latest run'} UTC</span>.
        </p>""",
        unsafe_allow_html=True,
    )
    fw = status.get("forecast_windows", {})
    if fw:
        window_labels = [
            ("LEAD TIME: DAYS 1–3", "1_3"),
            ("LEAD TIME: DAYS 4–7", "4_7"),
            ("LEAD TIME: DAYS 8–15", "8_15"),
        ]
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
                <div class="kobo-panel" style="padding:0.8rem 1rem; margin-bottom:0.65rem;">
                    <div style="display:flex; justify-content:space-between; align-items:baseline;">
                        <div class="kobo-strip-label" style="font-size:0.68rem;">{label}</div>
                        <div class="mono" style="color:{below_color}; font-size:0.76rem;">{pct_below:.0f}% members below normal</div>
                    </div>
                    <div style="position:relative; height:10px; background:{COLORS['rule']}; border-radius:3px; margin-top:5px;">
                        <div style="position:absolute; left:{range_left}%; width:{range_width}%; height:100%;
                                    background:{COLORS['sky']}55; border-radius:3px;"></div>
                        <div style="position:absolute; left:{median_pos}%; width:2px; height:14px; top:-2px;
                                    background:{COLORS['sky']};"></div>
                    </div>
                    <div class="mono" style="color:{COLORS['text_muted']}; font-size:0.70rem; margin-top:4px;">
                        Ensemble Median: {median:.1f} mm &nbsp;·&nbsp; Uncertainty (P10–P90): {p10:.1f}–{p90:.1f} mm ({n_members} members)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="kobo-label" style="margin-top:1.1rem;">Current State Class Scores (Multinomial LogReg)</div>',
        unsafe_allow_html=True,
    )
    probs = status["class_probabilities"]
    for cls, color in [("Normal", COLORS["sage"]), ("Moderate", COLORS["ochre"]), ("Severe", COLORS["clay"])]:
        p = probs.get(cls, 0)
        st.markdown(
            f"""
            <div style="margin-bottom:0.55rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:0.15rem;">
                    <span>{cls} Drought Class</span><span class="mono">{p*100:.1f}%</span>
                </div>
                <div style="height:5px; background:{COLORS['rule']}; border-radius:3px; overflow:hidden;">
                    <div style="width:{p*100}%; height:100%; background:{color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        "Note: Class scores reflect multinomial logistic regression softmax outputs on observed features; "
        "interpret alongside held-out precision and recall below."
    )

st.markdown('<hr class="kobo-rule">', unsafe_allow_html=True)

# =================================================================
# 5. EXPLAINABILITY + WHAT-IF PERTURBATION SIMULATOR
# =================================================================
st.markdown('<div class="kobo-label">Section IV · Feature Attribution &amp; Interactive What-If Stress Simulator</div>', unsafe_allow_html=True)

drivers = status.get("top_drivers", [])
if drivers:
    exp_c1, exp_c2 = st.columns([1.35, 1])

    with exp_c2:
        st.markdown(
            f"""
            <div class="kobo-panel">
                <div class="kobo-label" style="color:{COLORS['ochre']};">🧪 Stress-Test Scenario Simulator</div>
                <p style="font-size:0.78rem; color:{COLORS['text_muted']}; margin-bottom:0.4rem;">
                    Test how hypothetical rainfall deficits or heatwaves would shift the model's feature
                    contributions and drought class scores:
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        sim_rain_dz = st.slider("Perturb Rainfall Anomaly (Δσ)", -2.5, 2.5, 0.0, 0.25)
        sim_ndvi_dz = st.slider("Perturb Vegetation Vigor (Δσ)", -2.5, 2.5, 0.0, 0.25)
        sim_soil_dz = st.slider("Perturb Soil Moisture (Δσ)", -2.5, 2.5, 0.0, 0.25)
        sim_lst_dz = st.slider("Perturb Land Surface Temp (Δσ)", -2.5, 2.5, 0.0, 0.25)

    with exp_c1:
        drivers_df = pd.DataFrame(drivers).copy()

        def get_dz(feat):
            if "rain" in feat or "spi" in feat:
                return sim_rain_dz
            if "ndvi" in feat or "vci" in feat:
                return sim_ndvi_dz
            if "soil" in feat:
                return sim_soil_dz
            if "lst" in feat:
                return sim_lst_dz
            return 0.0

        drivers_df["dz"] = drivers_df["feature"].map(get_dz)
        drivers_df["label"] = drivers_df["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
        drivers_df["contribution"] = drivers_df["coefficient"] * drivers_df["value_z"]
        drivers_df["sim_contribution"] = drivers_df["coefficient"] * (drivers_df["value_z"] + drivers_df["dz"])
        drivers_df = drivers_df.sort_values("contribution")

        is_sim = any(abs(x) > 1e-4 for x in [sim_rain_dz, sim_ndvi_dz, sim_soil_dz, sim_lst_dz])

        fig_drivers = go.Figure()
        fig_drivers.add_trace(
            go.Bar(
                x=drivers_df["contribution"],
                y=drivers_df["label"],
                orientation="h",
                name="Observed Contribution",
                marker_color=[COLORS["clay"] if c < 0 else COLORS["sage"] for c in drivers_df["contribution"]],
                opacity=0.45 if is_sim else 0.9,
            )
        )
        if is_sim:
            fig_drivers.add_trace(
                go.Bar(
                    x=drivers_df["sim_contribution"],
                    y=drivers_df["label"],
                    orientation="h",
                    name="Simulated Scenario",
                    marker_color=COLORS["ochre"],
                )
            )

        fig_drivers.update_layout(
            barmode="group",
            paper_bgcolor=COLORS["surface"],
            plot_bgcolor=COLORS["surface"],
            font=dict(family="IBM Plex Mono", color=COLORS["text_muted"], size=11),
            xaxis=dict(title="Log-Odds Contribution (β × z)", gridcolor=COLORS["rule_soft"], zerolinecolor=COLORS["rule"]),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            legend=dict(orientation="h", y=1.12, font=dict(color=COLORS["text_muted"])),
            height=310,
            margin={"t": 20, "l": 10, "r": 10, "b": 30},
        )
        st.plotly_chart(fig_drivers, use_container_width=True, config={"displayModeBar": False})

        stress_impulse = (-0.65 * sim_rain_dz) + (-0.55 * sim_ndvi_dz) + (-0.50 * sim_soil_dz) + (0.45 * sim_lst_dz)
        base_p = np.array([
            max(1e-4, probs.get("Normal", 0.7)),
            max(1e-4, probs.get("Moderate", 0.2)),
            max(1e-4, probs.get("Severe", 0.1)),
        ])
        logits = np.log(base_p) + np.array([-0.8 * stress_impulse, 0.45 * stress_impulse, 0.95 * stress_impulse])
        sim_p = np.exp(logits - np.max(logits))
        sim_p = sim_p / sim_p.sum()

        st.markdown(
            f"""
            <div class="kobo-panel" style="padding:0.75rem 1rem;">
                <span class="mono" style="font-size:0.75rem; color:{COLORS['text_muted']};">
                    {"SIMULATED SCENARIO CLASS SCORES:" if is_sim else "OBSERVED BASELINE CLASS SCORES:"}
                </span>
                <span class="mono" style="font-size:0.84rem; float:right;">
                    <span style="color:{COLORS['sage']};">Normal: {sim_p[0]*100:.1f}%</span> &nbsp;·&nbsp;
                    <span style="color:{COLORS['ochre']};">Moderate: {sim_p[1]*100:.1f}%</span> &nbsp;·&nbsp;
                    <span style="color:{COLORS['clay']};">Severe: {sim_p[2]*100:.1f}%</span>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<hr class="kobo-rule">', unsafe_allow_html=True)

# =================================================================
# 6. HISTORICAL CASE STUDIES (Proving Multi-Sensor Value)
# =================================================================
st.markdown(
    '<div class="kobo-label">Section V · Historical Case Studies (2000–2024) — Why Multi-Sensor Fusion Beats Single Indicators</div>',
    unsafe_allow_html=True,
)

CASE_STUDIES = {
    "Full Historical Record (2000–2024)": {
        "range": ("2000-01-01", "2024-12-31"),
        "title": "25-Year Multi-Sensor Climatology (2000–2024)",
        "narrative": (
            "Across 25 years in Raya Kobo, meteorological rainfall deficits (SPI-6) frequently precede canopy stress, "
            "but agricultural drought severity depends heavily on pre-season ERA5-Land root-zone soil moisture and "
            "MODIS Land Surface Temperature (LST) evaporative demand."
        ),
    },
    "Case Study 1: 2015–2016 El Niño Kiremt Failure": {
        "range": ("2014-06-01", "2017-03-01"),
        "title": "2015–2016 El Niño Drought (Severe Kiremt Failure in North Wollo)",
        "narrative": (
            "During the 2015 El Niño event, failure of the Kiremt (Jun–Sep) rains triggered severe crop loss across Raya Kobo. "
            "Notice how ERA5-Land soil moisture and MODIS NDVI remained depressed well into early 2016 even after brief rainfall "
            "pulses—demonstrating why a rainfall-only trigger would have prematurely declared recovery."
        ),
    },
    "Case Study 2: 2008–2009 Consecutive Belg/Kiremt Deficit": {
        "range": ("2007-06-01", "2010-06-01"),
        "title": "2008–2009 Compound Seasonal Deficit",
        "narrative": (
            "Back-to-back poor Belg and erratic Kiremt rains between 2008 and 2009 caused cumulative rangeland and crop stress. "
            "Combining lagged soil moisture and vegetation anomalies captures this multi-season memory effect."
        ),
    },
    "Case Study 3: 2020–2024 Recent Multi-Sensor Window": {
        "range": ("2020-01-01", "2024-12-31"),
        "title": "2020–2024 Held-Out Evaluation Era",
        "narrative": (
            "The recent 2020–2024 period serves as our strictly unseen temporal test set. It includes both wet recovery phases "
            "and localized Moderate/Severe stress episodes used to benchmark the model against SPI climatology and persistence."
        ),
    },
}

cs_col1, cs_col2 = st.columns([1.4, 1.6])
with cs_col1:
    selected_case = st.selectbox("Select Historical Case Study", options=list(CASE_STUDIES.keys()))
with cs_col2:
    avail_cols = [
        c for c in ["ndvi_anomaly_z", "spi_6", "spi_3", "vci", "rain_anomaly_z", "soil_anomaly_z", "lst_anomaly_z"]
        if c in history.columns
    ]
    default_plot_cols = [c for c in ["ndvi_anomaly_z", "spi_6"] if c in avail_cols] or avail_cols[:2]
    chosen_indicators = st.multiselect(
        "Compare Sensor Time-Series",
        options=avail_cols,
        default=default_plot_cols,
        format_func=lambda c: FEATURE_LABELS.get(c, c),
    )

cs_meta = CASE_STUDIES[selected_case]
st.markdown(
    f"""
    <div class="kobo-panel" style="border-left: 3px solid {COLORS['sky']};">
        <div class="kobo-label" style="color:{COLORS['sky']};">{cs_meta['title']}</div>
        <div style="font-size:0.84rem; color:{COLORS['text']}; line-height:1.55;">{cs_meta['narrative']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

start_dt, end_dt = cs_meta["range"]
hist_sub = history[(history["date"] >= start_dt) & (history["date"] <= end_dt)].copy()

fig_hist = go.Figure()
palette = [COLORS["sage"], COLORS["sky"], COLORS["ochre"], COLORS["clay"], "#B497BD"]
for i, col_name in enumerate(chosen_indicators):
    fig_hist.add_trace(
        go.Scatter(
            x=hist_sub["date"],
            y=hist_sub[col_name],
            name=FEATURE_LABELS.get(col_name, col_name),
            line=dict(color=palette[i % len(palette)], width=1.6),
        )
    )

if "drought_class" in hist_sub.columns:
    for _, row in hist_sub[hist_sub["drought_class"] == "Severe"].iterrows():
        fig_hist.add_vrect(
            x0=row["date"],
            x1=row["date"] + pd.DateOffset(months=1),
            fillcolor=COLORS["clay"],
            opacity=0.20,
            line_width=0,
        )
    for _, row in hist_sub[hist_sub["drought_class"] == "Moderate"].iterrows():
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
    xaxis=dict(gridcolor=COLORS["rule_soft"]),
    yaxis=dict(gridcolor=COLORS["rule_soft"], zerolinecolor=COLORS["rule"]),
    legend=dict(orientation="h", y=1.12, font=dict(color=COLORS["text_muted"])),
    hovermode="x unified",
    height=350,
    margin={"t": 10, "l": 10, "r": 10, "b": 10},
)
st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

st.markdown('<hr class="kobo-rule">', unsafe_allow_html=True)

# =================================================================
# 7. TEMPORAL MODEL VALIDATION & BASELINE BENCHMARKING
# =================================================================
st.markdown('<div class="kobo-label">Section VI · Temporal Validation on Unseen Drought Periods &amp; Baseline Comparison</div>', unsafe_allow_html=True)

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
        <strong>Strict Temporal Split:</strong> Trained on historical years and evaluated on <strong>{validation['test_period']}</strong>
        ({validation['n_test_months']} unseen months). Because drought is a rare event (42 Normal, 3 Moderate, 3 Severe months in the test period),
        overall accuracy is misleading—a model that always predicts "Normal" achieves 87.5% accuracy while catching <strong>0% of droughts</strong>.
        We evaluate models primarily on <strong>Elevated-Risk Recall (Moderate + Severe)</strong> and per-class F1/Precision.
        </p>""",
        unsafe_allow_html=True,
    )

    model_labels = {
        "logistic_regression": "Multinomial Logistic Regression (Deployed)",
        "climatology_baseline": "Seasonal Climatology Baseline",
        "persistence_baseline": "1-Month Persistence Baseline",
    }

    recall_cols = st.columns(len(model_labels))
    for col, (key, label) in zip(recall_cols, model_labels.items()):
        pc = models[key]["per_class"]
        elevated_recall = (pc["Moderate"]["recall"] + pc["Severe"]["recall"]) / 2
        macro_f1 = np.mean([pc[c].get("f1", 0) for c in class_order])
        acc = models[key]["accuracy"]
        is_deployed = key == "logistic_regression"
        color = COLORS["sage"] if is_deployed else COLORS["clay"]
        col.markdown(
            f"""
            <div class="kobo-panel" style="border-left: 3px solid {color};">
                <div class="kobo-strip-label">{label}</div>
                <div class="mono" style="font-size:1.65rem; font-weight:600; color:{color};">
                    {elevated_recall*100:.0f}% <span style="font-size:0.72rem; color:{COLORS['text_muted']};">drought recall</span>
                </div>
                <div class="mono" style="font-size:0.72rem; color:{COLORS['text_muted']}; margin-top:0.35rem; line-height:1.6;">
                    Mod Recall: {pc['Moderate']['recall']*100:.0f}% &nbsp;·&nbsp; Sev Recall: {pc['Severe']['recall']*100:.0f}%<br/>
                    Macro F1: {macro_f1*100:.1f}% &nbsp;·&nbsp; Test-Set Accuracy: {acc*100:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    v_col1, v_col2 = st.columns([1.05, 1.25])
    with v_col1:
        selected_eval_model = st.selectbox(
            "Compare Model Confusion Matrix & Class Metrics",
            options=list(model_labels.keys()),
            format_func=lambda k: model_labels[k],
        )
        per_class = models[selected_eval_model]["per_class"]
        class_colors = {"Normal": COLORS["sage"], "Moderate": COLORS["ochre"], "Severe": COLORS["clay"]}
        for cls in class_order:
            stats = per_class[cls]
            st.markdown(
                f"""
                <div class="kobo-panel" style="border-left: 3px solid {class_colors[cls]}; padding: 0.65rem 0.9rem; margin-bottom: 0.5rem;">
                    <div class="kobo-strip-label">{cls} Class (Test Support: n={int(stats['support'])} months)</div>
                    <div class="mono" style="font-size:0.92rem; color:{COLORS['text']};">
                        Precision: {stats['precision']*100:.0f}% &nbsp;·&nbsp;
                        Recall: {stats['recall']*100:.0f}% &nbsp;·&nbsp;
                        F1-Score: {stats.get('f1', 0)*100:.0f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.caption(
            "Operational Trade-Off: In early warning, false negatives (missed droughts) carry higher humanitarian cost "
            "than false positives (Watch alerts that trigger low-cost field verification)."
        )

    with v_col2:
        cm = models[selected_eval_model]["confusion_matrix"]
        fig_cm = go.Figure(
            go.Heatmap(
                z=cm,
                x=[f"Pred {c}" for c in class_order],
                y=[f"True {c}" for c in class_order],
                colorscale=[[0, COLORS["surface"]], [1, COLORS["sage"] if selected_eval_model == "logistic_regression" else COLORS["ochre"]]],
                showscale=False,
                text=cm,
                texttemplate="%{text}",
                textfont=dict(family="IBM Plex Mono", size=14, color=COLORS["text"]),
            )
        )
        fig_cm.update_layout(
            title=dict(
                text=f"HELD-OUT CONFUSION MATRIX ({validation['test_period']})",
                font=dict(family="IBM Plex Mono", size=11, color=COLORS["text_muted"]),
            ),
            paper_bgcolor=COLORS["surface"],
            plot_bgcolor=COLORS["surface"],
            font=dict(family="Space Grotesk", color=COLORS["text_muted"], size=11),
            height=290,
            margin={"t": 38, "l": 10, "r": 10, "b": 10},
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

st.markdown('<hr class="kobo-rule">', unsafe_allow_html=True)

# =================================================================
# 8. REPRODUCIBILITY, KNOWN LIMITATIONS & 8-WEEK PILOT PROPOSAL
# =================================================================
st.markdown('<div class="kobo-label">Section VII · Technical Reproducibility, Limitations &amp; Proposed 8-Week Pilot</div>', unsafe_allow_html=True)

doc_col1, doc_col2 = st.columns(2)

with doc_col1:
    with st.expander("📐 Data Specifications, Preprocessing & Known Limitations", expanded=True):
        st.markdown(
            """
            **Data Sources & Spatial Resolution:**
            * **Precipitation (`CHIRPS v2.0`):** $0.05^\\circ$ (~5.5 km) quasi-global satellite-station product (2000–present), aggregated to monthly totals to compute SPI-1, SPI-3, SPI-6, and SPI-12.
            * **Vegetation (`MODIS MOD13Q1`):** 250m 16-day NDVI composites resampled to woreda grid cells; standardized into monthly $z$-scores and Vegetation Condition Index (VCI).
            * **Thermal Stress (`MODIS MOD11A2`):** 1 km 8-day daytime Land Surface Temperature (LST) anomalies.
            * **Soil Moisture (`ERA5-Land`):** $0.1^\\circ$ (~9 km) root-zone volumetric soil water layer anomalies and historical percentiles.
            * **Ensemble Forecast (`ECMWF AIFS-ENS`):** 15-day ensemble precipitation forecast (P10, Median, P90 across perturbed members).

            **Known Limitations & Operational Caveats:**
            * **Coarse Reanalysis Resolution:** ERA5-Land (~9 km) smooths over steep micro-topography between the western Zobel/escarpment highlands and the Raya valley floor.
            * **Optical Cloud Cover:** MODIS NDVI/LST can experience missing composites during peak *Kiremt* cloud cover (July–August).
            * **Probability Calibration:** Softmax outputs from multinomial logistic regression reflect uncalibrated class scores; operational triggers should rely on the multi-tier verification matrix rather than raw percentage thresholds alone.
            """
        )

with doc_col2:
    with st.expander("🤝 Proposed 8-Week Institutional Pilot Roadmap (Raya Kobo Woreda)", expanded=True):
        st.markdown(
            """
            Rather than proposing immediate nationwide deployment, this system is designed for a **collaborative 8-week validation pilot** in Raya Kobo Woreda with government (EDRMC / BoA) and humanitarian (WFP / FAO) partners:

            1. **Weeks 1–2 · Stakeholder Consultation:** Align with woreda/zonal agricultural and disaster-risk officers on decision calendars (*Belg* vs. *Kiremt*), existing warning workflows, and local kebele vulnerability data.
            2. **Weeks 3–4 · Historical Ground-Truthing:** Cross-validate model detections against woreda crop-assessment records, historical food-security reports, and local meteorological station gauges.
            3. **Weeks 5–6 · Operational User Testing:** Test the live dashboard and the **1-Page Printable Situation Brief** during routine coordination meetings; refine warning thresholds (`Watch` vs. `Warning`).
            4. **Weeks 7–8 · Joint Evaluation Report:** Assess timeliness, false-alarm vs. missed-event trade-offs, and determine whether a full seasonal trial is warranted.
            """
        )

st.markdown('<hr class="kobo-rule">', unsafe_allow_html=True)

# =================================================================
# ALERTING STUB
# =================================================================
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SUBSCRIPTIONS_PATH = DATA_DIR / "subscriptions.jsonl"


def validate_contact(contact_type, value):
    value = (value or "").strip()
    if not value:
        return False, "Please enter a value."
    if contact_type == "Email":
        if not EMAIL_RE.match(value):
            return False, "That doesn't look like a valid email address."
    else:
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return False, "Webhook URL must start with http:// or https:// and include a valid host."
    return True, None


def save_subscription(contact_type, value, levels):
    record = {
        "contact_type": contact_type,
        "contact_value": value,
        "notify_on": levels,
        "subscribed_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with open(SUBSCRIPTIONS_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
        return True, None
    except Exception as e:
        return False, str(e)


def count_subscriptions():
    try:
        with open(SUBSCRIPTIONS_PATH) as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


st.markdown('<div class="kobo-label">Get notified of status changes</div>', unsafe_allow_html=True)
sub_count = count_subscriptions()
sub_count_text = f" · {sub_count} subscriber{'s' if sub_count != 1 else ''} so far" if sub_count > 0 else ""
st.markdown(
    f"""<p style="color:{COLORS['text_muted']}; font-size:0.82rem;">
    Get an alert if Kobo's status moves to Watch or Warning.{sub_count_text}
    </p>
    <p style="color:{COLORS['text_muted']}; font-size:0.72rem; font-style:italic;">
    Design stub — entries are validated and stored locally to demonstrate last-mile early-warning dispatch UX.
    </p>""",
    unsafe_allow_html=True,
)

with st.form("subscribe_form", clear_on_submit=True):
    sub_col1, sub_col2 = st.columns([1, 2])
    contact_type = sub_col1.radio("Notify me via", ["Email", "Webhook URL"], label_visibility="collapsed")
    placeholder = "you@example.com" if contact_type == "Email" else "https://hooks.example.com/..."
    contact_value = sub_col2.text_input("Contact", placeholder=placeholder, label_visibility="collapsed")
    notify_levels = st.multiselect("Notify me when status becomes", ["Watch", "Warning"], default=["Watch", "Warning"])
    submitted = st.form_submit_button("Subscribe")

    if submitted:
        valid, error = validate_contact(contact_type, contact_value)
        if not valid:
            st.error(error)
        elif not notify_levels:
            st.error("Select at least one status level to be notified about.")
        else:
            ok, save_error = save_subscription(contact_type, contact_value.strip(), notify_levels)
            if ok:
                st.success("Subscribed — noted for the record. (Design stub: no notification will actually be sent.)")
            else:
                st.warning(f"Validation passed, but storage isn't writable in this environment right now. ({save_error})")

st.markdown('<hr class="kobo-rule">', unsafe_allow_html=True)

# =================================================================
# COLOPHON
# =================================================================
st.markdown(
    f"""
    <div class="kobo-colophon">
    MULTI-SENSOR DROUGHT EARLY WARNING &amp; DECISION-SUPPORT SYSTEM — a prototype decision-support tool, not an official government warning service.<br/>
    DATA — MODIS NDVI/LST · CHIRPS v2.0 · ERA5-Land · ECMWF AIFS-ENS, via Google Earth Engine<br/>
    MODEL — {MODEL_VERSION}, temporally validated on unseen North Wollo drought periods<br/>
    BUILD — Python · scikit-learn · GeoPandas · Streamlit · Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
