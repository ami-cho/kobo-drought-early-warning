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
        "livestock": "Assess browse and forage deficits in eastern lowland grazing corridors.",
    }
]
