"""
KOBO DROUGHT EARLY WARNING SYSTEM
Redesigned dashboard — "field instrument panel" aesthetic

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

STATUS_COLOR = {"Normal": COLORS["sage"], "Watch": COLORS["ochre"], "Warning": COLORS["clay"]}

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
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }}

    h1, h2, h3 {{
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
        margin: 2rem 0;
    }}

    /* Section label — functional instrument-placard style, used sparingly */
    .kobo-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        color: {COLORS['text_muted']};
        text-transform: uppercase;
        margin-bottom: 0.6rem;
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
        border-left: 3px solid var(--status-glow, {COLORS['rule']});
        border-radius: 4px;
        padding: 2.6rem 2.8rem;
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

    /* Indicator strip */
    .kobo-strip {{
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1.5rem;
        padding: 1.4rem 0;
    }}
    .kobo-strip-item {{
        flex: 1;
        min-width: 130px;
    }}
    .kobo-strip-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.06em;
        color: {COLORS['text_muted']};
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }}
    .kobo-strip-value {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.5rem;
        font-weight: 500;
    }}

    /* Streamlit expander restyle */
    div[data-testid="stExpander"] {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['rule']};
        border-radius: 4px;
    }}

    /* Footer colophon */
    .kobo-colophon {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: {COLORS['text_muted']};
        line-height: 1.8;
    }}

    @media (max-width: 768px) {{
        .kobo-hero {{ padding: 1.6rem 1.4rem; }}
        .kobo-strip {{ flex-direction: column; gap: 0.9rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =================================================================
# LOAD DATA
# =================================================================
# Single call site for which region this deployment shows. The loader
# functions and get_cached_status() below are fully parameterized by
# region_id -- this constant is the only place "kobo" is hardcoded for
# the UI layer. A second region wouldn't need a second UI (per scope),
# just a second value here (or a selector, if that's ever wanted later).
REGION_ID = "kobo"


@st.cache_data
def load_history(region_id):
    df = pd.read_csv(DATA_DIR / f"{region_id}_dashboard_history.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_grid(region_id):
    return gpd.read_file(DATA_DIR / f"{region_id}_grid_performance.geojson")


try:
    history = load_history(REGION_ID)
    grid = load_grid(REGION_ID)
except FileNotFoundError as e:
    st.error(f"Reference data not found. Make sure `data/` sits alongside this file.\n\nDetails: {e}")
    st.stop()

status, is_live, is_stale, read_error = get_cached_status(REGION_ID)
if status is None:
    st.error(f"No drought status data available.\n\nDetails: {read_error}")
    st.stop()

live_ok = is_live and not is_stale

status_color = STATUS_COLOR.get(status["status"], COLORS["text_muted"])

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
        <span id="kobo-live-text">{"LIVE" if live_ok else ("STALE — refresh job may be delayed" if is_live else "SNAPSHOT — refresh job hasn't run yet")}</span>
      </div>
      <h1 style="margin:0; font-size:1.95rem; line-height:1.2;">Multi-Sensor Drought<br/>Early Warning System</h1>
      <p class="mono" style="color:{COLORS['text_muted']}; font-size:0.85rem; margin-top:0.8rem;">
        RAYA KOBO WOREDA &nbsp;·&nbsp; NORTH WOLLO ZONE &nbsp;·&nbsp; AMHARA, ETHIOPIA<br/>
        12.11°N &nbsp;39.65°E &nbsp;·&nbsp; 1930 km²
      </p>
    </div>
    <div style="text-align:right;">
      <div class="kobo-label">Current status</div>
      <div class="mono" style="font-size:3.2rem; font-weight:600; color:{status_color}; line-height:1;">
        {status['status'].upper()}
      </div>
      <p style="color:{COLORS['text_muted']}; font-size:0.9rem; margin-top:0.4rem;">
        {status['current_class']} drought class &nbsp;·&nbsp; {status['current_confidence']*100:.0f}% model confidence
      </p>
    </div>
  </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# Live-ticking "updated Xs ago" counter + next-refresh progress bar — the
# clearest signal to a first-time visitor that this page is genuinely
# recomputing, not a static export.
if computed_at_iso:
    refresh_interval_seconds = status.get("refresh_interval_seconds", 6 * 3600)
    components.html(
        f"""
        <html>
        <head>
        <style>
            html, body {{
                margin: 0; padding: 0;
                background: {COLORS['bg']};
            }}
            #kobo-ago-wrap {{
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.72rem;
                letter-spacing: 0.08em;
                color: {COLORS['text_muted']};
                text-transform: uppercase;
                padding: 4px 0 3px 20px;
            }}
            #kobo-progress-wrap {{ padding: 0 20px 4px 20px; max-width: 280px; }}
            #kobo-progress-bar-bg {{
                height: 3px; background: {COLORS['rule']}; border-radius: 2px; overflow: hidden;
            }}
            #kobo-progress-bar-fg {{
                height: 100%; background: {live_dot_color}; width: 0%;
            }}
            #kobo-next-text {{
                font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
                color: {COLORS['text_muted']}; padding: 3px 0 0 20px;
            }}
        </style>
        </head>
        <body>
        <div id="kobo-ago-wrap">Updated <span id="kobo-ago" style="color:{live_dot_color}; font-weight:600;">just now</span></div>
        <div id="kobo-progress-wrap"><div id="kobo-progress-bar-bg"><div id="kobo-progress-bar-fg"></div></div></div>
        <div id="kobo-next-text">~Next scheduled refresh in <span id="kobo-next">—</span> <span style="opacity:0.6;">(external job, approximate)</span></div>
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
        height=58,
    )

if not live_ok:
    with st.expander("Why isn't this live right now?"):
        if is_live and is_stale:
            st.write(
                "A live status does exist, but it's older than expected — the scheduled refresh job "
                "(which runs independently of this app, roughly every 6 hours) may have failed or been "
                "delayed on its last run. The numbers shown are the last successful computation, not "
                "invented or interpolated."
            )
        else:
            st.write(
                "This app reads its status from a file written by a separate scheduled job (Earth Engine "
                "+ ECMWF AIFS run on a cron schedule, decoupled from this app's request path) — that job "
                "hasn't produced a result yet, so this is a bundled fallback snapshot instead."
            )

# =================================================================
# INDICATOR STRIP
# =================================================================
vec = status["current_vector"]


def safe_delta(cur, lag):
    if cur is None or lag is None:
        return None
    return cur - lag


metrics = [
    {"label": "NDVI anomaly", "value": vec.get("ndvi_anomaly_z"), "suffix": "σ", "decimals": 2, "sign": True,
     "color": COLORS["sage"], "delta": safe_delta(vec.get("ndvi_anomaly_z"), vec.get("ndvi_anomaly_z_lag1"))},
    {"label": "VCI", "value": vec.get("vci"), "suffix": "", "decimals": 2, "sign": False,
     "color": COLORS["sage"], "delta": safe_delta(vec.get("vci"), vec.get("vci_lag1"))},
    {"label": "Rainfall anomaly", "value": vec.get("rain_anomaly_z"), "suffix": "σ", "decimals": 2, "sign": True,
     "color": COLORS["sky"], "delta": safe_delta(vec.get("rain_anomaly_z"), vec.get("rain_anomaly_z_lag1"))},
    {"label": "LST anomaly", "value": vec.get("lst_anomaly_z"), "suffix": "σ", "decimals": 2, "sign": True,
     "color": COLORS["clay"], "delta": safe_delta(vec.get("lst_anomaly_z"), vec.get("lst_anomaly_z_lag1"))},
    {"label": "Soil moisture anomaly", "value": vec.get("soil_anomaly_z"), "suffix": "σ", "decimals": 2, "sign": True,
     "color": COLORS["ochre"], "delta": safe_delta(vec.get("soil_anomaly_z"), vec.get("soil_anomaly_z_lag1"))},
    {"label": "15-day forecast", "value": status.get("forecast_pct_below_normal"), "suffix": "% below normal", "decimals": 0, "sign": False,
     "color": COLORS["sky"], "delta": None},
]
metrics_json = json.dumps(metrics)

components.html(
    f"""
    <html><head><style>
    html, body {{ margin:0; padding:0; background:{COLORS['bg']}; font-family:'Space Grotesk',sans-serif; }}
    #kobo-strip-container {{
        display:flex; justify-content:space-between; flex-wrap:wrap; gap:1.5rem; padding: 6px 0;
    }}
    .kobo-strip-item {{ flex:1; min-width:130px; }}
    .kobo-strip-label {{
        font-family:'IBM Plex Mono',monospace; font-size:0.68rem; letter-spacing:0.06em;
        color:{COLORS['text_muted']}; text-transform:uppercase; margin-bottom:0.3rem;
    }}
    .kobo-strip-value {{ font-family:'IBM Plex Mono',monospace; font-size:1.5rem; font-weight:500; }}
    .kobo-strip-delta {{
        font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:{COLORS['text_muted']}; margin-top:0.2rem;
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

        if (m.delta !== null && m.delta !== undefined) {{
            const deltaDiv = document.createElement("div");
            deltaDiv.className = "kobo-strip-delta";
            const arrow = m.delta > 0.01 ? "\u25B2" : (m.delta < -0.01 ? "\u25BC" : "\u25AC");
            deltaDiv.textContent = arrow + " " + Math.abs(m.delta).toFixed(m.decimals) + " vs last month";
            item.appendChild(deltaDiv);
        }}

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
        animateValue("kobo-val-" + i, m.value, m.decimals, m.suffix, m.sign, 700 + i * 80);
    }});
    </script>
    </body></html>
    """,
    height=125,
)
st.markdown('<hr class="kobo-rule">', unsafe_allow_html=True)

st.markdown(
    f"""<p style="color:{COLORS['text_muted']}; font-size:0.95rem;">
    {status['forecast_note']}
    </p>""",
    unsafe_allow_html=True,
)

def freshness_indicator(date_str, green_days, amber_days):
    if not date_str or date_str == "unknown":
        return f'<span style="color:{COLORS["text_muted"]}">●</span> unknown'
    try:
        d = pd.to_datetime(date_str)
        now = pd.Timestamp.utcnow().tz_localize(None)
        age_days = (now - d).days
    except Exception:
        return f'<span style="color:{COLORS["text_muted"]}">●</span> {date_str}'
    if age_days <= green_days:
        color = COLORS["sage"]
    elif age_days <= amber_days:
        color = COLORS["ochre"]
    else:
        color = COLORS["clay"]
    return f'<span style="color:{color}">●</span> {date_str} <span style="color:{COLORS["text_muted"]}">({age_days}d ago)</span>'


with st.expander("Data currency — sources update at different real-world speeds"):
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<span class='mono'>NDVI</span><br/>{freshness_indicator(status.get('ndvi_date'), 20, 40)}", unsafe_allow_html=True)
    c2.markdown(f"<span class='mono'>LST</span><br/>{freshness_indicator(status.get('lst_date'), 15, 30)}", unsafe_allow_html=True)
    c3.markdown(f"<span class='mono'>Rainfall / soil moisture</span><br/>{freshness_indicator(status.get('rain_soil_date'), 45, 75)}", unsafe_allow_html=True)

st.markdown('<hr class="kobo-rule">', unsafe_allow_html=True)

# =================================================================
# SPATIAL RISK MAP + FORECAST WINDOWS (two columns)
# =================================================================
col_map, col_forecast = st.columns([1.3, 1])

with col_map:
    st.markdown('<div class="kobo-label">Spatial risk pattern</div>', unsafe_allow_html=True)
    geojson = json.loads(grid.to_json())

    with open(DATA_DIR / f"{REGION_ID}_boundary.geojson") as f:
        boundary_geojson = json.load(f)

    fig_map = go.Figure(
        go.Choroplethmap(
            geojson=geojson,
            locations=grid.index,
            z=grid["accuracy"],
            customdata=grid.index,
            colorscale=[
                [0.0, COLORS["clay"]],
                [0.35, "#A8683F"],
                [0.6, COLORS["ochre"]],
                [0.8, "#8FA377"],
                [1.0, COLORS["sage"]],
            ],
            zmin=0.70, zmax=0.95,
            marker_opacity=0.80,
            marker_line_width=0.6,
            marker_line_color=COLORS["bg"],
            hovertemplate="<b>CELL %{customdata}</b><br>Accuracy: %{z:.3f}<extra></extra>",
            colorbar=dict(
                title=dict(text="ACCURACY", font=dict(color=COLORS["text_muted"], size=10, family="IBM Plex Mono")),
                tickfont=dict(color=COLORS["text_muted"], size=9, family="IBM Plex Mono"),
                thickness=10, len=0.65, outlinewidth=0,
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
                fig_map.add_trace(go.Scattermap(
                    lon=lons, lat=lats, mode="lines",
                    line=dict(width=2, color=COLORS["text"]),
                    opacity=0.55, hoverinfo="skip", showlegend=False,
                ))

    fig_map.update_layout(
        map_style="carto-darkmatter",
        map_zoom=8.4,
        map_center={"lat": 12.1, "lon": 39.65},
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=400,
        font=dict(family="Space Grotesk", color=COLORS["text_muted"]),
        hoverlabel=dict(
            bgcolor=COLORS["surface"], bordercolor=COLORS["rule"],
            font=dict(family="IBM Plex Mono", size=11, color=COLORS["text"]),
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f"""<p style="color:{COLORS['text_muted']}; font-size:0.78rem;">
        Model accuracy by cell (validation period). Drought frequency is fairly uniform across
        Kobo by construction — accuracy is not.
        </p>""",
        unsafe_allow_html=True,
    )

with col_forecast:
    st.markdown('<div class="kobo-label">15-day forecast · ECMWF AIFS ENSEMBLE</div>', unsafe_allow_html=True)
    fw = status.get("forecast_windows", {})
    if fw:
        window_labels = [("1–3 DAYS", "1_3"), ("4–7 DAYS", "4_7"), ("8–15 DAYS", "8_15")]
        # Shared scale across all three windows so the bars are comparable
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

            below_color = COLORS["clay"] if pct_below > 65 else (COLORS["ochre"] if pct_below > 35 else COLORS["sage"])

            st.markdown(
                f"""
                <div style="margin-bottom:1.3rem;">
                    <div style="display:flex; justify-content:space-between; align-items:baseline;">
                        <div class="kobo-strip-label">{label}</div>
                        <div class="mono" style="color:{below_color}; font-size:0.78rem;">{pct_below:.0f}% below normal</div>
                    </div>
                    <div style="position:relative; height:10px; background:{COLORS['rule']}; border-radius:3px; margin-top:4px;">
                        <div style="position:absolute; left:{range_left}%; width:{range_width}%; height:100%;
                                    background:{COLORS['sky']}55; border-radius:3px;"></div>
                        <div style="position:absolute; left:{median_pos}%; width:2px; height:14px; top:-2px;
                                    background:{COLORS['sky']};"></div>
                    </div>
                    <div class="mono" style="color:{COLORS['text_muted']}; font-size:0.72rem; margin-top:3px;">
                        median {median:.1f} mm &nbsp;·&nbsp; range {p10:.1f}–{p90:.1f} mm ({n_members} members)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""<p style="color:{COLORS['text_muted']}; font-size:0.78rem; margin-top:0.5rem;">
            Ensemble spread across {fw.get('1_3', {}).get('n_members', 11)} AIFS-ENS members (control + a subset
            of perturbed members) — the shaded band is the 10th–90th percentile range, the line marks the median.
            </p>""",
            unsafe_allow_html=True,
        )
    else:
        st.write("Forecast data unavailable.")

    st.markdown('<div class="kobo-label" style="margin-top:1.8rem;">Class probability</div>', unsafe_allow_html=True)
    probs = status["class_probabilities"]
    for cls, color in [("Normal", COLORS["sage"]), ("Moderate", COLORS["ochre"]), ("Severe", COLORS["clay"])]:
        p = probs.get(cls, 0)
        st.markdown(
            f"""
            <div style="margin-bottom:0.7rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:0.2rem;">
                    <span>{cls}</span><span class="mono">{p*100:.0f}%</span>
                </div>
                <div style="height:5px; background:{COLORS['rule']}; border-radius:3px; overflow:hidden;">
                    <div style="width:{p*100}%; height:100%; background:{color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<hr class="kobo-rule">', unsafe_allow_html=True)

# =================================================================
# EXPLAINABILITY
# =================================================================
st.markdown('<div class="kobo-label">What\'s driving this prediction</div>', unsafe_allow_html=True)
FEATURE_LABELS = {
    "ndvi_anomaly_z": "NDVI anomaly",
    "vci": "VCI",
    "rainfall_mm": "Rainfall",
    "rain_anomaly_z": "Rainfall anomaly",
    "spi_1": "SPI-1",
    "spi_3": "SPI-3",
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

drivers = status.get("top_drivers", [])
if drivers:
    drivers_df = pd.DataFrame(drivers)
    drivers_df["feature"] = drivers_df["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
    drivers_df["contribution"] = drivers_df["coefficient"] * drivers_df["value_z"]
    drivers_df = drivers_df.sort_values("contribution")

    fig_drivers = go.Figure(
        go.Bar(
            x=drivers_df["contribution"],
            y=drivers_df["feature"],
            orientation="h",
            marker_color=[COLORS["clay"] if c < 0 else COLORS["sage"] for c in drivers_df["contribution"]],
        )
    )
    fig_drivers.update_layout(
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(family="IBM Plex Mono", color=COLORS["text_muted"], size=11),
        xaxis=dict(gridcolor=COLORS["rule_soft"], zerolinecolor=COLORS["rule"]),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=280,
        margin={"t": 10, "l": 10, "r": 10, "b": 30},
    )
    st.plotly_chart(fig_drivers, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f"""<p style="color:{COLORS['text_muted']}; font-size:0.78rem;">
        Multinomial logistic regression — a feature's contribution to one class is relative to the
        other two, not a standalone cause.
        </p>""",
        unsafe_allow_html=True,
    )

st.markdown('<hr class="kobo-rule">', unsafe_allow_html=True)

# =================================================================
# MODEL VALIDATION
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
        Evaluated on {validation['test_period']} ({validation['n_test_months']} months) — a period never
        used for model selection or tuning (that was the separate validation period's job).
        </p>""",
        unsafe_allow_html=True,
    )

    model_labels = {
        "logistic_regression": "Logistic regression (deployed)",
        "climatology_baseline": "Climatology baseline",
        "persistence_baseline": "Persistence baseline",
    }

    # Elevated-risk recall (Moderate + Severe) is the metric that actually matters for a
    # warning system — overall accuracy alone rewards a model that just predicts "Normal"
    # for a rare-event problem like this one. Lead with this, not raw accuracy.
    st.markdown(
        f"""<p style="color:{COLORS['text_muted']}; font-size:0.78rem;">
        Elevated-risk recall (Moderate + Severe) — the metric that matters for a warning system
        </p>""",
        unsafe_allow_html=True,
    )
    recall_cols = st.columns(len(model_labels))
    for col, (key, label) in zip(recall_cols, model_labels.items()):
        pc = models[key]["per_class"]
        elevated_recall = (pc["Moderate"]["recall"] + pc["Severe"]["recall"]) / 2
        is_deployed = key == "logistic_regression"
        color = COLORS["sage"] if is_deployed else COLORS["clay"]
        col.markdown(
            f"""<div class="kobo-strip-label">{label}</div>
            <div class="mono" style="font-size:1.7rem; font-weight:{'600' if is_deployed else '400'}; color:{color};">
                {elevated_recall*100:.0f}%
            </div>
            <div style="font-size:0.7rem; color:{COLORS['text_muted']};">
                Mod: {pc['Moderate']['recall']*100:.0f}% &nbsp;·&nbsp; Sev: {pc['Severe']['recall']*100:.0f}%
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        f"""<p style="color:{COLORS['text_muted']}; font-size:0.78rem;">
        Overall accuracy — shown for completeness, but read with caution: on an imbalanced test
        set like this one (42 Normal / 3 Moderate / 3 Severe months), a model that simply predicts
        "Normal" nearly every time scores well here while catching zero real drought events.
        The recall numbers above are the honest comparison.
        </p>""",
        unsafe_allow_html=True,
    )
    acc_cols = st.columns(len(model_labels))
    for col, (key, label) in zip(acc_cols, model_labels.items()):
        acc = models[key]["accuracy"]
        is_deployed = key == "logistic_regression"
        color = COLORS["text"] if is_deployed else COLORS["text_muted"]
        col.markdown(
            f"""<div class="kobo-strip-label">{label}</div>
            <div class="mono" style="font-size:1.2rem; color:{color};">
                {acc*100:.1f}%
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown(
        f"""<p style="color:{COLORS['text_muted']}; font-size:0.78rem; margin-top:0.4rem;">
        Per-class performance — deployed model
        </p>""",
        unsafe_allow_html=True,
    )
    per_class = models["logistic_regression"]["per_class"]
    class_colors = {"Normal": COLORS["sage"], "Moderate": COLORS["ochre"], "Severe": COLORS["clay"]}
    pc_cols = st.columns(3)
    for col, cls in zip(pc_cols, class_order):
        stats = per_class[cls]
        col.markdown(
            f"""
            <div style="border-left: 3px solid {class_colors[cls]}; padding-left: 0.8rem;">
                <div class="kobo-strip-label">{cls} (n={int(stats['support'])})</div>
                <div class="mono" style="font-size:1.1rem; color:{COLORS['text']};">
                    P {stats['precision']*100:.0f}% &nbsp; R {stats['recall']*100:.0f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    cm = models["logistic_regression"]["confusion_matrix"]
    fig_cm = go.Figure(
        go.Heatmap(
            z=cm,
            x=[f"Pred {c}" for c in class_order],
            y=[f"True {c}" for c in class_order],
            colorscale=[[0, COLORS["surface"]], [1, COLORS["sage"]]],
            showscale=False,
            text=cm,
            texttemplate="%{text}",
            textfont=dict(family="IBM Plex Mono", size=14, color=COLORS["text"]),
        )
    )
    fig_cm.update_layout(
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(family="Space Grotesk", color=COLORS["text_muted"], size=11),
        height=280,
        margin={"t": 20, "l": 10, "r": 10, "b": 10},
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f"""<p style="color:{COLORS['text_muted']}; font-size:0.78rem;">
        Confusion matrix, deployed model, held-out test period. Rows are the true class, columns the
        model's prediction.
        </p>""",
        unsafe_allow_html=True,
    )
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
# HISTORICAL RECORD
# =================================================================
st.markdown('<div class="kobo-label">Historical record · 2000–2024</div>', unsafe_allow_html=True)

fig_hist = go.Figure()
fig_hist.add_trace(go.Scatter(
    x=history["date"], y=history["ndvi_anomaly_z"], name="NDVI anomaly",
    line=dict(color=COLORS["sage"], width=1.4),
))
fig_hist.add_trace(go.Scatter(
    x=history["date"], y=history["spi_6"], name="SPI-6",
    line=dict(color=COLORS["sky"], width=1.4),
))

for _, row in history[history["drought_class"] == "Severe"].iterrows():
    fig_hist.add_vrect(x0=row["date"], x1=row["date"] + pd.DateOffset(months=1),
                        fillcolor=COLORS["clay"], opacity=0.18, line_width=0)
for _, row in history[history["drought_class"] == "Moderate"].iterrows():
    fig_hist.add_vrect(x0=row["date"], x1=row["date"] + pd.DateOffset(months=1),
                        fillcolor=COLORS["ochre"], opacity=0.12, line_width=0)

fig_hist.update_layout(
    paper_bgcolor=COLORS["surface"],
    plot_bgcolor=COLORS["surface"],
    font=dict(family="Space Grotesk", color=COLORS["text_muted"], size=11),
    xaxis=dict(gridcolor=COLORS["rule_soft"]),
    yaxis=dict(gridcolor=COLORS["rule_soft"]),
    legend=dict(orientation="h", y=1.12, font=dict(color=COLORS["text_muted"])),
    height=340,
    margin={"t": 10, "l": 10, "r": 10, "b": 10},
)
st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
st.markdown(
    f"""<p style="color:{COLORS['text_muted']}; font-size:0.78rem;">
    Shaded bands: classified drought months (clay = Severe, ochre = Moderate), from SPI-6.
    </p>""",
    unsafe_allow_html=True,
)

st.markdown('<hr class="kobo-rule">', unsafe_allow_html=True)

# =================================================================
# ALERTING STUB — validates and stores a subscription; does not send anything.
# This demonstrates the last-mile UX, not a production notification pipeline.
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
    Design stub — entries are validated and stored, but no notification is actually sent yet,
    and storage here is local to this deployment (it will not survive the next redeploy).
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
    MULTI-SENSOR DROUGHT EARLY WARNING SYSTEM — a portfolio project, not an official warning service.<br/>
    DATA — MODIS NDVI/LST · CHIRPS · ERA5-Land · ECMWF AIFS, via Google Earth Engine<br/>
    MODEL — logistic regression, validated against published North Wollo drought literature<br/>
    BUILD — Python · scikit-learn · Streamlit · Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
