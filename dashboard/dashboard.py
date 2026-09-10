"""
KOBO DROUGHT EARLY WARNING SYSTEM
Redesigned dashboard — "field instrument panel" aesthetic

Run locally:
    streamlit run dashboard.py

Expects a `data/` folder alongside this file (see README / DATA_SOURCES.md).
"""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from live_status import compute_live_status

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(
    page_title="Kobo Drought Early Warning",
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
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        animation: kobo-pulse 2.4s ease-in-out infinite;
    }}
    @keyframes kobo-pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.35; }}
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
@st.cache_data
def load_history():
    df = pd.read_csv(DATA_DIR / "kobo_dashboard_history.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_grid():
    return gpd.read_file(DATA_DIR / "kobo_grid_performance.geojson")


try:
    history = load_history()
    grid = load_grid()
except FileNotFoundError as e:
    st.error(f"Reference data not found. Make sure `data/` sits alongside this file.\n\nDetails: {e}")
    st.stop()

try:
    status = compute_live_status()
    live_ok = True
except Exception as e:
    live_ok = False
    live_error = str(e)
    try:
        with open(DATA_DIR / "kobo_latest_status.json") as f:
            status = json.load(f)
    except FileNotFoundError:
        st.error(f"Live computation failed and no fallback snapshot exists.\n\nDetails: {live_error}")
        st.stop()

status_color = STATUS_COLOR.get(status["status"], COLORS["text_muted"])

# =================================================================
# HERO MASTHEAD
# =================================================================
live_dot_color = COLORS["sage"] if live_ok else COLORS["ochre"]
live_text = (
    f"LIVE &nbsp;·&nbsp; recomputed {status.get('computed_at', '')}"
    if live_ok
    else f"SNAPSHOT &nbsp;·&nbsp; live pull unavailable"
)

hero_html = f"""
<div class="kobo-hero" style="--status-glow: {status_color}">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1.5rem;">
    <div>
      <div class="kobo-label" style="margin-bottom:0.9rem;">
        <span class="kobo-live-dot" style="background:{live_dot_color}"></span>{live_text}
      </div>
      <h1 style="margin:0; font-size:2.3rem; line-height:1.15;">Kobo Drought<br/>Early Warning</h1>
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

if not live_ok:
    with st.expander("Why is this a snapshot instead of live data?"):
        st.write(live_error)

# =================================================================
# INDICATOR STRIP
# =================================================================
vec = status["current_vector"]


def fmt(v, suffix="", sign=False):
    if v is None:
        return "—"
    if sign:
        return f"{v:+.2f}{suffix}"
    return f"{v:.2f}{suffix}"


strip_items = [
    ("NDVI anomaly", fmt(vec.get("ndvi_anomaly_z"), "σ", sign=True), COLORS["sage"]),
    ("VCI", fmt(vec.get("vci")), COLORS["sage"]),
    ("Rainfall anomaly", fmt(vec.get("rain_anomaly_z"), "σ", sign=True), COLORS["sky"]),
    ("LST anomaly", fmt(vec.get("lst_anomaly_z"), "σ", sign=True), COLORS["clay"]),
    ("Soil moisture anomaly", fmt(vec.get("soil_anomaly_z"), "σ", sign=True), COLORS["ochre"]),
    ("15-day forecast", f"{status['forecast_pct_of_normal']:.0f}% normal", COLORS["sky"]),
]

strip_html = '<div class="kobo-strip">' + "".join(
    f"""<div class="kobo-strip-item">
        <div class="kobo-strip-label">{label}</div>
        <div class="kobo-strip-value" style="color:{color}">{val}</div>
    </div>"""
    for label, val, color in strip_items
) + "</div>"
st.markdown(strip_html, unsafe_allow_html=True)
st.markdown('<hr class="kobo-rule">', unsafe_allow_html=True)

st.markdown(
    f"""<p style="color:{COLORS['text_muted']}; font-size:0.95rem;">
    {status['forecast_note']}
    </p>""",
    unsafe_allow_html=True,
)

with st.expander("Data currency — sources update at different real-world speeds"):
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<span class='mono'>NDVI</span><br/>{status.get('ndvi_date', 'unknown')}", unsafe_allow_html=True)
    c2.markdown(f"<span class='mono'>LST</span><br/>{status.get('lst_date', 'unknown')}", unsafe_allow_html=True)
    c3.markdown(f"<span class='mono'>Rainfall / soil moisture</span><br/>{status.get('rain_soil_date', 'unknown')}", unsafe_allow_html=True)

st.markdown('<hr class="kobo-rule">', unsafe_allow_html=True)

# =================================================================
# SPATIAL RISK MAP + FORECAST WINDOWS (two columns)
# =================================================================
col_map, col_forecast = st.columns([1.3, 1])

with col_map:
    st.markdown('<div class="kobo-label">Spatial risk pattern</div>', unsafe_allow_html=True)
    geojson = json.loads(grid.to_json())

    fig_map = go.Figure(
        go.Choroplethmap(
            geojson=geojson,
            locations=grid.index,
            z=grid["accuracy"],
            colorscale=[[0, COLORS["clay"]], [0.5, COLORS["ochre"]], [1, COLORS["sage"]]],
            zmin=0.70, zmax=0.95,
            marker_opacity=0.85,
            marker_line_width=1,
            marker_line_color=COLORS["bg"],
            colorbar=dict(
                title=dict(text="accuracy", font=dict(color=COLORS["text_muted"], size=10)),
                tickfont=dict(color=COLORS["text_muted"], size=9),
                thickness=12, len=0.7,
            ),
        )
    )
    fig_map.update_layout(
        map_style="carto-darkmatter",
        map_zoom=8.3,
        map_center={"lat": 12.1, "lon": 39.65},
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=380,
        font=dict(family="Space Grotesk", color=COLORS["text_muted"]),
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
    st.markdown('<div class="kobo-label">15-day forecast · ECMWF AIFS</div>', unsafe_allow_html=True)
    fw = status.get("forecast_windows", {})
    if fw:
        windows = [
            ("1–3 DAYS", fw.get("1_3_day_mm", 0)),
            ("4–7 DAYS", fw.get("4_7_day_mm", 0)),
            ("8–15 DAYS", fw.get("8_15_day_mm", 0)),
        ]
        max_mm = max(v for _, v in windows) or 1
        for label, mm in windows:
            pct = mm / max_mm * 100
            st.markdown(
                f"""
                <div style="margin-bottom:1.1rem;">
                    <div class="kobo-strip-label">{label}</div>
                    <div style="display:flex; align-items:center; gap:0.7rem;">
                        <div style="flex:1; height:6px; background:{COLORS['rule']}; border-radius:3px; overflow:hidden;">
                            <div style="width:{pct}%; height:100%; background:{COLORS['sky']};"></div>
                        </div>
                        <div class="mono" style="color:{COLORS['sky']}; font-size:0.95rem; min-width:60px; text-align:right;">
                            {mm:.1f} mm
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            f"""<p style="color:{COLORS['text_muted']}; font-size:0.78rem; margin-top:0.5rem;">
            Deterministic single-run forecast — one plausible scenario, not an ensemble probability.
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
drivers = status.get("top_drivers", [])
if drivers:
    drivers_df = pd.DataFrame(drivers)
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
# COLOPHON
# =================================================================
st.markdown(
    f"""
    <div class="kobo-colophon">
    KOBO DROUGHT EARLY WARNING SYSTEM — a portfolio project, not an official warning service.<br/>
    DATA — MODIS NDVI/LST · CHIRPS · ERA5-Land · ECMWF AIFS, via Google Earth Engine<br/>
    MODEL — logistic regression, validated against published North Wollo drought literature<br/>
    BUILD — Python · scikit-learn · Streamlit · Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
