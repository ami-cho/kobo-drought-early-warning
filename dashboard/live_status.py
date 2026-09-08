"""
live_status.py

Genuinely live drought status computation for the Kobo Drought Early Warning
dashboard. Pulls current NDVI/LST/CHIRPS/soil moisture from Earth Engine,
fetches the latest ECMWF AIFS forecast, scores both through the trained
model, and returns the same status dict shape the Colab notebook used to
export as kobo_latest_status.json.

Authentication: uses a GCP service account whose JSON key is stored in
Streamlit secrets (st.secrets["gee_service_account"]) — never committed
to the repo.
"""

import json
import pickle
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------
# Earth Engine init (service account, cached for the session)
# ---------------------------------------------------------------
@st.cache_resource
def init_earth_engine():
    sa_info = dict(st.secrets["gee_service_account"])
    credentials = ee.ServiceAccountCredentials(
        sa_info["client_email"], key_data=json.dumps(sa_info)
    )
    ee.Initialize(credentials, project=sa_info["project_id"])
    return True


# ---------------------------------------------------------------
# Load static artifacts (model, scaler, SPI params, climatology, boundary)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(DATA_DIR / "kobo_model.pkl", "rb") as f:
        model_bundle = pickle.load(f)
    with open(DATA_DIR / "kobo_spi_params.pkl", "rb") as f:
        spi_params = pickle.load(f)
    with open(DATA_DIR / "kobo_climatology.pkl", "rb") as f:
        clim = pickle.load(f)
    with open(DATA_DIR / "kobo_boundary.geojson") as f:
        boundary_geojson = json.load(f)

    kobo_geom = ee.Geometry(boundary_geojson["features"][0]["geometry"])

    return {
        "model": model_bundle["model"],
        "scaler": model_bundle["scaler"],
        "predictor_cols": model_bundle["predictor_cols"],
        "spi_params": spi_params,
        "clim": clim,
        "kobo_geom": kobo_geom,
    }


def score_spi(value, window, month, spi_params):
    key = (window, month)
    if key not in spi_params or pd.isna(value):
        return np.nan
    shape, loc, scale, prob_zero = spi_params[key]
    if value <= 0:
        cdf = prob_zero
    else:
        cdf = prob_zero + (1 - prob_zero) * stats.gamma.cdf(value, shape, loc=loc, scale=scale)
    cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
    return stats.norm.ppf(cdf)


def get_anomaly(value, month, clim_lookup):
    row = clim_lookup.loc[month]
    return (value - row["clim_mean"]) / row["clim_std"]


def get_percentile(value, month, hist_col, month_col):
    mask = month_col == month
    return (hist_col[mask] < value).mean() * 100


# ---------------------------------------------------------------
# Pull current-state satellite/reanalysis data
# ---------------------------------------------------------------
def pull_current_state(kobo_geom, months_back=15):
    end = ee.Date(pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    start = end.advance(-months_back, "month")

    # NDVI — last 3 composites (need current + a real lag1, not a fallback)
    ndvi_coll_recent = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterBounds(kobo_geom)
        .sort("system:time_start", False)
        .limit(3)
    )

    def ndvi_stat(img):
        qa = img.select("SummaryQA")
        val = img.select("NDVI").updateMask(qa.eq(0)).multiply(0.0001).reduceRegion(
            ee.Reducer.mean(), kobo_geom, 250, maxPixels=1e9
        ).get("NDVI")
        return ee.Feature(None, {"date": img.date().format("YYYY-MM-dd"), "ndvi": val})

    ndvi_recent_fc = ndvi_coll_recent.map(ndvi_stat)
    ndvi_recent_df = pd.DataFrame([f["properties"] for f in ndvi_recent_fc.getInfo()["features"]])
    ndvi_recent_df["date"] = pd.to_datetime(ndvi_recent_df["date"])
    ndvi_recent_df = ndvi_recent_df.sort_values("date").reset_index(drop=True)

    ndvi_val = ndvi_recent_df.iloc[-1]["ndvi"]
    ndvi_date = ndvi_recent_df.iloc[-1]["date"]
    ndvi_val_lag1 = ndvi_recent_df.iloc[-2]["ndvi"] if len(ndvi_recent_df) >= 2 else np.nan
    ndvi_date_lag1 = ndvi_recent_df.iloc[-2]["date"] if len(ndvi_recent_df) >= 2 else None

    # LST — most recent available composite
    lst_img = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterBounds(kobo_geom)
        .sort("system:time_start", False)
        .first()
    )
    lst_qc = lst_img.select("QC_Day")
    lst_val = (
        lst_img.select("LST_Day_1km").updateMask(lst_qc.bitwiseAnd(3).eq(0))
        .multiply(0.02).subtract(273.15)
        .reduceRegion(ee.Reducer.mean(), kobo_geom, 1000, maxPixels=1e9)
        .get("LST_Day_1km").getInfo()
    )
    lst_date = pd.to_datetime(lst_img.date().format("YYYY-MM-dd").getInfo())

    # CHIRPS — recent monthly totals (need history for SPI-1/3/12)
    chirps_live = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(kobo_geom)
        .select("precipitation")
    )
    n = months_back
    months_seq = ee.List.sequence(0, n - 1)

    def monthly_rain(m):
        m = ee.Number(m)
        s = start.advance(m, "month")
        e = s.advance(1, "month")
        img = chirps_live.filterDate(s, e).sum()
        stat = img.reduceRegion(ee.Reducer.mean(), kobo_geom, 5000, maxPixels=1e9)
        return ee.Feature(None, {"date": s.format("YYYY-MM-dd"), "rainfall_mm": stat.get("precipitation", -9999)})

    rain_fc = ee.FeatureCollection(months_seq.map(monthly_rain))
    rain_df = pd.DataFrame([f["properties"] for f in rain_fc.getInfo()["features"]])
    rain_df["date"] = pd.to_datetime(rain_df["date"])
    rain_df["rainfall_mm"] = rain_df["rainfall_mm"].replace(-9999, np.nan)
    rain_df["month"] = rain_df["date"].dt.month
    rain_df = rain_df.dropna(subset=["rainfall_mm"]).sort_values("date").reset_index(drop=True)

    # ERA5-Land soil moisture — recent months, guarded against not-yet-available months
    era5_live = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterBounds(kobo_geom).select(
        "volumetric_soil_water_layer_1"
    )

    def monthly_soil(m):
        m = ee.Number(m)
        s = start.advance(m, "month")
        e = s.advance(1, "month")
        filtered = era5_live.filterDate(s, e)
        has_data = filtered.size().gt(0)
        val = ee.Algorithms.If(
            has_data,
            ee.Image(filtered.first()).reduceRegion(ee.Reducer.mean(), kobo_geom, 11132, maxPixels=1e9)
            .get("volumetric_soil_water_layer_1"),
            -9999,
        )
        return ee.Feature(None, {"date": s.format("YYYY-MM-dd"), "soil_moisture": val})

    soil_fc = ee.FeatureCollection(months_seq.map(monthly_soil))
    soil_df = pd.DataFrame([f["properties"] for f in soil_fc.getInfo()["features"]])
    soil_df["date"] = pd.to_datetime(soil_df["date"])
    soil_df["soil_moisture"] = soil_df["soil_moisture"].replace(-9999, np.nan)
    soil_df["month"] = soil_df["date"].dt.month
    soil_df = soil_df.dropna(subset=["soil_moisture"]).sort_values("date").reset_index(drop=True)

    return {
        "ndvi_val": ndvi_val, "ndvi_date": ndvi_date,
        "ndvi_val_lag1": ndvi_val_lag1, "ndvi_date_lag1": ndvi_date_lag1,
        "lst_val": lst_val, "lst_date": lst_date,
        "rain_df": rain_df, "soil_df": soil_df,
    }


# ---------------------------------------------------------------
# AIFS forecast
# ---------------------------------------------------------------
def pull_aifs_forecast(kobo_geom, kobo_bounds):
    from ecmwf.opendata import Client
    import xarray as xr
    import tempfile

    client = Client(source="ecmwf", model="aifs-single")
    forecast_steps = list(range(24, 361, 24))

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        result = client.retrieve(type="fc", param="tp", step=forecast_steps, target=tmp.name)
        ds = xr.open_dataset(tmp.name, engine="cfgrib")

    minx, miny, maxx, maxy = kobo_bounds
    subset = ds.sel(latitude=slice(maxy, miny), longitude=slice(minx, maxx))
    tp_mean = subset["tp"].mean(dim=["latitude", "longitude"])

    tp_df = tp_mean.to_dataframe().reset_index()
    tp_df["tp_mm"] = tp_df["tp"]  # kg/m^2 == mm, no conversion
    tp_df["step_days"] = tp_df["step"].dt.days
    tp_df["daily_mm"] = tp_df["tp_mm"].diff().fillna(tp_df["tp_mm"].iloc[0])

    def window_total(days_range):
        mask = tp_df["step_days"].isin(days_range)
        return tp_df.loc[mask, "daily_mm"].sum()

    return {
        "issued": pd.to_datetime(result.datetime),
        "forecast_1_3": window_total([1, 2, 3]),
        "forecast_4_7": window_total([4, 5, 6, 7]),
        "forecast_8_15": window_total(list(range(8, 16))),
    }


# ---------------------------------------------------------------
# Combine current state + forecast + model into a status dict
# ---------------------------------------------------------------
@st.cache_data(ttl=6 * 3600, show_spinner="Computing live drought status from satellite and forecast data...")
def compute_live_status():
    init_earth_engine()
    artifacts = load_artifacts()
    kobo_geom = artifacts["kobo_geom"]
    kobo_bounds = kobo_geom.bounds().getInfo()["coordinates"][0]
    minx = min(p[0] for p in kobo_bounds)
    maxx = max(p[0] for p in kobo_bounds)
    miny = min(p[1] for p in kobo_bounds)
    maxy = max(p[1] for p in kobo_bounds)

    state = pull_current_state(kobo_geom)
    forecast = pull_aifs_forecast(kobo_geom, (minx, miny, maxx, maxy))

    rain_df = state["rain_df"]
    soil_df = state["soil_df"]
    spi_params = artifacts["spi_params"]
    clim = artifacts["clim"]

    rain_df["roll_1"] = rain_df["rainfall_mm"]
    rain_df["roll_3"] = rain_df["rainfall_mm"].rolling(3, min_periods=3).sum()
    rain_df["roll_12"] = rain_df["rainfall_mm"].rolling(12, min_periods=12).sum()
    rain_df["spi_1"] = rain_df.apply(lambda r: score_spi(r["roll_1"], 1, r["month"], spi_params), axis=1)
    rain_df["spi_3"] = rain_df.apply(lambda r: score_spi(r["roll_3"], 3, r["month"], spi_params) if pd.notna(r["roll_3"]) else np.nan, axis=1)
    rain_df["spi_12"] = rain_df.apply(lambda r: score_spi(r["roll_12"], 12, r["month"], spi_params) if pd.notna(r["roll_12"]) else np.nan, axis=1)

    current_rain = rain_df.iloc[-1]
    prev_rain = rain_df.iloc[-2]
    current_soil = soil_df.iloc[-1]
    prev_soil = soil_df.iloc[-2]
    spi12_row = rain_df.dropna(subset=["spi_12"])
    spi3_valid = rain_df.dropna(subset=["spi_3"])

    current_vector = {
        "ndvi_anomaly_z": get_anomaly(state["ndvi_val"], state["ndvi_date"].month, clim["ndvi_clim"]),
        "vci": (state["ndvi_val"] - clim["ndvi_clim"].loc[state["ndvi_date"].month, "clim_min"]) /
               (clim["ndvi_clim"].loc[state["ndvi_date"].month, "clim_max"] - clim["ndvi_clim"].loc[state["ndvi_date"].month, "clim_min"]) * 100,
        "rainfall_mm": current_rain["rainfall_mm"],
        "rain_anomaly_z": get_anomaly(current_rain["rainfall_mm"], current_rain["month"], clim["rain_clim_monthly"].set_index("month")),
        "spi_1": current_rain["spi_1"],
        "spi_3": current_rain["spi_3"],
        "spi_12": spi12_row["spi_12"].iloc[-1] if len(spi12_row) else np.nan,
        "lst_anomaly_z": get_anomaly(state["lst_val"], state["lst_date"].month, clim["lst_clim"]),
        "soil_anomaly_z": get_anomaly(current_soil["soil_moisture"], current_soil["month"], clim["soil_clim"]),
        "soil_percentile": get_percentile(
            current_soil["soil_moisture"], current_soil["month"],
            clim["soil_hist_by_month"]["soil_moisture"], clim["soil_hist_by_month"]["month"]
        ),
        "rain_anomaly_z_lag1": get_anomaly(prev_rain["rainfall_mm"], prev_rain["month"], clim["rain_clim_monthly"].set_index("month")),
        "spi_1_lag1": prev_rain["spi_1"],
        "spi_3_lag1": spi3_valid["spi_3"].iloc[-2] if len(spi3_valid) >= 2 else np.nan,
        "soil_anomaly_z_lag1": get_anomaly(prev_soil["soil_moisture"], prev_soil["month"], clim["soil_clim"]),
        "ndvi_anomaly_z_lag1": (
            get_anomaly(state["ndvi_val_lag1"], state["ndvi_date_lag1"].month, clim["ndvi_clim"])
            if state["ndvi_val_lag1"] is not None and pd.notna(state["ndvi_val_lag1"])
            else np.nan
        ),
    }

    predictor_cols = artifacts["predictor_cols"]
    X = pd.DataFrame([current_vector])[predictor_cols]

    missing = X.columns[X.isna().any()].tolist()
    if missing:
        raise ValueError(
            f"Live data pull is missing required values for: {missing}. "
            "This usually means one of the satellite/reanalysis sources doesn't have "
            "enough recent history available yet — try again later, or widen months_back."
        )

    X_scaled = artifacts["scaler"].transform(X)
    pred_class = artifacts["model"].predict(X_scaled)[0]
    pred_proba = artifacts["model"].predict_proba(X_scaled)[0]
    proba_dict = dict(zip(artifacts["model"].classes_, pred_proba))

    total_forecast_15d = forecast["forecast_1_3"] + forecast["forecast_4_7"] + forecast["forecast_8_15"]
    current_month = pd.Timestamp.utcnow().month
    daily_clim_rate = clim["rain_clim_monthly"].set_index("month").loc[current_month, "clim_mean"] / 30
    expected_15d = daily_clim_rate * 15
    forecast_pct = total_forecast_15d / expected_15d * 100 if expected_15d > 0 else np.nan

    if pred_class == "Severe":
        base_status = "Warning"
    elif pred_class == "Moderate":
        base_status = "Watch"
    else:
        base_status = "Normal"

    if forecast_pct < 50 and base_status != "Normal":
        note = "Forecast suggests continued/worsening deficit — no near-term relief expected"
    elif forecast_pct > 150 and base_status != "Normal":
        note = "Forecast suggests above-normal rainfall — possible relief, monitor for improvement"
    else:
        note = "Forecast near normal — no strong signal either way"

    return {
        "status": base_status,
        "current_class": pred_class,
        "current_confidence": float(proba_dict[pred_class]),
        "class_probabilities": {k: float(v) for k, v in proba_dict.items()},
        "forecast_pct_of_normal": float(forecast_pct),
        "forecast_note": note,
        "ndvi_date": state["ndvi_date"].strftime("%Y-%m-%d"),
        "lst_date": state["lst_date"].strftime("%Y-%m-%d"),
        "rain_soil_date": current_rain["date"].strftime("%Y-%m-%d"),
        "forecast_issued": forecast["issued"].strftime("%Y-%m-%d %H:%M"),
        "current_vector": {k: (float(v) if pd.notna(v) else None) for k, v in current_vector.items()},
        "forecast_windows": {
            "1_3_day_mm": float(forecast["forecast_1_3"]),
            "4_7_day_mm": float(forecast["forecast_4_7"]),
            "8_15_day_mm": float(forecast["forecast_8_15"]),
        },
        "computed_at": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    }
