"""
live_status.py

Pure computation module for the Kobo Drought Early Warning system: pulls
current NDVI/LST/CHIRPS/soil moisture from Earth Engine, fetches the latest
ECMWF AIFS ensemble forecast, and scores both through the trained model.

Deliberately has NO Streamlit dependency -- this makes it runnable as a
standalone scheduled job (see refresh_job.py), completely decoupled from
the Streamlit app's request path. The app only ever reads the JSON file
the job writes (see status_reader.py); it never calls compute_status()
itself.

Authentication: takes a GCP service account credential dict as a plain
argument (not st.secrets), so the caller decides where those credentials
come from -- an environment variable in refresh_job.py's case.
"""

import json
import pickle
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------
# Earth Engine init (service account)
# ---------------------------------------------------------------
def init_earth_engine(sa_info):
    credentials = ee.ServiceAccountCredentials(
        sa_info["client_email"], key_data=json.dumps(sa_info)
    )
    ee.Initialize(credentials, project=sa_info["project_id"])
    return True


# ---------------------------------------------------------------
# Load static artifacts (model, scaler, SPI params, climatology, boundary)
# ---------------------------------------------------------------
def load_artifacts(region_id, data_dir=None):
    """Loads the four artifact files for a given region_id, following the
    {region_id}_ prefix convention. This is the entire mechanism that makes
    a second region a config change instead of a code change."""
    data_dir = data_dir or DATA_DIR
    with open(data_dir / f"{region_id}_model.pkl", "rb") as f:
        model_bundle = pickle.load(f)
    with open(data_dir / f"{region_id}_spi_params.pkl", "rb") as f:
        spi_params = pickle.load(f)
    with open(data_dir / f"{region_id}_climatology.pkl", "rb") as f:
        clim = pickle.load(f)
    with open(data_dir / f"{region_id}_boundary.geojson") as f:
        boundary_geojson = json.load(f)

    region_geom = ee.Geometry(boundary_geojson["features"][0]["geometry"])

    return {
        "region_id": region_id,
        "model": model_bundle["model"],
        "scaler": model_bundle["scaler"],
        "predictor_cols": model_bundle["predictor_cols"],
        "spi_params": spi_params,
        "clim": clim,
        "region_geom": region_geom,
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
def pull_current_state(region_geom, months_back=15):
    end = ee.Date(pd.Timestamp.now("UTC").strftime("%Y-%m-%d"))
    start = end.advance(-months_back, "month")

    # NDVI — last 3 composites (need current + a real lag1, not a fallback)
    ndvi_coll_recent = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterBounds(region_geom)
        .sort("system:time_start", False)
        .limit(3)
    )

    def ndvi_stat(img):
        qa = img.select("SummaryQA")
        val = img.select("NDVI").updateMask(qa.eq(0)).multiply(0.0001).reduceRegion(
            ee.Reducer.mean(), region_geom, 250, maxPixels=1e9
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

    # LST — last 3 composites (need current + a real lag1 for trend arrows)
    lst_coll_recent = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterBounds(region_geom)
        .sort("system:time_start", False)
        .limit(3)
    )

    def lst_stat(img):
        qc = img.select("QC_Day")
        val = img.select("LST_Day_1km").updateMask(qc.bitwiseAnd(3).eq(0)).multiply(0.02).subtract(273.15).reduceRegion(
            ee.Reducer.mean(), region_geom, 1000, maxPixels=1e9
        ).get("LST_Day_1km")
        return ee.Feature(None, {"date": img.date().format("YYYY-MM-dd"), "lst_c": val})

    lst_recent_fc = lst_coll_recent.map(lst_stat)
    lst_recent_df = pd.DataFrame([f["properties"] for f in lst_recent_fc.getInfo()["features"]])
    lst_recent_df["date"] = pd.to_datetime(lst_recent_df["date"])
    lst_recent_df = lst_recent_df.sort_values("date").reset_index(drop=True)

    lst_val = lst_recent_df.iloc[-1]["lst_c"]
    lst_date = lst_recent_df.iloc[-1]["date"]
    lst_val_lag1 = lst_recent_df.iloc[-2]["lst_c"] if len(lst_recent_df) >= 2 else np.nan
    lst_date_lag1 = lst_recent_df.iloc[-2]["date"] if len(lst_recent_df) >= 2 else None

    # CHIRPS — recent monthly totals (need history for SPI-1/3/12)
    chirps_live = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(region_geom)
        .select("precipitation")
    )
    n = months_back
    months_seq = ee.List.sequence(0, n - 1)

    def monthly_rain(m):
        m = ee.Number(m)
        s = start.advance(m, "month")
        e = s.advance(1, "month")
        img = chirps_live.filterDate(s, e).sum()
        stat = img.reduceRegion(ee.Reducer.mean(), region_geom, 5000, maxPixels=1e9)
        return ee.Feature(None, {"date": s.format("YYYY-MM-dd"), "rainfall_mm": stat.get("precipitation", -9999)})

    rain_fc = ee.FeatureCollection(months_seq.map(monthly_rain))
    rain_df = pd.DataFrame([f["properties"] for f in rain_fc.getInfo()["features"]])
    rain_df["date"] = pd.to_datetime(rain_df["date"])
    rain_df["rainfall_mm"] = rain_df["rainfall_mm"].replace(-9999, np.nan)
    rain_df["month"] = rain_df["date"].dt.month
    rain_df = rain_df.dropna(subset=["rainfall_mm"]).sort_values("date").reset_index(drop=True)

    # ERA5-Land soil moisture — recent months, guarded against not-yet-available months
    era5_live = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterBounds(region_geom).select(
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
            ee.Image(filtered.first()).reduceRegion(ee.Reducer.mean(), region_geom, 11132, maxPixels=1e9)
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
        "lst_val_lag1": lst_val_lag1, "lst_date_lag1": lst_date_lag1,
        "rain_df": rain_df, "soil_df": soil_df,
    }


# ---------------------------------------------------------------
# AIFS forecast (ENSEMBLE — control + a subset of perturbed members)
# ---------------------------------------------------------------
def pull_aifs_forecast(region_geom, region_bounds):
    from ecmwf.opendata import Client
    import xarray as xr
    import tempfile

    client = Client(source="ecmwf", model="aifs-ens")
    forecast_steps = list(range(24, 361, 24))
    # A subset of the 50 perturbed members, evenly spaced — a full 50-member
    # download would meaningfully slow down a live app on a 6h cache cycle.
    member_numbers = list(range(1, 50, 5))  # 10 members: 1, 6, 11, ..., 46

    minx, miny, maxx, maxy = region_bounds

    def load_and_subset(path):
        ds = xr.open_dataset(path, engine="cfgrib")
        return ds.sel(latitude=slice(maxy, miny), longitude=slice(minx, maxx))

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp_cf:
        result_cf = client.retrieve(
            stream="enfo", type="cf", param="tp", step=forecast_steps, target=tmp_cf.name
        )
        ds_cf = load_and_subset(tmp_cf.name)

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp_pf:
        client.retrieve(
            stream="enfo", type="pf", param="tp", step=forecast_steps,
            number=member_numbers, target=tmp_pf.name,
        )
        ds_pf = load_and_subset(tmp_pf.name)

    def member_daily_series(tp_mean_series):
        df = tp_mean_series.to_dataframe(name="tp_mm").reset_index()
        if "tp" in df.columns and "tp_mm" not in df.columns:
            df = df.rename(columns={"tp": "tp_mm"})
        df["step_days"] = df["step"].dt.days
        df = df.sort_values("step_days").reset_index(drop=True)
        df["daily_mm"] = df["tp_mm"].diff().fillna(df["tp_mm"].iloc[0])
        return df

    def window_total(df, days_range):
        mask = df["step_days"].isin(days_range)
        return df.loc[mask, "daily_mm"].sum()

    windows = {"1_3": [1, 2, 3], "4_7": [4, 5, 6, 7], "8_15": list(range(8, 16))}
    member_totals = {k: [] for k in windows}

    cf_mean = ds_cf["tp"].mean(dim=["latitude", "longitude"])
    cf_df = member_daily_series(cf_mean)
    for key, days in windows.items():
        member_totals[key].append(window_total(cf_df, days))

    pf_mean = ds_pf["tp"].mean(dim=["latitude", "longitude"])  # dims: number, step
    for num in pf_mean["number"].values:
        member_df = member_daily_series(pf_mean.sel(number=num))
        for key, days in windows.items():
            member_totals[key].append(window_total(member_df, days))

    return {
        "issued": pd.to_datetime(result_cf.datetime),
        "member_totals": member_totals,  # {"1_3": [mm, mm, ...], "4_7": [...], "8_15": [...]}
        "n_members": len(member_totals["1_3"]),
    }


# ---------------------------------------------------------------
# Combine current state + forecast + model into a status dict
# ---------------------------------------------------------------
def compute_status(region_id, sa_info, data_dir=None):
    """Runs the full pipeline once, for the given region_id, and returns a
    status dict. This is the function the scheduled job calls -- it is
    never called directly from the Streamlit app's request path. No
    caching here; the caller decides how often to actually invoke this
    (e.g. a cron schedule)."""
    data_dir = data_dir or DATA_DIR
    init_earth_engine(sa_info)
    artifacts = load_artifacts(region_id, data_dir)
    region_geom = artifacts["region_geom"]
    region_bounds = region_geom.bounds().getInfo()["coordinates"][0]
    minx = min(p[0] for p in region_bounds)
    maxx = max(p[0] for p in region_bounds)
    miny = min(p[1] for p in region_bounds)
    maxy = max(p[1] for p in region_bounds)

    state = pull_current_state(region_geom)
    forecast = pull_aifs_forecast(region_geom, (minx, miny, maxx, maxy))

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
        "lst_anomaly_z_lag1": (
            get_anomaly(state["lst_val_lag1"], state["lst_date_lag1"].month, clim["lst_clim"])
            if state["lst_val_lag1"] is not None and pd.notna(state["lst_val_lag1"])
            else np.nan
        ),
        "vci_lag1": (
            (state["ndvi_val_lag1"] - clim["ndvi_clim"].loc[state["ndvi_date_lag1"].month, "clim_min"]) /
            (clim["ndvi_clim"].loc[state["ndvi_date_lag1"].month, "clim_max"] - clim["ndvi_clim"].loc[state["ndvi_date_lag1"].month, "clim_min"]) * 100
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

    current_month = pd.Timestamp.now("UTC").month
    daily_clim_rate = clim["rain_clim_monthly"].set_index("month").loc[current_month, "clim_mean"] / 30

    def ensemble_stats(values, normal_mm):
        arr = np.array(values, dtype=float)
        return {
            "median_mm": float(np.median(arr)),
            "p10_mm": float(np.percentile(arr, 10)),
            "p90_mm": float(np.percentile(arr, 90)),
            "pct_below_normal": float((arr < normal_mm).mean() * 100),
            "n_members": int(len(arr)),
        }

    window_stats = {
        "1_3": ensemble_stats(forecast["member_totals"]["1_3"], daily_clim_rate * 3),
        "4_7": ensemble_stats(forecast["member_totals"]["4_7"], daily_clim_rate * 4),
        "8_15": ensemble_stats(forecast["member_totals"]["8_15"], daily_clim_rate * 8),
    }
    # Overall below-normal signal across all three windows, for the status note
    overall_pct_below_normal = float(np.mean([
        window_stats["1_3"]["pct_below_normal"],
        window_stats["4_7"]["pct_below_normal"],
        window_stats["8_15"]["pct_below_normal"],
    ]))

    # Explainability: this class's coefficients x this observation's scaled values
    # (multinomial logistic regression -- see the UI caveat about relative-not-absolute contribution)
    class_idx = list(artifacts["model"].classes_).index(pred_class)
    coef_for_class = artifacts["model"].coef_[class_idx]
    drivers_df = pd.DataFrame({
        "feature": predictor_cols,
        "coefficient": coef_for_class,
        "value_z": X_scaled[0],
    }).sort_values("coefficient", key=abs, ascending=False)
    top_drivers = drivers_df.head(6).to_dict(orient="records")

    if pred_class == "Severe":
        base_status = "Warning"
    elif pred_class == "Moderate":
        base_status = "Watch"
    else:
        base_status = "Normal"

    if overall_pct_below_normal > 65 and base_status != "Normal":
        note = f"{overall_pct_below_normal:.0f}% of ensemble members show below-normal rainfall — high confidence of continued/worsening deficit"
    elif overall_pct_below_normal < 35 and base_status != "Normal":
        note = f"Only {overall_pct_below_normal:.0f}% of ensemble members show below-normal rainfall — possible relief, monitor for improvement"
    else:
        note = f"Forecast uncertain — ensemble members split ({overall_pct_below_normal:.0f}% below normal), no strong consensus either way"

    return {
        "region_id": region_id,
        "status": base_status,
        "current_class": pred_class,
        "current_confidence": float(proba_dict[pred_class]),
        "class_probabilities": {k: float(v) for k, v in proba_dict.items()},
        "forecast_pct_below_normal": overall_pct_below_normal,
        "forecast_note": note,
        "ndvi_date": state["ndvi_date"].strftime("%Y-%m-%d"),
        "lst_date": state["lst_date"].strftime("%Y-%m-%d"),
        "rain_soil_date": current_rain["date"].strftime("%Y-%m-%d"),
        "forecast_issued": forecast["issued"].strftime("%Y-%m-%d %H:%M"),
        "current_vector": {k: (float(v) if pd.notna(v) else None) for k, v in current_vector.items()},
        "forecast_windows": window_stats,
        "top_drivers": top_drivers,
        "computed_at": pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "computed_at_iso": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%S") + "Z",
        "refresh_interval_seconds": 6 * 3600,
    }
