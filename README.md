# Multi-Sensor Drought Early Warning System — Raya Kobo Woreda, North Wollo Zone, Amhara, Ethiopia

Machine Learning-Based Drought Risk Mapping and Early Warning System for Raya Kobo Woreda, North Wollo Zone, Amhara Region, Ethiopia.

**Live dashboard**: [kobo-drought-early-warning.streamlit.app](https://kobo-drought-early-warning-gdabadqt9yqzwa5pwurvgt.streamlit.app/)
**Methodology write-up**: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — data latency, why logistic regression, validation approach, known limitations, and what's next. Read this first if you're skimming.

## Project summary

An end-to-end system that combines satellite remote sensing, reanalysis climate data, and a short-range numerical weather forecast to detect and forecast agricultural drought risk at woreda and sub-woreda (grid-cell) resolution, validated against academic literature on North Wollo drought history.

- **Historical record**: 2000-2024, monthly, both woreda-mean and 25-cell spatial grid
- **Labels**: SPI-6 (6-month Standardized Precipitation Index), chosen specifically to avoid target leakage from the NDVI/LST/soil-moisture predictors
- **Model**: logistic regression — selected over Random Forest and XGBoost after honest head-to-head comparison (0.834 balanced accuracy / 0.687 macro F1 on validation, vs 0.610-0.657 for the tree ensembles); held-out test evaluation and naive-baseline comparison in `docs/METHODOLOGY.md`
- **Forecast layer**: ECMWF AIFS-ENS (10 perturbed members + control, open data), 1-15 day precipitation outlook with genuine ensemble spread, blended with the historical-data classifier
- **Backtesting**: median 3-month lead time ahead of documented Severe drought events, with an honestly-reported 64% false alarm rate
- **Architecture**: live computation is decoupled from the app's request path — a scheduled job (GitHub Actions) runs the Earth Engine/AIFS pull and writes a status file; the Streamlit app only ever reads it
- **Dashboard**: Streamlit, covering current status, spatial risk map, forecast breakdown, model explainability, and held-out validation metrics

## Key findings

- Every literature-documented North Wollo drought year (2004, 2005, 2008, 2009, 2010, 2011, 2013, 2015) shows a clear rainfall deficit in this dataset except 2005, which remained an unresolved mild case across every check performed (NDVI, VCI, seasonal rainfall, SPI-3, SPI-6) — reported as an honest limitation, not tuned away.
- Three additional stressed years not in the literature list (2006, 2012, 2017) were fully explained once SPI-1/SPI-3 (short-timescale) rainfall indices were checked — each had a genuine short dry spell hidden inside an otherwise-normal season.
- Drought *frequency* is spatially uniform across Kobo's grid cells by construction (SPI is normalized per cell) — but model *accuracy* is not (0.766-0.913 range), with the lowest accuracy in the lowland southeast, the area with the highest true Severe-drought exposure.
- Moving from a single woreda-mean model to a 25-cell spatial grid model traded some accuracy (0.834 → 0.761 balanced accuracy) for genuine spatial resolution — an explicit, documented trade-off.

## Repository structure

```
kobo-drought-early-warning/
├── README.md
├── .github/workflows/
│   ├── refresh_status.yml            # scheduled job: Earth Engine + AIFS -> status cache (cron, every 6h)
│   └── test_region_abstraction.yml   # CI: proves region_id parameterization on every push
├── data/
│   ├── raw/          # boundary file (COD-AB Kobo woreda extract)
│   ├── processed/     # combined feature tables (region-level + grid-level)
│   └── external/
├── models/            # trained model, scaler, SPI parameters, climatology lookups (pickled)
├── figures/            # exported charts (correlation matrix, time series, cell accuracy map)
├── maps/                # grid cell geometries + performance (GeoJSON)
├── notebooks/            # main Colab/Jupyter notebook, full pipeline
├── docs/                  # METHODOLOGY.md, CASE_STUDY.md, DATA_SOURCES.md
└── dashboard/               # deployed Streamlit app + its own data/ folder
    ├── dashboard.py             # UI only -- reads status via status_reader, never computes it
    ├── status_reader.py         # lightweight read-through cache the app actually uses
    ├── live_status.py           # pure computation module (Earth Engine + AIFS), zero Streamlit dependency
    ├── refresh_job.py           # CLI entry point the scheduled job runs (--region <id>)
    ├── region_config.py         # region registry -- adding a region is a config change here
    ├── test_region_abstraction.py  # committed proof that region_id genuinely parameterizes the pipeline
    ├── requirements.txt         # app-only deps (no Earth Engine/AIFS -- those aren't in the request path)
    └── requirements-job.txt     # job-only deps (Earth Engine, AIFS, cfgrib, etc.)
```

## Data sources

| Dataset | Source | Resolution | Period |
|---|---|---|---|
| Administrative boundary | OCHA Ethiopia COD-AB | vector (admin-3) | 2021 vintage |
| NDVI | MODIS MOD13Q1 | 250m, 16-day | 2000-2024 |
| Rainfall | CHIRPS daily | ~5km | 1981-2024 |
| Land surface temperature | MODIS MOD11A2 | ~1km, 8-day | 2000-2024 |
| Soil moisture | ERA5-Land monthly | ~11km | 2000-2024 |
| Forecast precipitation | ECMWF AIFS (open data, deterministic) | ~0.25°, 15-day horizon | real-time |

Full provenance and methodology notes: `docs/DATA_SOURCES.md`.

## Methodology and limitations

Full technical write-up — data latency table, why logistic regression was chosen over the tree ensembles, temporal validation approach, known failure modes, and what's next — lives in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md), kept as the single source of truth rather than duplicated here.

## Author

Amhasilasie Mulugeta Aemero — Environmental Engineering student, Addis Ababa Science and Technology University.
