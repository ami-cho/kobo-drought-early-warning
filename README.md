# Kobo Drought Early Warning System

Machine Learning-Based Drought Risk Mapping and Early Warning System for Raya Kobo Woreda, North Wollo Zone, Amhara Region, Ethiopia.

**Live status (as of last run):** see `dashboard/` — run `streamlit run dashboard/dashboard.py` after populating `dashboard/data/` (see below).

## Project summary

An end-to-end system that combines satellite remote sensing, reanalysis climate data, and a short-range numerical weather forecast to detect and forecast agricultural drought risk at woreda and sub-woreda (grid-cell) resolution, validated against academic literature on North Wollo drought history.

- **Historical record**: 2000-2024, monthly, both woreda-mean and 25-cell spatial grid
- **Labels**: SPI-6 (6-month Standardized Precipitation Index), chosen specifically to avoid target leakage from the NDVI/LST/soil-moisture predictors
- **Model**: logistic regression — selected over Random Forest and XGBoost after honest head-to-head comparison (0.834 balanced accuracy / 0.687 macro F1 on validation, vs 0.610-0.657 for the tree ensembles)
- **Forecast layer**: ECMWF AIFS (deterministic, open data), 1-15 day precipitation outlook blended with the historical-data classifier
- **Backtesting**: median 3-month lead time ahead of documented Severe drought events, with an honestly-reported 64% false alarm rate
- **Dashboard**: Streamlit, covering current status, spatial risk map, forecast breakdown, and model explainability

## Key findings

- Every literature-documented North Wollo drought year (2004, 2005, 2008, 2009, 2010, 2011, 2013, 2015) shows a clear rainfall deficit in this dataset except 2005, which remained an unresolved mild case across every check performed (NDVI, VCI, seasonal rainfall, SPI-3, SPI-6) — reported as an honest limitation, not tuned away.
- Three additional stressed years not in the literature list (2006, 2012, 2017) were fully explained once SPI-1/SPI-3 (short-timescale) rainfall indices were checked — each had a genuine short dry spell hidden inside an otherwise-normal season.
- Drought *frequency* is spatially uniform across Kobo's grid cells by construction (SPI is normalized per cell) — but model *accuracy* is not (0.766-0.913 range), with the lowest accuracy in the lowland southeast, the area with the highest true Severe-drought exposure.
- Moving from a single woreda-mean model to a 25-cell spatial grid model traded some accuracy (0.834 → 0.761 balanced accuracy) for genuine spatial resolution — an explicit, documented trade-off.

## Repository structure

```
kobo-drought-early-warning/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/          # boundary file (COD-AB Kobo woreda extract)
│   ├── processed/     # combined feature tables (region-level + grid-level)
│   └── external/
├── models/            # trained model, scaler, SPI parameters, climatology lookups (pickled)
├── figures/            # exported charts (correlation matrix, time series, cell accuracy map)
├── maps/                # grid cell geometries + performance (GeoJSON)
├── notebooks/            # main Colab/Jupyter notebook, full pipeline
├── docs/                  # DATA_SOURCES.md — data provenance and methodology notes
└── dashboard/               # Streamlit dashboard app + its own data/ folder
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

## Methodology highlights

1. **Leakage avoidance**: drought labels are derived solely from SPI-6 (a precipitation-only index), never from the NDVI/VCI/LST/soil-moisture features used as predictors.
2. **Temporal validation split**: train 2000-2012, validation 2013-2020, test 2021-2024 — redrawn from an initial naive split after discovering the original validation window happened to contain zero drought-class months.
3. **Baseline-first**: VCI threshold, SPI-3 threshold, and logistic regression were all evaluated before Random Forest/XGBoost, establishing an honest bar for the more complex models to clear.
4. **Spatial modeling**: a 25-cell (~11km, ERA5-Land-resolution) grid was built after an initial region-mean-only model, to produce genuine sub-woreda risk maps rather than a single areal average.
5. **Backtesting**: lead-time-to-event and false-alarm rate were both measured and reported, not just accuracy metrics.

## Limitations

- 25-year historical record with rare severe-drought events (13 Moderate / 4 Severe months in the training set) limits how much any model — especially tree ensembles — could learn about the rare classes.
- 64% false alarm rate on elevated (Moderate/Severe) predictions, a direct consequence of the `class_weight="balanced"` choice favoring recall over precision.
- The current live dashboard blends data sources with different real-world processing lags (NDVI/LST ~2-4 weeks, CHIRPS/soil moisture ~1-2 months) — displayed explicitly rather than presented as uniformly "current."
- The AIFS forecast is a single deterministic run, not an ensemble; it represents one plausible scenario, not a probabilistic forecast.

## Author

Amhasilasie Mulugeta Aemero — Environmental Engineering student, Addis Ababa Science and Technology University.
