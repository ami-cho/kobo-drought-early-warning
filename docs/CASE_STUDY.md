# Case Study: Multi-Sensor Drought Early Warning System — Raya Kobo Woreda, North Wollo Zone, Amhara, Ethiopia

**A live, self-updating drought early warning system for Raya Kobo Woreda, Ethiopia — built end to end, from satellite pixels to a production deployment.**

*Amhasilasie Mulugeta Aemero — Environmental Engineering, Addis Ababa Science and Technology University*
[Live dashboard](https://kobo-drought-early-warning-gdabadqt9yqzwa5pwurvgt.streamlit.app/) · [GitHub repository](https://github.com/ami-cho/kobo-drought-early-warning)

---

## The problem

Raya Kobo Woreda, in Ethiopia's North Wollo Zone, is one of the 48 woredas officially identified as most drought-prone and food-insecure in the Amhara Region. Its rainfall is bimodal — a smaller Belg season (February-April) and the main Kiremt rains (July-August) — which means agricultural drought here is rarely a single bad month; it's a cumulative seasonal failure that can hide inside monthly averages that look unremarkable on their own.

The question this project set out to answer: **can multi-source satellite and climate reanalysis data be combined with machine learning to detect drought conditions in Kobo, cross-validated against the documented history, and extended into a genuinely live, forecast-integrated warning system?**

## Data and indicators

Four independent data sources were built into a monthly (and later, spatial-grid) feature set spanning 2000-2024:

| Signal | Source | What it captures |
|---|---|---|
| Vegetation stress | MODIS NDVI (250m, 16-day) | Crop/vegetation health response to moisture |
| Precipitation | CHIRPS (5km, daily) | Rainfall totals and deficits (SPI at 1/3/6/12-month timescales) |
| Land surface temperature | MODIS LST (1km, 8-day) | Heat stress associated with dry conditions |
| Soil moisture | ERA5-Land (11km, monthly) | Root-zone water availability |

Each layer was independently validated against eight drought years documented in published North Wollo agricultural-drought literature (2004, 2005, 2008, 2009, 2010, 2011, 2013, 2015) before being trusted for modeling — including catching and resolving a case where three additional stressed years (2006, 2012, 2017) initially looked unexplained by monthly rainfall, until short-timescale SPI-1/SPI-3 analysis revealed genuine dry spells hidden inside otherwise-normal seasons.

## Avoiding a common pitfall: target leakage

The drought label needed to represent independent ground truth, not a restatement of the predictors. Rather than labeling drought from NDVI or VCI — the same vegetation-stress signal used as a model input — labels were built exclusively from **SPI-6**, a purely precipitation-based meteorological index. NDVI, LST, and soil moisture then serve as predictors describing the *symptoms* the model learns to recognize, while SPI-6 defines the *ground truth* being predicted.

## Model selection: letting the evidence decide

Three models were compared under a temporal (not random) train/validation/test split, with baselines evaluated first per standard ML practice:

| Model | Balanced accuracy | Macro F1 |
|---|---|---|
| VCI threshold rule | 0.583 | 0.441 |
| SPI-3 threshold rule | 0.622 | 0.672 |
| **Logistic regression** | **0.834** | **0.687** |
| Random Forest (tuned) | 0.610 | 0.657 |
| XGBoost (tuned) | 0.618 | 0.647 |

Logistic regression — the simplest model tested — outperformed both tuned tree ensembles. With only 13 Moderate and 4 Severe drought-months in the training period, tree-based models didn't have enough minority-class examples to out-learn a well-regularized linear model. The final model was chosen strictly on validation evidence, not algorithm sophistication.

## From a single number to a spatial map

The initial model produced one region-wide average per month. A 25-cell spatial grid (~11km resolution, matching the coarsest input dataset) was built to enable genuine sub-woreda risk mapping — trading some accuracy for spatial resolution (0.834 → 0.761 balanced accuracy), an explicit, documented cost.

The resulting map revealed a real spatial pattern a single areal average could never show: model accuracy ranges from 0.766 to 0.913 across Kobo's cells, and it is specifically **lower in the lowland southeast — the same area with the highest true severe-drought exposure.** The early-warning system is least reliable where it may be needed most.

## Forward-looking: forecast integration

The system integrates ECMWF's **AIFS** (AI Forecasting System) open-data deterministic forecast, extracting 1-3, 4-7, and 8-15 day precipitation outlooks and blending them with the historical-data classifier's current-state prediction into a single operational status.

## From notebook to production: making it genuinely live

The first deployed version served a static snapshot generated once in a Colab session — a common shortcut, but not a real early warning system. Rebuilding it to be genuinely live meant:

- **Authenticating a deployed web app against Google Earth Engine** using a GCP service account, working through an organization-level security policy (`iam.disableServiceAccountKeyCreation`) that initially blocked key generation entirely, and layering the correct IAM roles (Service Usage Consumer, Earth Engine Resource Writer) once the key existed
- **Refactoring the notebook's current-state and forecast logic into a callable module** the deployed app runs on every visit, cached for 6 hours so it recomputes from real satellite and forecast data without hammering the underlying APIs
- **Building honest failure handling**: if the live computation fails for any reason (a quota limit, a temporary data gap), the app falls back to the last known-good snapshot and says so explicitly in the UI, rather than crashing or silently showing stale data as if it were current
- **Fixing a real gap the redesign surfaced**: the live computation path was initially missing the model-explainability calculation entirely — a bug invisible until the UI was rebuilt to actually display it, caught and fixed rather than left unnoticed

The result is a dashboard that shows a visible **"🟢 Live — recomputed from Earth Engine + ECMWF AIFS"** banner with a timestamp, and the numbers genuinely move between visits as new satellite passes and forecast runs come in.

## Design: an instrument panel, not a template

The dashboard's visual design was built deliberately around the subject matter rather than a generic dashboard template: a soil-charcoal background, an earth-toned palette (sage for healthy conditions, ochre for watch-level dryness, clay-red for severe alerts, dusty blue for rainfall data), a serif display face paired with monospaced numerals throughout — reinforcing that this is a reading instrument, not a marketing page. Sections are separated by hairline rules rather than the boxed-card-with-shadow look most generated dashboards default to.

## Backtesting: how much warning does it actually give?

Running the trained model across the full historical record and measuring lead time to each documented Severe event: **median 3 months of advance warning**, holding up in genuinely out-of-sample validation and test periods. This came with a real, quantified cost: a **64% false alarm rate** on elevated predictions, the direct consequence of tuning the model to favor catching real events over avoiding false alarms. Both numbers are reported together, because a lead-time statistic without its false-alarm rate is an incomplete claim.

## Deliverables

- **[Live dashboard](https://kobo-drought-early-warning-gdabadqt9yqzwa5pwurvgt.streamlit.app/)** — genuinely recomputes from Earth Engine and ECMWF AIFS on every visit, not a static export
- **[GitHub repository](https://github.com/ami-cho/kobo-drought-early-warning)** — full pipeline notebook, trained model artifacts, documented data provenance
- **Validated historical record** — every model output cross-checked against independent literature and multiple physical indicators before being trusted

## Skills demonstrated

Google Earth Engine (Python API) · remote sensing (MODIS, CHIRPS, ERA5-Land) · numerical weather forecast integration (ECMWF AIFS) · feature engineering and time-series analysis · leakage-aware ML labeling · model comparison and selection (scikit-learn, XGBoost) · spatial/grid-based geospatial analysis (geopandas) · backtesting and operational validation · cloud service authentication and IAM (GCP service accounts) · production deployment and caching strategy · dashboard development and visual design (Streamlit, Plotly) · technical documentation and reproducible research practices

## Why this approach

This project deliberately mirrors the rigor of a companion flood-susceptibility mapping project for Addis Ababa — both apply GIS and machine learning to a distinct environmental hazard, with an emphasis on validating every intermediate result against independent evidence rather than trusting a model's output at face value. The goal throughout was not simply to produce a drought map, but to demonstrate an honest, defensible, end-to-end environmental data science workflow: from raw satellite pixels to a live, production-deployed, forecast-integrated warning system that keeps working after the notebook is closed.
