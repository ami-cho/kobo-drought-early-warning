# Data Sources & Provenance

## Kobo Woreda administrative boundary — CONFIRMED

- **Source**: Ethiopia - Subnational Administrative Boundaries (COD-AB), OCHA Ethiopia / HDX
  - URL: https://data.humdata.org/dataset/cod-ab-eth
  - File used: `eth_admin_boundaries.geojson.zip` → `eth_admin3.geojson` (admin level 3, woreda)
  - Schema: v04, lowercase field names (`adm3_name`, `adm2_name`, `adm1_name`, `area_sqkm`, etc.)
  - Filter used: `adm3_name == "Raya Kobo"`, `adm2_name == "North Wello"`, `adm1_name == "Amhara"`
  - **Note**: the correct feature is named "Raya Kobo," not "Kobo town" (a separate, much smaller urban-center admin unit also present in the same file — excluded as it does not represent the agricultural woreda referenced in the drought literature).

### Verification (completed)
- [x] Polygon represents Kobo Woreda — confirmed via adm3_name/adm2_name/adm1_name match
- [x] Geometry valid — single clean polygon, no self-intersections or fragments
- [x] Area/centroid plausible — EE-computed area 1929.9 km² (attribute table: 1922.36 km², Wikipedia-reported: 2001.57 km², all consistent within normal boundary-version variation); centroid [39.642°E, 12.106°N] vs Wikipedia's [39.650°E, 12.133°N], ~3km apart
- [x] Does not bleed into neighbors — visually confirmed bordered by Alamata/Tigray (north), Chifra/Afar (east), matching expected administrative neighbors
- [x] Displays correctly on basemap — confirmed via geemap outline-paint rendering

## Modeling grid

- 25 cells, 0.1° (~11km) resolution, matching ERA5-Land's native resolution as the coarsest/limiting dataset among NDVI (250m), CHIRPS (~5km), LST (~1km), and soil moisture (~11km)
- Clipped to the Kobo boundary; one cell (cell_id 0) excluded — confirmed as a 1.1 km² boundary sliver (vs ~123.2 km² for a full cell) causing 100% missing soil moisture and ~20% missing LST due to insufficient overlap with the coarser data sources' native pixel grids

## Satellite / reanalysis / forecast datasets

| Dataset | GEE ID / Source | Resolution | Period used | Access method |
|---|---|---|---|---|
| MODIS NDVI | `MODIS/061/MOD13Q1` | 250m, 16-day | 2000-2024 | Earth Engine, batched `.map()` + single `getInfo()` (region-level) / `Export.table.toDrive` (grid-level) |
| CHIRPS rainfall | `UCSB-CHG/CHIRPS/DAILY` | ~5km, daily→monthly | 1981-2024 | Earth Engine, same as above |
| MODIS LST | `MODIS/061/MOD11A2` | ~1km, 8-day | 2000-2024 | Earth Engine, same as above |
| ERA5-Land soil moisture | `ECMWF/ERA5_LAND/MONTHLY_AGGR`, band `volumetric_soil_water_layer_1` | ~11km, monthly | 2000-2024 | Earth Engine, same as above |
| ECMWF AIFS forecast | `ecmwf-opendata` Python package, `model="aifs-single"`, param `tp` | ~0.25°, daily steps to 15 days | Real-time | Direct API, GRIB2 via `cfgrib`/`xarray` |

### Known data-quality notes
- NDVI: QA-masked using `SummaryQA` band (0 = good quality only); ~4% of observations masked, normal cloud-contamination rate.
- LST: QA-masked using `QC_Day` bits 0-1 (mandatory QA flag); missing rates vary by cell, near-uniform except cell 0 (excluded, see above).
- ERA5-Land soil moisture: real-world processing lag of ~1-2 months from present; not available for the most recent 1-2 calendar months at any query time.
- CHIRPS: real-world processing lag of ~1 month from present.
- **AIFS unit note**: the `tp` (total precipitation) field is natively in `kg m**-2`, which is numerically equivalent to mm — no unit conversion needed. An earlier draft of the pipeline incorrectly assumed a meters-based convention and applied an erroneous ×1000 conversion, producing physically impossible values (600+ mm over 3 days); corrected before any downstream use.

## Drought label methodology

- **Basis**: SPI-6 (6-month Standardized Precipitation Index), computed via a gamma distribution fit per (grid cell × calendar month), transformed to a standard normal quantile.
- **Thresholds**: Severe ≤ -1.5, Moderate ≤ -1.0, Normal otherwise (standard SPI convention).
- **Why SPI-6 and not SPI-3 or VCI/NDVI**: SPI-3 (single-timescale) missed 4 of 8 literature-documented drought years (2004, 2005, 2010, 2011), which showed only mild single-month deficits but a clearer 6-month cumulative deficit — consistent with Kobo's bimodal (Belg + Kiremt) rainfall regime, where agricultural drought exposure is inherently a cumulative-seasonal phenomenon. NDVI/VCI were explicitly excluded from label construction to avoid target leakage, since they are used as model predictors.
- **Literature cross-check years**: 2004, 2005, 2008, 2009, 2010, 2011, 2013, 2015 (from published North Wollo drought studies using NDVI/VCI/TCI/VHI/SPEI, 2000-2019 period). 7 of 8 resolved at SPI-6; 2005 remains the sole unresolved case across every indicator tested.
