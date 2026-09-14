# Methodology

Technical write-up of the modeling and validation decisions behind the Kobo Drought Early Warning System. For the narrative version of this project, see [`CASE_STUDY.md`](CASE_STUDY.md). For data provenance details, see [`DATA_SOURCES.md`](DATA_SOURCES.md).

## Data sources and real-world latency

Every source updates on its own schedule, and the dashboard states each one's actual lag rather than implying everything reflects "right now":

| Source | Variable | Native resolution | Typical real-world lag |
|---|---|---|---|
| MODIS MOD13Q1 | NDVI | 250m, 16-day composite | ~2–4 weeks |
| MODIS MOD11A2 | Land surface temperature | 1km, 8-day composite | ~1–2 weeks |
| CHIRPS | Rainfall | ~5km, daily | ~1 month |
| ERA5-Land | Soil moisture | ~11km, monthly | ~1–2 months |
| ECMWF AIFS-ENS | Precipitation forecast | ~0.25°, 15-day horizon | Real-time (twice daily) |

This mismatch is a real operational constraint, not an oversight: a live status computed "now" is necessarily built from NDVI that's several weeks old sitting alongside a rainfall forecast that's hours old. The dashboard's "Data currency" panel makes each source's actual age visible rather than smoothing over the difference.

## Why logistic regression

Three models were compared under a **temporal** train/validation/test split (train 2000–2012, validation 2013–2020, test 2021–2024) — never a random split, since a random split would let information from the same drought episode leak across train and test.

Baselines were evaluated first, before any machine learning model, per standard practice:

| Model | Balanced accuracy (val) | Macro F1 (val) |
|---|---|---|
| VCI threshold rule | 0.583 | 0.441 |
| SPI-3 threshold rule | 0.622 | 0.672 |
| **Logistic regression** | **0.834** | **0.687** |
| Random Forest (tuned) | 0.610 | 0.657 |
| XGBoost (tuned) | 0.618 | 0.647 |

Logistic regression — the simplest model tested — outperformed both tuned tree ensembles. The training period contains only 13 Moderate and 4 Severe drought-months across 13 years; tree-based models did not have enough minority-class examples to out-learn a well-regularized linear model. The model was selected on validation evidence, not on the assumption that a more sophisticated algorithm would necessarily do better — in this case it didn't.

## Validation approach

- **Labels**: SPI-6 (6-month Standardized Precipitation Index) only — a purely precipitation-based index, deliberately excluding NDVI/VCI/LST/soil moisture, all of which are used as *predictors*. This separation avoids the model trivially recovering its own label.
- **Held-out test evaluation** (2021–2024, a period never touched during model selection): overall accuracy on this set was 81.2%, actually *lower* than a naive "always predict Normal"-style climatology baseline (87.5%). Read in isolation this looks like a regression — it is not. The test set is heavily imbalanced (42 Normal / 3 Moderate / 3 Severe months), and the climatology baseline achieves its accuracy by never once predicting Moderate or Severe. The metric that actually matters for a warning system — recall on the elevated-risk classes — tells the real story: the deployed model catches 100% of Moderate and 67% of Severe events in the test period, versus 0% and 0% for the climatology baseline. The dashboard's validation section leads with this recall comparison, not raw accuracy, specifically to avoid presenting a misleading headline number.
- **Backtesting**: running the model across the full historical record and measuring lead time to each documented Severe event gives a median of 3 months of advance warning, holding up in the genuinely out-of-sample validation and test periods. This comes with a real cost: a 64% false alarm rate on elevated predictions, the direct consequence of tuning the model (`class_weight="balanced"`) to favor catching real events over avoiding false alarms.
- **Cross-validation against independent literature**: every model output was checked against eight drought years documented in published North Wollo agricultural-drought studies (2004, 2005, 2008, 2009, 2010, 2011, 2013, 2015) before being trusted. Seven of eight were confirmed; 2005 remained unresolved across every indicator tested (NDVI, VCI, seasonal rainfall, SPI-3, SPI-6) and is reported as an honest limitation rather than tuned away.

## Known limitations and failure modes

- **Small event counts.** 25 years of monthly data yields only 17 Moderate/Severe months total across train+val+test combined. This is a hard ceiling on what any model — especially anything more complex than a linear one — can learn about the rare classes, and is a domain constraint, not a modeling error.
- **64% false alarm rate.** A direct, quantified consequence of prioritizing recall over precision. Reported alongside the lead-time statistic specifically because one number without the other is a misleading claim.
- **The 2005 case.** One literature-documented drought year that no indicator in this pipeline — NDVI, VCI, seasonal rainfall totals, SPI-3, or SPI-6 — flags as anomalous. Left unresolved and stated as such, rather than adjusting thresholds until it "worked."
- **Spatial grid trade-off.** Moving from a single woreda-wide average to a 25-cell spatial grid (to produce an actual risk map) cost real accuracy: 0.834 → 0.761 balanced accuracy. The resulting map's accuracy is lowest in the lowland southeast — the same area with the highest true severe-drought exposure. This is stated on the dashboard, not hidden.
- **Validation is against literature and remote-sensing cross-checks, not ground-truth field data.** No crop yield records, food security surveys, or household-level outcome data were available for this project. The validation performed is the best available given that constraint, not a substitute for it.
- **The forecast is a real ensemble now, but still a short-range one.** AIFS-ENS gives a genuine spread across ensemble members (10 perturbed + control, out of 50 available — a deliberate subset chosen to keep the live app's latency reasonable), but only out to 15 days. It says nothing about conditions a month or a season out.
- **Live computation depends on a scheduled job, not the request path.** If the scheduled refresh job fails silently for an extended period, the dashboard will show a "stale" warning past 8 hours, but there's no separate alerting on the job's own health beyond GitHub's own workflow-failure notifications.

## What I'd build next

- A genuine second region, not the structural proof-of-concept currently in place — ideally a woreda with a meaningfully different rainfall regime, to test whether the SPI-6 labeling approach and feature set generalize or need rework.
- Ground-truth validation against whatever food-security or crop outcome data can actually be sourced for North Wollo, even at a coarse level, to check the remote-sensing-only validation against something closer to real-world impact.
- A real notification pipeline behind the current alerting stub — the validation/storage layer exists; actually sending an email or firing a webhook does not yet.
- Monitoring on the scheduled refresh job itself (a dead-man's-switch style alert if it fails to produce a fresh result for longer than expected), rather than relying on a visitor noticing the dashboard's own staleness warning.
- Revisiting the false-alarm/recall trade-off with stakeholder input on which failure mode actually costs more in this specific context, rather than a single fixed `class_weight` choice.
