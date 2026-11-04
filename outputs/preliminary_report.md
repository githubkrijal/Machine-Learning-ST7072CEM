# Forest-fire detection prediction in Nepal: preliminary experiment

## Problem and scope
Predict whether a 0.25-degree forest-containing cell will have at least one recorded fire detection during the next local calendar day. Weather features end at the previous day. This predicts recorded satellite detections, not verified ignitions, burned area, damage, or real-time operational risk.

Weather coverage is incomplete: **108/238 requested cells**. These cells were downloaded in geographic order, so the subset is geographically biased and cannot establish performance across all Nepal. All results below are provisional and must be rerun after weather collection finishes.

## Data and preprocessing
- Main labels: supplied Active Fire Incidents in Nepal CSV, originally 13,781 rows. The source download URL and source monitoring completeness remain unverified.
- Detection duplicates removed: 224. Duplicates use latitude/longitude rounded to five decimals, acquisition timestamp, and satellite; retain highest confidence. This does not identify unique fire incidents.
- Points outside the supplied Nepal polygon: 1. Points not forest in the 2011 baseline: 3,298.
- Acquisition time is assumed UTC from MODIS-style fields and converted to Asia/Kathmandu. Confirm source metadata before final submission. Incomplete first/last acquisition days are excluded. Prepared period: 2012-02-02 to 2016-06-18.
- Forest: class 4 in the supplied ICIMOD/FRTC 2011 map, fixed for all years. This omits later forest gains and retains later losses. Other supplied map years are retained for future sensitivity checks.
- Boundary: geoBoundaries NPL ADM0, metadata year 2019, CC BY 4.0.
- Terrain: SRTM GL1 30 m. Bilinear alignment to the forest raster; slope/aspect calculated using approximate local metre spacing with block halos. Predictors use forest-pixel mean elevation, slope, aspect northness/eastness and baseline forest fraction. Terrain was calculated across all cells without using labels.
- Weather: ERA5 through Open-Meteo, daily Asia/Kathmandu, Celsius, mm, m/s; nearest weather cell, elevation adjustment disabled. Daily variables are shifted by one day; precipitation totals cover previous 1, 7 and 30 days; temperature/humidity means cover previous 7 days.
- Positive: one or more retained forest detections in a cell/day. Negative: no such recorded detection. Clouds, overpass gaps and missing source records can make these false negatives. No satellite observation mask is available.
- Dates with no detection anywhere nationally remain in the calendar, with a diagnostic national_detection_recorded flag. This flag is **not** a predictor. It cannot prove observation coverage.
- Confidence, FRP, brightness, source incident elevation, satellite, acquisition hour and detection counts are excluded from predictors. Confidence is not a non-fire label; all retained detection confidences are included.
- No negative downsampling or SMOTE: retain the natural recorded-label imbalance. Class weights used for the three learned models.
- Prepared 172,692 grid-day examples and 20 predictors. Target prevalence: 1.35%.

## Models and evaluation
Train: 2012–2014; validation: 2015; held-out test: available days in 2016. This measures later-time prediction at seen locations, not transfer to new areas.
- train: 114,912 grid-days, 1,497 positive grid-days.
- validation: 39,420 grid-days, 76 positive grid-days.
- test: 18,360 grid-days, 765 positive grid-days.

Implementations: logistic regression with training-only standardisation; random forest; histogram gradient boosting; and a constant-probability dummy baseline. Two modest hyperparameter configurations per learned model are compared on validation average precision. The classification threshold is selected by validation F1, never by test performance. Gradient boosting's internal random early stopping is disabled to preserve the temporal design. Fixed random seed: 42.

**Selected model: Gradient boosting**, chosen using 2015 validation average precision. Average precision (a summary of the precision–recall curve) is the main ranking metric. It is not numerically identical to trapezoidal PR-AUC. ROC-AUC, precision, recall, F1 and confusion matrices are supplementary. Weighted models' raw scores should not be treated as calibrated fire probabilities.

### Preliminary test results
| Model | Average precision | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Dummy baseline | 0.042 | 0.042 | 1.000 | 0.080 |
| Logistic regression | 0.400 | 0.364 | 0.639 | 0.464 |
| Random forest | 0.435 | 0.440 | 0.495 | 0.466 |
| Gradient boosting | 0.488 | 0.621 | 0.299 | 0.404 |

Thresholds differ by model and were fixed on validation. The test period is only part of 2016, and the source has unusually few 2015 detections; source completeness must be investigated. No claim of statistically significant superiority is made.

## What is still needed before final conclusions
1. Complete weather downloads and rerun both scripts; preserve a dated snapshot of these preliminary outputs for comparison.
2. Establish incident-source provenance, time convention and acquisition coverage. A no-detection grid-day is not confirmed absence.
3. Inspect performance by region/season and run a spatial holdout before claiming geographic generalisation. Test sensitivity to baseline forest definition, low-confidence detections and missing national detection days.
4. Add calibration and more temporal validation folds if interpreting probabilities or making operational claims. The current workflow is a course prototype.

## Sources
- Fire incidents and SRTM archive supplied by the user; retain their download provenance.
- FRTC (2022), Land cover of Nepal: https://doi.org/10.26066/RDS.1972729
- geoBoundaries metadata and attribution in data/sources.
- Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api
- Underlying ERA5: https://doi.org/10.24381/cds.adbb2d47
