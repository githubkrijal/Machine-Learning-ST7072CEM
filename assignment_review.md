# Coursework review and implementation plan

Reviewed 2 November 2026. Student: Krijal Shrestha, Coventry ID 11495103.
User-confirmed current deadline: **6 November 2026** (submission time unspecified).

## What the brief requires

The supplied brief specifies an individual report of 6,000 words, following the
IEEE conference manuscript template, plus a Python prototype developed in
Jupyter Notebook or Google Colab. Apply three different classification or
clustering methods, explain preprocessing and parameters, and critically
evaluate the results. Submit code as a ZIP with comprehensive execution
instructions and a short screencast/video or link. Report naming convention:
NAME_studentID; submission report formats: DOCX/PDF.

The listed deadline is 12 October 2026, 17:00. This predates the review date;
the user has confirmed 6 November 2026 as the current deadline. The brief
permits any available datasets; its Kaggle/UCI suggestions do not require
replacing the Nepal data. A classification project is sufficient; clustering
and PCA are discussed examples, not compulsory additions.

The visible marking table assigns 40% to technical quality, 20% to evaluation
and significance, and 5% to social, ethical, legal and professional issues.
Those visible allocations total 65%. Pages 7–10 were visually inspected;
page 8 is blank and no remaining 35% allocation is visible in this supplied
copy. Do not invent the absent criteria.

## Proposal review

The proposal provides a workable binary classification problem and specifies
Logistic Regression, Random Forest and Gradient Boosting, the
2012–2014/2015/2016 temporal split, imbalanced evaluation and spatial outputs.
The paper-aligned experiment retains those models and adds a Linear SVM to
address the user's request to compare a method beyond Random Forest.

Document these differences in the final Methods section rather than
silently describing the implementation as identical to the proposal:

| Proposal | Actual implementation / action |
|---|---|
| ERA5-Land, hourly aggregation and dewpoint/U/V conversion | Current model uses ERA5 daily aggregates from Open-Meteo. Humidity and wind speed arrive already derived. Correct dataset citation and report 0.25-degree resolution. Two downloaded ERA5-Land months are not mixed into this experiment. |
| Full Nepal coverage | Only the completed weather subset is currently modelled. Geographic download order biases coverage. Final national conclusions require completing all eligible cells. |
| Sampled no-fire observations | Retain all eligible grid-days without detections; no downsampling. Class weighting affects training only. Explain that these are unverified no-detection labels. |
| Forest cover optional | Fixed 2011 forest mask defines eligibility, avoiding future land-cover leakage. Document omitted later forest changes and assess sensitivity later. |
| 1 February 2012–19 June 2016 | Modelling excludes both partial endpoint days, using 2 February 2012–18 June 2016. Acquisition time interpreted as UTC pending confirmation. |
| Seven-day lookback | Implementation also uses 30-day rainfall. Weather starts 1 January 2012, providing the necessary history. |
| Daily probability/risk | Weighted classifier outputs are uncalibrated scores. Do not call them reliable probabilities or operational warnings without calibration and further validation. |
| Weather usefulness objective | Add a validation-only ablation comparing the same boosting model with and without lagged weather. |
| Feature importance and mapped predictions | Validation permutation importance exists; add a chart, coverage map and illustrative historical prediction map. |

The proposal is three pages, while the brief requests a proposal no longer
than one A4 page. Keep the supplied document unchanged; if a proposal must
be resubmitted, condense it and move references/data inventory to supporting
material, subject to tutor guidance.

## Blueprint review

Use the three papers to motivate and critique the study, not as experiments
that this coursework has reproduced:

1. Dhakal et al. (2024): compares knowledge-based and machine-learning
   susceptibility maps in southern Nepal. Publisher abstract confirms RF
   AUC 0.95 and accuracy 0.91. Those values are not comparable to our daily
   temporal test because labels, sampling, geography and validation differ.
   DOI: https://doi.org/10.1080/19475705.2024.2436540
2. Dahal et al. (2025): national susceptibility, carbon and biodiversity
   context. Our prototype does not estimate carbon losses or biodiversity
   impacts. DOI: https://doi.org/10.1016/j.infgeo.2025.100003
3. Sarkar et al. (2024): regional susceptibility comparison and ensembling
   in Northeast India; supports comparing nonlinear models. It does not
   establish daily forecasting performance in Nepal.
   DOI: https://doi.org/10.1016/j.ecoinf.2024.102598

The blueprint's section word ranges exceed the coursework's total if added
together. They are a reading scaffold, not the report word budget. Treat
claims about missing sections or unclear cross-validation as reading notes
requiring full-paper checks, not settled criticisms. Article identity and
high-level scope were checked with publisher/institutional pages; not every
blueprint detail or numerical claim has been independently verified.

A defensible project contribution is a reproducible comparison of four
classifiers for next-day recorded fire detections, with lagged weather,
chronological validation and explicit no-detection uncertainty. Do not
claim that no previous daily fire prediction research exists.

## Submission structure and working word budget

This is a proposed 6,000-word allocation, not an instruction from the brief:

| Section | Approximate words |
|---|---:|
| Abstract | 200 |
| Introduction and objectives | 650 |
| Related work and critical comparison | 850 |
| Problem, study area and datasets | 850 |
| Methods and experimental setup | 1,450 |
| Results | 850 |
| Discussion, limitations and professional issues | 900 |
| Conclusion | 250 |
| Total | 6,000 |

Confirm the module's policy on references, tables and appendices in the word
count; the supplied brief does not state exclusions. The PDF is confidential:
retain it locally and do not put it into a public repository or public ZIP.

## Work sequence

1. Preserve original PDFs and data. Deliver an executed notebook that explains
   the problem, source audit, EDA, preparation, models and interpretation.
2. Demonstrate chronological splits, no same-day fire predictors, lag checks,
   natural class prevalence and training-only transforms/weights.
3. Add validation weather ablation, ROC/PR and importance plots, and maps.
4. Complete weather coverage and freeze the final dataset. Investigate sparse
   2015 labels, missing observations, confidence sensitivity and spatial
   generalisation. Do not repeatedly tune against the existing test scores.
5. The paper-aligned four-model run and two-column LaTeX manuscript are now
   prepared from the 218-location snapshot. Complete weather collection,
   refresh the manifest, rerun and revise results before final national claims.
   Include false negatives/alarms, uncertainty, licensing and
   environmental/community harms.
6. Prepare a self-contained code ZIP and demo instructions. Record the actual
   screencast; a script or storyboard alone is not the required video.

## Source verification

The ICIMOD metadata confirms MODIS-based Nepal detections from February 2012
to June 2016, overlaid with Nepal land cover, CC BY 4.0:
https://rds.icimod.org/metadata/94a16f41-6b99-4f0c-aa49-4c15ecc398f1
The metadata does not establish uninterrupted observation coverage or confirm
the supplied CSV's file checksum. Acquisition-time convention remains to be
checked. The low 2015 count is an audit concern, not proof of a download error.

Open-Meteo documentation: https://open-meteo.com/en/docs/historical-weather-api
Underlying ERA5: https://doi.org/10.24381/cds.adbb2d47

Source documents reviewed locally:
- STW7072CEM Machine Learning March Intake 2026 Coursework CW (Regular), supplied PDF.
- 11495103_KrijalShrestha_Blueprint.pdf, supplied PDF.
- Assignment_proposal, supplied PDF.

## Confirmed working schedule

- 2 November: brief/proposal review, executed notebook, preliminary comparisons and maps.
- 3–4 November: complete weather and source audits; freeze data and evaluation design.
- 4–5 November: final experiments, IEEE report, reproducibility package and demonstration.
- 6 November: final checks and submission by the applicable platform cutoff; time still to confirm.

The schedule depends on weather access. If coverage remains partial, narrow the
stated study area transparently rather than invent missing data or claim national
performance. Original proposal and blueprint PDFs were not modified.
