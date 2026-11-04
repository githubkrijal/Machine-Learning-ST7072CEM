# Review of supplied coursework examples and project commitments

Reviewed all five supplied PDFs, the three-page proposal and nine-page blueprint, with text extraction and selected rendered pages. Samples are examples of presentation, not scientific sources to cite as evidence for Nepal fire prediction.

1. Online shopping (14 pages): SVM, k-NN, Random Forest, AdaBoost, Extra Trees. Explores undersampling, oversampling and SMOTE; scaling and PCA/Kernel PCA; grid search; confusion matrices and several metrics. Adopt: explicit comparison of preprocessing choices. Improve: resample training only and keep natural prevalence in validation/test.
2. Audio features (10 pages): k-NN, Decision Tree and Random Forest. Documents missing fields, metadata exclusion, scaling, feature selection and before/after results. Adopt: field-specific exclusion reasons. Do not copy its generic claims about model robustness or transferable accuracy.
3. Human activity (6 pages): Decision Tree, k-NN and linear/Gaussian SVM; standardisation and reduction of 5,625 raw features to 50 PCs. Adopt: variance plots and interpretation of lost feature meaning. Its dimensionality is unlike this study's 21 predictors. Its written fold description is inconsistent with conventional ten-fold training; use an explicit chronological protocol here.
4. Stock direction (6 pages): SVM, Logistic Regression and Random Forest, lagged indicators, walk-forward windows, parameter searches. Adopt: past-only feature construction and temporal diagnostics. Do not select this study's hyperparameters from test windows.
5. Mahakali water quality (16 pages): missing-value audit and k-NN imputation, standardisation, SMOTE, PCA, k-means, RF/SVM/k-NN, equations and colourful figures. Adopt: evidence for every processing decision. WQI categories constructed from environmental inputs are not equivalent to satellite-derived forest-fire labels; its scores are not a target for this experiment.

Proposal commitments retained: ICIMOD incidents; 2012–2014 training, 2015 validation, early 2016 test; Logistic Regression, RF and Gradient Boosting; lagged weather, forest and terrain; leakage controls; multiple metrics and interpretation. Linear SVM adds a distinct margin objective. ERA5 through Open-Meteo replaces the planned ERA5-Land hourly route; disclose this and retain the actual product's identity.

Blueprint studies: Dhakal et al. (2024), Dahal et al. (2025), Sarkar et al. (2024). Separate their stated contributions from limits of transferring their evidence to next-day detection. Do not label daily prediction as never studied. Each foundation study gets a gap/limitation/response discussion in the manuscript.

Lecturer feedback response: report exact source audit counts, resulting sample, feature construction, pipeline order, model settings, selection rules, measured PCA/SMOTE ablations, temporal/spatial diagnostics, uncertainty, and test-exposure limitations. Link every numerical statement to saved outputs.
