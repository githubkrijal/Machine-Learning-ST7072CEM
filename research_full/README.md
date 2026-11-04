# Full Nepal forest-fire analysis — November 2026

This full run uses 238 validated weather locations, 434,826 weather dates, and 380,562 modelling cell-days with 4,399 positive labels. Source observations and earlier 218-location experiments are preserved.

## Run in the existing workspace

1. Use Python 3.10 and install the versions in `research_full/requirements-lock.txt` into an isolated environment.
2. Open `Nepal_Forest_Fire_Full_Analysis.ipynb` with the project root as working directory. Its default cells verify and display the completed experiment.
3. Set `RETRAIN=True` to repeat the model comparison and diagnostic experiments from the prepared dataset.
4. Set `REBUILD_RAW=True` as well to rerun source integration. This requires the existing `data/sources`, `data/study_grid/lc2011.tif`, `data/terrain/nepal_SRTMGL1.tif`, and all `data/open_meteo_daily` inputs. These original files remain in the user's workspace; large raw rasters are not duplicated into the analysis ZIP.

## Commands you can run

From the extracted analysis folder, create an isolated environment using Python 3.10:

    python3.10 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r research_full/requirements-lock.txt

Show and verify the completed dataset/results:

    python run_nepal_analysis.py

Repeat model training and evaluation from the included prepared dataset, without needing the original rasters:

    python run_nepal_analysis.py --retrain --figures

Open `Nepal_Forest_Fire_Full_Analysis.ipynb` in VS Code and select the Python environment created above. Alternatively, install the optional browser notebook interface:

    python -m pip install notebook
    python -m jupyter notebook Nepal_Forest_Fire_Full_Analysis.ipynb

Model retraining overwrites the saved full-run model outputs and figures. It does not download anything. If original weather CSVs are absent from the extracted bundle, the archived audit is retained and is not presented as a new raw-file validation.

The publication figures use vertical panels sized to fit one LaTeX column. Every figure in the paper is a single-column figure.

Command-line equivalent for a full raw rebuild and model run (requires original weather and raster inputs):

    .weather-venv/bin/python -u reanalysis_2026.py
    .weather-venv/bin/python research_full_figures.py

`research_experiments.py`, `prepare_assignment.py`, and `download_daily_weather.py` supply shared preparation and fitting functions. The new entry point redirects outputs to `research_full`, including a new 238-location manifest. Earlier `research` outputs are not overwritten.

## Evidence

- `data/preparation_audit.json`: source exclusions, final sample, splits and dataset hash.
- `data/weather_snapshot.json`: source CSV hashes.
- `results/raw_audit.json`, `source_missingness.csv`, `weather_audit.csv`, `feature_audit.csv`: source and predictor checks.
- `results/tuning.csv`, `protocol.json`, `metrics.csv`, `test_predictions.csv`: configurations, thresholds, predictions and measured metrics.
- `results/preprocessing_ablation.csv`, `pca_variance.csv`: controlled class weighting, PCA and SMOTE comparisons.
- `results/forward_2014.csv`, `spatial_diagnostic.csv`, `ablation.csv`, `ap_intervals.csv`, `importance.csv`: robustness, feature and uncertainty evidence.
- `document_review/sample_review.md`: how the supplied examples informed the research design and where their protocols should not be copied.

Primary settings: 2012–2014 train, 2015 validation, 2016-01-01–2016-06-18 test; three candidates per each of four models; validation AP selects configuration, validation F1 selects decision threshold; class weights for main comparison; seed 42. PCA/SMOTE variants use fixed Logistic Regression C=1 and untouched validation/test prevalence.

The dataset is heavily imbalanced. Ordinary accuracy remains near 96%; balanced accuracy for the main comparison is 68.40–76.59%. These metrics are labelled separately. The 2016 period was explored in earlier project work, so it is a retrospective test, not a fresh independent external evaluation.

The paper remains in the user's selected Downloads LaTeX project. The Overleaf archive contains its current main source, references and figures. Add the repository URL after `GitHub Link:` in the appendix before submission. The coursework brief also requests a running-code screencast; this bundle does not include a recorded video.
