"""Temporal comparison of baseline, logistic regression, forest and boosting models."""
from pathlib import Path
import json
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'nepal-assignment-matplotlib'))
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             precision_score, recall_score, f1_score, confusion_matrix,
                             PrecisionRecallDisplay, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)


def threshold_for(y, probability):
    precision, recall, thresholds = precision_recall_curve(y, probability)
    f1 = np.divide(2*precision[:-1]*recall[:-1], precision[:-1]+recall[:-1],
                   out=np.zeros(len(thresholds)),where=precision[:-1]+recall[:-1]>0)
    return float(thresholds[np.argmax(f1)]) if len(thresholds) else .5


def evaluate(y, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0,1]).ravel()
    return dict(average_precision=average_precision_score(y, probability),
                roc_auc=roc_auc_score(y, probability),
                precision=precision_score(y, prediction, zero_division=0),
                recall=recall_score(y, prediction, zero_division=0),
                f1=f1_score(y, prediction, zero_division=0), threshold=threshold,
                true_negative=int(tn), false_positive=int(fp), false_negative=int(fn), true_positive=int(tp))


def main():
    data = pd.read_csv(ROOT/'data/prepared/forest_fire_grid_daily.csv')
    features = json.loads((ROOT/'data/prepared/feature_columns.json').read_text())
    parts = {s:data.loc[data['split']==s] for s in ['train','validation','test']}
    for s, part in parts.items():
        if part['fire_detected'].nunique()!=2:
            raise ValueError(f'{s} must contain both target classes')
    x, y = {}, {}
    for s, part in parts.items():
        x[s], y[s] = part[features], part['fire_detected']
    options = {'Dummy baseline':[DummyClassifier(strategy='prior')],
               'Logistic regression':[make_pipeline(StandardScaler(), LogisticRegression(C=c,class_weight='balanced',max_iter=2000,random_state=42)) for c in [.1,1]],
               'Random forest':[RandomForestClassifier(n_estimators=150,max_depth=d,min_samples_leaf=10,
                                   class_weight='balanced_subsample',n_jobs=2,random_state=42) for d in [12,None]],
               'Gradient boosting':[HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=n,
                                    learning_rate=.08,l2_regularization=1,early_stopping=False,
                                    class_weight='balanced',random_state=42) for n in [15,31]]}
    fitted = {}
    tuning=[]
    for name, candidates in options.items():
        best_score=-1
        for number, estimator in enumerate(candidates,1):
            print(f'Fitting {name}, configuration {number}/{len(candidates)}',flush=True)
            estimator.fit(x['train'], y['train'])
            prob = estimator.predict_proba(x['validation'])[:,1]
            score = average_precision_score(y['validation'],prob)
            tuning.append(dict(model=name,configuration=number,validation_average_precision=score,
                               parameters=str(estimator)))
            if score>best_score:
                best_score=score
                fitted[name]=(estimator,threshold_for(y['validation'],prob),score)
    # Choose model before examining test scores.
    selected=max((n for n in fitted if n!='Dummy baseline'), key=lambda n:fitted[n][2])
    print(f'Selected using validation average precision: {selected}',flush=True)
    metrics=[]
    fig, ax = plt.subplots(figsize=(7,5))
    predictions=parts['test'][['grid_id','date','fire_detected']].copy()
    for name,(estimator, threshold, score) in fitted.items():
        for s in ['validation','test']:
            prob=estimator.predict_proba(x[s])[:,1]
            metrics.append(dict(model=name,split=s,selected=name==selected,**evaluate(y[s],prob,threshold)))
            if s=='test':
                predictions[name+'_probability']=prob
                PrecisionRecallDisplay.from_predictions(y[s],prob,name=name,ax=ax)
    ax.set_title('Preliminary held-out 2016 precision–recall comparison')
    ax.axhline(float(y['test'].mean()),color='grey',linestyle=':',label='Test prevalence')
    ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'precision_recall.png',dpi=170);plt.close(fig)
    estimator,threshold,_=fitted[selected]
    selected_pred=(estimator.predict_proba(x['test'])[:,1]>=threshold).astype(int)
    fig,ax=plt.subplots(figsize=(5,4))
    ConfusionMatrixDisplay.from_predictions(y['test'],selected_pred,display_labels=['No recorded fire','Recorded fire'],ax=ax,colorbar=False)
    ax.set_title(f'Preliminary 2016 test: {selected}')
    fig.tight_layout();fig.savefig(OUT/'confusion_matrix.png',dpi=170);plt.close(fig)
    # Importance on validation only: no interpretation-driven test-set tuning.
    importance=permutation_importance(estimator,x['validation'],y['validation'],
                                     scoring='average_precision',n_repeats=3,random_state=42,n_jobs=2)
    pd.DataFrame({'feature':features,'mean_drop_in_validation_AP':importance.importances_mean,
                  'std':importance.importances_std}).sort_values('mean_drop_in_validation_AP',ascending=False).to_csv(OUT/'validation_feature_importance.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'model_comparison.csv',index=False)
    pd.DataFrame(tuning).to_csv(OUT/'validation_tuning.csv',index=False)
    predictions.to_csv(OUT/'test_predictions.csv',index=False)
    joblib.dump(dict(model=estimator,threshold=threshold,features=features,selected_by='2015 validation average precision'),OUT/'selected_model.joblib')
    audit=json.loads((ROOT/'data/prepared/preparation_audit.json').read_text())
    selected_test=next(m for m in metrics if m['model']==selected and m['split']=='test')
    comparison=pd.DataFrame(metrics).query("split == 'test'")
    table='| Model | Average precision | Precision | Recall | F1 |\n|---|---:|---:|---:|---:|\n'
    for _,r in comparison.iterrows():
        table+=f'| {r["model"]} | {r["average_precision"]:.3f} | {r["precision"]:.3f} | {r["recall"]:.3f} | {r["f1"]:.3f} |\n'
    counts='\n'.join(f'- {s}: {len(p):,} grid-days, {int(p.fire_detected.sum()):,} positive grid-days.' for s,p in parts.items())
    report=f'''# Forest-fire detection prediction in Nepal: preliminary experiment

## Problem and scope
Predict whether a 0.25-degree forest-containing cell will have at least one recorded fire detection during the next local calendar day. Weather features end at the previous day. This predicts recorded satellite detections, not verified ignitions, burned area, damage, or real-time operational risk.

Weather coverage is incomplete: **{audit['weather_complete_locations']}/238 requested cells**. These cells were downloaded in geographic order, so the subset is geographically biased and cannot establish performance across all Nepal. All results below are provisional and must be rerun after weather collection finishes.

## Data and preprocessing
- Main labels: supplied Active Fire Incidents in Nepal CSV, originally {audit['source_fire_rows']:,} rows. The source download URL and source monitoring completeness remain unverified.
- Detection duplicates removed: {audit['duplicate_detection_rows_removed']:,}. Duplicates use latitude/longitude rounded to five decimals, acquisition timestamp, and satellite; retain highest confidence. This does not identify unique fire incidents.
- Points outside the supplied Nepal polygon: {audit['outside_nepal_rows']:,}. Points not forest in the 2011 baseline: {audit['within_nepal_nonforest_rows']:,}.
- Acquisition time is assumed UTC from MODIS-style fields and converted to Asia/Kathmandu. Confirm source metadata before final submission. Incomplete first/last acquisition days are excluded. Prepared period: {audit['date_range'][0]} to {audit['date_range'][1]}.
- Forest: class 4 in the supplied ICIMOD/FRTC 2011 map, fixed for all years. This omits later forest gains and retains later losses. Other supplied map years are retained for future sensitivity checks.
- Boundary: geoBoundaries NPL ADM0, metadata year 2019, CC BY 4.0.
- Terrain: SRTM GL1 30 m. Bilinear alignment to the forest raster; slope/aspect calculated using approximate local metre spacing with block halos. Predictors use forest-pixel mean elevation, slope, aspect northness/eastness and baseline forest fraction. Terrain was calculated across all cells without using labels.
- Weather: ERA5 through Open-Meteo, daily Asia/Kathmandu, Celsius, mm, m/s; nearest weather cell, elevation adjustment disabled. Daily variables are shifted by one day; precipitation totals cover previous 1, 7 and 30 days; temperature/humidity means cover previous 7 days.
- Positive: one or more retained forest detections in a cell/day. Negative: no such recorded detection. Clouds, overpass gaps and missing source records can make these false negatives. No satellite observation mask is available.
- Dates with no detection anywhere nationally remain in the calendar, with a diagnostic national_detection_recorded flag. This flag is **not** a predictor. It cannot prove observation coverage.
- Confidence, FRP, brightness, source incident elevation, satellite, acquisition hour and detection counts are excluded from predictors. Confidence is not a non-fire label; all retained detection confidences are included.
- No negative downsampling or SMOTE: retain the natural recorded-label imbalance. Class weights used for the three learned models.
- Prepared {audit['daily_examples']:,} grid-day examples and {audit['predictor_count']} predictors. Target prevalence: {audit['positive_grid_days']/audit['daily_examples']:.2%}.

## Models and evaluation
Train: 2012–2014; validation: 2015; held-out test: available days in 2016. This measures later-time prediction at seen locations, not transfer to new areas.
{counts}

Implementations: logistic regression with training-only standardisation; random forest; histogram gradient boosting; and a constant-probability dummy baseline. Two modest hyperparameter configurations per learned model are compared on validation average precision. The classification threshold is selected by validation F1, never by test performance. Gradient boosting's internal random early stopping is disabled to preserve the temporal design. Fixed random seed: 42.

**Selected model: {selected}**, chosen using 2015 validation average precision. Average precision (a summary of the precision–recall curve) is the main ranking metric. It is not numerically identical to trapezoidal PR-AUC. ROC-AUC, precision, recall, F1 and confusion matrices are supplementary. Weighted models' raw scores should not be treated as calibrated fire probabilities.

### Preliminary test results
{table}
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
'''
    (OUT/'preliminary_report.md').write_text(report)
    (OUT/'run_summary.json').write_text(json.dumps(dict(selected_model=selected,provisional=True,
                        weather_locations=audit['weather_complete_locations'],selected_test=selected_test),indent=2))
    print(json.dumps(selected_test,indent=2),flush=True)


if __name__=='__main__':
    main()
