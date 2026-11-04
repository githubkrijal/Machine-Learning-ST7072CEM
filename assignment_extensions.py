"""Extra evidence requested by the proposal; no test-set model selection."""
from pathlib import Path
import json
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'nepal-assignment-matplotlib'))
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PlotPolygon
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, RocCurveDisplay

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'


def boundary_on(ax):
    geometry=json.loads((ROOT/'data/sources/geoBoundaries-NPL-ADM0.geojson').read_text())['features'][0]['geometry']
    rings=geometry['coordinates'] if geometry['type']=='Polygon' else [r for p in geometry['coordinates'] for r in p]
    for ring in rings:
        a=np.array(ring);ax.plot(a[:,0],a[:,1],color='black',lw=.7)
    ax.set(xlabel='Longitude (degrees east)',ylabel='Latitude (degrees north)')
    ax.set_aspect(1/np.cos(np.radians(28)))


def main():
    OUT.mkdir(exist_ok=True)
    d=pd.read_csv(ROOT/'data/prepared/forest_fire_grid_daily.csv',parse_dates=['date'])
    audit=json.loads((ROOT/'data/prepared/preparation_audit.json').read_text())
    features=json.loads((ROOT/'data/prepared/feature_columns.json').read_text())
    train=d[d.split=='train'];val=d[d.split=='validation']
    # Lock this comparison before using its outcome. It is exploratory validation,
    # not an excuse to reselect the model against the previously seen test set.
    spec=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,learning_rate=.08,
             l2_regularization=1,early_stopping=False,class_weight='balanced',random_state=42)
    nonweather=[f for f in features if not ('previous' in f)]
    experiments=[]
    for name,cols in [('Location, terrain and season',nonweather),('Same inputs plus lagged weather',features)]:
        model=clone(spec).fit(train[cols],train.fire_detected)
        ap=average_precision_score(val.fire_detected,model.predict_proba(val[cols])[:,1])
        experiments.append(dict(experiment=name,features=len(cols),validation_average_precision=ap))
    pd.DataFrame(experiments).to_csv(OUT/'weather_ablation_validation.csv',index=False)
    raw=pd.read_csv(ROOT/'data/sources/Active_Fire_Incidents_in_Nepal.csv')
    years=pd.to_datetime(raw.ACQ_DATE).dt.year.value_counts().sort_index()
    fig,ax=plt.subplots(figsize=(7,4));years.plot.bar(ax=ax,color='#476e91')
    ax.set(xlabel='Year',ylabel='Source detection rows',title='Source incident counts: partial 2012 and 2016')
    fig.tight_layout();fig.savefig(OUT/'source_year_counts.png',dpi=160);plt.close(fig)
    locations=pd.read_csv(ROOT/'data/study_grid/nepal_forest_weather_locations.csv')
    complete=set(d.grid_id)
    fig,ax=plt.subplots(figsize=(9,5));boundary_on(ax)
    for flag,color,label in [(False,'#d3d3d3','Weather pending'),(True,'#245a81','Weather included')]:
        p=locations[locations.grid_id.isin(complete)==flag]
        ax.scatter(p.longitude,p.latitude,c=color,s=18,label=label)
    ax.legend();ax.set_title(f'Preliminary coverage: {len(complete)}/238 nominal weather cell centres')
    fig.tight_layout();fig.savefig(OUT/'weather_coverage.png',dpi=170);plt.close(fig)
    pred=pd.read_csv(OUT/'test_predictions.csv')
    fig,ax=plt.subplots(figsize=(7,5))
    for c in pred:
        if c.endswith('_probability'):
            RocCurveDisplay.from_predictions(pred.fire_detected,pred[c],name=c.removesuffix('_probability'),ax=ax)
    ax.plot([0,1],[0,1],ls=':',color='grey');ax.set_title('Preliminary held-out 2016 ROC curves')
    fig.tight_layout();fig.savefig(OUT/'roc_curves.png',dpi=160);plt.close(fig)
    imp=pd.read_csv(OUT/'validation_feature_importance.csv').head(10).sort_values('mean_drop_in_validation_AP')
    fig,ax=plt.subplots(figsize=(9,5))
    ax.barh(imp.feature,imp.mean_drop_in_validation_AP,xerr=imp['std'],color='#476e91')
    ax.set(xlabel='Drop in validation average precision',title='Validation permutation importance (exploratory)')
    fig.tight_layout();fig.savefig(OUT/'feature_importance.png',dpi=170);plt.close(fig)
    selected=json.loads((OUT/'run_summary.json').read_text())['selected_model']
    day='2016-04-15'  # Fixed illustrative date, not chosen by model score.
    subset=pred[pred.date==day].merge(locations[['grid_id','latitude','longitude']],on='grid_id')
    fig,ax=plt.subplots(figsize=(9,5));boundary_on(ax)
    ax.scatter(locations.longitude,locations.latitude,s=20,color='#ddd',label='Eligible cell centres')
    sc=ax.scatter(subset.longitude,subset.latitude,c=subset[selected+'_probability'],vmin=0,vmax=1,cmap='YlOrRd',s=32)
    fig.colorbar(sc,ax=ax,label='Uncalibrated classifier score')
    ax.set_title(f'Historical prediction example: {day}\nAvailable cells only; not an operational warning')
    fig.tight_layout();fig.savefig(OUT/'historical_prediction_map.png',dpi=170);plt.close(fig)
    print(pd.DataFrame(experiments).to_string(index=False))


if __name__=='__main__':
    main()
