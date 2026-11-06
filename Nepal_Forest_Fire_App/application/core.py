"""Read-only access to the full research run; no fitting or downloading."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'research_full'
MODELS = ('Linear SVM', 'Logistic regression', 'Random forest', 'Gradient boosting')
LABELS = {'Linear SVM': 'Linear SVM', 'Logistic regression': 'Logistic Regression',
          'Random forest': 'Random Forest', 'Gradient boosting': 'Gradient Boosting'}


def load_data():
    features = json.loads((RUN / 'data/feature_columns.json').read_text())
    columns = ['grid_id', 'date', 'fire_detected', 'split', *features]
    data = pd.read_csv(RUN / 'data/forest_fire_grid_daily.csv', usecols=columns, parse_dates=['date'])
    data = data.loc[data['split'].eq('test')].copy()
    if data.empty or data.duplicated(['grid_id', 'date']).any():
        raise ValueError('Expected unique test-period grid-days in the full-run dataset.')
    if not np.isfinite(data[features].to_numpy()).all():
        raise ValueError('Predictors contain missing or non-finite values.')
    return data.sort_values(['date', 'grid_id']).reset_index(drop=True)


def load_model(name):
    if name not in MODELS:
        raise ValueError('Unknown model')
    return joblib.load(RUN / 'results' / (name.replace(' ', '_') + '_F1.joblib'))


def infer(bundle, rows):
    x = rows.loc[:, bundle['features']]
    with threadpool_limits(limits=2):
        model = bundle['model']
        scores = model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(x)
    result = rows[['grid_id', 'date', 'latitude', 'longitude', 'fire_detected']].copy()
    result['score'] = np.asarray(scores)
    result['prediction'] = (result.score >= bundle['threshold']).astype(int)
    # Ranking is relative to the supplied day's cells, never a probability.
    result['rank_percentile'] = result.score.rank(pct=True, method='average') * 100
    return result


def load_metrics():
    metrics = pd.read_csv(RUN / 'results/metrics.csv')
    return metrics.loc[metrics.policy.eq('F1') & metrics.model.isin(MODELS)].copy()


def boundary_rings():
    path = ROOT / 'data/sources/geoBoundaries-NPL-ADM0.geojson'
    if not path.exists():
        return []
    obj = json.loads(path.read_text())
    rings = []
    for feature in obj['features']:
        geom = feature['geometry']
        polygons = [geom['coordinates']] if geom['type'] == 'Polygon' else geom['coordinates']
        for polygon in polygons:
            rings.append(polygon[0])
    return rings
