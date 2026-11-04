"""Run the full-data Nepal experiment from the supplied prepared dataset.

Examples:
    python run_nepal_analysis.py
    python run_nepal_analysis.py --retrain
    python run_nepal_analysis.py --retrain --figures
    python run_nepal_analysis.py --rebuild-raw --retrain --figures
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
import pandas as pd
import reanalysis_2026 as analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrain', action='store_true', help='Repeat model fitting and evaluation from prepared data')
    parser.add_argument('--rebuild-raw', action='store_true', help='Rebuild predictors from original weather, fire and raster inputs')
    parser.add_argument('--figures', action='store_true', help='Regenerate the thirteen publication figures')
    args = parser.parse_args()
    if args.rebuild_raw:
        data, features = analysis.ex.prepare()
    else:
        data = pd.read_csv(analysis.DATA / 'forest_fire_grid_daily.csv', parse_dates=['date'])
        features = json.loads((analysis.DATA / 'feature_columns.json').read_text())
    assert data.grid_id.nunique() == 238, 'Expected the full 238-cell dataset'
    assert not data.duplicated(['grid_id', 'date']).any()
    print(f'{len(data):,} grid-days; {int(data.fire_detected.sum()):,} positives; {len(features)} predictors')
    if args.retrain:
        analysis.ex.run(data, features)
        analysis.extra(data, features)
    if args.figures:
        import runpy
        runpy.run_path(str(ROOT / 'research_full_figures.py'), run_name='__main__')
    results = pd.read_csv(analysis.OUT / 'metrics.csv')
    print(results.query("split == 'test' and policy == 'F1'")[['model', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'average_precision']].to_string(index=False))


if __name__ == '__main__':
    main()
