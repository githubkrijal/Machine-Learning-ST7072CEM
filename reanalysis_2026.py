"""Full 238-location reanalysis; original snapshots and sources remain intact."""
from pathlib import Path
import json,sys,time,hashlib
import research_experiments as ex
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'research_full';DATA=BASE/'data';OUT=BASE/'results'
for path in [DATA,OUT]:path.mkdir(parents=True,exist_ok=True)
ex.BASE=BASE;ex.DATA=DATA;ex.OUT=OUT

def extra(d,features):
    tr=d[d.split.eq('train')];va=d[d.split.eq('validation')];te=d[d.split.eq('test')]
    variants={
      'Unweighted Logistic':Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=1,max_iter=3000,random_state=42))]),
      'Weighted Logistic':Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=1,class_weight='balanced',max_iter=3000,random_state=42))]),
      'PCA Weighted Logistic':Pipeline([('scale',StandardScaler()),('pca',PCA(n_components=.95,svd_solver='full')),('model',LogisticRegression(C=1,class_weight='balanced',max_iter=3000,random_state=42))]),
      'SMOTE Logistic':Pipeline([('scale',StandardScaler()),('smote',SMOTE(sampling_strategy=.1,k_neighbors=5,random_state=42)),('model',LogisticRegression(C=1,max_iter=3000,random_state=42))]),
      'PCA SMOTE Logistic':Pipeline([('scale',StandardScaler()),('pca',PCA(n_components=.95,svd_solver='full')),('smote',SMOTE(sampling_strategy=.1,k_neighbors=5,random_state=42)),('model',LogisticRegression(C=1,max_iter=3000,random_state=42))])}
    rows=[]
    for name,model in variants.items():
      start=time.perf_counter();model.fit(tr[features],tr.fire_detected);s=ex.scores(model,va[features]);t=ex.threshold(va.fire_detected,s,'f1')
      for split,g in [('validation',va),('test',te)]:rows.append(dict(variant=name,split=split,seconds=time.perf_counter()-start,**ex.metrics(g.fire_detected,ex.scores(model,g[features]),t),pca_components=int(model.named_steps['pca'].n_components_) if 'pca' in model.named_steps else len(features)))
      print('Ablation',name,rows[-2]['average_precision'],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'preprocessing_ablation.csv',index=False)
    # Earlier forward-year check: use each family's fixed middle configuration, no retuning.
    early=tr[tr.date.dt.year<=2013];later=tr[tr.date.dt.year==2014];forward=[]
    for name,choices in ex.candidates().items():
      model=clone(choices[1]).fit(early[features],early.fire_detected)
      forward.append(dict(model=name,training_rows=len(early),evaluation_rows=len(later),positive_days=int(later.fire_detected.sum()),average_precision=average_precision_score(later.fire_detected,ex.scores(model,later[features]))))
    pd.DataFrame(forward).to_csv(OUT/'forward_2014.csv',index=False)
    z=StandardScaler().fit_transform(tr[features]);pca=PCA().fit(z)
    pd.DataFrame({'component':np.arange(1,len(features)+1),'variance':pca.explained_variance_ratio_,'cumulative':np.cumsum(pca.explained_variance_ratio_)}).to_csv(OUT/'pca_variance.csv',index=False)
    pd.DataFrame({'feature':features,'skewness':tr[features].skew(),'minimum':tr[features].min(),'maximum':tr[features].max(),'missing':tr[features].isna().sum()}).to_csv(OUT/'feature_audit.csv',index=False)
    weather=[]
    from download_daily_weather import VARIABLES
    for f in sorted((ROOT/'data/open_meteo_daily').glob('*.csv')):
      w=pd.read_csv(f);weather.append(dict(grid_id=f.stem,rows=len(w),duplicate_dates=int(w.date.duplicated().sum()),missing=int(w[VARIABLES].isna().sum().sum()),finite=bool(np.isfinite(w[VARIABLES]).all().all()),temperature_order_bad=int((w.temperature_2m_mean>w.temperature_2m_max).sum()),humidity_order_bad=int((w.relative_humidity_2m_min>w.relative_humidity_2m_mean).sum()),wind_order_bad=int((w.wind_speed_10m_mean>w.wind_speed_10m_max).sum())))
    if weather:
      assert len(weather) == 238, 'Provide all 238 source weather files for a fresh weather audit'
      pd.DataFrame(weather).to_csv(OUT/'weather_audit.csv',index=False)
    elif not (OUT/'weather_audit.csv').exists():
      raise FileNotFoundError('Source weather files or the archived weather audit are required')
    summary=json.loads((OUT/'summary.json').read_text());summary['provisional']=False;summary['run_date']='2026-11-04';(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    audit=json.loads((DATA/'preparation_audit.json').read_text());audit['snapshot_date']='2026-11-04';(DATA/'preparation_audit.json').write_text(json.dumps(audit,indent=2))
    print('FULL REANALYSIS COMPLETE',flush=True)

if __name__=='__main__':
    d,f=ex.prepare();assert d.grid_id.nunique()==238
    ex.run(d,f);extra(d,f)
