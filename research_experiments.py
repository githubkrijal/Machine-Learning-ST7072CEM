"""Frozen, reproducible chronological comparison for the two-column paper."""
from pathlib import Path
import os,json,hashlib,time,shutil
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/nepal-research-mpl')
import numpy as np,pandas as pd,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,confusion_matrix,accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score
from sklearn.inspection import permutation_importance
from sklearn.base import clone
import prepare_assignment as prep
from download_daily_weather import VARIABLES
ROOT=Path(__file__).resolve().parent;BASE=ROOT/'research';DATA=BASE/'data';OUT=BASE/'results'
for p in [DATA,OUT]:p.mkdir(parents=True,exist_ok=True)

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
    manifest=DATA/'weather_snapshot.json'
    if not manifest.exists():
        paths=sorted((ROOT/'data/open_meteo_daily').glob('*.csv'))
        manifest.write_text(json.dumps({p.stem:digest(p) for p in paths},indent=2))
    frozen=json.loads(manifest.read_text());expected=pd.date_range('2012-01-01','2016-12-31')
    for gid,sha in frozen.items():
        p=ROOT/'data/open_meteo_daily'/f'{gid}.csv';w=pd.read_csv(p,parse_dates=['date'])
        assert digest(p)==sha and len(w)==1827 and w.date.tolist()==list(expected)
        assert np.isfinite(w[VARIABLES]).all().all() and w.grid_id.eq(gid).all()
        assert w.precipitation_sum.ge(0).all() and w.relative_humidity_2m_mean.between(0,100).all()
    # Existing routine reads available inputs; freeze the final table to the manifest.
    prep.OUT=DATA;prep.main()
    d=pd.read_csv(DATA/'forest_fire_grid_daily.csv',parse_dates=['date']);d=d[d.grid_id.isin(frozen)].copy()
    dry=[]
    for gid in frozen:
        w=pd.read_csv(ROOT/'data/open_meteo_daily'/f'{gid}.csv',parse_dates=['date'])
        isdry=w.precipitation_sum.lt(1);runs=isdry.groupby((~isdry).cumsum()).cumsum()
        # Cap at 30 days so every retained target has a complete lookback.
        dry.append(pd.DataFrame({'grid_id':gid,'date':w.date,'dry_days_previous':runs.shift(1).clip(upper=30)}))
    d=d.merge(pd.concat(dry),on=['grid_id','date'],validate='one_to_one')
    features=json.loads((DATA/'feature_columns.json').read_text())+['dry_days_previous']
    assert d[features].notna().all().all() and not d.duplicated(['grid_id','date']).any()
    assert np.isfinite(d[features]).all().all()
    d.to_csv(DATA/'forest_fire_grid_daily.csv',index=False)
    (DATA/'feature_columns.json').write_text(json.dumps(features,indent=2))
    audit=json.loads((DATA/'preparation_audit.json').read_text())
    audit.update(weather_complete_locations=len(frozen),modelling_locations=d.grid_id.nunique(),daily_examples=len(d),positive_grid_days=int(d.fire_detected.sum()),predictor_count=len(features),provisional=len(frozen)<238,splits=d.groupby('split').agg(rows=('fire_detected','size'),positive_days=('fire_detected','sum')).to_dict('index'))
    audit['dataset_sha256']=digest(DATA/'forest_fire_grid_daily.csv');audit['snapshot_date']='2026-11-03'
    (DATA/'preparation_audit.json').write_text(json.dumps(audit,indent=2))
    pd.read_csv(ROOT/'data/sources/Active_Fire_Incidents_in_Nepal.csv').isna().sum().rename('missing').to_csv(OUT/'source_missingness.csv')
    d.loc[d.split.eq('train'),features].describe().T.to_csv(OUT/'training_descriptives.csv')
    d.loc[d.split.eq('train'),features].corr(method='spearman').to_csv(OUT/'training_correlations.csv')
    # Exact independently recomputed lag/rolling checks at each location.
    for gid,g in d.groupby('grid_id'):
        w=pd.read_csv(ROOT/'data/open_meteo_daily'/f'{gid}.csv',parse_dates=['date']).set_index('date')
        np.testing.assert_allclose(g.temperature_2m_mean_previous_day,w.temperature_2m_mean.reindex(g.date-pd.Timedelta(days=1)))
        np.testing.assert_allclose(g.precipitation_previous_30d_mm,w.precipitation_sum.shift(1).rolling(30).sum().reindex(g.date))
    print(json.dumps(audit,indent=2),flush=True)
    return d,features

def scores(model,x):
    return model.predict_proba(x)[:,1] if hasattr(model,'predict_proba') else model.decision_function(x)

def threshold(y,s,objective):
    y=np.asarray(y);s=np.asarray(s);order=np.argsort(-s,kind='stable');ys=y[order];ss=s[order]
    ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1];tp=np.cumsum(ys)[ends];fp=ends+1-tp;fn=y.sum()-tp;tn=len(y)-y.sum()-fp
    values=(tp+tn)/len(y) if objective=='accuracy' else 2*tp/np.maximum(2*tp+fp+fn,1)
    # Include all-negative rule; prefer highest threshold at ties.
    thresholds=np.r_[np.nextafter(ss[0],np.inf),ss[ends]]
    base=(y==0).mean() if objective=='accuracy' else 0
    return float(thresholds[np.argmax(np.r_[base,values])])

def metrics(y,s,t):
    pred=np.asarray(s)>=t;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(accuracy=accuracy_score(y,pred),balanced_accuracy=balanced_accuracy_score(y,pred),average_precision=average_precision_score(y,s),roc_auc=roc_auc_score(y,s),precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),threshold=t,tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def candidates():
    return {'Logistic regression':[make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight='balanced',max_iter=3000,random_state=42)) for c in [.01,.1,1]],
    'Linear SVM':[make_pipeline(StandardScaler(),LinearSVC(C=c,class_weight='balanced',dual=False,max_iter=10000,random_state=42)) for c in [.01,.1,1]],
    'Random forest':[RandomForestClassifier(n_estimators=150,max_depth=n,min_samples_leaf=10,class_weight='balanced_subsample',n_jobs=2,random_state=42) for n in [8,12,None]],
    'Gradient boosting':[HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=n,learning_rate=.08,l2_regularization=1,early_stopping=False,class_weight='balanced',random_state=42) for n in [7,15,31]]}

def run(d,features):
    parts={s:d[d.split.eq(s)] for s in ['train','validation','test']};tr=parts['train'];va=parts['validation'];te=parts['test']
    assert tr.date.max()<va.date.min()<va.date.max()<te.date.min()
    fitted={};tuning=[]
    protocol={'seed':42,'families':list(candidates()),'configurations_per_family':3,'primary_selection':'2015 average precision','secondary_selection':'2015 accuracy with validation-accuracy threshold','threshold_rules':['F1','accuracy'],'train':'2012-02-02 to 2014-12-31','validation':'2015','test':'2016-01-01 to 2016-06-18','features':features,'data_sha256':digest(DATA/'forest_fire_grid_daily.csv'),'previous_test_exposure':'Earlier 108-location exploratory analysis used overlapping 2016 labels; this is not a pristine untouched external test.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for name,options in candidates().items():
        rows=[]
        for i,m in enumerate(options):
            start=time.perf_counter();m.fit(tr[features],tr.fire_detected);elapsed=time.perf_counter()-start
            s=scores(m,va[features]);tf=threshold(va.fire_detected,s,'f1');ta=threshold(va.fire_detected,s,'accuracy')
            row=dict(model=name,configuration=i+1,fit_seconds=elapsed,validation_ap=average_precision_score(va.fire_detected,s),validation_accuracy=accuracy_score(va.fire_detected,s>=ta),f1_threshold=tf,accuracy_threshold=ta,parameters=str(m))
            tuning.append(row);rows.append((row,m));print(name,i+1,row['validation_ap'],flush=True)
        best_ap=max(rows,key=lambda v:v[0]['validation_ap']);best_acc=max(rows,key=lambda v:v[0]['validation_accuracy'])
        fitted[name]={'F1':best_ap,'accuracy':best_acc}
    pd.DataFrame(tuning).to_csv(OUT/'tuning.csv',index=False)
    selected=max(fitted,key=lambda n:fitted[n]['F1'][0]['validation_ap'])
    selected_acc=max(fitted,key=lambda n:fitted[n]['accuracy'][0]['validation_accuracy'])
    results=[];preds=te[['grid_id','date','fire_detected','latitude','longitude']].copy()
    for name,policies in fitted.items():
        for policy,(row,m) in policies.items():
            t=row['f1_threshold' if policy=='F1' else 'accuracy_threshold']
            for split,p in [('validation',va),('test',te)]:
                s=scores(m,p[features]);results.append(dict(model=name,policy=policy,configuration=row['configuration'],split=split,**metrics(p.fire_detected,s,t)))
                if split=='test':
                    preds[name+'_'+policy+'_score']=s;preds[name+'_'+policy+'_prediction']=(s>=t).astype(int)
            joblib.dump({'model':m,'features':features,'threshold':t},OUT/(name.replace(' ','_')+'_'+policy+'.joblib'))
    for split,p in [('validation',va),('test',te)]:
        results.append(dict(model='Always no detection',policy='baseline',configuration=0,split=split,**metrics(p.fire_detected,np.zeros(len(p)),1)))
    pd.DataFrame(results).to_csv(OUT/'metrics.csv',index=False);preds.to_csv(OUT/'test_predictions.csv',index=False)
    best=fitted[selected]['F1'][1]
    imp=permutation_importance(best,va[features],va.fire_detected,scoring='average_precision',n_repeats=3,n_jobs=2,random_state=42)
    pd.DataFrame({'feature':features,'mean':imp.importances_mean,'std':imp.importances_std}).sort_values('mean',ascending=False).to_csv(OUT/'importance.csv',index=False)
    # Validation-only weather ablation: same fixed boosting complexity for both sets.
    ab=[]
    base=[f for f in features if 'previous' not in f]
    for label,cols in [('Terrain, location and season',base),('Plus lagged weather',features)]:
        m=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,learning_rate=.08,l2_regularization=1,early_stopping=False,class_weight='balanced',random_state=42).fit(tr[cols],tr.fire_detected)
        ab.append({'features':label,'count':len(cols),'validation_ap':average_precision_score(va.fire_detected,scores(m,va[cols]))})
    pd.DataFrame(ab).to_csv(OUT/'ablation.csv',index=False)
    # Fixed hyperparameters from 2015: geographical diagnostic within 2014 only.
    # Longitudes >=85 held out of training; do not use these scores to retune.
    sptrain=tr[(tr.date.dt.year<=2013)&(tr.longitude<85)];sptest=tr[(tr.date.dt.year==2014)&(tr.longitude>=85)]
    spatial=[]
    for name in fitted:
        m=clone(fitted[name]['F1'][1]).fit(sptrain[features],sptrain.fire_detected)
        spatial.append({'model':name,'training_rows':len(sptrain),'evaluation_rows':len(sptest),'positive_days':int(sptest.fire_detected.sum()),'average_precision':average_precision_score(sptest.fire_detected,scores(m,sptest[features])),'roc_auc':roc_auc_score(sptest.fire_detected,scores(m,sptest[features]))})
    pd.DataFrame(spatial).to_csv(OUT/'spatial_diagnostic.csv',index=False)
    # Paired weekly block bootstrap for AP difference (models fixed).
    rng=np.random.default_rng(42);weeks=te.date.dt.to_period('W').astype(str).to_numpy();blocks=[np.flatnonzero(weeks==w) for w in np.unique(weeks)]
    intervals=[]
    for name in fitted:
        vals=[];sc=preds[name+'_F1_score'].to_numpy();ys=te.fire_detected.to_numpy()
        for b in range(200):
            ix=np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))])
            if len(np.unique(ys[ix]))==2: vals.append(average_precision_score(ys[ix],sc[ix]))
        intervals.append({'model':name,'ap_lower':float(np.quantile(vals,.025)),'ap_upper':float(np.quantile(vals,.975)),'bootstrap_repeats':len(vals)})
    pd.DataFrame(intervals).to_csv(OUT/'ap_intervals.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps({'selected_by_validation_AP':selected,'selected_by_validation_accuracy':selected_acc,'data_sha256':digest(DATA/'forest_fire_grid_daily.csv'),'weather_locations':d.grid_id.nunique(),'features':len(features),'provisional':True},indent=2))
    print(pd.DataFrame(results).query("split=='test'").to_string(index=False),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    d,f=prepare()
    if not a.prepare_only:run(d,f)
