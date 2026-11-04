from research_experiments import *
d=pd.read_csv(DATA/"forest_fire_grid_daily.csv",parse_dates=["date"])
features=json.loads((DATA/"feature_columns.json").read_text())
tr=d[d.split.eq("train")];te=d[d.split.eq("test")]
preds=pd.read_csv(OUT/"test_predictions.csv")
met=pd.read_csv(OUT/"metrics.csv")
fitted={}
for n in candidates():
 m=joblib.load(OUT/(n.replace(" ","_")+"_F1.joblib"))["model"]
 fitted[n]={"F1":(None,m)}
selected=met.query("split=='validation' and policy=='F1'").sort_values("average_precision",ascending=False).iloc[0].model
selected_acc=met.query("split=='validation' and policy=='accuracy'").sort_values("accuracy",ascending=False,kind="stable").iloc[0].model
results=met.to_dict("records")
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