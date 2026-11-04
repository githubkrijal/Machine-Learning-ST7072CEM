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
