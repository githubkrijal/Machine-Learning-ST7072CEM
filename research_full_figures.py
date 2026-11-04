"""Publication figures from measured frozen-run outputs."""
from reanalysis_2026 import ROOT,DATA,OUT,BASE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.metrics import precision_recall_curve,roc_curve
import json,numpy as np,pandas as pd
FIG=BASE/'overleaf/images';FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white','pdf.fonttype':42,'axes.labelcolor':'#263244','text.color':'#263244','axes.grid':True,'grid.alpha':.17})
COLORS={'Logistic regression':'#0072B2','Linear SVM':'#D55E00','Random forest':'#009E73','Gradient boosting':'#8E44AD'}
SHORT={'Logistic regression':'Logistic','Linear SVM':'Linear SVM','Random forest':'Forest','Gradient boosting':'Boosting'}
d=pd.read_csv(DATA/'forest_fire_grid_daily.csv',parse_dates=['date']);m=pd.read_csv(OUT/'metrics.csv');p=pd.read_csv(OUT/'test_predictions.csv',parse_dates=['date']);loc=pd.read_csv(ROOT/'data/study_grid/nepal_forest_weather_locations.csv');summary=json.loads((OUT/'summary.json').read_text());chosen=summary['selected_by_validation_AP']
def save(fig,name):
 fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');fig.savefig(FIG/(name+'.png'),dpi=190,bbox_inches='tight');plt.close(fig)
def boundary(ax):
 g=json.loads((ROOT/'data/sources/geoBoundaries-NPL-ADM0.geojson').read_text())['features'][0]['geometry'];rings=g['coordinates'] if g['type']=='Polygon' else [r for pp in g['coordinates'] for r in pp]
 for ring in rings:
  a=np.array(ring);ax.plot(a[:,0],a[:,1],c='#263244',lw=.6)
 ax.set(xlabel='Longitude (°E)',ylabel='Latitude (°N)');ax.set_aspect(1/np.cos(np.radians(28)))
fig,axes=plt.subplots(2,1,figsize=(4.4,5.6));boundary(axes[0]);counts=d.groupby('grid_id').fire_detected.sum();l=loc.merge(counts.rename('count'),on='grid_id',how='inner');sc=axes[0].scatter(l['longitude'],l['latitude'],c=l['count'],cmap='plasma',s=22);fig.colorbar(sc,ax=axes[0],shrink=.65,label='Recorded positive grid-days');axes[0].set_title(f'A. Forest-grid cells (n={len(l)})')
raw=pd.read_csv(ROOT/'data/sources/Active_Fire_Incidents_in_Nepal.csv');yr=pd.to_datetime(raw.ACQ_DATE).dt.year.value_counts().sort_index();bars=axes[1].bar(yr.index.astype(str),yr.values,color=['#0072B2','#009E73','#E69F00','#CC79A7','#D55E00']);axes[1].bar_label(bars,padding=3,fontsize=9);axes[1].set(ylim=(0,6500),ylabel='Source detection rows',title='B. Annual source counts');axes[1].text(.03,.91,'2012 and 2016 cover partial years',transform=axes[1].transAxes,fontsize=8);fig.tight_layout();save(fig,'coverage_counts')
tr=d[d.split.eq('train')];fig,axes=plt.subplots(3,1,figsize=(4.4,6.5));v=[('temperature_2m_mean_previous_day','Previous-day mean temperature (°C)','#D55E00'),('precipitation_previous_7d_mm','Previous 7-day precipitation (mm)','#0072B2'),('dry_days_previous','Previous dry-day run (cap 30)','#009E73')]
for ax,(col,label,c) in zip(axes,v):ax.hist(tr[col],bins=35,color=c,alpha=.85,edgecolor='white',linewidth=.3);ax.set(xlabel=label,ylabel='Training grid-days')
fig.tight_layout();save(fig,'distributions')
cols=['temperature_2m_mean_previous_day','temperature_2m_max_previous_day','relative_humidity_2m_mean_previous_day','relative_humidity_2m_min_previous_day','wind_speed_10m_mean_previous_day','wind_speed_10m_max_previous_day','precipitation_previous_7d_mm','precipitation_previous_30d_mm','dry_days_previous','elevation_mean_m','slope_mean_degrees','forest_fraction_2011'];labels=['Mean T','Max T','Mean RH','Min RH','Mean wind','Max wind','Rain 7d','Rain 30d','Dry days','Elevation','Slope','Forest fraction'];corr=tr[cols].corr(method='spearman');fig,ax=plt.subplots(figsize=(4.8,4.5));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r');ax.set_xticks(range(len(cols)),labels,rotation=60,ha='right',fontsize=8);ax.set_yticks(range(len(cols)),labels,fontsize=8);ax.grid(False);fig.colorbar(im,ax=ax,shrink=.8,label='Spearman correlation');ax.set_title('Training predictors: selected correlations');fig.tight_layout();save(fig,'correlation')
fig,axes=plt.subplots(2,1,figsize=(4.4,6));
for name,c in COLORS.items():
 s=p[name+'_F1_score'];pr,re,_=precision_recall_curve(p.fire_detected,s);fpr,tpr,_=roc_curve(p.fire_detected,s);r=m[(m.model==name)&(m.split=='test')&(m.policy=='F1')].iloc[0];axes[0].plot(re,pr,c=c,label=f'{SHORT[name]} (AP {r.average_precision:.3f})');axes[1].plot(fpr,tpr,c=c,label=f'{SHORT[name]} (AUC {r.roc_auc:.3f})')
axes[0].axhline(p.fire_detected.mean(),c='#777',ls='--',label='Prevalence');axes[1].plot([0,1],[0,1],c='#777',ls='--');axes[0].set(xlabel='Recall',ylabel='Precision',title='A. 2016 precision–recall');axes[1].set(xlabel='False-positive rate',ylabel='True-positive rate',title='B. 2016 ROC');
for ax in axes:ax.legend(fontsize=8);ax.set_xlim(0,1);ax.set_ylim(0,1.02)
fig.tight_layout();save(fig,'curves')
fig,axes=plt.subplots(2,1,figsize=(4.4,5.5));names=list(COLORS);x=np.arange(4)
r=m[(m.split=='test')&(m.policy=='F1')].set_index('model').loc[names]
for ax,metric,title in [(axes[0],'balanced_accuracy','A. Balanced accuracy'),(axes[1],'recall','B. Recorded fire days detected')]:
 bars=ax.bar(x,100*r[metric],color=[COLORS[n] for n in names]);ax.bar_label(bars,fmt='%.2f',padding=4,fontsize=9);ax.set_xticks(x,[SHORT[n] for n in names]);ax.set(ylim=(0,100),ylabel='Balanced accuracy (%)' if metric=='balanced_accuracy' else 'Recall (%)',title=title)
fig.tight_layout();save(fig,'accuracy_recall')
fig,axes=plt.subplots(2,1,figsize=(4.4,6.2));
for ax,name,cmap in zip(axes,[chosen,'Gradient boosting'],['Oranges','Purples']):
 r=m[(m.model==name)&(m.split=='test')&(m.policy=='F1')].iloc[0];a=np.array([[r.tn,r.fp],[r.fn,r.tp]],int);ax.imshow(a,cmap=cmap,norm=LogNorm(vmin=1,vmax=a.max()));
 for i in range(2):
  for j in range(2):ax.text(j,i,f'{a[i,j]:,}',ha='center',va='center',color='white' if a[i,j]>10000 else '#17202A',fontsize=13,fontweight='bold')
 ax.grid(False);ax.set_xticks([0,1],['No detection','Detection']);ax.set_yticks([0,1],['No detection','Detection']);ax.set(xlabel='Predicted',ylabel='Recorded',title=name)
fig.tight_layout();save(fig,'confusions')
imp=pd.read_csv(OUT/'importance.csv').head(9).iloc[::-1];labels={'day_of_year_sin':'Season: sine','day_of_year_cos':'Season: cosine','forest_fraction_2011':'Forest fraction','elevation_mean_m':'Mean elevation','slope_mean_degrees':'Mean slope','longitude':'Longitude','latitude':'Latitude','dry_days_previous':'Dry-day run','relative_humidity_2m_min_previous_day':'Previous minimum RH','temperature_2m_max_previous_day':'Previous maximum temperature','temperature_mean_previous_7d':'7-day mean temperature','aspect_northness_mean':'Northness','humidity_mean_previous_7d':'7-day mean humidity','relative_humidity_2m_mean_previous_day':'Previous mean RH'};fig,ax=plt.subplots(figsize=(4.8,3.9));ax.barh([labels.get(x,x.replace('_',' ')) for x in imp.feature],imp['mean'],xerr=imp['std'],color=plt.cm.viridis(np.linspace(.2,.85,len(imp))));ax.axvline(0,c='#444',lw=.6);ax.set(xlabel='Decrease in validation average precision',title=f'{chosen}: permutation importance');fig.tight_layout();save(fig,'importance')
fig,axes=plt.subplots(2,1,figsize=(4.4,5.6));ab=pd.read_csv(OUT/'ablation.csv');axes[0].bar(['Spatial + season','Plus weather'],ab.validation_ap,color=['#56B4E9','#CC79A7']);axes[0].set(ylabel='2015 average precision',title='A. Weather-feature comparison');sp=pd.read_csv(OUT/'spatial_diagnostic.csv');axes[1].bar([SHORT[n] for n in sp.model],sp.average_precision,color=[COLORS[n] for n in sp.model]);axes[1].axhline(sp.positive_days.iloc[0]/sp.evaluation_rows.iloc[0],c='#777',ls='--');axes[1].set(ylabel='2014 average precision',title='B. Geographic transfer diagnostic');fig.tight_layout();save(fig,'diagnostics')
# Illustrative SVM margins are not probabilities; show explicit within-day percentile rank.
sub=p[p.date.eq('2016-04-15')];fig,ax=plt.subplots(figsize=(4.8,3.3));boundary(ax);sc=ax.scatter(sub.longitude,sub.latitude,c=sub[chosen+'_F1_score'].rank(pct=True),cmap='plasma',vmin=0,vmax=1,s=32);pos=sub[sub.fire_detected.eq(1)];ax.scatter(pos.longitude,pos.latitude,facecolors='none',edgecolors='black',s=65,lw=1,label='Recorded detection');fig.colorbar(sc,ax=ax,label='Within-day score percentile');ax.legend(fontsize=8,loc='upper right');ax.set_title('Linear SVM ranking: 15 April 2016');fig.tight_layout();save(fig,'historical_map')
print('Figures saved:',len(list(FIG.glob('*.pdf'))))

# Additional evidence for preprocessing decisions.
ab=pd.read_csv(OUT/'preprocessing_ablation.csv');fig,axes=plt.subplots(2,1,figsize=(4.8,5.7));order=ab.variant.drop_duplicates();cs=plt.cm.viridis(np.linspace(.1,.9,len(order)))
for ax,split,col,title in [(axes[0],'validation','average_precision','A. Validation ranking'),(axes[1],'test','balanced_accuracy','B. Test balanced accuracy')]:
 a=ab[ab.split.eq(split)].set_index('variant').loc[order];bars=ax.barh(range(len(order)),a[col]*(100 if col=='balanced_accuracy' else 1),color=cs);ax.set_yticks(range(len(order)),[v.replace(' Logistic','').replace('Weighted','Class weights') for v in order],fontsize=8);ax.set(xlabel='Balanced accuracy (%)' if col=='balanced_accuracy' else 'Average precision',title=title);ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8);ax.set_xlim(0,100 if col=='balanced_accuracy' else .075)
fig.tight_layout();save(fig,'preprocessing')
pv=pd.read_csv(OUT/'pca_variance.csv');fig,axes=plt.subplots(2,1,figsize=(4.4,5.2));axes[0].bar(pv.component,100*pv.variance,color=plt.cm.plasma(np.linspace(.05,.85,len(pv))));axes[1].plot(pv.component,100*pv.cumulative,color='#0072B2',marker='o',ms=3);axes[1].axhline(95,color='#D55E00',ls='--');axes[1].axvline(11,color='#009E73',ls=':');axes[0].set(xlabel='Principal component',ylabel='Explained variance (%)',title='A. Training variance by component');axes[1].set(xlabel='Number of components',ylabel='Cumulative variance (%)',title='B. Eleven PCs retain 95.09%');fig.tight_layout();save(fig,'pca')
fig,axes=plt.subplots(2,1,figsize=(4.4,5.2));parts=['train','validation','test'];a=d.groupby('split').fire_detected.agg(['size','sum']).loc[parts];axes[0].bar(parts,a['size']-a['sum'],label='No recorded detection',color='#56B4E9');axes[0].bar(parts,a['sum'],bottom=a['size']-a['sum'],label='Recorded detection',color='#D55E00');axes[0].set(ylabel='Grid-days',title='A. Observed class counts');axes[0].legend(fontsize=8);bars=axes[1].bar(parts,100*a['sum']/a['size'],color=['#0072B2','#E69F00','#CC79A7']);axes[1].bar_label(bars,fmt='%.3f%%',padding=3);axes[1].set(ylim=(0,4.5),ylabel='Recorded positive days (%)',title='B. Prevalence changes across periods');fig.tight_layout();save(fig,'imbalance')
fig,ax=plt.subplots(figsize=(4.8,4.1));ax.axis('off');labels=['13,781 source detections','224 repeated keys removed','1 point outside boundary removed','3,298 non-forest points removed','10,258 retained forest detections','238 weather cells x 1,827 dates','380,562 model grid-days; 4,399 positive']
for i,label in enumerate(labels):
 y=1-i*.145;ax.text(.5,y,label,ha='center',va='center',transform=ax.transAxes,bbox=dict(boxstyle='round,pad=.35',fc=plt.cm.viridis(.12+i*.12),ec='none'),color='white' if i<5 else '#17202a',fontsize=10)
 if i<len(labels)-1:ax.annotate('',xy=(.5,y-.105),xytext=(.5,y-.04),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#334155'))
fig.tight_layout();save(fig,'preparation_flow')
