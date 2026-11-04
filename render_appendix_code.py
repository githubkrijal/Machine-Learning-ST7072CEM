"""Render faithful source excerpts as black-on-white appendix images."""
from pathlib import Path
import textwrap,json,os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/nepal-research-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'research_full/overleaf/images'
EXCERPTS=[
 ('code_cleaning','prepare_assignment.py',106,116,'Date conversion and repeated-detection removal.'),
 ('code_weather_features','prepare_assignment.py',137,145,'Past-only weather features and rolling summaries.'),
 ('code_linear_models','research_experiments.py',78,80,'Scaled Logistic Regression and Linear SVM candidates.'),
 ('code_tree_models','research_experiments.py',81,82,'Random Forest and Gradient Boosting settings.'),
 ('code_preprocessing','reanalysis_2026.py',23,25,'Controlled PCA and SMOTE pipeline variants.'),
 ('code_model_selection','research_experiments.py',90,100,'Training candidates and selecting validation settings.'),
 ('code_evaluation','research_experiments.py',74,76,'Confusion counts and performance-metric calculation.')]
records=[]
for name,filename,start,end,caption in EXCERPTS:
 original=(ROOT/filename).read_text().splitlines();lines=original[start-1:end];shown=[]
 for number,line in enumerate(lines,start):
  wrapped=textwrap.wrap(line.expandtabs(4),width=48,subsequent_indent='    ',replace_whitespace=False,drop_whitespace=True,break_long_words=True,break_on_hyphens=False) or ['']
  shown.extend([(f'{number:3d}' if j==0 else '   ')+'  '+part for j,part in enumerate(wrapped)])
 height=.62+.205*len(shown)
 fig=plt.figure(figsize=(5.5,height),facecolor='white');ax=fig.add_axes([0,0,1,1]);ax.axis('off')
 ax.text(.025,1-.12/height,f'{filename}  |  lines {start}-{end}',fontfamily='DejaVu Sans Mono',fontsize=10,color='black',va='top',transform=ax.transAxes)
 ax.plot([.02,.98],[1-.38/height]*2,color='#a0a0a0',lw=.5,transform=ax.transAxes)
 ax.text(.025,1-.48/height,'\n'.join(shown),fontfamily='DejaVu Sans Mono',fontsize=11.5,color='black',va='top',linespacing=1.28,transform=ax.transAxes)
 fig.savefig(OUT/(name+'.png'),dpi=240,facecolor='white');plt.close(fig)
 records.append(dict(image=name,source=filename,start=start,end=end,caption=caption))
(ROOT/'research_full/code_excerpt_manifest.json').write_text(json.dumps(records,indent=2))
print('Rendered',len(records),'source-code images from actual project files.')
