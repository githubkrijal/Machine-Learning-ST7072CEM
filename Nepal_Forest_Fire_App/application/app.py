"""Nepal forest-fire research dashboard. Run with streamlit run application/app.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import datetime as dt
import html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import ROOT, RUN, MODELS, LABELS, load_data, load_model, infer, load_metrics, boundary_rings

st.set_page_config(page_title='Nepal • Forest Fire Explorer', page_icon='🌲', layout='wide')
st.markdown('''<style>
.stApp {background:#f5f7f4; color:#18332b;}
.block-container {padding-top:2rem; max-width:1450px;}
h1,h2,h3 {color:#153c30; letter-spacing:-.025em;}
[data-testid="stMetric"] {background:white; padding:18px 20px; border:1px solid #e0e8df; border-radius:14px;}
[data-testid="stSidebar"] {background:#edf2eb;}
.hero {background:linear-gradient(115deg,#153e31,#286448);color:#fff;padding:28px 32px;border-radius:20px;margin-bottom:22px;}
.hero h1 {color:white;margin:5px 0 8px;font-size:2.25rem;}
.hero p {color:#d9eadc;margin:0;max-width:900px;}
.eyebrow {font-size:12px;letter-spacing:.15em;text-transform:uppercase;font-weight:700;}
.result {background:#fff;border:1px solid #dae4da;border-left:5px solid #2e8059;padding:20px;border-radius:12px;margin:10px 0 18px;}
.result strong {font-size:22px;display:block;margin-top:7px;}
small {color:#567064;}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def dataset():
    return load_data()

@st.cache_resource
def model(name):
    return load_model(name)

@st.cache_data
def day_predictions(date, name):
    rows = dataset().loc[dataset().date.eq(pd.Timestamp(date))]
    return infer(model(name), rows).reset_index(drop=True)

@st.cache_data
def rings():
    return boundary_rings()


def styled(fig, height=400):
    fig.update_layout(template='plotly_white', height=height, paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=15,r=15,t=35,b=20),
                      font=dict(family='Arial',color='#244739'), legend_title_text='')
    return fig

try:
    data = dataset()
    metrics = load_metrics()
    for name in MODELS:
        model(name)
except (FileNotFoundError, ValueError) as exc:
    st.error(f'The full research assets could not be loaded: {exc}')
    st.info('Run this app from the original project containing research_full/data and research_full/results. See application/README.md.')
    st.stop()

with st.sidebar:
    st.markdown('### 🌲 Forest Fire Explorer')
    st.caption('NEPAL · HISTORICAL RESEARCH DEMO')
    page = st.radio('Explore', ['Prediction explorer', 'Model performance', 'About the study'], label_visibility='collapsed')
    st.divider()
    st.markdown('**What is predicted?**')
    st.write('Whether a satellite fire detection will be recorded in a forest-containing grid cell on the next day.')
    st.caption('Historical test period: 1 January–18 June 2016. This is not a live warning service.')
    st.markdown('[Project on GitHub](https://github.com/githubkrijal/Machine-Learning-ST7072CEM)')

st.markdown('''<div class="hero"><div class="eyebrow">Nepal / Machine learning research</div>
<h1>Forest Fire Explorer</h1><p>Explore next-day detection predictions across Nepal’s forest grid. Compare the models, inspect their inputs and check predictions against historical records.</p></div>''', unsafe_allow_html=True)

if page == 'Prediction explorer':
    a,b,c = st.columns([1.1,1.3,1.6])
    date = a.date_input('Target day', value=dt.date(2016,4,15), min_value=data.date.min().date(), max_value=data.date.max().date())
    name = b.selectbox('Prediction model', MODELS, format_func=lambda n: LABELS[n])
    day = day_predictions(date, name)
    locations = day.set_index('grid_id')
    if 'pending_cell' in st.session_state:
        st.session_state['cell'] = st.session_state.pop('pending_cell')
    grid = c.selectbox('Forest grid cell', day.grid_id.tolist(),
                      format_func=lambda g: f'{g} · {locations.loc[g,"latitude"]:.2f}°N, {locations.loc[g,"longitude"]:.2f}°E', key='cell')
    left,right = st.columns([1.8,1], gap='large')
    with left:
        st.subheader('Nepal forest grid')
        st.caption('Select a dot, or use the grid-cell selector. Colours show relative score rank for this day.')
        show = st.toggle('Show recorded detections on the map', value=False)
        fig = go.Figure()
        for ring in rings():
            xy=np.asarray(ring)
            fig.add_trace(go.Scatter(x=xy[:,0],y=xy[:,1],mode='lines',line=dict(color='#6b8173',width=1),hoverinfo='skip',showlegend=False))
        fig.add_trace(go.Scatter(x=day.longitude,y=day.latitude,mode='markers',customdata=day[['grid_id','rank_percentile','prediction']],
            marker=dict(size=10,color=day.rank_percentile,colorscale=[[0,'#a9d9d2'],[.45,'#369e8d'],[.75,'#f4b942'],[1,'#dc5d3b']],cmin=0,cmax=100,
                colorbar=dict(title='Score rank',ticksuffix='%'),line=dict(width=.6,color='white')),
            hovertemplate='%{customdata[0]}<br>%{y:.2f}°N, %{x:.2f}°E<br>Daily score rank: %{customdata[1]:.1f}%<extra></extra>',showlegend=False))
        chosen=day.loc[day.grid_id.eq(grid)].iloc[0]
        fig.add_trace(go.Scatter(x=[chosen.longitude],y=[chosen.latitude],mode='markers',marker=dict(size=18,color='rgba(0,0,0,0)',line=dict(color='#112e27',width=3)),hoverinfo='skip',showlegend=False))
        if show:
            pos=day.loc[day.fire_detected.eq(1)]
            fig.add_trace(go.Scatter(x=pos.longitude,y=pos.latitude,mode='markers',marker=dict(symbol='x',size=12,color='#142c25'),name='Recorded detection',hoverinfo='skip'))
        fig.update_layout(dragmode='select',clickmode='event+select')
        fig.update_xaxes(title='Longitude (°E)',range=[79.6,88.5],showgrid=False)
        fig.update_yaxes(title='Latitude (°N)',range=[26.0,30.9],scaleanchor='x',scaleratio=1.12,showgrid=False)
        event=st.plotly_chart(styled(fig,470),width='stretch',key=f'map_{date}_{name}_{grid}',on_select='rerun',selection_mode='points')
        points=event.selection.points if event else []
        clicked=next((p['customdata'][0] for p in points if p.get('customdata')),None)
        if clicked and clicked != grid:
            # Apply before selectbox creation on the next rerun.
            st.session_state['pending_cell']=clicked
            st.rerun()
        st.caption('Grid centres represent forest-containing cells; a centre can fall outside Nepal while the cell intersects forest inside the boundary. Scores are not probabilities.')
    with right:
        st.subheader('Run a prediction')
        previous=date-dt.timedelta(days=1)
        st.write(f'Use weather through **{previous:%d %B %Y}** to predict the recorded outcome for **{date:%d %B %Y}**.')
        st.caption('Linear SVM is the default because it led validation average precision. All models use their saved validation-F1 thresholds.')
        if st.button('Predict next-day detection',type='primary',width='stretch'):
            row=data.loc[data.date.eq(pd.Timestamp(date)) & data.grid_id.eq(grid)]
            result=infer(model(name),row).iloc[0]
            st.session_state['last_prediction']={'date':str(date),'grid':grid,'model':name,'score':float(result.score),'prediction':int(result.prediction)}
        saved=st.session_state.get('last_prediction')
        if saved and (saved['date'],saved['grid'],saved['model']) == (str(date),grid,name):
            text='Detection predicted' if saved['prediction'] else 'No detection predicted'
            st.markdown(f'<div class="result"><small>{html.escape(LABELS[name])} · {html.escape(grid)}</small><strong>{text}</strong><p>Target: {date:%d %B %Y}</p></div>',unsafe_allow_html=True)
            st.metric('Model score',f'{saved["score"]:.4f}')
            st.caption(f'Detection threshold: {model(name)["threshold"]:.4f}. Scores are model-specific and should not be compared numerically across models.')
            if st.checkbox('Reveal recorded outcome',key=f'reveal_{date}_{grid}_{name}'):
                actual=int(chosen.fire_detected)
                st.write('**Recorded outcome:** '+('Detection recorded' if actual else 'No detection recorded'))
                outcome='Correct detection' if actual and saved['prediction'] else 'Missed detection' if actual else 'False alarm' if saved['prediction'] else 'Correct no-detection prediction'
                st.info(outcome)
        else:
            st.info('Choose a cell and click Predict to run the saved model on its 21 prepared inputs.')
    st.divider()
    tab1,tab2,tab3=st.tabs(['Weather & landscape','Compare this prediction','Download daily predictions'])
    row=data.loc[data.date.eq(pd.Timestamp(date)) & data.grid_id.eq(grid)]
    with tab1:
        r=row.iloc[0]
        cols=st.columns(4)
        for col,label,value in zip(cols,['Previous-day maximum temperature','Previous-day minimum humidity','Previous 7-day rainfall','Forest cover in cell'],[f'{r.temperature_2m_max_previous_day:.1f} °C',f'{r.relative_humidity_2m_min_previous_day:.0f}%',f'{r.precipitation_previous_7d_mm:.1f} mm',f'{r.forest_fraction_2011:.1%}']):
            col.metric(label,value)
        st.caption(f'Mean forest elevation: {r.elevation_mean_m:,.0f} m · slope: {r.slope_mean_degrees:.1f}° · previous-day mean wind: {r.wind_speed_10m_mean_previous_day:.2f} m/s')
        with st.expander('See all 21 model inputs'):
            st.dataframe(row[model(name)['features']].T.rename(columns={row.index[0]:'Value'}),width='stretch')
    with tab2:
        comparison=[]
        for n in MODELS:
            prediction=infer(model(n),row).iloc[0]
            comparison.append({'Model':LABELS[n],'Prediction':'Detection' if prediction.prediction else 'No detection','Score':prediction.score,'Threshold':model(n)['threshold']})
        st.dataframe(pd.DataFrame(comparison),hide_index=True,width='stretch')
        st.caption('These are four real model inferences for the same inputs. Raw scores use different scales; compare the decisions, not the score magnitudes.')
    with tab3:
        export=day.copy();export['model']=name;export['threshold']=model(name)['threshold']
        st.download_button('Download all 238 cells for this day',export.to_csv(index=False),file_name=f'nepal_{date}_{name.replace(" ","_")}.csv',mime='text/csv')
        st.caption('Includes predictions and historical recorded labels. Daily percentile ranks are relative scores, not calibrated fire probabilities.')

elif page == 'Model performance':
    st.subheader('How well did the models perform?')
    st.caption('Held-out chronological test: 1 January–18 June 2016 · 40,460 grid-days · 1,571 recorded positive grid-days. Prior exploratory exposure limits test independence.')
    te=metrics.loc[metrics.split.eq('test')].copy()
    va=metrics.loc[metrics.split.eq('validation')]
    cols=st.columns(3)
    cols[0].metric('Validation-selected model','Linear SVM')
    cols[1].metric('Highest test balanced accuracy',f'{te.balanced_accuracy.max():.2%}')
    cols[2].metric('Highest test F1',f'{te.f1.max():.3f}')
    metric=st.selectbox('Compare measured performance',['balanced_accuracy','recall','precision','f1','average_precision','roc_auc','accuracy'],format_func=lambda x:{'accuracy':'Ordinary accuracy','balanced_accuracy':'Balanced accuracy','average_precision':'Average precision','roc_auc':'ROC-AUC'}.get(x,x.replace('_',' ').title()))
    te['Model']=te.model.map(LABELS)
    fig=px.bar(te,x='Model',y=metric,color='Model',color_discrete_sequence=['#258b83','#8871b1','#e7a63e','#dd6947'],text_auto='.3f')
    fig.update_yaxes(range=[0,1],title=metric.replace('_',' ').title())
    st.plotly_chart(styled(fig,350),width='stretch')
    st.caption('Ordinary accuracy is high because most grid-days have no recorded detection. Balanced accuracy weights both classes equally; recall measures the fraction of recorded positive days recovered.')
    a,b=st.columns(2)
    n=a.selectbox('Inspect model',MODELS,format_func=lambda x:LABELS[x])
    m=te.loc[te.model.eq(n)].iloc[0]
    matrix=np.array([[m.tn,m.fp],[m.fn,m.tp]],dtype=int)
    with a:
        fig=px.imshow(matrix,x=['No detection','Detection'],y=['No detection','Detection'],text_auto=True,color_continuous_scale='Teal',labels=dict(x='Predicted',y='Recorded'))
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(styled(fig,350),width='stretch')
    with b:
        st.markdown('#### What these numbers mean')
        st.write(f'**{int(m.tp):,} correct detections** — a detection was predicted and recorded.')
        st.write(f'**{int(m.fn):,} missed detections** — no detection was predicted, but one was recorded.')
        st.write(f'**{int(m.fp):,} false alarms** — a detection was predicted, but none was recorded.')
        st.write(f'**{int(m.tn):,} correct no-detection predictions.**')
        st.caption('Counts refer to location–date combinations, not individual fires.')
    with st.expander('All test metrics and validation selection evidence'):
        st.dataframe(te[['Model','accuracy','balanced_accuracy','precision','recall','f1','average_precision','roc_auc']],hide_index=True)
        st.dataframe(va[['model','average_precision','threshold']],hide_index=True)

else:
    st.subheader('From research data to a next-day prediction')
    st.write('The study combines ICIMOD’s MODIS-based Active Fire Incidents in Nepal (February 2012–June 2016) with daily ERA5 weather accessed through Open-Meteo, a 2011 forest mask, SRTM terrain and a Nepal boundary.')
    cols=st.columns(3)
    for c,label,value in zip(cols,['Forest-containing grid cells','Prepared predictors','Research grid-days'],['238','21','380,562']):c.metric(label,value)
    st.markdown('''1. **Prepare:** check dates and duplicates, restrict detections to Nepal’s forest mask, and aggregate to cell-days.
2. **Use past information:** target-day features use weather through the preceding day, alongside fixed terrain and forest context.
3. **Fit and select:** train on 2012–2014; select settings by 2015 average precision and thresholds by validation F1.
4. **Evaluate:** compare predictions with recorded outcomes in January–June 2016.
5. **Demonstrate:** this app loads the fitted models and performs inference without retraining or downloading weather.''')
    st.info('The demo is restricted to the historical test period. It does not predict current conditions, ignition time, burned area or damage. No recorded detection does not establish that no fire occurred.')
    st.markdown('[ICIMOD Regional Database System](https://rds.icimod.org/) · [Open-Meteo](https://open-meteo.com/) · [OpenTopography](https://opentopography.org/) · [geoBoundaries](https://www.geoboundaries.org/)')

st.divider()
st.caption('Research demonstration • Next-day recorded detections, not a live fire warning • Predictions use the saved full-data experiment')
