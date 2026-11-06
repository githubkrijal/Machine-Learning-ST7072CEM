"""Validate real inference parity and the user-facing Streamlit workflow."""
from pathlib import Path
import numpy as np
import pandas as pd
from core import ROOT, RUN, MODELS, load_data, load_model, infer

def main():
    data=load_data()
    saved=pd.read_csv(RUN/'results/test_predictions.csv',parse_dates=['date'])
    for name in MODELS:
        actual=infer(load_model(name),data)
        joined=actual.merge(saved,on=['grid_id','date'],validate='one_to_one')
        assert len(joined)==40460
        np.testing.assert_allclose(joined.score,joined[name+'_F1_score'],rtol=1e-10,atol=1e-10)
        np.testing.assert_array_equal(joined.prediction,joined[name+'_F1_prediction'])
        print(f'{name}: all 40,460 saved scores and decisions match.')
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(Path(__file__).parent/'app.py'),default_timeout=60).run()
    assert not app.exception, app.exception
    app.button[0].click().run()
    assert not app.exception, app.exception
    assert any('Model score' in m.label for m in app.metric)
    app.checkbox[0].check().run()
    assert not app.exception, app.exception
    app.radio[0].set_value('Model performance').run()
    assert not app.exception, app.exception
    app.selectbox[0].set_value('accuracy').run()
    app.selectbox[1].set_value('Gradient boosting').run()
    assert not app.exception, app.exception
    app.radio[0].set_value('About the study').run()
    assert not app.exception, app.exception
    print('Prediction, outcome reveal, metric/model selection and all three pages passed.')

if __name__=='__main__':main()
