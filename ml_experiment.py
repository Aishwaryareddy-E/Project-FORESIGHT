from pathlib import Path
import numpy as np, pandas as pd, json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT=Path(__file__).resolve().parents[1]; PROCESSED=ROOT/'data'/'processed'; REPORTS=ROOT/'reports'
NUM=['lag_1','lag_2','lag_4','lag_8','lag_13','roll_mean_4','roll_mean_8','roll_mean_13','promotion_days','week_sin','week_cos','month_sin','month_cos']
LAGS=[1,2,4,8,13]

def wape(y,p): return 100*np.abs(np.asarray(y)-np.asarray(p)).sum()/np.abs(np.asarray(y)).sum()
def add_features(df):
    d=df.copy().sort_values(['sku','week_start'])
    g=d.groupby('sku')['units_sold']
    for l in LAGS:d[f'lag_{l}']=g.shift(l)
    shifted=g.shift(1)
    for w in [4,8,13]: d[f'roll_mean_{w}']=shifted.groupby(d['sku']).rolling(w).mean().reset_index(level=0,drop=True)
    d['promotion_days']=d['promotion_days'].fillna(0)
    d['week_sin']=np.sin(2*np.pi*d['week_of_year']/52); d['week_cos']=np.cos(2*np.pi*d['week_of_year']/52)
    d['month_sin']=np.sin(2*np.pi*d['month']/12); d['month_cos']=np.cos(2*np.pi*d['month']/12)
    return d

def model():
    pre=ColumnTransformer([('sku',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['sku']),('num','passthrough',NUM)])
    reg=HistGradientBoostingRegressor(max_iter=180,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=1.0,random_state=42)
    return Pipeline([('pre',pre),('model',reg)])

def predict_recursive(train,test):
    trainf=add_features(train).dropna(subset=NUM+['units_sold']).copy(); m=model(); m.fit(trainf[['sku']+NUM],trainf['units_sold'])
    hist=train[['sku','week_start','units_sold','promotion_days','week_of_year','month']].copy(); rows=[]
    for ws in sorted(test.week_start.unique()):
      for r in test[test.week_start==ws].sort_values('sku').itertuples():
        vals=hist[hist.sku==r.sku].set_index('week_start')['units_sold']
        f={f'lag_{l}':vals.get(ws-pd.Timedelta(weeks=l),np.nan) for l in LAGS}
        prev=np.array([vals.get(ws-pd.Timedelta(weeks=i),np.nan) for i in range(1,14)],float)
        for w in [4,8,13]: f[f'roll_mean_{w}']=np.nanmean(prev[:w])
        f['promotion_days']=0 if pd.isna(r.promotion_days) else r.promotion_days
        f['week_sin']=np.sin(2*np.pi*r.week_of_year/52); f['week_cos']=np.cos(2*np.pi*r.week_of_year/52)
        f['month_sin']=np.sin(2*np.pi*r.month/12); f['month_cos']=np.cos(2*np.pi*r.month/12)
        X=pd.DataFrame([{'sku':r.sku,**f}])
        pred=vals.get(ws-pd.Timedelta(weeks=52),vals.mean()) if X[NUM].isna().any().any() else float(m.predict(X[['sku']+NUM])[0])
        pred=max(0,float(pred)); rows.append({'sku':r.sku,'week_start':ws,'ml_forecast':pred,'actual':r.units_sold})
        hist=pd.concat([hist,pd.DataFrame([{'sku':r.sku,'week_start':ws,'units_sold':pred,'promotion_days':r.promotion_days,'week_of_year':r.week_of_year,'month':r.month}])],ignore_index=True)
    return pd.DataFrame(rows)

def main():
    df=pd.read_csv(PROCESSED/'analysis_ready_weekly.csv',parse_dates=['week_start'])
    df=df[~df.is_partial_week].copy()
    starts=pd.to_datetime(['2025-01-06','2025-04-07','2025-07-07','2025-09-29'])
    out=[]
    for i,start in enumerate(starts,1):
      end=start+pd.Timedelta(weeks=12); tr=df[df.week_start<start].copy(); te=df[(df.week_start>=start)&(df.week_start<=end)].copy()
      pred=predict_recursive(tr,te); out.append({'fold':i,'test_start':start.date().isoformat(),'test_end':end.date().isoformat(),'ml_wape':wape(pred.actual,pred.ml_forecast)})
    res=pd.DataFrame(out); res.to_csv(REPORTS/'ml_backtest_results.csv',index=False)
    summary={'folds':res.to_dict('records'),'average_ml_wape':float(res.ml_wape.mean()),'seasonal_naive_rolling_average_wape':10.840223,'beats_seasonal_naive':bool(res.ml_wape.mean()<10.840223)}
    (REPORTS/'ml_experiment_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
