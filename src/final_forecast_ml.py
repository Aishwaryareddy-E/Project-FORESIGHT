from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT=Path('/mnt/data/Project_FORESIGHT'); PROCESSED=ROOT/'data'/'processed'
LAGS=[1,2,4,8,13]; NUM=['lag_1','lag_2','lag_4','lag_8','lag_13','roll_mean_4','roll_mean_8','roll_mean_13','promotion_days','week_sin','week_cos','month_sin','month_cos']

def feat(d):
 d=d.copy().sort_values(['sku','week_start']); g=d.groupby('sku')['units_sold']
 for l in LAGS:d[f'lag_{l}']=g.shift(l)
 sh=g.shift(1)
 for w in [4,8,13]:d[f'roll_mean_{w}']=sh.groupby(d['sku']).rolling(w).mean().reset_index(level=0,drop=True)
 d['promotion_days']=d['promotion_days'].fillna(0); d['week_sin']=np.sin(2*np.pi*d.week_of_year/52); d['week_cos']=np.cos(2*np.pi*d.week_of_year/52); d['month_sin']=np.sin(2*np.pi*d.month/12); d['month_cos']=np.cos(2*np.pi*d.month/12)
 return d

df=pd.read_csv(PROCESSED/'analysis_ready_weekly.csv',parse_dates=['week_start'])
complete=df[~df.is_partial_week].copy().sort_values(['sku','week_start'])
trainf=feat(complete).dropna(subset=NUM+['units_sold'])
pre=ColumnTransformer([('sku',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['sku']),('num','passthrough',NUM)])
model=Pipeline([('pre',pre),('model',HistGradientBoostingRegressor(max_iter=180,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=1.0,random_state=42))])
model.fit(trainf[['sku']+NUM],trainf.units_sold)
# Future calendar fields from known calendar/analysis pattern: ISO week, month, promotion_days.
last=complete.week_start.max(); future=pd.date_range(last+pd.Timedelta(weeks=2),periods=13,freq='7D')
base_hist=complete[['sku','week_start','units_sold','promotion_days','week_of_year','month']].copy()
rows=[]
for ws in future:
 # create future calendar features from week date. promotion_days unknown beyond history; assume 0 for future
 for sku in sorted(complete.sku.unique()):
  vals=base_hist[base_hist.sku==sku].set_index('week_start').units_sold
  f={f'lag_{l}':vals.get(ws-pd.Timedelta(weeks=l),np.nan) for l in LAGS}
  prev=np.array([vals.get(ws-pd.Timedelta(weeks=i),np.nan) for i in range(1,14)],float)
  for w in [4,8,13]: f[f'roll_mean_{w}']=np.nanmean(prev[:w])
  iso_week=int(ws.isocalendar().week); month=int(ws.month)
  f['promotion_days']=0; f['week_sin']=np.sin(2*np.pi*iso_week/52); f['week_cos']=np.cos(2*np.pi*iso_week/52); f['month_sin']=np.sin(2*np.pi*month/12); f['month_cos']=np.cos(2*np.pi*month/12)
  X=pd.DataFrame([{'sku':sku,**f}])
  if X[NUM].isna().any().any(): pred=vals.get(ws-pd.Timedelta(weeks=52),vals.mean())
  else: pred=float(model.predict(X[['sku']+NUM])[0])
  pred=max(0,float(pred)); rows.append({'sku':sku,'forecast_week':ws,'forecast_demand':pred,'forecast_method':'HistGradientBoostingRegressor'})
  base_hist=pd.concat([base_hist,pd.DataFrame([{'sku':sku,'week_start':ws,'units_sold':pred,'promotion_days':0,'week_of_year':iso_week,'month':month}])],ignore_index=True)
final=pd.DataFrame(rows)
latest=complete.sort_values('inventory_snapshot_date').groupby('sku',as_index=False).tail(1)[['sku','inventory_snapshot_date','on_hand_units','on_order_units','lead_time_days','safety_stock','reorder_point','inventory_value','product_name','category','subcategory','launch_date','cost_price','selling_price','gross_margin_per_unit']]
latest=latest.rename(columns={'inventory_snapshot_date':'latest_inventory_snapshot_date','on_hand_units':'latest_known_on_hand_units','on_order_units':'latest_known_on_order_units','inventory_value':'latest_known_inventory_value'})
final['reference_week']=final['forecast_week']-pd.Timedelta(weeks=52)
final=final.merge(latest,on='sku',how='left',validate='many_to_one')
final=final[['sku','product_name','category','subcategory','forecast_week','reference_week','forecast_demand','forecast_method','latest_inventory_snapshot_date','latest_known_on_hand_units','latest_known_on_order_units','lead_time_days','safety_stock','reorder_point','latest_known_inventory_value','cost_price','selling_price','gross_margin_per_unit']]
final.to_csv(PROCESSED/'demand_forecast_results.csv',index=False)
print(final.shape, final.forecast_demand.isna().sum(), final.forecast_week.min(), final.forecast_week.max())
