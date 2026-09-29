"""Thesis-era price construction: linear interpolation of hourly day-ahead prices onto the 10-minute grid.
Each 10-min value between hour h and h+1 depends on the price of hour h+1 (look-ahead). All other columns unchanged."""
import pandas as pd, numpy as np
from pathlib import Path
P5=Path(__file__).resolve().parents[1]/'Prototype5'/'evaluation_dataset_ffill'
OUT=Path(__file__).resolve().parent/'data'
for name in ['unseendata.csv','unseendata_v2.csv']:
    d=pd.read_csv(P5/name,parse_dates=['timestamp'])
    hourly=d.set_index('timestamp').price.resample('h').first()            # hourly DA value
    idx=pd.DatetimeIndex(d.timestamp); interp=hourly.reindex(idx.union(hourly.index)).interpolate(method='time').reindex(idx)
    d2=d.copy(); d2['price']=interp.values
    d2.to_csv(OUT/name.replace('.csv','_interpolated.csv'),index=False)
    print(name,'mean |interp - ffill|',np.abs(d2.price-d.price).mean().round(3),'changed share',np.mean(np.abs(d2.price-d.price)>1e-9).round(3))
