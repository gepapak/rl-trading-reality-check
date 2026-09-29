import pandas as pd, numpy as np, glob, json
from pathlib import Path
A=Path(__file__).resolve().parent/'results'
NAMES={'baseline_1':'Markowitz','baseline_2':'Rule-based heuristic','baseline_3':'Buy-and-hold','baseline_4':'Capped-long'}
rows=[]
for d in sorted(A.glob('V*')):
    f=d/'final_paper_baseline_summary_latest.csv'
    if not d.is_dir() or not f.exists(): continue
    s=pd.read_csv(f)
    for _,r in s.iterrows():
        rows.append(dict(variant=d.name,region=r.region,strategy=NAMES.get(r.baseline,r.baseline),fund_return=r.total_return_pct,
                         sleeve_return=r.sleeve_trading_return_pct,sleeve_sharpe=r.sleeve_trading_sharpe_ratio,sleeve_dd=r.sleeve_trading_max_drawdown_pct))
R=pd.DataFrame(rows); R.to_csv(A/'variant_summary_long.csv',index=False)
strict=R[R.variant=='V00_strict'].set_index(['region','strategy']).sleeve_return
R['sleeve_return_minus_strict_pp']=R.apply(lambda r:r.sleeve_return-strict.get((r.region,r.strategy),np.nan),axis=1)
pd.set_option('display.width',250)
piv=R.pivot_table(index=['variant'],columns=['region','strategy'],values='sleeve_return').round(2)
print('SLEEVE RETURN % by variant (rows) x region/strategy'); print(piv.to_string())
win=R[R.strategy!='Buy-and-hold'].sort_values('sleeve_return',ascending=False).groupby(['variant','region']).strategy.first().unstack()
print('\nTOP-RANKED active strategy per variant'); print(win.to_string())
