"""Decision replay: hold each agent's logged decisions fixed and re-price them under evaluation/accounting shortcuts.
Validates the replay against the engine's booked horizon-settlement P&L first."""
import pandas as pd, numpy as np, glob, re
from pathlib import Path
P5=Path(__file__).resolve().parents[1]/'Prototype5'; OUT=Path(__file__).resolve().parent/'results'
SLEEVE_DKK=None
SETTLE={'2025':P5/'evaluation_dataset_ffill/unseendata_settlement_real_v2.csv','2025_v2':P5/'evaluation_dataset_ffill/unseendata_v2_settlement_real_v2.csv'}
cols=['timestep','decision_step','price_current','position_signed','liquidity_requested_volume_mwh','liquidity_executed_volume_mwh',
      'horizon_settlement_pnl_dkk','cumulative_volume_transaction_fees_dkk','cumulative_market_access_fees_dkk','cumulative_impact_costs_dkk',
      'cumulative_collateral_funding_costs_dkk','trading_sleeve_value_dkk','pnl_battery']
def replay(path,region):
    d=pd.read_csv(path,usecols=lambda c:c in cols)
    S=pd.read_csv(SETTLE[region]).settlement_price.values[:len(d)]
    dec=d[(d.decision_step==1)&(d.liquidity_executed_volume_mwh>0)].copy()
    sgn=np.sign(dec.position_signed.values); t=dec.timestep.values.astype(int)
    spread=S[t]-dec.price_current.values
    pnl_exec=sgn*dec.liquidity_executed_volume_mwh.values*spread
    pnl_req=sgn*dec.liquidity_requested_volume_mwh.values*spread
    booked=d.horizon_settlement_pnl_dkk.sum()
    costs=d.cumulative_volume_transaction_fees_dkk.iloc[-1]+d.cumulative_market_access_fees_dkk.iloc[-1]+d.cumulative_impact_costs_dkk.iloc[-1]+d.cumulative_collateral_funding_costs_dkk.iloc[-1]
    sleeve0=d.trading_sleeve_value_dkk.iloc[0]
    # daily and 10-min P&L series (booked at maturity t+6)
    step=np.zeros(len(d)); np.add.at(step,np.minimum(t+6,len(d)-1),pnl_exec)
    day=np.arange(len(d))//144; daily=pd.Series(step).groupby(day).sum()
    batt=d.pnl_battery.fillna(0).values
    stepB=step+batt; dailyB=pd.Series(stepB).groupby(day).sum()
    sh=lambda x,ann: x.mean()/x.std()*np.sqrt(ann) if x.std()>0 else np.nan
    return dict(decisions=len(dec),booked_pnl=booked,replay_pnl=pnl_exec.sum(),replay_err_pct=100*(pnl_exec.sum()-booked)/max(abs(booked),1),
                net_return_pct=100*(booked-costs)/sleeve0,no_liquidity_cap_gross_return_pct=100*pnl_req.sum()/sleeve0,
                inflation_x_no_cap=pnl_req.sum()/max(pnl_exec.sum(),1e-9) if pnl_exec.sum()>0 else np.nan,
                sharpe_daily_annualised=sh(daily,365),sharpe_10min_annualised=sh(pd.Series(step),52596),
                sharpe_daily_with_battery=sh(dailyB,365),battery_share_of_gain=batt.sum()/(booked+batt.sum()) if booked+batt.sum()!=0 else np.nan)
rows=[]
for region,sub in [('2025','evaluations_2025'),('2025_v2','evaluations_2025_v2')]:
    for f in glob.glob(str(P5/f'Ablations/batch_tier_phase_runs/prototype5_mechanism_ablations_final_v1/focal_anchor/seed*/{sub}/*/env_logs/tier1_debug_ep0.csv')):
        rows.append(dict(method='anchor',region=region,seed=re.search(r'seed(\d+)',f).group(1),**replay(f,region)))
    for f in sorted(glob.glob(str(P5/f'batch_tier_phase_runs/prototype5_mappo_marl_final_v1/seed*/{sub}/tier1/env_logs/tier1_debug_ep0.csv'))):
        rows.append(dict(method='MARL',region=region,seed=re.search(r'seed(\d+)',f).group(1),**replay(f,region)))
R=pd.DataFrame(rows); R.to_csv(OUT/'decision_replay.csv',index=False)
pd.set_option('display.width',250)
print(R.round(3).to_string(index=False))
print('\nMEANS'); print(R.groupby(['method','region']).mean(numeric_only=True).round(3).T.to_string())
