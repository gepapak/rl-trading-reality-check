import subprocess, sys, time, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
A=Path(__file__).resolve().parent
vs=sys.argv[1:] or ['V00_strict','V01_percent_payoff_reference_price','V02_no_liquidity_cap','V03_load_proxy_liquidity','V04_legacy_flat_fee',
   'V05_no_collateral_no_margin','V06_no_impact_no_spread','V07_legacy_settlement_clip','V08_interpolated_prices','V09_cash_sweeper',
   'V10_continuous_mtm','V11_percent_capped_mtm','V12_thesis_like_cumulative']
env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
def run(v):
    t=time.time(); log=A/'results'/f'{v}.log'
    with open(log,'w',encoding='utf-8') as f:
        r=subprocess.run([sys.executable,str(A/'audit_run_variant.py'),v],stdout=f,stderr=subprocess.STDOUT,env=env)
    return v,r.returncode,round(time.time()-t)
with ThreadPoolExecutor(max_workers=4) as ex:
    for v,rc,sec in ex.map(run,vs): print(v,'exit',rc,'seconds',sec,flush=True)
