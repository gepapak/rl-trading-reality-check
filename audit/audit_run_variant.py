"""Run the UNMODIFIED Prototype5 final-baseline pipeline with exactly one protocol shortcut re-enabled.
Usage: python audit_run_variant.py <variant>. Writes only to SIMULATOR_AUDIT_2026-09-28/results/<variant>."""
import sys, json, importlib.util
from pathlib import Path
AUD=Path(__file__).resolve().parent; P5=AUD.parent/'Prototype5'
VARIANTS={
 'V00_strict':{},
 'V01_percent_payoff_reference_price':{'mtm_horizon_payoff_denominator_mode':'reference_price'},
 'V02_no_liquidity_cap':{'liquidity_participation_cap_fraction':1e9},
 'V03_load_proxy_liquidity':{'liquidity_volume_source':'load','liquidity_min_volume_mwh':100.0},
 'V04_legacy_flat_fee':{'market_fee_model':'legacy_notional_fixed'},
 'V05_no_collateral_no_margin':{'enable_collateral_cash_drag':False,'enable_trading_sleeve_margin':False},
 'V06_no_impact_no_spread':{'impact_coef_bp':0.0,'half_spread_bp':0.0,'liquidity_tail_impact_multiplier':1.0},
 'V07_legacy_settlement_clip':{'mtm_external_settlement_min_price_dkk_per_mwh':-1000.0,'mtm_external_settlement_max_price_dkk_per_mwh':10000.0},
 'V08_interpolated_prices':{'__interp__':True},
 'V09_cash_sweeper':{'distribution_rate':0.10,'eval_distribution_rate':0.10},
 'V10_continuous_mtm':{'mtm_return_model':'horizon_settlement_continuous'},
 'V11_percent_capped_mtm':{'mtm_return_model':'percent_capped'},
 'V12_thesis_like_cumulative':{'__interp__':True,'mtm_return_model':'percent_capped','distribution_rate':0.10,'eval_distribution_rate':0.10,
     'liquidity_participation_cap_fraction':1e9,'market_fee_model':'legacy_notional_fixed','enable_collateral_cash_drag':False,
     'enable_trading_sleeve_margin':False,'impact_coef_bp':0.0,'liquidity_tail_impact_multiplier':1.0},
}
v=sys.argv[1]; ov=dict(VARIANTS[v]); interp=ov.pop('__interp__',False)
sys.path.insert(0,str(P5)); sys.path.insert(0,str(P5/'scripts'))
spec=importlib.util.spec_from_file_location('rfpb',P5/'scripts'/'run_final_paper_baselines.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
unknown=[k for k in ov if k not in m.FINAL_PROTOCOL]
if unknown: raise SystemExit(f'unknown protocol keys {unknown}')
m.FINAL_PROTOCOL.update(ov); m.FINAL_PROTOCOL['protocol_name']=f'simulator_audit_{v}'
if interp:
    orig=m._available_regions
    def patched(include_v2=True):
        r=orig(include_v2)
        for x in r: x['eval_data']=AUD/'data'/Path(x['eval_data']).name.replace('.csv','_interpolated.csv')
        return r
    m._available_regions=patched
out=AUD/'results'/v; out.mkdir(parents=True,exist_ok=True)
(out/'audit_overrides.json').write_text(json.dumps({'variant':v,'overrides':ov,'interpolated_prices':interp},indent=1))
import os; os.chdir(P5)
sys.argv=['run_final_paper_baselines.py','--output_root',str(out),'--skip_rule_price_entry']
sys.exit(m.main())
