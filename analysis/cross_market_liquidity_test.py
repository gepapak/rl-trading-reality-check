"""Cross-market liquidity-illusion test (pre-registered addendum in PREREGISTRATION.md).
Downloads eSett EXP13 net imbalance volumes for the 12 Nordic zones into Paper1, validates the volume proxy on DK against
Energinet activation volumes, then computes price-taker inflation I and value-vs-volume fill ratio R per zone."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

AUD = Path(__file__).resolve().parent
REPO = AUD.parent
ES = REPO / "data" / "esett"
OUT = REPO / "results" / "cross_market"
MBAS = {"DK1": "10YDK-1--------W", "DK2": "10YDK-2--------M", "FI": "10YFI_1________U", "NO1": "10YNO_1________2",
        "NO2": "10YNO_2________T", "NO3": "10YNO_3________J", "NO4": "10YNO_4________9", "NO5": "10Y1001A1001A48H",
        "SE1": "10Y1001A1001A44P", "SE2": "10Y1001A1001A45N", "SE3": "10Y1001A1001A46L", "SE4": "10Y1001A1001A47J"}
T = ["DK1", "DK2", "FI"]                      # recorded deviation: SE4 excluded (no public volume data)
H = ["NO1", "NO2", "NO3", "NO4", "NO5"]       # recorded deviation: SE1, SE2 excluded
ZONES = T + H


def download_volumes() -> None:
    start = pd.Timestamp("2025-03-04", tz="UTC"); end = pd.Timestamp("2026-09-28", tz="UTC")
    edges = [start] + [e for e in pd.date_range(start, end, freq="MS", tz="UTC") if e > start] + [end]
    manifest = {"source": "https://api.opendata.esett.com/EXP13/ImbalancePowerVolume", "areas": {}}
    for name, code in MBAS.items():
        if name not in ZONES or name.startswith("DK"):
            continue
        f = ES / f"esett_EXP13_{name}.csv.gz"
        if f.exists():
            continue
        frames = []
        for a, b in zip(edges[:-1], edges[1:]):
            q = {"start": a.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "end": b.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "mba": code}
            url = "https://api.opendata.esett.com/EXP13/ImbalancePowerVolume?" + urllib.parse.urlencode(q)
            for attempt in range(4):
                try:
                    raw = urllib.request.urlopen(url, timeout=120).read(); break
                except Exception:
                    time.sleep(5 * (attempt + 1))
            else:
                raise RuntimeError(f"download failed {name} {a}")
            frames.append(pd.DataFrame(json.loads(raw)))
            time.sleep(0.5)
        df = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
        df.to_csv(f, index=False)
        manifest["areas"][name] = {"rows": len(df), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        print("downloaded", name, len(df), flush=True)
    mf = ES / "esett_EXP13_download_manifest.json"
    if manifest["areas"]:
        mf.write_text(json.dumps(manifest, indent=1))


def zone_frame(name: str) -> pd.DataFrame:
    if name.startswith("DK"):  # recorded deviation: Energinet balancing volume |SatisfiedDemand| (eSett EXP13 has no DK)
        e = pd.read_csv(REPO / "data" / "energinet" / f"ImbalancePrice_{name}_post_golive.csv.gz",
                        parse_dates=["TimeUTC"]).drop_duplicates("TimeUTC").set_index("TimeUTC").sort_index()
        d = pd.DataFrame({"s": e.ImbalancePriceEUR - e.SpotPriceEUR, "imb": e.SatisfiedDemand,
                          "purchase": np.nan, "sales": np.nan}).dropna(subset=["s", "imb"])
        return d
    p = pd.read_csv(ES / f"esett_EXP14_{name}.csv.gz", parse_dates=["timestampUTC"]).set_index("timestampUTC")
    v = pd.read_csv(ES / f"esett_EXP13_{name}.csv.gz")
    v["t"] = pd.to_datetime(v["timestamp"]).dt.tz_localize("Europe/Copenhagen" if name.startswith("DK") else
                                                            ("Europe/Helsinki" if name == "FI" else
                                                             ("Europe/Oslo" if name.startswith("NO") else "Europe/Stockholm")),
                                                            ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    v = v.dropna(subset=["t"]).drop_duplicates("t").set_index("t")
    d = p[["imblSpotDifferencePrice"]].join(v[["imbalance", "imbalancePurchase", "imbalanceSales"]], how="inner").dropna()
    d.columns = ["s", "imb", "purchase", "sales"]
    return d


def metrics(d: pd.DataFrame, cap_frac: float = 0.25, size_mult: float = 10.0) -> dict:
    c = cap_frac * d.imb.abs().values
    q = size_mult * np.median(c[c > 0]) if np.any(c > 0) else np.nan
    s = d.s.values
    fill = np.minimum(q, c)
    I = np.sum(q * np.abs(s)) / np.sum(fill * np.abs(s))
    R = (np.sum(fill * np.abs(s)) / np.sum(q * np.abs(s))) / (np.sum(fill) / np.sum(q * np.ones_like(s)))
    a = np.sign(pd.Series(s).shift(8).fillna(0).values)                    # 2-hour persistence (secondary)
    pnl_cap, pnl_nocap = np.sum(a * fill * s), np.sum(a * q * s)
    top = np.abs(s) >= np.quantile(np.abs(s), 0.99)
    return dict(n=len(d), median_exec_volume_mwh=float(np.median(c)), I_price_taker_inflation=float(I), R_value_vs_volume_fill=float(R),
                fill_ratio_top1pct_spreads=float(np.mean(fill[top] / q)), fill_ratio_other=float(np.mean(fill[~top] / q)),
                spearman_abs_s_vs_exec_volume=float(spearmanr(np.abs(s), c)[0]),
                persistence_pnl_capped=float(pnl_cap), persistence_pnl_price_taker=float(pnl_nocap))


def main() -> int:
    download_volumes()
    # proxy validation on DK: eSett |net imbalance| vs Energinet activation volume used by the engine
    val = {}
    for name, fn in [("DK1", "unseendata_liquidity_volume_real_v1.csv"), ("DK2", "unseendata_v2_liquidity_volume_real_v1.csv")]:
        eng = pd.read_csv(REPO / "data" / "engine_liquidity" / fn, parse_dates=["timestamp"])
        eng["t"] = eng.timestamp.dt.tz_localize("UTC") if eng.timestamp.dt.tz is None else eng.timestamp
        e_h = eng.set_index("t").market_volume_mwh.resample("h").mean()
        z = zone_frame(name).imb.abs().resample("h").sum()
        j = pd.concat([e_h, z], axis=1, join="inner").dropna()
        val[name] = dict(hours=len(j), spearman=float(spearmanr(j.iloc[:, 0], j.iloc[:, 1])[0]))
    rows = []
    struct = pd.read_csv(REPO / "data" / "derived" / "cross_market_structure.csv").set_index("area")
    for name in ZONES:
        m = metrics(zone_frame(name))
        m.update(zone=name, group="T" if name in T else ("H" if name in H else "-"), alpha_upper=float(struct.loc[name, "alpha_upper"]))
        rows.append(m)
    R = pd.DataFrame(rows).set_index("zone")
    R.to_csv(OUT / "cross_market_liquidity_test.csv")
    pd.set_option("display.width", 250)
    print("proxy validation (Spearman eSett |net imbalance| vs Energinet activation volume, hourly):", val)
    print(R.round(3).to_string())
    t, h = R[R.group == "T"], R[R.group == "H"]
    v = [
        f"X1 mean R: T {t.R_value_vs_volume_fill.mean():.3f} vs H {h.R_value_vs_volume_fill.mean():.3f} -> {'SUPPORTED' if t.R_value_vs_volume_fill.mean() < h.R_value_vs_volume_fill.mean() else 'NOT SUPPORTED'}",
        f"X2 mean I: T {t.I_price_taker_inflation.mean():.2f} vs H {h.I_price_taker_inflation.mean():.2f} -> {'SUPPORTED' if t.I_price_taker_inflation.mean() > h.I_price_taker_inflation.mean() else 'NOT SUPPORTED'}",
        f"X3 DK: I {R.loc[['DK1','DK2'],'I_price_taker_inflation'].round(2).tolist()}, R {R.loc[['DK1','DK2'],'R_value_vs_volume_fill'].round(3).tolist()} -> {'SUPPORTED' if (R.loc[['DK1','DK2'],'I_price_taker_inflation']>1).all() and (R.loc[['DK1','DK2'],'R_value_vs_volume_fill']<1).all() else 'NOT SUPPORTED'}",
    ]
    rho = spearmanr(R.alpha_upper, R.I_price_taker_inflation)[0]
    v.append(f"X4 Spearman(alpha_upper, I) across 8 zones = {rho:+.3f} -> {'SUPPORTED' if rho < 0 else 'NOT SUPPORTED'}")
    print("\n".join(v))
    (OUT / "CROSS_MARKET_VERDICTS.md").write_text("# Cross-market liquidity-illusion verdicts\n\nProxy validation: " + json.dumps(val) + "\n\n" + "\n".join(f"- {x}" for x in v) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
