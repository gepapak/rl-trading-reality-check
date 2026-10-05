"""Study J data: 15-minute panels for two non-Danish Nordic bidding zones from eSett Open Data (already in Paper1).

Zones (fixed before any modelling): FI, the only Finnish zone, whose spreads resemble Denmark's; and NO2, the
Norwegian zone with the largest mean absolute imbalance volume, a hydro-dominated zone with much smaller spreads.

Sources (copied unchanged from INVESTOR_ACCURACY_TRAP_2026-09-28/data_external_esett/ into data_esett_j/):
    EXP14 (prices):  imblSalesPrice (= imblPurchasePrice, single price), imblSpotDifferencePrice (imbalance - spot)
    EXP13 (volumes): imbalance (net area imbalance, MWh per quarter-hour)
Output: data_panel_j/panel_{FI,NO2}.pkl with the columns of the Danish panels (GENERALIZATION_STUDY/build_panel.py)
    imb      imbalance price, EUR/MWh
    spot     day-ahead price = imbalance price - imbalance-spot difference, EUR/MWh (validated on DK1: identical to
             Energinet's SpotPriceEUR in 100% of quarter-hours)
    bal_mw   |net area imbalance| x 4, MW (the Danish panels use |SatisfiedDemand|, the TSO's net balancing in MW)
    wind_act, wind_fc   0 (not used by the liability environment)
    split    'train' (2025-03-04 .. 2025-12-31) or 'test' (2026-01-01 .. 2026-08-17), as for DK1/DK2
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC0 = HERE.parent / "INVESTOR_ACCURACY_TRAP_2026-09-28" / "data_external_esett"
SRC = HERE / "data_esett_j"
OUT = HERE / "data_panel_j"
ZONES = ("FI", "NO2")
TRAIN_END = pd.Timestamp("2026-01-01", tz="UTC")
TEST_END = pd.Timestamp("2026-08-18", tz="UTC")


def build(zone: str) -> pd.DataFrame:
    a = pd.read_csv(SRC / f"esett_EXP14_{zone}.csv.gz", usecols=["timestampUTC", "imblSalesPrice", "imblSpotDifferencePrice"])
    v = pd.read_csv(SRC / f"esett_EXP13_{zone}.csv.gz", usecols=["timestampUTC", "imbalance"])
    for d in (a, v):
        d["t"] = pd.to_datetime(d.timestampUTC, utc=True)
    a = a.drop_duplicates("t").set_index("t").sort_index()
    v = v.drop_duplicates("t").set_index("t").sort_index()
    p = pd.DataFrame({"imb": a.imblSalesPrice, "spot": a.imblSalesPrice - a.imblSpotDifferencePrice})
    p = p.join(v.imbalance.abs().mul(4.0).rename("bal_mw"), how="left")
    idx = pd.date_range(p.index.min(), p.index.max(), freq="15min", tz="UTC")
    p = p.reindex(idx)
    p["wind_act"] = 0.0
    p["wind_fc"] = 0.0
    p = p[(p.index >= pd.Timestamp("2025-03-04", tz="UTC")) & (p.index < TEST_END)]
    p["split"] = np.where(p.index < TRAIN_END, "train", "test")
    n0 = len(p)
    p = p.dropna(subset=["imb", "spot", "bal_mw"])
    print(f"{zone}: {len(p)} quarters kept of {n0} ({n0 - len(p)} with missing inputs dropped); "
          f"train {int((p.split == 'train').sum())}, test {int((p.split == 'test').sum())}; "
          f"|spread| mean {np.abs(p.imb - p.spot).mean():.1f} EUR/MWh; bal_mw mean {p.bal_mw.mean():.0f}")
    return p


def main() -> int:
    SRC.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    for zone in ZONES:
        for exp in ("EXP13", "EXP14"):
            f = f"esett_{exp}_{zone}.csv.gz"
            if not (SRC / f).exists():
                shutil.copy2(SRC0 / f, SRC / f)
            print(f, hashlib.sha256((SRC / f).read_bytes()).hexdigest())
        build(zone).to_pickle(OUT / f"panel_{zone}.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
