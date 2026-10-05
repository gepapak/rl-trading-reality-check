"""Build the 15-minute panel per region from the Energinet files in data_energinet/ (PREREGISTRATION_A.md, Data).

Output: data_panel/panel_{DK1,DK2}.pkl with columns
    imb, spot            imbalance and spot price, EUR/MWh
    bal_mw               balancing volume |SatisfiedDemand|, MW
    wind_act, wind_fc    actual and day-ahead forecast wind, MW, scaled to a 100 MW peak (train-period maximum)
    split                'train' (2025-03-04 .. 2025-12-31) or 'test' (2026-01-01 .. 2026-08-17)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "data_energinet"
OUT = HERE / "data_panel"
TRAIN_END = pd.Timestamp("2026-01-01", tz="UTC")
TEST_END = pd.Timestamp("2026-08-18", tz="UTC")
WIND_COLS = ["OffshoreWindLt100MW_MWh", "OffshoreWindGe100MW_MWh", "OnshoreWindLt50kW_MWh", "OnshoreWindGe50kW_MWh"]


def build(region: str) -> pd.DataFrame:
    ip = pd.read_csv(SRC / f"ImbalancePrice_{region}_post_golive.csv.gz", usecols=["TimeUTC", "ImbalancePriceEUR", "SpotPriceEUR", "SatisfiedDemand"])
    ip["t"] = pd.to_datetime(ip.TimeUTC, utc=True)
    ip = ip.drop_duplicates("t").set_index("t").sort_index()
    p = pd.DataFrame({"imb": ip.ImbalancePriceEUR, "spot": ip.SpotPriceEUR, "bal_mw": ip.SatisfiedDemand.abs()})
    idx = pd.date_range(p.index.min(), p.index.max(), freq="15min", tz="UTC")
    p = p.reindex(idx)

    pcs = pd.read_csv(SRC / f"ProductionConsumptionSettlement_{region}_post_golive.csv.gz", usecols=["HourUTC"] + WIND_COLS)
    pcs["t"] = pd.to_datetime(pcs.HourUTC, utc=True)
    act = pcs.drop_duplicates("t").set_index("t").sort_index()[WIND_COLS].sum(axis=1, min_count=1)  # MWh per hour = average MW
    fc = pd.read_csv(SRC / "Forecasts_Hour_post_golive.csv.gz", usecols=["HourUTC", "PriceArea", "ForecastType", "ForecastDayAhead"])
    fc = fc[(fc.PriceArea == region) & fc.ForecastType.isin(["Offshore Wind", "Onshore Wind"])]
    fc["t"] = pd.to_datetime(fc.HourUTC, utc=True)
    fc = fc.groupby("t").ForecastDayAhead.sum(min_count=1)
    hourly = pd.DataFrame({"wind_act": act, "wind_fc": fc})
    p = p.join(hourly.reindex(p.index, method="ffill", limit=3))  # hourly value held for the 4 quarters of the hour

    p = p[(p.index >= pd.Timestamp("2025-03-04", tz="UTC")) & (p.index < TEST_END)]
    p["split"] = np.where(p.index < TRAIN_END, "train", "test")
    scale = 100.0 / p.loc[p.split == "train", "wind_act"].max()
    p["wind_act"] *= scale
    p["wind_fc"] = (p["wind_fc"] * scale).clip(lower=0.0)
    n0 = len(p)
    p = p.dropna(subset=["imb", "spot", "bal_mw", "wind_act", "wind_fc"])
    print(f"{region}: {len(p)} quarters kept of {n0} ({n0 - len(p)} with missing inputs dropped); "
          f"train {int((p.split == 'train').sum())}, test {int((p.split == 'test').sum())}; wind scale {scale:.5f}")
    return p


def main() -> int:
    OUT.mkdir(exist_ok=True)
    for r in ("DK1", "DK2"):
        build(r).to_pickle(OUT / f"panel_{r}.pkl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
