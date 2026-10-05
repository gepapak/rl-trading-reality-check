"""Study J markets: Study B's environment (envs_b.LiabilityEnv, unchanged) on the FI and NO2 panels (build_panel_j.py).

Only the data differ. The placebo market uses fresh sign seeds per zone and split.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from envs_b import Market, NoEdgeMarket

HERE = Path(__file__).resolve().parent
PANELS = HERE / "data_panel_j"
NE_SEED_J = {("FI", "train"): 9501, ("FI", "test"): 9502, ("NO2", "train"): 9601, ("NO2", "test"): 9602}


@lru_cache(maxsize=None)
def _panel(region: str) -> pd.DataFrame:
    return pd.read_pickle(PANELS / f"panel_{region}.pkl")


def load_market_j(region: str, split: str, market: str, draw: int = 0) -> Market:
    panel = _panel(region)
    return Market(panel, split) if market == "RM" else NoEdgeMarket(panel, split, NE_SEED_J[(region, split)] + 1000 * draw)
