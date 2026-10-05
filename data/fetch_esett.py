"""Re-download the eSett Open Data files used by analysis/cross_market_liquidity_test.py and by Study J
(LIABILITY_STUDY_2026-09-29/get_esett_j.py copies the FI and NO2 files into place).

eSett's terms of use (https://opendata.esett.com/terms) make the data public without authorization but grant no
explicit redistribution license, so the raw files are not shipped in this repository. This script fetches exactly the
queries used in the study and checks them against the hashes recorded at download time:
- EXP14 (imbalance prices): per-call SHA-256 of every raw API response, from esett_download_manifest.json;
- EXP13 (imbalance volumes): SHA-256 of the decompressed CSV content, from CONTENT_SHA256.json.

Data published after the study's download is cut at the study's last timestamp, so an unrevised re-download
reproduces the study files exactly. Values eSett corrected later are reported as a warning, not an error.

Usage: python data/fetch_esett.py            (writes data/esett/esett_EXP1{3,4}_<zone>.csv.gz)
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "esett"
API = "https://api.opendata.esett.com"
MBAS = {"FI": "10YFI_1________U", "NO1": "10YNO_1________2", "NO2": "10YNO_2________T", "NO3": "10YNO_3________J",
        "NO4": "10YNO_4________9", "NO5": "10Y1001A1001A48H"}
START, END = pd.Timestamp("2025-03-04", tz="UTC"), pd.Timestamp("2026-09-28", tz="UTC")  # study window, fixed


def edges() -> list[pd.Timestamp]:
    e = [x for x in pd.date_range(START, END, freq="MS", tz="UTC") if x > START]
    return [START] + e + [END]


def get(url: str) -> bytes:
    for attempt in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "paper1-release-fetch"}), timeout=120).read()
        except Exception:  # noqa: BLE001
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}")


def fetch(endpoint: str, code: str) -> tuple[list[pd.DataFrame], list[str]]:
    frames, hashes = [], []
    for a, b in zip(edges()[:-1], edges()[1:]):
        q = {"start": a.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "end": b.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "mba": code}
        raw = get(f"{API}/{endpoint}?" + urllib.parse.urlencode(q))
        hashes.append(hashlib.sha256(raw).hexdigest())
        frames.append(pd.DataFrame(json.loads(raw)))
        time.sleep(0.5)
    return frames, hashes


def cutoff(df: pd.DataFrame, rec: dict) -> tuple[pd.DataFrame, int]:
    """Drop rows published after the study's download (timestamps later than the study's last timestamp)."""
    t = rec["time_column"]
    keep = df[t].astype(str) <= rec["last_timestamp"]
    return df[keep], int((~keep).sum())


def verify(path: Path, content: dict) -> bool:
    return hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == content[path.name]["sha256_uncompressed_csv"]


def main() -> int:
    OUT.mkdir(exist_ok=True)
    m14 = json.loads((OUT / "esett_download_manifest.json").read_text(encoding="utf-8"))["areas"]
    content = json.loads((OUT / "CONTENT_SHA256.json").read_text(encoding="utf-8"))
    warnings = 0
    for zone, code in MBAS.items():
        # EXP14 prices: same processing as the original download (dedupe on timestampUTC, sort)
        frames, hashes = fetch("EXP14/Prices", code)
        rec = [c["sha256"] for c in m14[zone]["calls"]]
        same = sum(h == r for h, r in zip(hashes, rec))
        df = pd.concat(frames, ignore_index=True).drop_duplicates("timestampUTC").sort_values("timestampUTC")
        df, cut14 = cutoff(df, content[f"esett_EXP14_{zone}.csv.gz"])
        f14 = OUT / f"esett_EXP14_{zone}.csv.gz"
        df.to_csv(f14, index=False)
        ok14 = same == len(rec) == len(hashes) or verify(f14, content)
        # EXP13 volumes: same processing as cross_market_liquidity_test.download_volumes (dedupe on timestamp)
        frames, _ = fetch("EXP13/ImbalancePowerVolume", code)
        dv = pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")
        dv, cut13 = cutoff(dv, content[f"esett_EXP13_{zone}.csv.gz"])
        f13 = OUT / f"esett_EXP13_{zone}.csv.gz"
        dv.to_csv(f13, index=False)
        ok13 = verify(f13, content)
        warnings += (not ok14) + (not ok13)
        print(f"{zone}: EXP14 {len(df)} rows ({cut14} later rows cut), "
              f"{'identical to the study data' if ok14 else f'{same}/{len(rec)} API responses identical; values REVISED by eSett since the study'}; "
              f"EXP13 {len(dv)} rows ({cut13} later rows cut), {'identical' if ok13 else 'values REVISED by eSett since the study'}", flush=True)
    print("all files identical to the study data" if not warnings else f"{warnings} file(s) differ from the study data (see above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
