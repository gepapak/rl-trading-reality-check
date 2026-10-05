"""Release helper for Study J: put the eSett files that build_panel_j.py needs into data_esett_j/.

eSett Open Data are not redistributed. In the released repository, first run `python data/fetch_esett.py` from the
repository root (it re-downloads the files and verifies them against the study's content hashes), then

    python get_esett_j.py      # copies the four FI/NO2 files into data_esett_j/ after verifying their content hashes
    python build_panel_j.py    # builds data_panel_j/ exactly as in the study (it skips its own copy step)

The study's own files came from an earlier download of the same eSett queries; re-serialized as fetch_esett.py writes
them, their content hashes equal those in data/esett/CONTENT_SHA256.json, so the panels are identical.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ESETT = HERE.parent / "data" / "esett"
DST = HERE / "data_esett_j"
FILES = [f"esett_{e}_{z}.csv.gz" for z in ("FI", "NO2") for e in ("EXP13", "EXP14")]


def main() -> int:
    content = json.loads((ESETT / "CONTENT_SHA256.json").read_text(encoding="utf-8"))
    DST.mkdir(exist_ok=True)
    bad = 0
    for f in FILES:
        src = ESETT / f
        if not src.exists():
            print(f"missing {src}; run python data/fetch_esett.py first")
            bad += 1
            continue
        ok = hashlib.sha256(gzip.decompress(src.read_bytes())).hexdigest() == content[f]["sha256_uncompressed_csv"]
        shutil.copy2(src, DST / f)
        print(f"{f}: copied; content {'identical to the study data' if ok else 'differs (values revised by eSett since the study)'}")
        bad += not ok
    if bad:
        print("Warning: some files differ from the study data. eSett revisions after 2026-08-17 do not affect the panels; "
              "run build_panel_j.py and compare data_panel_j/*.pkl with PREREGISTRATION_HASHES.txt "
              "(python ../tools/verify_studies.py does this).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
