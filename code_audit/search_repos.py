"""Step 1 of the code audit (CODE_AUDIT_PROTOCOL.md): run the fixed GitHub repository searches and save candidate metadata.
Reads only public metadata; nothing is cloned or executed. Output: code_audit/candidates.csv"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
QUERIES = [
    "reinforcement learning electricity market trading",
    "reinforcement learning energy trading environment",
    "battery arbitrage reinforcement learning",
    "electricity price arbitrage reinforcement learning",
    "intraday electricity market reinforcement learning",
    "imbalance market reinforcement learning",
    "energy trading gym",
    "power trading deep reinforcement learning",
    "bidding strategy reinforcement learning electricity",
    "energy storage arbitrage deep reinforcement learning",
]
# Deviation 1 (CODE_AUDIT_PROTOCOL.md): the same queries with a README qualifier, plus shorter supplementary queries
SUPPLEMENTARY = [
    "electricity trading reinforcement", "energy trading reinforcement", "battery arbitrage", "electricity market reinforcement",
    "energy arbitrage reinforcement", "power market reinforcement learning", "day-ahead market reinforcement",
    "virtual power plant reinforcement", "energy storage reinforcement learning price", "bidding reinforcement electricity market",
]
QUALIFIER = " in:name,description,topics,readme"
ALL_QUERIES = QUERIES + [q + QUALIFIER for q in QUERIES] + [q + QUALIFIER for q in SUPPLEMENTARY]


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "paper1-code-audit"})
    for attempt in range(5):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=60).read())
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                time.sleep(65)  # unauthenticated search limit: 10 requests / minute
                continue
            raise
    raise RuntimeError(url)


def main() -> int:
    rows = []
    for q in ALL_QUERIES:
        url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode({"q": q, "sort": "stars", "order": "desc", "per_page": 30})
        res = get(url)
        for rank, it in enumerate(res.get("items", []), 1):
            rows.append(dict(query=q, rank=rank, full_name=it["full_name"], stars=it["stargazers_count"], fork=it["fork"],
                             language=it.get("language"), default_branch=it["default_branch"], pushed_at=it["pushed_at"],
                             description=(it.get("description") or "")[:200], html_url=it["html_url"]))
        print(f"{q}: {res.get('total_count')} total, {len(res.get('items', []))} kept", flush=True)
        time.sleep(7)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "candidates_by_query.csv", index=False)
    u = (df.sort_values("stars", ascending=False).groupby("full_name", as_index=False)
         .agg(stars=("stars", "first"), fork=("fork", "first"), language=("language", "first"), default_branch=("default_branch", "first"),
              pushed_at=("pushed_at", "first"), description=("description", "first"), html_url=("html_url", "first"),
              queries=("query", lambda x: " | ".join(sorted(set(x))))).sort_values("stars", ascending=False))
    u.to_csv(OUT / "candidates.csv", index=False)
    print(f"{len(u)} unique candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
