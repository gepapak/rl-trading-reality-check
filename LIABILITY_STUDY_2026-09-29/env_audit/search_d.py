"""Study D search (PREREGISTRATION_D.md): the ten fixed GitHub repository queries, top 30 by stars each, merged.
Writes candidates.csv (public repository metadata only)."""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUERIES = ["trading environment reinforcement learning", "gym trading", "trading gym environment",
           "stock trading reinforcement learning", "crypto trading reinforcement learning", "forex reinforcement learning",
           "portfolio reinforcement learning environment", "futures trading reinforcement learning", "gymnasium trading",
           "algorithmic trading deep reinforcement learning"]


def search(q: str) -> list[dict]:
    url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(
        {"q": f"{q} language:Python", "sort": "stars", "order": "desc", "per_page": 30})
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "study-d-audit"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["items"]


def main() -> int:
    rows = {}
    for i, q in enumerate(QUERIES):
        items = search(q)
        for it in items:
            k = it["full_name"]
            rows.setdefault(k, dict(full_name=k, stars=it["stargazers_count"], default_branch=it["default_branch"],
                                    description=(it.get("description") or "").replace("\n", " ")[:200], queries=[]))
            rows[k]["queries"].append(i + 1)
        print(f"query {i + 1}: {len(items)} results", flush=True)
        time.sleep(7)  # unauthenticated search limit: 10 requests per minute
    import pandas as pd
    D = pd.DataFrame(rows.values()).sort_values("stars", ascending=False)
    D["queries"] = D.queries.apply(lambda v: ",".join(map(str, v)))
    D["searched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    D.to_csv(HERE / "candidates.csv", index=False)
    print(len(D), "unique repositories;", int((D.stars >= 200).sum()), "with >= 200 stars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
