"""Study D helper: pin a repository's commit, list candidate environment files, and print keyword lines (with line
numbers) from chosen files for static coding. Reads public content only; nothing is executed or saved except pins.json.

    python inspect_d.py tree  owner/repo            # pin commit, list .py files that look like environments
    python inspect_d.py grep  owner/repo path.py    # keyword lines of one file at the pinned commit
    python inspect_d.py show  owner/repo path.py A B  # lines A..B of one file at the pinned commit
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = HERE / "pins.json"
KEY = re.compile(r"balance|net_?worth|equity|cash|portfolio_?val|valuation|capital|reward|done|terminat|truncat|bankrupt|"
                 r"margin|leverage|short|borrow|max\(|min\(|clip|isfinite|isnan|nan|<= ?0|< ?0|stop|liquidat|profit", re.I)


def get(url: str, api: bool = False):
    h = {"User-Agent": "study-d-audit"}
    if api:
        h["Accept"] = "application/vnd.github+json"
    with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=30) as r:
        data = r.read()
    return json.loads(data) if api else data.decode("utf-8", errors="replace")


def pins() -> dict:
    return json.loads(PINS.read_text()) if PINS.exists() else {}


def tree(repo: str) -> None:
    import csv
    branch = next(r["default_branch"] for r in csv.DictReader(open(HERE / "candidates.csv", encoding="utf-8")) if r["full_name"] == repo)
    sha = get(f"https://api.github.com/repos/{repo}/commits/{branch}", api=True)["sha"]
    p = pins()
    p[repo] = sha
    PINS.write_text(json.dumps(p, indent=1))
    t = get(f"https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1", api=True)
    py = [x for x in t["tree"] if x["type"] == "blob" and x["path"].endswith(".py")]
    envlike = [x for x in py if re.search(r"env|gym|trad|market|simul|account|portfolio|broker|exchange", x["path"], re.I)]
    print(f"{repo} @ {sha[:12]}: {len(py)} .py files, {len(envlike)} environment-like")
    for x in envlike[:60]:
        print(f"  {x.get('size', 0):>7}  {x['path']}")


def lines(repo: str, path: str) -> list[str]:
    return get(f"https://raw.githubusercontent.com/{repo}/{pins()[repo]}/{path}").splitlines()


def grep(repo: str, path: str) -> None:
    for i, ln in enumerate(lines(repo, path), 1):
        if KEY.search(ln) and not ln.strip().startswith("#"):
            print(f"{i:>5}: {ln.rstrip()[:150]}")


def show(repo: str, path: str, a: int, b: int) -> None:
    for i, ln in enumerate(lines(repo, path)[a - 1:b], a):
        print(f"{i:>5}: {ln.rstrip()[:160]}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "tree":
        tree(sys.argv[2])
    elif cmd == "grep":
        grep(sys.argv[2], sys.argv[3])
    else:
        show(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
