"""Step 2 of the code audit (CODE_AUDIT_PROTOCOL.md): stage-2 screening material and automated candidate flags.

For each repository in stage1_keep.txt:
- pin the default-branch commit with `git ls-remote`;
- read the file tree (one GitHub API call);
- read README and .py/.ipynb sources from raw.githubusercontent.com at that commit, in memory only.

Third-party code is never executed. Short evidence excerpts are stored for manual coding.

Outputs:
- code_audit/stage2_repo_summary.csv
- code_audit/evidence_flags.csv
- code_audit/evidence/<owner>__<repo>.txt (README head, flagged lines, step() bodies)
"""
from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

import pandas as pd

AUD = Path(__file__).resolve().parent
EVD = AUD / "evidence"
EVD.mkdir(exist_ok=True)
UA = {"User-Agent": "paper1-code-audit"}
MONEY = r"\b(balance|budget|cash|equity|wealth|capital|net_worth|portfolio_value|bank|money|funds?)\w*"
PATTERNS = {
    "money_state": re.compile(MONEY, re.I),
    "floor_candidate": re.compile(r"(max\(\s*0(\.0*)?\s*,|np\.maximum\(\s*0|clip\(|min\(\s*0(\.0*)?\s*,).*" + MONEY + "|" + MONEY + r".*(max\(\s*0(\.0*)?\s*,|np\.maximum\(\s*0|clip\()", re.I),
    "solvency": re.compile(r"bankrupt|insolv|margin|liquidat|\bruin|if .*" + MONEY + r"\s*(<|<=)\s*0|done.*" + MONEY, re.I),
    "percent_payoff": re.compile(r"pct_change|/\s*\w*price\w*(\[[^\]]*\])?\s*-\s*1\b|\w*price\w*\[[^\]]*\]\s*/\s*\w*price\w*\[|log\(\s*\w*price|return\w*\s*=.*price.*/", re.I),
    "cash_settlement": re.compile(r"\w*price\w*(\[[^\]]*\])?\s*\*\s*\w*(power|energy|quantity|volume|action|charg|discharg|mwh|kwh|bid|amount|dispatch)\w*|\w*(power|energy|quantity|volume|action|charg|discharg|mwh|kwh|bid|amount|dispatch)\w*(\[[^\]]*\])?\s*\*\s*\w*price", re.I),
    "market_clearing": re.compile(r"clear|merit.?order|auction|order.?book|impact|slippage|liquidity|market.?depth|volume.?limit|uniform.?pric|pay.?as.?bid", re.I),
    "lookahead": re.compile(r"interpolat|bfill|backfill|shift\(\s*-|fit_transform\(|\.fit\(|MinMaxScaler|StandardScaler|normali[sz]", re.I),
    "metric": re.compile(r"sharpe|cumulative.?reward|total.?profit|episode.?reward|cum.?profit|total.?revenue", re.I),
}
PUB = re.compile(r"arxiv|doi\.org|\b10\.\d{4,}/|journal|conference|proceedings|ieee|elsevier|applied energy|energy and ai|neurips|icml|iclr|aaai|ijcai|e-energy|smartgridcomm|isgt|thesis", re.I)
SKIP = re.compile(r"(^|/)(tests?|docs?|site-packages|venv|\.venv|node_modules|build|dist|examples?/data)(/|$)", re.I)


def fetch(url: str, is_json: bool = False):
    for attempt in range(4):
        try:
            raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
            return json.loads(raw) if is_json else raw.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (403, 429):
                time.sleep(120)
                continue
            time.sleep(3)
        except Exception:
            time.sleep(3)
    return None


def code_lines(path: str, text: str) -> list[tuple[int, str]]:
    if path.endswith(".ipynb"):
        try:
            nb = json.loads(text)
        except Exception:
            return []
        out, n = [], 0
        for ci, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            src = cell.get("source", "")
            src = "".join(src) if isinstance(src, list) else src
            for ln in src.splitlines():
                n += 1
                out.append((n, f"[cell{ci}] {ln}"))
        return out
    return list(enumerate(text.splitlines(), 1))


def step_bodies(lines: list[tuple[int, str]], limit: int = 90) -> list[str]:
    out = []
    for i, (n, ln) in enumerate(lines):
        m = re.match(r"(\s*(\[cell\d+\] )?\s*)def (step|_step|reset|_get_reward|reward|_reward|calculate_reward|compute_reward)\b", ln)
        if m:
            ind = len(ln) - len(ln.lstrip())
            body = [f"{n}: {ln}"]
            for n2, l2 in lines[i + 1:i + 1 + limit]:
                if l2.strip() and (len(l2) - len(l2.lstrip())) <= ind and not l2.strip().startswith(("#", ")", "]")):
                    break
                body.append(f"{n2}: {l2}")
            out.append("\n".join(body))
    return out


def main() -> int:
    repos = [r.strip() for r in (AUD / "stage1_keep.txt").read_text().splitlines() if r.strip()]
    summ, flags = [], []
    for repo in repos:
        tag = repo.replace("/", "__")
        ls = subprocess.run(["git", "ls-remote", f"https://github.com/{repo}", "HEAD"], capture_output=True, text=True, timeout=120)
        sha = ls.stdout.split()[0] if ls.returncode == 0 and ls.stdout.strip() else None
        if not sha:
            summ.append(dict(repo=repo, sha=None, status="unreachable")); print(repo, "unreachable", flush=True); continue
        tree = fetch(f"https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1", is_json=True) or {}
        items = [t for t in tree.get("tree", []) if t.get("type") == "blob"]
        readme_path = next((t["path"] for t in items if re.match(r"readme(\.\w+)?$", t["path"], re.I)), None)
        readme = fetch(f"https://raw.githubusercontent.com/{repo}/{sha}/{readme_path}") if readme_path else ""
        readme = readme or ""
        code = [t for t in items if t["path"].endswith((".py", ".ipynb")) and not SKIP.search(t["path"]) and t.get("size", 0) <= 1_500_000]
        code = sorted(code, key=lambda t: (0 if re.search(r"env|market|trad|batter|agent|sim|reward|bid", t["path"], re.I) else 1, t["path"]))[:400]
        per_cat = {k: 0 for k in PATTERNS}
        dump = [f"# {repo} @ {sha}\n# files: {len(items)} total, {len(code)} code scanned\n\n## README (head)\n{readme[:2500]}\n"]
        steps, env_like = [], 0
        for t in code:
            text = fetch(f"https://raw.githubusercontent.com/{repo}/{sha}/{t['path']}")
            if not text:
                continue
            lines = code_lines(t["path"], text)
            if re.search(r"gym(nasium)?\.Env|def step\(|class \w*Env\b|class \w*Environment\b|def _step\(", text):
                env_like += 1
                for b in step_bodies(lines):
                    steps.append(f"### {t['path']}\n{b}")
            for n, ln in lines:
                s = ln.strip()
                if not s or s.startswith("#"):
                    continue
                for k, rx in PATTERNS.items():
                    if per_cat[k] < 60 and rx.search(s):
                        per_cat[k] += 1
                        flags.append(dict(repo=repo, sha=sha, path=t["path"], line=n, category=k, excerpt=s[:200]))
        dump.append("## flagged lines")
        for k in PATTERNS:
            dump.append(f"\n### {k}")
            dump += [f"{f['path']}:{f['line']}: {f['excerpt']}" for f in flags if f["repo"] == repo and f["category"] == k]
        dump.append("\n## step/reward/reset bodies (first 25)")
        dump += steps[:25]
        (EVD / f"{tag}.txt").write_text("\n".join(dump), encoding="utf-8")
        summ.append(dict(repo=repo, sha=sha, status="ok", n_files=len(items), n_code_scanned=len(code), env_like_files=env_like,
                         readme_has_publication=bool(PUB.search(readme)), readme_pub_match=(PUB.search(readme).group(0) if PUB.search(readme) else ""),
                         **{f"n_{k}": v for k, v in per_cat.items()}))
        print(repo, sha[:10], f"code={len(code)} env_like={env_like} pub={bool(PUB.search(readme))}", flush=True)
    pd.DataFrame(summ).to_csv(AUD / "stage2_repo_summary.csv", index=False)
    pd.DataFrame(flags).to_csv(AUD / "evidence_flags.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
