"""Download full texts (public arXiv PDFs) of candidate papers and extract verbatim passages for each coding
question of LITERATURE_SURVEY_PROTOCOL.md. Coding itself is done by reading these passages (evidence/*.md)."""
import re, time, json, urllib.request
from pathlib import Path
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
PDF = HERE / "pdfs"; EVD = HERE / "evidence"
PDF.mkdir(exist_ok=True); EVD.mkdir(exist_ok=True)
PAPERS = ["2111.13609", "2004.05940", "2510.16021", "2401.00015", "2404.18821", "2402.19110", "2212.06551",
          "2410.11180", "2404.17683", "2303.16266", "2411.15422", "2411.16519", "2606.27032", "2507.16479",
          "2605.23964", "2106.02396", "2510.03657", "2604.19580", "2608.26122", "2504.06932", "2402.01215",
          "2505.14133"]
CODES = {
    "C1_liquidity": r"price[- ]taker|liquidit|market depth|order book|market volume|traded volume|price impact|market impact|slippage|volume constraint|participation",
    "C2_costs": r"transaction cost|trading cost|fee|commission|degradation cost|bid-ask|spread cost",
    "C3_solvency": r"margin|bankrupt|ruin|collateral|capital constraint|budget constraint|credit|drawdown",
    "C4_lookahead": r"perfect foresight|look-?ahead|interpolat|realised price|realized price|ex[- ]post|oracle|future price",
    "C5_spikes": r"spike|outlier|clip|winsor|extreme price|cap(ped)? (at|to)|removed",
    "C6_metrics": r"sharpe|annuali[sz]|profit|revenue|return",
    "C7_seeds": r"random seed|seeds|independent runs|runs with different|repetitions|trials",
    "C8_baselines": r"baseline|benchmark|rule-based|heuristic|perfect foresight|MILP|optimi[sz]ation-based",
}


def fetch(aid: str) -> Path:
    p = PDF / f"{aid}.pdf"
    if not p.exists():
        req = urllib.request.Request(f"https://arxiv.org/pdf/{aid}", headers={"User-Agent": "Mozilla/5.0 (research survey)"})
        p.write_bytes(urllib.request.urlopen(req, timeout=120).read())
        time.sleep(3)
    return p


def main():
    index = {}
    for aid in PAPERS:
        try:
            p = fetch(aid)
            txt = " ".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
        except Exception as e:
            index[aid] = {"error": str(e)}
            print(aid, "ERROR", e)
            continue
        txt = re.sub(r"\s+", " ", txt)
        title = txt[:300]
        out = [f"# {aid}\n\nFirst 300 chars: {title}\n"]
        counts = {}
        for code, pat in CODES.items():
            hits = list(re.finditer(pat, txt, flags=re.I))
            counts[code] = len(hits)
            out.append(f"\n## {code} ({len(hits)} hits)\n")
            seen = 0
            for m in hits:
                if seen >= 8:
                    break
                s = txt[max(0, m.start() - 220): m.end() + 220]
                out.append(f"- ...{s}...\n")
                seen += 1
        (EVD / f"{aid}.md").write_text("".join(out), encoding="utf-8")
        index[aid] = {"chars": len(txt), "counts": counts, "title_snippet": title[:160]}
        print(aid, len(txt), counts)
    (HERE / "evidence_index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
