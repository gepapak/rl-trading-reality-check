"""Study D: appendix table of the coding of every screened repository (from coding_d.csv) -> sections/tab_envaudit.tex."""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = next((HERE.parents[1] / "Overleaf Projects (1 items)").glob("*"), HERE.parent / "outputs") / "sections" / "tab_envaudit.tex"


def esc(s: str) -> str:
    return str(s).replace("\\", "/").replace("_", "\\_").replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")


def short(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "."


def main() -> int:
    D = pd.read_csv(HERE / "coding_d.csv").fillna("")
    lines = [r"\begin{table}[!htbp]", r"\centering",
             r"\caption{Coding of the 20 screened public RL trading environments (static reading at pinned commits; file-and-line evidence is released with the code). Account: NOACC no capital account, ACC capital account. Loss accounting: UNLEV no borrowing, FL-cont/FL-term/FL-recap losses booked (continuing, terminating, recapitalization booked as debt), LL loss capped and trading stops, NF loss of the bankruptcy step not rewarded. Sizing: ABS absolute quantity, EQ fraction of equity, CASH bounded by cash. Prone: both conditions of Section~\ref{sec:theory}.}",
             r"\label{tab:envauditfull}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{@{}lrlllll@{}}", r"\toprule",
             r"Repository & Stars & Account & Borrow & Loss accounting & Sizing & Prone \\", r"\midrule"]
    for r in D.itertuples():
        if r.included != "yes":
            reason = str(r.exclusion_reason).split("(")[0].strip()
            lines.append(f"{esc(short(r.full_name, 30))} & {r.stars} & \\multicolumn{{5}}{{l}}{{excluded: {esc(reason)}}} \\\\")
            continue
        borrow = "yes" if str(r.borrowing).startswith("yes") else "no"
        loss = "--" if r.loss_accounting in ("n/a", "") else esc(short(r.loss_accounting.split("(")[0], 16))
        if r.loss_accounting.startswith("OTHER"):
            loss = "short-payoff cap"
        sizing = "--" if r.sizing in ("n/a", "") else esc(short(str(r.sizing).split("(")[0], 16))
        lines.append(f"{esc(short(r.full_name, 30))} & {r.stars} & {r.account} & {borrow} & {loss} & {sizing} & {'yes' if r.gambling_prone == 'YES' else 'no'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
