"""Deep-review rigor checks (exploratory, post hoc; not part of any pre-registration).

1. B3 with a two-way bootstrap: the 20 placebo sign draws are SHARED by all seeds, so seed-only intervals omit the
   between-draw variance. Resample draws and seeds jointly (same draw indices for every seed in a resample).
2. B2 effect size: RS(ZF) - RS(FL) pooled over the placebo market, with a seed-level permutation test, per algorithm.
3. The leverage probe as a diagnostic: how well does RS separate floor-trained from full-liability agents (AUC),
   in Study B (400k) and Study C (1.6M snapshot), placebo market.
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

rng = np.random.default_rng(20260930)
N = 10_000


def jobs_b() -> pd.DataFrame:
    rows = [json.load(open(f)) for f in glob.glob("results/jobs/*__NE__*.json")]
    return pd.DataFrame(rows)


def twoway(mat: np.ndarray) -> tuple[float, float]:
    """mat: seeds x draws. Resample seeds and draws independently; mean over the resampled grid."""
    s, d = mat.shape
    out = np.empty(N)
    for b in range(N):
        i = rng.integers(0, s, s)
        j = rng.integers(0, d, d)
        out[b] = mat[np.ix_(i, j)].mean()
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def perm_p(a: np.ndarray, b: np.ndarray) -> float:
    obs = a.mean() - b.mean()
    pool = np.r_[a, b]
    cnt = 0
    for _ in range(N):
        p = rng.permutation(pool)
        cnt += (p[: len(a)].mean() - p[len(a):].mean()) >= obs
    return (cnt + 1) / (N + 1)


def main() -> None:
    J = jobs_b()
    print("1. B3 two-way bootstrap (seeds x shared draws), placebo market, reported return %")
    for (algo, region, liab), g in J.groupby(["algo", "region", "liability"]):
        mat = np.array(g.draws_reported_return_pct.tolist())
        led = np.array(g.draws_ledger_return_pct.tolist())
        lo, hi = twoway(mat)
        print(f"   {algo} {region} {liab}: reported {mat.mean():+7.1f} [{lo:+.0f}, {hi:+.0f}]  ledger {led.mean():+7.1f}  "
              f"inflation {(mat - led).mean():+6.1f}")
    print("\n2. B2 effect size RS(ZF) - RS(FL), placebo market, seed-level runs (both zones)")
    J["RS"] = J["probe_e0.1"] - J["probe_e1"]
    for algo in ("PPO", "DQN", "both"):
        g = J if algo == "both" else J[J.algo == algo]
        z, f = g[g.liability == "ZF"].RS.to_numpy(), g[g.liability == "FL"].RS.to_numpy()
        diff = z.mean() - f.mean()
        bs = [rng.choice(z, len(z)).mean() - rng.choice(f, len(f)).mean() for _ in range(N)]
        print(f"   {algo}: ZF {z.mean():+.2f} (n={len(z)}), FL {f.mean():+.2f} (n={len(f)}), diff {diff:+.2f} "
              f"[{np.quantile(bs, .025):+.2f}, {np.quantile(bs, .975):+.2f}], one-sided permutation p = {perm_p(z, f):.4f}")
    print("\n3. Leverage probe as a diagnostic: AUC of RS for floor-trained vs full-liability agents (placebo market)")
    print(f"   Study B (400k): all {auc(J[J.liability=='ZF'].RS, J[J.liability=='FL'].RS):.2f}, "
          f"PPO {auc(J[(J.liability=='ZF')&(J.algo=='PPO')].RS, J[(J.liability=='FL')&(J.algo=='PPO')].RS):.2f}, "
          f"DQN {auc(J[(J.liability=='ZF')&(J.algo=='DQN')].RS, J[(J.liability=='FL')&(J.algo=='DQN')].RS):.2f}")
    S = pd.read_csv("scaling/results/snapshots_c.csv")
    for step in (400_000, 1_600_000):
        s = S[S.step == step]
        print(f"   Study C ({step // 1000}k): all {auc(s[s.liability=='ZF'].RS, s[s.liability=='FL'].RS):.2f}, "
              f"PPO {auc(s[(s.liability=='ZF')&(s.algo=='PPO')].RS, s[(s.liability=='FL')&(s.algo=='PPO')].RS):.2f}, "
              f"DQN {auc(s[(s.liability=='ZF')&(s.algo=='DQN')].RS, s[(s.liability=='FL')&(s.algo=='DQN')].RS):.2f}")
    # combined evidence for the equity-dependence, fresh seeds at 1.6M
    s = S[S.step == 1_600_000]
    z, f = s[s.liability == "ZF"].RS.to_numpy(), s[s.liability == "FL"].RS.to_numpy()
    print(f"   Study C 1.6M: ZF {z.mean():+.2f}, FL {f.mean():+.2f}, permutation p = {perm_p(z, f):.4f}")


if __name__ == "__main__":
    main()
