"""Shared runner for the four parallel campaign harnesses.

Each harness executes its job list sequentially with the existing, audited
wrappers (marl.py, baselines.py, Ablations/mechanism_ablations.py, ...) so every
output layout, provenance manifest, and runtime-contract check stays exactly
as the aggregation scripts expect. Failures do not stop the lane: independent
jobs continue, and the harness exits non-zero with a summary so nothing fails
silently. Every wrapper call is resume-safe; re-running the same harness after
an interruption continues where it stopped.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SEEDS_FULL = ["7", "42", "123", "2025", "3007", "5001", "8102", "9005", "10001", "11202"]
SEEDS_MECH = ["7", "42", "123"]
SEEDS_DETERMINISTIC = ["7"]


def run_jobs(harness_name: str, jobs: list[tuple[str, list[str], bool]]) -> int:
    """Run (label, argv, supports_dry_run) jobs sequentially; return exit code."""
    dry_run = any(flag in sys.argv[1:] for flag in {"--dry_run", "--dry-run"})
    results: list[tuple[str, str, float]] = []
    t_start = time.time()
    for label, argv, supports_dry in jobs:
        if dry_run and not supports_dry:
            print(f"\n[{harness_name}] SKIP (no dry-run support): {label}")
            results.append((label, "skipped-dry", 0.0))
            continue
        cmd = [sys.executable, "-u", *argv]
        if dry_run:
            cmd.append("--dry_run")
        print("\n" + "=" * 88)
        print(f"[{harness_name}] START: {label}")
        print(" ".join(cmd))
        print("=" * 88, flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        hours = (time.time() - t0) / 3600.0
        status = "OK" if rc == 0 else f"FAILED rc={rc}"
        print(f"\n[{harness_name}] END:   {label}  ({status}, {hours:.2f} h)", flush=True)
        results.append((label, status, hours))
    total_h = (time.time() - t_start) / 3600.0
    print("\n" + "#" * 88)
    print(f"[{harness_name}] SUMMARY ({total_h:.2f} h total)")
    for label, status, hours in results:
        print(f"  {status:14s} {hours:7.2f} h  {label}")
    failures = [r for r in results if r[1].startswith("FAILED")]
    if failures:
        print(f"[{harness_name}] {len(failures)} job(s) FAILED - re-run this harness to resume/retry.")
        return 1
    print(f"[{harness_name}] all jobs completed.")
    return 0
