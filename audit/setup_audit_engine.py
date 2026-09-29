"""Build the audit engine inside Paper1: a CODE copy of Paper1/Prototype5 with data folders linked (Windows
junctions to Paper1/Prototype5, no duplication) and a minimal, clearly marked audit patch to evaluation.py:

  1. SIM_AUDIT_ALLOW_PROTOCOL_MISMATCH=1 turns the training/eval runtime-contract mismatch error into a logged
     warning, so a policy trained under the strict protocol can be evaluated under a shortcut protocol
     (the audit question: "what would a lax simulator report for this exact policy?").
  2. SIM_AUDIT_CFG_OVERRIDES='{"attr": value}' sets config attributes right before the evaluation environment is
     built (used for solvency switches that have no CLI flag). Unknown attributes raise.
  3. Every audit evaluation records {"sim_audit": {...}} in its tier-metrics JSON.

Paper1/Prototype5 is only read. Idempotent: re-running verifies instead of re-copying.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

AUD = Path(__file__).resolve().parent
P5 = AUD.parent / "Prototype5"
ENG = AUD / "engine"
DATA_DIRS = [
    "training_dataset_ffill", "evaluation_dataset_ffill", "rolling_past_history_dataset_ffill",
    "forecast_cache_settlement_hourly_v2", "forecast_models_settlement_hourly_v2",
    "forecast_training_dataset_settlement_hourly_v2", "forecast_cache_input_settlement_v2",
    "settlement_price_dataset_real_v2", "liquidity_volume_dataset_real_v1", "forecast_training_dataset_ffill",
    "dataset generation",
]
PATCH_MARK = "# [SIM_AUDIT PATCH]"

OLD_CHECK = (
    "    if train_hash and eval_runtime_hash != train_hash and eval_runtime_match_hash != train_hash "
    "and not overlay_contract_exception:\n        raise RuntimeError(\n"
)
NEW_CHECK = (
    f"    {PATCH_MARK} contract mismatch may be allowed only for audit evaluations\n"
    "    _sim_audit_mismatch = bool(train_hash and eval_runtime_hash != train_hash "
    "and eval_runtime_match_hash != train_hash and not overlay_contract_exception)\n"
    "    if _sim_audit_mismatch and os.environ.get('SIM_AUDIT_ALLOW_PROTOCOL_MISMATCH') == '1':\n"
    "        print(f'[SIM_AUDIT] protocol mismatch ALLOWED for audit evaluation: train={train_hash}, eval={eval_runtime_hash}')\n"
    "    elif _sim_audit_mismatch:\n        raise RuntimeError(\n"
)
OLD_ENV = "    env = create_evaluation_environment(\n        eval_data,\n        output_dir=tier_out,\n"
NEW_ENV = (
    f"    {PATCH_MARK} optional config overrides for audit evaluations\n"
    "    _sim_audit_overrides = json.loads(os.environ.get('SIM_AUDIT_CFG_OVERRIDES', '') or '{}')\n"
    "    for _k, _v in _sim_audit_overrides.items():\n"
    "        if not hasattr(cfg, _k):\n"
    "            raise AttributeError(f'[SIM_AUDIT] unknown config attribute {_k}')\n"
    "        setattr(cfg, _k, _v)\n"
    "    if _sim_audit_overrides:\n"
    "        print(f'[SIM_AUDIT] config overrides applied: {_sim_audit_overrides}')\n"
    + OLD_ENV
)
OLD_META = '        tier_metrics["engine_file_hashes"] = engine_file_hashes()\n'
NEW_META = (
    OLD_META
    + f"        {PATCH_MARK} provenance of audit evaluations\n"
    "        tier_metrics['sim_audit'] = {'protocol_mismatch': bool(_sim_audit_mismatch), "
    "'mismatch_allowed': os.environ.get('SIM_AUDIT_ALLOW_PROTOCOL_MISMATCH') == '1', "
    "'cfg_overrides': _sim_audit_overrides, 'variant': os.environ.get('SIM_AUDIT_VARIANT', '')}\n"
)


def _junction(link: Path, target: Path) -> None:
    if link.exists():
        return
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)


def _patch_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"[setup] patch anchor '{label}' found {n} times (expected 1); aborting")
    return text.replace(old, new)


def main() -> int:
    if not P5.is_dir():
        raise SystemExit(f"Paper1/Prototype5 not found: {P5}")
    ENG.mkdir(exist_ok=True)
    for sub in ["", "Ablations", "scripts"]:
        (ENG / sub).mkdir(exist_ok=True)
        for f in (P5 / sub).glob("*.py"):
            dst = ENG / sub / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
    for f in P5.glob("*.json"):
        if not (ENG / f.name).exists():
            shutil.copy2(f, ENG / f.name)
    if not (ENG / "baselines").exists():
        shutil.copytree(P5 / "baselines", ENG / "baselines", ignore=shutil.ignore_patterns("__pycache__"))
    for d in DATA_DIRS:
        if (P5 / d).is_dir():
            _junction(ENG / d, P5 / d)

    ev = ENG / "evaluation.py"
    text = ev.read_text(encoding="utf-8")
    if PATCH_MARK not in text:
        text = _patch_once(text, OLD_CHECK, NEW_CHECK, "contract check")
        text = _patch_once(text, OLD_ENV, NEW_ENV, "env creation")
        text = _patch_once(text, OLD_META, NEW_META, "tier metrics")
        ev.write_text(text, encoding="utf-8")
    subprocess.run([sys.executable, "-m", "py_compile", str(ev)], check=True)
    orig = (P5 / "evaluation.py").read_text(encoding="utf-8")
    unpatched = ev.read_text(encoding="utf-8")
    for new, old in [(NEW_META, OLD_META), (NEW_ENV, OLD_ENV), (NEW_CHECK, OLD_CHECK)]:
        unpatched = unpatched.replace(new, old)
    if unpatched != orig:
        raise SystemExit("[setup] engine/evaluation.py differs from Prototype5 beyond the audit patch")
    for f in P5.glob("*.py"):
        if f.name != "evaluation.py" and (ENG / f.name).read_bytes() != f.read_bytes():
            raise SystemExit(f"[setup] engine/{f.name} differs from Prototype5")
    missing = [d for d in DATA_DIRS if (P5 / d).is_dir() and not (ENG / d).is_dir()]
    if missing:
        raise SystemExit(f"[setup] missing data junctions: {missing}")
    print(f"[setup] audit engine ready at {ENG}; evaluation.py patched (3 marked blocks); all other code byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
