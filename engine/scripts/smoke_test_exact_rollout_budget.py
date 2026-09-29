#!/usr/bin/env python3
"""Lightweight regression test for exact PPO rollout budgeting."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metacontroller import MultiESGAgent


class _Memory:
    def should_cleanup(self):
        return None, 0.0

    def cleanup(self, _level):
        return 0.0


class _Validator:
    def validate_observation_dict(self, _obs):
        return True


def main() -> int:
    agent = object.__new__(MultiESGAgent)
    buffer = SimpleNamespace(pos=0, buffer_size=256, full=False)
    policy = SimpleNamespace(mode="PPO", agent_name="investor_0", n_steps=256, rollout_buffer=buffer)
    agent.config = SimpleNamespace(rollout_cap=256)
    agent.n_steps = 256
    agent.policies = [policy]
    agent.memory_tracker = _Memory()
    agent.obs_validator = _Validator()
    agent._last_obs = {"investor_0": np.zeros(1, dtype=np.float32)}
    agent._training_metrics = {"observation_fixes": 0}
    agent.logger = logging.getLogger("rollout_budget_smoke")
    finalized = []

    agent._collect_actions_enhanced = MethodType(lambda self: ({}, {}), agent)
    agent._execute_environment_step_enhanced = MethodType(
        lambda self, _actions: (self._last_obs, {}, {}, {}, {}), agent
    )

    def _add(self, *_args):
        buffer.pos += 1
        buffer.full = buffer.pos >= buffer.buffer_size

    agent._add_experiences_enhanced = MethodType(_add, agent)
    agent._update_state_enhanced = MethodType(lambda self, *_args: None, agent)
    agent._finalize_rollouts_enhanced = MethodType(lambda self: finalized.append(buffer.pos), agent)

    first = agent._collect_rollouts_enhanced([], max_steps=160)
    assert first == 160 and buffer.pos == 160 and not finalized
    second = agent._collect_rollouts_enhanced([], max_steps=96)
    assert second == 96 and buffer.pos == 256 and finalized == [256]
    print("EXACT_ROLLOUT_BUDGET_SMOKE_OK: 160 + 96 = one causal 256-step PPO rollout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
