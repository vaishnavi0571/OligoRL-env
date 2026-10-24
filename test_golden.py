"""
Golden Test: rules behaviour must not change unintentionally
============================================================
Replays every built-in agent on seeds 0-19 (Medium and Hard) and compares
each firm's cash at every step, the final net worth and the composite
reward against stored snapshots, for two rule sets:
  legacy : npc_growth=False -- the round-1 rules (snapshot never rewritten)
  v2     : the current default rules (NPC growth + smooth quits)

Any opt-in experiment flag (NPC profiles, lobbying switch, PPO reward)
must leave both passing.

Run:            python test_golden.py
Re-snapshot v2: python test_golden.py --write   (only for intended changes)
"""

import json
import os
import sys

from simulator import EconSimulator
from reward_integrator import compute_step_reward
from agents import AGENTS, terminal_net_worth

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
MODES = {
    # name: (simulator flags, snapshot file, may be rewritten by --write)
    "legacy": ({"npc_growth": False}, "default_rules_seeds_0_19.json", False),
    "v2": ({}, "v2_default_rules_seeds_0_19.json", True),
}
GOLDEN_AGENTS = [
    "rule_based", "hold", "market+eps", "always_floor", "always_max",
    "poach_to_spill", "spill_parasite", "random",
]
DIFFICULTIES = ["medium", "hard"]
SEEDS = range(20)


def snapshot(sim_kwargs):
    out = {}
    for difficulty in DIFFICULTIES:
        for name in GOLDEN_AGENTS:
            for seed in SEEDS:
                sim = EconSimulator(difficulty=difficulty, seed=seed, **sim_kwargs)
                obs = sim.reset()
                agent = AGENTS[name]()
                agent.reset(seed)
                cash, reward = [], 0.0
                while not (sim.terminated or sim.truncated):
                    obs, info = sim.step(agent.act(obs))
                    reward += compute_step_reward(info, sim).composite
                    cash.append([round(c, 6) for c in obs["agent_wealth"]])
                out[f"{difficulty}/{name}/{seed}"] = {
                    "net_worth": round(terminal_net_worth(sim), 6),
                    "composite": round(reward, 9),
                    "cash": cash,
                }
    return out


def main():
    failed = False
    for mode, (sim_kwargs, filename, writable) in MODES.items():
        path = os.path.join(GOLDEN_DIR, filename)
        current = snapshot(sim_kwargs)
        if "--write" in sys.argv and writable:
            os.makedirs(GOLDEN_DIR, exist_ok=True)
            with open(path, "w") as f:
                json.dump(current, f)
            print(f"  [{mode}] wrote {len(current)} episodes to {path}")
            continue
        with open(path) as f:
            golden = json.load(f)
        mismatches = [k for k in golden if golden[k] != current.get(k)]
        missing = [k for k in golden if k not in current]
        print(f"  [{mode}] golden episodes: {len(golden)}   mismatches: {len(mismatches)}"
              f"   missing: {len(missing)}")
        for k in mismatches[:10]:
            g, c = golden[k], current.get(k, {})
            print(f"  [FAIL]  {mode} {k}: net worth {g['net_worth']} -> {c.get('net_worth')}")
        failed |= bool(mismatches or missing)
    if failed:
        print("\n  GOLDEN TEST FAILED -- rules changed.\n")
        return 1
    print("\n  GOLDEN TEST PASSED -- rules unchanged.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
