"""
Benchmark selection on VALIDATION seeds (frozen before any test run)
====================================================================
Runs every scripted agent under the PPO experiment rules (lobbying off)
on the default-profile and heldout-profile validation ranges, picks the
best agent by mean final net worth for each, and freezes the choice in
results/ppo/benchmark.json.

Also reports (validation seeds only):
  - Premium pricer (spill_parasite) vs Rule-based, spillover on and off
  - wage vs profit: fixed wage/market ratio sweeps and correlations

Run:  python -m ppo.benchmark            (current rules, v2)
      python -m ppo.benchmark --legacy   (round-1 rules -> results/ppo/round1/)
"""

import json
import math
import os
import random
import statistics
import sys

from agents import AGENTS, Agent, run_episode
from config import SPILLOVER_FRACTION, SPILLOVER_FRACTION_V1
from ppo.train import DEFAULT_CONFIG, load_config

LEGACY = "--legacy" in sys.argv
OUT = os.path.join("results", "ppo", "round1") if LEGACY else os.path.join("results", "ppo")


def ci95(values):
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    return {"mean": statistics.mean(d), "ci": ci95(d), "std": statistics.stdev(d),
            "win_rate": sum(x > 0 for x in d) / len(d)}


def pearson(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


class WageOverride(Agent):
    """Any agent, but always targeting market wage x ratio."""

    def __init__(self, base: Agent, ratio: float):
        self.base, self.ratio = base, ratio

    def reset(self, seed=0):
        self.base.reset(seed)

    def act(self, obs):
        action = dict(self.base.act(obs))
        action["set_wage"] = obs["market_wage"] * self.ratio
        return action


def worths(agent_factory, seeds, difficulty, profile, spill=None):
    return [run_episode(agent_factory(), difficulty, s, spillover_fraction=spill,
                        npc_profile=profile, lobbying_enabled=False,
                        npc_growth=not LEGACY)["net_worth"]
            for s in seeds]


def main():
    cfg = load_config(DEFAULT_CONFIG)
    diff = cfg["difficulty"]
    ranges = {"default": cfg["seeds"]["validation"], "heldout": cfg["seeds"]["validation_heldout"]}
    lines, frozen = [], {}

    def out(s=""):
        print(s)
        lines.append(s)

    for profile, (lo, hi) in ranges.items():
        seeds = list(range(lo, hi))
        out(f"VALIDATION -- profile={profile}, seeds {lo}..{hi - 1}, lobbying off")
        rows = {}
        for name, cls in AGENTS.items():
            w = worths(cls, seeds, diff, profile)
            rows[name] = {"mean": statistics.mean(w), "std": statistics.stdev(w), "ci": ci95(w),
                          "values": w}
        best = max(rows, key=lambda k: rows[k]["mean"])
        for name in sorted(rows, key=lambda k: -rows[k]["mean"]):
            r = rows[name]
            mark = "  <- benchmark" if name == best else ""
            out(f"  {AGENTS[name].label:16s} {r['mean']:10,.0f} +/- {r['ci']:6,.0f}  (std {r['std']:7,.0f}){mark}")
        frozen[profile] = {
            "agent": best, "label": AGENTS[best].label, "seeds": [lo, hi],
            "mean_net_worth": rows[best]["mean"], "ci": rows[best]["ci"],
            "hold_mean_net_worth": rows["hold"]["mean"],
            "all_means": {k: v["mean"] for k, v in rows.items()},
        }
        out()

    out("Premium pricer vs Rule-based (paired, validation default seeds)")
    seeds = list(range(*ranges["default"]))
    margins = {}
    for spill in (None, 0.0):
        para = worths(AGENTS["spill_parasite"], seeds, diff, "default", spill)
        rule = worths(AGENTS["rule_based"], seeds, diff, "default", spill)
        p = paired(para, rule)
        default_spill = SPILLOVER_FRACTION_V1 if LEGACY else SPILLOVER_FRACTION
        label = f"spill={default_spill} (default)" if spill is None else f"spill={spill}"
        margins[label] = p
        out(f"  {label}: pricer {statistics.mean(para):9,.0f}  rule {statistics.mean(rule):9,.0f}"
            f"  diff {p['mean']:+8,.0f} +/- {p['ci']:6,.0f}  (pricer wins {p['win_rate']:.0%})")
    out()

    out("Wage vs profit: fixed wage/market ratio sweeps (validation default, first 100 seeds)")
    ratios = [0.90, 0.95, 0.97, 1.00, 1.03, 1.05, 1.10, 1.15, 1.20]
    sweep_seeds = seeds[:100]
    sweep = {}
    for base in ("rule_based", "spill_parasite"):
        xs, ys, means = [], [], []
        for r in ratios:
            w = worths(lambda: WageOverride(AGENTS[base](), r), sweep_seeds, diff, "default")
            xs += [r] * len(w)
            ys += w
            means.append(statistics.mean(w))
        sweep[base] = {"ratios": ratios, "mean_net_worth": means, "pearson": pearson(xs, ys)}
        out(f"  {AGENTS[base].label:16s} " + "  ".join(f"{r:.2f}:{m / 1000:5.1f}k" for r, m in zip(ratios, means))
            + f"   r = {sweep[base]['pearson']:+.2f}")
    # Step-level: Random agent, wage/market vs that step's net cash flow
    from simulator import EconSimulator
    xs, ys = [], []
    for s in sweep_seeds:
        sim = EconSimulator(diff, seed=s, lobbying_enabled=False, npc_growth=not LEGACY)
        obs = sim.reset()
        agent = AGENTS["random"]()
        agent.reset(s)
        while not (sim.terminated or sim.truncated):
            obs, info = sim.step(agent.act(obs))
            xs.append(info.wage / info.market_wage)
            ys.append(info.net_cash_flow)
    step_r = pearson(xs, ys)
    out(f"  Random agent, per step: corr(wage/market, net cash flow) = {step_r:+.2f}  ({len(xs)} steps)")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "benchmark.json"), "w") as f:
        json.dump({"frozen_benchmark": frozen, "pricer_vs_rule": margins,
                   "wage_profit": {"sweeps": sweep, "random_step_pearson": step_r}}, f, indent=2)
    with open(os.path.join(OUT, "benchmark_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    out(f"\nFrozen -> {os.path.join(OUT, 'benchmark.json')}")


if __name__ == "__main__":
    main()
