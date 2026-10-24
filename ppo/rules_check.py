"""
Rules check: scripted suite + market detectors, legacy vs v2
============================================================
Validation seeds only.  For every scripted agent and rule set reports
final net worth and these detectors (averaged over episodes):
  spill share    units sold via spillover / all units sold (all firms)
  max emp share  largest share of the labor force held by one firm
  flow rate      workers moving per step / labor force
  bankrupt       player bankruptcy rate
  NPC exits      share of NPCs that went bankrupt
  mkt wage end   market (median) wage at the last step
  NPC size end   mean NPC headcount at the last step

Run:  python -m ppo.rules_check [num_seeds]
"""

import os
import statistics
import sys

from simulator import EconSimulator
from agents import AGENTS, terminal_net_worth
from config import LABOR_FORCE_TOTAL

RULES = {"legacy": {"npc_growth": False}, "v2": {}}


def episode(name, seed, profile, sim_kwargs):
    sim = EconSimulator("medium", seed=seed, npc_profile=profile, lobbying_enabled=False,
                        **sim_kwargs)
    obs = sim.reset()
    agent = AGENTS[name]()
    agent.reset(seed)
    spill = units = flows = 0.0
    max_emp = 0.0
    while not (sim.terminated or sim.truncated):
        obs, info = sim.step(agent.act(obs))
        spill += info.spill_pool
        units += info.total_units_all
        flows += sum(k for _, _, k in info.labor_flows)
        max_emp = max(max_emp, max(info.firm_workers) / LABOR_FORCE_TOTAL)
    npcs = sim.npc_firms
    return {
        "worth": terminal_net_worth(sim), "bankrupt": sim.terminated,
        "spill_share": spill / units if units else 0.0, "max_emp": max_emp,
        "flow_rate": flows / sim.step_count / LABOR_FORCE_TOTAL,
        "npc_exit": sum(not n.active for n in npcs) / len(npcs),
        "mkt_wage": sim.market_wage,
        "npc_size": statistics.mean(n.worker_count for n in npcs),
    }


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    cases = [("legacy", "default", 50000), ("v2", "default", 50000),
             ("v2", "train", 70000), ("v2", "heldout", 60000)]
    for rules, profile, base in cases:
        seeds = range(base, base + n)
        out(f"rules={rules}  profile={profile}  seeds {base}..{base + n - 1}")
        out(f"  {'agent':16s} {'net worth':>10s} {'bankrupt':>8s} {'spill':>6s} {'max emp':>7s}"
            f" {'flow':>6s} {'NPC exit':>8s} {'mkt wage':>8s} {'NPC size':>8s}")
        for name, cls in AGENTS.items():
            rs = [episode(name, s, profile, RULES[rules]) for s in seeds]
            m = lambda k: statistics.mean(r[k] for r in rs)
            out(f"  {cls.label:16s} {m('worth'):10,.0f} {m('bankrupt'):8.0%} {m('spill_share'):6.1%}"
                f" {m('max_emp'):7.0%} {m('flow_rate'):6.3f} {m('npc_exit'):8.0%}"
                f" {m('mkt_wage'):8.0f} {m('npc_size'):8.1f}")
        out()
    os.makedirs(os.path.join("results", "ppo"), exist_ok=True)
    with open(os.path.join("results", "ppo", "rules_check.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
