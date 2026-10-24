"""
Rules v2 report (validation seeds only)
=======================================
A. Detectors per scripted agent (default + heldout validation)
B. Poacher vs NPC growth (legacy vs v2 trajectories)
C. Why fixed-wage agents are drained (Hold trajectory)
D. Is more than one strategy viable?  (wage-matching variants)
E. H2: held-out as a stress test (drops, harsh subset, bankruptcies)
F. Gap-closed denominators: Benchmark - Hold vs Benchmark - Market+1%

Run:  python -m ppo.v2_report [num_seeds]
"""

import math
import os
import statistics as st
import sys

from simulator import EconSimulator
from agents import AGENTS, terminal_net_worth
from config import LABOR_FORCE_TOTAL
from ppo.benchmark import WageOverride


def ci95(v):
    return 1.96 * st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


def episode(agent, seed, profile="default", npc_growth=True, trace=False):
    sim = EconSimulator("medium", seed=seed, npc_profile=profile, lobbying_enabled=False,
                        npc_growth=npc_growth)
    obs = sim.reset()
    anchors = [n.price_anchor for n in sim.npc_firms]
    agent.reset(seed)
    spill = units = demand = flows = 0.0
    max_emp = 0.0
    steps = []
    while not (sim.terminated or sim.truncated):
        obs, info = sim.step(agent.act(obs))
        spill += info.spill_pool
        units += info.total_units_all
        demand += info.total_demand_all
        flows += sum(k for _, _, k in info.labor_flows)
        max_emp = max(max_emp, max(info.firm_workers) / LABOR_FORCE_TOTAL)
        if trace:
            steps.append({"workers": obs["worker_count"], "market_wage": obs["market_wage"],
                          "wage": obs["wage"], "quits": info.quits, "poached": info.poached_out,
                          "npc_size": st.mean(obs["firm_workers"][1:]),
                          "npc_wage": st.mean(obs["firm_wages"][1:])})
    return {"worth": terminal_net_worth(sim), "bankrupt": sim.terminated,
            "spill_units": spill / units if units else 0.0,
            "spill_demand": spill / demand if demand else 0.0,
            "max_emp": max_emp, "flow": flows / sim.step_count / LABOR_FORCE_TOTAL,
            "npc_exit": sum(not n.active for n in sim.npc_firms) / len(sim.npc_firms),
            "mkt_wage": sim.market_wage,
            "npc_size": st.mean(n.worker_count for n in sim.npc_firms),
            "anchors": anchors, "steps": steps, "len": sim.step_count}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    val = list(range(50000, 50000 + n))
    hval = list(range(60000, 60000 + n))
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    # ---------------- A
    res = {}
    for prof, seeds in (("default", val), ("heldout", hval)):
        out(f"A. Detectors -- rules v2, profile={prof}, validation seeds {seeds[0]}..{seeds[-1]}")
        out(f"  {'agent':16s} {'net worth':>10s} {'+/-':>6s} {'bankrupt':>8s} {'spill/units':>11s}"
            f" {'spill/demand':>12s} {'max emp':>7s} {'flow':>6s} {'NPC exit':>8s} {'mkt wage':>8s} {'NPC size':>8s}")
        for name, cls in AGENTS.items():
            rs = [episode(cls(), s, prof) for s in seeds]
            res[(prof, name)] = rs
            m = lambda k: st.mean(r[k] for r in rs)
            out(f"  {cls.label:16s} {m('worth'):10,.0f} {ci95([r['worth'] for r in rs]):6,.0f}"
                f" {m('bankrupt'):8.0%} {m('spill_units'):11.1%} {m('spill_demand'):12.1%}"
                f" {m('max_emp'):7.0%} {m('flow'):6.3f} {m('npc_exit'):8.1%} {m('mkt_wage'):8.0f} {m('npc_size'):8.1f}")
        out()

    # ---------------- B
    out("B. Poacher vs NPC growth (means over the first 50 validation seeds)")
    for label, growth in (("legacy", False), ("v2", True)):
        rs = [episode(AGENTS["poach_to_spill"](), s, npc_growth=growth, trace=True) for s in val[:50]]
        out(f"  {label}: bankrupt {st.mean(r['bankrupt'] for r in rs):.0%},"
            f" mean bankrupt step {st.mean(r['len'] for r in rs if r['bankrupt']):.1f}")
        for t in (0, 2, 4, 6, 8):
            alive = [r["steps"][t] for r in rs if len(r["steps"]) > t]
            if alive:
                out(f"    step {t + 1:2d}: poacher workers {st.mean(a['workers'] for a in alive):5.1f}"
                    f"  NPC size {st.mean(a['npc_size'] for a in alive):4.1f}"
                    f"  poacher wage {st.mean(a['wage'] for a in alive):4.0f}"
                    f"  NPC wage {st.mean(a['npc_wage'] for a in alive):4.0f}  (n={len(alive)})")
    out()

    # ---------------- C
    out("C. Why fixed-wage agents are drained: Hold under v2 (first 50 validation seeds)")
    rs = [episode(AGENTS["hold"](), s, trace=True) for s in val[:50]]
    for t in (0, 2, 4, 6, 9, 14, 19, 29):
        a = [r["steps"][t] for r in rs]
        gap = st.mean((x["market_wage"] - x["wage"]) / x["market_wage"] for x in a)
        out(f"    step {t + 1:2d}: workers {st.mean(x['workers'] for x in a):4.1f}  market wage"
            f" {st.mean(x['market_wage'] for x in a):4.0f}  gap {gap:5.1%}  quits {st.mean(x['quits'] for x in a):4.2f}"
            f"  poached {st.mean(x['poached'] for x in a):4.2f}  NPC size {st.mean(x['npc_size'] for x in a):4.1f}")
    out()

    # ---------------- D
    out("D. Viable strategies under v2 (default validation, paired vs best)")
    variants = {
        "Rule-based": lambda: AGENTS["rule_based"](),
        "Market + 1% (grow to 12)": lambda: AGENTS["market+eps"](),
        "Hold, wage 1.03x market": lambda: WageOverride(AGENTS["hold"](), 1.03),
        "Premium pricer, wage 1.03x": lambda: WageOverride(AGENTS["spill_parasite"](), 1.03),
        "Market + 1%, wage 1.03x": lambda: WageOverride(AGENTS["market+eps"](), 1.03),
        "Rule-based, wage 1.03x": lambda: WageOverride(AGENTS["rule_based"](), 1.03),
    }
    vres = {k: [episode(f(), s)["worth"] for s in val[:100]] for k, f in variants.items()}
    best = max(vres, key=lambda k: st.mean(vres[k]))
    for k in sorted(vres, key=lambda k: -st.mean(vres[k])):
        d = [a - b for a, b in zip(vres[k], vres[best])]
        rel = st.mean(vres[k]) / st.mean(vres[best])
        out(f"  {k:30s} {st.mean(vres[k]):9,.0f}  {rel:5.0%} of best"
            + ("" if k == best else f"   diff {st.mean(d):+8,.0f} +/- {ci95(d):6,.0f}"))
    viable = [k for k in vres if st.mean(vres[k]) >= 0.9 * st.mean(vres[best])]
    out(f"  Within 10% of best: {len(viable)} -> {viable}")
    out()

    # ---------------- E
    out("E. H2 stress test (validation)")
    for name in ("rule_based", "market+eps", "spill_parasite", "hold"):
        d = st.mean(r["worth"] for r in res[("default", name)])
        h = st.mean(r["worth"] for r in res[("heldout", name)])
        out(f"  {AGENTS[name].label:16s} default {d:9,.0f}  heldout {h:9,.0f}  drop {1 - h / d:+6.1%}"
            f"  heldout bankrupt {st.mean(r['bankrupt'] for r in res[('heldout', name)]):4.0%}")
    rb = res[("heldout", "rule_based")]
    harsh = [r for r in rb if sum(a < 7.0 for a in r["anchors"]) >= 2]
    mild = [r for r in rb if sum(a < 7.0 for a in r["anchors"]) == 0]
    out(f"  Rule-based on heldout subsets: harsh (>=2 price-war NPCs, n={len(harsh)}) "
        f"{st.mean(r['worth'] for r in harsh):9,.0f}  bankrupt {st.mean(r['bankrupt'] for r in harsh):.0%}"
        f" | no price-war NPCs (n={len(mild)}) {st.mean(r['worth'] for r in mild):9,.0f}")
    q = sorted(r["worth"] for r in rb)
    out(f"  Rule-based heldout 10th percentile {q[len(q) // 10]:9,.0f} vs default 10th percentile"
        f" {sorted(r['worth'] for r in res[('default', 'rule_based')])[n // 10]:9,.0f}")
    out()

    # ---------------- F
    out("F. Gap-closed denominators (default validation)")
    bench = [r["worth"] for r in res[("default", "rule_based")]]
    for ref in ("hold", "market+eps"):
        rv = [r["worth"] for r in res[("default", ref)]]
        d = [a - b for a, b in zip(bench, rv)]
        out(f"  Benchmark - {AGENTS[ref].label:12s} = {st.mean(d):9,.0f} +/- {ci95(d):5,.0f}"
            f"  (reference std {st.stdev(rv):6,.0f}, reference bankrupt "
            f"{st.mean(r['bankrupt'] for r in res[('default', ref)]):.0%})")

    os.makedirs(os.path.join("results", "ppo"), exist_ok=True)
    with open(os.path.join("results", "ppo", "v2_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
