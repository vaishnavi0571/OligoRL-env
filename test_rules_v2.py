"""
Rules v2 Tests: NPC growth + smooth quits (config 7f)
=====================================================
Run:  python test_rules_v2.py
"""

import math
import random
import sys

from simulator import EconSimulator, quit_rate
from config import (
    LABOR_FORCE_TOTAL, MAX_QUIT_RATE, QUIT_TOLERANCE, QUIT_SENSITIVITY,
    NPC_MIN_WORKERS, WORKER_PRODUCTIVITY, INPUT_COST_PER_UNIT, MIN_WAGE, MAX_WAGE,
    NPC_PROFILES, NPC_PROFILES_V1,
)

passed = 0
failed = 0


def check(label, condition, detail=""):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}]  {label}"
    if detail:
        msg += f"  --  {detail}"
    print(msg)
    if condition:
        passed += 1
    else:
        failed += 1


def heading(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ============================================================
heading("SMOOTH QUITS")
# ============================================================

check("Zero quits at equal wages", quit_rate(0.0, smooth=True) == 0.0)
check("Zero quits when overpaying", quit_rate(-0.2, smooth=True) == 0.0)
check("No free band: a 3% gap now costs quits", quit_rate(0.03, smooth=True) > 0.005,
      f"rate at 3% = {quit_rate(0.03, smooth=True):.4f} (v1: {quit_rate(0.03, smooth=False):.4f})")
check("Small gaps cost little", quit_rate(0.01, smooth=True) < 0.003,
      f"rate at 1% = {quit_rate(0.01, smooth=True):.4f}")
check("Matches v1 beyond the tolerance (10% gap)",
      abs(quit_rate(0.10, smooth=True) - quit_rate(0.10, smooth=False)) < 0.005)
gaps = [i / 1000 for i in range(0, 400)]
rates = [quit_rate(g, smooth=True) for g in gaps]
check("Monotone in the gap", all(b >= a for a, b in zip(rates, rates[1:])))
jumps = max(abs(b - a) for a, b in zip(rates, rates[1:]))
check("Continuous (no kink jump)", jumps < 0.003, f"max step change = {jumps:.4f}")
check("Capped at MAX_QUIT_RATE", max(rates) <= MAX_QUIT_RATE + 1e-12)


# ============================================================
heading("NPC GROWTH RULE")
# ============================================================

def sim_with(seed=0, **kw):
    s = EconSimulator("medium", seed=seed, **kw)
    s.reset()
    return s

sim = sim_with()
for npc in sim.npc_firms:
    npc.growth_cap, npc.growth_fill, npc.growth_buffer, npc.growth_margin = 2, 1.0, 1.0, 0.0
sim.step({"set_price": 10.0})
first_targets = [n.target_workers for n in sim.npc_firms]
check("One-step lag: no demand history -> no change at step 1",
      first_targets == [5, 5, 5, 5], f"targets = {first_targets}")
sim.step({"set_price": 10.0})
sizes = [n.target_workers for n in sim.npc_firms]
check("Unserved demand raises the target, at most `cap` per step",
      all(5 < t <= 7 for t in sizes), f"targets = {sizes}")

sim = sim_with()
for npc in sim.npc_firms:
    npc.growth_cap, npc.growth_margin = 3, 0.0
    npc.cash = 100.0                       # cannot afford a bigger payroll
sim.step({"set_price": 10.0})
sim.step({"set_price": 10.0})
check("Affordability gate: no cash -> no growth",
      all(n.target_workers <= 5 for n in sim.npc_firms),
      f"targets = {[n.target_workers for n in sim.npc_firms]}")

sim = sim_with()
npc = sim.npc_firms[0]
npc.growth_cap = 2
npc.price_anchor = 6.0                     # MRP ~ 300 < wage 500
npc.max_wage_share = 2.0                   # wage cap stays above MRP: keeps paying 500
fired = []
for _ in range(3):
    _, info = sim.step({"set_price": 10.0})
    fired.append(info.firm_fired[1])
mrp = WORKER_PRODUCTIVITY * (npc.price - INPUT_COST_PER_UNIT)
check("Loss-making NPC downsizes by firing (MRP < wage)", sum(fired) > 0 and npc.worker_count < 5,
      f"MRP {mrp:.0f}, wage {npc.wage:.0f}, workers {npc.worker_count}, fired per step {fired}")
check("Downsizing is capped per step", max(fired) <= npc.growth_cap)

sim = sim_with()
for npc in sim.npc_firms:
    npc.growth_cap, npc.price_anchor, npc.max_wage_share = 3, 5.0, 3.0
for _ in range(20):
    if sim.terminated or sim.truncated:
        break
    sim.step({"set_price": 5.0})
check("Never below NPC_MIN_WORKERS",
      all(n.worker_count >= NPC_MIN_WORKERS or not n.active for n in sim.npc_firms),
      f"sizes = {[n.worker_count for n in sim.npc_firms]}")

sim = sim_with(seed=3)
grew = False
for _ in range(30):
    obs, info = sim.step({"set_price": 10.0})
    grew |= max(obs["firm_workers"][1:]) > 5
check("NPCs grow past the old fixed size of 5 under excess demand", grew)

sim = sim_with()
sim.npc_firms[0].growth_cap = 2
sim.npc_firms[0].price_anchor = 6.0
sim.npc_firms[0].max_wage_share = 2.0
charged = False
for _ in range(3):
    _, info = sim.step({"set_price": 10.0})
    if info.firm_fired[1] > 0:
        charged = info.total_hiring_all >= info.firm_fired[1] * 500.0
check("NPC firing costs money (like the player's)", charged)


# ============================================================
heading("PROFILES")
# ============================================================

def inside(v, ranges):
    return any(lo <= v <= hi for lo, hi in ranges)

ok = True
for key, ranges in NPC_PROFILES["heldout"].items():
    train = NPC_PROFILES["train"][key]
    for lo, hi in ranges:
        for v in (lo, hi, (lo + hi) / 2):
            if inside(v, train) and not (v == train[0][0] or v == train[0][1]):
                ok = False
check("v2 heldout ranges lie outside v2 train ranges", ok)
check("Legacy profiles unchanged (round 1)",
      NPC_PROFILES_V1["train"]["price_anchor"] == [(9.0, 11.0)])
for prof in ("train", "heldout"):
    s = sim_with(seed=5, npc_profile=prof)
    vals = [(n.growth_cap, round(n.growth_fill, 3), round(n.price_anchor, 2)) for n in s.npc_firms]
    check(f"{prof} profile draws growth parameters",
          all(inside(n.growth_cap, NPC_PROFILES[prof]["growth_cap"]) for n in s.npc_firms), f"{vals}")


# ============================================================
heading("INVARIANTS UNDER V2 (random play, all profiles)")
# ============================================================

rng = random.Random(1)
workers_ok = flows_ok = money_ok = True
for prof in ("default", "train", "heldout"):
    for seed in range(8):
        sim = EconSimulator("hard", seed=seed, npc_profile=prof)
        obs = sim.reset()
        total_before = sum(obs["agent_wealth"])
        pool_before = obs["unemployed"]
        for t in range(60):
            obs, info = sim.step({
                "set_price": rng.uniform(6.0, 14.0),
                "set_wage": rng.uniform(MIN_WAGE, MAX_WAGE),
                "hire_workers": rng.randint(-3, 6),
            })
            workers_ok &= sum(obs["firm_workers"]) + obs["unemployed"] == LABOR_FORCE_TOTAL
            into = sum(k for s_, d, k in info.labor_flows if d == -1)
            out = sum(k for s_, d, k in info.labor_flows if s_ == -1)
            flows_ok &= obs["unemployed"] - pool_before == into - out
            pool_before = obs["unemployed"]
            total_after = sum(obs["agent_wealth"])
            expected = (info.total_revenue_all - info.total_input_cost_all - info.total_tax_all
                        - info.total_capex_all - info.total_hiring_all - info.worker_consumption)
            money_ok &= abs((total_after - total_before) - expected) < 0.1
            total_before = total_after
            if sim.terminated or sim.truncated:
                break
check("Employed + unemployed = LABOR_FORCE_TOTAL every step", workers_ok)
check("labor_flows explain every pool change (incl. NPC firing)", flows_ok)
check("Money conservation holds (incl. NPC firing costs)", money_ok)


# ============================================================
heading("RESULTS")
# ============================================================
total = passed + failed
print(f"\n  {passed}/{total} checks passed, {failed} failed.\n")
sys.exit(1 if failed else 0)
