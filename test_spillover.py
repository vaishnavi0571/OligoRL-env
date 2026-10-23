"""
Demand Spillover + Training Solvency Tests
==========================================
Run:  python test_spillover.py
"""

import sys
from simulator import EconSimulator
from reward_integrator import compute_step_reward
from rewards import wage_runway_score, activity_factor
from config import (
    INITIAL_WORKER_COUNT, WORKER_PRODUCTIVITY, MIN_WAGE,
    WAGE_RUNWAY_TARGET_STEPS, WAGE_RUNWAY_CAP_MULT, ACTIVITY_REFERENCE_UNITS,
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


def big_player_sim(frac=None, seed=42):
    """Player holds 10 workers (whole pool) -> spare capacity at $11."""
    s = EconSimulator(difficulty="easy", seed=seed, spillover_fraction=frac)
    s.reset()
    s.player.worker_count += s.unemployed
    s.unemployed = 0
    return s


# ============================================================
heading("SPILLOVER: switch and baseline")
# ============================================================

def trajectory(frac, policy):
    s = EconSimulator(difficulty="easy", seed=42, spillover_fraction=frac)
    s.reset()
    out = []
    for t in range(30):
        obs, info = s.step(policy(t))
        out.append((round(obs["firm_cash"], 6), info.spill_pool))
    return out

hold = lambda t: {"set_price": 10.0}
check("Default play: everyone capacity-bound, nothing spills",
      all(sp == 0 for _, sp in trajectory(0.4, hold)))
check("Default play identical with spillover on/off",
      [c for c, _ in trajectory(0.4, hold)] == [c for c, _ in trajectory(0.0, hold)])

sim = big_player_sim(frac=0.0)
_, info = sim.step({"set_price": 11.0})
check("Fraction 0.0 disables spillover (ablation)", info.spill_pool == 0.0)


# ============================================================
heading("SPILLOVER: allocation and guards")
# ============================================================

sim = big_player_sim()
_, info = sim.step({"set_price": 11.0})
spare = info.max_output - (info.units_sold - info.spill_units)
check("Firm with spare capacity receives spilled customers",
      info.spill_units > 0, f"spill = {info.spill_units:.0f} units")
check("Spill never exceeds spare capacity", info.spill_units <= spare + 1e-6,
      f"spill = {info.spill_units:.0f}, spare = {spare:.0f}")
check("Units sold never exceed capacity", info.units_sold <= info.max_output + 1e-6)

sim = EconSimulator(difficulty="easy", seed=42)
sim.reset()
_, info = sim.step({"set_price": 11.0, "hire_workers": 5})
check("Hiring lag: workers hired this step don't serve spill",
      info.spill_units == 0, f"spill = {info.spill_units:.0f}")
_, info = sim.step({"set_price": 11.0})
check("...but they do next step", info.spill_units > 0,
      f"spill = {info.spill_units:.0f}")

sim = big_player_sim()
sim.player.cash = 100.0   # cannot cover a 10-worker wage bill
_, info = sim.step({"set_price": 11.0})
check("Firm that can't cover its wage bill gets no spill", info.spill_units == 0)

sim = big_player_sim()
_, info = sim.step({"set_price": 50.0})
check("Turned-away customers won't pay a $50 price",
      info.spill_units < 1.0, f"spill = {info.spill_units:.2f} units")

sim = big_player_sim()
_, info = sim.step({"set_price": 11.0})
check("Total spill <= fraction x unserved demand",
      info.spill_pool <= 0.4 * 4 * 2000 + 1e-6, f"pool = {info.spill_pool:.0f}")

sim = EconSimulator(difficulty="hard", seed=3)
sim.reset()
money_ok = True
total_before = sum(sim._build_observation()["agent_wealth"])
for t in range(60):
    obs, info = sim.step({"set_price": 11.0 if t % 2 else 9.0,
                          "hire_workers": 2 if t < 4 else 0})
    total_after = sum(obs["agent_wealth"])
    expected = (info.total_revenue_all - info.total_input_cost_all - info.total_tax_all
                - info.total_capex_all - info.total_hiring_all - info.worker_consumption)
    money_ok &= abs((total_after - total_before) - expected) < 0.1
    total_before = total_after
    if sim.terminated:
        break
check("Money conservation holds with spillover", money_ok)


# ============================================================
heading("TRAINING SOLVENCY: wage runway x activity")
# ============================================================

target = WAGE_RUNWAY_TARGET_STEPS
cap = WAGE_RUNWAY_CAP_MULT * target
s_mid, r_mid = wage_runway_score(2 * 2500.0, 2500.0, target)
check("Runway = cash / wage bill", abs(r_mid - 2.0) < 1e-9)
check("Score = 0.5 at target runway", abs(s_mid - 0.5) < 1e-9)
s_rich, r_rich = wage_runway_score(1e9, 2500.0, target)
check("Runway capped (no reward for hoarding cash)", r_rich == cap)
s_cap, _ = wage_runway_score(cap * 2500.0, 2500.0, target)
check("Score at cap == score with infinite cash", abs(s_rich - s_cap) < 1e-12)
s_broke, _ = wage_runway_score(-500.0, 2500.0, target)
check("Broke firm scores near 0", s_broke < 0.06, f"S = {s_broke:.3f}")

check("Activity = 1 at baseline output", activity_factor(ACTIVITY_REFERENCE_UNITS) == 1.0)
check("Activity capped at 1", activity_factor(5 * ACTIVITY_REFERENCE_UNITS) == 1.0)
check("Activity = 0 when nothing sold", activity_factor(0.0) == 0.0)

# Shell firm: cut wages until every worker leaves, keep the cash
sim = EconSimulator(difficulty="easy", seed=42)
sim.reset()
for t in range(30):
    obs, info = sim.step({"set_price": 10.0, "set_wage": MIN_WAGE})
    rr = compute_step_reward(info, sim)
check("Shell firm (no workers) gets ~0 solvency", obs["worker_count"] == 0
      and rr.solvency < 0.01, f"workers = {obs['worker_count']}, S = {rr.solvency:.3f}")
check("Shell firm feedback says it is barely operating", "BARELY OPERATING" in rr.feedback)

# Units SOLD, not produced: idle workers earn no credit
sim = EconSimulator(difficulty="easy", seed=42, spillover_fraction=0.0)
sim.reset()
_, info = sim.step({"set_price": 40.0})
rr = compute_step_reward(info, sim)
check("Overpriced firm with idle workers gets low activity",
      rr.activity < 0.05, f"activity = {rr.activity:.3f}, units sold = {info.units_sold:.0f}")

sim = EconSimulator(difficulty="easy", seed=42)
sim.reset()
for t in range(10):
    _, info = sim.step({"set_price": 10.0})
rr = compute_step_reward(info, sim)
check("Healthy operating firm: full activity, high solvency",
      rr.activity == 1.0 and rr.solvency > 0.9, f"S = {rr.solvency:.3f}")


# ============================================================
heading("RESULTS")
# ============================================================
total = passed + failed
print(f"\n  {passed}/{total} checks passed, {failed} failed.\n")
sys.exit(1 if failed else 0)
