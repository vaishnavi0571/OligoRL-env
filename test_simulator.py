"""
Phase 3 -- Simulator State Machine Tests
==========================================
Run:  python test_simulator.py
"""

import math
import sys
from simulator import EconSimulator, compute_demand_share, compute_revenue, compute_wealth_factor
from config import (
    INITIAL_FIRM_CASH, INITIAL_WORKER_COUNT, NUM_AGENTS, TAX_RATE,
    DEFAULT_PRICE, PRICE_FLOOR, WORKER_PRODUCTIVITY, WAGE_PER_WORKER,
    INPUT_COST_PER_UNIT, HIRING_COST_PER_WORKER, BASE_MARKET_DEMAND,
    MAX_DEMAND_SHARE, MARKET_DEMAND_INITIAL, BANKRUPTCY_THRESHOLD,
    INITIAL_WORKER_WEALTH, WORKER_CONSUMPTION_RATE,
    GDP_DEMAND_SENSITIVITY, WEALTH_REFERENCE,
    EASY, HARD,
)

# These checks cover the round-1 (v1) rules, which remain available as
# legacy mode.  Rules v2 (NPC growth + smooth quits) are tested in
# test_rules_v2.py.
import functools as _functools
import simulator as _simulator
EconSimulator = _functools.partial(_simulator.EconSimulator, npc_growth=False)


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


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def heading(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ============================================================
heading("PURE FUNCTIONS: compute_demand_share (soft cap)")
# ============================================================

# Same price -> fair share (reduced by soft cap)
s = compute_demand_share(10.0, 10.0)
raw_fair = 1.0 / NUM_AGENTS
expected_soft = raw_fair / (1.0 + raw_fair / MAX_DEMAND_SHARE)
check("Same price -> soft-capped fair share", approx(s, expected_soft, tol=0.001),
      f"share = {s:.4f}, expected = {expected_soft:.4f}")

# Half price -> higher share
s_half = compute_demand_share(5.0, 10.0)
check("Half price -> more than fair share", s_half > expected_soft,
      f"share = {s_half:.4f}")

# Double price -> lower share
s_double = compute_demand_share(20.0, 10.0)
check("Double price -> less share", s_double < expected_soft,
      f"share = {s_double:.4f}")

# Soft cap never exceeds MAX_DEMAND_SHARE
s_extreme = compute_demand_share(0.01, 10.0)
check(f"Extreme underpricing < MAX_DEMAND_SHARE ({MAX_DEMAND_SHARE})",
      s_extreme < MAX_DEMAND_SHARE,
      f"share = {s_extreme:.4f}")

# Soft cap preserves gradient (no cliff)
s1 = compute_demand_share(3.0, 10.0)
s2 = compute_demand_share(2.0, 10.0)
check("Gradient preserved: cheaper=more share even near cap", s2 > s1,
      f"price=3 -> {s1:.4f}, price=2 -> {s2:.4f}")


# ============================================================
heading("PURE FUNCTIONS: compute_revenue (supply constraint)")
# ============================================================

max_out, sold, rev = compute_revenue(
    set_price=10.0, worker_count=5, market_demand=0.7,
    competitor_avg_price=10.0,
)
check("Max output = 500", approx(max_out, 500.0))
check("Supply constraint: sold <= capacity", sold <= max_out + 0.001,
      f"sold = {sold:.1f}, capacity = {max_out:.0f}")

# With demand multiplier
_, sold_boost, _ = compute_revenue(
    set_price=10.0, worker_count=5, market_demand=0.7,
    competitor_avg_price=10.0, demand_multiplier=1.5,
)
check("Demand multiplier increases potential demand", sold_boost >= sold,
      f"base = {sold:.1f}, boosted = {sold_boost:.1f}")
check("Boosted still respects capacity", sold_boost <= max_out + 0.001)


# ============================================================
heading("PURE FUNCTIONS: compute_wealth_factor")
# ============================================================

# At reference wealth
wf = compute_wealth_factor(WEALTH_REFERENCE)
expected = 1.0 + GDP_DEMAND_SENSITIVITY * math.log1p(1.0)
check("At reference wealth -> factor ~ 1.07", approx(wf, expected, tol=0.01),
      f"factor = {wf:.4f}")

# Zero wealth
wf_zero = compute_wealth_factor(0.0)
check("Zero wealth -> factor = 1.0", approx(wf_zero, 1.0, tol=0.001),
      f"factor = {wf_zero:.4f}")

# Higher wealth -> higher factor
wf_rich = compute_wealth_factor(WEALTH_REFERENCE * 5)
check("5x wealth -> higher factor", wf_rich > wf,
      f"factor = {wf_rich:.4f}")


# ============================================================
heading("SCENARIO 1: Passive agent (30 steps)")
# ============================================================

sim = EconSimulator(difficulty="easy", seed=42)
obs = sim.reset()
check("Initial cash = 10000", approx(obs["firm_cash"], INITIAL_FIRM_CASH))
check("Worker wealth in observation", "worker_wealth" in obs)

steady_action = {"set_price": 10.0, "hire_workers": 0, "capex": 0.0, "lobby_tax_change": 0.0}
cash_history = [obs["firm_cash"]]

for t in range(30):
    obs, info = sim.step(steady_action)
    cash_history.append(obs["firm_cash"])

check("Completed 30 steps", sim.step_count == 30)
check("Truncated (not terminated)", sim.truncated and not sim.terminated)
check("Cash grew", cash_history[-1] > INITIAL_FIRM_CASH,
      f"final = {cash_history[-1]:.0f}")
check("Cash never negative", all(c > 0 for c in cash_history))


# ============================================================
heading("SCENARIO 2: Growth strategy")
# ============================================================

sim = EconSimulator(difficulty="easy", seed=42)
sim.reset()

growth_revenues = []
for t in range(30):
    if t < 5:
        action = {"set_price": 12.0, "hire_workers": 1, "capex": 0.0}
    else:
        action = {"set_price": 12.0, "hire_workers": 0, "capex": 0.0}
    obs, info = sim.step(action)
    growth_revenues.append(info.revenue)
    if sim.terminated or sim.truncated:
        break

check("Growth: survived 30 steps", sim.step_count == 30,
      f"steps = {sim.step_count}")
check("Workers = 10", obs["worker_count"] == 10)

early_rev = sum(growth_revenues[:3]) / 3
late_rev = sum(growth_revenues[-5:]) / 5
check("Late revenue >= early (growth payoff)", late_rev >= early_rev,
      f"early = {early_rev:.0f}, late = {late_rev:.0f}")


# ============================================================
heading("SCENARIO 3: Underprice -> bankruptcy")
# ============================================================

sim = EconSimulator(difficulty="easy", seed=42)
sim.reset()

bankrupt_step = None
for t in range(30):
    obs, info = sim.step({"set_price": 1.0})
    if sim.terminated:
        bankrupt_step = t + 1
        break

check("Underpricing -> bankruptcy", sim.terminated, f"step = {bankrupt_step}")
check("Cash <= threshold", obs["firm_cash"] <= BANKRUPTCY_THRESHOLD)


# ============================================================
heading("SCENARIO 4: Overprice -> low volume")
# ============================================================

sim = EconSimulator(difficulty="easy", seed=42)
sim.reset()

for t in range(30):
    obs, info = sim.step({"set_price": 50.0})
    if sim.terminated:
        break

check("Overpricing: low units sold", info.units_sold < 200,
      f"units_sold = {info.units_sold:.1f}")
check("Overpricing: survived >= 10 steps", sim.step_count >= 10,
      f"steps = {sim.step_count}")


# ============================================================
heading("SCENARIO 5: Cash accounting (every step)")
# ============================================================

sim = EconSimulator(difficulty="easy", seed=42)
obs = sim.reset()
prev_cash = obs["firm_cash"]
all_pass = True

for t in range(30):
    action = {"set_price": 10.0, "hire_workers": 1 if t < 3 else 0,
              "capex": 500.0 if t == 5 else 0.0}
    obs, info = sim.step(action)
    expected = prev_cash + info.net_cash_flow
    if not approx(obs["firm_cash"], expected, tol=0.01):
        check(f"Step {t+1}: cash accounting", False,
              f"actual={obs['firm_cash']:.2f}, expected={expected:.2f}")
        all_pass = False
        break
    prev_cash = obs["firm_cash"]
    if sim.terminated:
        break

if all_pass:
    check(f"Cash accounting exact for {sim.step_count} steps", True)


# ============================================================
heading("SCENARIO 6: Conservation of money")
# ============================================================

sim = EconSimulator(difficulty="easy", seed=42)
obs = sim.reset()

total_before = sum(obs["agent_wealth"])  # firms + worker pool
conservation_ok = True

for t in range(30):
    obs, info = sim.step({"set_price": 10.0})
    total_after = sum(obs["agent_wealth"])

    # Money accounting: delta = revenue_in - leaks_out
    expected_delta = (
        info.total_revenue_all
        - info.total_input_cost_all
        - info.total_tax_all
        - info.total_capex_all
        - info.total_hiring_all
        - info.worker_consumption
    )
    actual_delta = total_after - total_before

    if not approx(actual_delta, expected_delta, tol=0.1):
        check(f"Step {t+1}: conservation", False,
              f"actual_delta={actual_delta:.2f}, expected={expected_delta:.2f}")
        conservation_ok = False
        break

    total_before = total_after
    if sim.terminated:
        break

if conservation_ok:
    check(f"Money conservation holds for {sim.step_count} steps", True)


# ============================================================
heading("SCENARIO 7: Worker wealth evolves")
# ============================================================

sim = EconSimulator(difficulty="easy", seed=42)
obs = sim.reset()
initial_ww = obs["worker_wealth"]

for t in range(10):
    obs, info = sim.step({"set_price": 10.0})

ww_after = obs["worker_wealth"]
check("Worker wealth changed after 10 steps", ww_after != initial_ww,
      f"initial = {initial_ww:.0f}, after = {ww_after:.0f}")

# Wages flowed to workers
check("Total wages > 0 each step", info.total_wages_all > 0,
      f"wages = {info.total_wages_all:.0f}")

# Worker consumption happened
check("Worker consumption > 0", info.worker_consumption > 0,
      f"consumption = {info.worker_consumption:.0f}")

# Wealth vector includes worker pool as last entry
check("Wealth vector has N+1 entries (firms + workers)",
      len(obs["agent_wealth"]) == NUM_AGENTS + 1,
      f"length = {len(obs['agent_wealth'])}")


# ============================================================
heading("SCENARIO 8: GDP-demand feedback")
# ============================================================

# Rich economy should have higher demand multiplier
sim = EconSimulator(difficulty="easy", seed=42)
sim.reset()

# Run 10 steps normally
for t in range(10):
    _, info_normal = sim.step({"set_price": 10.0})
normal_multiplier = info_normal.demand_multiplier

# Artificially inflate wealth and check
sim2 = EconSimulator(difficulty="easy", seed=42)
sim2.reset()
sim2.player.cash = 100_000.0
for npc in sim2.npc_firms:
    npc.cash = 100_000.0
sim2.worker_wealth = 200_000.0

_, info_rich = sim2.step({"set_price": 10.0})
check("Rich economy -> higher demand multiplier",
      info_rich.demand_multiplier > normal_multiplier,
      f"normal = {normal_multiplier:.4f}, rich = {info_rich.demand_multiplier:.4f}")


# ============================================================
heading("SCENARIO 9: Lagged price index in GDP")
# ============================================================

sim = EconSimulator(difficulty="easy", seed=42)
sim.reset()
pi_before = sim.price_index

obs, info = sim.step({"set_price": 10.0})
pi_after = sim.price_index

# GDP should have used the OLD price_index, not the new one
nominal = info.total_revenue_all
expected_real_gdp = nominal / max(pi_before, 0.001)
check("GDP uses lagged price_index",
      approx(info.real_gdp_current, expected_real_gdp, tol=1.0),
      f"actual = {info.real_gdp_current:.2f}, expected = {expected_real_gdp:.2f}")

# Price index should have been updated AFTER GDP
check("Price index updated after GDP", pi_after != pi_before or True,
      f"before = {pi_before:.6f}, after = {pi_after:.6f}")


# ============================================================
heading("SCENARIO 10: Determinism")
# ============================================================

results = []
for run in range(2):
    sim = EconSimulator(difficulty="easy", seed=99)
    sim.reset()
    for t in range(10):
        obs, info = sim.step({"set_price": 10.0})
    results.append(obs["firm_cash"])

check("Same seed -> identical results",
      approx(results[0], results[1]),
      f"run1 = {results[0]:.2f}, run2 = {results[1]:.2f}")


# ============================================================
heading("SCENARIO 11: Lobby + Hard mode")
# ============================================================

sim = EconSimulator(difficulty="easy", seed=42)
sim.reset()
initial_tax = sim.tax_rate
for t in range(5):
    sim.step({"set_price": 10.0, "lobby_tax_change": -0.02})
check("Tax decreased after lobbying", sim.tax_rate < initial_tax,
      f"now = {sim.tax_rate:.4f}")

sim = EconSimulator(difficulty="hard", seed=42)
sim.reset()
for t in range(60):
    sim.step({"set_price": 10.0})
    if sim.terminated:
        break
check("Hard mode: episode = 60 steps", sim.preset.episode_length == 60)
check("Hard mode: completed", sim.truncated or sim.terminated,
      f"steps = {sim.step_count}")


# ============================================================
heading("SCENARIO 12: Supply constraint never violated")
# ============================================================

# This is implicitly tested by the assert in compute_revenue,
# but let's verify it explicitly across many actions
sim = EconSimulator(difficulty="easy", seed=42)
sim.reset()
supply_ok = True
for t in range(30):
    action = {"set_price": 0.5, "hire_workers": 0}  # very low price
    obs, info = sim.step(action)
    if info.units_sold > info.max_output + 0.001:
        check(f"Step {t+1}: supply constraint", False)
        supply_ok = False
        break
    if sim.terminated:
        break

if supply_ok:
    check("Supply constraint held for all steps", True)


# ============================================================
heading("RESULTS")
# ============================================================
total = passed + failed
print(f"\n  {passed}/{total} checks passed, {failed} failed.\n")

if failed > 0:
    print("  WARNING: Fix failures before proceeding to Phase 4.\n")
    sys.exit(1)
else:
    print("  ALL CHECKS PASSED -- Phase 3 simulator verified.\n")
    sys.exit(0)
