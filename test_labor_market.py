"""
Labor Market Tests
==================
Run:  python test_labor_market.py
"""

import random
import sys
from simulator import EconSimulator, reservation_share, weighted_median
from config import (
    INITIAL_WORKER_COUNT, NUM_AGENTS, WAGE_PER_WORKER, HIRING_COST_PER_WORKER,
    LABOR_FORCE_TOTAL, MIN_WAGE, MAX_WAGE, WAGE_MAX_CHANGE, MAX_POACH_RATE,
    WORKER_PRODUCTIVITY, INPUT_COST_PER_UNIT,
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


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def heading(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def fresh(seed=42, difficulty="easy"):
    s = EconSimulator(difficulty=difficulty, seed=seed)
    s.reset()
    return s


EMPLOYED = NUM_AGENTS * INITIAL_WORKER_COUNT
POOL0 = LABOR_FORCE_TOTAL - EMPLOYED


# ============================================================
heading("SETUP")
# ============================================================

sim = EconSimulator(difficulty="easy", seed=42)
obs = sim.reset()
check("Unemployed = labor force - employed", obs["unemployed"] == POOL0,
      f"unemployed = {obs['unemployed']}")
check("All firms start at default wage",
      all(w == WAGE_PER_WORKER for w in obs["firm_wages"]))
steps = [n.wage_step for n in sim.npc_firms]
check("NPCs draw different rule parameters (no lockstep)",
      len(set(round(x, 6) for x in steps)) == len(steps),
      f"wage steps = {[round(x, 3) for x in steps]}")


# ============================================================
heading("PURE FUNCTIONS")
# ============================================================

check("Weighted median ignores a tiny extreme firm",
      weighted_median([(500, 20), (1000, 1)]) == 500)
check("Weighted median with no workers -> default",
      weighted_median([(700, 0)]) == WAGE_PER_WORKER)
check("Labor supply: nobody at 0.85x market", reservation_share(425, 500) == 0.0)
check("Labor supply: a third at 0.90x market",
      approx(reservation_share(450, 500), 1 / 3))
check("Labor supply: everyone at market", reservation_share(500, 500) == 1.0)


# ============================================================
heading("EQUAL WAGES: nobody moves (original behaviour)")
# ============================================================

sim = fresh()
moves = 0
for t in range(30):
    obs, info = sim.step({"set_price": 10.0})
    moves += sum(k for _, _, k in info.labor_flows)
check("Zero worker moves over 30 steps", moves == 0, f"moves = {moves}")
check("NPC wages unchanged", all(w == WAGE_PER_WORKER for w in obs["firm_wages"]))
check("Omitting set_wage keeps current wage", obs["wage"] == WAGE_PER_WORKER)


# ============================================================
heading("STICKY WAGES: max change per step")
# ============================================================

sim = fresh()
_, info = sim.step({"set_price": 10.0, "set_wage": MIN_WAGE})
check("Big cut limited to -10% per step",
      approx(info.wage, WAGE_PER_WORKER * (1 - WAGE_MAX_CHANGE)), f"wage = {info.wage}")
sim = fresh()
_, info = sim.step({"set_price": 10.0, "set_wage": MAX_WAGE})
check("Big raise limited to +10% per step",
      approx(info.wage, WAGE_PER_WORKER * (1 + WAGE_MAX_CHANGE)), f"wage = {info.wage}")
for _ in range(12):
    _, info = sim.step({"set_price": 10.0, "set_wage": 99_999.0})
check("Wage clamped to MAX_WAGE eventually", info.wage == MAX_WAGE)
check("Labor cost = workers x own wage",
      approx(info.labor_cost, sim.player.worker_count * MAX_WAGE))


# ============================================================
heading("QUITS: tolerance band, bigger gap -> more quits")
# ============================================================

def quits_over(wage, steps=10, seeds=range(20)):
    total = 0
    for seed in seeds:
        s = fresh(seed)
        for _ in range(steps):
            _, inf = s.step({"set_price": 10.0, "set_wage": wage})
            total += inf.quits
    return total

q_tiny = quits_over(490.0)   # 2% below market: inside tolerance
q_small = quits_over(460.0)  # 8% below
q_big = quits_over(350.0)    # 30% below
check("Gap within tolerance -> no quits", q_tiny == 0, f"quits at $490 = {q_tiny}")
check("Underpaying beyond tolerance causes quits", q_small > 0, f"quits at $460 = {q_small}")
check("Bigger wage gap -> more quits", q_big > q_small,
      f"$350 -> {q_big}, $460 -> {q_small}")

# A firm losing quitters cannot rehire in the same step
sim = fresh()
for _ in range(3):
    sim.step({"set_price": 10.0, "set_wage": 350.0})
blocked_ok = True
for _ in range(10):
    _, info = sim.step({"set_price": 10.0, "set_wage": 350.0, "hire_workers": 5})
    if info.quits > 0 and info.hires_filled > 0:
        blocked_ok = False
check("No same-step quit-and-rehire", blocked_ok)

sim = fresh()
for _ in range(4):
    _, info = sim.step({"set_price": 10.0, "set_wage": MIN_WAGE, "hire_workers": 3})
check("Workers refuse jobs far below market wage",
      info.hires_filled == 0, f"filled = {info.hires_filled}, wage = {info.wage:.0f}")


# ============================================================
heading("HIRING FROM THE POOL: soft wage priority, random ties")
# ============================================================

def scarce_pool_trial(seed, player_wage):
    """NPCs each 2 short (pool = 13); player also wants 10 -> oversubscribed."""
    s = fresh(seed)
    for npc in s.npc_firms:
        npc.worker_count -= 2
        s.unemployed += 2
    s.player.wage = player_wage
    _, inf = s.step({"set_price": 10.0, "hire_workers": 10})
    return inf.hires_filled, sum(inf.firm_hires[1:])

tie = [scarce_pool_trial(seed, 500.0) for seed in range(60)]
tie_share = sum(p for p, _ in tie) / sum(p + n for p, n in tie)
check("Ties are not always won by the player", 0.25 < tie_share < 0.75,
      f"player share at equal wage = {tie_share:.0%}")

eps = [scarce_pool_trial(seed, 505.0) for seed in range(60)]
eps_share = sum(p for p, _ in eps) / sum(p + n for p, n in eps)
check("+1% wage does NOT capture the whole pool (no knife-edge)", eps_share < 0.9,
      f"player share at +1% = {eps_share:.0%}")

hi = [scarce_pool_trial(seed, 550.0) for seed in range(60)]
hi_share = sum(p for p, _ in hi) / sum(p + n for p, n in hi)
check("Higher wage wins MORE of the pool", hi_share > tie_share,
      f"+10% -> {hi_share:.0%} vs tie {tie_share:.0%}")

sim = fresh()
_, info = sim.step({"set_price": 10.0, "hire_workers": 10})
check("Hiring limited by pool when nobody can be poached",
      info.hires_filled == POOL0 and info.unemployed == 0, f"filled = {info.hires_filled}")
check("Hiring cost charged only for filled positions",
      approx(info.hiring_cost, info.hires_filled * HIRING_COST_PER_WORKER))


# ============================================================
heading("POACHING: job-to-job moves")
# ============================================================

def empty_pool_sim(seed=42, player_wage=600.0):
    s = fresh(seed)
    s.unemployed = 0
    s.player.worker_count += POOL0   # keep labor force constant
    s.player.wage = player_wage
    return s

sim = empty_pool_sim(player_wage=600.0)
_, info = sim.step({"set_price": 10.0, "hire_workers": 3})
check("Paying 20% more poaches workers from rivals",
      info.poached_in == 3 and sum(info.firm_poached_out[1:]) == 3,
      f"poached_in = {info.poached_in}")
check("Per-firm poaching cap respected",
      max(info.firm_poached_out) <= int(MAX_POACH_RATE * INITIAL_WORKER_COUNT) + 1)

sim = empty_pool_sim(player_wage=520.0)
_, info = sim.step({"set_price": 10.0, "hire_workers": 3})
check("+4% is below the poaching premium -> nobody switches",
      info.poached_in == 0, f"poached_in = {info.poached_in}")

sim = empty_pool_sim(player_wage=600.0)
_, info = sim.step({"set_price": 10.0, "hire_workers": 3})
victims = [i for i, k in enumerate(info.firm_poached_out) if k > 0 and i > 0]
_, info2 = sim.step({"set_price": 10.0})
raised = all(info2.firm_wages[i] > WAGE_PER_WORKER for i in victims)
check("Poached NPCs raise their wage next step", bool(victims) and raised,
      f"victims = {victims}, wages = {[round(info2.firm_wages[i]) for i in victims]}")


# ============================================================
heading("NPC WAGE RULE: cap, relax")
# ============================================================

sim = fresh(seed=7, difficulty="hard")
cap_ok = True
max_npc_wage = 0.0
for t in range(20):
    obs, info = sim.step({"set_price": 10.0, "set_wage": 700.0,
                          "hire_workers": 3 if sim.player.worker_count < 14 else 0})
    for npc in sim.npc_firms:
        cap = max(npc.max_wage_share * WORKER_PRODUCTIVITY
                  * (npc.price - INPUT_COST_PER_UNIT), MIN_WAGE)
        cap_ok &= npc.wage <= cap + 1e-6
        max_npc_wage = max(max_npc_wage, npc.wage)
    if sim.terminated:
        break
check("NPCs raise wages when competing for workers", max_npc_wage > WAGE_PER_WORKER,
      f"max NPC wage = ${max_npc_wage:.0f}")
check("NPC wages never exceed their marginal-revenue cap", cap_ok)

peak = sum(n.wage for n in sim.npc_firms) / len(sim.npc_firms)
for _ in range(40):
    if sim.terminated or sim.truncated:
        break
    obs, info = sim.step({"set_price": 10.0, "set_wage": MIN_WAGE, "hire_workers": 0})
after = sum(n.wage for n in sim.npc_firms) / len(sim.npc_firms)
check("NPC wages relax once competition stops", after < peak,
      f"avg ${peak:.0f} -> ${after:.0f}")
check("NPC wages never relax below default",
      all(n.wage >= WAGE_PER_WORKER - 1e-6 for n in sim.npc_firms))


# ============================================================
heading("NPC EXIT: bankrupt NPC releases workers")
# ============================================================

sim = fresh()
sim.npc_firms[0].cash = -1e6
obs, info = sim.step({"set_price": 10.0})
check("Bankrupt NPC marked inactive", obs["firm_active"][1] is False)
check("Its workers return to the pool",
      obs["firm_workers"][1] == 0 and obs["unemployed"] == POOL0 + INITIAL_WORKER_COUNT,
      f"unemployed = {obs['unemployed']}")
obs, info = sim.step({"set_price": 10.0, "hire_workers": 5})
check("Released workers can be hired by others", info.hires_filled == 5)
check("Exited NPC no longer sells", info.firm_hires[1] == 0
      and obs["firm_workers"][1] == 0)


# ============================================================
heading("INVARIANTS: workers, flows, money (random play)")
# ============================================================

workers_ok = True
flows_ok = True
money_ok = True
rng = random.Random(0)
for seed in range(15):
    sim = EconSimulator(difficulty="hard", seed=seed)
    obs = sim.reset()
    total_before = sum(obs["agent_wealth"])
    pool_before = obs["unemployed"]
    for t in range(60):
        action = {
            "set_price": rng.uniform(6.0, 14.0),
            "set_wage": rng.uniform(MIN_WAGE, MAX_WAGE),
            "hire_workers": rng.randint(-3, 6),
        }
        obs, info = sim.step(action)
        if sum(obs["firm_workers"]) + obs["unemployed"] != LABOR_FORCE_TOTAL:
            workers_ok = False
        into_pool = sum(k for s, d, k in info.labor_flows if d == -1)
        out_pool = sum(k for s, d, k in info.labor_flows if s == -1)
        if obs["unemployed"] - pool_before != into_pool - out_pool:
            flows_ok = False
        pool_before = obs["unemployed"]
        total_after = sum(obs["agent_wealth"])
        expected_delta = (
            info.total_revenue_all - info.total_input_cost_all
            - info.total_tax_all - info.total_capex_all
            - info.total_hiring_all - info.worker_consumption
        )
        if not approx(total_after - total_before, expected_delta, tol=0.1):
            money_ok = False
        total_before = total_after
        if sim.terminated or sim.truncated:
            break

check("Employed + unemployed = LABOR_FORCE_TOTAL every step", workers_ok)
check("labor_flows exactly explain pool changes", flows_ok)
check("Money conservation holds with NPC hiring", money_ok)


# ============================================================
heading("RESULTS")
# ============================================================
total = passed + failed
print(f"\n  {passed}/{total} checks passed, {failed} failed.\n")
sys.exit(1 if failed else 0)
