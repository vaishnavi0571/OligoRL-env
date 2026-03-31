"""
Phase 6 -- Grader Alignment Tests
===================================
Verifies that the dense reward signal provided by the environment
correlates properly with the expected sparse metrics (survival time, market share).

Any divergence here means the RL agent will learn the wrong behavior
despite 'technically' maximizing the dense reward.

Run: python test_grader_alignment.py
"""

import sys
import math
from simulator import EconSimulator
from reward_integrator import compute_step_reward
from config import DEFAULT_PRICE

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
heading("TEST 1: Easy Mode (Survival Alignment)")
# ============================================================
# Agent A: "Always Survive" (healthy pricing)
# Agent B: "Suicide" (sells below cost, bankrupts early)
#
# Hypothesis: Agent A's cumulative dense reward should be MUCH higher
# because the Easy preset strongly weights the Solvency component.

sim_a = EconSimulator(difficulty="easy", seed=42)
sim_a.reset()
reward_sum_a = 0.0
survived_a = 0

for _ in range(30):
    obs, info = sim_a.step({"set_price": 10.0})
    r = compute_step_reward(info, sim_a)
    reward_sum_a += r.composite
    survived_a += 1
    if sim_a.terminated: break

sim_b = EconSimulator(difficulty="easy", seed=42)
sim_b.reset()
reward_sum_b = 0.0
survived_b = 0

for _ in range(30):
    # Selling below cost guarantees rapid bankruptcy
    obs, info = sim_b.step({"set_price": 1.0})
    r = compute_step_reward(info, sim_b)
    reward_sum_b += r.composite
    survived_b += 1
    if sim_b.terminated: break

check("Agent A survived longer than B", survived_a > survived_b,
      f"A={survived_a} steps, B={survived_b} steps")
check("Agent A cumulative dense reward > Agent B", reward_sum_a > reward_sum_b,
      f"A_reward_sum={reward_sum_a:.2f}, B_reward_sum={reward_sum_b:.2f}")

sparse_a = survived_a / 30.0
sparse_b = survived_b / 30.0

check("Dense aligns with Sparse (Survival %)", 
      (reward_sum_a > reward_sum_b) == (sparse_a > sparse_b),
      "If sparse is higher, dense must be higher")


# ============================================================
heading("TEST 2: Medium Mode (Market Share / Profit Alignment)")
# ============================================================
# Agent C: Passive (DEFAULT_PRICE)
# Agent D: Aggressive (undercuts competitors slightly, but stays profitable)
#
# Hypothesis: On Medium (profit-heavy), Agent D captures more market share
# and thus generates higher total profit AND higher cumulative dense reward.

sim_c = EconSimulator(difficulty="medium", seed=99)
sim_c.reset()
reward_sum_c = 0.0
market_share_sum_c = 0.0
profit_score_c = 0.0

for _ in range(30):
    obs, info = sim_c.step({"set_price": DEFAULT_PRICE})
    r = compute_step_reward(info, sim_c)
    reward_sum_c += r.composite
    profit_score_c += r.profit
    
    total_market = info.total_revenue_all
    if total_market > 0:
        market_share_sum_c += (info.revenue / total_market)
        
    if sim_c.terminated: break


sim_d = EconSimulator(difficulty="medium", seed=99)
sim_d.reset()
reward_sum_d = 0.0
market_share_sum_d = 0.0
profit_score_d = 0.0

for step_idx in range(30):
    # Undercutting slightly (e.g., 9.5 vs 10.0), stays above input + labor costs
    action_d = {"set_price": 9.5}
    if step_idx == 0:
        action_d["hire_workers"] = 2
    
    obs, info = sim_d.step(action_d)
    r = compute_step_reward(info, sim_d)
    reward_sum_d += r.composite
    profit_score_d += r.profit
    
    total_market = info.total_revenue_all
    if total_market > 0:
        market_share_sum_d += (info.revenue / total_market)
        
    if sim_d.terminated: break


check("Agent D captured MORE market share than C", market_share_sum_d > market_share_sum_c,
      f"D_share={market_share_sum_d/30:.1%}, C_share={market_share_sum_c/30:.1%}")

check("Agent D achieved HIGHER pure profit score", profit_score_d > profit_score_c,
      f"D_profit={profit_score_d:.2f}, C_profit={profit_score_c:.2f}")

check("Agent D earned HIGHER cumulative dense reward on Medium", reward_sum_d > reward_sum_c,
      f"D_reward={reward_sum_d:.2f}, C_reward={reward_sum_c:.2f}")

check("Dense aligns with Sparse (Market Share / Profit)", 
      (reward_sum_d > reward_sum_c) == (market_share_sum_d > market_share_sum_c),
      "Higher market share policy correctly translated to higher dense reward on Medium")


# ============================================================
heading("RESULTS")
# ============================================================
total = passed + failed
print(f"\n  {passed}/{total} Grader Alignment checks passed, {failed} failed.\n")

if failed > 0:
    print("  [ALERT] ALIGNMENT FAILURE: The RL agent will optimize the wrong thing.\n")
    sys.exit(1)
else:
    print("  ALL CHECKS PASSED -- Phase 6 Grader Alignment verified.\n")
    sys.exit(0)
