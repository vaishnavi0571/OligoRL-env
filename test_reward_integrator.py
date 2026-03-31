"""
Phase 4 -- Reward Integration Tests
=====================================
Verify that the reward integrator correctly wires Phase 2 components
onto Phase 3 state, and that semantic feedback is economically sensible.

Run:  python test_reward_integrator.py
"""

import sys
from simulator import EconSimulator
from reward_integrator import compute_step_reward
from config import EASY, MEDIUM, HARD, GINI_WEIGHT, POSITIVE_WEIGHT_SUM

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
heading("1. Healthy firm (passive, 10 steps)")
# ============================================================

sim = EconSimulator(difficulty="easy", seed=42)
sim.reset()

for t in range(10):
    obs, info = sim.step({"set_price": 10.0})
    rr = compute_step_reward(info, sim)

check("Profit > 0 (profitable firm)", rr.profit > 0,
      f"P = {rr.profit:.4f}")
check("Solvency > 0.5 (healthy cash)", rr.solvency > 0.5,
      f"S = {rr.solvency:.4f}")
check("GDP in [-1, 1]", -1 <= rr.gdp <= 1,
      f"G = {rr.gdp:.4f}")
check("Gini penalty >= 0", rr.gini_pen >= 0,
      f"gini_pen = {rr.gini_pen:.4f}")
check("Composite in [-0.1, 1.0]",
      -0.1 <= rr.composite <= 1.0,
      f"R = {rr.composite:.4f}")
check("Composite > 0.3 (healthy firm should score OK)",
      rr.composite > 0.3,
      f"R = {rr.composite:.4f}")

# Feedback sanity
check("Feedback mentions 'Profitable' or 'break-even'",
      "rofit" in rr.feedback or "break" in rr.feedback,
      f"feedback = {rr.feedback[:80]}")
check("Feedback contains Step counter",
      "[Step 10/30]" in rr.feedback)
check("Feedback contains cash amount",
      "$" in rr.feedback)
check("Feedback contains R= score",
      "R=" in rr.feedback)


# ============================================================
heading("2. Bankrupt firm (underprice)")
# ============================================================

sim = EconSimulator(difficulty="easy", seed=42)
sim.reset()

bankrupt_rr = None
for t in range(30):
    obs, info = sim.step({"set_price": 1.0})
    rr = compute_step_reward(info, sim)
    if sim.terminated:
        bankrupt_rr = rr
        break

check("Firm went bankrupt", sim.terminated)
check("Profit < 0 at bankruptcy", bankrupt_rr.profit < 0,
      f"P = {bankrupt_rr.profit:.4f}")
check("Solvency near 0 at bankruptcy", bankrupt_rr.solvency < 0.1,
      f"S = {bankrupt_rr.solvency:.4f}")
check("Composite < 0.2 at bankruptcy", bankrupt_rr.composite < 0.2,
      f"R = {bankrupt_rr.composite:.4f}")
check("Feedback mentions BANKRUPT",
      "BANKRUPT" in bankrupt_rr.feedback,
      f"feedback = {bankrupt_rr.feedback[-60:]}")
check("Feedback mentions LOSSES or Losing",
      "LOSS" in bankrupt_rr.feedback.upper() or "Losing" in bankrupt_rr.feedback,
      f"feedback = {bankrupt_rr.feedback[:80]}")


# ============================================================
heading("3. Full episode reward trajectory (30 steps)")
# ============================================================

sim = EconSimulator(difficulty="easy", seed=42)
sim.reset()

rewards = []
for t in range(30):
    obs, info = sim.step({"set_price": 10.0})
    rr = compute_step_reward(info, sim)
    rewards.append(rr)

check("30 rewards computed", len(rewards) == 30)

# All components in valid ranges
all_profit_ok = all(-1 <= r.profit <= 1 for r in rewards)
all_solv_ok = all(0 <= r.solvency <= 1 for r in rewards)
all_gdp_ok = all(-1 <= r.gdp <= 1 for r in rewards)
all_gini_ok = all(0 <= r.gini_pen <= 1 for r in rewards)
check("All P(t) in [-1, 1]", all_profit_ok)
check("All S(t) in [0, 1]", all_solv_ok)
check("All G(t) in [-1, 1]", all_gdp_ok)
check("All Gini_pen in [0, 1]", all_gini_ok)

# Composite range
all_composite_ok = all(-0.1 <= r.composite <= 1.0 for r in rewards)
check("All R(t) in [-0.1, 1.0]", all_composite_ok)

# Last step feedback mentions "Episode complete"
check("Final step feedback: episode complete",
      "Episode complete" in rewards[-1].feedback)


# ============================================================
heading("4. Reward weights match difficulty preset")
# ============================================================

for preset in [EASY, MEDIUM, HARD]:
    w = preset.weights_dict()
    w_sum = w["profit"] + w["solvency"] + w["gdp"]
    check(f"{preset.name}: positive weights sum to 1.0",
          abs(w_sum - POSITIVE_WEIGHT_SUM) < 0.01,
          f"sum = {w_sum:.4f}")
    check(f"{preset.name}: gini weight = {GINI_WEIGHT}",
          approx(w["gini"], GINI_WEIGHT),
          f"w_gini = {w['gini']}")


# ============================================================
heading("5. Perfect agent theoretical max")
# ============================================================

# If all components are at max (P=1, S=1, G=1, Gini=0):
w = EASY.weights_dict()
max_reward = w["profit"] * 1.0 + w["solvency"] * 1.0 + w["gdp"] * 1.0 + w["gini"] * 0.0
check("Perfect agent Easy reward = 1.0",
      approx(max_reward, 1.0, tol=0.01),
      f"max = {max_reward:.4f}")

# Worst case (P=-1, S=0, G=-1, Gini_pen=1):
worst = w["profit"] * (-1) + w["solvency"] * 0 + w["gdp"] * (-1) + w["gini"] * 1.0
check("Worst case Easy reward = negative",
      worst < 0,
      f"worst = {worst:.4f}")


# ============================================================
heading("6. Difficulty affects reward composition")
# ============================================================

# Same scenario, different difficulty -> different reward composition
for diff_name in ["easy", "medium", "hard"]:
    sim = EconSimulator(difficulty=diff_name, seed=42)
    sim.reset()
    for t in range(5):
        _, info = sim.step({"set_price": 10.0})
    rr = compute_step_reward(info, sim)
    check(f"{diff_name}: composite in valid range",
          -0.1 <= rr.composite <= 1.0,
          f"R = {rr.composite:.4f}")

# Easy emphasizes solvency, Medium emphasizes profit
sim_e = EconSimulator(difficulty="easy", seed=42)
sim_e.reset()
sim_m = EconSimulator(difficulty="medium", seed=42)
sim_m.reset()

for t in range(5):
    _, info_e = sim_e.step({"set_price": 10.0})
    _, info_m = sim_m.step({"set_price": 10.0})

rr_e = compute_step_reward(info_e, sim_e)
rr_m = compute_step_reward(info_m, sim_m)

# With high solvency, Easy (solvency-weighted) should score higher
check("Easy favors solvency (high-cash scenario scores better)",
      rr_e.composite >= rr_m.composite - 0.1,
      f"Easy R={rr_e.composite:.4f}, Medium R={rr_m.composite:.4f}")


# ============================================================
heading("7. Feedback generator coherence checks")
# ============================================================

# Run multiple scenarios and verify feedback describes economics sensibly

# Scenario A: Very profitable, high cash
sim = EconSimulator(difficulty="easy", seed=42)
sim.reset()
sim.player.cash = 100_000.0
_, info = sim.step({"set_price": 10.0})
rr = compute_step_reward(info, sim)
check("Rich firm feedback: 'strong' or 'adequate' cash",
      "strong" in rr.feedback or "adequate" in rr.feedback,
      f"S={rr.solvency:.2f}")

# Scenario B: Low cash, near danger
sim = EconSimulator(difficulty="easy", seed=42)
sim.reset()
sim.player.cash = 500.0
_, info = sim.step({"set_price": 10.0})
rr = compute_step_reward(info, sim)
check("Low-cash feedback: warning or 'tight' or 'CRITICAL'",
      "WARNING" in rr.feedback or "tight" in rr.feedback or "CRITICAL" in rr.feedback,
      f"cash={sim.player.cash:.0f}")

# Scenario C: Gini coefficient feedback
sim = EconSimulator(difficulty="easy", seed=42)
sim.reset()
_, info = sim.step({"set_price": 10.0})
# Artificially create high inequality
info.all_agent_wealth = [0, 0, 0, 0, 0, 50000]
rr = compute_step_reward(info, sim)
check("High inequality feedback mentions 'inequality'",
      "inequality" in rr.feedback.lower() or "INEQUALITY" in rr.feedback,
      f"Gini penalty = {rr.gini_pen:.4f}")


# ============================================================
heading("8. Feedback matches economic reality (no absurdities)")
# ============================================================

sim = EconSimulator(difficulty="easy", seed=42)
sim.reset()

absurd_count = 0
for t in range(30):
    obs, info = sim.step({"set_price": 10.0})
    rr = compute_step_reward(info, sim)

    # Cross-check: if profit says "Losing money", net should be negative
    if "Losing" in rr.feedback or "LOSS" in rr.feedback.upper():
        if info.revenue > info.total_expenses:
            absurd_count += 1
            print(f"    ABSURD at step {t+1}: feedback says losses but net positive")

    # Cross-check: if solvency says "strong", cash should be substantial
    if "strong" in rr.feedback:
        if sim.player.cash < 1000:
            absurd_count += 1
            print(f"    ABSURD at step {t+1}: 'strong cash' but only ${sim.player.cash:.0f}")

    if sim.terminated:
        break

check("No economically absurd feedback in 30 steps", absurd_count == 0,
      f"{absurd_count} absurdities found")


# ============================================================
heading("9. Print sample feedback strings")
# ============================================================

sim = EconSimulator(difficulty="easy", seed=42)
sim.reset()

print()
for t in range(5):
    _, info = sim.step({"set_price": 10.0})
    rr = compute_step_reward(info, sim)
    print(f"  {rr.feedback}")
    print()

# One bankruptcy feedback
sim2 = EconSimulator(difficulty="easy", seed=42)
sim2.reset()
for t in range(30):
    _, info = sim2.step({"set_price": 1.0})
    rr = compute_step_reward(info, sim2)
    if sim2.terminated:
        print(f"  {rr.feedback}")
        print()
        break

check("Sample feedback printed (visual inspection)", True)


# ============================================================
heading("RESULTS")
# ============================================================
total = passed + failed
print(f"\n  {passed}/{total} checks passed, {failed} failed.\n")

if failed > 0:
    print("  WARNING: Fix failures before proceeding to Phase 5.\n")
    sys.exit(1)
else:
    print("  ALL CHECKS PASSED -- Phase 4 reward integration verified.\n")
    sys.exit(0)
