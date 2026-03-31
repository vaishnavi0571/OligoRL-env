"""
Phase 2 -- Reward Component Tests
===================================
Feeds known inputs to each reward function and verifies the outputs
land in the expected ranges.  Acts as the safety net for later debugging.

Run:  python test_rewards.py
"""

import math
import sys
from rewards import (
    profit_score, solvency_score, gdp_score, gini_penalty,
    update_ema, discrete_gini,
)
from config import (
    REVENUE_FLOOR, REVENUE_EMA_INITIAL, IHS_EMA_DECAY,
    CAPEX_AMORTIZATION_STEPS, IHS_MAX_RATIO,
    SOLVENCY_STEEPNESS, BURN_RATE_EPSILON,
    GDP_FLOOR_EPSILON,
    GINI_THRESHOLD, GINI_EXPONENT, GINI_WEALTH_FLOOR,
    EASY, HARD,
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


# ============================================================
heading("HELPERS: update_ema")
# ============================================================

# First step: ema starts at REVENUE_EMA_INITIAL
ema = update_ema(REVENUE_EMA_INITIAL, 1000.0, IHS_EMA_DECAY)
expected = IHS_EMA_DECAY * REVENUE_EMA_INITIAL + (1 - IHS_EMA_DECAY) * 1000.0
check("EMA step 1 from cold start", approx(ema, expected),
      f"ema = {ema:.2f}, expected = {expected:.2f}")

# Converges toward input after many steps
ema = REVENUE_EMA_INITIAL
for _ in range(100):
    ema = update_ema(ema, 2000.0, IHS_EMA_DECAY)
check("EMA converges to input after 100 steps", approx(ema, 2000.0, tol=1.0),
      f"ema = {ema:.2f}")


# ============================================================
heading("HELPERS: discrete_gini")
# ============================================================

# Perfect equality
check("Gini of equal values = 0",
      discrete_gini([100, 100, 100, 100, 100]) == 0.0)

# All zeros
check("Gini of all zeros = 0",
      discrete_gini([0, 0, 0, 0, 0]) == 0.0)

# Maximum inequality: one agent has everything
g = discrete_gini([0, 0, 0, 0, 1000])
check("Gini of [0,0,0,0,1000] = 0.80", approx(g, 0.80, tol=0.01),
      f"gini = {g:.4f}")
# Theoretical max for n=5: (n-1)/n = 4/5 = 0.80

# Moderate inequality
g = discrete_gini([100, 200, 300, 400, 500])
check("Gini of [100..500] in (0.2, 0.3)", 0.2 < g < 0.3,
      f"gini = {g:.4f}")

# Single agent
check("Gini of single agent = 0", discrete_gini([500]) == 0.0)

# Two agents, one has all
g = discrete_gini([0, 1000])
check("Gini of [0, 1000] = 0.50", approx(g, 0.50, tol=0.01),
      f"gini = {g:.4f}")

# One agent has zero, rest equal -- realistic new-agent scenario
g = discrete_gini([0, 1000, 1000, 1000, 1000])
check("Gini of [0, 1k, 1k, 1k, 1k] in (0.15, 0.25)", 0.15 < g < 0.25,
      f"gini = {g:.4f}")


# ============================================================
heading("1. PROFIT SCORE")
# ============================================================

# --- Zero everything: P(t) = 0 ---
p = profit_score(
    revenue=0, labor_cost=0, input_cost=0, tax_rate=0.15,
    capex=0, baseline_revenue=REVENUE_FLOOR,
)
check("All zeros -> P = 0", approx(p, 0.0), f"P = {p:.6f}")

# --- Break-even: revenue covers costs exactly, no capex ---
p = profit_score(
    revenue=1000, labor_cost=500, input_cost=500, tax_rate=0.15,
    capex=0, baseline_revenue=1000,
)
# gross_profit = 0, tax = 0, net_income = 0
check("Break-even -> P = 0", approx(p, 0.0), f"P = {p:.6f}")

# --- ema_revenue exactly at floor boundary ---
p = profit_score(
    revenue=600, labor_cost=100, input_cost=100, tax_rate=0.15,
    capex=0, baseline_revenue=500.0,  # exactly floor
)
# gross_profit = 400, tax = 60, net_income = 340
# denom = max(500, 500) = 500, ratio = 340/500 = 0.68
check("EMA exactly at floor -> valid P", 0.1 < p < 0.5,
      f"P = {p:.4f}  (ema=floor={REVENUE_FLOOR})")

# --- Profitable firm ---
p = profit_score(
    revenue=10000, labor_cost=3000, input_cost=2000, tax_rate=0.15,
    capex=0, baseline_revenue=10000,
)
# gross_profit = 5000, tax = 750, net_income = 4250
# ratio = 4250/10000 = 0.425
check("Profitable firm -> P in (0.1, 0.3)", 0.1 < p < 0.3,
      f"P = {p:.4f}")

# --- Very profitable firm ---
p = profit_score(
    revenue=10000, labor_cost=1000, input_cost=500, tax_rate=0.15,
    capex=0, baseline_revenue=10000,
)
# gross_profit = 8500, tax = 1275, net_income = 7225
# ratio = 7225/10000 = 0.7225
check("Very profitable -> P in (0.2, 0.4)", 0.2 < p < 0.4,
      f"P = {p:.4f}")

# --- Losing firm ---
p = profit_score(
    revenue=1000, labor_cost=3000, input_cost=2000, tax_rate=0.15,
    capex=0, baseline_revenue=1000,
)
# gross_profit = -4000, tax = 0 (losses), net_income = -4000
# ratio = -4000/1000 = -4.0
check("Losing firm -> P < 0", p < 0, f"P = {p:.4f}")
check("Losing firm -> P > -1", p > -1, f"P = {p:.4f}")

# --- Heavy capex investment ---
p = profit_score(
    revenue=5000, labor_cost=1000, input_cost=1000, tax_rate=0.15,
    capex=10000, baseline_revenue=5000,
)
# gross_profit = 3000, tax = 450, amortized_capex = 10000/10 = 1000
# net_income = 3000 - 450 - 1000 = 1550
check("Heavy capex still positive (amortized)", p > 0,
      f"P = {p:.4f}")

# --- Extreme profit (clipping test) ---
p = profit_score(
    revenue=100000, labor_cost=0, input_cost=0, tax_rate=0.0,
    capex=0, baseline_revenue=500,  # ema << revenue
)
check("Extreme profit clipped to 1.0", p == 1.0,
      f"P = {p:.6f}")

# --- Extreme loss (clipping test) ---
p = profit_score(
    revenue=0, labor_cost=50000, input_cost=50000, tax_rate=0.0,
    capex=0, baseline_revenue=500,
)
check("Extreme loss clipped to -1.0", p == -1.0,
      f"P = {p:.6f}")

# --- No clipping mode ---
p = profit_score(
    revenue=100000, labor_cost=0, input_cost=0, tax_rate=0.0,
    capex=0, baseline_revenue=500, clip=False,
)
check("No-clip mode allows P > 1.0", p > 1.0,
      f"P = {p:.4f}")

# --- EMA at floor: cold start with small revenue ---
p = profit_score(
    revenue=100, labor_cost=50, input_cost=20, tax_rate=0.15,
    capex=0, baseline_revenue=100,  # small but above floor? No, floor=500
)
# ema_revenue=100 < floor=500, so denominator = 500
check("Small revenue uses floor as denominator", 0 < p < 0.1,
      f"P = {p:.4f}")


# ============================================================
heading("2. SOLVENCY SCORE")
# ============================================================

ideal_easy = EASY.ideal_runway  # 15.0

# --- At ideal runway -> S = 0.5 ---
s = solvency_score(
    firm_cash=15000, total_expenses=2000, revenue=1000,
    ideal_runway=ideal_easy,
)
check("At ideal runway -> S = 0.5", approx(s, 0.5, tol=0.01),
      f"S = {s:.4f}")

# --- Comfortable runway (2x ideal) -> S > 0.9 ---
s = solvency_score(
    firm_cash=30000, total_expenses=2000, revenue=1000,
    ideal_runway=ideal_easy,
)
check("2x ideal runway -> S > 0.9", s > 0.9,
      f"S = {s:.4f}")

# --- Zero cash -> near 0 ---
s = solvency_score(
    firm_cash=0, total_expenses=2000, revenue=1000,
    ideal_runway=ideal_easy,
)
check("Zero cash -> S < 0.1", s < 0.1,
      f"S = {s:.4f}")

# --- Negative cash is floored to 0 (Bug 3 fix) ---
s_neg = solvency_score(
    firm_cash=-1000000, total_expenses=2000, revenue=1000,
    ideal_runway=ideal_easy,
)
s_zero = solvency_score(
    firm_cash=0, total_expenses=2000, revenue=1000,
    ideal_runway=ideal_easy,
)
check("Negative cash floored to 0 (same as zero)",
      approx(s_neg, s_zero),
      f"S(-1M) = {s_neg:.4f}, S(0) = {s_zero:.4f}")

# --- Small negative cash same as zero ---
s_small_neg = solvency_score(
    firm_cash=-1, total_expenses=2000, revenue=1000,
    ideal_runway=ideal_easy,
)
check("Small negative cash also floored",
      approx(s_small_neg, s_zero),
      f"S(-1) = {s_small_neg:.4f}, S(0) = {s_zero:.4f}")

# --- Firm is profitable (revenue > expenses) ---
s = solvency_score(
    firm_cash=10000, total_expenses=1000, revenue=2000,
    ideal_runway=ideal_easy,
)
check("Profitable firm (revenue > expenses) -> S near 1.0", s > 0.99,
      f"S = {s:.4f}")

# --- Crisis: huge expenses, tiny cash ---
s = solvency_score(
    firm_cash=500, total_expenses=5000, revenue=0,
    ideal_runway=ideal_easy,
)
check("Crisis: 0.1 steps runway -> S near 0", s < 0.05,
      f"S = {s:.4f}")

# --- Output is always in [0, 1] ---
s_low = solvency_score(firm_cash=0, total_expenses=99999, revenue=0,
                        ideal_runway=ideal_easy)
s_high = solvency_score(firm_cash=1e12, total_expenses=1, revenue=0,
                         ideal_runway=ideal_easy)
check("S is always in [0, 1]", 0 <= s_low <= 1 and 0 <= s_high <= 1,
      f"S_low = {s_low:.6f}, S_high = {s_high:.6f}")

# --- Hard difficulty has higher ideal runway ---
ideal_hard = HARD.ideal_runway  # 30.0
s_easy = solvency_score(firm_cash=15000, total_expenses=2000, revenue=1000,
                         ideal_runway=ideal_easy)
s_hard = solvency_score(firm_cash=15000, total_expenses=2000, revenue=1000,
                         ideal_runway=ideal_hard)
check("Same cash, harder target -> lower S", s_hard < s_easy,
      f"S_easy = {s_easy:.4f}, S_hard = {s_hard:.4f}")


# ============================================================
heading("3. GDP SCORE (relative delta)")
# ============================================================

# --- No change -> G = 0 ---
g = gdp_score(real_gdp_current=50000, real_gdp_previous=50000)
check("No GDP change -> G = 0", approx(g, 0.0),
      f"G = {g:.6f}")

# --- 10% growth ---
g = gdp_score(real_gdp_current=55000, real_gdp_previous=50000)
# delta = 5000, denom = 50000, G = 5000/50000 = 0.10
check("10% growth -> G = 0.10", approx(g, 0.10, tol=0.001),
      f"G = {g:.4f}")

# --- 10% contraction ---
g = gdp_score(real_gdp_current=45000, real_gdp_previous=50000)
# delta = -5000, denom = 50000, G = -0.10
check("10% contraction -> G = -0.10", approx(g, -0.10, tol=0.001),
      f"G = {g:.4f}")

# --- Scale invariance: 10% growth from different bases ---
g_small = gdp_score(real_gdp_current=1100, real_gdp_previous=1000)
g_large = gdp_score(real_gdp_current=110000, real_gdp_previous=100000)
check("Scale invariant: 10% from 1k == 10% from 100k",
      approx(g_small, g_large, tol=0.001),
      f"G(1k) = {g_small:.4f}, G(100k) = {g_large:.4f}")

# --- This was the bug: same absolute delta, different bases ---
g_from_small = gdp_score(real_gdp_current=6000, real_gdp_previous=5000)
g_from_large = gdp_score(real_gdp_current=51000, real_gdp_previous=50000)
# 20% growth vs 2% growth -> must be different
check("Same abs delta, different bases -> different scores",
      abs(g_from_small - g_from_large) > 0.1,
      f"G(5k->6k) = {g_from_small:.4f}, G(50k->51k) = {g_from_large:.4f}")

# --- Small growth ---
g = gdp_score(real_gdp_current=50100, real_gdp_previous=50000)
check("Small growth (0.2%) -> G ~ 0.002", approx(g, 0.002, tol=0.001),
      f"G = {g:.6f}")

# --- Massive growth (clipping to 1.0) ---
g = gdp_score(real_gdp_current=200000, real_gdp_previous=50000)
# delta = 150000, denom = 50000, raw = 3.0 -> clipped to 1.0
check("Massive growth clipped to 1.0", g == 1.0,
      f"G = {g:.4f}")

# --- Massive contraction (clipping to -1.0) ---
g = gdp_score(real_gdp_current=0, real_gdp_previous=200000)
check("Massive contraction clipped to -1.0", g == -1.0,
      f"G = {g:.4f}")

# --- Symmetry holds ONLY when called with the same previous value ---
# NOTE: In real gameplay, growth from 45k to 50k uses denom=45k (+11.1%),
# while fall from 50k to 45k uses denom=50k (-10%). This asymmetry is
# economically correct: a 10% fall is harder to recover from proportionally.
# This test uses the same previous value so symmetry holds here.
g_up = gdp_score(real_gdp_current=55000, real_gdp_previous=50000)
g_down = gdp_score(real_gdp_current=45000, real_gdp_previous=50000)
check("Symmetric when same base: G(+5k) = -G(-5k)",
      approx(g_up, -g_down, tol=0.001),
      f"G_up = {g_up:.4f}, G_down = {g_down:.4f}")

# --- Asymmetry with different bases (economic reality) ---
g_fall = gdp_score(real_gdp_current=45000, real_gdp_previous=50000)   # -10%
g_recover = gdp_score(real_gdp_current=50000, real_gdp_previous=45000)  # +11.1%
check("Asymmetric recovery: recovering from fall scores higher",
      g_recover > abs(g_fall),
      f"G(fall) = {g_fall:.4f}, G(recover) = {g_recover:.4f}")

# --- Previous GDP = 0: uses floor denominator ---
g = gdp_score(real_gdp_current=1000, real_gdp_previous=0)
# denom = max(0, 1.0) = 1.0, delta = 1000, raw = 1000 -> clipped to 1.0
check("Previous GDP = 0 -> uses floor, clips to 1.0", g == 1.0,
      f"G = {g:.4f}")

# --- Previous GDP = 0, no change ---
g = gdp_score(real_gdp_current=0, real_gdp_previous=0)
check("Both GDP = 0 -> G = 0", approx(g, 0.0),
      f"G = {g:.6f}")


# ============================================================
heading("4. GINI PENALTY")
# ============================================================

# --- Perfect equality -> penalty = 0 ---
p = gini_penalty([1000, 1000, 1000, 1000, 1000])
check("Perfect equality -> penalty = 0", approx(p, 0.0),
      f"penalty = {p:.6f}")

# --- Below threshold -> penalty = 0 ---
p = gini_penalty([800, 900, 1000, 1100, 1200])
gini_val = discrete_gini([800, 900, 1000, 1100, 1200])
check(f"Low inequality (Gini={gini_val:.3f}) -> penalty = 0",
      approx(p, 0.0), f"penalty = {p:.6f}")

# --- Moderate inequality ---
wealth = [100, 200, 500, 1000, 5000]
gini_val = discrete_gini(wealth)
p = gini_penalty(wealth)
check(f"Moderate inequality (Gini={gini_val:.3f}) -> penalty in (0, 0.5)",
      0 < p < 0.5, f"penalty = {p:.4f}")

# --- High inequality ---
wealth = [0, 0, 0, 0, 10000]
gini_val = discrete_gini([max(w, GINI_WEALTH_FLOOR) for w in wealth])
p = gini_penalty(wealth)
check(f"High inequality (Gini={gini_val:.3f}) -> penalty in (0.2, 0.4)",
      0.2 < p < 0.4, f"penalty = {p:.4f}")

# --- Maximum inequality with 2 agents ---
p = gini_penalty([0, 10000])
gini_val = discrete_gini([0, 10000])
check(f"2-agent max inequality (Gini={gini_val:.3f}) -> penalty > 0",
      p > 0, f"penalty = {p:.4f}")

# --- Negative wealth is floored ---
p_neg = gini_penalty([-500, -500, -500, -500, 10000])
p_zero = gini_penalty([0, 0, 0, 0, 10000])
check("Negative wealth floored to 0 (same as zero wealth)",
      approx(p_neg, p_zero),
      f"p_neg = {p_neg:.4f}, p_zero = {p_zero:.4f}")

# --- Penalty is in [0, 1] ---
check("Penalty at perfect equality = 0", gini_penalty([1, 1, 1, 1, 1]) == 0.0)
check("Penalty at max inequality >= 0", gini_penalty([0, 0, 0, 0, 1]) >= 0.0)
check("Penalty at max inequality <= 1", gini_penalty([0, 0, 0, 0, 1]) <= 1.0)

# --- Penalty is monotonically increasing with inequality ---
p1 = gini_penalty([500, 500, 500, 500, 1000])
p2 = gini_penalty([100, 100, 100, 100, 5000])
p3 = gini_penalty([0, 0, 0, 0, 10000])
check("Penalty increases with inequality",
      p1 <= p2 <= p3,
      f"p1={p1:.4f} <= p2={p2:.4f} <= p3={p3:.4f}")


# ============================================================
heading("5. COMPONENT INTERACTION SANITY CHECKS")
# ============================================================

# A healthy firm scenario: all components should be positive / low penalty
w = EASY.weights_dict()

p_score = profit_score(
    revenue=8000, labor_cost=2000, input_cost=1500, tax_rate=0.15,
    capex=2000, baseline_revenue=7000,
)
s_score = solvency_score(
    firm_cash=20000, total_expenses=4500, revenue=8000,
    ideal_runway=EASY.ideal_runway,
)
g_score = gdp_score(real_gdp_current=52000, real_gdp_previous=50000)
gini_pen = gini_penalty([8000, 9000, 10000, 11000, 12000])

check("Healthy firm: P > 0", p_score > 0, f"P = {p_score:.4f}")
check("Healthy firm: S > 0.5", s_score > 0.5, f"S = {s_score:.4f}")
check("Healthy firm: G > 0", g_score > 0, f"G = {g_score:.4f}")
check("Healthy firm: Gini penalty = 0", approx(gini_pen, 0.0),
      f"penalty = {gini_pen:.4f}")

reward_healthy = (w["profit"] * p_score +
                  w["solvency"] * s_score +
                  w["gdp"] * g_score +
                  w["gini"] * gini_pen)
check("Healthy firm Easy reward in (0.5, 1.0)", 0.5 < reward_healthy < 1.0,
      f"R = {reward_healthy:.4f}")

# A struggling firm
p_score = profit_score(
    revenue=1000, labor_cost=3000, input_cost=2000, tax_rate=0.15,
    capex=5000, baseline_revenue=2000,
)
s_score = solvency_score(
    firm_cash=2000, total_expenses=6000, revenue=1000,
    ideal_runway=EASY.ideal_runway,
)
g_score = gdp_score(real_gdp_current=40000, real_gdp_previous=50000)
gini_pen = gini_penalty([0, 0, 0, 500, 20000])

check("Struggling firm: P < 0", p_score < 0, f"P = {p_score:.4f}")
check("Struggling firm: S < 0.5", s_score < 0.5, f"S = {s_score:.4f}")
check("Struggling firm: G < 0", g_score < 0, f"G = {g_score:.4f}")
check("Struggling firm: Gini penalty > 0", gini_pen > 0,
      f"penalty = {gini_pen:.4f}")

reward_struggling = (w["profit"] * p_score +
                     w["solvency"] * s_score +
                     w["gdp"] * g_score +
                     w["gini"] * gini_pen)
check("Struggling firm Easy reward in (-0.1, 0.3)", -0.1 < reward_struggling < 0.3,
      f"R = {reward_struggling:.4f}")

print(f"\n  Healthy firm reward:    {reward_healthy:.4f}")
print(f"  Struggling firm reward: {reward_struggling:.4f}")
print(f"  Spread:                 {reward_healthy - reward_struggling:.4f}")


# ============================================================
heading("RESULTS")
# ============================================================
total = passed + failed
print(f"\n  {passed}/{total} checks passed, {failed} failed.\n")

if failed > 0:
    print("  WARNING: Fix failures before proceeding to Phase 3.\n")
    sys.exit(1)
else:
    print("  ALL CHECKS PASSED -- Phase 2 reward components verified.\n")
    sys.exit(0)
