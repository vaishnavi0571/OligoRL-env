"""
Phase 1 -- Verification Script
===============================
Validates that every constant in config.py is self-consistent and that
the formulas produce values in the expected ranges.

Run:  python verify_config.py
"""

import math
import sys
from config import (
    IHS_EMA_DECAY, REVENUE_FLOOR, REVENUE_EMA_INITIAL, IHS_MAX_RATIO,
    IHS_CLIP, CAPEX_AMORTIZATION_STEPS,
    RUNWAY_TARGET_RATIO, SOLVENCY_STEEPNESS, BURN_RATE_EPSILON,
    PRICE_INERTIA, INITIAL_PRICE_INDEX, GDP_FLOOR_EPSILON,
    GINI_THRESHOLD, GINI_EXPONENT, GINI_WEALTH_FLOOR,
    GINI_WEIGHT, POSITIVE_WEIGHT_SUM,
    EASY, MEDIUM, HARD, PRESETS,
    INITIAL_FIRM_CASH, TAX_RATE, NUM_AGENTS, DEFAULT_PRICE,
    LOBBY_TAX_CHANGE_MIN, LOBBY_TAX_CHANGE_MAX,
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
heading("1. IHS Profit Component")
# ============================================================

check("EMA decay in (0, 1)", 0 < IHS_EMA_DECAY < 1, f"a = {IHS_EMA_DECAY}")
check("Revenue floor > 0", REVENUE_FLOOR > 0, f"floor = {REVENUE_FLOOR}")
check("IHS max ratio > 1", IHS_MAX_RATIO > 1, f"max_ratio = {IHS_MAX_RATIO}")

# EMA cold-start behavior
check("EMA initial = revenue floor",
      REVENUE_EMA_INITIAL == REVENUE_FLOOR,
      f"ema_init = {REVENUE_EMA_INITIAL}, floor = {REVENUE_FLOOR}")
check("max(ema_init, floor) = floor at step 0",
      max(REVENUE_EMA_INITIAL, REVENUE_FLOOR) == REVENUE_FLOOR)

# IHS clipping decision is explicit
check("IHS_CLIP is explicitly set", isinstance(IHS_CLIP, bool),
      f"clip = {IHS_CLIP}")

# Verify the IHS range mapping (before clipping)
lo = math.asinh(-IHS_MAX_RATIO) / math.asinh(IHS_MAX_RATIO)
hi = math.asinh(IHS_MAX_RATIO) / math.asinh(IHS_MAX_RATIO)
check("P(t) at +/-max_ratio = +/-1.0 exactly",
      abs(lo - (-1.0)) < 1e-10 and abs(hi - 1.0) < 1e-10,
      f"[{lo:.6f}, {hi:.6f}]")

# Verify clipping caps beyond-max values
if IHS_CLIP:
    extreme = math.asinh(100.0) / math.asinh(IHS_MAX_RATIO)
    clipped = max(-1.0, min(1.0, extreme))
    check("Clipping caps ratio=100 to 1.0", clipped == 1.0,
          f"raw = {extreme:.4f}, clipped = {clipped:.4f}")

# Spot-check: profit = 0 -> P(t) = 0
val = math.asinh(0.0) / math.asinh(IHS_MAX_RATIO)
check("Zero profit -> P(t) = 0", abs(val) < 1e-10, f"P(0) = {val:.6f}")

# Spot-check: moderate positive profit
ratio = 2.0  # net_income = 2x ema_revenue
val = math.asinh(ratio) / math.asinh(IHS_MAX_RATIO)
check("Moderate profit (2x rev) -> P(t) in (0.3, 0.7)", 0.3 < val < 0.7,
      f"P(2.0) = {val:.4f}")

check("Amortization steps > 0", CAPEX_AMORTIZATION_STEPS > 0,
      f"n = {CAPEX_AMORTIZATION_STEPS}")
check("Amortization < Easy episode length",
      CAPEX_AMORTIZATION_STEPS < EASY.episode_length,
      f"{CAPEX_AMORTIZATION_STEPS} < {EASY.episode_length}")
check("Amortization < Hard episode length",
      CAPEX_AMORTIZATION_STEPS < HARD.episode_length,
      f"{CAPEX_AMORTIZATION_STEPS} < {HARD.episode_length}")


# ============================================================
heading("2. Solvency Sigmoid")
# ============================================================

for preset in PRESETS.values():
    ideal = preset.ideal_runway
    check(
        f"{preset.name}: ideal runway in [5, {preset.episode_length}]",
        5 <= ideal <= preset.episode_length,
        f"target = {ideal:.1f} steps"
    )

    # Value at exactly the ideal runway should be 0.5 (sigmoid inflection)
    sig = 1.0 / (1.0 + math.exp(-SOLVENCY_STEEPNESS * (ideal - ideal)))
    check(f"{preset.name}: S(ideal) = 0.5", abs(sig - 0.5) < 1e-10,
          f"S({ideal:.0f}) = {sig:.4f}")

    # Value well above ideal
    sig_high = 1.0 / (1.0 + math.exp(-SOLVENCY_STEEPNESS * (ideal * 2 - ideal)))
    check(f"{preset.name}: S(2x ideal) > 0.8", sig_high > 0.8,
          f"S({ideal * 2:.0f}) = {sig_high:.4f}")

    # Value at zero runway
    sig_zero = 1.0 / (1.0 + math.exp(-SOLVENCY_STEEPNESS * (0 - ideal)))
    check(f"{preset.name}: S(0) < 0.15", sig_zero < 0.15,
          f"S(0) = {sig_zero:.4f}")


# ============================================================
heading("3. Price Index / GDP Delta")
# ============================================================

check("Price inertia in (0.5, 1)", 0.5 < PRICE_INERTIA < 1.0,
      f"inertia = {PRICE_INERTIA}")
check("Initial price index = DEFAULT_PRICE",
      INITIAL_PRICE_INDEX == DEFAULT_PRICE,
      f"index = {INITIAL_PRICE_INDEX}, default_price = {DEFAULT_PRICE}")

# Verify shock propagation: after N steps what fraction of a shock remains?
shock = INITIAL_PRICE_INDEX * 2.0  # price doubles
p = INITIAL_PRICE_INDEX
for _ in range(10):
    p = PRICE_INERTIA * p + (1 - PRICE_INERTIA) * shock
absorbed_pct = (p - INITIAL_PRICE_INDEX) / (shock - INITIAL_PRICE_INDEX) * 100
check("10-step shock propagation > 70%", absorbed_pct > 70,
      f"after 10 steps, {absorbed_pct:.1f}% of a 2x shock absorbed")

# Single-step propagation <= 20%
single_step = PRICE_INERTIA * INITIAL_PRICE_INDEX + (1 - PRICE_INERTIA) * shock
single_pct = (single_step - INITIAL_PRICE_INDEX) / (shock - INITIAL_PRICE_INDEX) * 100
check("Single-step shock <= 20%", single_pct <= 20,
      f"single step = {single_pct:.1f}%")

# GDP floor epsilon (relative-delta denominator floor)
check("GDP floor epsilon > 0", GDP_FLOOR_EPSILON > 0,
      f"floor = {GDP_FLOOR_EPSILON}")
check("GDP floor epsilon <= 10", GDP_FLOOR_EPSILON <= 10,
      f"floor = {GDP_FLOOR_EPSILON}  (should be small)")


# ============================================================
heading("4. Gini Penalty")
# ============================================================

check("Gini threshold in (0, 1)", 0 < GINI_THRESHOLD < 1,
      f"threshold = {GINI_THRESHOLD}")
check("Gini exponent in [2, 5]", 2 <= GINI_EXPONENT <= 5,
      f"exp = {GINI_EXPONENT}")

# Penalty at threshold
excess = max(GINI_THRESHOLD - GINI_THRESHOLD, 0)
penalty = (excess / (1 - GINI_THRESHOLD)) ** GINI_EXPONENT
check("Penalty at threshold = 0", penalty == 0.0)

# Penalty at Gini=0.7
gini = 0.7
excess = max(gini - GINI_THRESHOLD, 0)
penalty = (excess / (1 - GINI_THRESHOLD)) ** GINI_EXPONENT
check("Penalty at Gini=0.7 in (0.05, 0.6)", 0.05 < penalty < 0.6,
      f"penalty = {penalty:.4f}  (contributes {GINI_WEIGHT * penalty:.4f} to reward)")

# Penalty at Gini=1.0
excess = max(1.0 - GINI_THRESHOLD, 0)
penalty = (excess / (1 - GINI_THRESHOLD)) ** GINI_EXPONENT
check("Penalty at Gini=1.0 = 1.0", abs(penalty - 1.0) < 1e-10,
      f"penalty = {penalty:.6f}")

# Penalty at Gini=0.3 (below threshold)
excess = max(0.3 - GINI_THRESHOLD, 0)
penalty = (excess / (1 - GINI_THRESHOLD)) ** GINI_EXPONENT
check("Penalty at Gini=0.3 = 0 (below threshold)", penalty == 0.0)


# ============================================================
heading("5. Difficulty Presets & Weight Normalization")
# ============================================================

for name, preset in PRESETS.items():
    w = preset.weights_dict()
    pos_sum = w["profit"] + w["solvency"] + w["gdp"]

    check(f"{name}: positive weights sum ~ {POSITIVE_WEIGHT_SUM}",
          abs(pos_sum - POSITIVE_WEIGHT_SUM) < 0.02,
          f"Sum(P,S,G) = {pos_sum:.4f}")

    check(f"{name}: gini weight = {GINI_WEIGHT}",
          w["gini"] == GINI_WEIGHT)

    check(f"{name}: all positive weights > 0",
          all(w[k] > 0 for k in ["profit", "solvency", "gdp"]),
          f"P={w['profit']:.4f} S={w['solvency']:.4f} G={w['gdp']:.4f}")

    # Reward range: perfect agent scores POSITIVE_WEIGHT_SUM, worst scores GINI_WEIGHT
    r_max = pos_sum  # all components = 1.0, gini penalty = 0
    r_min = w["gini"]  # all components = 0, max gini penalty
    check(f"{name}: max reward ~ 1.00", abs(r_max - 1.00) < 0.02,
          f"R_max = {r_max:.4f}")
    check(f"{name}: min reward ~ -0.10", abs(r_min - (-0.10)) < 0.02,
          f"R_min = {r_min:.4f}")


# Print preset details
print()
for name, preset in PRESETS.items():
    print(f"  {preset}")


# ============================================================
heading("6. Cross-Component Sanity Checks")
# ============================================================

# Realistic scenario: small firm spending 5% of cash per step, no revenue
expense_rate_5pct = INITIAL_FIRM_CASH * 0.05
burn_5pct = max(expense_rate_5pct - 0, BURN_RATE_EPSILON)
runway_5pct = INITIAL_FIRM_CASH / burn_5pct
check("Runway at 5% expense rate in [15, 25]", 15 <= runway_5pct <= 25,
      f"runway = {runway_5pct:.1f} steps (comfortable)")

# Stressed scenario: 20% of cash per step, no revenue
expense_rate_20pct = INITIAL_FIRM_CASH * 0.20
burn_20pct = max(expense_rate_20pct - 0, BURN_RATE_EPSILON)
runway_20pct = INITIAL_FIRM_CASH / burn_20pct
check("Runway at 20% expense rate in [3, 10]", 3 <= runway_20pct <= 10,
      f"runway = {runway_20pct:.1f} steps (stressed)")

# Crisis scenario: 50% of cash per step
expense_rate_50pct = INITIAL_FIRM_CASH * 0.50
burn_50pct = max(expense_rate_50pct - 0, BURN_RATE_EPSILON)
runway_50pct = INITIAL_FIRM_CASH / burn_50pct
check("Runway at 50% expense rate in [1, 3]", 1 <= runway_50pct <= 3,
      f"runway = {runway_50pct:.1f} steps (crisis)")

# Tax rate sanity
check("Tax rate in [0.05, 0.40]", 0.05 <= TAX_RATE <= 0.40,
      f"tax = {TAX_RATE}")

# Number of agents >= 2 (needed for Gini)
check("NUM_AGENTS >= 2", NUM_AGENTS >= 2, f"n = {NUM_AGENTS}")


# ============================================================
heading("7. Action Bounds")
# ============================================================

check("Lobby min < 0", LOBBY_TAX_CHANGE_MIN < 0,
      f"min = {LOBBY_TAX_CHANGE_MIN}")
check("Lobby max > 0", LOBBY_TAX_CHANGE_MAX > 0,
      f"max = {LOBBY_TAX_CHANGE_MAX}")
check("Lobby range is symmetric",
      abs(LOBBY_TAX_CHANGE_MIN + LOBBY_TAX_CHANGE_MAX) < 1e-10,
      f"[{LOBBY_TAX_CHANGE_MIN}, {LOBBY_TAX_CHANGE_MAX}]")
check("Lobby max < TAX_RATE (can't zero tax in one step)",
      LOBBY_TAX_CHANGE_MAX < TAX_RATE,
      f"{LOBBY_TAX_CHANGE_MAX} < {TAX_RATE}")


# ============================================================
heading("RESULTS")
# ============================================================
total = passed + failed
print(f"\n  {passed}/{total} checks passed, {failed} failed.\n")

if failed > 0:
    print("  WARNING: Fix the failing checks before proceeding to Phase 2.\n")
    sys.exit(1)
else:
    print("  ALL CHECKS PASSED -- Phase 1 math is locked.\n")
    sys.exit(0)
