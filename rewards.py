"""
OpenEnv -- Phase 2: Reward Components
======================================
Four pure functions, each taking only numbers and returning a float.
No environment state, no Pydantic, no FastAPI.

All defaults come from config.py but every parameter is overridable
for testing.
"""

import math
from typing import List

from config import (
    IHS_MAX_RATIO, REVENUE_FLOOR, CAPEX_AMORTIZATION_STEPS, IHS_CLIP,
    SOLVENCY_STEEPNESS, BURN_RATE_EPSILON,
    GDP_FLOOR_EPSILON,
    GINI_THRESHOLD, GINI_EXPONENT, GINI_WEALTH_FLOOR,
)


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def update_ema(ema_prev: float, new_value: float, decay: float) -> float:
    """Compute one step of exponential moving average.

    ema_t = decay * ema_prev + (1 - decay) * new_value
    """
    return decay * ema_prev + (1.0 - decay) * new_value


def discrete_gini(values: List[float]) -> float:
    """Compute the Gini coefficient for a list of non-negative values.

    Uses the sorted-array formula (O(n log n)):
        G = (2 * sum_{i=1}^{n} i * v_i) / (n * sum(v)) - (n + 1) / n
    where v is sorted ascending.

    Returns 0.0 if all values are zero (perfect equality).
    Expects all values >= 0.
    """
    n = len(values)
    if n < 2:
        return 0.0
    total = sum(values)
    if total == 0.0:
        return 0.0  # all agents have zero wealth = perfect equality
    v = sorted(values)
    weighted_sum = sum((i + 1) * v[i] for i in range(n))
    return (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n


# -----------------------------------------------------------------
# 1. Profit Score  P(t)
# -----------------------------------------------------------------

def profit_score(
    revenue: float,
    labor_cost: float,
    input_cost: float,
    tax_rate: float,
    capex: float,
    baseline_revenue: float,
    amortization_n: int = CAPEX_AMORTIZATION_STEPS,
    max_ratio: float = IHS_MAX_RATIO,
    revenue_floor: float = REVENUE_FLOOR,
    clip: bool = IHS_CLIP,
) -> float:
    """Inverse Hyperbolic Sine transform of net income, normalized by EMA revenue.

    Args:
        revenue:       Gross revenue this timestep.
        labor_cost:    Total labor cost this timestep.
        input_cost:    Total input/material cost this timestep.
        tax_rate:      Flat tax rate (applied only when gross_profit > 0).
        capex:         Capital expenditure to amortize.
        baseline_revenue: Target normalization scale (prevents margin penalty).
        amortization_n: Steps over which capex is spread.
        max_ratio:     Normalizer for the asinh mapping.
        revenue_floor: Minimum denominator for normalization.
        clip:          If True, hard-clip output to [-1, 1].

    Returns:
        P(t) in [-1, 1] (if clip=True).
    """
    gross_profit = revenue - labor_cost - input_cost
    tax = max(gross_profit, 0.0) * tax_rate
    amortized_capex = capex / amortization_n if amortization_n > 0 else 0.0
    net_income = gross_profit - tax - amortized_capex

    denominator = max(baseline_revenue, revenue_floor)
    ratio = net_income / denominator

    score = math.asinh(ratio) / math.asinh(max_ratio)

    if clip:
        score = max(-1.0, min(1.0, score))

    return score


# -----------------------------------------------------------------
# 2. Solvency Score  S(t)
# -----------------------------------------------------------------

def solvency_score(
    firm_cash: float,
    total_expenses: float,
    revenue: float,
    ideal_runway: float,
    steepness: float = SOLVENCY_STEEPNESS,
    burn_rate_epsilon: float = BURN_RATE_EPSILON,
) -> float:
    """Logistic sigmoid of runway (steps of cash remaining).

    Args:
        firm_cash:        Current cash on hand.
        total_expenses:   Total expenses this timestep (labor + inputs + tax + capex).
        revenue:          Revenue this timestep.
        ideal_runway:     Target runway in timestep units (from preset).
        steepness:        Logistic steepness parameter k.
        burn_rate_epsilon: Floor for burn rate to avoid division by zero.

    Returns:
        S(t) in (0, 1).  0.5 at exactly ideal_runway.
    """
    burn_rate = max(total_expenses - revenue, burn_rate_epsilon)
    runway = max(firm_cash, 0.0) / burn_rate

    # Clamp the exponent to avoid overflow in math.exp for very negative runways
    exponent = -steepness * (runway - ideal_runway)
    exponent = max(-500.0, min(500.0, exponent))

    return 1.0 / (1.0 + math.exp(exponent))


# -----------------------------------------------------------------
# 3. GDP Delta Score  G(t)
# -----------------------------------------------------------------

def gdp_score(
    real_gdp_current: float,
    real_gdp_previous: float,
    floor: float = GDP_FLOOR_EPSILON,
) -> float:
    """Relative change in real GDP, clipped to [-1, 1].

    Computes (current - previous) / max(|previous|, floor).
    This is a true percentage-change measure: 10% growth always
    scores ~0.1 regardless of the absolute GDP level.

    Note: the caller is responsible for deflating nominal GDP by the
    price index before passing values here.

    Args:
        real_gdp_current:  Real GDP at timestep t.
        real_gdp_previous: Real GDP at timestep t-1.
        floor:             Minimum denominator (prevents div-by-zero
                           when previous GDP is 0 at episode start).

    Returns:
        G(t) in [-1, 1].
    """
    delta = real_gdp_current - real_gdp_previous
    denominator = max(abs(real_gdp_previous), floor)
    score = delta / denominator
    return max(-1.0, min(1.0, score))


# -----------------------------------------------------------------
# 4. Gini Penalty
# -----------------------------------------------------------------

def gini_penalty(
    wealth_vector: List[float],
    threshold: float = GINI_THRESHOLD,
    exponent: float = GINI_EXPONENT,
    wealth_floor: float = GINI_WEALTH_FLOOR,
) -> float:
    """Exponential penalty on inequality above a threshold.

    Returns:
        Penalty in [0, 1].  0 if Gini <= threshold, 1 if Gini = 1.0.
    """
    _, pen = gini_with_penalty(wealth_vector, threshold, exponent, wealth_floor)
    return pen


def gini_with_penalty(
    wealth_vector: List[float],
    threshold: float = GINI_THRESHOLD,
    exponent: float = GINI_EXPONENT,
    wealth_floor: float = GINI_WEALTH_FLOOR,
) -> tuple:
    """Compute Gini coefficient AND penalty in a single pass.

    Returns:
        (gini_coefficient, penalty)  -- avoids double computation.
    """
    floored = [max(w, wealth_floor) for w in wealth_vector]
    gini = discrete_gini(floored)

    excess = max(gini - threshold, 0.0)
    denom = 1.0 - threshold
    if denom <= 0:
        return gini, 0.0

    pen = (excess / denom) ** exponent
    return gini, pen


# -----------------------------------------------------------------
# 5. Shared Helpers (used by solvency_score and reward_integrator)
# -----------------------------------------------------------------

def compute_burn_rate(
    total_expenses: float,
    revenue: float,
    epsilon: float = BURN_RATE_EPSILON,
) -> float:
    """Net cash outflow rate, floored at epsilon."""
    return max(total_expenses - revenue, epsilon)


def compute_runway(firm_cash: float, burn_rate: float) -> float:
    """Steps of cash remaining at current burn rate."""
    return max(firm_cash, 0.0) / burn_rate

