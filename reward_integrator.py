"""
OpenEnv -- Phase 4: Reward Integrator
======================================
Connects Phase 2 (reward components) with Phase 3 (simulator).
After each step(), call compute_step_reward() to get the composite
reward and a deterministic semantic feedback string.

The feedback string is a debugging tool: if the sentence describes
something economically absurd, you have a bug.
"""

from dataclasses import dataclass

from rewards import (
    profit_score, solvency_score, gdp_score,
    gini_with_penalty,
    compute_burn_rate, compute_runway,
)
from simulator import EconSimulator, StepInfo
from config import INITIAL_FIRM_CASH


@dataclass
class RewardResult:
    """Complete reward breakdown for one timestep."""
    profit: float            # P(t) in [-1, 1]
    solvency: float          # S(t) in [0, 1]
    gdp: float               # G(t) in [-1, 1]
    gini_pen: float          # penalty in [0, 1]
    composite: float         # weighted sum
    feedback: str            # human-readable explanation
    gini_coefficient: float  # raw Gini before penalty
    runway_steps: float      # firm cash / burn rate


def compute_step_reward(info: StepInfo, sim: EconSimulator) -> RewardResult:
    """Compute all four reward components and generate feedback.

    Must be called after sim.step() returns, using the returned StepInfo
    and the updated sim state.
    """
    preset = sim.preset
    w = preset.weights_dict()

    # --- 1. Profit score ---
    p = profit_score(
        revenue=info.revenue,
        labor_cost=info.labor_cost,
        input_cost=info.input_cost,
        tax_rate=sim.tax_rate,
        capex=info.capex_spent,
        baseline_revenue=INITIAL_FIRM_CASH / 2.0,
    )

    # --- 2. Solvency score ---
    s = solvency_score(
        firm_cash=sim.player.cash,
        total_expenses=info.total_expenses,
        revenue=info.revenue,
        ideal_runway=preset.ideal_runway,
    )

    # --- 3. GDP score ---
    g = gdp_score(
        real_gdp_current=info.real_gdp_current,
        real_gdp_previous=info.real_gdp_previous,
    )

    # --- 4. Gini (single computation for both coefficient and penalty) ---
    gini_coeff, gp = gini_with_penalty(info.all_agent_wealth)

    # --- Composite reward ---
    # Defensive: gini weight is always subtracted (penalty), regardless of sign
    composite = (
        w["profit"]    * p
        + w["solvency"] * s
        + w["gdp"]      * g
        - abs(w["gini"]) * gp
    )

    # --- Centralized burn_rate / runway (single source of truth) ---
    burn_rate = compute_burn_rate(info.total_expenses, info.revenue)
    runway = compute_runway(sim.player.cash, burn_rate)

    # --- Generate semantic feedback ---
    feedback = _generate_feedback(
        p, s, g, gp, composite, gini_coeff, runway, info, sim,
    )

    return RewardResult(
        profit=p,
        solvency=s,
        gdp=g,
        gini_pen=gp,
        composite=composite,
        feedback=feedback,
        gini_coefficient=gini_coeff,
        runway_steps=runway,
    )


# -----------------------------------------------------------------
# Semantic feedback generator (rule-based, deterministic, fast)
# -----------------------------------------------------------------

def _generate_feedback(
    p: float, s: float, g: float, gp: float, composite: float,
    gini_coeff: float, runway: float,
    info: StepInfo, sim: EconSimulator,
) -> str:
    """Build a human-readable feedback string with causal reasoning.

    Structure:
      [Step X/Y] Financial summary | Qualitative | Causes | Score
    """
    parts = []
    step = sim.step_count
    ep_len = sim.preset.episode_length

    # --- Header ---
    parts.append(f"[Step {step}/{ep_len}]")

    # --- Financial summary ---
    net = info.revenue - info.total_expenses
    sign = "+" if net >= 0 else ""
    parts.append(
        f"Rev ${info.revenue:,.0f} | "
        f"Costs ${info.total_expenses:,.0f} | "
        f"Net {sign}${net:,.0f} | "
        f"Cash ${sim.player.cash:,.0f}"
    )

    # --- Qualitative assessments ---
    if p > 0.5:
        profit_msg = "Highly profitable"
    elif p > 0.15:
        profit_msg = "Profitable"
    elif p > -0.05:
        profit_msg = "Near break-even"
    elif p > -0.5:
        profit_msg = "Losing money"
    else:
        profit_msg = "SEVERE LOSSES"

    if s > 0.9:
        solv_msg = f"cash strong ({runway:.0f} steps runway)"
    elif s > 0.5:
        solv_msg = f"cash adequate ({runway:.0f} steps runway)"
    elif s > 0.2:
        solv_msg = f"cash tight ({runway:.1f} steps runway)"
    else:
        solv_msg = f"CASH CRITICAL ({runway:.1f} steps runway)"

    if g > 0.1:
        gdp_msg = "economy booming"
    elif g > 0.02:
        gdp_msg = "economy growing"
    elif g > -0.02:
        gdp_msg = "economy flat"
    elif g > -0.1:
        gdp_msg = "economy contracting"
    else:
        gdp_msg = "ECONOMIC CRISIS"

    assessment = [profit_msg, solv_msg, gdp_msg]

    if gp > 0.3:
        assessment.append(f"HIGH INEQUALITY (Gini {gini_coeff:.2f})")
    elif gp > 0.1:
        assessment.append(f"rising inequality (Gini {gini_coeff:.2f})")
    elif gp > 0.01:
        assessment.append(f"slight inequality (Gini {gini_coeff:.2f})")

    parts.append(" | ".join(assessment))

    # --- Causal reasoning (WHY things happened) ---
    causes = []

    # Price vs competitors
    player_price = sim.player.price
    comp_avg = info.competitor_avg_price
    if comp_avg > 0:
        price_ratio = player_price / comp_avg
        if price_ratio > 1.15:
            causes.append(
                f"pricing ${player_price:.0f} is {(price_ratio-1)*100:.0f}% "
                f"above competitors (${comp_avg:.0f}), reducing demand"
            )
        elif price_ratio < 0.85:
            causes.append(
                f"aggressive pricing ${player_price:.0f} vs competitors "
                f"${comp_avg:.0f} is capturing extra demand"
            )

    # Revenue trend vs EMA
    if sim.ema_revenue > 0:
        rev_ratio = info.revenue / sim.ema_revenue
        if rev_ratio > 1.15:
            causes.append("revenue trending up vs recent average")
        elif rev_ratio < 0.85:
            causes.append("revenue declining vs recent average")

    # Hiring effects
    if info.hiring_cost > 0:
        new_workers = int(info.hiring_cost / 500)  # HIRING_COST_PER_WORKER
        if info.revenue > info.labor_cost:
            causes.append(f"hired {new_workers} workers, boosting output")
        else:
            causes.append(f"hired {new_workers} workers, straining margins")

    # Capex effects
    if info.capex_spent > 0:
        causes.append(f"invested ${info.capex_spent:,.0f} in capital")

    # Capacity utilization
    if info.max_output > 0:
        utilization = info.units_sold / info.max_output
        if utilization < 0.5:
            causes.append(
                f"only using {utilization:.0%} of production capacity — "
                f"demand too low or price too high"
            )

    if causes:
        parts.append("Because: " + "; ".join(causes))

    # --- Overall score ---
    if composite > 0.7:
        parts.append(f"EXCELLENT (R={composite:.3f})")
    elif composite > 0.4:
        parts.append(f"Solid (R={composite:.3f})")
    elif composite > 0.1:
        parts.append(f"Below average (R={composite:.3f})")
    elif composite > -0.05:
        parts.append(f"Poor (R={composite:.3f})")
    else:
        parts.append(f"CRITICAL (R={composite:.3f})")

    # --- Termination events ---
    if info.terminated:
        parts.append(">>> BANKRUPT -- episode terminated.")
    elif info.truncated:
        parts.append(">>> Episode complete.")
    elif sim.player.cash < 2000:
        parts.append("WARNING: Cash below $2,000.")

    return " | ".join(parts)
