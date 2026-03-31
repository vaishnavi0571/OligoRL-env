"""
OpenEnv -- Phase 1 Configuration
=================================
Single source of truth for every mathematical constant and design decision.
All reward functions, the simulator state machine, and the API layer import
from this file.  Change a number here => it changes everywhere.

Design decisions are documented inline; the full rationale lives in logs.txt.
"""

from dataclasses import dataclass
from typing import Dict

# ---------------------------------------------------------------------
# 1.  IHS PROFIT COMPONENT  (weight: see difficulty presets)
# ---------------------------------------------------------------------
# Normalization: rolling EMA of revenue with a cold-start floor.
#   profit_input = net_income / max(ema_revenue, REVENUE_FLOOR)
#   P(t) = clip(asinh(profit_input) / asinh(IHS_MAX_RATIO), -1, 1)

IHS_EMA_DECAY: float = 0.9
"""Exponential moving average decay for revenue normalization.
ema_{t} = alpha * ema_{t-1} + (1-alpha) * revenue_t"""

REVENUE_FLOOR: float = 500.0
"""Minimum divisor to prevent division-by-near-zero in early steps."""

REVENUE_EMA_INITIAL: float = REVENUE_FLOOR
"""Starting value of the revenue EMA before any observations.
Set equal to REVENUE_FLOOR so max(ema, floor) = floor at step 0,
giving a well-defined denominator from the first timestep."""

IHS_MAX_RATIO: float = 10.0
"""Assumed max reasonable value of net_income / ema_revenue.
asinh(IHS_MAX_RATIO) ~ 3.0 is used to map the raw IHS into [-1, 1].
Values beyond this are hard-clipped (see IHS_CLIP)."""

IHS_CLIP: bool = True
"""If True, P(t) is clipped to [-1, 1] after the asinh mapping.
This guarantees deterministic bounds on the profit component.
If False, extreme ratios (>10x revenue) produce P(t) slightly above 1.0."""

# Capex amortization
CAPEX_AMORTIZATION_STEPS: int = 10
"""Number of timesteps over which a capital expenditure is spread.
Each step bears capex / CAPEX_AMORTIZATION_STEPS of the cost.
Chosen so investments made at step 1 are fully amortized before a
30-step Easy episode ends."""


# ---------------------------------------------------------------------
# 2.  SOLVENCY SIGMOID  (weight: see difficulty presets)
# ---------------------------------------------------------------------
# Runway = firm_cash / max(burn_rate, BURN_RATE_EPSILON)   in timestep units.
# S(t) = sigmoid(k * (runway - ideal_runway))
# ideal_runway = RUNWAY_TARGET_RATIO * episode_length

RUNWAY_TARGET_RATIO: float = 0.5
"""Ideal runway expressed as a fraction of episode length.
On Easy (30 steps) -> target = 15 steps of runway."""

SOLVENCY_STEEPNESS: float = 0.2
"""Logistic steepness parameter k.  Controls how sharply the sigmoid
transitions around the ideal runway.  At k=0.2 the danger zone
(steep gradient) spans roughly 5-25 steps of runway."""

BURN_RATE_EPSILON: float = 1.0
"""Floor for burn-rate divisor to avoid division by zero when the
firm is profitable (negative burn rate)."""


# ---------------------------------------------------------------------
# 3.  GDP DELTA COMPONENT  (weight: see difficulty presets)
# ---------------------------------------------------------------------
# G(t) = clip((real_GDP_t - real_GDP_{t-1}) / max(|real_GDP_{t-1}|, floor), -1, 1)
# This is a *relative* delta: 10% growth always scores ~0.1 regardless of
# absolute GDP level.  The floor prevents division by zero at episode start.
#
# The price index P_t is exogenous with inertia:
#   P_t = PRICE_INERTIA * P_{t-1} + (1 - PRICE_INERTIA) * market_avg_price_t

PRICE_INERTIA: float = 0.85
"""Weight on the previous price index.  A shock today moves the index
by at most 15% of the gap per step; full propagation takes ~10 steps."""

INITIAL_PRICE_INDEX: float = 10.0
"""Starting value of the price index.
Set to DEFAULT_PRICE so that real_GDP = nominal_GDP / price_index
is consistent from step 0 (no artificial deflation shock)."""

GDP_FLOOR_EPSILON: float = 1.0
"""Floor for the GDP denominator to prevent division by zero when
previous real GDP is zero (e.g. at episode start before any production).
The denominator is max(|real_gdp_previous|, GDP_FLOOR_EPSILON)."""


# ---------------------------------------------------------------------
# 4.  GINI PENALTY  (weight: always -0.10, never renormalized)
# ---------------------------------------------------------------------
# penalty = (max(gini - GINI_THRESHOLD, 0) / (1 - GINI_THRESHOLD)) ^ GINI_EXP
# Wealth vector entries are floored at 0 before Gini computation.

GINI_THRESHOLD: float = 0.40
"""Gini coefficient below which no penalty applies."""

GINI_EXPONENT: float = 3.0
"""Exponent on the excess above the threshold.  Must be in [2, 5].
  ^2 -> light tax (agent ignores it)
  ^3 -> meaningful pressure at Gini ~ 0.7  (~0.125 penalty)
  ^4 -> numerically explosive past 0.8, causes training instability
We use 3 as the middle ground."""

GINI_WEALTH_INCLUDES_CAPITAL: bool = False
"""If True, wealth = firm_cash + capital_stock.
If False, wealth = firm_cash only (chosen for interpretability --
the LLM agent can see cash directly in the observation)."""

GINI_WEALTH_FLOOR: float = 0.0
"""Floor applied to each agent's wealth before computing Gini.
Prevents negative values (debt) from producing Gini > 1."""


# ---------------------------------------------------------------------
# 5.  DIFFICULTY PRESETS  (weight normalization)
# ---------------------------------------------------------------------
# Rule: positive weights (P, S, G) are normalized to sum to 1.00.
#        Gini weight is always fixed at -0.10.
#        A perfect agent (all components=1.0, zero Gini penalty) scores 1.0.
#        Worst case (all components=0, max Gini penalty) scores -0.1.
#        Total reward range: [-0.10,  1.00]  regardless of difficulty.

GINI_WEIGHT: float = -0.10
"""Fixed Gini penalty weight.  Never changes across presets."""

POSITIVE_WEIGHT_SUM: float = 1.00
"""Target sum of the three positive component weights.
A perfect agent scores exactly POSITIVE_WEIGHT_SUM = 1.0.
Gini penalty can push the score down to GINI_WEIGHT = -0.1."""


@dataclass(frozen=True)
class DifficultyPreset:
    """Immutable container for a single difficulty level."""
    name: str
    episode_length: int
    # Raw (unnormalized) weights -- only their ratios matter
    raw_profit: float
    raw_solvency: float
    raw_gdp: float

    @property
    def _raw_sum(self) -> float:
        return self.raw_profit + self.raw_solvency + self.raw_gdp

    @property
    def w_profit(self) -> float:
        """Normalized profit weight (sums with solvency + gdp to POSITIVE_WEIGHT_SUM)."""
        return round(self.raw_profit / self._raw_sum * POSITIVE_WEIGHT_SUM, 4)

    @property
    def w_solvency(self) -> float:
        """Normalized solvency weight."""
        return round(self.raw_solvency / self._raw_sum * POSITIVE_WEIGHT_SUM, 4)

    @property
    def w_gdp(self) -> float:
        """Normalized GDP-delta weight."""
        return round(self.raw_gdp / self._raw_sum * POSITIVE_WEIGHT_SUM, 4)

    @property
    def w_gini(self) -> float:
        """Gini penalty weight (always fixed)."""
        return GINI_WEIGHT

    @property
    def ideal_runway(self) -> float:
        """Ideal runway in timestep units for this difficulty."""
        return RUNWAY_TARGET_RATIO * self.episode_length

    def weights_dict(self) -> Dict[str, float]:
        return {
            "profit":   self.w_profit,
            "solvency": self.w_solvency,
            "gdp":      self.w_gdp,
            "gini":     self.w_gini,
        }

    def __repr__(self) -> str:
        return (
            f"DifficultyPreset({self.name}: "
            f"P={self.w_profit:.2f}  S={self.w_solvency:.2f}  "
            f"G={self.w_gdp:.2f}  Gini={self.w_gini:.2f}  "
            f"ep_len={self.episode_length}  runway_target={self.ideal_runway:.0f})"
        )


# -- Concrete presets -------------------------------------------------
EASY = DifficultyPreset(
    name="easy",
    episode_length=30,
    raw_profit=0.1,
    raw_solvency=0.7,
    raw_gdp=0.1,
)

MEDIUM = DifficultyPreset(
    name="medium",
    episode_length=30,
    raw_profit=0.6,
    raw_solvency=0.2,
    raw_gdp=0.1,
)

HARD = DifficultyPreset(
    name="hard",
    episode_length=60,
    raw_profit=0.4,
    raw_solvency=0.2,
    raw_gdp=0.3,
)

PRESETS: Dict[str, DifficultyPreset] = {
    "easy":   EASY,
    "medium": MEDIUM,
    "hard":   HARD,
}

DEFAULT_DIFFICULTY: str = "easy"


# ---------------------------------------------------------------------
# 6.  SIMULATOR DEFAULTS
# ---------------------------------------------------------------------
INITIAL_FIRM_CASH: float = 10_000.0
INITIAL_WORKER_COUNT: int = 5
NUM_AGENTS: int = 5

TAX_RATE: float = 0.15
"""Flat tax rate applied to gross profit each step.
Only applied when gross_profit > 0 (firms with losses pay no tax).
gross_profit = revenue - labor_cost - input_cost"""


# ---------------------------------------------------------------------
# 7.  PRODUCTION & MARKET MECHANICS  (Phase 3)
# ---------------------------------------------------------------------

# -- Production --
WORKER_PRODUCTIVITY: float = 100.0
"""Units of output per worker per timestep."""

WAGE_PER_WORKER: float = 500.0
"""Labor cost per worker per timestep."""

INPUT_COST_PER_UNIT: float = 3.0
"""Variable cost per unit sold (materials, energy, etc.)."""

HIRING_COST_PER_WORKER: float = 500.0
"""One-time cost incurred when hiring or firing a worker."""

# -- Market --
BASE_MARKET_DEMAND: float = 10_000.0
"""Total market size in units per timestep at market_demand=1.0.
Increased from 5000 to compensate for the soft demand cap."""

PRICE_ELASTICITY: float = 7.0
"""Exponent controlling demand sensitivity to relative price.
Higher = demand shifts more aggressively to cheaper firms."""

MAX_DEMAND_SHARE: float = 0.40
"""Asymptotic maximum share a single firm can capture.
Used as the ceiling of a SOFT cap (not hard clip):
  share = raw_share / (1 + raw_share / MAX_DEMAND_SHARE)
This preserves gradient at all demand levels."""

DEFAULT_PRICE: float = 10.0
"""Starting price for all agents and NPC reference price."""

PRICE_FLOOR: float = 0.01
"""Minimum allowed price. Prevents division by zero in demand model."""

# -- GDP-demand feedback --
GDP_DEMAND_SENSITIVITY: float = 0.1
"""How strongly total economy wealth affects market demand.
demand_factor = 1 + sensitivity * ln(1 + wealth / reference)
At baseline wealth (50k), this gives ~7% demand boost."""

WEALTH_REFERENCE: float = 50_000.0
"""Reference total wealth for demand scaling.
Set to INITIAL_FIRM_CASH * NUM_AGENTS (initial total firm cash).
If the economy grows, demand rises; if it shrinks, demand falls."""

# -- Worker economy --
WORKER_CONSUMPTION_RATE: float = 0.80
"""Fraction of worker wealth consumed (spent) each step.
The remainder accumulates as worker savings."""

INITIAL_WORKER_WEALTH: float = 50_000.0
"""Starting total worker savings. Set equal to initial total firm
cash so the Gini starts below the penalty threshold."""

# -- Demand noise --
DEMAND_NOISE_STD: float = 0.02
"""Per-firm demand noise standard deviation. Prevents deterministic
exploitation by the agent (adds ~2% randomness to demand)."""

# -- Market demand evolution (mean-reverting) --
MARKET_DEMAND_INITIAL: float = 0.70
MARKET_DEMAND_MEAN: float = 0.70
MARKET_DEMAND_REVERSION: float = 0.10
MARKET_DEMAND_VOLATILITY: float = 0.05
MARKET_DEMAND_MIN: float = 0.20
MARKET_DEMAND_MAX: float = 1.00

# -- Action clamps --
MAX_HIRE_PER_STEP: int = 10
"""Maximum workers that can be hired in a single step.
Firing is limited to current worker_count."""

LOBBY_TAX_CHANGE_MIN: float = -0.10
"""Minimum per-step change to tax rate via lobby action."""

LOBBY_TAX_CHANGE_MAX: float = 0.10
"""Maximum per-step change to tax rate via lobby action."""

# -- Bankruptcy --
BANKRUPTCY_THRESHOLD: float = 0.0
"""firm_cash at or below this after paying expenses triggers termination."""


# ---------------------------------------------------------------------
# 8.  FORMULA REFERENCE  (no code -- just documentation)
# ---------------------------------------------------------------------
#
#  Composite reward:
#    R(t) = w_P * P(t) + w_S * S(t) + w_G * G(t) + w_Gini * Gini_penalty(t)
#    Range: [-0.10, 1.00]  (perfect agent scores 1.0, worst scores -0.1)
#
#  P(t) -- Profit score:
#    net_income   = revenue - labor_cost - input_cost
#                   - max(gross_profit, 0) * TAX_RATE
#                   - (capex / CAPEX_AMORTIZATION_STEPS)
#    ema_revenue  = IHS_EMA_DECAY * ema_prev + (1 - IHS_EMA_DECAY) * revenue
#                   (initialized to REVENUE_EMA_INITIAL at step 0)
#    ratio        = net_income / max(ema_revenue, REVENUE_FLOOR)
#    P(t)         = clip(asinh(ratio) / asinh(IHS_MAX_RATIO), -1, 1)
#
#  S(t) -- Solvency score:
#    burn_rate    = max(total_expenses - revenue, BURN_RATE_EPSILON)
#    runway       = max(firm_cash, 0) / burn_rate    (in timestep units)
#    S(t)         = 1 / (1 + exp(-SOLVENCY_STEEPNESS * (runway - ideal_runway)))
#
#  G(t) -- GDP delta (relative):
#    P_index_t    = PRICE_INERTIA * P_index_{t-1}
#                   + (1 - PRICE_INERTIA) * market_avg_price_t
#    real_GDP_t   = nominal_GDP_t / P_index_t
#    G(t)         = clip((real_GDP_t - real_GDP_{t-1})
#                        / max(|real_GDP_{t-1}|, GDP_FLOOR_EPSILON), -1, 1)
#
#  Gini_penalty(t):
#    wealth_vec   = [max(cash_i, GINI_WEALTH_FLOOR) for i in agents]
#    gini         = discrete_gini(wealth_vec)
#    excess       = max(gini - GINI_THRESHOLD, 0)
#    penalty      = (excess / (1 - GINI_THRESHOLD)) ^ GINI_EXPONENT
#    (Result is in [0, 1].  Multiplied by w_Gini = -0.10.)
#
# ---------------------------------------------------------------------
