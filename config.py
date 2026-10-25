"""
OligoRL -- Configuration
========================
Single source of truth for every mathematical constant and design decision.
All reward functions, the simulator state machine, and the API layer import
from this file.  Change a number here => it changes everywhere.

Design decisions and their rationale are documented inline, section by
section; the README summarizes the model and the PPO experiment.
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

    @property
    def wage_runway_target(self) -> float:
        """Target steps of wages covered by cash (training solvency, 7c)."""
        return WAGE_RUNWAY_TARGET_STEPS

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
"""Default wage per worker per timestep.  Every firm starts at this wage;
the player can change it with the set_wage action (see LABOR MARKET)."""

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

# -- Demand spillover --
SPILLOVER_FRACTION: float = 0.0
"""Rules v2 default: spillover OFF.  PPO showed in rounds 1 and 2 that any
material spillover is exploitable: total demand (~8,000 units/step) is
~2.7x the economy's maximum output (30 workers x 100), and demand depends
only on relative prices, so turned-away customers never disappear and a
premium-priced firm holding idle capacity harvests them.  Known limitation:
with spillover off, a rival's capacity shortage no longer sends its
customers to anyone (no demand coupling between firms).

SPILLOVER_FRACTION_V1 is the round-1 value, used by legacy mode
(npc_growth=False) so the round-1 golden snapshot reproduces exactly.

Fraction of a capacity-constrained firm's unserved demand that spills
to firms with spare capacity.  0.0 disables spillover (fixed per-firm
demand).  Guards (when > 0):
  - spare capacity uses start-of-step headcount (one-step hiring lag),
    so workers poached this step cannot serve spilled customers yet
  - one simultaneous round, split proportional to spare capacity
  - firms whose cash cannot cover this step's wage bill get no spillover"""

SPILLOVER_FRACTION_V1: float = 0.4

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
# 7b. LABOR MARKET
# ---------------------------------------------------------------------
# Workers are a finite pool shared by all firms.  Phase order each step:
#   1. Post wages: player via set_wage (max +/-WAGE_MAX_CHANGE per step),
#      NPCs via their reaction rule (based on LAST step's outcomes).
#   2. market_wage = employment-weighted MEDIAN of posted wages.
#   3. Quits: a firm paying below the median of the OTHER firms by more
#      than QUIT_TOLERANCE loses workers to the unemployed pool:
#        quit_rate = min(MAX_QUIT_RATE, QUIT_SENSITIVITY * (gap - tolerance))
#   4. Hiring from the pool: unemployed workers have reservation wages
#      spread uniformly over [RESERVATION_WAGE_LOW, RESERVATION_WAGE_HIGH]
#      x market_wage (a labor supply curve).  When firms want more workers
#      than are willing, each worker picks a firm by wage-weighted lottery.
#      A firm that just lost quitters cannot rehire this step.
#   5. Poaching: openings still unfilled pull employed workers from firms
#      paying at least POACH_PREMIUM less (job-to-job moves).
# If every firm pays the same wage, nobody moves (original behaviour).
# All labor randomness uses a separate RNG, so the price/demand random
# sequence is identical to the fixed-wage model for a given seed.

LABOR_FORCE_TOTAL: int = 30
"""Total workers in the economy (employed + unemployed).
At start: NUM_AGENTS * INITIAL_WORKER_COUNT = 25 employed, 5 unemployed.
The small pool means growing past ~10 workers requires poaching from NPCs."""

MIN_WAGE: float = 300.0
"""Legal minimum wage.  Lower bound of the set_wage action."""

MAX_WAGE: float = 1000.0
"""Upper bound of the set_wage action."""

WAGE_MAX_CHANGE: float = 0.10
"""Wages are sticky: the player's wage moves at most 10% per step.
Blocks bait-and-switch (hire high, cut to the floor next step)."""

QUIT_TOLERANCE: float = 0.03
"""Workers ignore wage gaps up to 3% below the market.  Prevents
permanent churn from tiny wage differences between firms."""

QUIT_SENSITIVITY: float = 2.0
"""How strongly workers react to being underpaid beyond the tolerance.
Paying 13% below market -> 2 * (13% - 3%) = 20% of workers quit."""

MAX_QUIT_RATE: float = 0.5
"""At most this fraction of a firm's workers can quit in one step."""

RESERVATION_WAGE_LOW: float = 0.85
RESERVATION_WAGE_HIGH: float = 1.00
"""Unemployed workers' reservation wages, as fractions of the market wage.
A firm paying 0.85x market attracts nobody from the pool; at 0.90x a
third of the pool is willing; at market wage everyone is."""

HIRING_WAGE_ELASTICITY: float = 10.0
"""When workers choose between firms, P(firm) ~ (wage / market)^k.
Paying 5% more makes a firm ~1.6x as likely to be picked: higher pay
wins MORE workers, not ALL of them (no +$1 knife-edge)."""

POACH_PREMIUM: float = 0.05
"""Employed workers switch firms only for at least a 5% raise."""

MAX_POACH_RATE: float = 0.4
"""At most this fraction of a firm's workers can be poached per step."""

# NPC wage rule.  Each NPC draws its own parameters per episode from these
# ranges, so NPCs are not identical and the rule is harder to exploit.
NPC_WAGE_STEP_RANGE: tuple = (0.02, 0.04)
"""Raise applied after losing workers or failing to fill openings."""

NPC_WAGE_RELAX_RANGE: tuple = (0.005, 0.015)
"""Cut applied (never below WAGE_PER_WORKER) after a calm streak."""

NPC_RELAX_STEPS_RANGE: tuple = (2, 4)
"""Calm steps (no losses, no unfilled openings) before relaxing."""

NPC_MAX_WAGE_SHARE_RANGE: tuple = (0.85, 0.95)
"""NPCs never pay more than this fraction of a worker's marginal revenue
product: WORKER_PRODUCTIVITY * (price - INPUT_COST_PER_UNIT).
At price $10 and share 0.9 this caps NPC wages at 0.9 * 100 * 7 = $630."""


# ---------------------------------------------------------------------
# 7c. TRAINING SOLVENCY: activity-gated wage runway
# ---------------------------------------------------------------------
# The original solvency score divides cash by net burn rate.  A profitable
# firm has ~zero burn, so runway explodes and S(t) saturates: in practice
# it only checks cash > 0, and a shell firm with no workers scores as
# "safe".  The training reward instead uses:
#   wage_runway = min(cash / wage_bill, CAP_MULT * target)   (steps)
#   S_runway    = sigmoid(STEEPNESS * (wage_runway - target) / target)
#   activity    = min(units_sold / ACTIVITY_REFERENCE_UNITS, 1)
#   S(t)        = S_runway * activity          (multiplicative gate)
# Units SOLD, not produced: hoarding idle workers earns no credit.
# Leaderboards rank on terminal net worth, not on this reward.

WAGE_RUNWAY_TARGET_STEPS: float = 2.0
"""Target: cash covers 2 steps of payroll.  Same on every difficulty --
runway is about surviving a bad step, not about episode length.
Calibrated by sweep: a 6-step target punished hiring for ~20 steps
(growing to 10 workers scored below holding at 5 despite +60% cash)."""

WAGE_RUNWAY_CAP_MULT: float = 2.0
"""Runway beyond 2x target (4 steps) earns nothing extra: solvency flags
distress, it does not reward cash hoarding."""

WAGE_RUNWAY_STEEPNESS: float = 3.0
"""Sigmoid steepness in units of target.  S_runway = 0.05 at zero runway,
0.5 at target, 0.95 at the cap -- the same shape on every difficulty."""

ACTIVITY_REFERENCE_UNITS: float = INITIAL_WORKER_COUNT * WORKER_PRODUCTIVITY
"""Units sold that earn full solvency credit (the starting firm's output)."""

# ---------------------------------------------------------------------
# 7d. EXPERIMENT PROFILES (opt-in; defaults reproduce the leaderboard)
# ---------------------------------------------------------------------
# NPC rule parameters per profile, as lists of (low, high) intervals: a
# value is drawn by picking one interval uniformly at random, then a value
# uniformly inside it (integers for relax_steps).
#   default : the NPC_*_RANGE values above, fixed NPC price anchor $10.
#             Used by the leaderboard and golden test.  Unchanged.
#   train   : each range CONTAINS the default range (wider).
#   heldout : outside the train ranges on BOTH sides -- NPC behaviour
#             the PPO agent never saw, for out-of-distribution eval.

# Round-1 profiles (rules v1, npc_growth=False).  Kept unchanged so the
# round-1 numbers reproduce exactly.
NPC_PROFILES_V1: Dict[str, Dict[str, list]] = {
    "train": {
        "wage_step":      [(0.01, 0.06)],
        "wage_relax":     [(0.0025, 0.025)],
        "relax_steps":    [(1, 6)],
        "max_wage_share": [(0.80, 1.00)],
        "price_anchor":   [(9.0, 11.0)],
    },
    "heldout": {
        "wage_step":      [(0.0, 0.01), (0.06, 0.09)],
        "wage_relax":     [(0.0, 0.0025), (0.025, 0.04)],
        "relax_steps":    [(0, 0), (7, 9)],
        "max_wage_share": [(0.70, 0.80), (1.00, 1.10)],
        "price_anchor":   [(8.0, 9.0), (11.0, 12.0)],
    },
}

# Rules v2 (npc_growth=True).  Train includes price-war NPCs (low price
# anchors), overpaying NPCs (wage cap above marginal revenue product) and
# frozen-headcount NPCs (growth_cap 0, cannot downsize) -- the combination
# goes bankrupt and releases workers.  Heldout stays outside train on both
# sides, except growth_cap, which cannot go below 0 (upper side only).
# Growth keys are described in 7f.
NPC_PROFILES: Dict[str, Dict[str, list]] = {
    "train": {
        "wage_step":       [(0.01, 0.06)],
        "wage_relax":      [(0.0025, 0.025)],
        "relax_steps":     [(1, 6)],
        "max_wage_share":  [(0.70, 1.25)],
        "price_anchor":    [(6.5, 12.5)],
        "growth_cap":      [(0, 3)],
        "growth_fill":     [(0.6, 1.0)],
        "growth_buffer":   [(1.0, 5.0)],
        "growth_margin":   [(0.0, 0.25)],
    },
    "heldout": {
        "wage_step":       [(0.0, 0.01), (0.06, 0.09)],
        "wage_relax":      [(0.0, 0.0025), (0.025, 0.04)],
        "relax_steps":     [(0, 0), (7, 9)],
        "max_wage_share":  [(0.55, 0.70), (1.25, 1.40)],
        "price_anchor":    [(5.5, 6.5), (12.5, 13.5)],
        "growth_cap":      [(4, 6)],
        "growth_fill":     [(0.4, 0.6), (1.0, 1.2)],
        "growth_buffer":   [(0.0, 1.0), (5.0, 7.0)],
        "growth_margin":   [(-0.1, 0.0), (0.25, 0.4)],
    },
}

# ---------------------------------------------------------------------
# 7f. RULES V2: NPC GROWTH + SMOOTH QUITS  (npc_growth=True, the default)
# ---------------------------------------------------------------------
# Round 1 showed NPCs stuck at INITIAL_WORKER_COUNT leave a permanent
# overflow of customers that a premium-priced firm with idle capacity can
# harvest via spillover.  In v2 each NPC sizes its workforce to its own
# demand from the PREVIOUS step (one-step lag):
#   desired  = ceil(fill * last_own_demand / WORKER_PRODUCTIVITY)
#   grow     only if affordable:
#              MRP = WORKER_PRODUCTIVITY * (price - INPUT_COST_PER_UNIT)
#              MRP >= wage * (1 + margin)   and
#              cash >= buffer * (workers + 1) * wage
#   shrink   symmetrically when desired < workers, or when MRP < wage
#   change   at most `cap` workers per step in either direction, never
#            below NPC_MIN_WORKERS; fired workers go to the pool and cost
#            HIRING_COST_PER_WORKER, like the player's
# Each NPC draws (cap, fill, buffer, margin) per episode.
#
# Smooth quits replace the kink at the tolerance edge (round 1: paying
# 0.97x market cost nothing, 0.96x started quits):
#   rate = min(MAX_QUIT_RATE, SENS * w * (softplus((gap - tol) / w)
#                                          - softplus(-tol / w)))   gap > 0
# Zero at equal wages, ~1.3%/step at a 3% gap, matches v1 beyond ~5%.
#
# npc_growth=False reproduces the round-1 rules exactly (golden test).

NPC_GROWTH_CAP_RANGE: tuple = (1, 2)
NPC_GROWTH_FILL_RANGE: tuple = (0.7, 1.0)
NPC_GROWTH_BUFFER_RANGE: tuple = (2.0, 4.0)
NPC_GROWTH_MARGIN_RANGE: tuple = (0.05, 0.2)
NPC_MIN_WORKERS: int = 1

QUIT_SMOOTH_WIDTH: float = 0.01
"""Softplus width w for the v2 quit curve (in units of relative wage gap)."""

# -- Leaderboard (evaluation, not training) --
BANKRUPTCY_PENALTY: float = INITIAL_FIRM_CASH
"""Terminal net worth of a bankrupt firm = final cash - this penalty.
Stops 'die early' from beating 'bleed slowly' on the leaderboard."""


# ---------------------------------------------------------------------
# 7e. PPO TRAINING REWARD (the leaderboard metric, made dense)
# ---------------------------------------------------------------------
#   r_t   = (net_worth_t - net_worth_{t-1}) / INITIAL_FIRM_CASH + hinge_t
#   hinge = -k * max(0, 1 - runway_t / PPO_HINGE_RUNWAY_STEPS)
#   runway_t = cash_t / wage_bill_t     (steps of payroll covered)
#   k     = PPO_HINGE_MAX_TOTAL * (BANKRUPTCY_PENALTY / INITIAL_FIRM_CASH)
#           / episode_length
# With gamma = 1 the deltas telescope: the return is the final net worth
# change (incl. the bankruptcy penalty) plus the hinge.  The hinge is zero
# whenever cash covers a step of wages, and its worst-case total over an
# episode is half the bankruptcy penalty, so it can never outweigh going
# bankrupt.  The legacy composite reward is still computed for display.

PPO_HINGE_RUNWAY_STEPS: float = 1.0
"""Hinge penalty starts when cash covers less than 1 step of wages."""

PPO_HINGE_MAX_TOTAL: float = 0.5
"""Worst-case cumulative hinge, as a fraction of the bankruptcy penalty."""


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
