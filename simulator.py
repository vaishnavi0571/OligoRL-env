"""
OligoRL -- Simulator State Machine
===================================
Pure economic simulation.  No reward logic, no API layer.
Testable in isolation by running step() with hardcoded actions.

Step order within each step() call:
    1. Validate/clamp actions
    2. Generate NPC actions (prices, then wages)
    3. Labor market: firing, quits, hiring from the unemployed pool
    4. Compute wealth-driven demand multiplier
    5. Calculate revenue (all agents, with demand noise)
    6. Pay expenses, update cash
    7. Update worker wealth (wages in, consumption out)
    8. Update tax rate (lobby action)
    9. Compute real GDP (using LAGGED price_index)
   10. Update price index (for next step)
   11. Update market demand (exogenous)
   12. Update EMA revenue
   13. Check termination
   14. Advance time

NPC POLICY:
    - NPCs are "dumb followers" anchored to DEFAULT_PRICE +/- 2% noise
    - NPCs never fire; they reopen positions up to INITIAL_WORKER_COUNT
      when they lose workers
    - NPC wages: raise after losing workers or leaving openings unfilled,
      relax toward WAGE_PER_WORKER after a calm streak, capped at an
      affordable share of a worker's marginal revenue product.  Each NPC
      draws its own rule parameters per episode (no lockstep).
    - A bankrupt NPC exits: its workers return to the pool and it stops
      competing for customers.
    - NPCs never invest (capex = 0) and never lobby (tax_change = 0)
    - NPCs pay identical input/tax rates as the player
    - This creates a stable competitive baseline; the player differentiates
      through pricing, wages, hiring, investment, and tax lobbying.

LABOR MARKET (see config 7b for the full phase order):
    - Workers are finite (LABOR_FORCE_TOTAL): employed + unemployed is
      constant every step.
    - Quits to unemployment when paying clearly below the market median;
      hiring from the pool follows a labor supply curve; unfilled openings
      poach workers from firms paying at least POACH_PREMIUM less.
    - If all firms pay the same wage nobody moves, so the original
      fixed-wage behaviour is reproduced exactly.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
    
from config import (
    INITIAL_FIRM_CASH, INITIAL_WORKER_COUNT, NUM_AGENTS, TAX_RATE,
    DEFAULT_PRICE, INITIAL_PRICE_INDEX,
    WORKER_PRODUCTIVITY, WAGE_PER_WORKER, INPUT_COST_PER_UNIT,
    HIRING_COST_PER_WORKER,
    BASE_MARKET_DEMAND, PRICE_ELASTICITY, MAX_DEMAND_SHARE, PRICE_FLOOR,
    GDP_DEMAND_SENSITIVITY, WEALTH_REFERENCE,
    WORKER_CONSUMPTION_RATE, INITIAL_WORKER_WEALTH, DEMAND_NOISE_STD,
    MARKET_DEMAND_INITIAL, MARKET_DEMAND_MEAN, MARKET_DEMAND_REVERSION,
    MARKET_DEMAND_VOLATILITY, MARKET_DEMAND_MIN, MARKET_DEMAND_MAX,
    PRICE_INERTIA,
    REVENUE_EMA_INITIAL, IHS_EMA_DECAY,
    MAX_HIRE_PER_STEP, LOBBY_TAX_CHANGE_MIN, LOBBY_TAX_CHANGE_MAX,
    BANKRUPTCY_THRESHOLD,
    LABOR_FORCE_TOTAL, MIN_WAGE, MAX_WAGE, WAGE_MAX_CHANGE,
    QUIT_TOLERANCE, QUIT_SENSITIVITY, MAX_QUIT_RATE,
    RESERVATION_WAGE_LOW, RESERVATION_WAGE_HIGH, HIRING_WAGE_ELASTICITY,
    POACH_PREMIUM, MAX_POACH_RATE,
    NPC_WAGE_STEP_RANGE, NPC_WAGE_RELAX_RANGE, NPC_RELAX_STEPS_RANGE,
    NPC_MAX_WAGE_SHARE_RANGE,
    SPILLOVER_FRACTION, SPILLOVER_FRACTION_V1, NPC_PROFILES, NPC_PROFILES_V1,
    NPC_GROWTH_CAP_RANGE, NPC_GROWTH_FILL_RANGE, NPC_GROWTH_BUFFER_RANGE,
    NPC_GROWTH_MARGIN_RANGE, NPC_MIN_WORKERS, QUIT_SMOOTH_WIDTH,
    PRESETS, DEFAULT_DIFFICULTY,
)
from rewards import update_ema


# -----------------------------------------------------------------
# Pure functions (no state)
# -----------------------------------------------------------------

def compute_demand_share(
    agent_price: float,
    competitor_avg_price: float,
    num_agents: int = NUM_AGENTS,
    elasticity: float = PRICE_ELASTICITY,
    max_share: float = MAX_DEMAND_SHARE,
) -> float:
    """Fraction of total market demand captured by one agent.

    Uses a SOFT cap: share = raw / (1 + raw / max_share).
    This asymptotically approaches max_share without a hard cliff,
    preserving the gradient at all demand levels.
    """
    effective_price = max(agent_price, PRICE_FLOOR)
    if competitor_avg_price <= 0:
        competitor_avg_price = effective_price
    price_ratio = competitor_avg_price / effective_price
    fair_share = 1.0 / num_agents
    raw_share = fair_share * (price_ratio ** elasticity)
    # Soft cap (asymptotic above 75% of max_share to preserve baseline elasticity)
    threshold = max_share * 0.75
    if raw_share <= threshold:
        return raw_share
    else:
        import math
        margin = max_share - threshold
        excess = raw_share - threshold
        return threshold + margin * (1.0 - math.exp(-excess / margin))


def compute_revenue(
    set_price: float,
    worker_count: int,
    market_demand: float,
    competitor_avg_price: float,
    demand_multiplier: float = 1.0,
    num_agents: int = NUM_AGENTS,
    rng: Optional[random.Random] = None,
    return_demand: bool = False,
):
    """Compute production and sales for one agent.

    Args:
        demand_multiplier: wealth/GDP-driven scaling of total demand.
        rng: if provided, adds small per-firm demand noise.
        return_demand: also return the firm's potential demand (units),
            used by the spillover stage.

    Returns:
        (max_output, units_sold, revenue), plus demand if return_demand
    """
    max_output = float(worker_count * WORKER_PRODUCTIVITY)
    share = compute_demand_share(set_price, competitor_avg_price, num_agents)
    potential_demand = BASE_MARKET_DEMAND * market_demand * demand_multiplier * share

    # Per-firm demand noise (prevents deterministic exploitation)
    if rng is not None:
        noise = 1.0 + rng.gauss(0, DEMAND_NOISE_STD)
        potential_demand *= max(noise, 0.0)  # clamp noise to non-negative

    units_sold = min(max_output, max(0.0, potential_demand))

    # Supply constraint: units_sold can NEVER exceed production capacity
    assert units_sold <= max_output + 0.001, (
        f"Supply constraint violation: sold {units_sold:.2f} > capacity {max_output:.2f}"
    )

    effective_price = max(set_price, PRICE_FLOOR)
    revenue = units_sold * effective_price
    if return_demand:
        return max_output, units_sold, revenue, max(0.0, potential_demand)
    return max_output, units_sold, revenue


def compute_wealth_factor(total_wealth: float) -> float:
    """GDP-demand feedback: wealthier economy generates more demand."""
    return 1.0 + GDP_DEMAND_SENSITIVITY * math.log1p(
        max(total_wealth, 0.0) / WEALTH_REFERENCE
    )


def weighted_median(pairs: List[Tuple[float, int]]) -> float:
    """Median wage weighted by headcount.  pairs = [(wage, workers), ...].

    Robust to a tiny firm posting an extreme wage.  Falls back to
    WAGE_PER_WORKER when nobody is employed.
    """
    total = sum(w for _, w in pairs)
    if total <= 0:
        return WAGE_PER_WORKER
    cum = 0
    for wage, weight in sorted(pairs):
        cum += weight
        if cum * 2 >= total:
            return wage
    return pairs[-1][0]


def reservation_share(wage: float, market_wage: float) -> float:
    """Fraction of the unemployed willing to work at `wage` (labor supply)."""
    ratio = wage / market_wage
    span = RESERVATION_WAGE_HIGH - RESERVATION_WAGE_LOW
    return min(max((ratio - RESERVATION_WAGE_LOW) / span, 0.0), 1.0)


def _softplus(x: float) -> float:
    return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))


def quit_rate(gap: float, smooth: bool) -> float:
    """Share of a firm's workers that quit, given its relative wage gap
    below the reference wage (gap > 0 means underpaying).

    v1 (smooth=False): zero inside QUIT_TOLERANCE, linear beyond (a kink).
    v2 (smooth=True):  softplus-smoothed version of the same curve, zero
                       at gap = 0 (config 7f).
    """
    if gap <= 1e-9:
        return 0.0
    if not smooth:
        excess = gap - QUIT_TOLERANCE
        return 0.0 if excess <= 0 else min(MAX_QUIT_RATE, QUIT_SENSITIVITY * excess)
    w = QUIT_SMOOTH_WIDTH
    soft = _softplus((gap - QUIT_TOLERANCE) / w) - _softplus(-QUIT_TOLERANCE / w)
    return min(MAX_QUIT_RATE, QUIT_SENSITIVITY * w * soft)


def stochastic_round(x: float, rng: random.Random) -> int:
    """Round x up with probability equal to its fractional part."""
    base = int(x)
    frac = x - base
    if frac > 0 and rng.random() < frac:
        base += 1
    return base


# -----------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------

@dataclass
class FirmState:
    """Per-agent financial state."""
    cash: float = INITIAL_FIRM_CASH
    worker_count: int = INITIAL_WORKER_COUNT
    price: float = DEFAULT_PRICE
    capital_stock: float = 0.0
    wage: float = WAGE_PER_WORKER
    active: bool = True      # False once an NPC goes bankrupt and exits
    last_lost: int = 0       # workers lost (quit + poached) last step
    last_unfilled: int = 0   # openings left unfilled last step
    # NPC wage-rule parameters (drawn per episode in reset())
    calm_steps: int = 0
    wage_step: float = 0.03
    wage_relax: float = 0.01
    relax_steps: int = 3
    max_wage_share: float = 0.9
    price_anchor: float = DEFAULT_PRICE   # NPC prices = anchor x (1 +/- 2% noise)
    # NPC growth rule (rules v2, config 7f)
    growth_cap: int = 1
    growth_fill: float = 1.0
    growth_buffer: float = 3.0
    growth_margin: float = 0.1
    target_workers: int = INITIAL_WORKER_COUNT
    last_demand: float = 0.0              # own demand last step (lag input)


@dataclass
class StepInfo:
    """Per-step details for reward computation (Phase 4) and debugging."""
    # Player financials
    revenue: float = 0.0
    labor_cost: float = 0.0
    input_cost: float = 0.0
    hiring_cost: float = 0.0
    capex_spent: float = 0.0
    gross_profit: float = 0.0
    tax_paid: float = 0.0
    total_expenses: float = 0.0
    net_cash_flow: float = 0.0
    # Production
    units_sold: float = 0.0     # includes spillover units
    max_output: float = 0.0
    demand: float = 0.0         # player's own potential demand (units)
    spill_units: float = 0.0    # extra units sold from rivals' unserved demand
    spill_pool: float = 0.0     # total units spilled this step
    total_units_all: float = 0.0  # units sold by all firms (incl. spillover)
    total_demand_all: float = 0.0  # own potential demand summed over all firms
    # Labor market (player)
    wage: float = 0.0
    market_wage: float = 0.0
    hires_requested: int = 0
    hires_filled: int = 0    # from the pool + poached from rivals
    poached_in: int = 0      # of hires_filled, taken from rivals
    poached_out: int = 0     # player workers taken by rivals
    fired: int = 0
    quits: int = 0           # player workers who quit to unemployment
    unemployed: int = 0
    # Labor market (all firms, player first)
    firm_wages: List[float] = field(default_factory=list)
    firm_workers: List[int] = field(default_factory=list)
    firm_quits: List[int] = field(default_factory=list)
    firm_poached_out: List[int] = field(default_factory=list)
    firm_hires: List[int] = field(default_factory=list)
    firm_fired: List[int] = field(default_factory=list)
    firm_active: List[bool] = field(default_factory=list)
    labor_flows: List[List[int]] = field(default_factory=list)
    """[src, dst, count] moves this step; -1 = unemployed pool."""
    # Market / macro
    real_gdp_current: float = 0.0
    real_gdp_previous: float = 0.0
    market_avg_price: float = 0.0
    competitor_avg_price: float = 0.0
    demand_multiplier: float = 1.0
    # Aggregates (for conservation checks)
    total_revenue_all: float = 0.0
    total_input_cost_all: float = 0.0
    total_tax_all: float = 0.0
    total_wages_all: float = 0.0
    total_capex_all: float = 0.0
    total_hiring_all: float = 0.0
    worker_consumption: float = 0.0
    # Wealth (for Gini: firms + worker pool as last entry)
    all_agent_wealth: List[float] = field(default_factory=list)
    # Termination
    terminated: bool = False
    truncated: bool = False


# -----------------------------------------------------------------
# Simulator
# -----------------------------------------------------------------

class EconSimulator:
    """Economic simulator state machine."""

    def __init__(
        self,
        difficulty: str = DEFAULT_DIFFICULTY,
        seed: int = 42,
        spillover_fraction: Optional[float] = None,
        npc_profile: str = "default",
        lobbying_enabled: bool = True,
        npc_growth: bool = True,
        smooth_quits: Optional[bool] = None,
    ):
        """Experiment flags:
            npc_profile:      "default" | "train" | "heldout" (config 7d)
            lobbying_enabled: False ignores every lobby_tax_change action
            npc_growth:       rules v2 (config 7f, the default; spillover
                              off).  False reproduces the round-1 rules
                              exactly (incl. spillover 0.4).
            smooth_quits:     v2 quit curve; defaults to npc_growth
        """
        self.profiles = NPC_PROFILES if npc_growth else NPC_PROFILES_V1
        if npc_profile != "default" and npc_profile not in self.profiles:
            raise ValueError(f"Unknown npc_profile '{npc_profile}'")
        self.preset = PRESETS[difficulty]
        self._seed = seed
        self.npc_profile = npc_profile
        self.lobbying_enabled = lobbying_enabled
        self.npc_growth = npc_growth
        self.smooth_quits = npc_growth if smooth_quits is None else smooth_quits
        default_spill = SPILLOVER_FRACTION if npc_growth else SPILLOVER_FRACTION_V1
        self.spillover_fraction = (
            default_spill if spillover_fraction is None else spillover_fraction
        )
        self.last_units_sold: float = 0.0
        self.last_capacity: float = 0.0
        self.last_demand: float = 0.0
        self.rng = random.Random(seed)
        self.player: Optional[FirmState] = None
        self.npc_firms: List[FirmState] = []
        self.price_index: float = INITIAL_PRICE_INDEX
        self.market_demand: float = MARKET_DEMAND_INITIAL
        self.tax_rate: float = TAX_RATE
        self.ema_revenue: float = REVENUE_EMA_INITIAL
        self.real_gdp_previous: float = 0.0
        self.worker_wealth: float = INITIAL_WORKER_WEALTH
        self.labor_rng = random.Random(seed * 7919 + 1)
        self.unemployed: int = 0
        self.market_wage: float = WAGE_PER_WORKER
        self.step_count: int = 0
        self.terminated: bool = False
        self.truncated: bool = False

    def reset(self, seed: Optional[int] = None) -> Dict:
        """Initialize a fresh episode.  Returns observation dict."""
        if seed is not None:
            self._seed = seed
        self.rng = random.Random(self._seed)
        # Separate stream so labor randomness never shifts price/demand noise
        self.labor_rng = random.Random(self._seed * 7919 + 1)

        npc_count = NUM_AGENTS - 1
        self.player = FirmState()
        self.npc_firms = [FirmState() for _ in range(npc_count)]
        for npc in self.npc_firms:
            npc.wage_step = self.labor_rng.uniform(*NPC_WAGE_STEP_RANGE)
            npc.wage_relax = self.labor_rng.uniform(*NPC_WAGE_RELAX_RANGE)
            npc.relax_steps = self.labor_rng.randint(*NPC_RELAX_STEPS_RANGE)
            npc.max_wage_share = self.labor_rng.uniform(*NPC_MAX_WAGE_SHARE_RANGE)
        if self.npc_growth:
            # Separate stream: growth draws never shift the v1 sequences
            grng = random.Random(self._seed * 15485863 + 3)
            for npc in self.npc_firms:
                npc.growth_cap = grng.randint(*NPC_GROWTH_CAP_RANGE)
                npc.growth_fill = grng.uniform(*NPC_GROWTH_FILL_RANGE)
                npc.growth_buffer = grng.uniform(*NPC_GROWTH_BUFFER_RANGE)
                npc.growth_margin = grng.uniform(*NPC_GROWTH_MARGIN_RANGE)
        if self.npc_profile != "default":
            # Separate stream: profile draws never shift the default sequence
            self._draw_npc_profile(random.Random(self._seed * 104729 + 7))
        self.price_index = INITIAL_PRICE_INDEX
        self.market_demand = MARKET_DEMAND_INITIAL
        self.tax_rate = TAX_RATE
        self.ema_revenue = REVENUE_EMA_INITIAL
        self.worker_wealth = INITIAL_WORKER_WEALTH
        self.unemployed = max(LABOR_FORCE_TOTAL - NUM_AGENTS * INITIAL_WORKER_COUNT, 0)
        self.market_wage = WAGE_PER_WORKER
        self.last_units_sold = 0.0
        self.last_capacity = float(INITIAL_WORKER_COUNT * WORKER_PRODUCTIVITY)
        self.last_demand = 0.0
        self.step_count = 0
        self.terminated = False
        self.truncated = False

        # Initial GDP estimate: all agents at capacity, default price
        output_per_agent = INITIAL_WORKER_COUNT * WORKER_PRODUCTIVITY
        self.real_gdp_previous = (
            NUM_AGENTS * output_per_agent * DEFAULT_PRICE / self.price_index
        )

        return self._build_observation()

    def step(self, action: Dict) -> Tuple[Dict, StepInfo]:
        """Execute one timestep."""
        if self.player is None:
            raise RuntimeError("Call reset() before step()")
        if self.terminated or self.truncated:
            raise RuntimeError("Episode is over. Call reset().")

        p = self.player
        info = StepInfo()
        info.real_gdp_previous = self.real_gdp_previous  # capture before update

        # ===== 1. Validate/clamp actions =====
        set_price = max(float(action.get("set_price", p.price)), PRICE_FLOOR)
        hire = int(action.get("hire_workers", 0))
        hire = max(-p.worker_count, min(MAX_HIRE_PER_STEP, hire))
        capex = max(float(action.get("capex", 0.0)), 0.0)
        lobby = float(action.get("lobby_tax_change", 0.0)) if self.lobbying_enabled else 0.0
        lobby = max(LOBBY_TAX_CHANGE_MIN, min(LOBBY_TAX_CHANGE_MAX, lobby))
        wage = action.get("set_wage")
        wage = p.wage if wage is None else float(wage)
        wage = max(p.wage * (1 - WAGE_MAX_CHANGE), min(p.wage * (1 + WAGE_MAX_CHANGE), wage))
        wage = max(MIN_WAGE, min(MAX_WAGE, wage))

        # ===== 2. Generate NPC actions =====
        npc_prices = []
        for npc in self.npc_firms:
            noise = 1.0 + self.rng.gauss(0, 0.02)
            npc.price = max(npc.price_anchor * noise, PRICE_FLOOR)
            npc_prices.append(npc.price)
        self._update_npc_wages()
        if self.npc_growth:
            self._update_npc_targets()

        # ===== 3. Labor market: firing, quits, hiring, poaching =====
        firms = [p] + self.npc_firms
        start_workers = [f.worker_count for f in firms]  # for spillover lag
        start_cash = [f.cash for f in firms]
        p.wage = wage
        p.price = set_price
        self._run_labor_market(hire, info)
        info.hiring_cost = (info.hires_filled + info.fired) * HIRING_COST_PER_WORKER

        # ===== 4. Compute wealth-driven demand multiplier =====
        total_economy_wealth = (
            p.cash + sum(npc.cash for npc in self.npc_firms) + self.worker_wealth
        )
        info.demand_multiplier = compute_wealth_factor(total_economy_wealth)

        # ===== 5. Calculate revenue (all agents) =====
        # Exited (bankrupt) NPCs no longer compete for customers
        all_prices = [set_price] + npc_prices
        active = [npc.active for npc in self.npc_firms]
        active_npc_prices = [pr for pr, a in zip(npc_prices, active) if a]
        n_competing = 1 + len(active_npc_prices)

        # Player revenue
        competitor_avg = (
            sum(active_npc_prices) / len(active_npc_prices)
            if active_npc_prices else DEFAULT_PRICE
        )
        info.competitor_avg_price = competitor_avg
        info.max_output, info.units_sold, info.revenue, info.demand = compute_revenue(
            set_price, p.worker_count, self.market_demand, competitor_avg,
            demand_multiplier=info.demand_multiplier, num_agents=n_competing,
            rng=self.rng, return_demand=True,
        )

        # NPC revenue
        npc_revenues = []
        npc_units_sold_list = []
        npc_demands = []
        for i, npc in enumerate(self.npc_firms):
            if not npc.active:
                npc_revenues.append(0.0)
                npc_units_sold_list.append(0.0)
                npc_demands.append(0.0)
                continue
            others = [set_price] + [
                pr for j, pr in enumerate(npc_prices) if j != i and active[j]
            ]
            other_avg = sum(others) / len(others)
            _, n_sold, n_rev, n_dem = compute_revenue(
                npc.price, npc.worker_count, self.market_demand, other_avg,
                demand_multiplier=info.demand_multiplier, num_agents=n_competing,
                rng=self.rng, return_demand=True,
            )
            npc_revenues.append(n_rev)
            npc_units_sold_list.append(n_sold)
            npc_demands.append(n_dem)
            npc.last_demand = n_dem   # growth-rule input for next step

        # ===== 5b. Demand spillover =====
        if self.spillover_fraction > 0:
            sold = [info.units_sold] + npc_units_sold_list
            revenues = [info.revenue] + npc_revenues
            demands = [info.demand] + npc_demands
            spill = self._spill_demand(firms, sold, demands, start_workers, start_cash)
            info.spill_pool = sum(spill)
            for i, extra in enumerate(spill):
                if extra <= 0:
                    continue
                sold[i] += extra
                revenues[i] += extra * max(firms[i].price, PRICE_FLOOR)
            info.spill_units = spill[0]
            info.units_sold, info.revenue = sold[0], revenues[0]
            npc_units_sold_list, npc_revenues = sold[1:], revenues[1:]

        info.total_units_all = info.units_sold + sum(npc_units_sold_list)
        info.total_demand_all = info.demand + sum(npc_demands)

        # ===== 6. Pay expenses, update cash =====
        # Player
        info.labor_cost = p.worker_count * p.wage
        info.input_cost = info.units_sold * INPUT_COST_PER_UNIT
        info.gross_profit = info.revenue - info.labor_cost - info.input_cost
        info.tax_paid = max(info.gross_profit, 0.0) * self.tax_rate
        info.capex_spent = capex
        p.capital_stock += capex

        info.total_expenses = (
            info.labor_cost + info.input_cost + info.tax_paid
            + info.capex_spent + info.hiring_cost
        )
        info.net_cash_flow = info.revenue - info.total_expenses
        p.cash += info.net_cash_flow

        # NPCs
        total_npc_wages = 0.0
        total_npc_input = 0.0
        total_npc_tax = 0.0
        total_npc_hiring = 0.0
        for i, npc in enumerate(self.npc_firms):
            npc_labor = npc.worker_count * npc.wage
            npc_input = npc_units_sold_list[i] * INPUT_COST_PER_UNIT
            npc_hiring = (info.firm_hires[i + 1] + info.firm_fired[i + 1]) * HIRING_COST_PER_WORKER
            npc_gross = npc_revenues[i] - npc_labor - npc_input
            npc_tax = max(npc_gross, 0.0) * self.tax_rate
            npc.cash += npc_revenues[i] - npc_labor - npc_input - npc_tax - npc_hiring
            total_npc_wages += npc_labor
            total_npc_input += npc_input
            total_npc_tax += npc_tax
            total_npc_hiring += npc_hiring

        # NPC exit: a bankrupt NPC releases its workers and leaves the market
        for i, npc in enumerate(self.npc_firms):
            if npc.active and npc.cash <= BANKRUPTCY_THRESHOLD:
                npc.active = False
                if npc.worker_count:
                    info.labor_flows.append([i + 1, -1, npc.worker_count])
                self.unemployed += npc.worker_count
                npc.worker_count = 0
                info.unemployed = self.unemployed
                info.firm_workers[i + 1] = 0
                info.firm_active[i + 1] = False

        # ===== 7. Update worker wealth =====
        info.total_wages_all = info.labor_cost + total_npc_wages
        self.worker_wealth += info.total_wages_all
        info.worker_consumption = self.worker_wealth * WORKER_CONSUMPTION_RATE
        self.worker_wealth -= info.worker_consumption

        # Aggregate accounting (for conservation checks)
        info.total_revenue_all = info.revenue + sum(npc_revenues)
        info.total_input_cost_all = info.input_cost + total_npc_input
        info.total_tax_all = info.tax_paid + total_npc_tax
        info.total_capex_all = info.capex_spent  # NPCs don't invest
        info.total_hiring_all = info.hiring_cost + total_npc_hiring

        # ===== 8. Update tax rate =====
        self.tax_rate = max(0.0, min(1.0, self.tax_rate + lobby))

        # ===== 9. Compute real GDP using LAGGED price_index =====
        all_sold = [info.units_sold] + npc_units_sold_list
        total_sold = sum(all_sold)
        if total_sold > 0:
            info.market_avg_price = sum(
                all_prices[i] * all_sold[i] for i in range(len(all_prices))
            ) / total_sold
        else:
            info.market_avg_price = DEFAULT_PRICE

        nominal_gdp = info.total_revenue_all
        # Use OLD price_index (from previous step) to deflate — avoids
        # circular dependency where inflation dampens itself instantly
        info.real_gdp_current = nominal_gdp / max(self.price_index, 0.001)

        # ===== 10. Update price index (for next step) =====
        self.price_index = (
            PRICE_INERTIA * self.price_index
            + (1 - PRICE_INERTIA) * info.market_avg_price
        )

        # ===== 11. Update market demand =====
        drift = MARKET_DEMAND_REVERSION * (MARKET_DEMAND_MEAN - self.market_demand)
        shock = self.rng.gauss(0, MARKET_DEMAND_VOLATILITY)
        self.market_demand = max(
            MARKET_DEMAND_MIN,
            min(MARKET_DEMAND_MAX, self.market_demand + drift + shock),
        )

        # ===== 12. Update EMA revenue =====
        self.ema_revenue = update_ema(self.ema_revenue, info.revenue, IHS_EMA_DECAY)

        # ===== 13. Check termination =====
        # Wealth vector: firms + worker pool (as aggregate entry)
        info.all_agent_wealth = (
            [p.cash] + [npc.cash for npc in self.npc_firms] + [self.worker_wealth]
        )

        if p.cash <= BANKRUPTCY_THRESHOLD:
            self.terminated = True
            info.terminated = True
        if self.step_count + 1 >= self.preset.episode_length:
            self.truncated = True
            info.truncated = True

        # ===== 14. Advance time =====
        self.real_gdp_previous = info.real_gdp_current
        self.last_units_sold = info.units_sold
        self.last_capacity = info.max_output
        self.last_demand = info.demand
        self.step_count += 1

        return self._build_observation(), info

    def _draw_npc_profile(self, rng: random.Random) -> None:
        """Overwrite NPC rule parameters with draws from an experiment profile."""
        ranges = self.profiles[self.npc_profile]

        def draw(key: str, integer: bool = False):
            lo, hi = rng.choice(ranges[key])
            return rng.randint(lo, hi) if integer else rng.uniform(lo, hi)

        for npc in self.npc_firms:
            npc.wage_step = draw("wage_step")
            npc.wage_relax = draw("wage_relax")
            npc.relax_steps = draw("relax_steps", integer=True)
            npc.max_wage_share = draw("max_wage_share")
            npc.price_anchor = draw("price_anchor")
            if self.npc_growth:
                npc.growth_cap = draw("growth_cap", integer=True)
                npc.growth_fill = draw("growth_fill")
                npc.growth_buffer = draw("growth_buffer")
                npc.growth_margin = draw("growth_margin")

    def _update_npc_targets(self) -> None:
        """Rules v2: NPC headcount follows last step's own demand (config 7f)."""
        for npc in self.npc_firms:
            if not npc.active:
                continue
            w = npc.worker_count
            cap = npc.growth_cap
            mrp = WORKER_PRODUCTIVITY * max(npc.price - INPUT_COST_PER_UNIT, 0.0)
            if npc.last_demand > 0:
                desired = max(NPC_MIN_WORKERS,
                              math.ceil(npc.growth_fill * npc.last_demand / WORKER_PRODUCTIVITY))
            else:
                desired = w
            affordable = (mrp >= npc.wage * (1.0 + npc.growth_margin)
                          and npc.cash >= npc.growth_buffer * (w + 1) * npc.wage)
            if mrp < npc.wage:
                target = max(NPC_MIN_WORKERS, w - cap)       # losing money per worker
            elif desired > w:
                target = min(w + cap, desired) if affordable else w
            elif desired < w:
                target = max(NPC_MIN_WORKERS, w - cap, desired)
            else:
                target = w
            npc.target_workers = target

    def _spill_demand(
        self,
        firms: List[FirmState],
        sold: List[float],
        demands: List[float],
        start_workers: List[int],
        start_cash: List[float],
    ) -> List[float]:
        """Extra units each firm sells from rivals' unserved demand.

        One simultaneous round:
          pool  = spillover_fraction * sum(unserved demand)
          spare = start-of-step capacity - units already sold
                  (hiring lag: workers gained this step don't count)
          accept_i = min(1, (p_ref / p_i) ^ PRICE_ELASTICITY)
                  (turned-away customers are still price-sensitive;
                   p_ref = their original firms' unserved-weighted price)
          firm i gets min(spare_i, pool * spare_i / total_spare * accept_i)
        Firms that can't cover this step's wage bill from start-of-step
        cash get nothing.  Undistributed spill is lost.
        """
        n = len(firms)
        unserved = [
            max(demands[i] - sold[i], 0.0) if firms[i].active else 0.0
            for i in range(n)
        ]
        pool = self.spillover_fraction * sum(unserved)
        if pool <= 0:
            return [0.0] * n
        p_ref = sum(unserved[i] * firms[i].price for i in range(n)) / sum(unserved)
        spare = [0.0] * n
        for i, f in enumerate(firms):
            if not f.active or start_cash[i] < f.worker_count * f.wage:
                continue
            lagged_capacity = min(start_workers[i], f.worker_count) * WORKER_PRODUCTIVITY
            spare[i] = max(lagged_capacity - sold[i], 0.0)
        total_spare = sum(spare)
        if total_spare <= 0:
            return [0.0] * n
        result = []
        for i, f in enumerate(firms):
            accept = min(1.0, (p_ref / max(f.price, PRICE_FLOOR)) ** PRICE_ELASTICITY)
            result.append(min(spare[i], pool * spare[i] / total_spare * accept))
        return result

    def _update_npc_wages(self) -> None:
        """NPC wage rule, reacting to LAST step's labor outcomes.

        Raise after losing workers or failing to fill openings; relax
        toward the default after a calm streak; never pay more than an
        affordable share of a worker's marginal revenue product.
        """
        for npc in self.npc_firms:
            if not npc.active:
                continue
            mrp = WORKER_PRODUCTIVITY * max(npc.price - INPUT_COST_PER_UNIT, 0.0)
            cap = max(npc.max_wage_share * mrp, MIN_WAGE)
            if npc.last_lost > 0 or npc.last_unfilled > 0:
                npc.wage *= 1.0 + npc.wage_step
                npc.calm_steps = 0
            else:
                npc.calm_steps += 1
                if npc.calm_steps >= npc.relax_steps:
                    npc.wage = max(WAGE_PER_WORKER, npc.wage * (1.0 - npc.wage_relax))
            npc.wage = max(MIN_WAGE, min(cap, npc.wage))

    def _run_labor_market(self, hire: int, info: StepInfo) -> None:
        """Move workers between firms and the unemployed pool.

        Wages are already posted.  Phase order (see config 7b):
          1. Player firing -> pool
          2. Quits: underpaid workers (beyond tolerance) -> pool
          3. Hiring from the pool (labor supply curve + wage lottery)
          4. Poaching: unfilled openings pull workers from lower payers
        Fills the labor fields of `info`.
        """
        firms = [self.player] + self.npc_firms
        n = len(firms)
        rng = self.labor_rng
        POOL = -1
        quits = [0] * n
        poached_out = [0] * n
        hires = [0] * n
        poached_in = [0] * n
        flows: Dict[Tuple[int, int], int] = {}

        def move(src: int, dst: int, k: int = 1) -> None:
            flows[(src, dst)] = flows.get((src, dst), 0) + k

        # 1. Firing: fired workers go straight to the pool.  The player fires
        #    by action; in rules v2 NPCs downsize toward their target.
        fired_all = [0] * n
        fired = max(-hire, 0)
        if fired:
            self.player.worker_count -= fired
            self.unemployed += fired
            fired_all[0] = fired
            move(0, POOL, fired)
        if self.npc_growth:
            for i, npc in enumerate(self.npc_firms, start=1):
                cut = max(npc.worker_count - npc.target_workers, 0) if npc.active else 0
                if cut:
                    npc.worker_count -= cut
                    self.unemployed += cut
                    fired_all[i] = cut
                    move(i, POOL, cut)

        market = weighted_median(
            [(f.wage, f.worker_count) for f in firms if f.active]
        )

        # 2. Quits: compare each firm to the median of the OTHER firms, so
        #    a firm can't move its own reference wage
        for i, f in enumerate(firms):
            if not f.active or f.worker_count == 0:
                continue
            ref = weighted_median([
                (g.wage, g.worker_count)
                for j, g in enumerate(firms) if j != i and g.active
            ])
            rate = quit_rate((ref - f.wage) / ref, self.smooth_quits)
            if rate <= 0:
                continue
            q = min(stochastic_round(f.worker_count * rate, rng), f.worker_count)
            if q:
                f.worker_count -= q
                self.unemployed += q
                quits[i] = q
                move(i, POOL, q)

        # Openings: player's request; NPCs fill up to their target headcount
        # (v1: always INITIAL_WORKER_COUNT; v2: demand-driven target)
        openings = [max(hire, 0)] + [
            max((npc.target_workers if self.npc_growth else INITIAL_WORKER_COUNT)
                - npc.worker_count, 0) if npc.active else 0
            for npc in self.npc_firms
        ]

        # 3. Hiring from the pool.  Each firm can only reach the share of the
        #    pool willing to work at its wage; a firm that just lost
        #    quitters cannot rehire this step.
        pool_start = self.unemployed
        reach = [
            0 if quits[i] or not openings[i] else min(
                openings[i],
                int(pool_start * reservation_share(f.wage, market) + 1e-9),
            )
            for i, f in enumerate(firms)
        ]
        if sum(reach) <= self.unemployed:
            for i in range(n):
                hires[i] = reach[i]
                if reach[i]:
                    move(POOL, i, reach[i])
        else:
            # Oversubscribed: each worker picks a firm, P ~ (wage/market)^k
            while self.unemployed - sum(hires) > 0:
                cands = [i for i in range(n) if hires[i] < reach[i]]
                if not cands:
                    break
                weights = [(firms[i].wage / market) ** HIRING_WAGE_ELASTICITY for i in cands]
                i = rng.choices(cands, weights=weights)[0]
                hires[i] += 1
                move(POOL, i)
        for i, f in enumerate(firms):
            f.worker_count += hires[i]
        self.unemployed -= sum(hires)

        # 4. Poaching: openings still unfilled pull employed workers from
        #    firms paying at least POACH_PREMIUM less
        remaining = [openings[i] - hires[i] for i in range(n)]
        if any(r > 0 for r in remaining):
            poachable = [
                stochastic_round(MAX_POACH_RATE * f.worker_count, rng) if f.active else 0
                for f in firms
            ]
            while True:
                pairs = [
                    (i, j) for i in range(n) if remaining[i] > 0
                    for j in range(n)
                    if j != i and poachable[j] > 0 and firms[j].worker_count > 0
                    and firms[i].wage >= firms[j].wage * (1 + POACH_PREMIUM)
                ]
                if not pairs:
                    break
                weights = [
                    (firms[i].wage / firms[j].wage) ** HIRING_WAGE_ELASTICITY
                    for i, j in pairs
                ]
                i, j = rng.choices(pairs, weights=weights)[0]
                firms[j].worker_count -= 1
                firms[i].worker_count += 1
                poachable[j] -= 1
                remaining[i] -= 1
                poached_out[j] += 1
                poached_in[i] += 1
                hires[i] += 1
                move(j, i)

        for i, f in enumerate(firms):
            f.last_lost = quits[i] + poached_out[i]
            f.last_unfilled = max(openings[i] - hires[i], 0)
        self.market_wage = market

        info.wage = self.player.wage
        info.market_wage = market
        info.hires_requested = max(hire, 0)
        info.hires_filled = hires[0]
        info.poached_in = poached_in[0]
        info.poached_out = poached_out[0]
        info.fired = fired
        info.quits = quits[0]
        info.unemployed = self.unemployed
        info.firm_wages = [f.wage for f in firms]
        info.firm_workers = [f.worker_count for f in firms]
        info.firm_quits = quits
        info.firm_poached_out = poached_out
        info.firm_hires = hires
        info.firm_fired = fired_all
        info.firm_active = [f.active for f in firms]
        info.labor_flows = [[s, d, k] for (s, d), k in flows.items()]

    def _build_observation(self) -> Dict:
        """Build the observation dict visible to the agent."""
        p = self.player
        return {
            "step": self.step_count,
            "firm_cash": p.cash,
            "worker_count": p.worker_count,
            "capital_stock": p.capital_stock,
            "current_price": p.price,
            "price_index": self.price_index,
            "market_demand": self.market_demand,
            "tax_rate": self.tax_rate,
            "ema_revenue": self.ema_revenue,
            "worker_wealth": self.worker_wealth,
            "episode_length": self.preset.episode_length,
            "agent_wealth": (
                [p.cash] + [npc.cash for npc in self.npc_firms] + [self.worker_wealth]
            ),
            # Labor market (lists are player first, then NPCs)
            "wage": p.wage,
            "market_wage": self.market_wage,
            "unemployed": self.unemployed,
            "labor_force": LABOR_FORCE_TOTAL,
            "workers_lost_last_step": p.last_lost,
            "firm_active": [True] + [npc.active for npc in self.npc_firms],
            "firm_wages": [p.wage] + [npc.wage for npc in self.npc_firms],
            "firm_workers": [p.worker_count] + [npc.worker_count for npc in self.npc_firms],
            "firm_prices": [p.price] + [npc.price for npc in self.npc_firms],
            # Last step's sales (player): own demand excludes spillover
            "last_units_sold": self.last_units_sold,
            "last_capacity": self.last_capacity,
            "last_demand": self.last_demand,
        }
