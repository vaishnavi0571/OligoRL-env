"""
OpenEnv -- Phase 3: Simulator State Machine
=============================================
Pure economic simulation.  No reward logic, no API layer.
Testable in isolation by running step() with hardcoded actions.

Step order within each step() call:
    1. Validate/clamp actions
    2. Generate NPC actions
    3. Update hiring/firing
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
    - NPCs never hire/fire (maintain INITIAL_WORKER_COUNT)
    - NPCs never invest (capex = 0) and never lobby (tax_change = 0)
    - NPCs pay identical wage/input/tax rates as the player
    - This creates a stable competitive baseline; the player differentiates
      through pricing, hiring, investment, and tax lobbying.
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
) -> Tuple[float, float, float]:
    """Compute production and sales for one agent.

    Args:
        demand_multiplier: wealth/GDP-driven scaling of total demand.
        rng: if provided, adds small per-firm demand noise.

    Returns:
        (max_output, units_sold, revenue)
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
    return max_output, units_sold, revenue


def compute_wealth_factor(total_wealth: float) -> float:
    """GDP-demand feedback: wealthier economy generates more demand."""
    return 1.0 + GDP_DEMAND_SENSITIVITY * math.log1p(
        max(total_wealth, 0.0) / WEALTH_REFERENCE
    )


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
    units_sold: float = 0.0
    max_output: float = 0.0
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

    def __init__(self, difficulty: str = DEFAULT_DIFFICULTY, seed: int = 42):
        self.preset = PRESETS[difficulty]
        self._seed = seed
        self.rng = random.Random(seed)
        self.player: Optional[FirmState] = None
        self.npc_firms: List[FirmState] = []
        self.price_index: float = INITIAL_PRICE_INDEX
        self.market_demand: float = MARKET_DEMAND_INITIAL
        self.tax_rate: float = TAX_RATE
        self.ema_revenue: float = REVENUE_EMA_INITIAL
        self.real_gdp_previous: float = 0.0
        self.worker_wealth: float = INITIAL_WORKER_WEALTH
        self.step_count: int = 0
        self.terminated: bool = False
        self.truncated: bool = False

    def reset(self, seed: Optional[int] = None) -> Dict:
        """Initialize a fresh episode.  Returns observation dict."""
        if seed is not None:
            self._seed = seed
        self.rng = random.Random(self._seed)

        npc_count = NUM_AGENTS - 1
        self.player = FirmState()
        self.npc_firms = [FirmState() for _ in range(npc_count)]
        self.price_index = INITIAL_PRICE_INDEX
        self.market_demand = MARKET_DEMAND_INITIAL
        self.tax_rate = TAX_RATE
        self.ema_revenue = REVENUE_EMA_INITIAL
        self.worker_wealth = INITIAL_WORKER_WEALTH
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
        lobby = float(action.get("lobby_tax_change", 0.0))
        lobby = max(LOBBY_TAX_CHANGE_MIN, min(LOBBY_TAX_CHANGE_MAX, lobby))

        # ===== 2. Generate NPC actions =====
        npc_prices = []
        for npc in self.npc_firms:
            noise = 1.0 + self.rng.gauss(0, 0.02)
            npc.price = max(DEFAULT_PRICE * noise, PRICE_FLOOR)
            npc_prices.append(npc.price)

        # ===== 3. Update hiring/firing =====
        p.worker_count += hire
        info.hiring_cost = abs(hire) * HIRING_COST_PER_WORKER
        p.price = set_price

        # ===== 4. Compute wealth-driven demand multiplier =====
        total_economy_wealth = (
            p.cash + sum(npc.cash for npc in self.npc_firms) + self.worker_wealth
        )
        info.demand_multiplier = compute_wealth_factor(total_economy_wealth)

        # ===== 5. Calculate revenue (all agents) =====
        all_prices = [set_price] + npc_prices

        # Player revenue
        competitor_avg = sum(npc_prices) / len(npc_prices) if npc_prices else DEFAULT_PRICE
        info.competitor_avg_price = competitor_avg
        info.max_output, info.units_sold, info.revenue = compute_revenue(
            set_price, p.worker_count, self.market_demand, competitor_avg,
            demand_multiplier=info.demand_multiplier, rng=self.rng,
        )

        # NPC revenue
        npc_revenues = []
        npc_units_sold_list = []
        for i, npc in enumerate(self.npc_firms):
            others = [pr for j, pr in enumerate(all_prices) if j != i + 1]
            other_avg = sum(others) / len(others) if others else DEFAULT_PRICE
            _, n_sold, n_rev = compute_revenue(
                npc.price, npc.worker_count, self.market_demand, other_avg,
                demand_multiplier=info.demand_multiplier, rng=self.rng,
            )
            npc_revenues.append(n_rev)
            npc_units_sold_list.append(n_sold)

        # ===== 6. Pay expenses, update cash =====
        # Player
        info.labor_cost = p.worker_count * WAGE_PER_WORKER
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
        for i, npc in enumerate(self.npc_firms):
            npc_labor = npc.worker_count * WAGE_PER_WORKER
            npc_input = npc_units_sold_list[i] * INPUT_COST_PER_UNIT
            npc_gross = npc_revenues[i] - npc_labor - npc_input
            npc_tax = max(npc_gross, 0.0) * self.tax_rate
            npc.cash += npc_revenues[i] - npc_labor - npc_input - npc_tax
            total_npc_wages += npc_labor
            total_npc_input += npc_input
            total_npc_tax += npc_tax

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
        info.total_hiring_all = info.hiring_cost  # NPCs don't hire

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
        self.step_count += 1

        return self._build_observation(), info

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
        }
