"""
OpenEnv -- Agents and Leaderboard Scoring
==========================================
Every agent maps an observation dict to an action dict.  Agents may keep
state between steps; call reset() at the start of each episode.

Leaderboards rank on terminal_net_worth(), NOT on the training reward.
"""

from typing import Dict

from simulator import EconSimulator, weighted_median
from config import (
    MIN_WAGE, MAX_WAGE, DEFAULT_PRICE, INPUT_COST_PER_UNIT,
    MAX_HIRE_PER_STEP, POACH_PREMIUM, BANKRUPTCY_PENALTY,
)


def terminal_net_worth(sim: EconSimulator) -> float:
    """Leaderboard score: cash + capital, minus a penalty if bankrupt."""
    worth = sim.player.cash + sim.player.capital_stock
    if sim.terminated:
        worth -= BANKRUPTCY_PENALTY
    return worth


class Agent:
    name = "agent"

    def reset(self) -> None:
        pass

    def act(self, obs: Dict) -> Dict:
        raise NotImplementedError


# -----------------------------------------------------------------
# Rule-based agent (the reference strategy)
# -----------------------------------------------------------------

class RuleBasedAgent(Agent):
    """Hand-written strategy, deliberately different from the NPC rule.

    Price:  +3% while selling out with 20%+ excess demand; -3% when using
            under 90% of capacity (never below input cost + $1).
    Wage:   match the market wage; bid POACH_PREMIUM above it only when
            it wants workers and the unemployed pool is empty.
    Hiring: hire (up to 2) while demand exceeds capacity by 20%+ and cash
            covers 4 steps of the larger payroll; fire 1 below 70% use.
    """
    name = "rule_based"

    def act(self, obs: Dict) -> Dict:
        price = obs["current_price"]
        workers = obs["worker_count"]
        if obs["step"] == 0:
            return {"set_price": price}

        capacity = max(obs["last_capacity"], 1.0)
        utilization = obs["last_units_sold"] / capacity
        excess_demand = obs["last_demand"] > 1.2 * capacity
        sold_out = utilization >= 0.99

        if sold_out and excess_demand:
            price = min(price * 1.03, 2.0 * DEFAULT_PRICE)
        elif utilization < 0.9:
            price = max(price * 0.97, INPUT_COST_PER_UNIT + 1.0)

        wage = obs["market_wage"]
        hire = 0
        if sold_out and excess_demand:
            if obs["unemployed"] == 0:
                wage *= 1.0 + POACH_PREMIUM + 0.01
            for h in (2, 1):
                if obs["firm_cash"] >= 4 * (workers + h) * wage:
                    hire = h
                    break
        elif utilization < 0.7 and workers > 1:
            hire = -1

        return {"set_price": price, "set_wage": wage, "hire_workers": hire}


# -----------------------------------------------------------------
# Scripted sanity baselines and exploit probes
# -----------------------------------------------------------------

class HoldAgent(Agent):
    """Keeps every default: price $10, wage $500, no hiring."""
    name = "hold"

    def act(self, obs: Dict) -> Dict:
        return {"set_price": DEFAULT_PRICE}


class AlwaysFloorAgent(Agent):
    """Parasite: pays the minimum wage forever."""
    name = "always_floor"

    def act(self, obs: Dict) -> Dict:
        return {"set_price": DEFAULT_PRICE, "set_wage": MIN_WAGE}


class AlwaysMaxAgent(Agent):
    """Pays the maximum wage and keeps hiring."""
    name = "always_max"

    def act(self, obs: Dict) -> Dict:
        return {"set_price": DEFAULT_PRICE, "set_wage": MAX_WAGE, "hire_workers": 2}


class MarketPlusEpsAgent(Agent):
    """Pays market + 1% and grows to 12 workers."""
    name = "market+eps"

    def act(self, obs: Dict) -> Dict:
        return {"set_price": DEFAULT_PRICE, "set_wage": obs["market_wage"] * 1.01,
                "hire_workers": 2 if obs["worker_count"] < 12 else 0}


def rivals_wage(obs: Dict) -> float:
    """Headcount-weighted median wage of active rivals (excludes self)."""
    return weighted_median([
        (w, n) for w, n, a in zip(obs["firm_wages"][1:], obs["firm_workers"][1:],
                                  obs["firm_active"][1:]) if a
    ])


class PoachToSpillAgent(Agent):
    """Exploit probe: outbid rivals for workers, starve their capacity,
    then serve the customers they turn away.  Bids against RIVALS' wages
    (bidding against the market median would chase its own wage)."""
    name = "poach_to_spill"

    def act(self, obs: Dict) -> Dict:
        return {"set_price": DEFAULT_PRICE, "set_wage": rivals_wage(obs) * 1.15,
                "hire_workers": 3 if obs["worker_count"] < 15 else 0}


class SpillParasiteAgent(Agent):
    """Exploit probe: price above rivals to keep idle capacity, and live
    off the customers that capacity-constrained rivals turn away."""
    name = "spill_parasite"

    def act(self, obs: Dict) -> Dict:
        return {"set_price": 1.1 * DEFAULT_PRICE,
                "hire_workers": 1 if obs["worker_count"] < 8 else 0}


AGENTS = {
    cls.name: cls for cls in (
        RuleBasedAgent, HoldAgent, AlwaysFloorAgent, AlwaysMaxAgent,
        MarketPlusEpsAgent, PoachToSpillAgent, SpillParasiteAgent,
    )
}


def run_episode(agent: Agent, difficulty: str = "medium", seed: int = 0,
                spillover_fraction=None) -> Dict:
    """Play one full episode; returns summary stats."""
    sim = EconSimulator(difficulty=difficulty, seed=seed,
                        spillover_fraction=spillover_fraction)
    obs = sim.reset()
    agent.reset()
    while not (sim.terminated or sim.truncated):
        obs, _ = sim.step(agent.act(obs))
    return {
        "net_worth": terminal_net_worth(sim),
        "bankrupt": sim.terminated,
        "steps": sim.step_count,
        "workers": sim.player.worker_count,
    }


assert MAX_HIRE_PER_STEP >= 3  # PoachToSpillAgent hires 3 per step
