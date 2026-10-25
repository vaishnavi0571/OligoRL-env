"""
OligoRL -- Agents and Leaderboard Scoring
==========================================
Every agent maps an observation dict to an action dict.  Agents may keep
state between steps; call reset(seed) at the start of each episode.

Leaderboards rank on terminal_net_worth(), NOT on the training reward.
Every agent plays the same seeds (common random numbers), and results
are reported as a mean with a 95% confidence interval.
"""

import os
import random
import statistics
from functools import lru_cache
from typing import Dict, List, Optional

from simulator import EconSimulator, weighted_median
from reward_integrator import compute_step_reward
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
    label = "Agent"
    description = ""

    def reset(self, seed: int = 0) -> None:
        pass

    def act(self, obs: Dict) -> Dict:
        raise NotImplementedError


def rivals_wage(obs: Dict) -> float:
    """Headcount-weighted median wage of active rivals (excludes self)."""
    return weighted_median([
        (w, n) for w, n, a in zip(obs["firm_wages"][1:], obs["firm_workers"][1:],
                                  obs["firm_active"][1:]) if a
    ])


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
    label = "Rule-based"
    description = "Raises price and hires while demand outruns capacity; matches market wage"

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
    label = "Hold"
    description = "Does nothing: default price, default wage, never hires"

    def act(self, obs: Dict) -> Dict:
        return {"set_price": DEFAULT_PRICE}


class MarketPlusEpsAgent(Agent):
    """Pays market + 1% and grows to 12 workers."""
    name = "market+eps"
    label = "Market + 1%"
    description = "Pays 1% above market and grows to 12 workers"

    def act(self, obs: Dict) -> Dict:
        return {"set_price": DEFAULT_PRICE, "set_wage": obs["market_wage"] * 1.01,
                "hire_workers": 2 if obs["worker_count"] < 12 else 0}


class AlwaysFloorAgent(Agent):
    """Parasite: pays the minimum wage forever."""
    name = "always_floor"
    label = "Minimum wage"
    description = "Parasite probe: pays the legal minimum wage forever"

    def act(self, obs: Dict) -> Dict:
        return {"set_price": DEFAULT_PRICE, "set_wage": MIN_WAGE}


class AlwaysMaxAgent(Agent):
    """Pays the maximum wage and keeps hiring."""
    name = "always_max"
    label = "Maximum wage"
    description = "Pays the maximum wage and hires every step"

    def act(self, obs: Dict) -> Dict:
        return {"set_price": DEFAULT_PRICE, "set_wage": MAX_WAGE, "hire_workers": 2}


class PoachToSpillAgent(Agent):
    """Exploit probe: outbid rivals for workers, starve their capacity,
    then serve the customers they turn away.  Bids against RIVALS' wages
    (bidding against the market median would chase its own wage)."""
    name = "poach_to_spill"
    label = "Poacher"
    description = "Exploit probe: outbids rivals by 15% to steal workers and their customers"

    def act(self, obs: Dict) -> Dict:
        return {"set_price": DEFAULT_PRICE, "set_wage": rivals_wage(obs) * 1.15,
                "hire_workers": 3 if obs["worker_count"] < 15 else 0}


class SpillParasiteAgent(Agent):
    """Prices 10% above rivals and grows to 8 workers.

    Built as an exploit probe for demand spillover (keep idle capacity,
    live off rivals' turned-away customers), but checkpoint 1 showed its
    lead over the rule-based agent is LARGER with spillover off -- it is
    simply a strong premium-pricing strategy.  Internal name kept so the
    golden test and saved results stay comparable."""
    name = "spill_parasite"
    label = "Premium pricer"
    description = "Prices 10% above rivals and grows to 8 workers (best scripted agent)"

    def act(self, obs: Dict) -> Dict:
        return {"set_price": 1.1 * DEFAULT_PRICE,
                "hire_workers": 1 if obs["worker_count"] < 8 else 0}


class RandomAgent(Agent):
    """Uniformly random actions within sensible ranges (seeded per episode)."""
    name = "random"
    label = "Random"
    description = "Random price, wage and hiring every step"

    def __init__(self):
        self.rng = random.Random(0)

    def reset(self, seed: int = 0) -> None:
        self.rng = random.Random(seed + 12345)

    def act(self, obs: Dict) -> Dict:
        return {"set_price": self.rng.uniform(7.0, 13.0),
                "set_wage": self.rng.uniform(MIN_WAGE, 800.0),
                "hire_workers": self.rng.randint(-2, 3)}


PPO_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ppo_medium")


class PPOAgent(Agent):
    """Trained PPO policy (numpy-only inference; see ppo/).  Mean action."""
    name = "ppo"
    label = "PPO (trained)"
    description = "Trained with PPO on the net-worth reward; best of 5 runs by validation"

    def __init__(self):
        from ppo.policy_numpy import NumpyPolicy   # numpy only, no torch
        self.policy = NumpyPolicy(PPO_MODEL + ".npz", PPO_MODEL + ".json")

    def act(self, obs: Dict) -> Dict:
        return self.policy.act(obs)


AGENTS = {
    cls.name: cls for cls in (
        RuleBasedAgent, HoldAgent, MarketPlusEpsAgent, AlwaysFloorAgent,
        AlwaysMaxAgent, PoachToSpillAgent, SpillParasiteAgent, RandomAgent,
    )
}
if os.path.exists(PPO_MODEL + ".npz") and os.path.exists(PPO_MODEL + ".json"):
    AGENTS[PPOAgent.name] = PPOAgent


# -----------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------

def run_episode(agent: Agent, difficulty: str = "medium", seed: int = 0,
                spillover_fraction=None, record: bool = False,
                npc_profile: str = "default", lobbying_enabled: bool = True,
                npc_growth: bool = True) -> Dict:
    """Play one full episode; returns summary stats.

    With record=True also returns the player's net worth after every step
    (index 0 = start) under "worth_curve".  npc_profile / lobbying_enabled
    are the opt-in experiment flags (defaults = leaderboard rules).
    """
    sim = EconSimulator(difficulty=difficulty, seed=seed,
                        spillover_fraction=spillover_fraction,
                        npc_profile=npc_profile, lobbying_enabled=lobbying_enabled,
                        npc_growth=npc_growth)
    obs = sim.reset()
    agent.reset(seed)
    total_reward = 0.0
    curve = [terminal_net_worth(sim)]
    while not (sim.terminated or sim.truncated):
        obs, info = sim.step(agent.act(obs))
        total_reward += compute_step_reward(info, sim).composite
        if record:
            curve.append(terminal_net_worth(sim))
    result = {
        "net_worth": terminal_net_worth(sim),
        "bankrupt": sim.terminated,
        "steps": sim.step_count,
        "workers": sim.player.worker_count,
        "total_reward": total_reward,
    }
    if record:
        result["worth_curve"] = curve
    return result


def _mean_ci(values: List[float]):
    mean = statistics.mean(values)
    ci = 1.96 * statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return mean, ci


@lru_cache(maxsize=64)
def evaluate_agent(name: str, difficulty: str, num_seeds: int) -> Dict:
    """Leaderboard row for one agent over seeds 0..num_seeds-1 (cached:
    the simulator is deterministic per seed)."""
    runs = [run_episode(AGENTS[name](), difficulty, seed, record=(seed == 0))
            for seed in range(num_seeds)]
    worth, worth_ci = _mean_ci([r["net_worth"] for r in runs])
    reward, reward_ci = _mean_ci([r["total_reward"] for r in runs])
    cls = AGENTS[name]
    return {
        "agent": name,
        "label": cls.label,
        "description": cls.description,
        "net_worth": worth,
        "net_worth_ci": worth_ci,
        "bankrupt_rate": sum(r["bankrupt"] for r in runs) / len(runs),
        "total_reward": reward,
        "total_reward_ci": reward_ci,
        "end_workers": statistics.mean(r["workers"] for r in runs),
        "steps_survived": statistics.mean(r["steps"] for r in runs),
        "worth_curve_seed0": runs[0]["worth_curve"],
    }


def leaderboard(names: Optional[List[str]] = None, difficulty: str = "medium",
                num_seeds: int = 20) -> List[Dict]:
    """Rows for the given agents, ranked by mean terminal net worth."""
    names = names or list(AGENTS)
    rows = [evaluate_agent(n, difficulty, num_seeds) for n in names]
    return sorted(rows, key=lambda r: r["net_worth"], reverse=True)


assert MAX_HIRE_PER_STEP >= 3  # PoachToSpillAgent hires 3 per step
