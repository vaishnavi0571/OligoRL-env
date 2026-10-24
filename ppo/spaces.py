"""
PPO observation and action spaces (pure Python, no numpy/torch)
================================================================
Shared by the training environment and by the numpy inference agent, so
training and deployment encode observations and decode actions with the
exact same code.

Observation: 14 features built from the public observation dict, each
    normalized as clip((x - offset) / scale, -5, 5).  No NPC internals.

Action: 3 raw values from the policy.  Each is squashed with tanh to
    a in [-1, 1], then decoded:
      price = rivals' average price x 2^a          -> [0.5x, 2x], log scale
      wage  = market wage x (1 + 0.1a if a < 0 else 1 + 0.2a) -> [0.9x, 1.2x]
      hire  = 0 inside the dead zone |a| < 0.1; otherwise at least 1:
              sign(a) * max(1, round(|a| * f * max(workers, 1)))
              f = 0.4 when hiring, 0.2 when firing, capped per step
    raw = 0 (the initial policy mean) decodes to "rival price, market
    wage, no change".  Lobbying and capex are not PPO actions.
"""

import math
from typing import Dict, List, Sequence, Tuple

from config import (
    INITIAL_FIRM_CASH, WAGE_PER_WORKER, DEFAULT_PRICE, MAX_HIRE_PER_STEP,
)

# (name, offset, scale)
OBS_SPEC: List[Tuple[str, float, float]] = [
    ("cash_vs_start",          3.0, 3.0),
    ("net_worth_vs_start",     3.0, 3.0),
    ("wage_vs_market",         1.0, 0.2),
    ("price_vs_rivals",        1.0, 0.3),
    ("progress",               0.5, 0.5),
    ("workers_share",          0.2, 0.2),
    ("unemployed_share",       0.1, 0.1),
    ("demand_vs_capacity",     2.0, 2.0),
    ("utilization",            0.8, 0.4),
    ("market_wage_vs_default", 1.0, 0.2),
    ("rival_price_vs_default", 1.0, 0.1),
    ("tax_rate",               0.15, 0.1),
    ("workers_lost_share",     0.0, 0.2),
    ("payroll_runway_steps",   5.0, 5.0),
]
OBS_DIM = len(OBS_SPEC)
OBS_CLIP = 5.0

ACTION_NAMES = ["price", "wage", "hire"]
ACTION_DIM = len(ACTION_NAMES)
RAW_ACTION_BOUND = 3.0          # policy outputs are clipped here before tanh
PRICE_LOG2_RANGE = 1.0          # price ratio = 2^a
WAGE_DOWN, WAGE_UP = 0.1, 0.2   # wage ratio in [0.9, 1.2]
HIRE_DEADZONE = 0.1
HIRE_FRACTION, FIRE_FRACTION = 0.4, 0.2
BOUND_MARGIN = 0.05             # "at bound" = within 5% of the range of an edge


def rivals_avg_price(obs: Dict) -> float:
    prices = [p for p, a in zip(obs["firm_prices"][1:], obs["firm_active"][1:]) if a]
    return sum(prices) / len(prices) if prices else DEFAULT_PRICE


def raw_features(obs: Dict) -> List[float]:
    """Unnormalized features, in OBS_SPEC order."""
    workers = obs["worker_count"]
    capacity = max(obs["last_capacity"], 1.0)
    wage_bill = max(workers * obs["wage"], 1.0)
    rival_price = rivals_avg_price(obs)
    return [
        obs["firm_cash"] / INITIAL_FIRM_CASH,
        (obs["firm_cash"] + obs["capital_stock"]) / INITIAL_FIRM_CASH,
        obs["wage"] / obs["market_wage"],
        obs["current_price"] / rival_price,
        obs["step"] / obs["episode_length"],
        workers / obs["labor_force"],
        obs["unemployed"] / obs["labor_force"],
        obs["last_demand"] / capacity,
        obs["last_units_sold"] / capacity,
        obs["market_wage"] / WAGE_PER_WORKER,
        rival_price / DEFAULT_PRICE,
        obs["tax_rate"],
        obs["workers_lost_last_step"] / max(workers, 1),
        max(obs["firm_cash"], 0.0) / wage_bill,
    ]


def encode_observation(obs: Dict) -> List[float]:
    """Normalized feature vector fed to the policy."""
    return [
        max(-OBS_CLIP, min(OBS_CLIP, (x - off) / scale))
        for x, (_, off, scale) in zip(raw_features(obs), OBS_SPEC)
    ]


def squash(raw: Sequence[float]) -> List[float]:
    return [math.tanh(max(-RAW_ACTION_BOUND, min(RAW_ACTION_BOUND, float(x)))) for x in raw]


def decode_action(raw: Sequence[float], obs: Dict) -> Dict:
    """Policy output -> simulator action dict."""
    a_price, a_wage, a_hire = squash(raw)
    price = rivals_avg_price(obs) * 2.0 ** (PRICE_LOG2_RANGE * a_price)
    wage_ratio = 1.0 + (WAGE_DOWN * a_wage if a_wage < 0 else WAGE_UP * a_wage)
    wage = obs["market_wage"] * wage_ratio

    hire = 0
    if abs(a_hire) >= HIRE_DEADZONE:
        base = max(obs["worker_count"], 1)
        frac = HIRE_FRACTION if a_hire > 0 else FIRE_FRACTION
        size = max(1, round(abs(a_hire) * frac * base))
        hire = int(math.copysign(min(size, MAX_HIRE_PER_STEP), a_hire))
        if hire < 0:
            hire = max(hire, -obs["worker_count"])
    return {"set_price": price, "set_wage": wage, "hire_workers": hire}


def at_bounds(raw: Sequence[float]) -> List[bool]:
    """Per action dimension: is the decoded action within BOUND_MARGIN of
    its range from either edge?  Measured on the decoded scale:
      price: log2(price / rivals) in [-1, 1]
      wage:  wage / market in [1 - WAGE_DOWN, 1 + WAGE_UP]
      hire:  squashed value in [-1, 1] (the count also depends on headcount)
    """
    a_price, a_wage, a_hire = squash(raw)
    m = BOUND_MARGIN
    lo_w, hi_w = 1.0 - WAGE_DOWN, 1.0 + WAGE_UP
    wage_ratio = 1.0 + (WAGE_DOWN * a_wage if a_wage < 0 else WAGE_UP * a_wage)
    span_w = hi_w - lo_w
    return [
        abs(PRICE_LOG2_RANGE * a_price) >= PRICE_LOG2_RANGE * (1 - m),
        wage_ratio <= lo_w + m * span_w or wage_ratio >= hi_w - m * span_w,
        abs(a_hire) >= 1 - 2 * m,
    ]


def spec_dict() -> Dict:
    """Everything needed to reproduce encoding/decoding, saved with weights."""
    return {
        "obs_spec": [{"name": n, "offset": o, "scale": s} for n, o, s in OBS_SPEC],
        "obs_clip": OBS_CLIP,
        "action_names": ACTION_NAMES,
        "raw_action_bound": RAW_ACTION_BOUND,
        "price_log2_range": PRICE_LOG2_RANGE,
        "wage_down": WAGE_DOWN,
        "wage_up": WAGE_UP,
        "hire_deadzone": HIRE_DEADZONE,
        "hire_fraction": HIRE_FRACTION,
        "fire_fraction": FIRE_FRACTION,
        "bound_margin": BOUND_MARGIN,
    }
