"""
Labor Market Diagnostics
========================
Runs sanity-baseline policies over many seeds and reports stagnation
detectors (floor collapse, cap ratchet, hoarding, frozen pool, NPC
lockstep, leaderboard separation, early bankruptcy).

Run:  python labor_diagnostics.py [difficulty] [num_seeds]
"""

import random
import statistics
import sys

from simulator import EconSimulator
from reward_integrator import compute_step_reward
from config import MIN_WAGE, MAX_WAGE, LABOR_FORCE_TOTAL, WORKER_PRODUCTIVITY, INPUT_COST_PER_UNIT


def hold(obs, rng):
    return {"set_price": 10.0}


def always_floor(obs, rng):
    return {"set_price": 10.0, "set_wage": MIN_WAGE}


def always_max(obs, rng):
    return {"set_price": 10.0, "set_wage": MAX_WAGE, "hire_workers": 2}


def market_plus_eps(obs, rng):
    return {"set_price": 10.0, "set_wage": obs["market_wage"] * 1.01,
            "hire_workers": 2 if obs["worker_count"] < 12 else 0}


def poacher(obs, rng):
    return {"set_price": 10.0, "set_wage": obs["market_wage"] * 1.15,
            "hire_workers": 3 if obs["worker_count"] < 12 else 0}


def random_policy(obs, rng):
    return {"set_price": rng.uniform(7.0, 13.0),
            "set_wage": rng.uniform(MIN_WAGE, 800.0),
            "hire_workers": rng.randint(-2, 3)}


POLICIES = {
    "hold": hold,
    "always_floor": always_floor,
    "always_max": always_max,
    "market+eps": market_plus_eps,
    "poacher": poacher,
    "random": random_policy,
}


def run_episode(policy, difficulty, seed):
    sim = EconSimulator(difficulty=difficulty, seed=seed)
    obs = sim.reset()
    rng = random.Random(seed + 12345)
    stats = {"reward": 0.0, "steps": 0, "at_floor": 0, "at_cap": 0,
             "npc_at_cap": 0, "npc_steps": 0, "flows": 0,
             "max_emp_share": 0.0, "wage_std": []}
    while not (sim.terminated or sim.truncated):
        obs, info = sim.step(policy(obs, rng))
        stats["reward"] += compute_step_reward(info, sim).composite
        stats["steps"] += 1
        stats["at_floor"] += info.wage <= MIN_WAGE + 1e-6
        stats["at_cap"] += info.wage >= MAX_WAGE - 1e-6
        for npc in sim.npc_firms:
            if not npc.active:
                continue
            cap = npc.max_wage_share * WORKER_PRODUCTIVITY * (npc.price - INPUT_COST_PER_UNIT)
            stats["npc_at_cap"] += npc.wage >= cap - 1e-6
            stats["npc_steps"] += 1
        stats["flows"] += sum(k for _, _, k in info.labor_flows)
        employed = [w for w, a in zip(info.firm_workers, info.firm_active) if a]
        stats["max_emp_share"] = max(stats["max_emp_share"], max(employed) / LABOR_FORCE_TOTAL)
        stats["wage_std"].append(statistics.pstdev(
            [w for w, a in zip(info.firm_wages, info.firm_active) if a]))
    stats["early_end"] = sim.terminated
    stats["final_cash"] = sim.player.cash
    return stats


def main():
    difficulty = sys.argv[1] if len(sys.argv) > 1 else "medium"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    seeds = range(n_seeds)
    print(f"\nLabor diagnostics -- difficulty={difficulty}, seeds={n_seeds}\n")
    header = (f"{'policy':14s} {'reward':>14s} {'final cash':>12s} {'early end':>9s} "
              f"{'@floor':>7s} {'@cap':>6s} {'NPC@cap':>7s} {'flow/step':>9s} "
              f"{'max emp':>7s} {'wage std':>8s}")
    print(header)
    print("-" * len(header))
    means = {}
    noise = []
    for name, policy in POLICIES.items():
        runs = [run_episode(policy, difficulty, s) for s in seeds]
        rewards = [r["reward"] for r in runs]
        mean = statistics.mean(rewards)
        sd = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        ci = 1.96 * sd / len(rewards) ** 0.5
        means[name] = mean
        noise.append(ci)
        steps = sum(r["steps"] for r in runs)
        npc_steps = max(sum(r["npc_steps"] for r in runs), 1)
        print(f"{name:14s} {mean:6.2f} +/-{ci:5.2f} "
              f"{statistics.mean(r['final_cash'] for r in runs):12,.0f} "
              f"{sum(r['early_end'] for r in runs) / len(runs):9.0%} "
              f"{sum(r['at_floor'] for r in runs) / steps:7.0%} "
              f"{sum(r['at_cap'] for r in runs) / steps:6.0%} "
              f"{sum(r['npc_at_cap'] for r in runs) / npc_steps:7.0%} "
              f"{sum(r['flows'] for r in runs) / steps:9.2f} "
              f"{max(r['max_emp_share'] for r in runs):7.0%} "
              f"{statistics.mean(statistics.mean(r['wage_std']) for r in runs):8.1f}")
    spread = max(means.values()) - min(means.values())
    print(f"\nLeaderboard spread (best - worst mean reward): {spread:.2f}")
    print(f"Typical seed noise (95% CI half-width):        {statistics.mean(noise):.2f}")
    print("Separation OK" if spread > 3 * statistics.mean(noise)
          else "WARNING: policies barely separate from seed noise")


if __name__ == "__main__":
    main()
