"""
Checkpoint 1 -- reward check (no learning involved)
===================================================
Runs every scripted agent on Medium seeds with the PPO training reward
(config 7e) computed alongside, and reports:
  1. final net worth mean/std, return mean/std (delta + hinge parts)
  2. paired differences vs the best scripted agent (std, 95% CI, win rate)
  3. whether ranking by return == ranking by final net worth
  4. price / rivals' average and wage / market distributions per agent
     (to choose PPO action bounds no known strategy touches)
  5. Overflow parasite vs Rule-based, with spillover on and off

Run:  python -m ppo.reward_check [num_seeds]
"""

import os
import statistics
import sys

from simulator import EconSimulator
from rewards import net_worth_reward
from agents import AGENTS, terminal_net_worth

RESULTS_DIR = os.path.join("results", "ppo")


def rivals_avg_price(obs):
    p = [pr for pr, a in zip(obs["firm_prices"][1:], obs["firm_active"][1:]) if a]
    return sum(p) / len(p) if p else obs["current_price"]


def run(name, seed, difficulty="medium", spillover=None):
    sim = EconSimulator(difficulty=difficulty, seed=seed, spillover_fraction=spillover)
    obs = sim.reset()
    agent = AGENTS[name]()
    agent.reset(seed)
    prev_worth = terminal_net_worth(sim)
    ret = delta_sum = hinge_sum = 0.0
    price_ratios, wage_ratios = [], []
    while not (sim.terminated or sim.truncated):
        action = agent.act(obs)
        price_ratios.append(action["set_price"] / rivals_avg_price(obs))
        target_wage = action.get("set_wage")
        wage_ratios.append((obs["wage"] if target_wage is None else target_wage) / obs["market_wage"])
        obs, info = sim.step(action)
        worth = terminal_net_worth(sim)
        r, d, h = net_worth_reward(prev_worth, worth, sim.player.cash, info.labor_cost,
                                   sim.preset.episode_length)
        prev_worth = worth
        ret += r
        delta_sum += d
        hinge_sum += h
    return {"worth": terminal_net_worth(sim), "ret": ret, "delta": delta_sum,
            "hinge": hinge_sum, "prices": price_ratios, "wages": wage_ratios,
            "bankrupt": sim.terminated}


def ci95(values):
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / len(values) ** 0.5


def pctl(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(q * (len(s) - 1))))]


def paired(a, b):
    diffs = [x - y for x, y in zip(a, b)]
    return {"mean": statistics.mean(diffs), "std": statistics.stdev(diffs),
            "ci": ci95(diffs), "win": sum(d > 0 for d in diffs) / len(diffs)}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    seeds = list(range(n))
    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    runs = {name: [run(name, s) for s in seeds] for name in AGENTS}
    worth = {k: [r["worth"] for r in v] for k, v in runs.items()}
    ret = {k: [r["ret"] for r in v] for k, v in runs.items()}
    best = max(AGENTS, key=lambda k: statistics.mean(worth[k]))

    out(f"CHECKPOINT 1 -- Medium, seeds 0..{n - 1}, default rules")
    out(f"Best scripted agent (benchmark): {best}\n")
    out(f"{'agent':15s} {'net worth mean':>15s} {'std':>8s} {'return mean':>12s} {'std':>6s}"
        f" {'delta':>7s} {'hinge':>7s} {'bankrupt':>8s} | {'diff vs best':>13s} {'diff std':>9s}"
        f" {'95% CI':>9s} {'win':>5s}")
    for k in sorted(AGENTS, key=lambda k: -statistics.mean(worth[k])):
        rs = runs[k]
        p = paired(worth[k], worth[best]) if k != best else None
        out(f"{k:15s} {statistics.mean(worth[k]):15,.0f} {statistics.stdev(worth[k]):8,.0f}"
            f" {statistics.mean(ret[k]):12.3f} {statistics.stdev(ret[k]):6.3f}"
            f" {statistics.mean(r['delta'] for r in rs):7.3f} {statistics.mean(r['hinge'] for r in rs):7.3f}"
            f" {sum(r['bankrupt'] for r in rs) / n:8.0%} | "
            + (f"{p['mean']:13,.0f} {p['std']:9,.0f} {p['ci']:9,.0f} {p['win']:5.0%}" if p else f"{'(benchmark)':>13s}"))

    by_worth = sorted(AGENTS, key=lambda k: -statistics.mean(worth[k]))
    by_ret = sorted(AGENTS, key=lambda k: -statistics.mean(ret[k]))
    out(f"\nRanking by net worth: {by_worth}")
    out(f"Ranking by return:    {by_ret}")
    out("RANKING MATCH" if by_worth == by_ret else "RANKING MISMATCH -- investigate")
    max_hinge = max(-r["hinge"] for v in runs.values() for r in v)
    out(f"Largest total hinge penalty seen: {max_hinge:.3f} (cap = 0.5 x bankruptcy penalty = 0.500)")

    out("\nAction distributions (all steps, all seeds)")
    out(f"{'agent':15s} {'price/rivals: min':>18s} {'p1':>6s} {'p50':>6s} {'p99':>6s} {'max':>6s}"
        f" | {'wage/market: min':>17s} {'p1':>6s} {'p50':>6s} {'p99':>6s} {'max':>6s}")
    for k in AGENTS:
        pr = [x for r in runs[k] for x in r["prices"]]
        wr = [x for r in runs[k] for x in r["wages"]]
        out(f"{k:15s} {min(pr):18.3f} {pctl(pr, .01):6.3f} {pctl(pr, .5):6.3f} {pctl(pr, .99):6.3f} {max(pr):6.3f}"
            f" | {min(wr):17.3f} {pctl(wr, .01):6.3f} {pctl(wr, .5):6.3f} {pctl(wr, .99):6.3f} {max(wr):6.3f}")

    out("\nOverflow parasite vs Rule-based (paired, final net worth)")
    for spill in (None, 0.0):
        para = [run("spill_parasite", s, spillover=spill)["worth"] for s in seeds]
        rule = [run("rule_based", s, spillover=spill)["worth"] for s in seeds]
        p = paired(para, rule)
        label = "spill=0.4 (default)" if spill is None else "spill=0.0"
        out(f"  {label:20s} parasite {statistics.mean(para):9,.0f}  rule {statistics.mean(rule):9,.0f}"
            f"  diff {p['mean']:+8,.0f} +/- {p['ci']:6,.0f}  (std {p['std']:6,.0f}, parasite wins {p['win']:.0%})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "checkpoint1_reward_check.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
