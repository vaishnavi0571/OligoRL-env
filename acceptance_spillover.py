"""
Spillover Acceptance Test
=========================
Do exploit probes beat the rule-based agent by a wide margin at a given
spillover fraction?  Ranked on terminal net worth over many seeds
(common random numbers: every agent plays the same seeds).

Decision rule:
  - neither probe wins by a wide margin  -> keep spillover
  - a probe is competitive              -> lower the fraction, retest
  - still competitive at 10-20%         -> cut spillover (fraction 0)

"Wide margin" = probe mean > rule mean by 10%+ AND the 95% CIs do not
overlap.

Run:  python acceptance_spillover.py [difficulty] [num_seeds]
"""

import statistics
import sys

from agents import AGENTS, run_episode

FRACTIONS = [0.4, 0.3, 0.2, 0.1, 0.0]
PROBES = ["always_floor", "poach_to_spill", "spill_parasite"]
REFERENCE = "rule_based"
SHOWN = [REFERENCE, "hold", "market+eps"] + PROBES


def summarize(values):
    mean = statistics.mean(values)
    ci = 1.96 * statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return mean, ci


def main():
    difficulty = sys.argv[1] if len(sys.argv) > 1 else "medium"
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    print(f"\nSpillover acceptance -- difficulty={difficulty}, seeds={n_seeds}")
    print("Score: terminal net worth (cash + capital, -$10k if bankrupt)\n")

    verdicts = {}
    for frac in FRACTIONS:
        print(f"spillover_fraction = {frac}")
        results = {}
        for name in SHOWN:
            runs = [run_episode(AGENTS[name](), difficulty, s, frac) for s in range(n_seeds)]
            mean, ci = summarize([r["net_worth"] for r in runs])
            bankrupt = sum(r["bankrupt"] for r in runs) / len(runs)
            workers = statistics.mean(r["workers"] for r in runs)
            results[name] = (mean, ci)
            print(f"  {name:15s} {mean:10,.0f} +/- {ci:7,.0f}   bankrupt {bankrupt:4.0%}"
                  f"   end workers {workers:4.1f}")
        ref_mean, ref_ci = results[REFERENCE]
        winners = [
            p for p in PROBES
            if results[p][0] > 1.10 * ref_mean
            and results[p][0] - results[p][1] > ref_mean + ref_ci
        ]
        verdicts[frac] = winners
        print(f"  -> {'EXPLOITED by ' + ', '.join(winners) if winners else 'no probe wins by a wide margin'}\n")

    safe = [f for f in FRACTIONS if f > 0 and not verdicts[f]]
    if safe:
        print(f"VERDICT: keep spillover; highest safe fraction = {max(safe)}")
    else:
        print("VERDICT: cut spillover (set SPILLOVER_FRACTION = 0.0)")


if __name__ == "__main__":
    main()
