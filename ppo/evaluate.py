"""
Final evaluation on TEST seeds (run once, after the 5 training runs)
====================================================================
Implements results/ppo/preregistration.md:
  - exports each run's selected checkpoint to numpy and checks parity
  - paired comparisons vs every scripted agent on the default and heldout
    test seeds (same seeds, same NPC draws), mean diff + 95% CI + win rate
  - criteria P1, P2 (+ gap closed), P3, H1, H2 (a, b, c), R1, B1, B2
  - learning curves (mean and min-max band across runs, references)
  - behavior audit: action bounds, wage/market distribution, hiring
    pattern, endgame counterfactual, best/worst episodes, return vs
    net-worth consistency
Writes results/ppo/final/ and models/ppo_medium.{npz,json} (leaderboard
policy = best run by validation).

Run:  python -m ppo.evaluate [--runs 1 2 3 4 5]
Dry run (validation seeds + pilot checkpoint, no test seeds, no model copy):
      python -m ppo.evaluate --dry-run --runs-dir results/ppo/pilot_v2_spill0 --runs 1
"""

import argparse
import csv
import json
import math
import os
import shutil
import statistics as st

import numpy as np

from agents import AGENTS, run_episode
from ppo import spaces
from ppo.env import FirmEnv, rollout
from ppo.export import export, parity
from ppo.policy_numpy import NumpyPolicy
from ppo.train import DEFAULT_CONFIG, load_config

OUT = os.path.join("results", "ppo", "final")
RUNS_DIR = os.path.join("results", "ppo", "runs")
BASELINES = ["random", "always_floor", "always_max", "poach_to_spill", "hold", "market+eps"]
EXTRA_ROWS = ["rule_based", "spill_parasite"]
BENCHMARK = "rule_based"          # frozen on validation (benchmark.json)
GAP_ANCHOR, GAP_ANCHOR_ALT = "market+eps", "hold"


# ---------------------------------------------------------------- stats
def ci95(v):
    return 1.96 * st.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    m, c = st.mean(d), ci95(d)
    return {"mean_diff": m, "ci": c, "lo": m - c, "hi": m + c,
            "win_rate": sum(x > 0 for x in d) / len(d)}


def gap_closed(p, bench, ref, n_boot=5000, seed=0):
    """(mean P - mean ref) / (mean bench - mean ref), paired bootstrap CI."""
    p, bench, ref = map(np.asarray, (p, bench, ref))
    point = (p.mean() - ref.mean()) / (bench.mean() - ref.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(p), size=(n_boot, len(p)))
    ratios = (p[idx].mean(1) - ref[idx].mean(1)) / (bench[idx].mean(1) - ref[idx].mean(1))
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi)}


# ---------------------------------------------------------------- episodes
def scripted(name, seeds, profile):
    return [run_episode(AGENTS[name](), "medium", s, npc_profile=profile,
                        lobbying_enabled=False) for s in seeds]


def ppo_mean(policy, seeds, profile):
    return rollout(policy.mean_action, seeds, "medium", profile, record=True)


def ppo_sampled(policy, seeds, profile, seed=0):
    """Episodes with sampled actions (audit only)."""
    rng = np.random.default_rng(seed)
    env = FirmEnv(npc_profile=profile)
    bounds, worths = [], []
    for s in seeds:
        obs, _ = env.reset(options={"episode_seed": s})
        done = False
        while not done:
            raw = policy.sample_action(obs.astype(np.float64), rng)
            obs, _, done, _, info = env.step(raw)
            bounds.append(info["at_bounds"])
        worths.append(info["net_worth"])
    return np.mean(bounds, axis=0).tolist(), worths


def endgame_counterfactual(policy, seeds, profile):
    """Net worth with vs without any hiring in the last 3 steps."""
    env = FirmEnv(npc_profile=profile)
    diffs, hired_late = [], 0
    for s in seeds:
        out = []
        for suppress in (False, True):
            obs, _ = env.reset(options={"episode_seed": s})
            done, late = False, False
            while not done:
                raw = policy.mean_action(obs.astype(np.float64))
                if env._obs["step"] >= env._obs["episode_length"] - 3:
                    if suppress:
                        raw = raw.copy()
                        raw[2] = 0.0          # squashes into the no-change dead zone
                    else:
                        late |= spaces.decode_action(raw.tolist(), env._obs)["hire_workers"] > 0
                obs, _, done, _, info = env.step(raw)
            out.append(info["net_worth"])
            if not suppress and late:
                hired_late += 1
        diffs.append(out[0] - out[1])
    return {"share_episodes_hiring_late": hired_late / len(seeds),
            "mean_cost_of_late_hiring": -st.mean(diffs),
            "ci": ci95(diffs)}


# ---------------------------------------------------------------- audit
def behavior_audit(res):
    steps = [s for r in res for s in r["trace"]]
    wage = np.array([s["wage_vs_market"] for s in steps])
    price_ratio = [s["action"]["set_price"] for s in steps]
    hist_edges = np.arange(0.875, 1.2251, 0.025)
    hist, _ = np.histogram(wage, bins=hist_edges)
    by_step = {}
    for r in res:
        for t, s in enumerate(r["trace"]):
            by_step.setdefault(t + 1, []).append((s["action"]["hire_workers"], s["workers"]))
    ordered = sorted(res, key=lambda r: r["net_worth"])
    returns = [r["return"] for r in res]
    worths = [r["net_worth"] for r in res]
    telescoping = max(abs(r["return"] - (r["net_worth"] - 10_000) / 10_000 - r["hinge"]) for r in res)
    return {
        "mean_action_at_bound": np.mean([s["at_bounds"] for s in steps], axis=0).tolist(),
        "wage_vs_market": {"p5": float(np.percentile(wage, 5)), "p50": float(np.median(wage)),
                           "p95": float(np.percentile(wage, 95)),
                           "histogram_edges": hist_edges.round(3).tolist(),
                           "histogram_counts": hist.tolist()},
        "price_set": {"p5": float(np.percentile(price_ratio, 5)),
                      "p50": float(np.median(price_ratio)),
                      "p95": float(np.percentile(price_ratio, 95))},
        "hiring_by_step": {t: {"mean_hire_request": st.mean(h for h, _ in v),
                               "mean_workers": st.mean(w for _, w in v)}
                           for t, v in sorted(by_step.items())},
        "worst_episodes": [(r["seed"], round(r["net_worth"])) for r in ordered[:3]],
        "best_episodes": [(r["seed"], round(r["net_worth"])) for r in ordered[-3:]],
        "corr_return_networth": float(np.corrcoef(returns, worths)[0, 1]),
        "max_telescoping_residual": float(telescoping),
    }


# ---------------------------------------------------------------- learning curve
def learning_curves(runs, bench_means):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logs = {}
    for k in runs:
        with open(os.path.join(RUNS_DIR, f"seed_{k}", "log.csv")) as f:
            logs[k] = list(csv.DictReader(f))
    steps = sorted(set.intersection(*[{int(r["timesteps"]) for r in v} for v in logs.values()]))
    rows = []
    for t in steps:
        row = {"timesteps": t}
        for col in ("val_net_worth", "val_heldout_net_worth", "train_return"):
            vals = [float(next(r[col] for r in logs[k] if int(r["timesteps"]) == t)) for k in runs]
            row[f"{col}_mean"], row[f"{col}_min"], row[f"{col}_max"] = st.mean(vals), min(vals), max(vals)
        rows.append(row)
    with open(os.path.join(OUT, "learning_curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    x = [r["timesteps"] / 1000 for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for col, color, label in (("val_net_worth", "#2a78d6", "PPO, default validation"),
                              ("val_heldout_net_worth", "#eb6834", "PPO, held-out validation")):
        ax1.plot(x, [r[f"{col}_mean"] for r in rows], color=color, lw=2, label=label)
        ax1.fill_between(x, [r[f"{col}_min"] for r in rows], [r[f"{col}_max"] for r in rows],
                         color=color, alpha=0.12)
    for name, ls in (("rule_based", "-"), ("market+eps", "--"), ("hold", ":")):
        ax1.axhline(bench_means["default"][name], color="#52514e", lw=1, ls=ls,
                    label=f"{AGENTS[name].label} (default validation)")
    ax1.set_xlabel("training steps (thousands)")
    ax1.set_ylabel("mean final net worth ($)")
    ax1.set_title(f"Validation net worth, mean and min-max over {len(runs)} runs")
    ax1.grid(color="#e1e0d9", lw=0.6)
    ax1.legend(fontsize=8, frameon=False)
    ax2.plot(x, [r["train_return_mean"] for r in rows], color="#1baf7a", lw=2, label="training return (mean)")
    ax2.fill_between(x, [r["train_return_min"] for r in rows], [r["train_return_max"] for r in rows],
                     color="#1baf7a", alpha=0.15)
    ax2.set_xlabel("training steps (thousands)")
    ax2.set_ylabel("episode return (net worth change / $10k)")
    ax2.set_title("Training reward (sampled actions)")
    ax2.grid(color="#e1e0d9", lw=0.6)
    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "learning_curve.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, nargs="+", default=None)
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="use VALIDATION seeds and write to results/ppo/dryrun (code check only)")
    args = ap.parse_args()
    global OUT, RUNS_DIR
    cfg = load_config(DEFAULT_CONFIG)
    runs = args.runs or cfg["seeds"]["training_runs"]
    RUNS_DIR = args.runs_dir or RUNS_DIR
    if args.dry_run:
        OUT = os.path.join("results", "ppo", "dryrun")
        test = {"default": list(range(*cfg["seeds"]["validation"])),
                "heldout": list(range(*cfg["seeds"]["validation_heldout"]))}
    else:
        test = {"default": list(range(*cfg["seeds"]["test_default"])),
                "heldout": list(range(*cfg["seeds"]["test_heldout"]))}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join("results", "ppo", "benchmark.json")) as f:
        frozen = json.load(f)["frozen_benchmark"]
    assert frozen["default"]["agent"] == BENCHMARK and frozen["heldout"]["agent"] == BENCHMARK

    print("Scripted agents on test seeds...", flush=True)
    sc = {(prof, n): scripted(n, seeds, prof) for prof, seeds in test.items() for n in AGENTS}
    sw = {k: [r["net_worth"] for r in v] for k, v in sc.items()}

    report = {"runs": {}, "scripted": {}}
    for prof in test:
        for n in AGENTS:
            report["scripted"][f"{prof}/{n}"] = {
                "mean": st.mean(sw[(prof, n)]), "ci": ci95(sw[(prof, n)]),
                "bankrupt": st.mean(r["bankrupt"] for r in sc[(prof, n)])}

    for k in runs:
        print(f"Run {k}...", flush=True)
        run_dir = os.path.join(RUNS_DIR, f"seed_{k}")
        with open(os.path.join(run_dir, "run.json")) as f:
            run_meta = json.load(f)
        prefix = "dryrun_" if args.dry_run else ""
        model, npz, js = export(os.path.join(run_dir, "model_best.zip"), f"{prefix}ppo_medium_run{k}",
                                {"run_seed": k, "best_val_net_worth": run_meta["best_val_net_worth"],
                                 "best_step": run_meta["best_step"]})
        par = parity(model, npz, js, n_obs=1000)
        pol = NumpyPolicy(npz, js)
        R = {"best_val_net_worth": run_meta["best_val_net_worth"],
             "best_step": run_meta["best_step"], "parity": par}
        res = {prof: ppo_mean(pol, seeds, prof) for prof, seeds in test.items()}
        pw = {prof: [r["net_worth"] for r in res[prof]] for prof in test}
        for prof in test:
            R[prof] = {
                "mean": st.mean(pw[prof]), "ci": ci95(pw[prof]),
                "bankrupt": st.mean(r["bankrupt"] for r in res[prof]),
                "vs": {n: paired(pw[prof], sw[(prof, n)]) for n in AGENTS},
            }
        d, h = R["default"], R["heldout"]
        bd, bh = st.mean(sw[("default", BENCHMARK)]), st.mean(sw[("heldout", BENCHMARK)])
        R["gap_closed"] = gap_closed(pw["default"], sw[("default", BENCHMARK)], sw[("default", GAP_ANCHOR)])
        R["gap_closed_hold"] = gap_closed(pw["default"], sw[("default", BENCHMARK)], sw[("default", GAP_ANCHOR_ALT)])
        ratio_d, ratio_h = d["mean"] / bd, h["mean"] / bh
        R["criteria"] = {
            "P1": all(d["vs"][n]["lo"] > 0 for n in BASELINES),
            "P2_pass": d["mean"] >= 0.90 * bd,
            "P2_strong": d["mean"] >= 0.95 * bd,
            "P2_stretch": d["vs"][BENCHMARK]["lo"] > 0,
            "P3": d["bankrupt"] <= 0.01,
            "H1": all(h["vs"][n]["lo"] > 0 for n in BASELINES),
            "H2a": h["mean"] >= 0.85 * bh,
            "H2b": ratio_h >= ratio_d - 0.10,
            "H2c": h["bankrupt"] <= 0.05,
        }
        R["criteria"]["H2"] = all(R["criteria"][c] for c in ("H2a", "H2b", "H2c"))
        R["ratios"] = {"ppo_over_bench_default": ratio_d, "ppo_over_bench_heldout": ratio_h}
        sampled_bounds, sampled_worths = ppo_sampled(pol, test["default"], "default", seed=k)
        audit = behavior_audit(res["default"])
        R["B1"] = {"mean_action_at_bound": audit["mean_action_at_bound"],
                   "sampled_action_at_bound": sampled_bounds,
                   "sampled_policy_mean_net_worth": st.mean(sampled_worths),
                   "pass": max(audit["mean_action_at_bound"]) <= 0.05}
        R["B2"] = endgame_counterfactual(pol, test["default"], "default")
        R["audit_default"] = audit
        R["audit_heldout"] = behavior_audit(res["heldout"])
        report["runs"][k] = R

    # ---- aggregate
    means = {k: report["runs"][k]["default"]["mean"] for k in runs}
    ordered = sorted(runs, key=lambda k: means[k])
    headline = ordered[len(ordered) // 2]
    leaderboard = max(runs, key=lambda k: report["runs"][k]["best_val_net_worth"])
    report["aggregate"] = {
        "R1_runs_meeting_P1_and_P3": sum(report["runs"][k]["criteria"]["P1"]
                                          and report["runs"][k]["criteria"]["P3"] for k in runs),
        "R1_pass": sum(report["runs"][k]["criteria"]["P1"] and report["runs"][k]["criteria"]["P3"]
                       for k in runs) >= 4,
        "runs_meeting_P2_pass": sum(report["runs"][k]["criteria"]["P2_pass"] for k in runs),
        "run_means": means,
        "spread": {"min": min(means.values()), "median": st.median(means.values()),
                   "max": max(means.values()),
                   "std": st.stdev(means.values()) if len(runs) > 1 else 0.0},
        "headline_run": headline,
        "leaderboard_run": leaderboard,
    }
    if args.dry_run:
        learning_curves(runs, {"default": frozen["default"]["all_means"]})
        with open(os.path.join(OUT, "evaluation.json"), "w") as f:
            json.dump(report, f, indent=2, default=float)
        write_markdown(report, runs, dry_run=True)
        print(f"Dry run done -> {OUT}")
        return
    shutil.copy(os.path.join("models", f"ppo_medium_run{leaderboard}.npz"), os.path.join("models", "ppo_medium.npz"))
    with open(os.path.join("models", f"ppo_medium_run{leaderboard}.json")) as f:
        meta = json.load(f)
    meta.update({"name": "ppo_medium", "selected_as": "leaderboard: best run by validation",
                 "rules": "v2, spillover 0, lobbying off"})
    with open(os.path.join("models", "ppo_medium.json"), "w") as f:
        json.dump(meta, f, indent=2)

    learning_curves(runs, {"default": frozen["default"]["all_means"]})
    with open(os.path.join(OUT, "evaluation.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    write_markdown(report, runs)
    print(f"Done -> {OUT}")


def write_markdown(rep, runs, dry_run=False):
    L = []
    agg = rep["aggregate"]
    hk, lk = agg["headline_run"], agg["leaderboard_run"]
    L.append("# PPO evaluation DRY RUN (validation seeds, code check only)\n" if dry_run
             else "# PPO final evaluation (test seeds)\n")
    L.append(f"Headline = median run (**run {hk}**); leaderboard = best run by validation (**run {lk}**).\n")
    L.append("## Criteria per run\n")
    L.append("| Run | Default mean | Held-out mean | P1 | P2 pass/strong/stretch | P3 | H1 | H2 a/b/c | B1 | Gap closed (vs Market+1%) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    yn = lambda b: "✓" if b else "✗"
    for k in runs:
        R = rep["runs"][k]
        c = R["criteria"]
        g = R["gap_closed"]
        L.append(f"| {k} | {R['default']['mean']:,.0f} | {R['heldout']['mean']:,.0f} | {yn(c['P1'])} | "
                 f"{yn(c['P2_pass'])}/{yn(c['P2_strong'])}/{yn(c['P2_stretch'])} | {yn(c['P3'])} | {yn(c['H1'])} | "
                 f"{yn(c['H2a'])}/{yn(c['H2b'])}/{yn(c['H2c'])} | {yn(R['B1']['pass'])} | "
                 f"{g['point']:.2f} [{g['lo']:.2f}, {g['hi']:.2f}] |")
    L.append(f"\nR1: {agg['R1_runs_meeting_P1_and_P3']}/5 runs meet P1 and P3 "
             f"({'pass' if agg['R1_pass'] else 'fail'}); {agg['runs_meeting_P2_pass']}/5 meet P2 (pass). "
             f"Run means: min {agg['spread']['min']:,.0f}, median {agg['spread']['median']:,.0f}, "
             f"max {agg['spread']['max']:,.0f}, std {agg['spread']['std']:,.0f}.\n")
    for prof in ("default", "heldout"):
        R = rep["runs"][hk][prof]
        L.append(f"## Headline run {hk} vs every agent — {prof} test seeds\n")
        L.append("| Agent | Agent mean | PPO − agent (95% CI) | PPO wins |")
        L.append("|---|---|---|---|")
        for n in BASELINES + EXTRA_ROWS:
            s = rep["scripted"][f"{prof}/{n}"]
            v = R["vs"][n]
            L.append(f"| {AGENTS[n].label} | {s['mean']:,.0f} | {v['mean_diff']:+,.0f} "
                     f"[{v['lo']:+,.0f}, {v['hi']:+,.0f}] | {v['win_rate']:.0%} |")
        L.append(f"\nPPO mean {R['mean']:,.0f} ± {R['ci']:,.0f}; bankrupt {R['bankrupt']:.1%}.\n")
    L.append("## Scripted agents: default → held-out (test)\n")
    L.append("| Agent | Default | Held-out | Ratio | Held-out bankrupt |")
    L.append("|---|---|---|---|---|")
    for n in AGENTS:
        dflt, held = rep["scripted"][f"default/{n}"], rep["scripted"][f"heldout/{n}"]
        ratio = f"{held['mean'] / dflt['mean']:.2f}" if dflt["mean"] > 0 and held["mean"] > 0 else "n/a"
        L.append(f"| {AGENTS[n].label} | {dflt['mean']:,.0f} | {held['mean']:,.0f} | {ratio} | {held['bankrupt']:.0%} |")
    R = rep["runs"][hk]
    a = R["audit_default"]
    L.append(f"\n## Behavior audit (run {hk}, default test)\n")
    L.append(f"- At bound, mean action (price/wage/hire): {', '.join(f'{x:.1%}' for x in R['B1']['mean_action_at_bound'])}; "
             f"sampled: {', '.join(f'{x:.1%}' for x in R['B1']['sampled_action_at_bound'])}")
    L.append(f"- Wage / market: p5 {a['wage_vs_market']['p5']:.3f}, median {a['wage_vs_market']['p50']:.3f}, "
             f"p95 {a['wage_vs_market']['p95']:.3f}")
    L.append(f"- Price set: p5 ${a['price_set']['p5']:.2f}, median ${a['price_set']['p50']:.2f}, p95 ${a['price_set']['p95']:.2f}")
    L.append(f"- Endgame: {R['B2']['share_episodes_hiring_late']:.0%} of episodes hire in the last 3 steps; "
             f"cost {R['B2']['mean_cost_of_late_hiring']:+,.0f} ± {R['B2']['ci']:,.0f} net worth")
    L.append(f"- Return vs net worth correlation {a['corr_return_networth']:.4f}; max telescoping residual "
             f"{a['max_telescoping_residual']:.2e}")
    L.append(f"- Worst episodes (seed, net worth): {a['worst_episodes']}; best: {a['best_episodes']}")
    L.append(f"- Export parity (run {hk}): max |torch − numpy| {R['parity']['max_abs_diff_raw_mean']:.2e} "
             f"on {R['parity']['n_obs']} observations")
    with open(os.path.join(OUT, "evaluation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
