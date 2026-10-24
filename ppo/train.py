"""
Train one PPO run (Stable-Baselines3, in-process simulator)
===========================================================
Every eval_freq steps the deterministic (mean-action) policy is scored on
the validation seeds; the checkpoint with the best mean validation net
worth is kept as model_best.zip.  Test seeds are never touched here.

Outputs in --out:
  log.csv          timesteps, training return / net worth, validation
                   net worth / return / bankruptcies, action-bound shares
  model_best.zip   best checkpoint by validation net worth
  model_final.zip  last checkpoint
  run.json         config + run seed + best checkpoint info

Run:  python -m ppo.train --run-seed 1 [--timesteps 10000] [--out DIR]
"""

import argparse
import csv
import json
import os
import time
from collections import deque

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from ppo.env import FirmEnv, rollout
from ppo import spaces

DEFAULT_CONFIG = os.path.join("ppo", "configs", "medium.json")


def load_config(path):
    with open(path) as f:
        return json.load(f)


class ValidationCallback(BaseCallback):
    """Validation scoring, checkpoint selection and CSV logging."""

    def __init__(self, cfg, out_dir, eval_freq, val_seeds, heldout_val_seeds=None):
        super().__init__()
        self.cfg = cfg
        self.out_dir = out_dir
        self.eval_freq = eval_freq
        self.val_seeds = val_seeds
        self.heldout_val_seeds = heldout_val_seeds
        self.best = -float("inf")
        self.best_step = 0
        self.episodes = deque(maxlen=100)       # (return, net worth)
        self.bound_hits = np.zeros(spaces.ACTION_DIM)
        self.action_count = 0
        self.rows = []
        self.last_eval = 0
        self.t0 = time.time()

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            self.bound_hits += np.asarray(info["at_bounds"], dtype=float)
            self.action_count += 1
            if "episode" in info:
                self.episodes.append((info["episode"]["r"], info["net_worth"]))
        return True

    def _on_rollout_start(self) -> None:
        # Runs right after each policy update, so the scored policy is the
        # one that would be saved
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self._evaluate()

    def _on_training_end(self) -> None:
        if self.num_timesteps != self.last_eval:
            self._evaluate()

    def _evaluate(self):
        policy = lambda obs: self.model.predict(obs, deterministic=True)[0]
        sk = self.cfg.get("sim_kwargs", {})
        res = rollout(policy, self.val_seeds, self.cfg["difficulty"], self.cfg["val_profile"],
                      record=True, sim_kwargs=sk)
        worth = float(np.mean([r["net_worth"] for r in res]))
        mean_bounds = np.mean([s["at_bounds"] for r in res for s in r["trace"]], axis=0)
        row = {
            "timesteps": self.num_timesteps,
            "wall_s": round(time.time() - self.t0, 1),
            "train_return": float(np.mean([e[0] for e in self.episodes])) if self.episodes else float("nan"),
            "train_net_worth": float(np.mean([e[1] for e in self.episodes])) if self.episodes else float("nan"),
            "val_net_worth": worth,
            "val_return": float(np.mean([r["return"] for r in res])),
            "val_bankrupt": float(np.mean([r["bankrupt"] for r in res])),
            **{f"at_bound_sampled_{n}": float(self.bound_hits[i] / max(self.action_count, 1))
               for i, n in enumerate(spaces.ACTION_NAMES)},
            **{f"at_bound_mean_{n}": float(mean_bounds[i])
               for i, n in enumerate(spaces.ACTION_NAMES)},
            "action_std": float(torch.exp(self.model.policy.log_std.detach()).mean()),
        }
        if self.heldout_val_seeds:
            res_h = rollout(policy, self.heldout_val_seeds, self.cfg["difficulty"], "heldout",
                            sim_kwargs=sk)
            row["val_heldout_net_worth"] = float(np.mean([r["net_worth"] for r in res_h]))
        self.last_eval = self.num_timesteps
        self.bound_hits[:] = 0
        self.action_count = 0
        self.rows.append(row)
        # Pre-registered selection: best default-validation net worth among
        # checkpoints with validation bankruptcy <= 1% (ties -> earlier)
        if worth > self.best and row["val_bankrupt"] <= 0.01:
            self.best, self.best_step = worth, self.num_timesteps
            self.model.save(os.path.join(self.out_dir, "model_best.zip"))
        with open(os.path.join(self.out_dir, "log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(self.rows)
        held = f"  heldout {row['val_heldout_net_worth']:9,.0f}" if "val_heldout_net_worth" in row else ""
        print(f"  [{row['timesteps']:>7d}] train return {row['train_return']:6.3f}"
              f"  val net worth {worth:9,.0f}{held}  bankrupt {row['val_bankrupt']:4.0%}"
              f"  at-bound sampled p/w/h {row['at_bound_sampled_price']:.2f}/"
              f"{row['at_bound_sampled_wage']:.2f}/{row['at_bound_sampled_hire']:.2f}"
              f"  mean {row['at_bound_mean_price']:.2f}/{row['at_bound_mean_wage']:.2f}/"
              f"{row['at_bound_mean_hire']:.2f}  std {row['action_std']:.2f}", flush=True)


def make_env(cfg, rank, run_seed):
    def _init():
        env = Monitor(FirmEnv(cfg["difficulty"], cfg["train_profile"],
                              sim_kwargs=cfg.get("sim_kwargs", {})))
        env.reset(seed=run_seed * 1000 + rank)
        return env
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--run-seed", type=int, default=1)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--eval-freq", type=int, default=None)
    ap.add_argument("--val-episodes", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--spillover", type=float, default=None,
                    help="override the simulator's spillover fraction (train + validation)")
    ap.add_argument("--val-heldout", action="store_true",
                    help="also score the heldout-profile validation range (logging only)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("sim_kwargs", {})
    if args.spillover is not None:
        cfg["sim_kwargs"]["spillover_fraction"] = args.spillover
    p = cfg["ppo"]
    timesteps = args.timesteps or cfg["total_timesteps"]
    eval_freq = args.eval_freq or cfg["eval_freq"]
    n_val = args.val_episodes or cfg["val_episodes"]
    lo, hi = cfg["seeds"]["validation"]
    val_seeds = list(range(lo, min(hi, lo + n_val)))
    heldout_val = None
    if args.val_heldout:
        hlo, hhi = cfg["seeds"]["validation_heldout"]
        heldout_val = list(range(hlo, min(hhi, hlo + n_val)))
    out = args.out or os.path.join("results", "ppo", "runs", f"seed_{args.run_seed}")
    os.makedirs(out, exist_ok=True)

    torch.manual_seed(args.run_seed)
    np.random.seed(args.run_seed)
    torch.set_num_threads(1)
    venv = DummyVecEnv([make_env(cfg, i, args.run_seed) for i in range(p["n_envs"])])
    model = PPO(
        "MlpPolicy", venv, seed=args.run_seed, verbose=0,
        n_steps=p["n_steps"], batch_size=p["batch_size"], n_epochs=p["n_epochs"],
        learning_rate=p["learning_rate"], gamma=p["gamma"], gae_lambda=p["gae_lambda"],
        clip_range=p["clip_range"], ent_coef=p["ent_coef"], vf_coef=p["vf_coef"],
        max_grad_norm=p["max_grad_norm"],
        policy_kwargs={"net_arch": p["net_arch"], "log_std_init": p["log_std_init"]},
    )
    cb = ValidationCallback(cfg, out, eval_freq, val_seeds, heldout_val)
    print(f"Run seed {args.run_seed}: {timesteps:,} steps, validating every {eval_freq:,} "
          f"on {len(val_seeds)} seeds -> {out}", flush=True)
    model.learn(total_timesteps=timesteps, callback=cb)
    model.save(os.path.join(out, "model_final.zip"))
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump({"config": cfg, "run_seed": args.run_seed, "timesteps": timesteps,
                   "best_val_net_worth": cb.best, "best_step": cb.best_step,
                   "spaces": spaces.spec_dict()}, f, indent=2)
    print(f"Best validation net worth {cb.best:,.0f} at step {cb.best_step:,}")


if __name__ == "__main__":
    main()
