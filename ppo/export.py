"""
Export an SB3 PPO checkpoint to numpy (npz + json) and verify parity
====================================================================
Writes models/<name>.npz (weights) and models/<name>.json (architecture,
observation/action spec, provenance), then compares the torch mean
action with the numpy one on >= N observations (validation rollouts plus
random encoded observations) and reports the max absolute difference.

Run:  python -m ppo.export --model results/ppo/runs/seed_1/model_best.zip
                           [--name ppo_medium] [--n-obs 1000]
"""

import argparse
import json
import os

import numpy as np
import torch
from stable_baselines3 import PPO

from ppo import spaces
from ppo.env import FirmEnv
from ppo.policy_numpy import NumpyPolicy

MODELS_DIR = "models"


def export(model_path: str, name: str, extra_meta=None):
    model = PPO.load(model_path, device="cpu")
    pol = model.policy
    if not isinstance(pol.activation_fn(), torch.nn.Tanh):
        raise ValueError("Exporter assumes Tanh activations")
    linears = [m for m in pol.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    arrays = {}
    for i, layer in enumerate(linears):
        arrays[f"W{i}"] = layer.weight.detach().double().numpy()
        arrays[f"b{i}"] = layer.bias.detach().double().numpy()
    arrays["W_out"] = pol.action_net.weight.detach().double().numpy()
    arrays["b_out"] = pol.action_net.bias.detach().double().numpy()
    arrays["log_std"] = pol.log_std.detach().double().numpy()   # sampled-action audit only

    os.makedirs(MODELS_DIR, exist_ok=True)
    npz_path = os.path.join(MODELS_DIR, f"{name}.npz")
    json_path = os.path.join(MODELS_DIR, f"{name}.json")
    np.savez(npz_path, **arrays)
    meta = {
        "name": name,
        "source_checkpoint": model_path.replace("\\", "/"),
        "n_hidden_layers": len(linears),
        "hidden_sizes": [int(l.out_features) for l in linears],
        "activation": "tanh",
        "obs_dim": spaces.OBS_DIM,
        "action_dim": spaces.ACTION_DIM,
        "eval_action": "mean (deterministic), then clip to raw bound, tanh, decode",
        "spaces": spaces.spec_dict(),
        **(extra_meta or {}),
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    return model, npz_path, json_path


def parity(model, npz_path, json_path, n_obs=1000, seed=0):
    """Max |torch mean - numpy mean| over real and random observations."""
    rng = np.random.default_rng(seed)
    obs = []
    env = FirmEnv(npc_profile="default")
    policy = lambda o: model.predict(o, deterministic=True)[0]
    s = 50_000
    while len(obs) < n_obs // 2:
        o, _ = env.reset(options={"episode_seed": s})
        done = False
        while not done and len(obs) < n_obs // 2:
            obs.append(o)
            o, _, done, _, _ = env.step(policy(o))
        s += 1
    obs += list(rng.uniform(-spaces.OBS_CLIP, spaces.OBS_CLIP,
                            size=(n_obs - len(obs), spaces.OBS_DIM)))
    X = np.asarray(obs, dtype=np.float32)
    with torch.no_grad():
        dist = model.policy.get_distribution(torch.as_tensor(X))
        torch_mean = dist.distribution.mean.double().numpy()
    np_mean = NumpyPolicy(npz_path, json_path).mean_action(X.astype(np.float64))
    raw_diff = float(np.max(np.abs(torch_mean - np_mean)))
    clip = spaces.RAW_ACTION_BOUND
    squashed = lambda a: np.tanh(np.clip(a, -clip, clip))
    sq_diff = float(np.max(np.abs(squashed(torch_mean) - squashed(np_mean))))
    return {"n_obs": len(X), "max_abs_diff_raw_mean": raw_diff,
            "max_abs_diff_squashed": sq_diff}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="ppo_medium")
    ap.add_argument("--n-obs", type=int, default=1000)
    args = ap.parse_args()
    model, npz_path, json_path = export(args.model, args.name)
    result = parity(model, npz_path, json_path, args.n_obs)
    with open(json_path) as f:
        meta = json.load(f)
    meta["parity"] = result
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Exported {npz_path} + {json_path}")
    print(f"Parity on {result['n_obs']} observations: max |torch - numpy| raw mean "
          f"{result['max_abs_diff_raw_mean']:.2e}, after tanh {result['max_abs_diff_squashed']:.2e}")


if __name__ == "__main__":
    main()
