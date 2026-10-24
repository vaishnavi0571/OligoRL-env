"""
Numpy-only PPO policy (inference path: no torch, no SB3, no gymnasium)
=====================================================================
Loads the exported weights (models/<name>.npz) and metadata
(models/<name>.json) and computes the MEAN action of the actor network:
    h = tanh(W_i h + b_i) for each hidden layer;  mean = W_out h + b_out
The raw mean is then decoded with ppo/spaces.py, exactly as in training.
"""

import json
from typing import Dict

import numpy as np

from ppo import spaces


class NumpyPolicy:
    def __init__(self, npz_path: str, json_path: str):
        with open(json_path) as f:
            self.meta = json.load(f)
        weights = np.load(npz_path)
        n_hidden = self.meta["n_hidden_layers"]
        self.hidden = [(weights[f"W{i}"], weights[f"b{i}"]) for i in range(n_hidden)]
        self.out = (weights["W_out"], weights["b_out"])
        self.log_std = weights["log_std"] if "log_std" in weights.files else None
        if self.meta["spaces"] != spaces.spec_dict():
            raise ValueError("Model was exported with different observation/action spaces")

    def mean_action(self, obs_vec: np.ndarray) -> np.ndarray:
        """Raw (pre-tanh) mean action for one or many encoded observations."""
        h = np.asarray(obs_vec, dtype=np.float64)
        for W, b in self.hidden:
            h = np.tanh(h @ W.T + b)
        W, b = self.out
        return h @ W.T + b

    def sample_action(self, obs_vec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Raw action sampled from the training distribution (audit only;
        evaluation and deployment use the mean action)."""
        mean = self.mean_action(obs_vec)
        return mean + np.exp(self.log_std) * rng.standard_normal(mean.shape)

    def act(self, obs: Dict) -> Dict:
        """Simulator observation dict -> simulator action dict."""
        x = np.asarray(spaces.encode_observation(obs), dtype=np.float64)
        return spaces.decode_action(self.mean_action(x).tolist(), obs)
