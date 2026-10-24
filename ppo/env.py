"""
Gymnasium environment for PPO, running the simulator in-process
===============================================================
- Observation / action encoding: ppo/spaces.py (shared with inference)
- Reward: rewards.net_worth_reward (config 7e), gamma = 1
- Lobbying disabled; NPC parameters drawn from an experiment profile
- The episode end (bankruptcy or the final step) is reported as
  `terminated`, never `truncated`: the horizon is fixed and the agent
  sees step/T, so nothing should be bootstrapped past the last step.

Episode seeds: a fixed list (validation / test) or, by default, drawn
from TRAIN_SEED_RANGE with the env's own RNG.
"""

from typing import Callable, Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np

from simulator import EconSimulator
from rewards import net_worth_reward
from agents import terminal_net_worth
from ppo import spaces

TRAIN_SEED_RANGE = (100_000, 1_000_000_000)


class FirmEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, difficulty: str = "medium", npc_profile: str = "train",
                 episode_seeds: Optional[Sequence[int]] = None,
                 sim_kwargs: Optional[Dict] = None):
        """sim_kwargs: extra EconSimulator flags, e.g. spillover_fraction."""
        super().__init__()
        self.difficulty = difficulty
        self.npc_profile = npc_profile
        self.sim_kwargs = dict(sim_kwargs or {})
        self.episode_seeds = list(episode_seeds) if episode_seeds is not None else None
        self._next_seed_idx = 0
        self.observation_space = gym.spaces.Box(
            -spaces.OBS_CLIP, spaces.OBS_CLIP, shape=(spaces.OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            -spaces.RAW_ACTION_BOUND, spaces.RAW_ACTION_BOUND,
            shape=(spaces.ACTION_DIM,), dtype=np.float32)
        self.sim: Optional[EconSimulator] = None
        self._obs: Optional[Dict] = None
        self._worth = 0.0

    def _pick_seed(self) -> int:
        if self.episode_seeds is not None:
            seed = self.episode_seeds[self._next_seed_idx % len(self.episode_seeds)]
            self._next_seed_idx += 1
            return seed
        return int(self.np_random.integers(*TRAIN_SEED_RANGE))

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        episode_seed = (options or {}).get("episode_seed", self._pick_seed())
        self.sim = EconSimulator(difficulty=self.difficulty, seed=episode_seed,
                                 npc_profile=self.npc_profile, lobbying_enabled=False,
                                 **self.sim_kwargs)
        self._obs = self.sim.reset()
        self._worth = terminal_net_worth(self.sim)
        return np.asarray(spaces.encode_observation(self._obs), dtype=np.float32), {
            "episode_seed": episode_seed}

    def step(self, raw_action):
        raw = np.asarray(raw_action, dtype=np.float64).tolist()
        action = spaces.decode_action(raw, self._obs)
        self._obs, info = self.sim.step(action)
        worth = terminal_net_worth(self.sim)
        reward, delta, hinge = net_worth_reward(
            self._worth, worth, self.sim.player.cash, info.labor_cost,
            self.sim.preset.episode_length)
        self._worth = worth
        done = self.sim.terminated or self.sim.truncated
        return (
            np.asarray(spaces.encode_observation(self._obs), dtype=np.float32),
            float(reward), bool(done), False,
            {"net_worth": worth, "delta": delta, "hinge": hinge,
             "bankrupt": self.sim.terminated, "at_bounds": spaces.at_bounds(raw),
             "action": action},
        )


def rollout(policy: Callable[[np.ndarray], np.ndarray], seeds: Sequence[int],
            difficulty: str = "medium", npc_profile: str = "default",
            record: bool = False, sim_kwargs: Optional[Dict] = None) -> List[Dict]:
    """Play one episode per seed with a deterministic policy
    (obs vector -> raw action).  Returns per-episode summaries."""
    results = []
    env = FirmEnv(difficulty=difficulty, npc_profile=npc_profile, sim_kwargs=sim_kwargs)
    for seed in seeds:
        obs, _ = env.reset(options={"episode_seed": seed})
        ret = hinge = 0.0
        steps = []
        done = False
        while not done:
            raw = np.asarray(policy(obs), dtype=np.float64)
            obs, r, done, _, info = env.step(raw)
            ret += r
            hinge += info["hinge"]
            if record:
                steps.append({"raw": raw.tolist(), "action": info["action"],
                              "at_bounds": info["at_bounds"], "reward": r,
                              "net_worth": info["net_worth"],
                              "wage_vs_market": env._obs["wage"] / env._obs["market_wage"],
                              "workers": env._obs["worker_count"]})
        results.append({"seed": seed, "net_worth": info["net_worth"], "return": ret,
                        "hinge": hinge, "bankrupt": info["bankrupt"],
                        "steps": env.sim.step_count, **({"trace": steps} if record else {})})
    return results
