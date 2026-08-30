"""Wrapper che concatena le masse vere all'osservazione, per la policy oracolo.

masses="links" -> thigh, leg, foot (le tre randomizzate)
masses="all"   -> anche il torso
"""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

MASS_SLICES = {"links": slice(1, 4), "all": slice(0, 4)}


class OracleWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, masses: str = "links"):
        super().__init__(env)
        if masses not in MASS_SLICES:
            raise ValueError(f"masses deve essere uno di {list(MASS_SLICES)}")
        self._slice = MASS_SLICES[masses]
        self._n_extra = self._slice.stop - self._slice.start

        base = env.observation_space
        low = np.concatenate([base.low, np.full(self._n_extra, -np.inf)])
        high = np.concatenate([base.high, np.full(self._n_extra, np.inf)])
        self.observation_space = Box(low=low, high=high, dtype=np.float64)

    def _augment(self, obs: np.ndarray) -> np.ndarray:
        true_masses = self.env.unwrapped.get_parameters()[self._slice]
        return np.concatenate([obs, true_masses]).astype(np.float64)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(obs), reward, terminated, truncated, info
