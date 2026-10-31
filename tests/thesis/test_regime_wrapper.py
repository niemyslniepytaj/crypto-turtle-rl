"""Z1.6a — per-env etykiety (nie tile); zera dla rl."""
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym


class _ToyEnv(gym.Env):
    """Proste środowisko Gym do testów ConcatRegimeVecEnv."""
    def __init__(self, t0=0):
        super().__init__()
        self.t = int(t0)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        return np.zeros(2, dtype=np.float32), 0.0, False, False, {}


def test_per_env_different_labels():
    from rl_module.services.rl.regime_obs_wrapper import ConcatRegimeVecEnv

    # serie: t=0 -> [1,0,0], t=1 -> [0,1,0], t=2 -> [0,0,1]
    series = np.eye(3, dtype=np.float32)
    # dwa env startujące z różnym t
    def make0():
        return _ToyEnv(t0=0)

    def make1():
        return _ToyEnv(t0=2)

    venv = DummyVecEnv([make0, make1])
    wrapped = ConcatRegimeVecEnv(venv, regime_k=3, regime_onehot_series=series, zeros_only=False)
    obs = wrapped.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    assert obs.shape == (2, 5)  # 2 market + 3 regime
    # env0 t=0 → e0; env1 t=2 → e2
    assert np.allclose(obs[0, 2:], [1, 0, 0])
    assert np.allclose(obs[1, 2:], [0, 0, 1])
    # po step: t=1 i t=3 (clamp do 2)
    obs2, *_ = wrapped.step(np.zeros((2, 1), dtype=np.float32))
    assert np.allclose(obs2[0, 2:], [0, 1, 0])
    assert np.allclose(obs2[1, 2:], [0, 0, 1])


def test_zeros_only_rl():
    from rl_module.services.rl.regime_obs_wrapper import ConcatRegimeVecEnv

    series = np.eye(3, dtype=np.float32)
    venv = DummyVecEnv([lambda: _ToyEnv(0), lambda: _ToyEnv(1)])
    wrapped = ConcatRegimeVecEnv(venv, regime_k=3, regime_onehot_series=series, zeros_only=True)
    obs = wrapped.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    assert np.allclose(obs[:, 2:], 0.0)
