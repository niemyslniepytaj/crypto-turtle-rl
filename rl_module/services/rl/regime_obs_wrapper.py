"""Dokleja kanał reżimu PO VecNormalize — per-env wg indeksu czasu t (I2 / Z1.1)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnvWrapper


class ConcatRegimeVecEnv(VecEnvWrapper):
    """
    venv zwraca obs rynkową (już znormowaną); tu doklejamy one-hot reżimu długości K.

    regime_onehot_series: (T, K) — lookup po get_attr('t') dla KAŻDEGO środowiska
    (nie tile jednego wektora). Brak serii / mode zeros → zera (R2/M2).
    """

    def __init__(
        self,
        venv,
        regime_k: int = 3,
        regime_onehot_series: Optional[np.ndarray] = None,
        zeros_only: bool = False,
    ):
        super().__init__(venv)
        self.regime_k = int(max(regime_k, 0))
        self.zeros_only = bool(zeros_only)
        self._series: Optional[np.ndarray] = None
        if regime_onehot_series is not None and not self.zeros_only:
            s = np.asarray(regime_onehot_series, dtype=np.float32)
            if s.ndim == 1:
                s = s.reshape(-1, self.regime_k) if self.regime_k else s.reshape(-1, 1)
            self._series = s
        if self.regime_k <= 0:
            return
        old = venv.observation_space
        low = np.concatenate(
            [np.asarray(old.low, dtype=np.float32).ravel(), np.zeros(self.regime_k, dtype=np.float32)]
        )
        high = np.concatenate(
            [np.asarray(old.high, dtype=np.float32).ravel(), np.ones(self.regime_k, dtype=np.float32)]
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def set_regime_series(self, series: Optional[np.ndarray], zeros_only: Optional[bool] = None):
        if zeros_only is not None:
            self.zeros_only = bool(zeros_only)
        if series is None or self.zeros_only:
            self._series = None
            return
        s = np.asarray(series, dtype=np.float32)
        if s.ndim == 1 and self.regime_k:
            s = s.reshape(-1, self.regime_k)
        self._series = s

    def _lookup_batch(self, n: int) -> np.ndarray:
        if self.regime_k <= 0:
            return np.zeros((n, 0), dtype=np.float32)
        out = np.zeros((n, self.regime_k), dtype=np.float32)
        if self.zeros_only or self._series is None or len(self._series) == 0:
            return out
        try:
            ts = self.venv.get_attr("t")
        except Exception:
            ts = [0] * n
        if not isinstance(ts, (list, tuple)):
            ts = list(ts)
        tmax = len(self._series) - 1
        for i in range(n):
            ti = int(ts[i]) if i < len(ts) else 0
            ti = min(max(ti, 0), tmax)
            vec = self._series[ti]
            m = min(self.regime_k, int(np.asarray(vec).ravel().shape[0]))
            out[i, :m] = np.asarray(vec, dtype=np.float32).ravel()[:m]
        return out

    def _concat(self, obs):
        if self.regime_k <= 0:
            return obs
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        reg = self._lookup_batch(obs.shape[0])
        return np.concatenate([obs, reg], axis=1)

    def reset(self, **kwargs):
        obs = self.venv.reset(**kwargs)
        if isinstance(obs, tuple):
            o, info = obs
            return self._concat(o), info
        return self._concat(obs)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._concat(obs), rewards, dones, infos
