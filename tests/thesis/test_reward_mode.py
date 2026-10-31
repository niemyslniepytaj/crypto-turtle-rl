"""reward_mode: legacy (bez zmian) vs excess (port − HODL bench)."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from rl_module.services.rl.agent import TradingEnv
from rl_module.services.thesis.knobs import (
    PPO_HPARAMS,
    build_pretrain_ppo_kwargs,
    thesis_train_config_hash,
)


def _stub_env(
    *,
    reward_mode: str,
    prev_value: float,
    current_value: float,
    closes_by_symbol: dict,
    t: int = 1,
):
    """Minimalny self pod TradingEnv._calculate_reward (bez pełnego env)."""
    models = {
        sym: SimpleNamespace(close_prices=np.asarray(prices, dtype=float))
        for sym, prices in closes_by_symbol.items()
    }
    return SimpleNamespace(
        core_config=SimpleNamespace(reward_mode=reward_mode),
        symbols=list(closes_by_symbol.keys()),
        t=int(t),
        engine=SimpleNamespace(
            prev_value=float(prev_value),
            current_value=float(current_value),
            models=models,
        ),
    )


def test_legacy_exact_gain_and_loss():
    # +1%: log(1.01)*100, bez ×2
    env_up = _stub_env(
        reward_mode="legacy",
        prev_value=1000.0,
        current_value=1010.0,
        closes_by_symbol={"ETH": [100.0, 101.0], "BTC": [200.0, 202.0]},
    )
    r_up = TradingEnv._calculate_reward(env_up)
    expected_up = float(np.log(1010.0 / 1000.0) * 100.0)
    assert abs(r_up - expected_up) < 1e-12

    # −1%: log(0.99)*100*2
    env_dn = _stub_env(
        reward_mode="legacy",
        prev_value=1000.0,
        current_value=990.0,
        closes_by_symbol={"ETH": [100.0, 99.0], "BTC": [200.0, 198.0]},
    )
    r_dn = TradingEnv._calculate_reward(env_dn)
    expected_dn = float(np.log(990.0 / 1000.0) * 100.0 * 2.0)
    assert abs(r_dn - expected_dn) < 1e-12


def test_excess_hodl_policy_reward_near_zero():
    """Równa waga HODL: portfel rośnie jak średnia log-returnów → excess ≈ 0."""
    # ETH +1%, BTC +1% → bench = log(1.01); portfel 1000→1010
    env = _stub_env(
        reward_mode="excess",
        prev_value=1000.0,
        current_value=1010.0,
        closes_by_symbol={"ETH": [100.0, 101.0], "BTC": [50.0, 50.5]},
    )
    r = TradingEnv._calculate_reward(env)
    assert abs(r) < 1e-9, f"HODL excess powinien być ~0, jest {r}"


def test_excess_full_cash_on_rising_market_is_negative():
    """Gotówka (portfel flat) przy rosnącym rynku → ujemny excess."""
    env = _stub_env(
        reward_mode="excess",
        prev_value=1000.0,
        current_value=1000.0,
        closes_by_symbol={"ETH": [100.0, 105.0], "BTC": [200.0, 210.0]},
    )
    r = TradingEnv._calculate_reward(env)
    assert r < 0.0, f"cash na rising powinien dawać r<0, jest {r}"
    bench = 0.5 * (np.log(105 / 100) + np.log(210 / 200))
    assert abs(r - float(-bench * 100.0)) < 1e-12


def test_reward_mode_in_hparams_kwargs_and_hash():
    assert PPO_HPARAMS.get("reward_mode") == "excess"
    kw = build_pretrain_ppo_kwargs(seed=396379, learning_rate=1.0)
    assert kw["reward_mode"] == "excess"

    h1 = thesis_train_config_hash()
    assert "reward_mode" in str(
        {**PPO_HPARAMS}
    )  # w hashu przez PPO_HPARAMS
    # zmiana reward_mode zmienia hash
    old = PPO_HPARAMS["reward_mode"]
    try:
        PPO_HPARAMS["reward_mode"] = "legacy"
        h2 = thesis_train_config_hash()
        assert h1 != h2
        assert len(h1) == 64 and len(h2) == 64
    finally:
        PPO_HPARAMS["reward_mode"] = old
    assert thesis_train_config_hash() == h1
