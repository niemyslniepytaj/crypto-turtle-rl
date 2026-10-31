"""
Wspólna przestrzeń knobów MP: LOCK (R1) i M1.

6 free = te same 6 pokręteł co akcja agenta RL (v5rl): bias, scale, confidence,
urgency, signal_dz, rebalance_dz. Limity ekspozycji (wlim, maxc) są STAŁE.
Reszta constant.

Teza = portfel ETH+BTC: knoby SHARED (nie per-coin ×N).
Szczegóły: thesis.assets.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

from .assets import max_wallet_optuna_range, n_symbols as _n_symbols

# Free — tunowane naraz (flat 6F, bez O1→O2)
MP_FREE = (
    "mp_alloc_bias",
    "mp_alloc_scale",
    "mp_confidence",
    "mp_urgency",
    "mp_signal_deadzone",
    "mp_rebalance_deadzone",
)

# Zakresy Optuna [lo, hi] — identyczne z zakresem fizycznym akcji agenta RL
# (version_registry.v5rl_decision_rl_config: u∈[0,1]; signal_dz=max(0.05,0.5u); reb_dz=max(0.12,0.5u))
MP_FREE_RANGES = {
    "mp_alloc_bias": (0.00, 1.00),
    "mp_alloc_scale": (0.00, 1.00),
    "mp_confidence": (0.00, 1.00),
    "mp_urgency": (0.00, 1.00),
    "mp_signal_deadzone": (0.05, 0.50),
    "mp_rebalance_deadzone": (0.12, 0.50),
}

# Constant — warsztat / LOCK defaults
MP_CONST = {
    "add_fee": True,
    "slippage": 0.0,
    "threshold_buy": 0.5,
    "threshold_sell": 0.5,
    "stop_loss": 0.03,
    "stop_loss_cooldown": 120,
    "mincost": 12.0,
    # limity ekspozycji: wspólne, stałe dla R1/M1/R2/M2 (RL ich nie steruje)
    "wallet_percentage_limit": 0.85,
    # 0.50 do decyzji po eksperymencie A1–A2 (0.75 vs 0.5); patrz plgrid_compare_mwpc
    "max_wallet_percentage_crypto": 0.50,
    "mp_ema_alpha_min": 0.02,
    "mp_ema_alpha_span": 0.08,
}

# Teza: bez klampu 1/N (CoreConfig.clamp_wallet_cap_to_1n=False) — rotacja maxc>0.5
CLAMP_WALLET_CAP_TO_1N = False

# Domyślne wartości free (gdy brak LOCK)
MP_FREE_DEFAULTS = {
    "mp_alloc_bias": 0.5,
    "mp_alloc_scale": 0.70,
    "mp_confidence": 0.5,
    "mp_urgency": 0.5,
    "mp_signal_deadzone": 0.10,
    "mp_rebalance_deadzone": 0.25,
}

# Wspólne wagi / bramki Optuny (R1 LOCK entry + M1) — jedna prawda
OPTUNA_OBJECTIVE = {
    "sharpe_weight": 5.0,
    "dd_weight": 0.75,
    "invalid_excess_weight": 0.05,
    "ruin_return_floor": -20.0,
    "ruin_penalty_weight": 2.0,
    "invalid_floor": -400.0,
    "corr_weight": 0.0,
    "cash_weight": 0.0,
    "shape_penalty": 0.0,
}

TUNER_GATES = {
    "min_tx_per_day": 0.7,
    "max_tx_per_day": 6.0,
    "overtrade_hard_mult": 1.75,
    "min_tx_floor": 12,
    "min_closed_per_day": 0.05,
    "min_closed_floor": 5,
    "hodl_corr_soft": 0.90,
    "hodl_corr_hard": 0.985,
}

# Binance spot retail (praca 2.x / 20.10): 0.1% od notional
HODL_ENTRY_FEE_RATE = 0.001
HODL_ENTRY_FEE_DEFAULT = False  # R0 = czyste odniesienie rynku (= hodl_return w excess)

# Rozbieg cech modelu predykcyjnego: pierwsze 130 świec każdego okna 28-dniowego (40 320)
# nie ma sygnału, więc boty handlują na 40 190 ostatnich świecach (SimulationService:
# num_ticks = min(len(okno), len(predykcje)), dane tnie się od końca). R0 liczy HODL
# na tym samym zakresie, żeby R0 == hodl_return zapisany przy botach.
HODL_WARMUP_TICKS = 130

# M3 / UL — K z shared/files/regime_lock.json (źródło prawdy).
# REGIME_K_DEFAULT tylko w testach jednostkowych bez pliku lock.
REGIME_K_DEFAULT = 3
REGIME_K_MIN = 2
REGIME_K_MAX = 5
REGIME_FIT_ROLLING_WINDOWS = 3  # refit rolling w sekwencji (jak M1)

# Meta-kontroler v6+rl / v6+rlul — identyczne w pretrain, M2 continue i eval R2/M2
V5RL_KNOB_HOLD_BARS = 1  # konfiguracja bazowa (diagnostyka hold=30 odrzucona 25.10)
V5RL_TRADE_LOCK_BARS = 15

# Hiperparametry PPO (arbitralne, bez Optuny) — źródło prawdy dla pretrainu
# learning_rate: liniowo start→end przez get_linear_fn(..., 1.0)
PPO_HPARAMS: Dict[str, Any] = {
    "n_steps": 4096,
    "batch_size": 256,
    "gamma": 0.99,
    "ent_coef": 0.015,
    "ent_coef_end": 0.001,
    "log_std_init": 0.0,  # domyslne SB3 (std=1) - konfiguracja bazowa
    "learning_rate_start": 2.0e-4,
    "learning_rate_end": 4.0e-5,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "n_epochs": 10,
    "episode_timesteps": 10080,
    "total_timesteps": 4_000_000,  # 12M diagnostyka (26.10) nie poprawila wyniku ponad szum
    # UWAGA: w v6 rl/rlul NIEAKTYWNE. Rejestr podpina v5rl_env_calculate_reward
    # jako instancyjna _calculate_reward (zaslania TradingEnv._calculate_reward).
    # Faktyczna nagroda = v5rl_env_calculate_reward. Nie opisywac w pracy jako zmiany.
    "reward_mode": "excess",
}

# Early-stop na plateau A1 sim_best (pretrain)
PPO_EARLY_STOP: Dict[str, int] = {
    "patience": 20,
    "min_dumps": 30,
}

# Stałe portfela zapisywane w artefakcie agenta (podzbiór MP_CONST)
PORTFOLIO_CONST_KEYS = (
    "stop_loss",
    "stop_loss_cooldown",
    "mincost",
    "wallet_percentage_limit",
    "max_wallet_percentage_crypto",
    "threshold_buy",
    "threshold_sell",
    "add_fee",
    "slippage",
    "mp_ema_alpha_min",
    "mp_ema_alpha_span",
)


def portfolio_const_snapshot() -> Dict[str, Any]:
    return {k: MP_CONST[k] for k in PORTFOLIO_CONST_KEYS if k in MP_CONST}


def ppo_hparams_snapshot() -> Dict[str, Any]:
    return dict(PPO_HPARAMS)


def thesis_train_config_hash() -> str:
    """SHA256 kanonicznego JSON (MP_CONST + PPO_HPARAMS + early-stop + hold/lock)."""
    try:
        from test_module.services.core_logic.version_registry import (
            V6RL_EXTRA_OBS as _xobs,
            V6RL_EXTRA_PER_SYMBOL as _xper,
            V6RL_OBS_MIN as _xmin,
            V6RL_OBS_MIN_IDX as _xidx,
        )
        extra_obs = {"V6RL_EXTRA_OBS": bool(_xobs), "per_symbol": int(_xper),
                     "obs_min": bool(_xmin), "obs_min_idx": list(_xidx)}
    except Exception:
        extra_obs = None
    blob = {
        "V6RL_EXTRA_OBS": extra_obs,
        "MP_CONST": MP_CONST,
        "PPO_HPARAMS": PPO_HPARAMS,
        "PPO_EARLY_STOP": PPO_EARLY_STOP,
        "CLAMP_WALLET_CAP_TO_1N": CLAMP_WALLET_CAP_TO_1N,
        "V5RL_KNOB_HOLD_BARS": V5RL_KNOB_HOLD_BARS,
        "V5RL_TRADE_LOCK_BARS": V5RL_TRADE_LOCK_BARS,
    }
    raw = json.dumps(blob, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def ppo_learning_rate_schedule():
    """SB3 linear schedule: learning_rate_start → learning_rate_end."""
    from stable_baselines3.common.utils import get_linear_fn

    return get_linear_fn(
        float(PPO_HPARAMS["learning_rate_start"]),
        float(PPO_HPARAMS["learning_rate_end"]),
        1.0,
    )


def build_pretrain_ppo_kwargs(
    *,
    seed: int,
    sim_mode: str = "rl",
    num_envs: Optional[int] = None,
    learning_rate=None,
    live_logging: bool = True,
    experiment_version: str = "v6",
    window_id: str = "train",
    eval_window_id: str = "A1",
    early_stop_patience: Optional[int] = None,
    early_stop_min_dumps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Kwargs pod TrainAgentPPOService.train_agent_ppo — stałe z MP_CONST / PPO_HPARAMS.
    learning_rate=None → liniowy schedule z PPO_HPARAMS (wymaga SB3).
    """
    hp = PPO_HPARAMS
    lr = learning_rate if learning_rate is not None else ppo_learning_rate_schedule()
    pat = (
        int(early_stop_patience)
        if early_stop_patience is not None
        else int(PPO_EARLY_STOP["patience"])
    )
    mind = (
        int(early_stop_min_dumps)
        if early_stop_min_dumps is not None
        else int(PPO_EARLY_STOP["min_dumps"])
    )
    kw: Dict[str, Any] = {
        "total_timesteps": int(hp["total_timesteps"]),
        "learning_rate": lr,
        "n_steps": int(hp["n_steps"]),
        "batch_size": int(hp["batch_size"]),
        "gamma": float(hp["gamma"]),
        "ent_coef": float(hp["ent_coef"]),
        "ent_coef_end": float(hp["ent_coef_end"]),
        "gae_lambda": float(hp["gae_lambda"]),
        "clip_range": float(hp["clip_range"]),
        "n_epochs": int(hp["n_epochs"]),
        "episode_timesteps": int(hp["episode_timesteps"]),
        "mp_threshold": float(MP_CONST["threshold_buy"]),
        "live_logging": bool(live_logging),
        "experiment_version": experiment_version,
        "sim_mode": sim_mode,
        "window_id": window_id,
        "eval_window_id": eval_window_id,
        "stop_loss": float(MP_CONST["stop_loss"]),
        "stop_loss_cooldown": int(MP_CONST["stop_loss_cooldown"]),
        "wallet_percentage_limit": float(MP_CONST["wallet_percentage_limit"]),
        "max_wallet_percentage_crypto": float(MP_CONST["max_wallet_percentage_crypto"]),
        "mincost": float(MP_CONST["mincost"]),
        "clamp_wallet_cap_to_1n": bool(CLAMP_WALLET_CAP_TO_1N),
        "early_stop_patience": pat,
        "early_stop_min_dumps": mind,
        "v5rl_knob_hold_bars": int(V5RL_KNOB_HOLD_BARS),
        "v5rl_trade_lock_bars": int(V5RL_TRADE_LOCK_BARS),
        "reward_mode": str(hp.get("reward_mode") or "legacy"),
        "seed": int(seed),
    }
    if num_envs is not None:
        kw["num_envs"] = int(num_envs)
    return kw


def default_mp_params(balance_start=1000.0):
    """Płaski dict pod symulację / R1 bez pliku LOCK."""
    out = {"balance_start": float(balance_start)}
    out.update(MP_CONST)
    out.update(MP_FREE_DEFAULTS)
    out["clamp_wallet_cap_to_1n"] = bool(CLAMP_WALLET_CAP_TO_1N)
    return out


def build_optuna_config(
    balance_start,
    window,
    n_symbols=2,
    *,
    max_wallet_percentage_crypto: Optional[float] = None,
    stop_loss: Optional[float] = None,
):
    """
    Config pod SimulationTunerService.run_tuning — flat 6F SHARED.
    maxc / stop_loss są CONST; opcjonalne nadpisanie (--max-wallet / --stop-loss).
    Teza: clamp_wallet_cap_to_1n=False (WalletConfig.set).
    Format: "F"/ "C" jak w plgrid_simulation_tuner.
    """
    _n_symbols(symbols=["_"] * max(int(n_symbols), 1))  # walidacja N
    cfg = {
        "balance_start": ["C", float(balance_start)],
        "window": window,
        "clamp_wallet_cap_to_1n": ["C", bool(CLAMP_WALLET_CAP_TO_1N)],
    }
    for key, val in MP_CONST.items():
        cfg[key] = ["C", val]
    if max_wallet_percentage_crypto is not None:
        cfg["max_wallet_percentage_crypto"] = [
            "C", float(max_wallet_percentage_crypto),
        ]
    if stop_loss is not None:
        cfg["stop_loss"] = ["C", float(stop_loss)]
    for key in MP_FREE:
        lo, hi = MP_FREE_RANGES[key]
        cfg[key] = ["F", lo, hi]
    return cfg


def apply_tuner_settings(tuner) -> dict:
    """Ustawia wagi celu i bramki na tunerze; zwraca snapshot do manifestu."""
    snap = {}
    for key, val in OPTUNA_OBJECTIVE.items():
        setattr(tuner, key, val)
        snap[key] = val
    for key, val in TUNER_GATES.items():
        setattr(tuner, key, val)
        snap[key] = val
    return snap


def tuner_settings_snapshot() -> dict:
    """Wartości bez mutacji obiektu — do unit testów / manifestu."""
    out = {}
    out.update(OPTUNA_OBJECTIVE)
    out.update(TUNER_GATES)
    return out
