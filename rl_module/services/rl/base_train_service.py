import os
import pandas as pd
import multiprocessing
from datetime import datetime
from abc import ABC
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from test_module.services.core_logic.mp_signal import ModelSignal
from test_module.services.core_logic.structures import CoreConfig
from test_module.services.core_logic.windows import load_windows, slice_windows
from .agent import TradingEnv
from .training_callback import TrainingCallback

class BaseTrainService(ABC):
    """Wspólny pipeline treningu SB3: dane, VecEnv, callbacki, zapis agenta i obs_rms."""

    def __init__(self, base_dir=None, agent_type='ppo'):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.base_dir = base_dir
        self.agent_type = agent_type

        # te same CSV / okna co SimulationService
        self.DATA_DIR = os.path.join(base_dir, "shared", "files", "processed_simulation_data")
        self.AGENT_DIR = os.path.join(base_dir, "shared", "agents", agent_type)
        os.makedirs(self.AGENT_DIR, exist_ok=True)

        self.models = {}
        self.symbols = []
        self.num_ticks = 0


    def _load_data(self, model_files, data_files, window_id=None):
        """Ładuje CSV pełnego okresu i opcjonalnie tnie do okna (jak symulacja/Optuna)."""

        print(f"[TrainAgent{self.agent_type.upper()}] Przygotowywanie modeli i danych....")
        windows = load_windows(self.base_dir) if window_id else None

        for symbol, filename in data_files.items():
            filepath = os.path.join(self.DATA_DIR, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Brak pliku: {filepath}")

            df = pd.read_csv(filepath)
            df.dropna(inplace=True)
            if window_id:
                df = slice_windows(df, windows, window_id)
                print(f"[TrainAgent{self.agent_type.upper()}] window={window_id} {symbol}: {len(df)} rows")
                if len(df) == 0:
                    raise ValueError(f"Puste okno {window_id} dla {symbol}")

            len_df = len(df)
            if self.num_ticks == 0:
                self.num_ticks = len_df
            else:
                self.num_ticks = min(self.num_ticks, len_df)

            self.models[symbol] = ModelSignal(
                model_name=model_files[symbol],
                data=df.copy(),
                base_dir=self.base_dir
            )
            self.symbols.append(symbol)

            len_pred = len(self.models[symbol].predictions)
            self.num_ticks = min(self.num_ticks, len_pred)

        for symbol in self.symbols:
            self.models[symbol].process_models_data(self.num_ticks)

        print(
            f"[TrainAgent{self.agent_type.upper()}] Załadowano {len(self.symbols)} kryptowalut, "
            f"{self.num_ticks} timestamps (window={window_id or 'full'})."
        )
        return True

    def _setup_env(self,
                   total_timesteps,
                   base_dir,
                   live_logging,
                   episode_timesteps,
                   mp_threshold,
                   num_envs=None,
                   normalize=True,
                   gamma=0.99,
                   experiment_version="v0",
                   sim_mode=None,
                   train_meta=None,
                   stop_loss=0.0,
                   stop_loss_cooldown=120,
                   wallet_percentage_limit=0.9,
                   max_wallet_percentage_crypto=1.0,
                   mincost=1.0,
                   clamp_wallet_cap_to_1n=True,
                   v5rl_knob_hold_bars=1,
                   v5rl_trade_lock_bars=15,
                   mp_alloc_bias=0.5,
                   mp_alloc_scale=0.5,
                   mp_confidence=0.5,
                   mp_urgency=0.5,
                   mp_signal_deadzone=0.15,
                   mp_rebalance_deadzone=0.25,
                   reward_mode="legacy",
                   regime_k=None,
                   obs_rms_path=None,
                   regime_onehot_series=None,
                   regime_zeros_only=None):
        """Buduje VecEnv treningowe (opcjonalnie VecNormalize + kanał reżimu) oraz callback."""
        from test_module.services.core_logic.structures import coerce_experiment_pair
        from rl_module.services.rl.obs_rms_utils import load_obs_rms_into_vecnormalize
        from rl_module.services.rl.regime_obs_wrapper import ConcatRegimeVecEnv
        from rl_module.services.regime_service import require_regime_k

        if num_envs is None:
            num_envs = max(1, multiprocessing.cpu_count() - 5)

        print(f"[TrainAgent{self.agent_type.upper()}] Uruchamianie na {num_envs} równoległych środowiskach...")

        # v6: trening meta = sim_mode=rl; legacy v5rl/mprl → to samo
        ver, mode = coerce_experiment_pair(experiment_version, sim_mode)
        if ver == "v6" and mode == "mp":
            # train_rl bez jawnego mode → meta (dawne v5rl / mprl)
            if sim_mode is None and str(experiment_version) in ("v5rl", "v6"):
                mode = "rl"

        core_config = CoreConfig(
            sim_mode=mode or "",
            balance_start=1000.0,
            symbols=self.symbols,
            threshold_buy=mp_threshold,
            threshold_sell=mp_threshold,
            experiment_version=ver,
            stop_loss=float(stop_loss or 0.0),
            stop_loss_cooldown=int(stop_loss_cooldown or 120),
            wallet_percentage_limit=float(wallet_percentage_limit),
            max_wallet_percentage_crypto=float(max_wallet_percentage_crypto),
            mincost=float(mincost),
            clamp_wallet_cap_to_1n=bool(clamp_wallet_cap_to_1n),
            mp_alloc_bias=float(mp_alloc_bias),
            mp_alloc_scale=float(mp_alloc_scale),
            mp_confidence=float(mp_confidence),
            mp_urgency=float(mp_urgency),
            mp_signal_deadzone=float(mp_signal_deadzone),
            mp_rebalance_deadzone=float(mp_rebalance_deadzone),
            v5rl_knob_hold_bars=int(v5rl_knob_hold_bars if v5rl_knob_hold_bars is not None else 1),
            v5rl_trade_lock_bars=int(v5rl_trade_lock_bars if v5rl_trade_lock_bars is not None else 15),
            reward_mode=str(reward_mode or "legacy").strip().lower() or "legacy",
        )
        print(
            f"[TrainAgent{self.agent_type.upper()}] "
            f"experiment={ver}  sim_mode={core_config.sim_mode or '—'}  "
            f"reward_mode={core_config.reward_mode}"
        )

        # Sync startów: deterministyczny t0 z (salt + epoch_counter) — bez shared
        # memory (spawn na Windows psuł Value/Lock → każde env powtarzało to samo
        # okno przez num_envs epizodów). Powrót do niezależnych startów:
        # episode_sync["enabled"] = False.
        episode_sync = {
            "enabled": True,
            "salt": int(datetime.now().timestamp()) & 0x7FFFFFFF,
            "num_envs": int(num_envs),
        }

        def _env_factory():
            return TradingEnv(
                models=self.models,
                num_ticks=self.num_ticks,
                base_dir=self.base_dir,
                core_config=core_config,
                episode_timesteps=episode_timesteps,
                agent_type=self.agent_type,
                episode_sync=episode_sync,
            )

        env = make_vec_env(
            _env_factory,
            n_envs=num_envs,
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={"start_method": "spawn" if os.name == "nt" else "fork"},  # ← dodaj to
            wrapper_class=Monitor
        )
        torch.set_num_threads(num_envs)

        if normalize:
            env = VecNormalize(
                env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=5.0,
                clip_reward=5.0,
                gamma=gamma
            )
            if obs_rms_path:
                ok = load_obs_rms_into_vecnormalize(env, obs_rms_path)
                print(
                    f"[TrainAgent{self.agent_type.upper()}] "
                    f"obs_rms continue={'OK' if ok else 'BRAK'} ({obs_rms_path})"
                )

        # Kanał reżimu PO VecNormalize — TYLKO rlul (etap 1: rl = obs oryginalna, K=0)
        _rk = 0 if regime_k is None else int(regime_k)
        if regime_k is None and (mode or "") == "rlul":
            _rk = int(require_regime_k(self.base_dir))
        zeros_only = regime_zeros_only
        if zeros_only is None:
            zeros_only = (mode or "") != "rlul"
        if _rk > 0 and (mode or "") == "rlul":
            series = None if zeros_only else regime_onehot_series
            env = ConcatRegimeVecEnv(
                env,
                regime_k=int(_rk),
                regime_onehot_series=series,
                zeros_only=bool(zeros_only),
            )
            print(
                f"[TrainAgent{self.agent_type.upper()}] "
                f"regime_k={int(_rk)} zeros_only={bool(zeros_only)} "
                f"series={'tak' if series is not None else 'nie'}"
            )
        elif (mode or "") in ("rl", "mprl", ""):
            print(
                f"[TrainAgent{self.agent_type.upper()}] "
                f"regime_k=0 (tryb rl — bez kanału UL)"
            )

        meta = {
            "experiment_version": experiment_version,
            "symbols": list(self.symbols),
            "num_ticks": int(self.num_ticks),
            "gamma": gamma,
            "episode_start_sync": bool(episode_sync["enabled"]),
            "regime_k": int(_rk) if _rk else 0,
            **(train_meta or {}),
        }
        callback = TrainingCallback(
            total_timesteps=total_timesteps,
            num_cpu=env.num_envs,
            base_dir=base_dir,
            episode_timesteps=episode_timesteps,
            live=live_logging,
            train_meta=meta,
        )
        return env, callback

    def _save_agent(self, agent, tag="", meta=None):
        """Zapis agenta + sidecar JSON (seed, train_meta). tag np. 'interrupted'."""
        import json

        from rl_module.services.thesis_artifacts import agent_meta_path, make_agent_artifact

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tag_part = f"_{tag}" if tag else ""
        meta = dict(meta or {})
        seed = meta.get("seed")
        seed_part = f"_seed{int(seed)}" if seed is not None else ""
        mode = str(meta.get("sim_mode") or "")
        mode_part = "_rlul" if mode == "rlul" else ""
        filename = f"agent_{self.agent_type}{mode_part}{seed_part}{tag_part}_{timestamp}.zip"
        path = os.path.join(self.AGENT_DIR, filename)

        agent.save(path)
        print(f"TrainAgent[{self.agent_type.upper()}] Agent zapisany do: {path}")

        # obs_rms obok zip (I0.3) — z env treningowego jeśli podpięty
        try:
            from rl_module.services.rl.obs_rms_utils import (
                obs_rms_path_for_agent,
                save_vecnormalize_obs_rms,
            )
            env_ref = meta.get("_train_env")
            if env_ref is not None:
                rms_path = obs_rms_path_for_agent(path)
                saved = save_vecnormalize_obs_rms(env_ref, rms_path)
                if saved:
                    meta["obs_rms_file"] = os.path.basename(saved)
                    print(f"TrainAgent[{self.agent_type.upper()}] obs_rms: {saved}")
        except Exception as e:
            print(f"TrainAgent[{self.agent_type.upper()}] obs_rms skip: {e}")

        try:
            meta_for_json = {k: v for k, v in meta.items() if not str(k).startswith("_")}
            # Pola tezy na top-level artefaktu (obok seed / window) — czytelne dla manifestu
            top_keys = (
                "ppo_hparams", "portfolio_const", "git_commit", "config_hash",
                "v5rl_knob_hold_bars", "v5rl_trade_lock_bars",
            )
            extra = {k: meta_for_json[k] for k in top_keys if k in meta_for_json}
            artifact = make_agent_artifact(
                seed=int(seed if seed is not None else -1),
                agent_file=filename,
                agent_path=path,
                experiment_version=meta.get("experiment_version"),
                sim_mode=meta.get("sim_mode"),
                window_id=meta.get("window_id"),
                eval_window_id=meta.get("eval_window_id"),
                continue_from=meta.get("continue_from"),
                tag=tag or None,
                train_meta=meta_for_json,
                extra=extra or None,
            )
            if seed is None:
                artifact["seed"] = None
            meta_path = agent_meta_path(path)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(artifact, f, indent=2)
            print(f"TrainAgent[{self.agent_type.upper()}] Meta: {meta_path}")
        except Exception as e:
            print(f"TrainAgent[{self.agent_type.upper()}] Meta sidecar skip: {e}")

        return path, filename