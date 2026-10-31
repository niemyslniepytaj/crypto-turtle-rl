from .rl.base_train_service import BaseTrainService
import os
import time
from datetime import datetime, timedelta
from stable_baselines3 import PPO
import torch

from rl_module.services.thesis import console as thesis_console

class TrainAgentPPOService(BaseTrainService):
    """Trening agenta PPO (Stable-Baselines3) na bazie BaseTrainService."""

    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        super().__init__(base_dir=base_dir, agent_type='ppo')

    def train_agent_ppo(self,
                    rabbitmq_msg_id=None,
                    model_files=None,
                    data_files=None,
                    total_timesteps=2_000_000,
                    learning_rate=0.0003,
                    n_steps=2048,
                    batch_size=64,
                    gamma=0.99,
                    ent_coef=0.01,
                    ent_coef_end=None,
                    n_epochs=10,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    episode_timesteps=1024,
                    mp_threshold=0.45,
                    live_logging=True,
                    experiment_version="v0",
                    sim_mode=None,
                    window_id=None,
                    eval_window_id=None,
                    stop_loss=0.05,
                    stop_loss_cooldown=120,
                    wallet_percentage_limit=0.80,
                    max_wallet_percentage_crypto=0.70,
                    mincost=15.0,
                    clamp_wallet_cap_to_1n=True,
                    early_stop_patience=25,
                    early_stop_min_dumps=40,
                    v5rl_knob_hold_bars=1,
                    v5rl_trade_lock_bars=15,
                    reward_mode=None,
                    mp_alloc_bias=None,
                    mp_alloc_scale=None,
                    mp_confidence=None,
                    mp_urgency=None,
                    mp_signal_deadzone=None,
                    mp_rebalance_deadzone=None,
                    continue_from=None,
                    reset_num_timesteps=None,
                    num_envs=None,
                    seed=None,
                    regime_onehot_series=None,
                    regime_model_path=None,
                    regime_zeros_only=None,
                    align_callback_total=False):

        try:
            start_time = time.time()
            if self.models == {}:
                self._load_data(model_files, data_files, window_id=window_id)

            # continue-learn (M2): bez live watchera gdy brak eval_window
            if continue_from and live_logging and eval_window_id is None:
                live_logging = False

            if live_logging:
                import time as _time
                from z_maina_funkcje.plgrid_train_eval_watcher import (
                    ensure_eval_watcher_running,
                    _load_json,
                    _pid_alive,
                )
                ok = ensure_eval_watcher_running(self.base_dir)
                _time.sleep(0.8)
                pid_path = os.path.join(
                    self.base_dir, "shared", "agents", "actual_training_logs",
                    "eval_watcher.pid",
                )
                pid = int((_load_json(pid_path, {}) or {}).get("pid") or 0)
                if not ok or not _pid_alive(pid):
                    raise RuntimeError(
                        "EvalWatcher nie działa — bez watchera brak A1 snapshot / "
                        "early-stop. Uruchom: python z_maina_funkcje/plgrid_train_eval_watcher.py"
                    )

            # SB3 schedule dostaje progress_remaining: 1.0=start, 0.0=koniec
            lr_desc = learning_rate
            if callable(learning_rate):
                try:
                    lr_desc = (
                        f"{float(learning_rate(1.0)):.2e}→{float(learning_rate(0.0)):.2e}"
                    )
                except Exception:
                    lr_desc = "schedule"

            if live_logging and not eval_window_id:
                raise ValueError(
                    "przy live_logging=True ustaw eval_window_id "
                    "(okno automatycznych symulacji OOS, np. 'A1')"
                )

            ent_end = float(ent_coef if ent_coef_end is None else ent_coef_end)

            from rl_module.services.thesis.knobs import (
                PPO_HPARAMS,
                V5RL_KNOB_HOLD_BARS,
                V5RL_TRADE_LOCK_BARS,
            )
            _hold = int(
                v5rl_knob_hold_bars
                if v5rl_knob_hold_bars is not None
                else V5RL_KNOB_HOLD_BARS
            )
            _tlock = int(
                v5rl_trade_lock_bars
                if v5rl_trade_lock_bars is not None
                else V5RL_TRADE_LOCK_BARS
            )
            _reward_mode = str(
                reward_mode
                if reward_mode is not None
                else (PPO_HPARAMS.get("reward_mode") or "legacy")
            ).strip().lower() or "legacy"
            if _reward_mode not in ("legacy", "excess"):
                raise ValueError(
                    f"reward_mode musi być 'legacy' albo 'excess', dostano {_reward_mode!r}"
                )

            # seed DecisionConfig na centrum v5mp (gdy podane / defaulty champion)
            from test_module.services.core_logic.version_registry import V5MP_KNOB_CENTER
            knob_defaults = {
                "mp_alloc_bias": mp_alloc_bias if mp_alloc_bias is not None else V5MP_KNOB_CENTER[0],
                "mp_alloc_scale": mp_alloc_scale if mp_alloc_scale is not None else V5MP_KNOB_CENTER[1],
                "mp_confidence": mp_confidence if mp_confidence is not None else V5MP_KNOB_CENTER[2],
                "mp_urgency": mp_urgency if mp_urgency is not None else V5MP_KNOB_CENTER[3],
                "mp_signal_deadzone": mp_signal_deadzone if mp_signal_deadzone is not None else V5MP_KNOB_CENTER[4],
                "mp_rebalance_deadzone": mp_rebalance_deadzone if mp_rebalance_deadzone is not None else V5MP_KNOB_CENTER[5],
            }

            n_envs = int(num_envs) if num_envs is not None else 11

            from rl_module.services.rl.obs_rms_utils import resolve_obs_rms_beside_agent

            obs_rms_continue = None
            if continue_from:
                obs_rms_continue = resolve_obs_rms_beside_agent(
                    continue_from, agents_dir=self.AGENT_DIR,
                )

            # etap 1: rl → regime_k=0; rlul → K z lock + etykiety
            _mode = sim_mode or "rl"
            _zeros = regime_zeros_only
            _series = regime_onehot_series
            _rk = 0
            if _mode == "rlul":
                from rl_module.services.regime_service import require_regime_k
                _rk = int(require_regime_k(self.base_dir))
                if _zeros is None:
                    _zeros = False
                if _series is None and window_id and data_files:
                    from rl_module.services.thesis.regime_labels import build_onehot_for_window
                    _series, _rk, _meta_ul = build_onehot_for_window(
                        self.base_dir,
                        dict(data_files),
                        window_id,
                        model_path=regime_model_path,
                        refit=False,
                        seed=int(seed or 0),
                    )
                    thesis_console.print_kv(
                        "TrainPPO",
                        f"rlul etykiety window={window_id} T={_series.shape[0]} K={_rk}",
                        kind="info",
                    )
            else:
                if _zeros is None:
                    _zeros = True

            env, callback = self._setup_env(
                total_timesteps=total_timesteps,
                base_dir=self.base_dir,
                episode_timesteps=episode_timesteps,
                live_logging=live_logging,
                mp_threshold=mp_threshold,
                num_envs=n_envs,
                gamma=gamma,
                experiment_version=experiment_version,
                sim_mode=sim_mode,
                stop_loss=stop_loss,
                stop_loss_cooldown=stop_loss_cooldown,
                wallet_percentage_limit=wallet_percentage_limit,
                max_wallet_percentage_crypto=max_wallet_percentage_crypto,
                mincost=mincost,
                clamp_wallet_cap_to_1n=bool(clamp_wallet_cap_to_1n),
                v5rl_knob_hold_bars=_hold,
                v5rl_trade_lock_bars=_tlock,
                reward_mode=_reward_mode,
                mp_alloc_bias=knob_defaults["mp_alloc_bias"],
                mp_alloc_scale=knob_defaults["mp_alloc_scale"],
                mp_confidence=knob_defaults["mp_confidence"],
                mp_urgency=knob_defaults["mp_urgency"],
                mp_signal_deadzone=knob_defaults["mp_signal_deadzone"],
                mp_rebalance_deadzone=knob_defaults["mp_rebalance_deadzone"],
                regime_k=_rk,
                obs_rms_path=obs_rms_continue,
                regime_onehot_series=_series,
                regime_zeros_only=_zeros,
                train_meta={
                    "window_id": window_id,
                    "eval_window_id": eval_window_id,
                    "model_files": dict(model_files or {}),
                    "data_files": dict(data_files or {}),
                    "n_steps": n_steps,
                    "batch_size": batch_size,
                    "ent_coef": ent_coef,
                    "ent_coef_start": float(ent_coef),
                    "ent_coef_end": ent_end,
                    "gae_lambda": gae_lambda,
                    "clip_range": clip_range,
                    "n_epochs": n_epochs,
                    "mp_threshold": mp_threshold,
                    "learning_rate": lr_desc,
                    "agent_type": "ppo",
                    "experiment_version": experiment_version,
                    "sim_mode": sim_mode,
                    "stop_loss": stop_loss,
                    "stop_loss_cooldown": stop_loss_cooldown,
                    "wallet_percentage_limit": wallet_percentage_limit,
                    "max_wallet_percentage_crypto": max_wallet_percentage_crypto,
                    "mincost": mincost,
                    "early_stop_patience": int(early_stop_patience or 0),
                    "early_stop_min_dumps": int(early_stop_min_dumps or 0),
                    "v5rl_knob_hold_bars": _hold,
                    "v5rl_trade_lock_bars": _tlock,
                    "reward_mode": _reward_mode,
                    "continue_from": continue_from,
                    "obs_rms_continue": obs_rms_continue,
                    "seed": seed,
                },
            )

            policy_kwargs = dict(
                activation_fn=torch.nn.ReLU,
                net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                log_std_init=float(PPO_HPARAMS.get("log_std_init", 0.0)),
            )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            thesis_console.print_card(
                "Start treningu PPO",
                [
                    ("seed", seed),
                    ("device", device.upper()),
                    ("cuda", torch.cuda.is_available()),
                    ("timesteps", f"{int(total_timesteps):,}"),
                    ("window_id", window_id),
                    ("eval_window_id", eval_window_id),
                    ("continue_from", continue_from or "—"),
                    ("learning_rate", lr_desc),
                    ("n_steps", n_steps),
                    ("batch_size", batch_size),
                    ("ent_coef", f"{float(ent_coef)} → {ent_end}"),
                    ("experiment", f"{experiment_version}+{sim_mode}"),
                    ("reward_mode", _reward_mode),
                ],
                tag="TrainPPO",
            )

            if continue_from:
                load_path = continue_from
                if not os.path.isabs(load_path) and not os.path.isfile(load_path):
                    cand = os.path.join(self.AGENT_DIR, load_path)
                    if os.path.isfile(cand):
                        load_path = cand
                    elif not str(load_path).endswith(".zip"):
                        cand2 = os.path.join(self.AGENT_DIR, f"{load_path}.zip")
                        if os.path.isfile(cand2):
                            load_path = cand2
                if not os.path.isfile(load_path):
                    raise FileNotFoundError(f"continue_from not found: {continue_from}")
                thesis_console.print_kv(
                    "TrainPPO", f"Douczanie (continue) z {load_path}", kind="info",
                )
                agent = PPO.load(load_path, env=env, device=device)
                # Agenty pretrain (kopie sim_best) maja obs_rms.npz BEZ pola count ->
                # VecNormalize startowalby z count~0 i pierwsza paczka zastapilaby
                # statystyki pretrainu. Ustaw wage a-priori = kroki agenta.
                try:
                    import numpy as _np
                    from rl_module.services.rl.obs_rms_utils import find_vecnormalize
                    _vn = find_vecnormalize(env)
                    if _vn is not None and obs_rms_continue:
                        with _np.load(obs_rms_continue) as _d:
                            _has_count = "count" in _d.files
                        if not _has_count:
                            _vn.obs_rms.count = float(agent.num_timesteps)
                            thesis_console.print_kv(
                                "TrainPPO",
                                f"obs_rms bez count -> count={int(agent.num_timesteps)}",
                                kind="info",
                            )
                except Exception as _e:
                    print(f"[TrainPPO] obs_rms count fix skip: {_e}")
                # Agenty pretrain (kopie sim_best) maja obs_rms.npz BEZ pola count ->
                # VecNormalize startowalby z count~0 i pierwsza paczka zastapilaby
                # statystyki pretrainu. Ustaw wage a-priori = kroki agenta.
                try:
                    import numpy as _np
                    from rl_module.services.rl.obs_rms_utils import find_vecnormalize
                    _vn = find_vecnormalize(env)
                    if _vn is not None and obs_rms_continue:
                        with _np.load(obs_rms_continue) as _d:
                            _has_count = "count" in _d.files
                        if not _has_count:
                            _vn.obs_rms.count = float(agent.num_timesteps)
                            thesis_console.print_kv(
                                "TrainPPO",
                                f"obs_rms bez count -> count={int(agent.num_timesteps)}",
                                kind="info",
                            )
                except Exception as _e:
                    print(f"[TrainPPO] obs_rms count fix skip: {_e}")
                if learning_rate is not None:
                    agent.learning_rate = learning_rate
                    if not callable(learning_rate):
                        # SB3 po load() uzywa lr_schedule zapisanego w modelu;
                        # samo agent.learning_rate nic nie zmienia -> nadpisz schedule.
                        _lr_const = float(learning_rate)
                        agent.lr_schedule = lambda _progress: _lr_const
            else:
                agent = PPO(
                    policy="MlpPolicy",
                    env=env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    gamma=gamma,
                    ent_coef=float(ent_coef),
                    n_epochs=n_epochs,
                    gae_lambda=gae_lambda,
                    clip_range=clip_range,
                    policy_kwargs=policy_kwargs,
                    device=device,
                    verbose=1,
                )

            # Train — fresh: reset=True; continue (M2): reset=False
            if reset_num_timesteps is None:
                _reset_ts = continue_from is None
            else:
                _reset_ts = bool(reset_num_timesteps)
            thesis_console.print_kv(
                "TrainPPO",
                f"Nauka… reset_num_timesteps={_reset_ts}",
                kind="info",
            )
            if align_callback_total and continue_from and not _reset_ts:
                # dokończenie przerwanego runu: total_timesteps = kroki DO DODANIA;
                # callback (entropia, progress) liczy od num_timesteps -> cel absolutny
                callback.total_timesteps = int(total_timesteps) + int(agent.num_timesteps)
            elif continue_from and not _reset_ts:
                # douczanie: postep/entropia liczone od startu douczania, nie od 0 agenta
                callback.timestep_offset = int(agent.num_timesteps)
            interrupted = False
            early_stopped = False
            try:
                agent.learn(
                    total_timesteps=total_timesteps,
                    callback=callback,
                    log_interval=10000,
                    reset_num_timesteps=_reset_ts,
                )
                early_stopped = bool(getattr(callback, "_early_stopped", False))
            except KeyboardInterrupt:
                interrupted = True
                thesis_console.print_kv(
                    "TrainPPO", "Przerwano (Ctrl+C) — zapisuję model…", kind="warn",
                )

            tag = "interrupted" if interrupted else ("early_stop" if early_stopped else "final")
            try:
                from rl_module.services.thesis.knobs import (
                    portfolio_const_snapshot,
                    ppo_hparams_snapshot,
                    thesis_train_config_hash,
                )
                from rl_module.services.thesis.repro import git_commit

                agent_path, agent_file = self._save_agent(
                    agent,
                    tag=tag,
                    meta={
                        "seed": seed,
                        "experiment_version": experiment_version,
                        "sim_mode": sim_mode,
                        "window_id": window_id,
                        "eval_window_id": eval_window_id,
                        "continue_from": continue_from,
                        "total_timesteps": total_timesteps,
                        "learning_rate": lr_desc,
                        "n_steps": n_steps,
                        "batch_size": batch_size,
                        "gamma": gamma,
                        "ent_coef": float(ent_coef),
                        "ent_coef_end": ent_end,
                        "model_files": dict(model_files or {}),
                        "data_files": dict(data_files or {}),
                        "agent_type": "ppo",
                        "stop_loss": stop_loss,
                        "stop_loss_cooldown": stop_loss_cooldown,
                        "wallet_percentage_limit": wallet_percentage_limit,
                        "max_wallet_percentage_crypto": max_wallet_percentage_crypto,
                        "mincost": mincost,
                        "mp_threshold": mp_threshold,
                        "v5rl_knob_hold_bars": _hold,
                        "v5rl_trade_lock_bars": _tlock,
                        "reward_mode": _reward_mode,
                        "ppo_hparams": ppo_hparams_snapshot(),
                        "portfolio_const": portfolio_const_snapshot(),
                        "git_commit": git_commit(self.base_dir),
                        "config_hash": thesis_train_config_hash(),
                        "_train_env": env,
                    },
                )
            except Exception as save_err:
                thesis_console.print_kv(
                    "TrainPPO", f"Błąd zapisu: {save_err}", kind="err",
                )
                agent_path, agent_file = None, None

            # po early-stop: przypomnij / skopiuj sim_best jako preferowany artefakt
            if early_stopped:
                try:
                    from pathlib import Path
                    logs = Path(self.base_dir) / "shared" / "agents" / "actual_training_logs"
                    sb_meta = logs / "sim_best.json"
                    if sb_meta.is_file():
                        import json
                        with open(sb_meta, "r", encoding="utf-8") as f:
                            sb = json.load(f)
                        src = Path(self.AGENT_DIR) / (sb.get("agent_file") or "agent_ppo_workdir_sim_best.zip")
                        if src.is_file():
                            thesis_console.print_kv(
                                "TrainPPO",
                                f"early-stop → preferuj OOS best: {src.name} "
                                f"excess={sb.get('excess_pct')} (nie {agent_file})",
                                kind="best",
                            )
                except Exception as e:
                    thesis_console.print_kv(
                        "TrainPPO", f"info sim_best: {e}", kind="warn",
                    )

            try:
                env.close()
            except Exception:
                pass

            elapsed = timedelta(seconds=time.time() - start_time)
            wall_s = time.time() - start_time
            result_code = (
                "interrupted" if interrupted else ("early_stop" if early_stopped else "success")
            )
            thesis_console.print_card(
                "Koniec treningu PPO",
                [
                    ("result", result_code),
                    ("wall_time", str(elapsed)),
                    ("wall_time_s", f"{wall_s:.1f}"),
                    ("agent_file", agent_file or "—"),
                    ("agent_path", agent_path or "—"),
                    ("seed", seed),
                ],
                tag="TrainPPO",
                ok=(result_code == "success"),
            )
            if interrupted:
                return {
                    "rabbit_msg_id": rabbitmq_msg_id,
                    "result": "interrupted",
                    "message": "PPO training interrupted; model saved",
                    "agent_path": agent_path,
                    "agent_file": agent_file,
                    "seed": seed,
                    "wall_time_s": wall_s,
                    "timestamp": datetime.now().isoformat()
                }

            if early_stopped:
                return {
                    "rabbit_msg_id": rabbitmq_msg_id,
                    "result": "early_stop",
                    "message": "PPO early-stopped on A1 sim_best plateau; prefer workdir_sim_best",
                    "agent_path": agent_path,
                    "agent_file": agent_file,
                    "seed": seed,
                    "wall_time_s": wall_s,
                    "timestamp": datetime.now().isoformat()
                }

            return {
                "rabbit_msg_id": rabbitmq_msg_id,
                "result": "success",
                "message": "PPO training completed successfully",
                "agent_path": agent_path,
                "agent_file": agent_file,
                "seed": seed,
                "wall_time_s": wall_s,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            import traceback
            thesis_console.print_kv("TrainPPO", f"BŁĄD: {e}", kind="err")
            traceback.print_exc()

            return {
                "rabbit_msg_id": rabbitmq_msg_id,
                "result": "failed",
                "message": str(e),
                "agent_file": None,
                "agent_path": None,
                "timestamp": datetime.now().isoformat()
            }