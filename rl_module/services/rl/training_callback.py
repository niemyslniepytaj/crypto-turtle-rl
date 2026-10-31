import os
import sys
import json
import datetime
import numpy as np
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback


class TrainingCallback(BaseCallback):
    """Callback treningu SB3: ewaluacja, metryki epizodu, promocja sim_best."""
    def __init__(
        self,
        total_timesteps,
        num_cpu,
        base_dir,
        episode_timesteps,
        live=True,
        train_meta=None,
    ):
        super().__init__()
        self.base_dir = Path(base_dir)
        self.episode_timesteps = episode_timesteps
        self.total_timesteps = total_timesteps
        self.num_envs = num_cpu
        self.live = live
        self.train_meta = dict(train_meta or {})

        self.log_dir = self.base_dir / "shared" / "agents" / "actual_training_logs"
        self.full_log_path = self.log_dir / "full_training.json"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.progress_bar_length = 40
        self.last_progress = -1
        # douczanie: SB3 num_timesteps jest absolutny (nie od 0) - licz postep od startu
        self.timestep_offset = 0
        self.max_episodes = round(
            (np.ceil(self.total_timesteps / self.episode_timesteps) + 15) // 16 * 16, 0
        )
        self.last_dump_epoch = -1
        self.start_time = datetime.datetime.now().isoformat()
        self._best = None  # best avg_end portfolio across dumps

        # checkpoint roboczy + kolejka dla osobnego watchera symulacji
        agent_type = str(self.train_meta.get("agent_type") or "ppo")
        self.agent_type_name = agent_type
        self.workdir_agent_dir = self.base_dir / "shared" / "agents" / agent_type
        self.workdir_agent_dir.mkdir(parents=True, exist_ok=True)
        self.workdir_latest_name = f"agent_{agent_type}_workdir_latest.zip"
        self.workdir_actual_best_name = f"agent_{agent_type}_workdir_actual_best.zip"
        self.train_best_meta_path = self.log_dir / "train_actual_best.json"
        self.eval_request_path = self.log_dir / "eval_request.json"
        self.snapshot_every = int(self.train_meta.get("snapshot_every") or 10)

        # entropy anneal + early-stop na A1 (sim_best)
        self.ent_coef_start = float(self.train_meta.get("ent_coef_start")
                                    if self.train_meta.get("ent_coef_start") is not None
                                    else (self.train_meta.get("ent_coef") or 0.01))
        self.ent_coef_end = float(self.train_meta.get("ent_coef_end")
                                  if self.train_meta.get("ent_coef_end") is not None
                                  else self.ent_coef_start)
        self.early_stop_patience = int(self.train_meta.get("early_stop_patience") or 0)
        self.early_stop_min_dumps = int(self.train_meta.get("early_stop_min_dumps") or 40)
        self.sim_best_meta_path = self.log_dir / "sim_best.json"
        self.early_stop_path = self.log_dir / "early_stop.json"
        self._early_stopped = False

        if self.live:
            self._enable_windows_ansi()
            self.initialize_log_files()
            self._print_banner()

    # --- ANSI (jak Optuna) -------------------------------------------------
    @staticmethod
    def _enable_windows_ansi():
        if os.name != "nt":
            return
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass

    def _color(self, enabled=True):
        if not enabled or os.environ.get("NO_COLOR"):
            class _No:
                """Puste kody ANSI (wyłączone kolorowanie)."""
                reset = bold = dim = red = green = yellow = blue = magenta = cyan = ""
            return _No()

        class _Ansi:
            """Kody ANSI do kolorowego wyjścia konsoli."""
            reset = "\033[0m"
            bold = "\033[1m"
            dim = "\033[2m"
            red = "\033[31m"
            green = "\033[32m"
            yellow = "\033[33m"
            blue = "\033[34m"
            magenta = "\033[35m"
            cyan = "\033[36m"

        return _Ansi()

    def _print_banner(self):
        c = self._color(True)
        m = self.train_meta
        print(
            f"\n{c.cyan}{c.bold}── Train PPO{c.reset} "
            f"{c.blue}exp={m.get('experiment_version', '?')}{c.reset} "
            f"window={c.bold}{m.get('window_id') or 'full'}{c.reset} "
            f"syms={m.get('symbols', [])}\n"
            f"   {c.dim}data{c.reset}    ticks={m.get('num_ticks', '?')}  "
            f"ep={self.episode_timesteps}  envs={self.num_envs}  "
            f"total={self.total_timesteps}\n"
            f"   {c.dim}hparams{c.reset} gamma={m.get('gamma')}  "
            f"n_steps={m.get('n_steps')}  batch={m.get('batch_size')}  "
            f"ent={self.ent_coef_start}->{self.ent_coef_end}  "
            f"lr={m.get('learning_rate')}\n"
            f"   {c.dim}early{c.reset}   patience={self.early_stop_patience or 'off'}  "
            f"min_dumps={self.early_stop_min_dumps}  "
            f"hold={m.get('v5rl_knob_hold_bars', '?')}bars  "
            f"tlock={m.get('v5rl_trade_lock_bars', '?')}bars\n"
            f"   {c.dim}sync{c.reset}    episode_starts="
            f"{'ON (shared t0)' if m.get('episode_start_sync') else 'OFF (per-env random)'}\n"
            f"   {c.dim}models{c.reset} {m.get('model_files', {})}\n"
        )

    def _on_step(self) -> bool:
        _done = max(0, self.num_timesteps - self.timestep_offset)
        progress = int((_done / max(self.total_timesteps, 1)) * 10000)

        # w douczaniu rysuj pasek rzadko (co 10%) - bez spamu w konsoli
        _min_step = 1000 if self.timestep_offset > 0 else 1
        if abs(progress - self.last_progress) >= _min_step:
            self._print_progress_bar(progress)
            self.last_progress = progress

        # dump częściej przy długich epizodach: co ~episode/4 vec-stepów, min 512
        dump_every = max(512, self.episode_timesteps // 4)
        if self.live and self.n_calls % dump_every == 0:
            self._dump_logs()

        if self._early_stopped:
            return False
        if self.live and self.early_stop_patience > 0 and self.n_calls % dump_every == 0:
            if self._check_early_stop():
                return False

        return True

    def _on_rollout_end(self) -> None:
        """Liniowy anneal ent_coef (start→end) po progress timesteps."""
        if self.model is None:
            return
        try:
            progress = float(max(0, self.num_timesteps - self.timestep_offset)) / float(
                max(self.total_timesteps, 1)
            )
            progress = max(0.0, min(1.0, progress))
            ent = self.ent_coef_start + (self.ent_coef_end - self.ent_coef_start) * progress
            self.model.ent_coef = float(ent)
        except Exception:
            pass

    def _check_early_stop(self) -> bool:
        """
        Stop gdy A1 sim_best nie poprawia się przez `early_stop_patience` dumpów.
        Wymaga sim_best.json z TEJ sesji (train_start_time).
        """
        if self._early_stopped:
            return True
        sb = self._load_json(self.sim_best_meta_path, None)
        if not sb:
            return False
        # ignoruj leftover z poprzedniego runu
        sb_start = sb.get("train_start_time")
        if sb_start and sb_start != self.start_time:
            return False
        if not sb_start:
            # stary format bez pola — nie early-stopuj na cudzym best
            return False
        full = self._load_json(self.full_log_path, {})
        hist_len = len(full.get("history") or [])
        if hist_len < self.early_stop_min_dumps:
            return False
        try:
            best_dump = int(sb.get("global_dump") or 0)
        except (TypeError, ValueError):
            best_dump = 0
        stalled = hist_len - best_dump
        if stalled < self.early_stop_patience:
            return False

        c = self._color(True)
        reason = {
            "reason": "sim_best_plateau",
            "hist_len": hist_len,
            "sim_best_global_dump": best_dump,
            "stalled_dumps": stalled,
            "patience": self.early_stop_patience,
            "excess_pct": sb.get("excess_pct"),
            "epoch_sum": sb.get("epoch_sum"),
            "agent_file": sb.get("agent_file"),
            "num_timesteps": int(self.num_timesteps),
            "stopped_at": datetime.datetime.now().isoformat(),
        }
        self._save_json(self.early_stop_path, reason)
        self._early_stopped = True
        print(
            f"\n{c.yellow}{c.bold}[early-stop]{c.reset} A1 sim_best plateau "
            f"stalled={stalled}≥{self.early_stop_patience} dumps "
            f"(best excess={sb.get('excess_pct')} es={sb.get('epoch_sum')}) "
            f"— kończę learn; bierz {sb.get('agent_file')}"
        )
        return True

    def _print_progress_bar(self, progress: int):
        c = self._color(True)
        mean_reward = 0.0
        last_batch_avg = 0.0
        if self.model and getattr(self.model, "ep_info_buffer", None):
            rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            if rewards:
                mean_reward = sum(rewards) / len(rewards)
                last_n = rewards[-self.num_envs:]
                if last_n:
                    last_batch_avg = sum(last_n) / len(last_n)

        pct = progress / 100
        filled = int(self.progress_bar_length * progress / 10000)
        bar = "█" * filled + "░" * (self.progress_bar_length - filled)
        r_col = c.green if mean_reward >= 0 else c.red

        sys.stdout.write(
            f"\r{c.cyan}PPO{c.reset}×{self.num_envs} [{c.bold}{bar}{c.reset}] "
            f"{c.bold}{pct:5.1f}%{c.reset}  "
            f"R̄={r_col}{mean_reward:+.1f}{c.reset}  "
            f"batch={last_batch_avg:+.1f}  "
            f"t={max(0, self.num_timesteps - self.timestep_offset)}/{self.total_timesteps}"
        )
        sys.stdout.flush()

    def _print_epoch_card(self, entry):
        """Kolorowa karta po dumpie (jak Optuna trial card)."""
        c = self._color(True)
        avg_p = entry["portfolio"]["avg_end_p"]
        avg_r = entry["reward"]["avg_end_r"]
        tx = entry["avg_transactions"]
        tx_day = entry.get("avg_tx_per_day", 0.0)
        excess = entry.get("avg_excess_pct", 0.0)
        hodl = entry.get("avg_hodl_pct", 0.0)
        hodl_min = entry.get("min_hodl_pct", hodl)
        hodl_max = entry.get("max_hodl_pct", hodl)
        hodl_envs = entry.get("hodl_pct_per_env") or []
        strat = entry.get("avg_strat_pct", 0.0)
        dd = entry.get("avg_max_dd_pct", 0.0)
        closed = entry.get("avg_closed", 0.0)
        knobs = entry.get("knobs") or {}

        is_best = self._best is None or avg_p > self._best.get("avg_p", float("-inf"))
        if is_best:
            self._best = {"avg_p": avg_p, "excess": excess, "epoch_sum": entry["epoch_sum"]}
            best_txt = f" {c.yellow}{c.bold}★ NEW BEST${c.reset}"
        else:
            b = self._best
            best_txt = (
                f" {c.dim}| best ${b['avg_p']:.0f} excess={b['excess']:+.2f}% "
                f"(es={b['epoch_sum']}){c.reset}"
            )

        def _signed(v, nd=2, good_positive=True):
            try:
                x = float(v)
            except (TypeError, ValueError):
                return str(v)
            ok = (x >= 0) if good_positive else (x <= 0)
            col = c.green if ok else c.red
            return f"{col}{x:+.{nd}f}{c.reset}"

        tx_col = c.green if tx_day <= 6 else (c.yellow if tx_day <= 20 else c.red)
        exp = self.train_meta.get("experiment_version", "?")
        win = self.train_meta.get("window_id") or "full"

        knobs_txt = ""
        if knobs:
            knobs_txt = (
                f"\n   {c.dim}knobs{c.reset}  "
                f"bias={knobs.get('mp_alloc_bias', float('nan')):.3f} "
                f"scale={knobs.get('mp_alloc_scale', float('nan')):.3f} "
                f"conf={knobs.get('mp_confidence', float('nan')):.3f} "
                f"urg={knobs.get('mp_urgency', float('nan')):.3f} "
                f"sig_dz={knobs.get('mp_signal_deadzone', float('nan')):.3f} "
                f"reb_dz={knobs.get('mp_rebalance_deadzone', float('nan')):.3f}"
            )

        hodl_envs_txt = ""
        if hodl_envs:
            parts = ",".join(f"{v:+.4f}" for v in hodl_envs)
            hodl_envs_txt = f"\n   {c.dim}hodl_envs{c.reset} [{parts}]"

        # zejdź z progress-line
        sys.stdout.write("\n")
        print(
            f"{c.cyan}{c.bold}── Epoch dump{c.reset} es={entry['epoch_sum']} "
            f"{c.blue}exp={exp}{c.reset} window={win}{best_txt}\n"
            f"   {c.dim}result{c.reset}  {c.bold}${avg_p:.0f}{c.reset}  "
            f"ret={_signed(strat)}%  "
            f"hodl={hodl:+.4f}% (min={float(hodl_min):+.4f} max={float(hodl_max):+.4f})  "
            f"excess={_signed(excess)}%  "
            f"dd={_signed(-abs(dd), good_positive=False)}%\n"
            f"   {c.dim}act{c.reset}     {tx_col}tx/ep={tx:.0f}  tx/d={tx_day:.2f}{c.reset}  "
            f"closed={closed:.1f}  R̄_end={_signed(avg_r, 1)}"
            f"{hodl_envs_txt}"
            f"{knobs_txt}\n"
        )

    def _dump_logs(self):
        try:
            prev_epochs = self.training_env.get_attr("prev_epoch")
        except AttributeError:
            return

        valid_epochs = [p for p in prev_epochs if p and p.epoch > 0]
        if not valid_epochs:
            return

        for i, epoch_data in enumerate(valid_epochs):
            if hasattr(epoch_data, "to_dict"):
                data = epoch_data.to_dict()
                data["process_id"] = i
                self._save_json(self.log_dir / f"proc_{i}.json", data)

        current_epoch_sum = sum(p.epoch for p in valid_epochs)
        if current_epoch_sum <= self.last_dump_epoch:
            return
        self.last_dump_epoch = current_epoch_sum
        self._update_full_log(valid_epochs, current_epoch_sum)

    def _update_full_log(self, valid_epochs, epoch_sum):
        num_procs = len(valid_epochs)

        def aggregate_stats(attr_suffix):
            return {
                f"max_max_{attr_suffix}": max(getattr(p, f"max_{attr_suffix}") for p in valid_epochs),
                f"min_min_{attr_suffix}": min(getattr(p, f"min_{attr_suffix}") for p in valid_epochs),
                f"max_mean_{attr_suffix}": max(getattr(p, f"mean_{attr_suffix}") for p in valid_epochs),
                f"min_mean_{attr_suffix}": min(getattr(p, f"mean_{attr_suffix}") for p in valid_epochs),
                f"avg_end_{attr_suffix}": sum(getattr(p, f"end_{attr_suffix}") for p in valid_epochs) / num_procs,
                f"min_end_{attr_suffix}": min(getattr(p, f"end_{attr_suffix}") for p in valid_epochs),
                f"max_end_{attr_suffix}": max(getattr(p, f"end_{attr_suffix}") for p in valid_epochs),
            }

        def _avg(attr, default=0.0):
            vals = [getattr(p, attr, default) for p in valid_epochs]
            return float(sum(vals) / num_procs) if vals else default

        knobs_acc = {}
        knobs_n = 0
        for p in valid_epochs:
            kd = getattr(p, "knobs", None) or {}
            if not kd:
                continue
            knobs_n += 1
            for k, v in kd.items():
                knobs_acc[k] = knobs_acc.get(k, 0.0) + float(v)
        knobs_mean = {k: knobs_acc[k] / knobs_n for k in knobs_acc} if knobs_n else {}

        new_entry = {
            "epoch_sum": epoch_sum,
            "avg_transactions": sum(p.buys + p.sells for p in valid_epochs) / num_procs,
            "avg_buys": sum(p.buys for p in valid_epochs) / num_procs,
            "avg_sells": sum(p.sells for p in valid_epochs) / num_procs,
            "avg_tx_per_day": _avg("tx_per_day"),
            "avg_closed": _avg("closed_trades"),
            "avg_strat_pct": _avg("strat_return_pct"),
            "avg_hodl_pct": _avg("hodl_return_pct"),
            "min_hodl_pct": min(float(getattr(p, "hodl_return_pct", 0.0)) for p in valid_epochs),
            "max_hodl_pct": max(float(getattr(p, "hodl_return_pct", 0.0)) for p in valid_epochs),
            "hodl_pct_per_env": [
                round(float(getattr(p, "hodl_return_pct", 0.0)), 6) for p in valid_epochs
            ],
            "avg_excess_pct": _avg("excess_pct"),
            "avg_max_dd_pct": _avg("max_dd_pct"),
            "avg_episode_bars": _avg("episode_bars"),
            "knobs": knobs_mean,
            "portfolio": aggregate_stats("p"),
            "reward": aggregate_stats("r"),
        }

        full_data = self._load_json(self.full_log_path, default={"config": {}, "history": []})
        full_data["config"] = {
            "episode_length": self.episode_timesteps,
            "max_episodes": self.max_episodes,
            "num_processes": self.num_envs,
            "global_epochs": round(self.max_episodes / max(self.num_envs, 1), 0),
            "start_time": self.start_time,
            "total_timesteps": self.total_timesteps,
            **self.train_meta,
        }

        if "history" not in full_data:
            full_data["history"] = []
        full_data["history"].append(new_entry)
        self._save_json(self.full_log_path, full_data)
        self._print_epoch_card(new_entry)
        self._save_workdir_checkpoint(epoch_sum, new_entry)

    def _find_vecnormalize(self):
        env = self.training_env
        seen = set()
        while env is not None and id(env) not in seen:
            seen.add(id(env))
            cls = type(env).__name__
            if cls == "VecNormalize" or hasattr(env, "obs_rms"):
                return env
            env = getattr(env, "venv", None) or getattr(env, "env", None)
        return None

    def _save_workdir_checkpoint(self, epoch_sum, entry):
        """
        Zapis agenta:
          - always: workdir_latest (+ obs_rms) + eval_request
          - co snapshot_every (domyślnie 10) dumpów: workdir_es{N}.zip (trzymane)
          - gdy poprawa excess/portfela treningowego: workdir_actual_best.zip
        Sym-best aktualizuje osobny watcher po OOS.
        """
        if self.model is None:
            return
        import shutil
        c = self._color(True)
        try:
            latest_path = self.workdir_agent_dir / self.workdir_latest_name
            partial_stem = self.workdir_agent_dir / "_workdir_partial"
            for stale in (
                Path(str(partial_stem) + ".zip"),
                partial_stem,
            ):
                if stale.exists():
                    stale.unlink()
            self.model.save(str(partial_stem))
            produced = Path(str(partial_stem) + ".zip")
            if not produced.exists():
                raise FileNotFoundError(f"Brak zapisu SB3: {produced}")
            os.replace(produced, latest_path)

            obs_stats_name = "agent_workdir_latest_obs_rms.npz"
            obs_stats_path = self.workdir_agent_dir / obs_stats_name
            vn = self._find_vecnormalize()
            if vn is not None and getattr(vn, "obs_rms", None) is not None:
                np.savez(
                    obs_stats_path,
                    mean=np.asarray(vn.obs_rms.mean, dtype=np.float64),
                    var=np.asarray(vn.obs_rms.var, dtype=np.float64),
                    clip_obs=float(getattr(vn, "clip_obs", 10.0) or 10.0),
                    epsilon=float(getattr(vn, "epsilon", 1e-8) or 1e-8),
                )
            else:
                obs_stats_path = None

            hist_len = len(self._load_json(self.full_log_path, {}).get("history", []))
            notes = [f"latest es={epoch_sum}"]

            # snapshot co N dumpów (global epoch na dashboardzie)
            es_name = None
            if hist_len > 0 and hist_len % max(1, self.snapshot_every) == 0:
                es_name = f"agent_{self.agent_type_name}_workdir_es{int(epoch_sum)}.zip"
                shutil.copy2(latest_path, self.workdir_agent_dir / es_name)
                if obs_stats_path and obs_stats_path.exists():
                    shutil.copy2(
                        obs_stats_path,
                        self.workdir_agent_dir / f"agent_workdir_es{int(epoch_sum)}_obs_rms.npz",
                    )
                notes.append(f"snapshot/{self.snapshot_every}")

            # najlepszy względem treningu (excess, fallback kapitał)
            train_score = entry.get("avg_excess_pct")
            if train_score is None:
                train_score = entry.get("portfolio", {}).get("avg_end_p")
            try:
                train_score = float(train_score)
            except (TypeError, ValueError):
                train_score = float("-inf")

            best_meta = self._load_json(self.train_best_meta_path, {})
            prev_best = best_meta.get("score")
            try:
                prev_best = float(prev_best) if prev_best is not None else float("-inf")
            except (TypeError, ValueError):
                prev_best = float("-inf")

            if train_score > prev_best:
                best_path = self.workdir_agent_dir / self.workdir_actual_best_name
                shutil.copy2(latest_path, best_path)
                if obs_stats_path and obs_stats_path.exists():
                    shutil.copy2(
                        obs_stats_path,
                        self.workdir_agent_dir / "agent_workdir_actual_best_obs_rms.npz",
                    )
                self._save_json(self.train_best_meta_path, {
                    "score": train_score,
                    "score_key": "avg_excess_pct",
                    "epoch_sum": int(epoch_sum),
                    "global_dump": hist_len,
                    "avg_end_p": entry.get("portfolio", {}).get("avg_end_p"),
                    "avg_excess_pct": entry.get("avg_excess_pct"),
                    "avg_tx_per_day": entry.get("avg_tx_per_day"),
                    "agent_file": self.workdir_actual_best_name,
                    "updated_at": datetime.datetime.now().isoformat(),
                })
                notes.append(f"{c.yellow}★ actual_best{c.reset} score={train_score:+.3f}")

            request = {
                "epoch_sum": int(epoch_sum),
                "global_dump": hist_len,
                "agent_file": self.workdir_latest_name,
                "agent_es_file": es_name,
                "obs_rms_file": obs_stats_name if obs_stats_path else None,
                "train_meta": self.train_meta,
                "train_start_time": self.start_time,
                "metrics_snapshot": {
                    "avg_end_p": entry.get("portfolio", {}).get("avg_end_p"),
                    "avg_excess_pct": entry.get("avg_excess_pct"),
                    "avg_tx_per_day": entry.get("avg_tx_per_day"),
                },
                "created_at": datetime.datetime.now().isoformat(),
            }
            req_tmp = self.eval_request_path.with_suffix(".tmp")
            self._save_json(req_tmp, request)
            os.replace(req_tmp, self.eval_request_path)

            print(f"{c.dim}[ckpt]{c.reset} " + " | ".join(notes) + " | eval_request")
        except Exception as e:
            print(f"{c.red}[ckpt] zapis workdir nieudany: {e}{c.reset}")

    def initialize_log_files(self):
        initial_data = {
            "config": {
                "episode_length": self.episode_timesteps,
                "max_episodes": self.max_episodes,
                "num_processes": self.num_envs,
                "global_epochs": round(self.max_episodes / max(self.num_envs, 1), 0),
                "start_time": self.start_time,
                "total_timesteps": self.total_timesteps,
                **self.train_meta,
            },
            "history": [],
        }
        self._save_json(self.full_log_path, initial_data)

        # świeża sesja: zarchiwizuj stary sim_best, wyczyść early-stop / train_best
        # żeby nie early-stopować na cudzym +3.8% i nie blokować promocji OOS
        try:
            if self.sim_best_meta_path.exists():
                prev = self._load_json(self.sim_best_meta_path, {})
                if prev:
                    arch = self.log_dir / f"sim_best_prev_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    self._save_json(arch, prev)
                self.sim_best_meta_path.unlink(missing_ok=True)
            if self.early_stop_path.exists():
                self.early_stop_path.unlink(missing_ok=True)
            if self.train_best_meta_path.exists():
                self.train_best_meta_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[Init] ostrzeżenie przy resecie sesji best: {e}")

        initial_proc_data = {
            "epoch": 0,
            "metrics": {
                "buys": 0, "sells": 0, "holds": 0, "transactions": 0,
                "closed_trades": 0, "tx_per_day": 0.0,
                "strat_return_pct": 0.0, "hodl_return_pct": 0.0,
                "excess_pct": 0.0, "max_dd_pct": 0.0, "knobs": {},
            },
            "portfolio": {"portfolio_values": [], "min_p": 0.0, "mean_p": 0.0, "max_p": 0.0},
            "reward": {"reward_values": [], "min_r": 0.0, "mean_r": 0.0, "max_r": 0.0},
        }

        for i in range(self.num_envs):
            proc_data = json.loads(json.dumps(initial_proc_data))
            proc_data["process_id"] = i
            self._save_json(self.log_dir / f"proc_{i}.json", proc_data)

        c = self._color(True)
        print(f"{c.dim}[Init]{c.reset} logi dla {self.num_envs} procesow -> {self.log_dir}")
        print(f"{c.dim}[Init]{c.reset} swieza sesja (sim_best wyczyszczony, start={self.start_time})")

    @staticmethod
    def _save_json(path, data):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"[Callback Error] Nie udało się zapisać {path.name}: {e}")

    @staticmethod
    def _load_json(path, default=None):
        if not path.exists():
            return default if default is not None else {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Callback Warning] Błąd odczytu {path.name}: {e}, tworzę nowy.")
            return default if default is not None else {}
