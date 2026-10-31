"""
Osobny entry: trening / pretrain PPO (teza).
Hiperparametry z PPO_HPARAMS, stałe portfela z MP_CONST — bez ppo_hparams_lock / Optuny.

Uwaga: actual_training_logs + workdir_* + jeden EvalWatcher → NIE uruchamiaj
równolegle dwóch treningów (blokada train.lock). --all jest sekwencyjne.

Przykłady:
  python z_maina_funkcje/plgrid_train_agent_ppo.py --seed 396379
  python z_maina_funkcje/plgrid_train_agent_ppo.py --all
  python z_maina_funkcje/plgrid_train_agent_ppo.py --seed 951830 --sim-mode rlul
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rl_module.services.rl.obs_rms_utils import obs_rms_path_for_agent
from rl_module.services.thesis.assets import THESIS_DATA_FILES, THESIS_MODEL_PATHS, THESIS_SEEDS
from rl_module.services.thesis.knobs import (
    build_pretrain_ppo_kwargs,
    thesis_train_config_hash,
)
from rl_module.services.thesis.repro import git_commit
from rl_module.services.train_agent_ppo_service import TrainAgentPPOService

LOGS_DIR = os.path.join(BASE_DIR, "shared", "agents", "actual_training_logs")
TRAIN_LOCK_PATH = os.path.join(LOGS_DIR, "train.lock")
AGENT_DIR = os.path.join(BASE_DIR, "shared", "agents", "ppo")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Pretrain PPO (teza) — bez Optuny hparams")
    parser.add_argument(
        "--seed", type=int, action="append", default=None,
        help=f"Ziarno (można powtórzyć). Domyślnie: THESIS_SEEDS[0]={THESIS_SEEDS[0]}",
    )
    parser.add_argument(
        "--all", action="store_true",
        help=f"Pętla po THESIS_SEEDS {list(THESIS_SEEDS)} (sekwencyjnie, 1 proces)",
    )
    parser.add_argument(
        "--sim-mode", choices=("rl", "rlul"), default="rl",
        help="rl = R2/M2 (regime_k=0); rlul = R3/M3",
    )
    parser.add_argument(
        "--num-envs", type=int, default=None,
        help="Liczba env (domyślnie: bez zmian = serwis 11)",
    )
    return parser.parse_args(argv)


def _resolve_seeds(args) -> list:
    if args.all:
        return list(THESIS_SEEDS)
    if args.seed:
        return list(args.seed)
    return [THESIS_SEEDS[0]]


def _pid_alive(pid: int) -> bool:
    if pid is None or int(pid) <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes
            k = ctypes.windll.kernel32
            SYNCHRONIZE = 0x00100000
            h = k.OpenProcess(SYNCHRONIZE, False, int(pid))
            if h:
                k.CloseHandle(h)
                return True
            return False
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def acquire_train_lock(seed: int, lock_path: str = TRAIN_LOCK_PATH) -> None:
    """Zakłada train.lock; rzuca RuntimeError, gdy aktywny jest inny proces treningu."""
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    if os.path.isfile(lock_path):
        try:
            with open(lock_path, encoding="utf-8") as f:
                prev = json.load(f)
        except Exception:
            prev = {}
        prev_pid = int(prev.get("pid") or 0)
        if _pid_alive(prev_pid):
            raise RuntimeError(
                f"trening już trwa (pid={prev_pid}, ziarno={prev.get('seed')}) "
                f"— wspólne actual_training_logs/workdir; poczekaj lub usuń {lock_path}"
            )
        # stale lock
        try:
            os.remove(lock_path)
        except OSError:
            pass
    payload = {
        "pid": os.getpid(),
        "seed": int(seed),
        "acquired_at": datetime.now().isoformat(),
    }
    tmp = lock_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, lock_path)


def release_train_lock(lock_path: str = TRAIN_LOCK_PATH) -> None:
    """Usuwa train.lock, jeśli należy do bieżącego procesu."""
    try:
        if not os.path.isfile(lock_path):
            return
        with open(lock_path, encoding="utf-8") as f:
            prev = json.load(f)
        if int(prev.get("pid") or -1) == os.getpid():
            os.remove(lock_path)
    except OSError:
        pass


def _ensure_watcher_alive(base_dir: str) -> None:
    from z_maina_funkcje.plgrid_train_eval_watcher import (
        ensure_eval_watcher_running,
        _pid_alive as watcher_pid_alive,
        _load_json,
    )

    ok = ensure_eval_watcher_running(base_dir)
    pid_path = os.path.join(
        base_dir, "shared", "agents", "actual_training_logs", "eval_watcher.pid"
    )
    # krótki grace po Popen
    time.sleep(0.8)
    meta = _load_json(pid_path, {})
    pid = int(meta.get("pid") or 0)
    if not ok or not watcher_pid_alive(pid):
        raise RuntimeError(
            "EvalWatcher nie działa — bez watchera brak snapshotu A1 / early-stop. "
            f"Uruchom ręcznie: python z_maina_funkcje/plgrid_train_eval_watcher.py  "
            f"(pid_path={pid_path})"
        )


def promote_sim_best_for_seed(
    *,
    seed: int,
    logs_dir: str = LOGS_DIR,
    agent_dir: str = AGENT_DIR,
    train_start_time: Optional[str] = None,
    eval_window_id: str = "A1",
    sim_mode: str = "rl",
) -> Optional[Dict[str, Any]]:
    """
    Kopiuje workdir_sim_best → agent_ppo_seed{S}_simbest_{ts}.zip (+ obs_rms + json).
    Zwraca meta albo None gdy brak sim_best z tej sesji.
    """
    sb_path = os.path.join(logs_dir, "sim_best.json")
    if not os.path.isfile(sb_path):
        return None
    with open(sb_path, encoding="utf-8") as f:
        sb = json.load(f)
    if train_start_time and sb.get("train_start_time") and sb.get("train_start_time") != train_start_time:
        return None

    src_name = sb.get("agent_file") or "agent_ppo_workdir_sim_best.zip"
    src = os.path.join(agent_dir, src_name)
    if not os.path.isfile(src):
        return None

    # stała nazwa: najlepszy agent seeda (A1) = PPO_best_{seed}; nowy trening nadpisuje
    # rl (R2/M2) -> PPO_best_{seed}; rlul (R3/M3) -> PPO_best_UL_{seed}
    prefix = "PPO_best_UL" if sim_mode == "rlul" else "PPO_best"
    dst_name = f"{prefix}_{int(seed)}.zip"
    dst = os.path.join(agent_dir, dst_name)
    shutil.copy2(src, dst)

    src_rms = obs_rms_path_for_agent(src)
    # legacy name from watcher
    alt_rms = os.path.join(agent_dir, "agent_workdir_sim_best_obs_rms.npz")
    rms_src = src_rms if os.path.isfile(src_rms) else (alt_rms if os.path.isfile(alt_rms) else None)
    rms_dst = None
    if rms_src:
        rms_dst = obs_rms_path_for_agent(dst)
        shutil.copy2(rms_src, rms_dst)

    meta = {
        "schema": "turtle.artifact.v1",
        "kind": "agent_simbest",
        "seed": int(seed),
        "agent_file": dst_name,
        "agent_path": dst,
        "obs_rms_file": os.path.basename(rms_dst) if rms_dst else None,
        "source_workdir": src_name,
        "excess_pct": sb.get("excess_pct"),
        "epoch_sum": sb.get("epoch_sum"),
        "global_dump": sb.get("global_dump"),
        "eval_window_id": eval_window_id,
        "train_start_time": sb.get("train_start_time"),
        "config_hash": thesis_train_config_hash(),
        "git_commit": git_commit(BASE_DIR),
        "timestamp": datetime.now().isoformat(),
        "selection": "sim_best",
        "sim_mode": sim_mode,
    }
    meta_path = os.path.splitext(dst)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def run_pretrain(
    *,
    seed: int,
    sim_mode: str = "rl",
    num_envs=None,
    service=None,
    learning_rate=None,
    acquire_lock: bool = True,
    ensure_watcher: bool = True,
    logs_dir: str = LOGS_DIR,
    agent_dir: str = AGENT_DIR,
    lock_path: str = TRAIN_LOCK_PATH,
) -> Dict[str, Any]:
    """Pretrain PPO dla jednego ziarna; zwraca wynik train_agent_ppo oraz meta sim_best."""
    if int(seed) not in THESIS_SEEDS:
        print(
            f"[TrainPPO] WARN seed={seed} spoza THESIS_SEEDS {list(THESIS_SEEDS)} "
            f"— kontynuuję"
        )

    if acquire_lock:
        acquire_train_lock(int(seed), lock_path=lock_path)
    try:
        if ensure_watcher:
            _ensure_watcher_alive(BASE_DIR)

        kw = build_pretrain_ppo_kwargs(
            seed=int(seed),
            sim_mode=sim_mode,
            num_envs=num_envs,
            learning_rate=learning_rate,
        )
        trainer = service if service is not None else TrainAgentPPOService(BASE_DIR)
        result = trainer.train_agent_ppo(
            rabbitmq_msg_id=None,
            model_files=dict(THESIS_MODEL_PATHS),
            data_files=dict(THESIS_DATA_FILES),
            **kw,
        ) or {}

        # train_start_time z full_training / eval_request jeśli dostępne
        train_start = None
        ft = os.path.join(logs_dir, "full_training.json")
        if os.path.isfile(ft):
            try:
                with open(ft, encoding="utf-8") as f:
                    train_start = (json.load(f).get("config") or {}).get("start_time")
            except Exception:
                pass
        if train_start is None:
            er = os.path.join(logs_dir, "eval_request.json")
            if os.path.isfile(er):
                try:
                    with open(er, encoding="utf-8") as f:
                        train_start = json.load(f).get("train_start_time")
                except Exception:
                    pass

        tag = str(result.get("result") or "success")
        # promote dla success / early_stop / interrupted
        if tag in ("success", "early_stop", "interrupted"):
            promo = promote_sim_best_for_seed(
                seed=int(seed),
                logs_dir=logs_dir,
                agent_dir=agent_dir,
                train_start_time=train_start,
                eval_window_id=str(kw.get("eval_window_id") or "A1"),
                sim_mode=sim_mode,
            )
        else:
            promo = None

        if promo:
            result["simbest_file"] = promo["agent_file"]
            result["simbest_path"] = promo["agent_path"]
            result["selection"] = "sim_best"
            print(
                f"[TrainPPO] sim_best → {promo['agent_file']}  "
                f"(excess={promo.get('excess_pct')} es={promo.get('epoch_sum')})"
            )
            print(
                f"[TrainPPO] Wklej do thesis_sequence_config by_seed.{seed}.agent_rl = "
                f"{promo['agent_file']}"
            )
        else:
            print(
                "[TrainPPO] *** WARN: brak sim_best.json z tej sesji — "
                "używam finalnego agenta (selection=final_fallback). "
                "Sprawdź EvalWatcher / A1 eval. ***"
            )
            result["simbest_file"] = result.get("agent_file")
            result["simbest_path"] = result.get("agent_path")
            result["selection"] = "final_fallback"
            if result.get("agent_file"):
                print(
                    f"[TrainPPO] fallback agent_rl = {result['agent_file']}"
                )
        return result
    finally:
        if acquire_lock:
            release_train_lock(lock_path=lock_path)


def main(argv=None):
    args = _parse_args(argv)
    seeds = _resolve_seeds(args)
    for i, seed in enumerate(seeds):
        print(f"\n######## Pretrain PPO  seed={seed}  mode={args.sim_mode}  "
              f"({i + 1}/{len(seeds)}) ########")
        run_pretrain(
            seed=int(seed),
            sim_mode=args.sim_mode,
            num_envs=args.num_envs,
        )


if __name__ == "__main__":
    main()
