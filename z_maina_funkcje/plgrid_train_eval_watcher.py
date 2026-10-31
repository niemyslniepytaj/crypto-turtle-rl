"""
Watcher: osobny proces obok treningu PPO.

Czyta shared/agents/actual_training_logs/eval_request.json (pisany po każdym
global dumpie), odpala symulację sim_mode=rl / v5rl i zapisuje wynik z tagiem
`rltrain_es{N}` — widać w simulation_viewer_dashboard (filtr RL train).

Zwykle startuje sam z train_agent_ppo (live_logging=True).
Ręcznie: python z_maina_funkcje/plgrid_train_eval_watcher.py
"""
import os
import sys
import time
import json
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rl_module.services.thesis.knobs import (
    MP_CONST,
    V5RL_KNOB_HOLD_BARS as _V5RL_HOLD,
    V5RL_TRADE_LOCK_BARS as _V5RL_LOCK,
)

REQUEST_PATH = os.path.join(
    BASE_DIR, "shared", "agents", "actual_training_logs", "eval_request.json"
)
STATE_PATH = os.path.join(
    BASE_DIR, "shared", "agents", "actual_training_logs", "eval_watcher_state.json"
)
PID_PATH = os.path.join(
    BASE_DIR, "shared", "agents", "actual_training_logs", "eval_watcher.pid"
)
LOG_PATH = os.path.join(
    BASE_DIR, "shared", "agents", "actual_training_logs", "eval_watcher.log"
)
POLL_SEC = 20


def _pid_alive(pid: int) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            SYNCHRONIZE = 0x00100000
            handle = kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def ensure_eval_watcher_running(base_dir=None, reset_state=True) -> bool:
    """
    Uruchamia watchera w tle jeśli jeszcze nie działa.
    reset_state=True (domyślnie przy starcie treningu): czyści last_epoch_sum,
    żeby nowy run nie był blokowany przez stan poprzedniego treningu.
    """
    root = base_dir or BASE_DIR
    log_dir = os.path.join(root, "shared", "agents", "actual_training_logs")
    pid_path = os.path.join(log_dir, "eval_watcher.pid")
    log_path = os.path.join(log_dir, "eval_watcher.log")
    state_path = os.path.join(log_dir, "eval_watcher_state.json")
    os.makedirs(log_dir, exist_ok=True)

    if reset_state:
        _save_json(state_path, {
            "last_epoch_sum": -1,
            "train_start_time": None,
            "reset_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print("[EvalWatcher] state zresetowany pod nowy trening")

    old = _load_json(pid_path, {})
    old_pid = int(old.get("pid") or 0)
    if _pid_alive(old_pid):
        print(f"[EvalWatcher] już działa (pid={old_pid})")
        return True

    script = os.path.join(root, "z_maina_funkcje", "plgrid_train_eval_watcher.py")
    if not os.path.isfile(script):
        print(f"[EvalWatcher] brak skryptu: {script}")
        return False

    log_f = open(log_path, "a", encoding="utf-8")
    log_f.write(f"\n--- start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    log_f.flush()

    kwargs = {
        "args": [sys.executable, "-u", script, "--daemon"],
        "cwd": root,
        "stdin": subprocess.DEVNULL,
        "stdout": log_f,
        "stderr": subprocess.STDOUT,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        kwargs["start_new_session"] = True

    proc = subprocess.Popen(**kwargs)
    _save_json(pid_path, {
        "pid": proc.pid,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "log": log_path,
    })
    print(f"[EvalWatcher] auto-start pid={proc.pid}  log={log_path}")
    return True


def _load_json(path, default=None):
    if not os.path.isfile(path):
        return default if default is not None else {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _maybe_promote_sim_best(req: dict, sim_result: dict):
    """Jeśli OOS lepszy (excess, potem final $) → kopiuj agent → workdir_sim_best."""
    import shutil

    meta = req.get("train_meta") or {}
    agent_type = str(meta.get("agent_type") or "ppo")
    agent_dir = os.path.join(BASE_DIR, "shared", "agents", agent_type)
    latest = os.path.join(agent_dir, req.get("agent_file") or f"agent_{agent_type}_workdir_latest.zip")
    if not os.path.isfile(latest):
        return

    final = sim_result.get("final_portfolio_value")
    hodl = sim_result.get("hodl_return")
    try:
        final_f = float(final)
    except (TypeError, ValueError):
        return
    bal0 = 1000.0
    strat_ret = (final_f / bal0 - 1.0) * 100.0
    try:
        excess = strat_ret - float(hodl) if hodl is not None else strat_ret
    except (TypeError, ValueError):
        excess = strat_ret

    # ranking: najpierw excess, przy remisie final $
    score = (float(excess), final_f)

    best_meta_path = os.path.join(
        BASE_DIR, "shared", "agents", "actual_training_logs", "sim_best.json"
    )
    prev = _load_json(best_meta_path, {})
    prev_score = prev.get("score")
    req_start = req.get("train_start_time")
    prev_start = prev.get("train_start_time")
    # nie porównuj z best z innej sesji treningu
    same_session = (not prev_start) or (not req_start) or (prev_start == req_start)
    if same_session and isinstance(prev_score, (list, tuple)) and len(prev_score) >= 2:
        prev_t = (float(prev_score[0]), float(prev_score[1]))
    else:
        prev_t = (float("-inf"), float("-inf"))

    if score <= prev_t:
        print(
            f"[EvalWatcher] sim score excess={score[0]:+.2f}% ${score[1]:.0f} "
            f"< best {prev_t[0]:+.2f}% ${prev_t[1]:.0f} — bez promocji"
        )
        return

    best_name = f"agent_{agent_type}_workdir_sim_best.zip"
    best_path = os.path.join(agent_dir, best_name)
    shutil.copy2(latest, best_path)
    obs = req.get("obs_rms_file")
    if obs:
        src = os.path.join(agent_dir, obs)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(agent_dir, "agent_workdir_sim_best_obs_rms.npz"))

    _save_json(best_meta_path, {
        "score": [score[0], score[1]],
        "score_key": ["excess_pct", "final_portfolio_value"],
        "epoch_sum": req.get("epoch_sum"),
        "global_dump": req.get("global_dump"),
        "agent_file": best_name,
        "eval_window_id": meta.get("eval_window_id"),
        "strat_return_pct": strat_ret,
        "hodl_return_pct": hodl,
        "excess_pct": score[0],
        "final_portfolio_value": final_f,
        "train_start_time": req.get("train_start_time"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    print(
        f"[EvalWatcher] * sim_best <- es={req.get('epoch_sum')} "
        f"excess={score[0]:+.2f}% ${final_f:.0f}"
    )


def run_eval(req: dict):
    from test_module.services.simulation_service import SimulationService

    meta = req.get("train_meta") or {}
    model_files = dict(meta.get("model_files") or {})
    data_files = dict(meta.get("data_files") or {})
    if not model_files or not data_files:
        raise ValueError("eval_request bez model_files/data_files — pomijam")

    exp = meta.get("experiment_version") or "v6"
    # meta-kontroler = rl (ex-mprl); legacy requesty bez sim_mode → rl
    sim_mode = meta.get("sim_mode") or "rl"
    window_id = meta.get("eval_window_id")
    if not window_id:
        raise ValueError("brak train_meta.eval_window_id — ustaw w wywołaniu treningu")
    agent_file = req.get("agent_file") or "agent_ppo_workdir_latest.zip"
    epoch_sum = int(req.get("epoch_sum") or 0)
    name_tag = f"rltrain_es{epoch_sum}"

    print(
        f"[EvalWatcher] Start sim rltrain es={epoch_sum} "
        f"window={window_id} agent={agent_file} exp={exp} mode={sim_mode}"
    )

    # rlul (R3/M3): walidacja A1 musi dostac prawdziwe etykiety rezimu (nie zera)
    regime_kw = {}
    if sim_mode == "rlul":
        from rl_module.services.thesis.regime_labels import build_onehot_for_window
        _series, _rk, _ = build_onehot_for_window(
            BASE_DIR, data_files, window_id, refit=False,
            seed=int(meta.get("seed") or 0),
        )
        regime_kw = {"regime_k": int(_rk), "regime_onehot_series": _series}

    result = SimulationService(BASE_DIR).run_simulation(
        rabbit_msg_id=None,
        model_paths=model_files,
        data_files=data_files,
        agent_rl=agent_file,
        **regime_kw,
        balance_start=1000.0,
        add_fee=True,
        sim_mode=sim_mode,
        rlagent_deterministic=True,
        slippage=0,
        threshold_buy=float(meta.get("mp_threshold") or MP_CONST["threshold_buy"]),
        threshold_sell=float(meta.get("mp_threshold") or MP_CONST["threshold_sell"]),
        stop_loss_cooldown=int(
            meta.get("stop_loss_cooldown") or MP_CONST["stop_loss_cooldown"]
        ),
        wallet_percentage_limit=float(
            meta.get("wallet_percentage_limit") or MP_CONST["wallet_percentage_limit"]
        ),
        max_wallet_percentage_crypto=float(
            meta.get("max_wallet_percentage_crypto")
            or MP_CONST["max_wallet_percentage_crypto"]
        ),
        stop_loss=float(meta.get("stop_loss") or MP_CONST["stop_loss"]),
        mincost=float(meta.get("mincost") or MP_CONST["mincost"]),
        v5rl_knob_hold_bars=int(meta["v5rl_knob_hold_bars"]) if meta.get("v5rl_knob_hold_bars") is not None else _V5RL_HOLD,
        v5rl_trade_lock_bars=int(meta["v5rl_trade_lock_bars"]) if meta.get("v5rl_trade_lock_bars") is not None else _V5RL_LOCK,
        plots=False,
        logs=True,
        results=True,
        experiment_version=exp,
        window_id=window_id,
        name_tag=name_tag,
        obs_rms_file=req.get("obs_rms_file"),
    )
    if isinstance(result, dict) and result.get("result") == "success":
        _maybe_promote_sim_best(req, result)
    print(f"[EvalWatcher] Done {name_tag}")


def main():
    # zapisz własny pid (daemon / ręczny start)
    _save_json(PID_PATH, {
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "log": LOG_PATH,
    })
    print(f"[EvalWatcher] Poll {REQUEST_PATH} co {POLL_SEC}s (pid={os.getpid()})")
    state = _load_json(STATE_PATH, {"last_epoch_sum": -1, "train_start_time": None})
    while True:
        req = _load_json(REQUEST_PATH, None)
        if req:
            es = int(req.get("epoch_sum") or -1)
            # tylko jawny train_start_time (nie created_at — zmienia się co dump)
            train_start = req.get("train_start_time") or (req.get("train_meta") or {}).get("start_time")
            prev_start = state.get("train_start_time")
            last_es = int(state.get("last_epoch_sum", -1))

            # nowy run treningu albo epoch poszło wstecz → reset licznika
            new_session = bool(train_start and prev_start and train_start != prev_start)
            rewind = es >= 0 and last_es >= 0 and es < last_es
            if new_session or rewind:
                print(
                    f"[EvalWatcher] reset state "
                    f"(session_new={new_session} rewind={rewind} "
                    f"last_es={last_es} -> es={es})"
                )
                state["last_epoch_sum"] = -1
                if train_start:
                    state["train_start_time"] = train_start
                last_es = -1
                _save_json(STATE_PATH, state)
            elif train_start and not prev_start:
                state["train_start_time"] = train_start

            if es > last_es:
                try:
                    run_eval(req)
                    state["last_epoch_sum"] = es
                    state["train_start_time"] = train_start
                    state["last_run_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    _save_json(STATE_PATH, state)
                except Exception as e:
                    import traceback
                    print(f"[EvalWatcher] ERROR: {e}")
                    traceback.print_exc()
                    state["last_error_epoch"] = es
                    state["last_error"] = str(e)
                    state["last_epoch_sum"] = es
                    state["train_start_time"] = train_start
                    _save_json(STATE_PATH, state)
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    # --daemon: ten sam main (flaga tylko dla czytelności autostartu)
    main()
