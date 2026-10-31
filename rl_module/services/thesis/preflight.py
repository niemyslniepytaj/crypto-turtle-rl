"""Walidacja sequence config + dry-run planu (I0.6)."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rl_module.services.thesis import console as thesis_console
from rl_module.services.thesis.arms import (
    CANONICAL_ARMS,
    IMPLEMENTED_ARMS,
    PLACEHOLDER_ARMS,
    normalize_arm,
)
from rl_module.services.thesis.knobs import tuner_settings_snapshot
from rl_module.services.thesis_artifacts import load_sequence_config, resolve_path, seed_bundle
from test_module.services.core_logic.windows import load_windows, parse_window_ids, slice_windows


class PreflightError(ValueError):
    """Błąd walidacji wstępnej sekwencji / ramienia tezy."""
    pass


def _seed_from_name(path: str) -> Optional[int]:
    name = os.path.basename(str(path))
    m = re.search(r"seed(\d+)", name, re.I) or re.search(r"PPO_best_(?:UL_)?(\d+)", name)
    return int(m.group(1)) if m else None


def validate_sequence_config(
    cfg: Dict[str, Any],
    *,
    base_dir: str,
    check_files: bool = True,
) -> List[str]:
    """Zwraca listę błędów (pusta = OK)."""
    errors: List[str] = []
    windows = load_windows(base_dir)
    wmap = windows.get("windows") or {}
    seq = list(windows.get("sequence") or wmap.keys())

    seeds = cfg.get("seeds") or []
    if not seeds:
        errors.append("brak 'seeds' (wymagana niepusta lista)")

    arms_raw = cfg.get("arms") or cfg.get("arm") or []
    if isinstance(arms_raw, str):
        arms_raw = [arms_raw]
    if not arms_raw:
        errors.append("brak 'arms' / 'arm'")
    arms = []
    for a in arms_raw:
        try:
            arms.append(normalize_arm(a))
        except Exception:
            errors.append(f"nieznane ramię: {a} (znane: {CANONICAL_ARMS})")

    start = cfg.get("start_window")
    end = cfg.get("end_window")
    eval_folds = cfg.get("eval_folds")
    if eval_folds:
        for fid in parse_window_ids(eval_folds) or []:
            if fid not in wmap and fid not in (windows.get("aliases") or {}):
                errors.append(f"okno spoza windows.json: {fid}")
    else:
        for wid in (start, end):
            if wid and wid not in wmap and wid not in (windows.get("aliases") or {}):
                errors.append(f"okno spoza windows.json: {wid}")
        if start and end and start in seq and end in seq:
            if seq.index(start) > seq.index(end):
                errors.append(f"start_window={start} po end_window={end}")

    data_files = cfg.get("data_files") or {}
    model_paths = cfg.get("model_paths") or {}
    data_dir = os.path.join(base_dir, "shared", "files", "processed_simulation_data")
    model_dir = os.path.join(base_dir, "shared", "models")

    # etap 1: UL (lock/model) wymagane TYLKO dla R3/M3 — nie dla R0–M2
    needs_ul = any(a in ("R3", "M3") for a in arms)
    if check_files and needs_ul:
        from rl_module.services.regime_service import (
            regime_lock_path,
            regime_model_train_path,
            load_regime_lock,
        )
        lp = regime_lock_path(base_dir)
        mp = regime_model_train_path(base_dir)
        if not os.path.isfile(lp):
            errors.append(
                f"brak {lp} — uruchom z_maina_funkcje/plgrid_regime_lock.py"
            )
        else:
            try:
                lock = load_regime_lock(base_dir)
                if int(lock.get("k") or 0) < 2:
                    errors.append(f"regime_lock.json: niepoprawne k={lock.get('k')}")
            except Exception as e:
                errors.append(f"regime_lock.json: {e}")
        if not os.path.isfile(mp):
            errors.append(
                f"brak modelu UL {mp} — uruchom plgrid_regime_lock.py"
            )
    if check_files:
        for sym, name in data_files.items():
            path = name if os.path.isabs(name) else os.path.join(data_dir, name)
            if not os.path.isfile(path):
                errors.append(f"brak CSV [{sym}]: {path}")
        for sym, name in model_paths.items():
            path = name if os.path.isabs(name) else os.path.join(model_dir, name)
            found = os.path.isfile(path) or os.path.isdir(path)
            if not found and not os.path.isabs(name):
                # SimulationService/MPSignal szuka modelu w podfolderach shared/models
                found = any(
                    os.path.isdir(os.path.join(model_dir, sub, name))
                    for sub in os.listdir(model_dir)
                )
            if not found:
                errors.append(f"brak modelu MP [{sym}]: {path}")

    by_seed = cfg.get("by_seed") or {}
    for seed in seeds:
        bundle = dict(by_seed.get(str(int(seed))) or {})
        # merge top-level defaults
        for k in ("best_params_path", "agent_rl", "agent_rlul"):
            bundle.setdefault(k, cfg.get(k))
        for arm in arms:
            if arm in ("R1", "M1"):
                lock = bundle.get("best_params_path") or cfg.get("best_params_path")
                if check_files and lock:
                    try:
                        lp = resolve_path(lock, base_dir)
                        if not os.path.isfile(lp):
                            errors.append(f"seed={seed} brak LOCK: {lock}")
                        else:
                            s_name = _seed_from_name(lp)
                            if s_name is not None and int(s_name) != int(seed):
                                errors.append(
                                    f"seed={seed} LOCK nazwa seed{s_name} ≠ {seed}: {lock}"
                                )
                    except FileNotFoundError:
                        errors.append(f"seed={seed} brak LOCK: {lock}")
                elif arm == "R1" and check_files and not lock and not cfg.get("mp_params"):
                    # R1 może iść na default_mp_params — tylko ostrzeżenie nie błąd
                    pass
            if arm in ("R2", "M2"):
                agent = bundle.get("agent_rl") or cfg.get("agent_rl")
                if check_files and agent:
                    ap = agent if os.path.isabs(agent) else os.path.join(
                        base_dir, "shared", "agents", "ppo", agent
                    )
                    if not agent.endswith(".zip"):
                        ap_zip = ap if ap.endswith(".zip") else ap + ".zip"
                    else:
                        ap_zip = ap
                    if not os.path.isfile(ap_zip) and not os.path.isfile(ap):
                        errors.append(f"seed={seed} brak agenta: {agent}")
                    else:
                        s_name = _seed_from_name(agent)
                        if s_name is not None and int(s_name) != int(seed):
                            errors.append(
                                f"seed={seed} agent nazwa seed{s_name} ≠ {seed}: {agent}"
                            )
            if arm in ("R3", "M3"):
                agent = (
                    bundle.get("agent_rlul")
                    or cfg.get("agent_rlul")
                    or bundle.get("agent_rl")
                    or cfg.get("agent_rl")
                )
                if check_files and not agent:
                    errors.append(f"seed={seed} brak agent_rlul dla {arm}")
                elif check_files and agent:
                    ap = agent if os.path.isabs(agent) else os.path.join(
                        base_dir, "shared", "agents", "ppo", agent
                    )
                    ap_zip = ap if ap.endswith(".zip") else ap + ".zip"
                    if not os.path.isfile(ap_zip) and not os.path.isfile(ap):
                        errors.append(f"seed={seed} brak agenta {arm} (rlul): {agent}")
                    else:
                        s_name = _seed_from_name(agent)
                        if s_name is not None and int(s_name) != int(seed):
                            errors.append(
                                f"seed={seed} agent_rlul nazwa seed{s_name} ≠ {seed}: {agent}"
                            )

    return errors


def build_execution_plan(cfg: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    windows = load_windows(base_dir)
    seq = list(windows.get("sequence") or [])
    start = cfg.get("start_window") or (seq[0] if seq else None)
    end = cfg.get("end_window") or start
    if cfg.get("eval_folds"):
        folds = parse_window_ids(cfg["eval_folds"]) or []
    else:
        if start in seq and end in seq:
            folds = seq[seq.index(start) : seq.index(end) + 1]
        else:
            folds = [x for x in (start, end) if x]

    arms_raw = cfg.get("arms") or cfg.get("arm") or []
    if isinstance(arms_raw, str):
        arms_raw = [arms_raw]
    arms = [normalize_arm(a) for a in arms_raw]
    skip = bool(cfg.get("skip_placeholders", True))
    if skip:
        arms = [a for a in arms if a not in PLACEHOLDER_ARMS]

    seeds = [int(s) for s in (cfg.get("seeds") or [])]
    plan_rows = []
    required_files: List[str] = []
    for seed in seeds:
        bundle = seed_bundle(cfg, seed)
        for arm in arms:
            plan_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "portfolio_mode": cfg.get("portfolio_mode", "reset"),
                    "folds": list(folds),
                    "optuna_n_trials": int(cfg.get("optuna_n_trials", 0)) if arm == "M1" else 0,
                    "continue_timesteps": int(cfg.get("continue_timesteps", 0))
                    if arm in ("M2", "M3")
                    else 0,
                    "best_params_path": bundle.get("best_params_path") or cfg.get("best_params_path"),
                    "agent_rl": bundle.get("agent_rl") or cfg.get("agent_rl"),
                }
            )
    return {
        "arms": arms,
        "seeds": seeds,
        "folds": folds,
        "portfolio_mode": cfg.get("portfolio_mode", "reset"),
        "optuna_n_trials": int(cfg.get("optuna_n_trials", 50)),
        "continue_timesteps": int(cfg.get("continue_timesteps", 200_000)),
        "tuner_settings": tuner_settings_snapshot(),
        "rows": plan_rows,
        "n_jobs": len(plan_rows),
    }


def run_preflight(config_path: str, base_dir: str) -> int:
    """0 = OK, 1 = błędy. Nic nie uruchamia."""
    cfg = load_sequence_config(config_path, base_dir)
    errors = validate_sequence_config(cfg, base_dir=base_dir, check_files=True)
    plan = build_execution_plan(cfg, base_dir)

    thesis_console.print_card(
        "Preflight sekwencji",
        [
            ("config", config_path),
            ("arms", ", ".join(plan["arms"]) or "—"),
            ("seeds", ", ".join(str(s) for s in plan["seeds"]) or "—"),
            ("folds", ", ".join(plan["folds"]) or "—"),
            ("portfolio_mode", plan["portfolio_mode"]),
            ("optuna_n_trials (M1)", plan["optuna_n_trials"]),
            ("continue_timesteps", plan["continue_timesteps"]),
            ("n_runów (arm×seed)", plan["n_jobs"]),
        ],
        tag="Preflight",
        ok=(len(errors) == 0),
    )
    for row in plan["rows"]:
        thesis_console.print_kv(
            "Plan",
            f"{row['arm']} seed={row['seed']} folds={','.join(row['folds'])} "
            f"optuna={row['optuna_n_trials']} ppo_steps={row['continue_timesteps']}",
            kind="info",
        )
    if errors:
        for e in errors:
            thesis_console.print_kv("Preflight", e, kind="err")
        return 1
    thesis_console.print_kv("Preflight", "OK — brak błędów artefaktów", kind="ok")
    return 0
