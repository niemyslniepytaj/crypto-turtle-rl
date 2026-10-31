"""
Spójny schemat JSON artefaktów tezy (Optuna LOCK, agent PPO, sequence config).

schema: turtle.artifact.v1 | turtle.sequence_config.v1
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


SCHEMA_ARTIFACT = "turtle.artifact.v1"
SCHEMA_SEQUENCE = "turtle.sequence_config.v1"

KIND_OPTUNA = "optuna_best_params"
KIND_AGENT = "agent_ppo"


def _now() -> str:
    return datetime.now().isoformat()


def write_json(path: str, payload: Dict[str, Any]) -> str:
    """Zapisuje obiekt JSON (UTF-8, indent=2); zwraca ścieżkę."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def read_json(path: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
    """Wczytuje obiekt JSON; wymaga słownika w korzeniu pliku."""
    resolved = resolve_path(path, base_dir)
    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {resolved}")
    return data


def resolve_path(path: str, base_dir: Optional[str] = None) -> str:
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    if base_dir:
        cand = os.path.join(base_dir, path)
        if os.path.isfile(cand):
            return cand
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(path)


def make_optuna_artifact(
    *,
    seed: int,
    study_name: str,
    best_params: Dict[str, Any],
    best_metric: Any = None,
    best_trial: Any = None,
    static_params: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    experiment_version: Optional[str] = None,
    window: Optional[str] = None,
    best_agent: Any = None,
    stage_meta: Optional[Dict[str, Any]] = None,
    algorithm: str = "simulation",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Buduje payload artefaktu Optuna (schema turtle.artifact.v1)."""
    payload = {
        "schema": SCHEMA_ARTIFACT,
        "kind": KIND_OPTUNA,
        "seed": int(seed),
        "study_name": study_name,
        "algorithm": algorithm,
        "experiment_version": experiment_version,
        "window": window,
        "best_trial": best_trial,
        "best_metric": best_metric,
        "best_params": dict(best_params or {}),
        "static_params": dict(static_params or {}),
        "best_agent": best_agent,
        "config": config or {},
        "stage_meta": stage_meta or {},
        "timestamp": _now(),
    }
    if extra:
        payload.update(extra)
    return payload


def make_agent_artifact(
    *,
    seed: int,
    agent_file: str,
    agent_path: Optional[str] = None,
    experiment_version: Optional[str] = None,
    sim_mode: Optional[str] = None,
    window_id: Optional[str] = None,
    eval_window_id: Optional[str] = None,
    continue_from: Any = None,
    tag: Optional[str] = None,
    train_meta: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Buduje payload metadanych agenta PPO (schema turtle.artifact.v1)."""
    payload = {
        "schema": SCHEMA_ARTIFACT,
        "kind": KIND_AGENT,
        "seed": int(seed),
        "agent_file": agent_file,
        "agent_path": agent_path,
        "experiment_version": experiment_version,
        "sim_mode": sim_mode,
        "window_id": window_id,
        "eval_window_id": eval_window_id,
        "continue_from": continue_from,
        "tag": tag,
        "train_meta": dict(train_meta or {}),
        "timestamp": _now(),
    }
    if extra:
        payload.update(extra)
    return payload


def extract_best_params(loaded: Dict[str, Any]) -> Dict[str, Any]:
    """Wyodrębnia słownik best_params z artefaktu lub surowego JSON knobów."""
    if not isinstance(loaded, dict):
        raise ValueError("best_params JSON must be an object")
    if "best_params" in loaded and isinstance(loaded["best_params"], dict):
        return dict(loaded["best_params"])
    # raw knobs (legacy)
    skip = {
        "schema", "kind", "seed", "study_name", "algorithm", "experiment_version",
        "window", "best_trial", "best_metric", "static_params", "best_agent",
        "config", "stage_meta", "timestamp",
    }
    return {k: v for k, v in loaded.items() if k not in skip}


def extract_seed(loaded: Dict[str, Any], default: Optional[int] = None) -> Optional[int]:
    if not isinstance(loaded, dict):
        return default
    if "seed" in loaded and loaded["seed"] is not None:
        return int(loaded["seed"])
    return default


def agent_meta_path(agent_zip_path: str) -> str:
    """Sidecar next to zip: agent_ppo_....zip → agent_ppo_....json"""
    root, _ = os.path.splitext(agent_zip_path)
    return f"{root}.json"


def load_agent_artifact(
    agent_ref: str,
    base_dir: Optional[str] = None,
    agents_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load sidecar for agent zip/basename; None if missing."""
    candidates = []
    if agent_ref.endswith(".json"):
        candidates.append(agent_ref)
    else:
        candidates.append(agent_meta_path(agent_ref))
        if not os.path.isabs(agent_ref) and agents_dir:
            candidates.append(agent_meta_path(os.path.join(agents_dir, agent_ref)))
        if base_dir and not os.path.isabs(agent_ref):
            candidates.append(agent_meta_path(os.path.join(base_dir, agent_ref)))
    for path in candidates:
        try:
            resolved = resolve_path(path, base_dir)
            return read_json(resolved)
        except FileNotFoundError:
            continue
    return None


def load_sequence_config(path: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
    cfg = read_json(path, base_dir)
    if cfg.get("schema") and cfg.get("schema") != SCHEMA_SEQUENCE:
        # allow missing schema for drafts, but warn via kind check
        pass
    if "seeds" not in cfg or not cfg["seeds"]:
        raise ValueError(
            f"sequence config: brak niepustej listy 'seeds' w {path}"
        )
    try:
        cfg["seeds"] = [int(s) for s in cfg["seeds"]]
    except Exception as e:
        raise ValueError(f"sequence config: 'seeds' muszą być int — {e}") from e
    by_seed = cfg.get("by_seed") or {}
    try:
        cfg["by_seed"] = {str(int(k)): dict(v) for k, v in by_seed.items()}
    except Exception as e:
        raise ValueError(f"sequence config: 'by_seed' niepoprawny — {e}") from e
    arms = cfg.get("arms") or cfg.get("arm")
    if arms is not None:
        from rl_module.services.thesis.arms import CANONICAL_ARMS, normalize_arm
        raw = [arms] if isinstance(arms, str) else list(arms)
        for a in raw:
            try:
                normalize_arm(a)
            except Exception:
                raise ValueError(
                    f"sequence config: nieznane ramię {a!r} (znane: {CANONICAL_ARMS})"
                )
    return cfg


def seed_bundle(cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    return dict((cfg.get("by_seed") or {}).get(str(int(seed)) or {}) or {})


def load_ppo_hparams_lock(
    path: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Warsztatowy lock hiperparametrów PPO (jedna Optuna, wspólna dla seedów).
    Akceptuje artifact z best_params albo płaski dict.
    """
    if path is None:
        if not base_dir:
            raise ValueError("podaj path lub base_dir do ppo_hparams_lock.json")
        path = os.path.join(base_dir, "shared", "agents", "optuna_studies", "ppo_hparams_lock.json")
    data = read_json(path, base_dir)
    params = extract_best_params(data)
    # sensowne typy
    out = dict(params)
    for k in ("batch_size", "n_steps", "n_epochs", "episode_timesteps", "total_timesteps"):
        if k in out and out[k] is not None:
            out[k] = int(out[k])
    for k in ("learning_rate", "gamma", "ent_coef", "gae_lambda", "clip_range", "mp_threshold"):
        if k in out and out[k] is not None:
            out[k] = float(out[k])
    return out
