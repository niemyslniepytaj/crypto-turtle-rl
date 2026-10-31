"""Zapis / odczyt statystyk VecNormalize (obs_rms) obok agenta — I0.3."""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import numpy as np


def obs_rms_path_for_agent(agent_zip_path: str) -> str:
    """Ścieżka pliku ``*_obs_rms.npz`` odpowiadająca archiwum agenta."""
    root, _ = os.path.splitext(agent_zip_path)
    return f"{root}_obs_rms.npz"


def resolve_obs_rms_beside_agent(agent_ref: str, agents_dir: Optional[str] = None) -> Optional[str]:
    """Wyszukuje istniejący plik obs_rms obok wskazanego agenta."""
    candidates = []
    if agent_ref.endswith(".npz"):
        candidates.append(agent_ref)
    else:
        candidates.append(obs_rms_path_for_agent(agent_ref))
        if agent_ref.endswith(".zip"):
            candidates.append(obs_rms_path_for_agent(agent_ref))
        else:
            candidates.append(obs_rms_path_for_agent(f"{agent_ref}.zip"))
        if agents_dir and not os.path.isabs(agent_ref):
            base = agent_ref if agent_ref.endswith(".zip") else f"{agent_ref}.zip"
            candidates.append(obs_rms_path_for_agent(os.path.join(agents_dir, base)))
            candidates.append(os.path.join(agents_dir, "agent_workdir_latest_obs_rms.npz"))
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def save_vecnormalize_obs_rms(venv: Any, path: str, clip_obs: float = 5.0, epsilon: float = 1e-8) -> Optional[str]:
    """Z VecNormalize (lub env z obs_rms) → npz kompatybilny z DecisionEngine.load_obs_rms."""
    vn = venv
    # zejdź przez wrappery
    seen = set()
    while vn is not None and id(vn) not in seen:
        seen.add(id(vn))
        if getattr(vn, "obs_rms", None) is not None and type(vn).__name__ == "VecNormalize":
            break
        if hasattr(vn, "venv"):
            vn = vn.venv
        elif hasattr(vn, "env"):
            vn = vn.env
        else:
            break
    if vn is None or getattr(vn, "obs_rms", None) is None:
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(
        path,
        mean=np.asarray(vn.obs_rms.mean, dtype=np.float64),
        var=np.asarray(vn.obs_rms.var, dtype=np.float64),
        clip_obs=np.asarray(float(getattr(vn, "clip_obs", clip_obs)), dtype=np.float64),
        epsilon=np.asarray(float(getattr(vn, "epsilon", epsilon)), dtype=np.float64),
        count=np.asarray(float(getattr(vn.obs_rms, "count", 0.0)), dtype=np.float64),
    )
    return path


def load_obs_rms_into_vecnormalize(venv: Any, path: str) -> bool:
    """Wgrywa mean/var do VecNormalize.obs_rms (continue M2)."""
    if not path or not os.path.isfile(path):
        return False
    data = np.load(path)
    vn = venv
    seen = set()
    while vn is not None and id(vn) not in seen:
        seen.add(id(vn))
        if getattr(vn, "obs_rms", None) is not None and type(vn).__name__ == "VecNormalize":
            mean = np.asarray(data["mean"], dtype=np.float64)
            var = np.asarray(data["var"], dtype=np.float64)
            n = min(vn.obs_rms.mean.shape[0], mean.shape[0])
            vn.obs_rms.mean[:n] = mean[:n]
            vn.obs_rms.var[:n] = var[:n]
            if "count" in data:
                try:
                    vn.obs_rms.count = float(data["count"])
                except Exception:
                    pass
            return True
        if hasattr(vn, "venv"):
            vn = vn.venv
        elif hasattr(vn, "env"):
            vn = vn.env
        else:
            break
    return False


def find_vecnormalize(env: Any) -> Optional[Any]:
    seen = set()
    cur = env
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if type(cur).__name__ == "VecNormalize" or (
            getattr(cur, "obs_rms", None) is not None and hasattr(cur, "normalize_obs")
        ):
            return cur
        cur = getattr(cur, "venv", None) or getattr(cur, "env", None)
    return None
