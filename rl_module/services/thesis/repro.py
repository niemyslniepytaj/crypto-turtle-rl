"""Determinizm / odtwarzalność — hash danych, wersje lib, git (I0.5)."""

from __future__ import annotations

import hashlib
import os
import subprocess
from typing import Any, Dict, Optional


def file_sha256(path: str, chunk: int = 1 << 20) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_commit(base_dir: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=base_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


def library_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in (
        "numpy", "pandas", "scipy", "sklearn", "torch",
        "stable_baselines3", "optuna", "gymnasium",
    ):
        try:
            mod = __import__(name if name != "sklearn" else "sklearn")
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[name] = "missing"
    return out


def hardware_snapshot() -> Dict[str, Any]:
    import platform

    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["device"] = "cuda"
        else:
            info["device"] = "cpu"
    except Exception:
        info["device"] = "cpu"
        info["cuda_available"] = False
    return info


def hash_data_and_models(
    *,
    base_dir: str,
    data_files: Dict[str, str],
    model_paths: Dict[str, str],
) -> Dict[str, Any]:
    data_dir = os.path.join(base_dir, "shared", "files", "processed_simulation_data")
    model_dir = os.path.join(base_dir, "shared", "models")
    data_hashes = {}
    for sym, name in (data_files or {}).items():
        path = name if os.path.isabs(name) else os.path.join(data_dir, name)
        data_hashes[sym] = {"file": name, "sha256": file_sha256(path)}
    model_hashes = {}
    for sym, name in (model_paths or {}).items():
        # model może być katalogiem lub plikiem — hash pierwszego .pt/.pkl/.joblib jeśli katalog
        path = name if os.path.isabs(name) else os.path.join(model_dir, name)
        if os.path.isdir(path):
            candidates = []
            for root, _dirs, files in os.walk(path):
                for fn in files:
                    if fn.endswith((".pt", ".pth", ".pkl", ".joblib", ".h5")):
                        candidates.append(os.path.join(root, fn))
            candidates.sort()
            model_hashes[sym] = {
                "path": name,
                "sha256": file_sha256(candidates[0]) if candidates else None,
                "n_files": len(candidates),
            }
        else:
            model_hashes[sym] = {"path": name, "sha256": file_sha256(path)}
    return {"data": data_hashes, "models": model_hashes}


def build_repro_block(
    *,
    base_dir: str,
    data_files: Dict[str, str],
    model_paths: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "git_commit": git_commit(base_dir),
        "library_versions": library_versions(),
        "hardware": hardware_snapshot(),
        "hashes": hash_data_and_models(
            base_dir=base_dir,
            data_files=data_files,
            model_paths=model_paths,
        ),
    }
