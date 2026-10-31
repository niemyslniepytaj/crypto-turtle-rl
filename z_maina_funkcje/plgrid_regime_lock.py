"""
Narzędzie: wybór K (silhouette) + fit UL na T1–T6 → regime_lock.json + model.

Użytkownik uruchamia RAZ przed treningiem R2/M2/M3/R3.
  python z_maina_funkcje/plgrid_regime_lock.py
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rl_module.services.regime_service import (
    REGIME_LOCK_SEED,
    RegimeConfig,
    RegimeService,
    regime_lock_path,
    regime_model_train_path,
)
from rl_module.services.thesis.assets import THESIS_DATA_FILES
from rl_module.services.thesis import console as thesis_console
from test_module.services.core_logic.windows import load_windows, slice_windows
import pandas as pd


def _load_closes(base_dir: str, data_files: dict, window_spec: str) -> dict:
    windows = load_windows(base_dir)
    data_dir = os.path.join(base_dir, "shared", "files", "processed_simulation_data")
    out = {}
    for sym, name in data_files.items():
        path = name if os.path.isabs(name) else os.path.join(data_dir, name)
        df = pd.read_csv(path)
        df = slice_windows(df, windows, window_spec)
        out[sym] = df["close"].astype(float).to_numpy()
    return out


if __name__ == "__main__":
    data_files = dict(THESIS_DATA_FILES)
    closes = _load_closes(BASE_DIR, data_files, "train")
    thesis_console.print_card(
        "Regime lock (T1–T6)",
        [
            ("symbols", ", ".join(closes.keys())),
            ("n_ticks", min(len(v) for v in closes.values())),
            ("seed", REGIME_LOCK_SEED),
            ("K range", "2..5"),
        ],
        tag="RegimeLock",
    )

    svc = RegimeService(RegimeConfig(seed=REGIME_LOCK_SEED, k_locked=None))
    k = svc.fit_from_closes(closes, seed=REGIME_LOCK_SEED)
    model_path = regime_model_train_path(BASE_DIR)
    lock_path = regime_lock_path(BASE_DIR)
    svc.save(model_path)

    feature_stats = {
        "mean": None if svc.scaler_mean_ is None else svc.scaler_mean_.tolist(),
        "scale": None if svc.scaler_scale_ is None else svc.scaler_scale_.tolist(),
        "feature_names": svc.feature_names,
    }
    lock = {
        "k": int(k),
        "silhouette": float(svc.last_silhouette) if svc.last_silhouette == svc.last_silhouette else None,
        "seed": int(REGIME_LOCK_SEED),
        "window": "train",
        "method": "kmeans",
        "encoding": "onehot",
        "feature_stats": feature_stats,
        "model_path": os.path.relpath(model_path, BASE_DIR).replace("\\", "/"),
        "profile": svc.profile_,
        "vol_short": svc.config.vol_short,
        "vol_long": svc.config.vol_long,
        "ret_horizon": svc.config.ret_horizon,
    }
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2)

    thesis_console.print_kv("RegimeLock", f"K={k} silhouette={svc.last_silhouette}", kind="ok")
    thesis_console.print_kv("RegimeLock", f"lock -> {lock_path}", kind="ok")
    thesis_console.print_kv("RegimeLock", f"model -> {model_path}", kind="ok")
