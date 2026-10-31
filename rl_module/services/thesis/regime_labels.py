"""Budowa serii one-hot reżimu dla okna (z lookbackiem) — wspólne dla sim/train."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from rl_module.services.regime_service import (
    REGIME_VOL_LONG,
    RegimeConfig,
    RegimeService,
    load_regime_lock,
    regime_model_train_path,
    require_regime_k,
)
from test_module.services.core_logic.windows import load_windows, slice_windows


def _data_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "shared", "files", "processed_simulation_data")


def load_closes_with_lookback(
    base_dir: str,
    data_files: Dict[str, str],
    window_id: str,
    *,
    lookback_bars: int = REGIME_VOL_LONG,
) -> Tuple[Dict[str, np.ndarray], int, int]:
    """
    Ładuje lookback + okno. Zwraca (closes, out_start, out_end) względem
    zwróconej serii — cechy liczone na całości, etykiety tylko [out_start:out_end].
    """
    windows = load_windows(base_dir)
    seq = list(windows.get("sequence") or [])
    wmap = windows["windows"]
    if window_id not in wmap and window_id not in (windows.get("aliases") or {}):
        raise KeyError(f"Nieznane okno: {window_id}")

    # pełny zakres: od (start_okna - lookback) do end okna
    # prostsza ścieżka: załaduj prefix sequence do window_id włącznie
    if window_id in seq:
        idx = seq.index(window_id)
        # weź kilka poprzednich okien jako lookback (wystarczy rolling 3 + warm)
        prior = seq[max(0, idx - 3) : idx]
        spec = "+".join(prior + [window_id]) if prior else window_id
    else:
        spec = window_id

    closes_full: Dict[str, np.ndarray] = {}
    n_eval = None
    for sym, name in data_files.items():
        path = name if os.path.isabs(name) else os.path.join(_data_dir(base_dir), name)
        df = pd.read_csv(path)
        df_full = slice_windows(df, windows, spec)
        df_eval = slice_windows(df, windows, window_id)
        closes_full[sym] = df_full["close"].astype(float).to_numpy()
        n_eval = len(df_eval) if n_eval is None else min(n_eval, len(df_eval))

    n_full = min(len(v) for v in closes_full.values())
    out_end = n_full
    out_start = max(0, n_full - int(n_eval or 0))
    return closes_full, out_start, out_end


def build_onehot_for_window(
    base_dir: str,
    data_files: Dict[str, str],
    window_id: str,
    *,
    model_path: Optional[str] = None,
    refit: bool = False,
    fit_window_spec: Optional[str] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Zwraca (onehot[T_eval, K], K, meta).
    refit=False: ładuje zamrożony model (R3 / pretrain).
    refit=True: fit na fit_window_spec (rolling past), K z lock.
    """
    k = require_regime_k(base_dir)
    lock = load_regime_lock(base_dir)
    svc = RegimeService(RegimeConfig(k_locked=k, seed=int(lock.get("seed") or 0)))

    meta: Dict[str, Any] = {"k": k, "refit": bool(refit), "window_id": window_id}

    if refit:
        if not fit_window_spec:
            raise ValueError("refit wymaga fit_window_spec")
        windows = load_windows(base_dir)
        closes_fit: Dict[str, np.ndarray] = {}
        for sym, name in data_files.items():
            path = name if os.path.isabs(name) else os.path.join(_data_dir(base_dir), name)
            df = pd.read_csv(path)
            df = slice_windows(df, windows, fit_window_spec)
            closes_fit[sym] = df["close"].astype(float).to_numpy()
        t0 = __import__("time").time()
        svc.fit_from_closes(closes_fit, seed=int(seed), k=k)
        meta["ul_fit_time_s"] = __import__("time").time() - t0
        meta["fit_windows"] = fit_window_spec
    else:
        mp = model_path or regime_model_train_path(base_dir)
        if not os.path.isfile(mp):
            raise FileNotFoundError(
                f"Brak modelu UL: {mp} — uruchom plgrid_regime_lock.py"
            )
        svc.load(mp)
        meta["model_path"] = mp
        meta["ul_fit_time_s"] = 0.0

    closes, out_start, out_end = load_closes_with_lookback(
        base_dir, data_files, window_id,
    )
    onehot = svc.predict_onehot_from_closes(
        closes, out_start=out_start, out_end=out_end,
    )
    meta["profile"] = svc.profile_
    meta["n_labels"] = int(onehot.shape[0])
    return onehot, k, meta
