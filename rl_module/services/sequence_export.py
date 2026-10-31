"""
Jednolity eksport wyników sekwencji testowej (thesis).

- Nazwa runu: {ARM}_{label}_{seed}_{YYYYMMDD}_{start}_{end}
- Per okno: fold_*.json + seq_*.json.gz (simulation_viewer)
- Sekwencja B1+: sklejony equity (pomija A* w pliku ciągłym; A* jako osobne okna)
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence


def safe_token(s: Any) -> str:
    raw = str(s or "").strip()
    raw = raw.replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_\-]+", "", raw) or "run"


def make_run_id(
    *,
    arm: str,
    label: str,
    seed: int,
    start_window: str,
    end_window: str,
    date: Optional[str] = None,
) -> str:
    d = date or datetime.now().strftime("%Y%m%d")
    return "_".join(
        [
            safe_token(arm),
            safe_token(label),
            str(int(seed)),
            d,
            safe_token(start_window),
            safe_token(end_window),
        ]
    )


def window_gzip_basename(*, arm: str, label: str, seed: int, fold_id: str) -> str:
    """Prefix seq_ → kategoria w simulation_viewer."""
    return f"seq_{safe_token(arm)}_{safe_token(label)}_seed{int(seed)}_{safe_token(fold_id)}"


def sequence_b_gzip_basename(
    *,
    arm: str,
    label: str,
    seed: int,
    start_b: str,
    end_b: str,
) -> str:
    return (
        f"seq_{safe_token(arm)}_{safe_token(label)}_seed{int(seed)}"
        f"_sequence_{safe_token(start_b)}_{safe_token(end_b)}"
    )


def b_folds_only(folds: Sequence[str]) -> List[str]:
    return [f for f in folds if str(f).startswith("B")]


def write_viewer_gzip(path: str, payload: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def stitch_sequence_payload(
    *,
    fold_payloads: List[Dict[str, Any]],
    fold_ids: List[str],
    arm: str,
    seed: int,
    label: str,
    portfolio_mode: str,
) -> Dict[str, Any]:
    """
    Skleja outputy symulacji (format jak SimulationService.output) w jeden plik viewer.
    """
    if not fold_payloads:
        return {}

    # wykryj klucz mode/agent
    def _mode_key(p: Dict[str, Any]) -> str:
        if "agent" in p:
            return "agent"
        if "model" in p:
            return "model"
        for k in p:
            if k not in ("files", "config", "hodl") and isinstance(p.get(k), dict):
                return k
        return "model"

    mode = _mode_key(fold_payloads[0])
    portfolio: List[float] = []
    holdings: List[Any] = []
    hodl_refs: Dict[str, List[float]] = {}
    metrics_last = {}
    config = fold_payloads[0].get("config") or {}
    files = fold_payloads[0].get("files") or {}

    offset = 0.0
    for i, (fid, payload) in enumerate(zip(fold_ids, fold_payloads)):
        block = payload.get(mode) or {}
        vals = list(block.get("portfolio_values") or [])
        holds = list(block.get("holdings") or [])
        if not vals:
            continue
        # continuous stitch: przesuń kolejny segment, by stykał się z końcem poprzedniego
        if i > 0 and portfolio and vals:
            gap = float(portfolio[-1]) - float(vals[0])
            vals = [float(v) + gap for v in vals]
            # holdings nie przesuwamy numerycznie — to udziały; zostaw jak są
        elif i == 0:
            offset = float(vals[0]) if vals else 0.0
        portfolio.extend(float(v) for v in vals)
        holdings.extend(holds)
        metrics_last = dict(block.get("metrics") or {})
        refs = ((payload.get("hodl") or {}).get("crypto_ref_values") or {})
        for sym, series in refs.items():
            series = list(series or [])
            if not series:
                continue
            if sym not in hodl_refs:
                hodl_refs[sym] = []
            if hodl_refs[sym] and series:
                g = float(hodl_refs[sym][-1]) - float(series[0])
                series = [float(v) + g for v in series]
            hodl_refs[sym].extend(float(v) for v in series)

    return {
        "files": files,
        "config": {
            **config,
            "sequence_arm": arm,
            "sequence_seed": seed,
            "sequence_label": label,
            "sequence_folds": list(fold_ids),
            "portfolio_mode": portfolio_mode,
        },
        mode: {
            "metrics": metrics_last,
            "portfolio_values": portfolio,
            "holdings": holdings,
            "timestamps": len(portfolio),
        },
        "hodl": {"crypto_ref_values": hodl_refs},
    }


def copy_gzip_to_run(src_gzip: str, dest_dir: str, dest_name: Optional[str] = None) -> Optional[str]:
    if not src_gzip or not os.path.isfile(src_gzip):
        return None
    os.makedirs(dest_dir, exist_ok=True)
    name = dest_name or os.path.basename(src_gzip)
    dest = os.path.join(dest_dir, name)
    shutil.copy2(src_gzip, dest)
    return dest
