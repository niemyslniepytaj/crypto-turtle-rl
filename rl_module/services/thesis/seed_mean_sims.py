"""
Symulacje uśrednione po ziarnach (do Simulation Viewer / rysunków w pracy).

Z plików seq_{ARM}_{label}_seed{S}_{okno}.json.gz powstają
seq_{ARM}_{label}_mean_{okno}.json.gz: wartość portfela (i udziały) są średnią z ziaren,
HODL jest wspólny (identyczny dla wszystkich ziaren), metryki liczbowe to średnie z ziaren.
R0 jest pomijane (deterministyczne, jedno ziarno).
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_NAME_RE = re.compile(r"^seq_(?P<arm>[A-Z]\d+)_(?P<label>[^_]+)_seed(?P<seed>\d+)_(?P<tail>.+)\.json\.gz$")
SIM_KEYS = ("model", "agent")


def _load(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _save(path: str, data: Dict[str, Any]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as f:
        json.dump(data, f)


def _mean_arrays(seq: Sequence[Any]) -> List[Any]:
    arr = np.mean(np.asarray(seq, dtype=np.float64), axis=0)
    return arr.tolist()


def _mean_metrics(metric_list: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in metric_list[0].keys():
        vals = [m.get(k) for m in metric_list]
        nums = [float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if nums and len(nums) == len(vals):
            mean = float(np.mean(nums))
            out[k] = round(mean, 4) if not all(isinstance(v, int) for v in vals) else mean
        elif nums:
            out[k] = round(float(np.mean(nums)), 4)  # część ziaren bez wartości (np. profit factor)
        else:
            out[k] = vals[0]
    return out


def merge_group(paths: Sequence[str], arm: str, label: str, tail: str, seeds: Sequence[int]) -> Dict[str, Any]:
    datas = [_load(p) for p in paths]
    base = datas[0]
    merged = json.loads(json.dumps({k: v for k, v in base.items() if k not in SIM_KEYS}))
    for key in SIM_KEYS:
        if key not in base:
            continue
        blocks = [d[key] for d in datas]
        n = min(len(b["portfolio_values"]) for b in blocks)
        block = {k: v for k, v in base[key].items() if k not in ("portfolio_values", "holdings", "metrics")}
        block["portfolio_values"] = _mean_arrays([b["portfolio_values"][:n] for b in blocks])
        if all(b.get("holdings") for b in blocks):
            block["holdings"] = _mean_arrays([b["holdings"][:n] for b in blocks])
        if all(isinstance(b.get("metrics"), dict) for b in blocks):
            block["metrics"] = _mean_metrics([b["metrics"] for b in blocks])
        if isinstance(base[key].get("timestamps"), list):
            block["timestamps"] = base[key]["timestamps"][:n]
        merged[key] = block
    cfg = merged.setdefault("config", {})
    cfg["seed"] = "mean"
    cfg["seeds"] = [int(s) for s in seeds]
    cfg["n_seeds"] = len(seeds)
    cfg["averaged_over_seeds"] = True
    return merged


def build_seed_mean_sims(
    outputs_dir: str,
    *,
    arms: Optional[Sequence[str]] = None,
    label: str = "thesis",
    overwrite: bool = True,
) -> List[str]:
    """Zwraca listę zapisanych plików."""
    groups: Dict[tuple, Dict[int, str]] = defaultdict(dict)
    for path in glob.glob(os.path.join(outputs_dir, "seq_*_seed*.json.gz")):
        m = _NAME_RE.match(os.path.basename(path))
        if not m or m["label"] != label:
            continue
        arm = m["arm"]
        if arm == "R0" or (arms and arm not in arms):
            continue
        groups[(arm, m["tail"])][int(m["seed"])] = path
    written: List[str] = []
    for (arm, tail), by_seed in sorted(groups.items()):
        if len(by_seed) < 2:
            print(f"[seed-mean] pomijam {arm}/{tail}: tylko {len(by_seed)} ziarno")
            continue
        seeds = sorted(by_seed)
        out = os.path.join(outputs_dir, f"seq_{arm}_{label}_mean_{tail}.json.gz")
        if os.path.exists(out) and not overwrite:
            continue
        data = merge_group([by_seed[s] for s in seeds], arm, label, tail, seeds)
        _save(out, data)
        written.append(out)
        print(f"[seed-mean] {os.path.basename(out)} ({len(seeds)} ziarna)")
    return written
