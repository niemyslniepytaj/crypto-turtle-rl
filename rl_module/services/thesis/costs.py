"""Koszt adaptacji w godzinach rdzeniowych (Z3)."""

from __future__ import annotations

from typing import Any, Dict, Optional


def core_hours(wall_s: float, cores: float) -> float:
    return float(wall_s or 0.0) * float(cores or 0.0) / 3600.0


def attach_adapt_costs(art: Dict[str, Any], *, arm: str) -> Dict[str, Any]:
    """
    Uzupełnia artifacts o *_core_hours i sumy adapt_*.
    M1: Optuna; M2: douczanie; M3: douczanie+UL; R*: 0.
    """
    art = art if art is not None else {}
    opt_s = float(art.get("optuna_time_s") or 0.0)
    train_s = float(art.get("train_time_s") or 0.0)
    ul_s = float(art.get("ul_fit_time_s") or 0.0)
    opt_c = float(art.get("optuna_cores") or art.get("n_jobs") or 1.0)
    train_c = float(art.get("train_cores") or art.get("num_envs") or 1.0)
    ul_c = float(art.get("ul_cores") or 1.0)

    art["optuna_cores"] = opt_c
    art["train_cores"] = train_c
    art["ul_cores"] = ul_c
    art["optuna_core_hours"] = round(core_hours(opt_s, opt_c), 6)
    art["train_core_hours"] = round(core_hours(train_s, train_c), 6)
    art["ul_core_hours"] = round(core_hours(ul_s, ul_c), 6)

    if arm in ("R0", "R1", "R2", "R3"):
        art["adapt_wall_time_s"] = 0.0
        art["adapt_core_hours"] = 0.0
        art["adaptation_cost"] = 0
    elif arm == "M1":
        art["adapt_wall_time_s"] = round(opt_s, 2)
        art["adapt_core_hours"] = art["optuna_core_hours"]
        art["adaptation_cost"] = 1
    elif arm == "M2":
        art["adapt_wall_time_s"] = round(train_s, 2)
        art["adapt_core_hours"] = art["train_core_hours"]
        art["adaptation_cost"] = 1
    elif arm == "M3":
        art["adapt_wall_time_s"] = round(train_s + ul_s, 2)
        art["adapt_core_hours"] = round(
            art["train_core_hours"] + art["ul_core_hours"], 6
        )
        art["adaptation_cost"] = 1
    else:
        art["adapt_wall_time_s"] = round(opt_s + train_s + ul_s, 2)
        art["adapt_core_hours"] = round(
            art["optuna_core_hours"] + art["train_core_hours"] + art["ul_core_hours"], 6
        )
    return art
