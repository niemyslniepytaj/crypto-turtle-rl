"""Kontrola kompletności wyników po sekwencji (I4.1)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from rl_module.services.thesis import console as thesis_console
from rl_module.services.thesis.compare import collect_folds, discover_run_dirs


def completeness_table(
    run_dirs: Sequence[str],
    *,
    expected_arms: Optional[Sequence[str]] = None,
    expected_seeds: Optional[Sequence[int]] = None,
    expected_folds: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    df = collect_folds(run_dirs)
    if df.empty:
        return pd.DataFrame(
            columns=["arm", "seed", "portfolio_mode", "fold_id", "status", "note"]
        )

    arms = expected_arms or sorted(df["arm"].dropna().unique())
    seeds = expected_seeds or sorted(df["seed"].dropna().unique().astype(int))
    folds = expected_folds or sorted(df["fold_id"].dropna().unique())
    modes = sorted(df["portfolio_mode"].dropna().unique()) if "portfolio_mode" in df.columns else ["reset"]

    rows = []
    for mode in modes:
        sub = df[df["portfolio_mode"] == mode] if "portfolio_mode" in df.columns else df
        for arm in arms:
            # R0 (HODL) nie jest botem - deterministyczny, jedno uruchomienie (jeden seed)
            arm_seeds = seeds
            if arm == "R0":
                _r0 = sorted(sub[sub["arm"] == "R0"]["seed"].dropna().astype(int).unique())
                arm_seeds = list(_r0[:1]) or list(seeds[:1])
            for seed in arm_seeds:
                for fid in folds:
                    hit = sub[
                        (sub["arm"] == arm)
                        & (sub["seed"].astype(int) == int(seed))
                        & (sub["fold_id"].astype(str) == str(fid))
                    ]
                    if hit.empty:
                        rows.append(
                            {
                                "arm": arm,
                                "seed": int(seed),
                                "portfolio_mode": mode,
                                "fold_id": fid,
                                "status": "missing",
                                "note": "",
                            }
                        )
                    else:
                        st = str(hit["status"].iloc[0])
                        rows.append(
                            {
                                "arm": arm,
                                "seed": int(seed),
                                "portfolio_mode": mode,
                                "fold_id": fid,
                                "status": st,
                                "note": "" if st == "ok" else "failed_or_skip",
                            }
                        )
    return pd.DataFrame(rows)


def print_completeness_summary(table: pd.DataFrame) -> None:
    if table.empty:
        thesis_console.print_kv("Completeness", "brak danych", kind="warn")
        return
    n = len(table)
    n_ok = int((table["status"] == "ok").sum())
    n_miss = int((table["status"] == "missing").sum())
    n_fail = int(((table["status"] != "ok") & (table["status"] != "missing")).sum())
    thesis_console.print_card(
        "Kompletność wyników",
        [
            ("wierszy", n),
            ("ok", n_ok),
            ("missing", n_miss),
            ("failed/other", n_fail),
        ],
        tag="Completeness",
        ok=(n_miss == 0 and n_fail == 0),
    )
    bad = table[table["status"] != "ok"]
    for _, row in bad.head(30).iterrows():
        thesis_console.print_kv(
            "Brak/Fail",
            f"{row['arm']} seed={row['seed']} {row['portfolio_mode']} {row['fold_id']} → {row['status']}",
            kind="warn",
        )


def run_completeness_check(
    results_root: str,
    out_csv: Optional[str] = None,
    *,
    run_label: Optional[str] = None,
    portfolio_mode: Optional[str] = None,
) -> pd.DataFrame:
    dirs = discover_run_dirs(results_root, run_label=run_label, portfolio_mode=portfolio_mode)
    table = completeness_table(dirs)
    print_completeness_summary(table)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        table.to_csv(out_csv, index=False)
        thesis_console.print_kv("Completeness", f"zapisano {out_csv}", kind="ok")
    return table
