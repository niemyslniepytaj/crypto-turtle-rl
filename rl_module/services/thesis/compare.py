"""
Agregacja wielu run_dir (ramiona × seedy) → jedna tabela porównawcza.
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from rl_module.services.thesis import console as thesis_console

PAIRED = (
    ("M1", "R1", "M1_minus_R1"),  # Delta_MP
    ("M2", "R2", "M2_minus_R2"),  # Delta_RL
    ("M3", "R3", "M3_minus_R3"),  # Delta_UL
    ("M1", "M2", "M1_minus_M2"),
    ("M3", "M2", "M3_minus_M2"),
    ("R1", "R0", "R1_minus_R0"),
    ("R2", "R0", "R2_minus_R0"),
    ("R3", "R0", "R3_minus_R0"),
    ("M1", "R0", "M1_minus_R0"),
    ("M2", "R0", "M2_minus_R0"),
    ("M3", "R0", "M3_minus_R0"),
)


def _load_folds_from_run(run_dir: str) -> List[Dict[str, Any]]:
    rows = []
    windows_dir = os.path.join(run_dir, "windows")
    if not os.path.isdir(windows_dir):
        return rows
    manifest = {}
    mp = os.path.join(run_dir, "manifest.json")
    if os.path.isfile(mp):
        with open(mp, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    arm = manifest.get("arm")
    seed = manifest.get("seed")
    run_label = manifest.get("run_label")
    portfolio_mode = manifest.get("portfolio_mode")
    for name in sorted(os.listdir(windows_dir)):
        if not (name.startswith("fold_") and name.endswith(".json")):
            continue
        path = os.path.join(windows_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        metrics = rec.get("metrics") or {}
        art = rec.get("artifacts") or {}
        rows.append(
            {
                "run_dir": run_dir,
                "run_id": os.path.basename(run_dir),
                "arm": rec.get("arm") or arm,
                "seed": seed,
                "run_label": run_label,
                "portfolio_mode": portfolio_mode,
                "fold_id": rec.get("fold_id"),
                "status": rec.get("status"),
                "total_return": metrics.get("total_return"),
                "excess_return": rec.get("excess_return"),
                "hodl_return": rec.get("hodl_return"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "max_drawdown": metrics.get("max_drawdown"),
                "sortino_ratio": metrics.get("sortino_ratio"),
                "profit_factor": metrics.get("profit_factor"),
                "win_rate": metrics.get("win_rate"),
                "transaction_count": metrics.get("transaction_count"),
                "avg_transactions_per_day": metrics.get("avg_transactions_per_day"),
                "turnover": metrics.get("turnover"),
                "time_in_cash": metrics.get("time_in_cash"),
                "final_portfolio_value": metrics.get("final_portfolio_value"),
                "wall_time_s": art.get("wall_time_s"),
                "optuna_trials": art.get("optuna_trials"),
                "continue_timesteps": art.get("continue_timesteps"),
                "adapt_core_hours": art.get("adapt_core_hours"),
                "adapt_wall_time_s": art.get("adapt_wall_time_s"),
                "ul_fit_time_s": art.get("ul_fit_time_s"),
                "train_time_s": art.get("train_time_s"),
                "optuna_time_s": art.get("optuna_time_s"),
                "env_steps": art.get("env_steps"),
            }
        )
    return rows


def collect_folds(run_dirs: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rd in run_dirs:
        if not os.path.isdir(rd):
            thesis_console.print_kv("Compare", f"brak katalogu: {rd}", kind="warn")
            continue
        rows.extend(_load_folds_from_run(rd))
    df = pd.DataFrame(rows)
    if not df.empty and "fold_id" in df.columns:
        df = df.sort_values(["portfolio_mode", "seed", "arm", "fold_id"]).reset_index(drop=True)
    return df


def filter_b_folds(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["fold_id"].astype(str).str.startswith("B")].copy()


def summary_by_arm_seed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    metrics = ["total_return", "excess_return", "sharpe_ratio", "max_drawdown", "wall_time_s"]
    records = []
    for (arm, seed, mode), g in ok.groupby(["arm", "seed", "portfolio_mode"], dropna=False):
        row = {"arm": arm, "seed": seed, "portfolio_mode": mode, "n_folds": len(g)}
        for m in metrics:
            if m not in g.columns:
                continue
            s = pd.to_numeric(g[m], errors="coerce").dropna()
            row[f"{m}_mean"] = float(s.mean()) if len(s) else None
            row[f"{m}_std"] = float(s.std(ddof=1)) if len(s) > 1 else (0.0 if len(s) else None)
        records.append(row)
    return pd.DataFrame(records)


def summary_by_arm(df: pd.DataFrame) -> pd.DataFrame:
    """Średnia meanów po seedach."""
    s = summary_by_arm_seed(df)
    if s.empty:
        return s
    metrics = [c for c in s.columns if c.endswith("_mean")]
    records = []
    for (arm, mode), g in s.groupby(["arm", "portfolio_mode"], dropna=False):
        row = {"arm": arm, "portfolio_mode": mode, "n_seeds": len(g)}
        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce").dropna()
            row[m.replace("_mean", "_across_seeds_mean")] = float(vals.mean()) if len(vals) else None
            row[m.replace("_mean", "_across_seeds_median")] = float(vals.median()) if len(vals) else None
        records.append(row)
    return pd.DataFrame(records)


def paired_deltas(df: pd.DataFrame, metric: str = "excess_return") -> pd.DataFrame:
    """Paired Δ per fold×seed×mode dla par z PAIRED."""
    if df.empty:
        return pd.DataFrame()
    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
    rows = []
    for mode, gmode in ok.groupby("portfolio_mode", dropna=False):
        for seed, gseed in gmode.groupby("seed", dropna=False):
            pivot = gseed.pivot_table(
                index="fold_id", columns="arm", values=metric, aggfunc="first"
            )
            for left, right, name in PAIRED:
                if left not in pivot.columns or right not in pivot.columns:
                    continue
                delta = pivot[left] - pivot[right]
                for fold_id, val in delta.items():
                    if pd.isna(val):
                        continue
                    rows.append(
                        {
                            "portfolio_mode": mode,
                            "seed": seed,
                            "fold_id": fold_id,
                            "pair": name,
                            "metric": metric,
                            "delta": float(val),
                            "left_arm": left,
                            "right_arm": right,
                        }
                    )
    return pd.DataFrame(rows)


def _to_markdown_table(df: pd.DataFrame, max_cols: int = 12) -> str:
    if df is None or df.empty:
        return "_(brak danych)_\n"
    cols = list(df.columns)[:max_cols]
    sub = df[cols].copy()
    for c in sub.columns:
        if pd.api.types.is_float_dtype(sub[c]):
            sub[c] = sub[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    return sub.to_markdown(index=False) + "\n"


def compare_runs(
    run_dirs: Sequence[str],
    out_dir: str,
    *,
    only_b_folds: bool = True,
    label: str = "compare",
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    folds = collect_folds(run_dirs)
    if only_b_folds:
        folds = filter_b_folds(folds)

    folds_csv = os.path.join(out_dir, "folds_all.csv")
    summary_seed_csv = os.path.join(out_dir, "summary_arm_seed.csv")
    summary_arm_csv = os.path.join(out_dir, "summary_arm.csv")
    paired_csv = os.path.join(out_dir, "paired_deltas.csv")
    md_path = os.path.join(out_dir, "compare.md")

    s_seed = summary_by_arm_seed(folds)
    s_arm = summary_by_arm(folds)
    paired = paired_deltas(folds, metric="excess_return")

    if not folds.empty:
        folds.to_csv(folds_csv, index=False)
    if not s_seed.empty:
        s_seed.to_csv(summary_seed_csv, index=False)
    if not s_arm.empty:
        s_arm.to_csv(summary_arm_csv, index=False)
    if not paired.empty:
        paired.to_csv(paired_csv, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Porównanie ramion — {label}\n\n")
        f.write(f"Wygenerowano: {datetime.now().isoformat()}\n\n")
        f.write(f"Run dirs: {len(run_dirs)}; only_B={only_b_folds}\n\n")
        f.write("## Summary by arm (across seeds)\n\n")
        f.write(_to_markdown_table(s_arm))
        f.write("\n## Summary by arm × seed\n\n")
        f.write(_to_markdown_table(s_seed))
        f.write("\n## Paired deltas (excess_return)\n\n")
        if not paired.empty:
            agg = (
                paired.groupby(["portfolio_mode", "pair"])["delta"]
                .agg(["mean", "median", "count"])
                .reset_index()
            )
            f.write(_to_markdown_table(agg))
        else:
            f.write("_(brak sparowanych foldów)_\n")

    thesis_console.print_card(
        "Agregacja zakończona",
        [
            ("out_dir", out_dir),
            ("n_rows", len(folds)),
            ("folds_csv", folds_csv if not folds.empty else "—"),
            ("compare.md", md_path),
        ],
        tag="Compare",
        ok=True,
    )
    return {
        "out_dir": out_dir,
        "folds_csv": folds_csv if not folds.empty else None,
        "summary_arm_seed_csv": summary_seed_csv if not s_seed.empty else None,
        "summary_arm_csv": summary_arm_csv if not s_arm.empty else None,
        "paired_deltas_csv": paired_csv if not paired.empty else None,
        "compare_md": md_path,
        "n_folds": int(len(folds)),
    }


def discover_run_dirs(
    results_root: str,
    *,
    run_label: Optional[str] = None,
    portfolio_mode: Optional[str] = None,
) -> List[str]:
    """Szuka katalogów z manifest.json pod thesis_results/."""
    if not os.path.isdir(results_root):
        return []
    found = []
    for path in glob.glob(os.path.join(results_root, "*")):
        if not os.path.isdir(path):
            continue
        if os.path.basename(path).startswith("_"):
            continue
        mf = os.path.join(path, "manifest.json")
        if not os.path.isfile(mf):
            continue
        with open(mf, "r", encoding="utf-8") as f:
            man = json.load(f)
        if run_label and man.get("run_label") != run_label:
            continue
        if portfolio_mode and man.get("portfolio_mode") != portfolio_mode:
            continue
        found.append(path)
    return sorted(found)
