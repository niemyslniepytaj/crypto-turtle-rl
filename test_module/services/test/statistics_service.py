"""
Thesis-oriented aggregation for walk-forward fold results.

Produces CSV / Markdown / JSON artifacts that map cleanly to tables in the thesis:
  - folds.csv              one row per (arm, fold)
  - summary_by_arm.csv     mean ± std over OOS folds
  - summary_by_arm.md      same as Markdown table
  - vs_baseline.csv        paired excess vs hodl (or chosen baseline)
  - run_results.json       full structured dump

Wilcoxon / bootstrap helpers are available for later multi-seed runs.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd


# Metrics shown in the main thesis comparison table
THESIS_METRICS = [
    "total_return",
    "excess_return",
    "hodl_return",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "cagr",
    "profit_factor",
    "win_rate",
    "turnover",
    "avg_transactions_per_day",
    "time_in_cash",
    "closed_trades_count",
    "transaction_count",
    "final_portfolio_value",
    "score",
]


class StatisticsService:
    """Agregacje statystyczne wyników symulacji."""
    def folds_to_dataframe(self, fold_records: Sequence[Dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for rec in fold_records:
            metrics = rec.get("metrics") or {}
            row = {
                "arm": rec.get("arm"),
                "plan_id": rec.get("plan_id"),
                "arm_label": rec.get("arm_label"),
                "fold_id": rec.get("fold_id"),
                "train_window": rec.get("train_window"),
                "eval_window": rec.get("eval_window"),
                "status": rec.get("status"),
                "n_ticks": rec.get("n_ticks"),
                "hodl_return": rec.get("hodl_return"),
                "excess_return": rec.get("excess_return"),
                "experiment_version": rec.get("experiment_version"),
                "sim_mode": rec.get("sim_mode"),
                "timestamp": rec.get("timestamp"),
            }
            for key in THESIS_METRICS:
                if key in ("hodl_return", "excess_return"):
                    continue
                row[key] = metrics.get(key)
            # Prefer top-level excess/hodl when present
            if row.get("excess_return") is None and metrics.get("total_return") is not None:
                hr = row.get("hodl_return")
                if hr is not None:
                    row["excess_return"] = float(metrics["total_return"]) - float(hr)
            rows.append(row)
        df = pd.DataFrame(rows)
        if not df.empty and "fold_id" in df.columns:
            df = df.sort_values(["arm", "fold_id"]).reset_index(drop=True)
        return df

    def summary_by_arm(
        self,
        folds_df: pd.DataFrame,
        metrics: Optional[Sequence[str]] = None,
        status_ok: str = "ok",
    ) -> pd.DataFrame:
        if folds_df is None or folds_df.empty:
            return pd.DataFrame()
        metrics = list(metrics or THESIS_METRICS)
        ok = folds_df[folds_df["status"] == status_ok].copy() if "status" in folds_df.columns else folds_df
        if ok.empty:
            return pd.DataFrame()

        records = []
        for arm, g in ok.groupby("arm", sort=True):
            row = {
                "arm": arm,
                "plan_id": g["plan_id"].iloc[0] if "plan_id" in g.columns else None,
                "arm_label": g["arm_label"].iloc[0] if "arm_label" in g.columns else None,
                "n_folds": int(len(g)),
                "n_folds_ok": int(len(g)),
            }
            for m in metrics:
                if m not in g.columns:
                    row[f"{m}_mean"] = None
                    row[f"{m}_std"] = None
                    row[f"{m}_median"] = None
                    continue
                series = pd.to_numeric(g[m], errors="coerce").dropna()
                if series.empty:
                    row[f"{m}_mean"] = None
                    row[f"{m}_std"] = None
                    row[f"{m}_median"] = None
                else:
                    row[f"{m}_mean"] = float(series.mean())
                    row[f"{m}_std"] = float(series.std(ddof=1)) if len(series) > 1 else 0.0
                    row[f"{m}_median"] = float(series.median())
            records.append(row)
        return pd.DataFrame(records)

    def vs_baseline(
        self,
        folds_df: pd.DataFrame,
        baseline_arm: str = "R0",
        value_col: str = "total_return",
    ) -> pd.DataFrame:
        """Paired per-fold differences vs baseline (same fold_id)."""
        if folds_df is None or folds_df.empty:
            return pd.DataFrame()
        ok = folds_df[folds_df["status"] == "ok"].copy()
        if baseline_arm not in set(ok["arm"]):
            return pd.DataFrame()
        base = ok[ok["arm"] == baseline_arm][["fold_id", value_col]].rename(
            columns={value_col: "baseline_value"}
        )
        rows = []
        for arm, g in ok[ok["arm"] != baseline_arm].groupby("arm"):
            merged = g.merge(base, on="fold_id", how="inner")
            if merged.empty:
                continue
            merged["diff"] = pd.to_numeric(merged[value_col], errors="coerce") - pd.to_numeric(
                merged["baseline_value"], errors="coerce"
            )
            for _, r in merged.iterrows():
                rows.append(
                    {
                        "arm": arm,
                        "baseline_arm": baseline_arm,
                        "fold_id": r["fold_id"],
                        "metric": value_col,
                        "value": r[value_col],
                        "baseline_value": r["baseline_value"],
                        "diff": r["diff"],
                    }
                )
        return pd.DataFrame(rows)

    def wilcoxon_vs_baseline(
        self,
        folds_df: pd.DataFrame,
        baseline_arm: str = "R0",
        value_col: str = "total_return",
    ) -> pd.DataFrame:
        """
        Paired Wilcoxon signed-rank (arm vs baseline) on per-fold values.
        Requires scipy; returns empty frame if unavailable or n < 3.
        """
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            return pd.DataFrame(
                [{"note": "scipy not installed — wilcoxon skipped"}]
            )

        diff_df = self.vs_baseline(folds_df, baseline_arm=baseline_arm, value_col=value_col)
        if diff_df.empty:
            return pd.DataFrame()

        rows = []
        for arm, g in diff_df.groupby("arm"):
            diffs = pd.to_numeric(g["diff"], errors="coerce").dropna()
            if len(diffs) < 3:
                rows.append(
                    {
                        "arm": arm,
                        "baseline_arm": baseline_arm,
                        "metric": value_col,
                        "n": int(len(diffs)),
                        "mean_diff": float(diffs.mean()) if len(diffs) else None,
                        "statistic": None,
                        "p_value": None,
                        "note": "need >= 3 paired folds",
                    }
                )
                continue
            # zero_method='wilcox' drops zeros; alternative='two-sided'
            try:
                stat, p = wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
            except ValueError as e:
                rows.append(
                    {
                        "arm": arm,
                        "baseline_arm": baseline_arm,
                        "metric": value_col,
                        "n": int(len(diffs)),
                        "mean_diff": float(diffs.mean()),
                        "statistic": None,
                        "p_value": None,
                        "note": str(e),
                    }
                )
                continue
            rows.append(
                {
                    "arm": arm,
                    "baseline_arm": baseline_arm,
                    "metric": value_col,
                    "n": int(len(diffs)),
                    "mean_diff": float(diffs.mean()),
                    "statistic": float(stat),
                    "p_value": float(p),
                    "note": "",
                }
            )
        return pd.DataFrame(rows)

    def bootstrap_mean_ci(
        self,
        values: Iterable[float],
        n_boot: int = 2000,
        alpha: float = 0.05,
        seed: int = 0,
    ) -> Dict[str, Optional[float]]:
        import numpy as np

        arr = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=float)
        if arr.size == 0:
            return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
        rng = np.random.default_rng(seed)
        means = []
        for _ in range(int(n_boot)):
            sample = rng.choice(arr, size=arr.size, replace=True)
            means.append(float(sample.mean()))
        means = np.sort(np.asarray(means))
        lo = float(np.quantile(means, alpha / 2))
        hi = float(np.quantile(means, 1 - alpha / 2))
        return {"mean": float(arr.mean()), "ci_low": lo, "ci_high": hi, "n": int(arr.size)}

    def to_markdown_table(
        self,
        summary_df: pd.DataFrame,
        metrics: Optional[Sequence[str]] = None,
    ) -> str:
        """Compact mean±std Markdown table for the thesis draft."""
        if summary_df is None or summary_df.empty:
            return "_Brak wyników do tabeli._\n"

        metrics = list(metrics or [
            "total_return",
            "excess_return",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
            "avg_transactions_per_day",
        ])

        headers = ["Ramię", "plan", "n"]
        for m in metrics:
            headers.append(m)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, r in summary_df.iterrows():
            cells = [
                str(r.get("arm_label") or r.get("arm")),
                str(r.get("plan_id") or ""),
                str(int(r.get("n_folds") or 0)),
            ]
            for m in metrics:
                mean = r.get(f"{m}_mean")
                std = r.get(f"{m}_std")
                cells.append(self._fmt_mean_std(mean, std))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _fmt_mean_std(mean, std) -> str:
        if mean is None or (isinstance(mean, float) and not math.isfinite(mean)):
            return "—"
        if std is None or (isinstance(std, float) and not math.isfinite(std)):
            return f"{mean:.2f}"
        return f"{mean:.2f}±{std:.2f}"

    def build_and_save(
        self,
        *,
        fold_records: Sequence[Dict[str, Any]],
        run_dir: str,
        manifest: Optional[Dict[str, Any]] = None,
        baseline_arm: str = "R0",
    ) -> Dict[str, Any]:
        os.makedirs(run_dir, exist_ok=True)
        folds_df = self.folds_to_dataframe(fold_records)
        # Legacy runs used arm=hodl — fall back if R0 absent
        if (
            not folds_df.empty
            and baseline_arm == "R0"
            and "arm" in folds_df.columns
            and "R0" not in set(folds_df["arm"])
            and "hodl" in set(folds_df["arm"])
        ):
            baseline_arm = "hodl"
        summary_df = self.summary_by_arm(folds_df)
        vs_df = self.vs_baseline(folds_df, baseline_arm=baseline_arm, value_col="total_return")
        vs_excess_df = self.vs_baseline(folds_df, baseline_arm=baseline_arm, value_col="excess_return")
        wilcoxon_df = self.wilcoxon_vs_baseline(
            folds_df, baseline_arm=baseline_arm, value_col="total_return"
        )

        folds_csv = os.path.join(run_dir, "folds.csv")
        summary_csv = os.path.join(run_dir, "summary_by_arm.csv")
        vs_csv = os.path.join(run_dir, "vs_baseline.csv")
        wilcoxon_csv = os.path.join(run_dir, "wilcoxon_vs_baseline.csv")
        md_path = os.path.join(run_dir, "summary_by_arm.md")
        json_path = os.path.join(run_dir, "run_results.json")

        if not folds_df.empty:
            folds_df.to_csv(folds_csv, index=False)
        if not summary_df.empty:
            summary_df.to_csv(summary_csv, index=False)
        if not vs_df.empty:
            vs_df.to_csv(vs_csv, index=False)
        if not wilcoxon_df.empty:
            wilcoxon_df.to_csv(wilcoxon_csv, index=False)

        md = self.to_markdown_table(summary_df)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Walk-forward — tabela zbiorcza\n\n")
            f.write(f"Wygenerowano: {datetime.now().isoformat()}\n\n")
            if manifest:
                f.write(f"Run: `{manifest.get('run_id')}`\n\n")
                f.write(f"Folds: {', '.join(manifest.get('eval_folds') or [])}\n\n")
            f.write(md)
            f.write("\n## Uwagi\n\n")
            f.write("- Wartości: średnia ± odchylenie std po foldach OOS (`status=ok`).\n")
            f.write("- `excess_return` = total_return strategii − HODL w tym samym foldzie.\n")
            f.write("- Surowy dump: `folds.csv`, `run_results.json`.\n")

        payload = {
            "manifest": manifest or {},
            "folds": fold_records,
            "summary_by_arm": summary_df.to_dict(orient="records") if not summary_df.empty else [],
            "vs_baseline": vs_df.to_dict(orient="records") if not vs_df.empty else [],
            "vs_baseline_excess": vs_excess_df.to_dict(orient="records") if not vs_excess_df.empty else [],
            "wilcoxon_vs_baseline": wilcoxon_df.to_dict(orient="records") if not wilcoxon_df.empty else [],
            "artifacts": {
                "folds_csv": folds_csv if not folds_df.empty else None,
                "summary_csv": summary_csv if not summary_df.empty else None,
                "summary_md": md_path,
                "vs_baseline_csv": vs_csv if not vs_df.empty else None,
                "wilcoxon_csv": wilcoxon_csv if not wilcoxon_df.empty else None,
            },
            "generated_at": datetime.now().isoformat(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return payload
