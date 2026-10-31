"""
Analiza statystyczna tezy (4.1 / 4.4 / 4.5) — Wilcoxon na R, Holm, Kerby r_rb.
Wołane z plgrid_thesis_compare; compare.py zostaje do zbiorczych CSV.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from rl_module.services.thesis import console as thesis_console
from rl_module.services.thesis.compare import (
    PAIRED,
    collect_folds,
    discover_run_dirs,
    filter_b_folds,
)
from rl_module.services.thesis.hypotheses import (
    ALPHA,
    BEST_METHOD_RULE,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    METHOD_PAIRS,
    TREND_THRESHOLD,
    get_hypothesis_set,
    method_delta_arms,
)


def fold_sort_key(fid: Any) -> Tuple[str, int]:
    """Naturalny porz?dek okien: A1, A2, B1, B2, ..., B10 (nie leksykograficzny)."""
    m = re.match(r"^([A-Za-z]+)(\d+)$", str(fid))
    return (m.group(1), int(m.group(2))) if m else (str(fid), 0)


def aggregate_seeds(
    df: pd.DataFrame,
    *,
    metric: str = "total_return",
) -> pd.DataFrame:
    """
    Ramię × okno × portfolio_mode = średnia z ziaren (+ std).
    R0: zwykle 1 wartość — średnia i tak OK.
    """
    if df.empty:
        return pd.DataFrame()
    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
    rows = []
    keys = ["arm", "fold_id", "portfolio_mode"]
    for key, g in ok.groupby(keys, dropna=False):
        arm, fold_id, mode = key
        vals = pd.to_numeric(g[metric], errors="coerce").dropna()
        row = {
            "arm": arm,
            "fold_id": fold_id,
            "portfolio_mode": mode,
            "n_seeds": int(len(vals)),
            f"{metric}_mean": float(vals.mean()) if len(vals) else np.nan,
            f"{metric}_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            f"{metric}_median": float(vals.median()) if len(vals) else np.nan,
        }
        if "excess_return" in g.columns:
            ex = pd.to_numeric(g["excess_return"], errors="coerce").dropna()
            row["excess_return_mean"] = float(ex.mean()) if len(ex) else np.nan
            row["excess_return_std"] = float(ex.std(ddof=1)) if len(ex) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def arm_summary(agg: pd.DataFrame, *, metric: str = "total_return") -> pd.DataFrame:
    """Średnia/mediana z B1–B10; H5: # okien > HODL; średnia nadwyżka."""
    if agg.empty:
        return pd.DataFrame()
    col = f"{metric}_mean"
    b = agg[agg["fold_id"].astype(str).str.startswith("B")].copy()
    rows = []
    for (arm, mode), g in b.groupby(["arm", "portfolio_mode"], dropna=False):
        vals = pd.to_numeric(g[col], errors="coerce").dropna()
        excess = pd.to_numeric(g.get("excess_return_mean"), errors="coerce").dropna()
        # H5 vs R0 na tych samych foldach
        n_beat_hodl = np.nan
        if "R0" in set(b["arm"]):
            r0 = b[(b["arm"] == "R0") & (b["portfolio_mode"] == mode)].set_index("fold_id")
            beat = 0
            n = 0
            for _, row in g.iterrows():
                fid = row["fold_id"]
                if fid not in r0.index:
                    continue
                n += 1
                if float(row[col]) > float(r0.loc[fid, col]):
                    beat += 1
            n_beat_hodl = beat if n else np.nan
        rows.append(
            {
                "arm": arm,
                "portfolio_mode": mode,
                "n_folds": int(len(vals)),
                f"{metric}_mean": float(vals.mean()) if len(vals) else np.nan,
                f"{metric}_median": float(vals.median()) if len(vals) else np.nan,
                "excess_mean_pp": float(excess.mean()) if len(excess) else np.nan,
                "n_windows_beat_hodl": n_beat_hodl,
            }
        )
    return pd.DataFrame(rows)


def paired_differences(
    agg: pd.DataFrame,
    *,
    metric: str = "total_return",
    arm_x: str,
    arm_y: str,
    portfolio_mode: Optional[str] = None,
) -> pd.DataFrame:
    """di = R_x - R_y per fold (średnie po seedach)."""
    col = f"{metric}_mean"
    d = agg.copy()
    if portfolio_mode is not None:
        d = d[d["portfolio_mode"] == portfolio_mode]
    bx = d[d["arm"] == arm_x].set_index("fold_id")
    by = d[d["arm"] == arm_y].set_index("fold_id")
    folds = sorted(set(bx.index) & set(by.index), key=fold_sort_key)
    rows = []
    for fid in folds:
        rx = float(bx.loc[fid, col])
        ry = float(by.loc[fid, col])
        rows.append(
            {
                "fold_id": fid,
                "arm_x": arm_x,
                "arm_y": arm_y,
                "R_x": rx,
                "R_y": ry,
                "d_i": rx - ry,
                "portfolio_mode": portfolio_mode or bx.loc[fid].get("portfolio_mode"),
            }
        )
    return pd.DataFrame(rows)


def wilcoxon_signed_rank(d: np.ndarray) -> Dict[str, Any]:
    """
    Wilcoxon dwustronny; di=0 wykluczone z rang (zero_method='wilcox').
    Kerby matched-pairs rank-biserial: r = (W+ - W-) / (W+ + W-).
    """
    from scipy.stats import wilcoxon

    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    n_all = int(len(d))
    n_zero = int(np.sum(d == 0))
    d_nz = d[d != 0]
    n_nonzero = int(len(d_nz))
    if n_nonzero < 1:
        return {
            "n_all": n_all,
            "n_zero": n_zero,
            "n_nonzero": n_nonzero,
            "W_plus": np.nan,
            "W_minus": np.nan,
            "statistic": np.nan,
            "p_value": np.nan,
            "r_rb": np.nan,
            "median_d": float(np.median(d)) if n_all else np.nan,
            "n_positive": int(np.sum(d > 0)),
            "zero_method": "wilcox",
        }
    # scipy: statistic = min(W+, W-) by default (alternative two-sided)
    res = wilcoxon(d_nz, zero_method="wilcox", alternative="two-sided", method="auto")
    # Kerby: policz W+ / W- z rang
    abs_d = np.abs(d_nz)
    ranks = pd.Series(abs_d).rank(method="average").to_numpy()
    w_plus = float(np.sum(ranks[d_nz > 0]))
    w_minus = float(np.sum(ranks[d_nz < 0]))
    denom = w_plus + w_minus
    r_rb = float((w_plus - w_minus) / denom) if denom > 0 else np.nan
    return {
        "n_all": n_all,
        "n_zero": n_zero,
        "n_nonzero": n_nonzero,
        "W_plus": w_plus,
        "W_minus": w_minus,
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "r_rb": r_rb,
        "median_d": float(np.median(d_nz)),
        "n_positive": int(np.sum(d > 0)),
        "zero_method": "wilcox",
    }


def holm_adjust(p_by_id: Dict[str, float], family: Sequence[str]) -> Dict[str, float]:
    """Holm–Bonferroni w rodzinie; brakujące id pomijane."""
    items = [(hid, p_by_id[hid]) for hid in family if hid in p_by_id and np.isfinite(p_by_id[hid])]
    m = len(items)
    if m == 0:
        return {}
    items.sort(key=lambda x: x[1])
    out: Dict[str, float] = {}
    prev = 0.0
    for i, (hid, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        adj = max(adj, prev)  # monotoniczność Holma
        out[hid] = float(adj)
        prev = adj
    return out


def bootstrap_median_ci(
    d: np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Percentylowy 95% CI mediany (po oknach). Zwraca (median, lo, hi)."""
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    med = float(np.median(d))
    rng = np.random.RandomState(int(seed))
    boots = np.empty(int(n_boot), dtype=np.float64)
    n = len(d)
    for i in range(int(n_boot)):
        sample = d[rng.randint(0, n, size=n)]
        boots[i] = np.median(sample)
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return med, lo, hi


def method_deltas_table(
    agg: pd.DataFrame,
    *,
    portfolio_mode: str = "reset",
    metric: str = "total_return",
) -> pd.DataFrame:
    """Delta_i = R_M,i − R_R,i per metoda × okno."""
    rows = []
    for mid, ax, ay in METHOD_PAIRS:
        diffs = paired_differences(
            agg, metric=metric, arm_x=ax, arm_y=ay, portfolio_mode=portfolio_mode,
        )
        if diffs.empty:
            continue
        for _, r in diffs.iterrows():
            rows.append(
                {
                    "method": mid,
                    "arm_x": ax,
                    "arm_y": ay,
                    "fold_id": r["fold_id"],
                    "Delta": r["d_i"],
                    "R_x": r["R_x"],
                    "R_y": r["R_y"],
                    "portfolio_mode": portfolio_mode,
                }
            )
    return pd.DataFrame(rows)


def friedman_on_deltas(delta_wide: pd.DataFrame) -> Dict[str, Any]:
    """Friedman + Kendall W na kolumnach metod (wiersze = okna)."""
    from scipy.stats import friedmanchisquare

    cols = [c for c in delta_wide.columns if c != "fold_id"]
    if len(cols) < 2 or len(delta_wide) < 2:
        return {"status": "insufficient", "p_value": np.nan, "kendall_w": np.nan}
    arrays = [delta_wide[c].to_numpy(dtype=float) for c in cols]
    if any(np.isnan(a).any() for a in arrays):
        # drop rows with any nan
        mask = np.ones(len(delta_wide), dtype=bool)
        for a in arrays:
            mask &= np.isfinite(a)
        arrays = [a[mask] for a in arrays]
        if len(arrays[0]) < 2:
            return {"status": "insufficient", "p_value": np.nan, "kendall_w": np.nan}
    stat, p = friedmanchisquare(*arrays)
    # Kendall W: rangi metod liczone w obr?bie ka?dego okna (wiersza), sumy rang po metodach
    from scipy.stats import rankdata

    n, k = len(arrays[0]), len(arrays)
    mat = np.vstack(arrays).T  # (okna x metody)
    ranks = np.vstack([rankdata(row) for row in mat])
    rank_sums = ranks.sum(axis=0)
    ss = float(np.sum((rank_sums - rank_sums.mean()) ** 2))
    w = (12.0 * ss) / (n ** 2 * (k ** 3 - k)) if n > 0 and k > 1 else float("nan")
    return {
        "status": "ok",
        "statistic": float(stat),
        "p_value": float(p),
        "kendall_w": float(w),
        "methods": cols,
        "n_folds": int(len(arrays[0])),
    }


def run_hypothesis_tests(
    agg: pd.DataFrame,
    *,
    portfolio_mode: str = "reset",
    metric: str = "total_return",
    hypothesis_set: Optional[str] = None,
    costs_by_method: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Zwraca (tests_df, diffs_long, warnings, extras).
    extras: friedman, efficiency, hodl_h5, ...
    """
    cfg = get_hypothesis_set(hypothesis_set)
    arms = set(agg["arm"].unique()) if not agg.empty else set()
    warnings_list: List[str] = []
    test_rows: List[Dict[str, Any]] = []
    diff_frames: List[pd.DataFrame] = []
    extras: Dict[str, Any] = {"hypothesis_set": hypothesis_set or "intermediate"}
    p_raw: Dict[str, float] = {}
    deltas = method_deltas_table(agg, portfolio_mode=portfolio_mode, metric=metric)

    for spec in cfg["tests"]:
        hid = spec["id"]
        kind = spec["kind"]
        if kind == "wilcoxon_pair":
            ax, ay = spec["arm_x"], spec["arm_y"]
            req = spec.get("requires") or (ax, ay)
            if any(a not in arms for a in req):
                msg = f"{hid}: brak ramion {req} — pomijam"
                warnings_list.append(msg)
                thesis_console.print_kv("Analysis", msg, kind="warn")
                test_rows.append(
                    {
                        "hypothesis": hid,
                        "kind": kind,
                        "arm_x": ax,
                        "arm_y": ay,
                        "status": "skipped_missing_arms",
                        "p_value": np.nan,
                    }
                )
                continue
            diffs = paired_differences(
                agg, metric=metric, arm_x=ax, arm_y=ay, portfolio_mode=portfolio_mode,
            )
            diffs["hypothesis"] = hid
            diffs["delta_name"] = spec.get("delta_name")
            diff_frames.append(diffs)
            d = diffs["d_i"].to_numpy() if not diffs.empty else np.array([])
            stats = wilcoxon_signed_rank(d)
            med, lo, hi = bootstrap_median_ci(d)
            p_raw[hid] = stats["p_value"]
            test_rows.append(
                {
                    "hypothesis": hid,
                    "kind": kind,
                    "arm_x": ax,
                    "arm_y": ay,
                    "delta_name": spec.get("delta_name"),
                    "portfolio_mode": portfolio_mode,
                    "status": "ok",
                    **stats,
                    "median_d_ci_lo": lo,
                    "median_d_ci_hi": hi,
                    "median_d_boot": med,
                    "decision_raw": (
                        "reject"
                        if np.isfinite(stats["p_value"]) and stats["p_value"] < ALPHA
                        else "fail_to_reject"
                    ),
                    "holm_family": spec.get("holm_family"),
                }
            )
        elif kind == "friedman_deltas":
            methods = list(spec.get("methods") or ())
            present = []
            for mid in methods:
                ax, ay = method_delta_arms(mid)
                if ax in arms and ay in arms:
                    present.append(mid)
            if len(present) < 2:
                msg = f"{hid}: za mało metod do Friedmana ({present}) — pomijam"
                warnings_list.append(msg)
                thesis_console.print_kv("Analysis", msg, kind="warn")
                test_rows.append(
                    {"hypothesis": hid, "kind": kind, "status": "skipped_missing_arms"}
                )
                continue
            if "UL" not in present and spec.get("fallback_without_ul"):
                # Wilcoxon na Delta^RL − Delta^MP
                msg = f"{hid}: brak UL — fallback Wilcoxon Delta_RL−Delta_MP"
                warnings_list.append(msg)
                thesis_console.print_kv("Analysis", msg, kind="warn")
                wide = deltas.pivot_table(
                    index="fold_id", columns="method", values="Delta", aggfunc="mean"
                )
                if "RL" in wide.columns and "MP" in wide.columns:
                    d = (wide["RL"] - wide["MP"]).to_numpy()
                    stats = wilcoxon_signed_rank(d)
                    med, lo, hi = bootstrap_median_ci(d)
                    test_rows.append(
                        {
                            "hypothesis": hid,
                            "kind": "wilcoxon_delta_diff_fallback",
                            "status": "ok",
                            **stats,
                            "median_d_ci_lo": lo,
                            "median_d_ci_hi": hi,
                            "decision_raw": (
                                "reject"
                                if np.isfinite(stats["p_value"]) and stats["p_value"] < ALPHA
                                else "fail_to_reject"
                            ),
                        }
                    )
                continue
            wide = deltas.pivot_table(
                index="fold_id", columns="method", values="Delta", aggfunc="mean"
            )
            wide = wide[[c for c in present if c in wide.columns]].dropna()
            wide = wide.reset_index()
            fr = friedman_on_deltas(wide)
            extras["friedman"] = fr
            row = {
                "hypothesis": hid,
                "kind": kind,
                "status": fr.get("status"),
                "p_value": fr.get("p_value"),
                "statistic": fr.get("statistic"),
                "kendall_w": fr.get("kendall_w"),
                "decision_raw": (
                    "reject"
                    if np.isfinite(fr.get("p_value", np.nan)) and fr["p_value"] < ALPHA
                    else "fail_to_reject"
                ),
            }
            # post hoc Wilcoxon par z Holmem tylko po odrzuceniu
            if (
                spec.get("holm_posthoc")
                and row["decision_raw"] == "reject"
                and len(present) >= 2
            ):
                post_p = {}
                for i in range(len(present)):
                    for j in range(i + 1, len(present)):
                        a, b = present[i], present[j]
                        d = (wide[a] - wide[b]).to_numpy()
                        st = wilcoxon_signed_rank(d)
                        key = f"{a}_vs_{b}"
                        post_p[key] = st["p_value"]
                        test_rows.append(
                            {
                                "hypothesis": f"{hid}_posthoc_{key}",
                                "kind": "wilcoxon_posthoc",
                                "status": "ok",
                                **st,
                            }
                        )
                adj = holm_adjust(post_p, list(post_p.keys()))
                extras["friedman_posthoc_holm"] = adj
            test_rows.append(row)
        elif kind == "delta_diffs":
            wide = deltas.pivot_table(
                index="fold_id", columns="method", values="Delta", aggfunc="mean"
            )
            for a, b, name in spec.get("pairs") or ():
                if a not in wide.columns or b not in wide.columns:
                    warnings_list.append(f"{hid}/{name}: brak {a} lub {b}")
                    continue
                d = (wide[a] - wide[b]).to_numpy()
                stats = wilcoxon_signed_rank(d)
                med, lo, hi = bootstrap_median_ci(d)
                p_raw[f"{hid}_{name}"] = stats["p_value"]
                test_rows.append(
                    {
                        "hypothesis": f"{hid}_{name}",
                        "kind": kind,
                        "status": "ok",
                        **stats,
                        "median_d_ci_lo": lo,
                        "median_d_ci_hi": hi,
                        "decision_raw": (
                            "reject"
                            if np.isfinite(stats["p_value"]) and stats["p_value"] < ALPHA
                            else "fail_to_reject"
                        ),
                    }
                )
        elif kind == "hodl_descriptive":
            baseline = spec.get("baseline", "R0")
            if baseline not in arms:
                warnings_list.append(f"{hid}: brak {baseline}")
                continue
            # zbudowane później w arm_summary / h5 table
            extras["h5_enabled"] = True

    # Holm: decyzja (extended) lub tylko wrażliwość (intermediate)
    sens = cfg.get("holm_sensitivity_on") or ()
    families = cfg.get("holm_families") or {}
    if families:
        for fam_id, members in families.items():
            adj = holm_adjust(p_raw, members)
            for row in test_rows:
                if row.get("hypothesis") in adj:
                    row["p_holm"] = adj[row["hypothesis"]]
                    row["holm_family"] = fam_id
                    row["decision_holm"] = (
                        "reject"
                        if adj[row["hypothesis"]] < ALPHA
                        else "fail_to_reject"
                    )
    elif sens:
        adj = holm_adjust(p_raw, sens)
        for row in test_rows:
            if row.get("hypothesis") in adj:
                row["p_holm"] = adj[row["hypothesis"]]
                row["p_holm_sensitivity_only"] = True

    # Efektywność e = Delta / adapt_core_hours
    eff_rows = []
    if costs_by_method is not None and not deltas.empty:
        for mid, ax, _ay in METHOD_PAIRS:
            dsub = deltas[deltas["method"] == mid]
            if dsub.empty:
                continue
            # średni koszt adaptacji po oknach (z arm adaptive)
            if not costs_by_method.empty and ax in set(costs_by_method.get("arm", [])):
                c = costs_by_method[costs_by_method["arm"] == ax]
                ch = float(pd.to_numeric(c.get("adapt_core_hours"), errors="coerce").mean() or np.nan)
            else:
                ch = np.nan
            for _, r in dsub.iterrows():
                e = (float(r["Delta"]) / ch) if np.isfinite(ch) and ch > 0 else np.nan
                eff_rows.append(
                    {
                        "method": mid,
                        "fold_id": r["fold_id"],
                        "Delta": r["Delta"],
                        "adapt_core_hours": ch,
                        "e": e,
                    }
                )
    eff_df = pd.DataFrame(eff_rows)
    if not eff_df.empty:
        rank_eff = (
            eff_df.groupby("method")["e"]
            .median()
            .sort_values(ascending=False)
        )
        rank_delta = (
            deltas.groupby("method")["Delta"]
            .median()
            .sort_values(ascending=False)
        )
        extras["efficiency_rank"] = rank_eff.to_dict()
        extras["effectiveness_rank"] = rank_delta.to_dict()
        extras["best_method_rule"] = BEST_METHOD_RULE
        # kompromis: suma rang
        methods = sorted(set(rank_eff.index) | set(rank_delta.index))
        score = {}
        for i, m in enumerate(rank_eff.index):
            score[m] = score.get(m, 0) + i
        for i, m in enumerate(rank_delta.index):
            score[m] = score.get(m, 0) + i
        if rank_eff.index[0] == rank_delta.index[0]:
            extras["best_method"] = rank_eff.index[0]
        else:
            extras["best_method"] = min(score, key=score.get) if score else None

    tests = pd.DataFrame(test_rows)
    diffs_all = pd.concat(diff_frames, ignore_index=True) if diff_frames else pd.DataFrame()
    extras["warnings"] = warnings_list
    extras["deltas"] = deltas
    extras["efficiency"] = eff_df
    return tests, diffs_all, warnings_list, extras


def lag1_autocorr(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return float("nan")
    x0, x1 = d[:-1], d[1:]
    if np.std(x0) < 1e-15 or np.std(x1) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x0, x1)[0, 1])


def classify_trend(ret: float, thr: float = TREND_THRESHOLD) -> str:
    if not np.isfinite(ret):
        return "unknown"
    if ret > thr:
        return "up"
    if ret < -thr:
        return "down"
    return "sideways"


def market_context_from_folds(
    fold_rows: pd.DataFrame,
    *,
    btc_col: str = "btc_return",
    eth_col: str = "eth_return",
) -> pd.DataFrame:
    """Jeśli brak zwrotów rynku w folds — pusta tabela (uzupełnić przy eksporcie z danych)."""
    if fold_rows.empty:
        return pd.DataFrame()
    need = [c for c in (btc_col, eth_col) if c in fold_rows.columns]
    if len(need) < 2:
        return pd.DataFrame()
    rows = []
    for fid, g in fold_rows.groupby("fold_id"):
        btc = float(pd.to_numeric(g[btc_col], errors="coerce").dropna().mean())
        eth = float(pd.to_numeric(g[eth_col], errors="coerce").dropna().mean())
        avg = 0.5 * (btc + eth)
        rows.append(
            {
                "fold_id": fid,
                "btc_return": btc,
                "eth_return": eth,
                "avg_return": avg,
                "trend_class": classify_trend(avg / 100.0 if abs(avg) > 1 else avg),
            }
        )
    return pd.DataFrame(rows)


def efficiency_table(
    arm_sum: pd.DataFrame,
    costs: pd.DataFrame,
    *,
    pairs: Sequence[Tuple[str, str, str]] = (
        ("M1", "R1", "M1_vs_R1"),
        ("M2", "R2", "M2_vs_R2"),
        ("M3", "R3", "M3_vs_R3"),
        ("M3", "M2", "M3_vs_M2"),
    ),
) -> pd.DataFrame:
    """Przyrost średniego R / dodatkowy koszt (opisowe)."""
    if arm_sum.empty:
        return pd.DataFrame()
    rows = []
    for ax, ay, name in pairs:
        sx = arm_sum[arm_sum["arm"] == ax]
        sy = arm_sum[arm_sum["arm"] == ay]
        if sx.empty or sy.empty:
            continue
        for mode in set(sx["portfolio_mode"]) & set(sy["portfolio_mode"]):
            rx = float(sx[sx["portfolio_mode"] == mode]["total_return_mean"].iloc[0])
            ry = float(sy[sy["portfolio_mode"] == mode]["total_return_mean"].iloc[0])
            dR = rx - ry
            cx = cy = np.nan
            if not costs.empty:
                cxa = costs[(costs["arm"] == ax) & (costs["portfolio_mode"] == mode)]
                cya = costs[(costs["arm"] == ay) & (costs["portfolio_mode"] == mode)]
                if not cxa.empty and not cya.empty:
                    cx = float(cxa["wall_time_s_mean"].iloc[0])
                    cy = float(cya["wall_time_s_mean"].iloc[0])
            dC = cx - cy if np.isfinite(cx) and np.isfinite(cy) else np.nan
            rows.append(
                {
                    "pair": name,
                    "portfolio_mode": mode,
                    "delta_R": dR,
                    "delta_wall_time_s": dC,
                    "R_per_extra_second": (dR / dC) if np.isfinite(dC) and dC != 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _latex_tabular(df: pd.DataFrame, caption: str, label: str) -> str:
    if df.empty:
        body = "\\hline\n(brak danych)\\\\\n\\hline\n"
        cols = "l"
    else:
        cols = "l" + "c" * (len(df.columns) - 1)
        header = " & ".join(str(c) for c in df.columns) + " \\\\"
        lines = [header, "\\hline"]
        for _, row in df.iterrows():
            cells = []
            for v in row:
                if isinstance(v, float):
                    cells.append(f"{v:.4g}" if np.isfinite(v) else "---")
                else:
                    cells.append(str(v))
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\hline")
        body = "\n".join(lines)
    return (
        f"% {label}\n"
        f"\\begin{{table}}[ht]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{cols}}}\n\\hline\n{body}\n\\end{{tabular}}\n"
        f"\\end{{table}}\n"
    )


def _save_fig_diffs(diffs: pd.DataFrame, path: str, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        warnings.warn(f"matplotlib skip: {e}")
        return
    if diffs.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for hid, g in diffs.groupby("hypothesis"):
        g = g.sort_values("fold_id")
        ax.plot(g["fold_id"], g["d_i"], marker="o", label=hid)
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("d_i = R_x - R_y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def check_completeness(
    df: pd.DataFrame,
    *,
    arms: Sequence[str],
    seeds: Sequence[int],
    folds: Sequence[str],
    portfolio_mode: str,
    allow_incomplete: bool = False,
) -> List[str]:
    warnings_list = []
    if df.empty:
        warnings_list.append("brak foldów")
        return warnings_list
    sub = df[df["portfolio_mode"] == portfolio_mode] if "portfolio_mode" in df.columns else df
    for arm in arms:
        if arm == "R0":
            # HODL jest deterministyczny: wystarczy jedno ziarno na okno
            for fid in folds:
                hit = sub[(sub["arm"] == "R0") & (sub["fold_id"].astype(str) == str(fid))]
                if hit.empty:
                    warnings_list.append(f"brak: R0 {fid}")
                elif not (hit["status"] == "ok").any():
                    warnings_list.append(f"failed: R0 {fid}")
            continue
        for seed in seeds:
            for fid in folds:
                hit = sub[
                    (sub["arm"] == arm)
                    & (sub["seed"].astype(int) == int(seed))
                    & (sub["fold_id"].astype(str) == str(fid))
                ]
                if hit.empty:
                    warnings_list.append(f"brak: {arm} seed={seed} {fid}")
                elif (hit["status"] != "ok").any():
                    warnings_list.append(
                        f"failed: {arm} seed={seed} {fid} status={hit['status'].iloc[0]}"
                    )
    if warnings_list and not allow_incomplete:
        thesis_console.print_kv(
            "Analysis",
            f"niekompletne dane ({len(warnings_list)}) — użyj --allow-incomplete dla smoke",
            kind="warn",
        )
    return warnings_list


def generate_report(
    run_dirs: Sequence[str],
    out_dir: str,
    *,
    portfolio_mode: str = "reset",
    only_b_folds: bool = True,
    allow_incomplete: bool = False,
    label: str = "report",
    hypothesis_set: Optional[str] = None,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    df = collect_folds(run_dirs)
    if only_b_folds:
        df_work = filter_b_folds(df)
    else:
        df_work = df.copy()
    df_mode = df_work[df_work["portfolio_mode"] == portfolio_mode].copy() if "portfolio_mode" in df_work.columns else df_work

    incomplete = check_completeness(
        df_mode,
        arms=sorted(df_mode["arm"].dropna().unique()) if not df_mode.empty else [],
        seeds=sorted(df_mode["seed"].dropna().unique().astype(int)) if not df_mode.empty else [],
        folds=sorted(df_mode["fold_id"].dropna().unique()) if not df_mode.empty else [],
        portfolio_mode=portfolio_mode,
        allow_incomplete=allow_incomplete,
    )
    with open(os.path.join(out_dir, "completeness_warnings.json"), "w", encoding="utf-8") as f:
        json.dump(incomplete, f, indent=2)

    agg = aggregate_seeds(df_mode, metric="total_return")
    summary = arm_summary(agg, metric="total_return")

    # koszty + adapt_core_hours
    cost_rows = []
    if not df_mode.empty:
        for (arm, mode), g in df_mode.groupby(["arm", "portfolio_mode"]):
            ok = g[g["status"] == "ok"] if "status" in g.columns else g
            cost_rows.append(
                {
                    "arm": arm,
                    "portfolio_mode": mode,
                    "wall_time_s_mean": float(pd.to_numeric(ok.get("wall_time_s"), errors="coerce").mean()),
                    "adapt_core_hours": float(pd.to_numeric(ok.get("adapt_core_hours"), errors="coerce").mean()),
                    "adapt_wall_time_s": float(pd.to_numeric(ok.get("adapt_wall_time_s"), errors="coerce").mean()),
                    "optuna_trials_mean": float(pd.to_numeric(ok.get("optuna_trials"), errors="coerce").mean()),
                    "continue_timesteps_mean": float(pd.to_numeric(ok.get("continue_timesteps"), errors="coerce").mean()),
                }
            )
    costs = pd.DataFrame(cost_rows)

    tests, diffs, hyp_warn, extras = run_hypothesis_tests(
        agg,
        portfolio_mode=portfolio_mode,
        hypothesis_set=hypothesis_set,
        costs_by_method=costs,
    )
    incomplete = list(incomplete) + list(hyp_warn)

    # autokorelacja per hipoteza
    ac_rows = []
    if not diffs.empty:
        for hid, g in diffs.groupby("hypothesis"):
            ac_rows.append({"hypothesis": hid, "lag1_autocorr_d": lag1_autocorr(g.iloc[sorted(range(len(g)), key=lambda i: fold_sort_key(g["fold_id"].iloc[i]))]["d_i"].to_numpy())})
    ac_df = pd.DataFrame(ac_rows)

    deltas = extras.get("deltas")
    if not isinstance(deltas, pd.DataFrame):
        deltas = pd.DataFrame()
    eff_df = extras.get("efficiency")
    if not isinstance(eff_df, pd.DataFrame):
        eff_df = pd.DataFrame()
    eff = efficiency_table(summary, costs)

    # CSV (pełne dane; miary spoza 4.5, np. r Kerby'ego i W Kendalla, zostają tylko tutaj)
    paths = {}
    for name, frame in (
        ("folds_raw.csv", df_mode),
        ("agg_arm_fold.csv", agg),
        ("tab_metryki.csv", summary),
        ("tab_testy.csv", tests),
        ("tab_koszty.csv", costs),
        ("tab_efektywnosc.csv", eff_df if not eff_df.empty else eff),
        ("diffs_di.csv", diffs),
        ("deltas_methods.csv", deltas),
        ("autocorr_di.csv", ac_df),
    ):
        p = os.path.join(out_dir, name)
        frame.to_csv(p, index=False)
        paths[name] = p

    # pliki .tex (po jednym na hipotezę) i wykresy
    from rl_module.services.thesis.report_tex import write_report

    tex_files = write_report(
        out_dir,
        df_mode=df_mode,
        agg=agg,
        tests=tests,
        diffs=diffs,
        ac=ac_df,
        extras=extras,
        costs=costs,
    )

    meta = {
        "label": label,
        "portfolio_mode": portfolio_mode,
        "hypothesis_set": hypothesis_set or "intermediate",
        "n_run_dirs": len(run_dirs),
        "n_incomplete_warnings": len(incomplete),
        "allow_incomplete": allow_incomplete,
        "alpha": ALPHA,
        "effect": "kerby_rank_biserial (CSV only)",
        "best_method": extras.get("best_method"),
        "best_method_rule": extras.get("best_method_rule"),
        "paths": paths,
        "tex_files": tex_files,
    }
    with open(os.path.join(out_dir, "report_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    thesis_console.print_kv("Analysis", f"raport -> {out_dir}", kind="ok")
    return meta


# Uzupełnij PAIRED o M1−M2 (I3.3)
if ("M1", "M2", "M1_minus_M2") not in PAIRED:
    PAIRED_EXTENDED = PAIRED + (("M1", "M2", "M1_minus_M2"),)
else:
    PAIRED_EXTENDED = PAIRED
