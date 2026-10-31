"""
Raport wynikowy do pracy: pliki .tex (po jednym na hipotezę) i wykresy.

Układ zgodny z planem analizy (rozdz. 4.5): H1-H3 test Wilcoxona dla par na d_i (bez Holma),
H4 Friedman na Delta_MP/RL/UL (post hoc Holm tylko po odrzuceniu), H5 opisowo względem R0.
Tabele: pełne krawędzie (|l|c|...| + \\hline), wąskie (<= 9 kolumn).
Miary nieujęte w 4.5 (r Kerby'ego, W Kendalla) zostają w CSV, a w tekście tylko pod flagami.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ALPHA = 0.05
ARM_ORDER = ("R0", "R1", "R2", "R3", "M1", "M2", "M3")
SHOW_KERBY_R = False      # 4.5 nie wymienia r Kerby'ego; włączyć po dopisaniu do 4.5
SHOW_KENDALL_W = False    # 4.5 (PDF) nie wymienia W Kendalla
HODL_NA = ("win_rate", "avg_transactions_per_day", "turnover", "time_in_cash", "profit_factor")
TREND_PP = 5.0            # próg klasyfikacji okna po HODL (p.p.)
FILL, EDGE = "#C2E4F8", "#3B82C4"

HYP_META = {
    "H1": dict(
        x="M1", y="R1", fig="fig_h1_di.png", tab="tab:h1-wynik", fl="fig:h1-di",
        title="H1: strojenie parametrów botem bazowym",
        question=(
            "Czy cykliczne ponowne strojenie parametrów (Optuna, ramię M1) daje inną stopę zwrotu "
            "niż jednorazowe strojenie z zamrożeniem parametrów (ramię R1)?"
        ),
    ),
    "H2": dict(
        x="M2", y="R2", fig="fig_h2_di.png", tab="tab:h2-wynik", fl="fig:h2-di",
        title="H2: douczanie agenta PPO",
        question=(
            "Czy kontynuowanie uczenia agenta PPO na kolejnych oknach (ramię M2) daje inną stopę "
            "zwrotu niż agent zamrożony po treningu (ramię R2)?"
        ),
    ),
    "H3": dict(
        x="M3", y="R3", fig="fig_h3_di.png", tab="tab:h3-wynik", fl="fig:h3-di",
        title="H3: etykieta reżimu rynkowego",
        question=(
            "Czy agent PPO z etykietą reżimu z uczenia nienadzorowanego i kontynuacją uczenia "
            "(ramię M3) daje inną stopę zwrotu niż agent z etykietą zamrożony (ramię R3)?"
        ),
    ),
}


# ----------------------------------------------------------------- formatowanie
def num(x: Any, d: int = 2, *, sign: bool = False) -> str:
    """Liczba z przecinkiem dziesi?tnym i minusem matematycznym (tryb tekstowy)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "---"
    if not math.isfinite(v):
        return "---"
    s = f"{abs(v):.{d}f}".replace(".", "{,}")
    if round(v, d) == 0:
        return s
    if v < 0:
        return f"$-${s}"
    return f"$+${s}" if sign else s


def pval(p: Any) -> str:
    """Warto?? p do u?ycia WEWN?TRZ trybu matematycznego."""
    try:
        v = float(p)
    except (TypeError, ValueError):
        return "---"
    if not math.isfinite(v):
        return "---"
    return "<0{,}001" if v < 0.001 else f"{v:.3f}".replace(".", "{,}")


def pcell(p: Any) -> str:
    s = pval(p)
    return s if s == "---" else f"${s}$"


def tabela(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    caption: str,
    label: str,
    *,
    small: bool = False,
    first_left: bool = True,
    bold_last: int = 0,
) -> str:
    n = len(header)
    spec = "|" + ("l|" if first_left else "c|") + "c|" * (n - 1)
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
    ]
    if small:
        lines.append("\\small")
    lines += [f"\\begin{{tabular}}{{{spec}}}", "\\hline"]
    lines.append(" & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\ \\hline")
    for r in rows:
        lines.append(" & ".join(str(c) for c in r) + " \\\\ \\hline")
    lines += ["\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _arms(cols: Sequence[str]) -> List[str]:
    return [a for a in ARM_ORDER if a in set(cols)]


def _fold_key(fid: str):
    import re

    m = re.match(r"^([A-Za-z]+)(\d+)$", str(fid))
    return (m.group(1), int(m.group(2))) if m else (str(fid), 0)


# ----------------------------------------------------------------- wykresy
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Inter", "Arial", "DejaVu Sans"]
    return plt


def _title(ax, text: str) -> None:
    """Podpis wykresu: pogrubiony, wyśrodkowany, nad wykresem (jak w Simulation Viewer)."""
    ax.set_title(text, fontsize=13, fontweight="bold", color="#111827", loc="center", pad=12)


def fig_bar_di(diffs: pd.DataFrame, path: str, title: str, ylabel: str = "Różnica $d_i$ [p.p.]") -> None:
    plt = _plt()
    g = diffs.sort_values("fold_id", key=lambda s: s.map(_fold_key))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(g["fold_id"], g["d_i"], color=FILL, edgecolor=EDGE, linewidth=1.0)
    ax.axhline(0.0, color="#111827", lw=0.8)
    _title(ax, title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Okno testowe")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_bar_arms(summary: pd.DataFrame, path: str, title: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(summary["arm"], summary["mean_R"], color=FILL, edgecolor=EDGE, linewidth=1.0)
    ax.axhline(0.0, color="#111827", lw=0.8)
    _title(ax, title)
    ax.set_ylabel("Średnia stopa zwrotu [%]")
    ax.set_xlabel("Ramię")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_bar_delty(wide: pd.DataFrame, path: str, title: str) -> None:
    """Grupowy wykres s?upkowy Delta_MP/RL/UL w oknach (kolory grup wg schematy-kolory.md)."""
    plt = _plt()
    style = {
        "MP": ("M1 $-$ R1", "#F6CB6B", "#B7791F", ""),
        "RL": ("M2 $-$ R2", "#E48BB8", "#8E1F52", "//"),
        "UL": ("M3 $-$ R3", "#F0A55E", "#9C5410", "xx"),
    }
    methods = [m for m in ("MP", "RL", "UL") if m in wide.columns]
    x = np.arange(len(wide))
    w = 0.8 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for j, m in enumerate(methods):
        label, fill, edge, hatch = style[m]
        ax.bar(x + (j - (len(methods) - 1) / 2) * w, wide[m].to_numpy(), width=w,
               color=fill, edgecolor=edge, linewidth=1.0, hatch=hatch, label=label)
    ax.axhline(0.0, color="#111827", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(list(wide.index))
    _title(ax, title)
    ax.set_ylabel("Przyrost $\\Delta$ [p.p.]")
    ax.set_xlabel("Okno testowe")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=True, edgecolor="#333333", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure_block(fname: str, width: str = "0.85\\textwidth") -> str:
    """Rysunek bez \\caption: podpis jest na górze obrazu (pogrubiony)."""
    return "\n".join([
        "\\begin{figure}[ht]",
        "\\centering",
        f"\\includegraphics[width={width}]{{{fname}}}",
        "\\end{figure}",
        "",
    ])


# ----------------------------------------------------------------- sekcje (tylko tabele i wykresy)
def section_h123(hid: str, tests: pd.DataFrame, diffs: pd.DataFrame, ac: pd.DataFrame, fig_dir: str) -> str:
    meta = HYP_META[hid]
    t = tests[tests["hypothesis"] == hid]
    d = diffs[diffs["hypothesis"] == hid]
    if t.empty or d.empty:
        return f"% {hid}: brak danych\n"
    r = t.iloc[0]
    n = int(r["n_all"])
    x, y = meta["x"], meta["y"]
    fig_bar_di(
        d, os.path.join(fig_dir, meta["fig"]),
        f"Różnice stóp zwrotu {x} i {y} w oknach testowych",
    )
    rho = ac[ac["hypothesis"] == hid]["lag1_autocorr_d"]
    header = ["Mediana $d_i$ [p.p.]", "Okna $d_i>0$", "$W$", "$p$", "95\\% CI mediany", "$\\rho_1$"]
    row = [
        num(r["median_d"]),
        f"{int(r['n_positive'])} z {n}",
        num(r["statistic"], 1),
        pcell(r["p_value"]),
        f"[{num(r['median_d_ci_lo'])}; {num(r['median_d_ci_hi'])}]",
        num(rho.iloc[0], 2) if len(rho) else "---",
    ]
    if SHOW_KERBY_R:
        header.append("$r$")
        row.append(num(r["r_rb"], 2))
    return "\n".join([
        figure_block(meta["fig"]),
        tabela(header, [row], f"Wynik testu {hid} ({x} i {y})", meta["tab"], small=True, first_left=False),
    ])


def section_h4(tests: pd.DataFrame, extras: Dict[str, Any], deltas: pd.DataFrame, costs: pd.DataFrame, fig_dir: str) -> str:
    if deltas.empty:
        return "% H4: brak danych\n"
    wide = deltas.pivot_table(index="fold_id", columns="method", values="Delta", aggfunc="mean")
    wide = wide.loc[sorted(wide.index, key=_fold_key)]
    methods = [m for m in ("MP", "RL", "UL") if m in wide.columns]
    n = len(wide)
    fig_bar_delty(wide, os.path.join(fig_dir, "fig_h4_delty.png"),
                  "Przyrost wyniku metod samodoskonalenia w oknach testowych")
    names = {"MP": "$\\Delta_{MP}$", "RL": "$\\Delta_{RL}$", "UL": "$\\Delta_{UL}$"}
    rows = [[fid] + [num(wide.loc[fid, m]) for m in methods] for fid in wide.index]
    rows.append(["Mediana"] + [num(wide[m].median()) for m in methods])
    rows.append(["Okna $\\Delta>0$"] + [f"{int((wide[m] > 0).sum())} z {n}" for m in methods])
    t_delta = tabela(
        ["Okno"] + [names[m] for m in methods], rows,
        "Przyrost wyniku metod samodoskonalenia [p.p.]", "tab:h4-delty",
    )

    fr = extras.get("friedman") or {}
    p = fr.get("p_value", np.nan)
    rejected = bool(np.isfinite(p) and p < ALPHA)
    h4_hdr = ["$\\chi^2_F$", "Liczba okien", "$p$"]
    h4_row = [num(fr.get("statistic"), 2), str(fr.get("n_folds", n)), pcell(p)]
    if SHOW_KENDALL_W:
        h4_hdr.append("$W$ Kendalla")
        h4_row.append(num(fr.get("kendall_w"), 2))
    t_fr = tabela(h4_hdr, [h4_row], "Wynik testu Friedmana dla H4", "tab:h4-wynik", first_left=False)

    post = ""
    if rejected:
        adj = extras.get("friedman_posthoc_holm") or {}
        ph = tests[tests["hypothesis"].astype(str).str.startswith("H4_posthoc_")]
        prow = []
        for _, r in ph.iterrows():
            key = str(r["hypothesis"]).replace("H4_posthoc_", "")
            a, b = key.split("_vs_")
            prow.append([f"{a} i {b}", num(r["median_d"]), pcell(r["p_value"]), pcell(adj.get(key))])
        post = tabela(
            ["Para", "Mediana różnic [p.p.]", "$p$", "$p$ po Holmie"], prow,
            "Porównania parami metod (post hoc)", "tab:h4-posthoc",
        )

    eff = extras.get("efficiency")
    arm_of = {"MP": "M1", "RL": "M2", "UL": "M3"}
    crow = []
    for m in methods:
        ch = np.nan
        if not costs.empty and arm_of[m] in set(costs["arm"]):
            ch = float(pd.to_numeric(costs[costs["arm"] == arm_of[m]]["adapt_core_hours"], errors="coerce").mean())
        e = np.nan
        if isinstance(eff, pd.DataFrame) and not eff.empty:
            es = pd.to_numeric(eff[eff["method"] == m]["e"], errors="coerce")
            e = float(es.median()) if es.notna().any() else np.nan
        crow.append([m, num(wide[m].median()), num(ch, 2), num(e, 2)])
    t_cost = tabela(
        ["Metoda", "Mediana $\\Delta$ [p.p.]", "Rdzeniogodziny na okno", "$e$ [p.p./h]"],
        crow, "Skuteczność i koszt adaptacji", "tab:h4-koszt", small=True,
    )
    return "\n".join([figure_block("fig_h4_delty.png"), t_delta, t_fr, post, t_cost])


def table_h5(agg: pd.DataFrame, fig_dir: str) -> str:
    piv = agg.pivot_table(index="fold_id", columns="arm", values="total_return_mean", aggfunc="mean")
    piv = piv.loc[sorted(piv.index, key=_fold_key)]
    if "R0" not in piv.columns:
        return "% H5: brak R0\n"
    arms = _arms(piv.columns)
    n = len(piv)
    rows, sdf = [], []
    for a in arms:
        ex = piv[a] - piv["R0"]
        rows.append([
            a,
            "---" if a == "R0" else f"{int((ex > 0).sum())} z {n}",
            "---" if a == "R0" else num(ex.mean(), 2, sign=True),
        ])
        sdf.append({"arm": a, "mean_R": float(piv[a].mean())})
    fig_bar_arms(
        pd.DataFrame(sdf), os.path.join(fig_dir, "fig_h5_srednie.png"),
        "Średnia stopa zwrotu ramion i strategii HODL w oknach B1--B10".replace("--", "–"),
    )
    t = tabela(
        ["Ramię", "Okna lepsze niż HODL", "Średnia nadwyżka [p.p.]"],
        rows[1:], "Okna z wynikiem lepszym niż HODL (H5)", "tab:h5-hodl", small=True,
    )
    return "\n".join([t, figure_block("fig_h5_srednie.png")])


def table_windows(agg: pd.DataFrame) -> str:
    piv = agg.pivot_table(index="fold_id", columns="arm", values="total_return_mean", aggfunc="mean")
    piv = piv.loc[sorted(piv.index, key=_fold_key)]
    arms = _arms(piv.columns)
    # Kolumny wyrownane do prawej, 2 miejsca po przecinku => przecinki w jednej linii.
    # Najlepszy wynik w wierszu (takze Srednia i Mediana): szare tlo
    # (wymaga \usepackage[table]{xcolor}).
    def row_cells(label: str, vals: pd.Series) -> List[str]:
        b = vals.idxmax()
        out = [label]
        for a_ in arms:
            c = num(vals[a_])
            out.append(f"\\cellcolor{{gray!25}}{c}" if a_ == b else c)
        return out

    body = [row_cells(fid, piv.loc[fid]) for fid in piv.index]
    stats = [
        row_cells("Średnia", pd.Series({a_: piv[a_].mean() for a_ in arms})),
        row_cells("Mediana", pd.Series({a_: piv[a_].median() for a_ in arms})),
    ]
    spec = "|l|" + "r|" * len(arms)
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Stopa zwrotu ramion w oknach testowych [\\%]}",
        "\\label{tab:wyniki-okna}",
        "\\small",
        f"\\begin{{tabular}}{{{spec}}}",
        "\\hline",
        " & ".join(f"\\textbf{{{h}}}" for h in ["Okno"] + arms) + " \\\\ \\hline",
    ]
    for r in body[:-1]:
        lines.append(" & ".join(r) + " \\\\ \\hline")
    # gruba linia miedzy ostatnim oknem a statystykami (bez dodatkowych pakietow)
    lines.append(" & ".join(body[-1]) + " \\\\ \\noalign{\\hrule height 1.5pt}")
    for r in stats:
        lines.append(" & ".join(r) + " \\\\ \\hline")
    lines += ["\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def table_extra_metrics(df: pd.DataFrame) -> str:
    """Miary w wierszach, ramiona w kolumnach (jak tabela wyników okien)."""
    specs = [
        ("sharpe_ratio", "Sharpe", 2, "mean", 1.0),
        ("sortino_ratio", "Sortino", 2, "mean", 1.0),
        ("max_drawdown", "Max Drawdown [\\%]", 1, "mean", 1.0),
        ("profit_factor", "Profit Factor (mediana)", 2, "median", 1.0),
        ("turnover", "Obrót", 1, "mean", 1.0),
        ("avg_transactions_per_day", "Transakcje dziennie", 1, "mean", 1.0),
        ("time_in_cash", "Czas w gotówce [\\%]", 1, "mean", 100.0),
    ]
    specs = [s for s in specs if s[0] in df.columns]
    arms = _arms(df["arm"].unique())
    rows = []
    for col, label, d, how, mult in specs:
        row = [label]
        for a in arms:
            s = pd.to_numeric(df[df["arm"] == a][col], errors="coerce").dropna()
            if (a == "R0" and col in HODL_NA) or not len(s):
                row.append("---")
            else:
                row.append(num((s.mean() if how == "mean" else s.median()) * mult, d))
        rows.append(row)
    return tabela(["Miara"] + arms, rows, "Metryki uzupełniające ramion", "tab:metryki-dod", small=True)


def table_seeds(df: pd.DataFrame) -> str:
    g = df.groupby(["arm", "seed"])["total_return"].mean().unstack("seed")
    seeds = list(g.columns)
    rows = []
    for a in _arms(g.index):
        vals = g.loc[a]
        sd = vals.std(ddof=1) if vals.notna().sum() > 1 else np.nan
        rows.append([a] + [num(vals[s]) for s in seeds] + [num(sd)])
    return tabela(
        ["Ramię"] + [str(int(s)) for s in seeds] + ["$\\sigma$"], rows,
        "Średnia stopa zwrotu ramion dla ziaren [\\%]", "tab:ziarna-sigma", small=True,
    )


def table_trend(agg: pd.DataFrame) -> str:
    piv = agg.pivot_table(index="fold_id", columns="arm", values="total_return_mean", aggfunc="mean")
    if "R0" not in piv.columns:
        return "% brak R0\n"
    bots = [a for a in _arms(piv.columns) if a != "R0"]
    cls = piv["R0"].map(lambda r: "wzrost" if r > TREND_PP else ("spadek" if r < -TREND_PP else "boczny"))
    rows = []
    for c in ("wzrost", "boczny", "spadek"):
        idx = cls[cls == c].index
        if len(idx) == 0:
            continue
        rows.append([f"{c} ({len(idx)})"] + [num((piv.loc[idx, a] - piv.loc[idx, "R0"]).mean(), 1, sign=True) for a in bots])
    return tabela(
        ["Okna (liczba)"] + bots, rows,
        "Średnia nadwyżka nad HODL wg kierunku rynku [p.p.]", "tab:trend-nadwyzka", small=True,
    )


def write_report(
    out_dir: str,
    *,
    df_mode: pd.DataFrame,
    agg: pd.DataFrame,
    tests: pd.DataFrame,
    diffs: pd.DataFrame,
    ac: pd.DataFrame,
    extras: Dict[str, Any],
    costs: pd.DataFrame,
) -> List[str]:
    tex_dir = os.path.join(out_dir, "latex")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(tex_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    ok = df_mode[df_mode["status"] == "ok"] if "status" in df_mode.columns else df_mode
    deltas = extras.get("deltas")
    deltas = deltas if isinstance(deltas, pd.DataFrame) else pd.DataFrame()

    t_okna = table_windows(agg)
    t_dod = table_extra_metrics(ok)
    t_h5 = table_h5(agg, fig_dir)

    # 5.1: wyniki zbiorcze (tabele + wykres H5); porównania okien z Simulation Viewer wstawia autor
    w51 = "\n".join([
        t_okna,
        t_dod,
        t_h5,
        "% Porównanie jednego okna między metodami (PNG z Simulation Viewer, 'Pobierz PNG (praca)'):",
        "% \\begin{figure}[ht]\\centering\\includegraphics[width=\\textwidth]{rysunki/porownanie_B3}\\end{figure}",
        "% \\begin{figure}[ht]\\centering\\includegraphics[width=\\textwidth]{rysunki/porownanie_B10}\\end{figure}",
        "",
    ])

    files: Dict[str, str] = {
        "w5_1_wyniki_zbiorcze.tex": w51,
        "h1.tex": section_h123("H1", tests, diffs, ac, fig_dir),
        "h2.tex": section_h123("H2", tests, diffs, ac, fig_dir),
        "h3.tex": section_h123("H3", tests, diffs, ac, fig_dir),
        "h4.tex": section_h4(tests, extras, deltas, costs, fig_dir),
        "tab_ziarna_sigma.tex": table_seeds(ok),
        "tab_trend.tex": table_trend(agg),
    }
    for name, text in files.items():
        _write(os.path.join(tex_dir, name), text)
    return sorted(files)
