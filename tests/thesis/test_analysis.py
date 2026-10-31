"""I3.11 — Holm / Wilcoxon / agregacja na danych syntetycznych."""
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from rl_module.services.thesis.analysis import (
    aggregate_seeds,
    generate_report,
    holm_adjust,
    wilcoxon_signed_rank,
)


def test_wilcoxon_excludes_zeros_and_kerby():
    d = np.array([1.0, -2.0, 0.0, 3.0, -1.0, 0.0, 2.0])
    out = wilcoxon_signed_rank(d)
    assert out["n_zero"] == 2
    assert out["n_nonzero"] == 5
    assert np.isfinite(out["p_value"])
    assert -1.0 <= out["r_rb"] <= 1.0
    assert abs(out["W_plus"] + out["W_minus"] - out["n_nonzero"] * (out["n_nonzero"] + 1) / 2) < 1e-6


def test_bootstrap_deterministic():
    from rl_module.services.thesis.analysis import bootstrap_median_ci
    d = np.array([1.0, 2.0, -1.0, 0.5, 3.0, -0.2, 1.5, 0.0, 2.2, -0.5])
    a = bootstrap_median_ci(d, seed=123)
    b = bootstrap_median_ci(d, seed=123)
    assert a == b


def test_kerby_manual():
    # 3 dodatnie, 1 ujemne o mniejszych |d| → r_rb > 0
    d = np.array([3.0, 2.0, 1.0, -0.5])
    out = wilcoxon_signed_rank(d)
    assert out["r_rb"] > 0


def test_holm_monotonic():
    raw = {"H1": 0.01, "H2": 0.04}
    adj = holm_adjust(raw, ("H1", "H2"))
    assert adj["H1"] <= adj["H2"]
    assert abs(adj["H1"] - 0.02) < 1e-12  # 2*0.01
    assert abs(adj["H2"] - 0.04) < 1e-12  # max(0.02, 1*0.04)


def test_aggregate_seeds_mean():
    rows = []
    for seed in (1, 2):
        rows.append(
            {
                "arm": "R1",
                "fold_id": "B1",
                "portfolio_mode": "reset",
                "seed": seed,
                "status": "ok",
                "total_return": 10.0 * seed,
                "excess_return": 1.0,
            }
        )
    df = pd.DataFrame(rows)
    agg = aggregate_seeds(df)
    assert abs(agg.iloc[0]["total_return_mean"] - 15.0) < 1e-9


def test_generate_report_synthetic():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        # minimal fake run dirs with fold json + manifest
        for arm, rets in (("R1", [1, 2, 3]), ("M1", [2, 3, 4])):
            rd = tmp_path / f"{arm}_test_1"
            (rd / "windows").mkdir(parents=True)
            man = {
                "arm": arm,
                "seed": 1,
                "run_label": "synth",
                "portfolio_mode": "reset",
            }
            (rd / "manifest.json").write_text(__import__("json").dumps(man), encoding="utf-8")
            for i, r in enumerate(rets, start=1):
                fold = {
                    "arm": arm,
                    "fold_id": f"B{i}",
                    "status": "ok",
                    "metrics": {"total_return": float(r)},
                    "excess_return": float(r) - 1.0,
                    "artifacts": {"wall_time_s": 1.0},
                }
                (rd / "windows" / f"fold_B{i}.json").write_text(
                    __import__("json").dumps(fold), encoding="utf-8"
                )
        out = tmp_path / "report"
        meta = generate_report(
            [str(tmp_path / "R1_test_1"), str(tmp_path / "M1_test_1")],
            str(out),
            portfolio_mode="reset",
            only_b_folds=True,
            allow_incomplete=True,
            label="synth",
        )
        assert (out / "tab_testy.csv").exists()
        assert meta["n_incomplete_warnings"] >= 0
