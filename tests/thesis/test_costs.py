"""Z3.2 — core_hours i adapt=0 dla R*."""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from rl_module.services.thesis.costs import attach_adapt_costs, core_hours


def test_core_hours_formula():
    assert abs(core_hours(3600, 2) - 2.0) < 1e-9


def test_r_arms_zero_adapt():
    for arm in ("R0", "R1", "R2", "R3"):
        art = {"optuna_time_s": 10, "train_time_s": 20, "ul_fit_time_s": 5}
        attach_adapt_costs(art, arm=arm)
        assert art["adapt_core_hours"] == 0.0
        assert art["adapt_wall_time_s"] == 0.0


def test_m3_sum_parts():
    art = {
        "train_time_s": 100.0,
        "train_cores": 4,
        "ul_fit_time_s": 50.0,
        "ul_cores": 1,
        "optuna_time_s": 0.0,
    }
    attach_adapt_costs(art, arm="M3")
    assert abs(art["adapt_wall_time_s"] - 150.0) < 1e-6
    expected = core_hours(100, 4) + core_hours(50, 1)
    assert abs(art["adapt_core_hours"] - expected) < 1e-9
