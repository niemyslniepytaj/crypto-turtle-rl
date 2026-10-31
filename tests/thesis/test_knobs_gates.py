"""I0.1 — wspólne wagi/bramki Optuny."""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from rl_module.services.thesis.knobs import (
    OPTUNA_OBJECTIVE,
    TUNER_GATES,
    apply_tuner_settings,
    tuner_settings_snapshot,
)


class _FakeTuner:
    """Minimalny stub tunera do testów bramek Optuna / knobs."""
    pass


def test_snapshot_matches_constants():
    snap = tuner_settings_snapshot()
    for k, v in OPTUNA_OBJECTIVE.items():
        assert snap[k] == v
    for k, v in TUNER_GATES.items():
        assert snap[k] == v


def test_apply_tuner_settings():
    t = _FakeTuner()
    snap = apply_tuner_settings(t)
    assert t.sharpe_weight == 5.0
    assert t.dd_weight == 0.75
    assert t.min_tx_per_day == 0.7
    assert t.hodl_corr_hard == 0.985
    assert snap["sharpe_weight"] == t.sharpe_weight
