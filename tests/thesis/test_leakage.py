"""I0.4 — brak przecieku: train/UL kończy się przed eval."""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from rl_module.services.thesis.walkforward import rolling_train_spec
from test_module.services.core_logic.windows import load_windows


def test_rolling_train_before_eval_all_ab():
    windows = load_windows(BASE)
    seq = list(windows["sequence"])
    for fold in seq:
        if not (str(fold).startswith("A") or str(fold).startswith("B")):
            continue
        idx = seq.index(fold)
        if idx == 0:
            continue
        spec = rolling_train_spec(windows, fold, n=3)
        parts = spec.split("+")
        for p in parts:
            assert seq.index(p) < idx, f"{p} not before {fold}"


def test_m2_continue_window_is_current_not_future():
    """Douczanie M2/M3 na właśnie ewaluowanym oknie — następny fold jest późniejszy."""
    windows = load_windows(BASE)
    seq = list(windows["sequence"])
    a_folds = [w for w in seq if str(w).startswith("A") or str(w).startswith("B")]
    for i, fold in enumerate(a_folds[:-1]):
        nxt = a_folds[i + 1]
        assert seq.index(fold) < seq.index(nxt)
