"""I2.5 — UL fit nie widzi okna eval; one-hot / zera."""
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from rl_module.services.regime_service import RegimeConfig, RegimeService, features_from_closes
from rl_module.services.thesis.walkforward import rolling_train_spec
from test_module.services.core_logic.windows import load_windows


def test_neutral_zeros():
    svc = RegimeService(RegimeConfig(k_locked=3))
    z = svc.neutral_onehot(5, k=3)
    assert z.shape == (5, 3)
    assert np.allclose(z, 0.0)


def test_fit_predict_and_canonical_k():
    rng = np.random.RandomState(0)
    # dwa reżimy; n > vol_long (1440) żeby drop_warmup zostawił próbki
    n = 4000
    c1 = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.001, n)))
    c2 = 100 * np.exp(np.cumsum(rng.normal(-0.0001, 0.002, n)))
    closes = {"ETH": c1, "BTC": c2}
    svc = RegimeService(RegimeConfig(seed=0, k_min=2, k_max=3))
    k = svc.fit_from_closes(closes, seed=0)
    assert 2 <= k <= 3
    oh = svc.predict_onehot_from_closes(closes)
    assert oh.shape[1] == k
    assert np.all(oh.sum(axis=1) == 1.0)
    assert svc.scaler_mean_ is not None


def test_ul_windows_before_eval():
    windows = load_windows(BASE)
    for fold in ("A1", "A2", "B1"):
        if fold not in windows["sequence"]:
            continue
        spec = rolling_train_spec(windows, fold, n=3)
        seq = windows["sequence"]
        for p in spec.split("+"):
            assert seq.index(p) < seq.index(fold)
