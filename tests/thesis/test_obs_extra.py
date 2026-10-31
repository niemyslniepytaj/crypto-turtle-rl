"""Rozszerzone obs v6 rl: wymiary, przyczynowosc (brak wycieku z przyszlosci), zgodnosc z referencja."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from test_module.services.core_logic import version_registry as vr  # noqa: E402


def _engine(n=20000, seed=0):
    rng = np.random.RandomState(seed)
    syms = ["ETH", "BTC"]
    models = {}
    for s in syms:
        close = 100.0 * np.exp(np.cumsum(rng.randn(n) * 0.0005))
        mu = rng.randn(n) * 0.001
        models[s] = SimpleNamespace(close=close, xgb_mu=mu)
    probs = {s: [0.3, 0.4, 0.3] for s in syms}
    return SimpleNamespace(symbols=syms, models=models, current_probs=probs, BUY_IDX=0, SELL_IDX=2)


def _ref(eng, t):
    """Wolna referencja (wprost z definicji), dane tylko <= t."""
    out = []
    h = vr.V6RL_PRED_HORIZON
    for s in eng.symbols:
        close = eng.models[s].close
        mu = eng.models[s].xgb_mu
        probs = eng.current_probs[s]

        def mu_mean(w):
            lo = max(0, t - w + 1)
            return float(np.clip(np.mean(mu[lo:t + 1]) * 100.0, -5, 5))

        def hit(w):
            b = t - h
            a = max(0, b - w + 1)
            if b - a < 30:
                return 0.0
            m = mu[a:b + 1]
            r = np.log(close[a + h:b + h + 1] / close[a:b + 1])
            return float(np.clip((np.mean(np.sign(m) == np.sign(r)) - 0.5) * 4.0, -2, 2))

        def ret(w):
            return float(np.log(close[t] / close[t - w]) * 100.0) if t >= w else 0.0

        vol = 0.0
        if t >= vr._V6X_24H + 1:
            vd = np.std(np.diff(np.log(close[t - vr._V6X_24H: t + 1])))
            lo7 = max(0, t - vr._V6X_7D)
            vw = np.std(np.diff(np.log(close[lo7: t + 1])))
            vol = float(np.tanh(vd / (vw + 1e-8) - 1.0))
        lo7 = max(0, t - vr._V6X_7D)
        dd7 = float(np.tanh((close[t] / np.max(close[lo7: t + 1]) - 1.0) * 100.0 / 5.0))
        out += [
            probs[0], probs[2], mu_mean(240), mu_mean(1440), hit(240), hit(1440),
            float(np.tanh(ret(240) / 1.5)), float(np.tanh(ret(1440) / 3.0)), vol, dd7,
        ]
    return np.asarray(out, dtype=np.float32)


def test_dims_and_overlay():
    eng = _engine()
    f = vr.v6rl_extra_features(eng, 12000)
    assert f.shape == (vr.V6RL_EXTRA_PER_SYMBOL * 2,)
    assert np.isfinite(f).all()
    exp = vr.EXPERIMENTS["v6"]
    if vr.V6RL_OBS_MIN:
        per = len(vr.V6RL_OBS_MIN_IDX)
    else:
        per = 13 + (vr.V6RL_EXTRA_PER_SYMBOL if vr.V6RL_EXTRA_OBS else 0)
    for mode in ("rl", "mprl", "rlul"):
        assert exp["by_sim_mode"][mode]["obs_len_for_symbol"] == per
    assert exp["by_sim_mode"]["mp"]["obs_len_for_symbol"] == 13


def test_matches_reference_at_many_t():
    eng = _engine()
    for t in (50, 400, 1500, 5000, 12000, 19999):
        got = vr.v6rl_extra_features(eng, t)
        ref = _ref(eng, t)
        assert np.allclose(got, ref, atol=1e-4), (t, got, ref)


def test_causal_no_future_leak():
    eng = _engine()
    t = 12000
    a = vr.v6rl_extra_features(eng, t)
    for s in eng.symbols:
        c = eng.models[s].close.copy()
        m = eng.models[s].xgb_mu.copy()
        c[t + 1:] *= 5.0
        m[t + 1:] = 9.0
        eng.models[s].close = c
        eng.models[s].xgb_mu = m
    b = vr.v6rl_extra_features(eng, t)
    assert np.array_equal(a, b)


def test_hit_rate_ref_helper_bounded():
    eng = _engine()
    m = eng.models["ETH"]
    base = vr._v6_hit_rate(m.xgb_mu, m.close, 12000, 240)
    assert -2.0 <= base <= 2.0


def test_min_obs_dim_and_selection():
    if not vr.V6RL_OBS_MIN:
        return
    exp = vr.EXPERIMENTS["v6"]["by_sim_mode"]["rl"]
    assert exp["obs_len_global"] + 2 * exp["obs_len_for_symbol"] == 18
    assert tuple(vr.V6RL_OBS_MIN_IDX) == (2, 11, 8, 12)
