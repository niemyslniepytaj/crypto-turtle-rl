"""Etap 1: rl bez locka; K z agenta, nie z REGIME_K_DEFAULT."""
import os
import sys
from types import SimpleNamespace

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)


def test_append_regime_zero_passthrough():
    from test_module.services.core_logic.decision import DecisionEngine
    from test_module.services.core_logic.structures import CoreConfig

    # minimal stub — bez ładowania agenta
    eng = object.__new__(DecisionEngine)
    eng._regime_k = 0
    eng._regime_onehot_series = None
    eng._regime_onehot = None
    obs = np.arange(5, dtype=np.float32)
    out = DecisionEngine._append_regime(eng, obs)
    assert np.allclose(out, obs)


def test_infer_k_from_obs_space():
    from test_module.services.core_logic.decision import DecisionEngine

    eng = object.__new__(DecisionEngine)
    eng.BASE_MODEL_DIR = os.path.join(BASE, "shared", "agents")
    eng.agent_name = None
    eng.agent_path = ""
    eng.obs_len_global = 10
    eng.obs_len_for_symbol = 13
    eng.symbols = ["ETH", "BTC"]
    # market = 10 + 13*2 = 36; agent dim 39 → K=3
    eng.agent = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(39,)),
    )
    assert eng.infer_regime_k_from_agent() == 3
    eng.agent = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(36,)),
    )
    assert eng.infer_regime_k_from_agent() == 0


def test_preflight_r2_no_ul_required():
    """Preflight R0–M2 bez regime_lock nie zgłasza błędu UL."""
    from rl_module.services.thesis.preflight import validate_sequence_config

    cfg = {
        "seeds": [265],
        "arms": ["R0", "R1", "R2", "M1", "M2"],
        "start_window": "A1",
        "end_window": "A2",
        "data_files": {},
        "model_paths": {},
        "by_seed": {"265": {}},
    }
    # check_files=True: UL nie wymagane; mogą być inne braki (CSV/agent)
    errs = validate_sequence_config(cfg, base_dir=BASE, check_files=True)
    assert not any("regime_lock" in e or "regime_model" in e or "plgrid_regime_lock" in e for e in errs)

    cfg_ul = dict(cfg)
    cfg_ul["arms"] = ["R3", "M3"]
    errs_ul = validate_sequence_config(cfg_ul, base_dir=BASE, check_files=True)
    # R3/M3 wymagaja agent_rlul (bez fallbacku na agent_rl) lub artefaktow UL
    assert any(
        "agent_rlul" in e or "regime_lock" in e or "regime_model" in e or "plgrid_regime_lock" in e
        for e in errs_ul
    )
