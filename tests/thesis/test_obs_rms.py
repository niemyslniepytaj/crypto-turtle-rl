"""I0.3 — zapis/odczyt obs_rms."""
import os
import sys
import tempfile

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from rl_module.services.rl.obs_rms_utils import obs_rms_path_for_agent, resolve_obs_rms_beside_agent


def test_obs_rms_path_naming():
    p = obs_rms_path_for_agent("/tmp/agent_ppo_seed1_final.zip")
    assert p.endswith("agent_ppo_seed1_final_obs_rms.npz")


def test_resolve_beside_agent():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        zip_path = tmp_path / "agent_ppo_seed7.zip"
        zip_path.write_bytes(b"fake")
        rms = tmp_path / "agent_ppo_seed7_obs_rms.npz"
        mean = np.zeros(4)
        var = np.ones(4)
        np.savez(rms, mean=mean, var=var, clip_obs=5.0, epsilon=1e-8)
        found = resolve_obs_rms_beside_agent(str(zip_path))
        assert found == str(rms)
        with np.load(found) as data:
            assert data["mean"].shape == (4,)
