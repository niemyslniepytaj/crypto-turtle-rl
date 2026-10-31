"""Pretrain PPO — stałe, early-stop, sim_best copy, train.lock, LR desc, V5RL źródło."""
import json
import os
import sys
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from rl_module.services.thesis.assets import THESIS_SEEDS
from rl_module.services.thesis.knobs import (
    MP_CONST,
    PPO_EARLY_STOP,
    PPO_HPARAMS,
    V5RL_KNOB_HOLD_BARS,
    V5RL_TRADE_LOCK_BARS,
    build_pretrain_ppo_kwargs,
    ppo_hparams_snapshot,
    portfolio_const_snapshot,
    thesis_train_config_hash,
)
from rl_module.services.thesis_artifacts import make_agent_artifact
from z_maina_funkcje import plgrid_train_agent_ppo as entry
from z_maina_funkcje.plgrid_train_agent_ppo import (
    acquire_train_lock,
    promote_sim_best_for_seed,
    release_train_lock,
    run_pretrain,
)


REQUIRED_PPO_KEYS = (
    "n_steps",
    "batch_size",
    "gamma",
    "ent_coef",
    "ent_coef_end",
    "learning_rate_start",
    "learning_rate_end",
    "gae_lambda",
    "clip_range",
    "n_epochs",
    "episode_timesteps",
    "total_timesteps",
    "reward_mode",
)


def test_ppo_hparams_has_all_keys_and_4m():
    for k in REQUIRED_PPO_KEYS:
        assert k in PPO_HPARAMS, f"brak {k} w PPO_HPARAMS"
    assert PPO_HPARAMS["total_timesteps"] == 4_000_000
    assert PPO_HPARAMS["reward_mode"] == "excess"
    assert PPO_EARLY_STOP["patience"] == 20
    assert PPO_EARLY_STOP["min_dumps"] == 30


def test_build_pretrain_kwargs_match_mp_const_and_early_stop():
    kw = build_pretrain_ppo_kwargs(
        seed=THESIS_SEEDS[0], sim_mode="rl", learning_rate=1.0,
    )
    assert kw["stop_loss"] == MP_CONST["stop_loss"]
    assert kw["stop_loss_cooldown"] == MP_CONST["stop_loss_cooldown"]
    assert kw["mincost"] == MP_CONST["mincost"]
    assert kw["wallet_percentage_limit"] == MP_CONST["wallet_percentage_limit"]
    assert kw["max_wallet_percentage_crypto"] == MP_CONST["max_wallet_percentage_crypto"]
    assert kw["mp_threshold"] == MP_CONST["threshold_buy"]
    assert kw["v5rl_knob_hold_bars"] == V5RL_KNOB_HOLD_BARS
    assert kw["v5rl_trade_lock_bars"] == V5RL_TRADE_LOCK_BARS
    assert kw["early_stop_patience"] == 20
    assert kw["early_stop_min_dumps"] == 30
    assert kw["total_timesteps"] == 4_000_000
    assert kw["reward_mode"] == "excess"
    assert kw["window_id"] == "train"
    assert kw["eval_window_id"] == "A1"


def test_run_pretrain_passes_mp_const_to_service(tmp_path):
    captured = {}

    class FakeService:
        """Stub TrainAgentPPOService przechwytujący kwargs w teście pretrainu."""

        def train_agent_ppo(self, **kwargs):
            captured.update(kwargs)
            return {
                "agent_file": "fake_final.zip",
                "agent_path": "/tmp/fake_final.zip",
                "result": "success",
            }

    out = run_pretrain(
        seed=THESIS_SEEDS[0],
        sim_mode="rl",
        service=FakeService(),
        learning_rate=1.0,
        acquire_lock=False,
        ensure_watcher=False,
        logs_dir=str(tmp_path / "logs"),
        agent_dir=str(tmp_path / "ppo"),
    )
    assert captured["stop_loss"] == MP_CONST["stop_loss"]
    assert captured["v5rl_knob_hold_bars"] == V5RL_KNOB_HOLD_BARS
    assert captured["v5rl_trade_lock_bars"] == V5RL_TRADE_LOCK_BARS
    assert captured["early_stop_patience"] == 20
    assert out["selection"] == "final_fallback"


def test_ppo_hparams_in_agent_artifact():
    hp = ppo_hparams_snapshot()
    port = portfolio_const_snapshot()
    ch = thesis_train_config_hash()
    art = make_agent_artifact(
        seed=THESIS_SEEDS[0],
        agent_file="agent_ppo_seed396379_final_x.zip",
        window_id="train",
        eval_window_id="A1",
        train_meta={"ppo_hparams": hp, "portfolio_const": port},
        extra={
            "ppo_hparams": hp,
            "portfolio_const": port,
            "git_commit": "deadbeef",
            "config_hash": ch,
        },
    )
    assert art["ppo_hparams"]["total_timesteps"] == 4_000_000
    assert art["portfolio_const"]["stop_loss"] == MP_CONST["stop_loss"]
    assert len(art["config_hash"]) == 64


def test_promote_sim_best_copies_zip(tmp_path: Path = None):
    # pytest-free: lokalny tmp
    root = Path(BASE) / "_tmp_test_simbest"
    if tmp_path is not None:
        root = Path(tmp_path)
    else:
        root.mkdir(exist_ok=True)
    logs = root / "logs"
    agents = root / "agents"
    logs.mkdir(parents=True, exist_ok=True)
    agents.mkdir(parents=True, exist_ok=True)

    work = agents / "agent_ppo_workdir_sim_best.zip"
    work.write_bytes(b"zip-bytes")
    rms = agents / "agent_ppo_workdir_sim_best_obs_rms.npz"
    rms.write_bytes(b"rms")
    sb = {
        "agent_file": work.name,
        "excess_pct": 1.23,
        "epoch_sum": 99,
        "global_dump": 5,
        "train_start_time": "2026-01-01T00:00:00",
    }
    (logs / "sim_best.json").write_text(json.dumps(sb), encoding="utf-8")

    meta = promote_sim_best_for_seed(
        seed=THESIS_SEEDS[0],
        logs_dir=str(logs),
        agent_dir=str(agents),
        train_start_time="2026-01-01T00:00:00",
        eval_window_id="A1",
    )
    assert meta is not None
    assert meta["selection"] == "sim_best"
    assert meta["agent_file"] == f"PPO_best_{THESIS_SEEDS[0]}.zip"
    assert meta["seed"] == THESIS_SEEDS[0]
    dst = agents / meta["agent_file"]
    assert dst.is_file()
    assert dst.read_bytes() == b"zip-bytes"
    meta_json = Path(str(dst).replace(".zip", ".json"))
    assert meta_json.is_file()


def test_train_lock_blocks_second(tmp_path=None):
    root = Path(BASE) / "_tmp_test_trainlock"
    root.mkdir(exist_ok=True)
    lock = root / "train.lock"
    if lock.exists():
        lock.unlink()
    acquire_train_lock(THESIS_SEEDS[0], lock_path=str(lock))
    try:
        raised = False
        try:
            acquire_train_lock(THESIS_SEEDS[1], lock_path=str(lock))
        except RuntimeError as e:
            raised = True
            assert "trening już trwa" in str(e)
        assert raised
    finally:
        release_train_lock(lock_path=str(lock))


def test_lr_desc_progress_remaining_order():
    """SB3: fn(1)=start, fn(0)=end → opis 2e-4→4e-5."""
    start, end = 2.0e-4, 4.0e-5

    def schedule(progress_remaining: float) -> float:
        # jak get_linear_fn(start, end, 1.0): remaining=1 → start
        return end + (start - end) * float(progress_remaining)

    lr_desc = f"{float(schedule(1.0)):.2e}→{float(schedule(0.0)):.2e}"
    assert lr_desc.startswith("2.00e-04")
    assert "4.00e-05" in lr_desc


def test_v5rl_bars_single_source_in_kwargs_and_continue_defaults():
    kw = build_pretrain_ppo_kwargs(seed=1, learning_rate=1.0)
    assert kw["v5rl_knob_hold_bars"] == V5RL_KNOB_HOLD_BARS == 1
    assert kw["v5rl_trade_lock_bars"] == V5RL_TRADE_LOCK_BARS == 15
    # walkforward / watcher importują te same stałe
    from rl_module.services.thesis.walkforward import V5RL_KNOB_HOLD_BARS as W_H
    from rl_module.services.thesis.walkforward import V5RL_TRADE_LOCK_BARS as W_L
    from z_maina_funkcje.plgrid_train_eval_watcher import _V5RL_HOLD, _V5RL_LOCK
    assert W_H == V5RL_KNOB_HOLD_BARS
    assert W_L == V5RL_TRADE_LOCK_BARS
    assert _V5RL_HOLD == V5RL_KNOB_HOLD_BARS
    assert _V5RL_LOCK == V5RL_TRADE_LOCK_BARS


def test_promote_rlul_uses_ul_name(tmp_path):
    logs = tmp_path / "logs"
    agents = tmp_path / "agents"
    logs.mkdir()
    agents.mkdir()
    (agents / "agent_ppo_workdir_sim_best.zip").write_bytes(b"z")
    (logs / "sim_best.json").write_text(
        json.dumps({"agent_file": "agent_ppo_workdir_sim_best.zip", "excess_pct": 1.0,
                    "epoch_sum": 1, "global_dump": 1, "train_start_time": "t"}),
        encoding="utf-8",
    )
    meta = promote_sim_best_for_seed(
        seed=THESIS_SEEDS[0], logs_dir=str(logs), agent_dir=str(agents),
        train_start_time="t", sim_mode="rlul",
    )
    assert meta["agent_file"] == f"PPO_best_UL_{THESIS_SEEDS[0]}.zip"
    assert meta["sim_mode"] == "rlul"
