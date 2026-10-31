"""Entry CLI: sekwencja walk-forward tezy (ramiona R0–M3)."""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rl_module.services.walkforward_service import WalkForwardService

if __name__ == "__main__":
    # Sekwencja tezy — osobny entry (nie wołać z train/tuner).
    #
    # Kolejność: regime_lock → simulation_tuner (--lock-tag final) → train_agent_ppo
    #            → ten skrypt (thesis_sequence_config.json) → thesis_compare
    # Config: shared/files/thesis_sequence_config.json (+ .example.json)
    # Wyniki: shared/simulation/thesis_results/{ARM}_thesis_{seed}_{date}_{start}_{end}/
    # Opis składu: docs/wyniki-kampanii-tezy.md

    parser = argparse.ArgumentParser(description="Sekwencja tezy (walk-forward)")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Tylko walidacja configu + plan (bez treningu/symulacji)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Ścieżka do thesis_sequence_config.json",
    )
    parser.add_argument("--arms", nargs="*", default=None, help="Podzbior ramion, np. R0 R1")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Podzbior seedow")
    args, _unknown = parser.parse_known_args()

    config_path = args.config or os.path.join(
        BASE_DIR, "shared", "files", "thesis_sequence_config.json"
    )
    example_path = os.path.join(
        BASE_DIR, "shared", "files", "thesis_sequence_config.example.json"
    )

    if args.preflight:
        from rl_module.services.thesis.preflight import run_preflight

        path = config_path if os.path.isfile(config_path) else example_path
        if not os.path.isfile(path):
            print(f"[plgrid_test_sequence] Brak configu: {config_path}")
            sys.exit(1)
        sys.exit(run_preflight(path, BASE_DIR))

    wf = WalkForwardService(BASE_DIR)

    if os.path.isfile(config_path):
        _ov = {}
        if args.arms:
            _ov["arms"] = args.arms
        if args.seeds:
            _ov["seeds"] = args.seeds
        wf.run_from_config(config_path, **_ov)
    else:
        print(
            f"[plgrid_test_sequence] Brak {config_path}\n"
            f"  Skopiuj wzór: {example_path}\n"
            f"  albo odpal ręcznie pojedynczy seed poniżej."
        )
        # Fallback bez configu: smoke R0 (nie kampania tezy)
        wf.run(
            arm="R0",
            experiment_version="v6",
            start_window="A1",
            end_window="B10",
            run_label="test",
            portfolio_mode="reset",
            data_files={
                "ETH": "ETHUSDT_1m_20250511_20260831.csv",
                "BTC": "BTCUSDT_1m_20250511_20260831.csv",
            },
            model_paths={
                "ETH": "LR1449_eth_model_E114_2709",
                "BTC": "LR1414_btc_model_B104_2209",
            },
            agent_rl=None,
            stop_loss=0.06,
            mincost=12.0,
            mp_alloc_scale=0.70,
            mp_rebalance_deadzone=0.25,
            mp_alloc_bias=0.9990106449716608,
            mp_confidence=0.06696638913377573,
            mp_urgency=0.629175066726628,
            mp_signal_deadzone=0.16138637516790336,
            wallet_percentage_limit=0.7829232963447843,
            max_wallet_percentage_crypto=0.7141785451510417,
            mp_ema_alpha_min=0.02,
            mp_ema_alpha_span=0.08,
            optuna_n_trials=30,
            optuna_rolling_windows=3,
            continue_timesteps=200_000,
            learning_rate=1e-4,
            seed=396379,
            plots=True,
            logs=False,
            sim_results=True,
            quiet=True,
            resume=True,
        )
