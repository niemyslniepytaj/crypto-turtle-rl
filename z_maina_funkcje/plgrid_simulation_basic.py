"""Entry CLI: jednorazowa / podstawowa symulacja backtestu."""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from test_module.services.simulation_service import SimulationService

if __name__ == "__main__":
    # Warsztat poza protokołem tezy — knoby poniżej nie są lockiem R1.
    SimulationService(BASE_DIR).run_simulation(
        rabbit_msg_id=None,
        model_paths={
            "ETH": "LR1449_eth_model_E114_2709",
            "BTC": "LR1414_btc_model_B104_2209",
        },
        data_files={
            "ETH": "ETHUSDT_1m_20250511_20260831.csv",
            "BTC": "BTCUSDT_1m_20250511_20260831.csv",
        },
        agent_rl="agent_ppo_final_2026-10-06_20-29-19.zip",
        balance_start=1000.0,
        add_fee=True,
        sim_mode="mp",
        rlagent_deterministic=True,
        slippage=0,
        threshold_buy=0.5,
        threshold_sell=0.5,
        tx_fraction=0.06659684773593486,
        max_fraction_when_confident=0.20091679340937524,
        stop_loss_cooldown=120,
        stop_loss=0.06,
        mincost=12.0,
        wallet_percentage_limit=0.85,
        max_wallet_percentage_crypto=0.5,
        mp_alloc_bias=0.8912201604930934,
        mp_alloc_scale=0.3960287571347845,
        mp_confidence=0.4194913532017072,
        mp_urgency=0.07107010406757262,
        mp_signal_deadzone=0.30570861095350077,
        mp_rebalance_deadzone=0.341989459895197,
        mp_ema_alpha_min=0.02,
        mp_ema_alpha_span=0.08,
        plots=True,
        logs=True,
        results=True,
        experiment_version="v6",
        window_id="A1",
    )
