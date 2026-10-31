"""Entry CLI: symulacje na wytrenowanym agencie PPO."""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rl_module.services.train_agent_ppo_service import TrainAgentPPOService
from test_module.services.simulation_service import SimulationService

if __name__ == "__main__":
    # Pliki ml
    ml_btc_path = "btc_model_c_v1_1210_11.pkl"

    # Pliki data agent
    data_btc_agent_path = "processed_ROBUST_set4_BTCUSDT-agent_1m_5m_2025-10-11_2d07d1b6-baf3-4ed8-bdc0-390156e92dc6.csv"

    # Pliki data simulation
    data_btc_sim_path = "processed_ROBUST_set4_BTCUSDT-simulation_1m_2m_2025-10-11_8a076fe5-6d3c-4b1c-a490-9be3a3174517.csv"
    #data_btc_sim_path = "processed_ROBUST_set4_BTCUSDT_1s_2m_2026-01-11_9d42fe9e-b4ad-413f-ad84-aaaaaf53705c.csv"

    agent_return = TrainAgentPPOService(BASE_DIR).train_agent_ppo(
        rabbitmq_msg_id=None,
        model_files={
            "BTC": ml_btc_path,
        },
        data_files={
			"BTC": data_btc_agent_path
		},
        live_logging=True,
        total_timesteps=5000,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.97,
        ent_coef=0.05
    )
    agent_path = agent_return["agent_file"]

    SimulationService(BASE_DIR).run_simulation(
        rabbit_msg_id=None,
        model_paths={
            "BTC": ml_btc_path,
        },
        data_files={
            "BTC": data_btc_sim_path
        },
        agent_rl=agent_path,
        balance_start=1000.0,
        add_fee=True,
        sim_mode="rl",
        rlagent_deterministic=False,
        slippage=0,
        wallet_percentage_limit=0.9,
        max_wallet_percentage_crypto=1.0,
        stop_loss=1.0,
        threshold_buy=0.5,
        threshold_sell=0.5,
        mincost=1.0,
        tx_fraction=0.01,
        max_fraction_when_confident=0.5,
        plots=True,
        logs=True,
        results=True,
        experiment_version="v0"
    )

