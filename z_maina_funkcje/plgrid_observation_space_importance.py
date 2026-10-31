"""Entry CLI: analiza ważności wymiarów przestrzeni obserwacji."""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rl_module.services.obs_importance_service import ObsImportanceService

if __name__ == "__main__":

    ObsImportanceService(BASE_DIR).run_analysis(
        rabbit_msg_id=None,
        model_paths={
            "ETH": "X065_eth_xgb_model_E103_0903"
        },
        data_files={
            "ETH": "sym.csv"
        },
        agent_rl="agent_ppo_2026-03-30_18-29-15.zip",
        balance_start=1000.0,
        add_fee=True,
        rlagent_deterministic=True,
        slippage=0,
        wallet_percentage_limit=0.959,
        max_wallet_percentage_crypto=0.967,
        stop_loss=1.0,
        threshold_buy=0.5,
        threshold_sell=0.5,
        mincost=4.30,
        tx_fraction=0.0113,
        max_fraction_when_confident=0.369,
        experiment_version="v5",
        # Parametry analizy
        t_start=1,
        t_end=50000,
        step=1,         # co ile kroków próbkować (mniejszy = dokładniej, wolniej)
        method="both",   # "perturbation" | "gradient" | "both"
        save_plot=True,
        save_json=True,
    )