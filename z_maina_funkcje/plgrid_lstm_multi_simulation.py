"""Entry CLI: multi-symulacja lub Optuna po katalogu modeli LSTM."""

import os
import sys
import json
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from test_module.services.simulation_service import SimulationService
from rl_module.services.simulation_tuner_service import SimulationTunerService

# Target directories for rankings
OUTPUT_DIR = os.path.join(BASE_DIR, "shared", "simulation", "lstm_lr_rankings")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path to the LSTM models directory
LSTM_LR_MODELS_DIR = os.path.join(BASE_DIR, "shared", "models", "lstm_lr_models")

def get_available_models():
    """Scan the lstm_lr_models directory and return list of model folder names."""
    if not os.path.exists(LSTM_LR_MODELS_DIR):
        print(f"[Warning] Models directory {LSTM_LR_MODELS_DIR} does not exist.")
        return []
    models = [d for d in os.listdir(LSTM_LR_MODELS_DIR) if os.path.isdir(os.path.join(LSTM_LR_MODELS_DIR, d))]
    # Sort them to be deterministic
    models.sort()
    return models

def resolve_model_symbol_and_data(model_name):
    """Determine asset symbol and corresponding data file based on model name."""
    model_name_lower = model_name.lower()
    if "btc" in model_name_lower:
        return "BTC", "btc_test.csv"
    elif "eth" in model_name_lower:
        return "ETH", "xgb_eth_raw8_short.csv"
    else:
        # Default fallback
        return "ETH", "xgb_eth_raw8_short.csv"

def run_multi_simulation(models_to_run=None, limit_models=None):
    """
    Run simulation with static parameters for all specified models.
    Create a ranking of models sorted by final portfolio value descending.
    """
    all_models = get_available_models()
    if not all_models:
        print("No models found to simulate.")
        return []

    if models_to_run:
        models = [m for m in all_models if m in models_to_run]
    else:
        models = all_models

    if limit_models:
        models = models[:limit_models]

    results = []
    print(f"\n[Multi-Simulation] Starting evaluation of {len(models)} models...")

    for i, model_name in enumerate(models):
        symbol, data_file = resolve_model_symbol_and_data(model_name)
        print(f"[{i+1}/{len(models)}] Simulating model: {model_name} (Symbol: {symbol}, Data: {data_file})")
        
        try:
            sim_res = SimulationService(BASE_DIR).run_simulation(
                rabbit_msg_id=None,
                model_paths={symbol: model_name},
                data_files={symbol: data_file},
                agent_rl=None,
                balance_start=1000.0,
                add_fee=True,
                sim_mode="mp",
                rlagent_deterministic=True,
                slippage=0,
                wallet_percentage_limit=0.9,
                max_wallet_percentage_crypto=1.0,
                stop_loss=1.0,
                threshold_buy=0.5,
                threshold_sell=0.5,
                mincost=1.0,
                tx_fraction=0.01,
                max_fraction_when_confident=0.5,
                plots=False,
                logs=False,
                results=True,
                experiment_version="v0"
            )
            
            if sim_res and sim_res.get("result") == "success":
                final_val = sim_res.get("final_portfolio_value", 1000.0)
                sharpe = sim_res.get("sharpe_ratio", 0.0)
                max_dd = sim_res.get("max_drawdown", 0.0)
                
                results.append({
                    "model_name": model_name,
                    "symbol": symbol,
                    "final_portfolio_value": final_val,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": max_dd,
                    "status": "success"
                })
                print(f"  -> SUCCESS: Portfolio Value = ${final_val:.2f}, Sharpe = {sharpe:.2f}, Max DD = {max_dd:.2f}%")
            else:
                msg = sim_res.get("message", "Unknown error") if sim_res else "No response"
                results.append({
                    "model_name": model_name,
                    "symbol": symbol,
                    "status": "failed",
                    "error": msg
                })
                print(f"  -> FAILED: {msg}")

        except Exception as e:
            results.append({
                "model_name": model_name,
                "symbol": symbol,
                "status": "failed",
                "error": str(e)
            })
            print(f"  -> EXCEPTION: {e}")

    # Rank by final portfolio value descending (only successful simulations)
    rankings = [r for r in results if r["status"] == "success"]
    rankings.sort(key=lambda x: x["final_portfolio_value"], reverse=True)
    
    # Append failed models to the end
    failures = [r for r in results if r["status"] != "success"]
    final_ranking = rankings + failures

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filepath = os.path.join(OUTPUT_DIR, f"simulation_ranking_{timestamp}.json")
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(final_ranking, f, indent=2)

    # Print summary table
    print("\n" + "="*80)
    print(f"SIMULATION RANKING (Saved to {output_filepath})")
    print("="*80)
    print(f"{'Rank':<5} | {'Model Name':<45} | {'Symbol':<6} | {'Final Value':<12} | {'Sharpe':<8} | {'Max DD':<8}")
    print("-"*80)
    for idx, r in enumerate(rankings):
        print(f"{idx+1:<5} | {r['model_name']:<45} | {r['symbol']:<6} | ${r['final_portfolio_value']:<11.2f} | {r['sharpe_ratio']:<8.2f} | {r['max_drawdown']:<7.2f}%")
    if failures:
        print("-"*80)
        print("FAILED MODELS:")
        for r in failures:
            print(f"FAIL  | {r['model_name']:<45} | {r['symbol']:<6} | Error: {r.get('error')}")
    print("="*80 + "\n")

    return final_ranking


def run_multi_tuner(models_to_run=None, limit_models=None, n_trials=10):
    """
    Run Optuna hyperparameter tuning in a loop for all specified models.
    Create a ranking of models based on the best tuned metric.
    """
    all_models = get_available_models()
    if not all_models:
        print("No models found to tune.")
        return []

    if models_to_run:
        models = [m for m in all_models if m in models_to_run]
    else:
        models = all_models

    if limit_models:
        models = models[:limit_models]

    results = []
    print(f"\n[Multi-Tuner] Starting hyperparameter tuning of {len(models)} models with {n_trials} trials each...")

    # Default search space
    tuning_config = {
        "balance_start":                ["C", 1000.0],
        "threshold_buy":                ["F", 0.2, 0.8],
        "threshold_sell":               ["F", 0.2, 0.8],
        "wallet_percentage_limit":      ["F", 0.6, 1.0],
        "max_wallet_percentage_crypto": ["F", 0.8, 1.0],
        "stop_loss":                    ["C", 1.0],
        "tx_fraction":                  ["F", 0.0, 0.3],
        "max_fraction_when_confident":  ["F", 0.2, 1.0],
        "slippage":                     ["C", 0.0],
        "mincost":                      ["F", 1.0, 100.0],
    }

    for i, model_name in enumerate(models):
        symbol, data_file = resolve_model_symbol_and_data(model_name)
        print(f"[{i+1}/{len(models)}] Tuning model: {model_name} (Symbol: {symbol}, Data: {data_file})")
        
        try:
            study_name = f"tune_{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
            study = SimulationTunerService(BASE_DIR).run_tuning(
                model_files={symbol: model_name},
                eval_data_files={symbol: data_file},
                config=tuning_config,
                n_trials=n_trials,
                study_name=study_name,
                live_logging=False,
                experiment_version="v0"
            )
            
            best_value = study.best_value
            best_params = study.best_params
            
            # Retrieve detailed metrics from the best trial
            trial = study.best_trial
            final_val = trial.user_attrs.get('final_value', 1000.0)
            sharpe = trial.user_attrs.get('sharpe_ratio', 0.0)
            max_dd = trial.user_attrs.get('max_drawdown', 0.0)

            results.append({
                "model_name": model_name,
                "symbol": symbol,
                "best_return_metric": best_value,
                "best_portfolio_value": final_val,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "best_params": best_params,
                "status": "success"
            })
            print(f"  -> SUCCESS: Best Return Metric = {best_value:.2f}%, Sharpe = {sharpe:.2f}, Max DD = {max_dd:.2f}%")
            
        except Exception as e:
            results.append({
                "model_name": model_name,
                "symbol": symbol,
                "status": "failed",
                "error": str(e)
            })
            print(f"  -> EXCEPTION: {e}")

    # Rank by best return metric descending
    rankings = [r for r in results if r["status"] == "success"]
    rankings.sort(key=lambda x: x["best_return_metric"], reverse=True)
    
    # Append failures
    failures = [r for r in results if r["status"] != "success"]
    final_ranking = rankings + failures

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filepath = os.path.join(OUTPUT_DIR, f"tuner_ranking_{timestamp}.json")
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(final_ranking, f, indent=2)

    # Print summary table
    print("\n" + "="*80)
    print(f"TUNER RANKING (Saved to {output_filepath})")
    print("="*80)
    print(f"{'Rank':<5} | {'Model Name':<45} | {'Symbol':<6} | {'Best Return':<12} | {'Final Value':<12} | {'Sharpe':<8}")
    print("-"*80)
    for idx, r in enumerate(rankings):
        print(f"{idx+1:<5} | {r['model_name']:<45} | {r['symbol']:<6} | {r['best_return_metric']:<10.2f}% | ${r['best_portfolio_value']:<11.2f} | {r['sharpe_ratio']:<8.2f}")
    if failures:
        print("-"*80)
        print("FAILED MODELS:")
        for r in failures:
            print(f"FAIL  | {r['model_name']:<45} | {r['symbol']:<6} | Error: {r.get('error')}")
    print("="*80 + "\n")

    return final_ranking

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LSTM Models Multi-Simulation & Tuning Runner")
    parser.add_argument("--all", action="store_true", help="Run on all available models in the folder")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of models to execute")
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning instead of static simulation")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials per model when tuning")
    
    args = parser.parse_args()
    
    available = get_available_models()
    print(f"Found {len(available)} models in total.")
    
    if not available:
        print("No models available to run.")
    else:
        if args.all:
            models_to_run = available
            if args.limit:
                models_to_run = models_to_run[:args.limit]
            print(f"Running on models (Count: {len(models_to_run)})")
            if args.tune:
                run_multi_tuner(models_to_run=models_to_run, n_trials=args.trials)
            else:
                run_multi_simulation(models_to_run=models_to_run)
        else:
            # Default verification test (2 models)
            test_models = []
            eth_model = next((m for m in available if "eth" in m.lower()), None)
            btc_model = next((m for m in available if "btc" in m.lower()), None)
            
            if eth_model:
                test_models.append(eth_model)
            if btc_model:
                test_models.append(btc_model)
                
            if not test_models:
                test_models = available[:2]
                
            if args.limit:
                test_models = test_models[:args.limit]
                
            print(f"No arguments provided. Running default verification test on models: {test_models}")
            if args.tune:
                print(f"\n--- Running Optuna Tuner ({args.trials} trials) ---")
                run_multi_tuner(models_to_run=test_models, n_trials=args.trials)
            else:
                print("\n--- Running Multi-Simulation ---")
                run_multi_simulation(models_to_run=test_models)
