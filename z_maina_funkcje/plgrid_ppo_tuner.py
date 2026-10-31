"""
Osobny entry: Optuna hiperparametrów PPO (warsztat, 1×).
Wynik → shared/agents/optuna_studies/ppo_hparams_lock.json
NIE odpala sekwencji ani treningów seedowych.
"""
import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rl_module.services.train_agent_ppo_tuner_service import PPOTunerService
from rl_module.services.thesis_artifacts import make_optuna_artifact, write_json, extract_best_params

if __name__ == "__main__":

    SEED = 265  # seed study (nie linia OOS)
    stamp = datetime.now().strftime("%d-%m__%H-%M-%S")
    study_name = f"ppo_hparams_{stamp}_seed{SEED}"

    from rl_module.services.thesis.assets import THESIS_DATA_FILES, THESIS_MODEL_PATHS

    tuner = PPOTunerService(BASE_DIR)
    study = tuner.run_tuning(
        train_data_files=dict(THESIS_DATA_FILES),
        eval_data_files=dict(THESIS_DATA_FILES),
        model_files=dict(THESIS_MODEL_PATHS),
        config={
            "episode_timesteps": ["C", 10080],
            "learning_rate": ["FL", 1e-5, 3e-4],
            "gamma": ["F", 0.95, 0.999],
            "batch_size": ["C", 256, 512],
            "n_steps": ["C", 2048, 4096],
            "n_epochs": ["C", 10],
            "clip_range": ["C", 0.2],
            "ent_coef": ["F", 0.001, 0.02],
            "mp_threshold": ["C", 0.50],
            "total_timesteps": ["C", 400_000],
            "window": ["C", "train"],
        },
        n_trials=8,
        live_logging=True,
        experiment_version="v6",
        study_name=study_name,
        seed=SEED,
    )

    best_path = getattr(tuner, "best_params_file", None)
    if best_path and os.path.isfile(best_path):
        import json
        with open(best_path, encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {
            "best_params": dict(study.best_params) if study is not None else {},
            "best_metric": getattr(study, "best_value", None) if study else None,
            "best_trial": study.best_trial.number if study and study.best_trial else None,
        }

    lock_params = extract_best_params(payload) if isinstance(payload, dict) else dict(payload)
    lock_path = os.path.join(
        BASE_DIR, "shared", "agents", "optuna_studies", "ppo_hparams_lock.json",
    )
    art = make_optuna_artifact(
        seed=SEED,
        study_name=study_name,
        best_params=lock_params,
        best_metric=payload.get("best_metric") if isinstance(payload, dict) else None,
        best_trial=payload.get("best_trial") if isinstance(payload, dict) else None,
        experiment_version="v6",
        window="train",
        algorithm="ppo",
        extra={"role": "PPO_HPARAMS_LOCK", "source_study": study_name},
    )
    write_json(lock_path, art)
    print(f"\n=== PPO hparams LOCK → {lock_path} ===")
    print("  Użyj w plgrid_train_agent_ppo (wszystkie seedy OOS na tych hparams).")
