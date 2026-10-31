"""
R1 LOCK — Optuna MP flat 6F na oknie train.

Nazwy locków:
  domyślnie:  r1_lock_seed{S}[_mwpc…][_sl…].json
  --lock-tag final → r1_lock_final_seed{S}[_mwpc…][_sl…].json
    (NIGDY nie nadpisuje r1_lock_seed{S}.json)

Przykłady:
  python z_maina_funkcje/plgrid_simulation_tuner.py --seed 396379
  python z_maina_funkcje/plgrid_simulation_tuner.py --all-seeds --stop-loss 0.03 --lock-tag final
  python z_maina_funkcje/plgrid_simulation_tuner.py --seed 396379 --max-wallet 0.75
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rl_module.services.simulation_tuner_service import SimulationTunerService
from rl_module.services.thesis.assets import (
    THESIS_DATA_FILES,
    THESIS_MODEL_PATHS,
    THESIS_SEEDS,
    format_mwpc_tag,
    format_sl_tag,
)
from rl_module.services.thesis.knobs import (
    MP_CONST,
    apply_tuner_settings,
    build_optuna_config,
)
from rl_module.services.thesis_artifacts import make_optuna_artifact, write_json


def _safe_tag(tag: str) -> str:
    t = re.sub(r"[^A-Za-z0-9_\-]+", "", str(tag).strip())
    if not t:
        raise ValueError("pusty --lock-tag")
    return t


def _exp_suffix(max_wallet: float, stop_loss: float) -> str:
    """Sufiksy gdy wartość ≠ MP_CONST (czytelność / unik nadpisania baseline)."""
    parts = []
    if abs(float(max_wallet) - float(MP_CONST["max_wallet_percentage_crypto"])) >= 1e-12:
        parts.append(format_mwpc_tag(max_wallet))
    if abs(float(stop_loss) - float(MP_CONST["stop_loss"])) >= 1e-12:
        parts.append(format_sl_tag(stop_loss))
    return ("_" + "_".join(parts)) if parts else ""


def _lock_basename(
    seed: int,
    max_wallet: float,
    stop_loss: float,
    lock_tag: str | None = None,
) -> str:
    suffix = _exp_suffix(max_wallet, stop_loss)
    if lock_tag:
        return f"r1_lock_{_safe_tag(lock_tag)}_seed{seed}{suffix}.json"
    return f"r1_lock_seed{seed}{suffix}.json"


def run_r1_lock(
    seed: int,
    n_trials: int,
    max_wallet: float,
    stop_loss: float,
    lock_tag: str | None = None,
) -> str:
    """Uruchamia Optunę MP 6F na oknie train i zapisuje plik locka R1; zwraca ścieżkę JSON."""
    tuner = SimulationTunerService(BASE_DIR)
    apply_tuner_settings(tuner)

    mw = float(max_wallet)
    sl = float(stop_loss)
    suffix = _exp_suffix(mw, sl)
    tag_part = f"_{_safe_tag(lock_tag)}" if lock_tag else ""
    stamp = datetime.now().strftime("%d-%m__%H-%M-%S")
    study_name = f"simulation_tuning_{stamp}_seed{seed}_6f{tag_part}{suffix}"

    eval_data_files = dict(THESIS_DATA_FILES)
    model_files = dict(THESIS_MODEL_PATHS)
    config = build_optuna_config(
        balance_start=1000.0,
        window="train",
        n_symbols=len(eval_data_files),
        max_wallet_percentage_crypto=mw,
        stop_loss=sl,
    )

    print(
        f"\n=== Optuna MP flat 6F  trials={n_trials}  seed={seed}  "
        f"maxc={mw}  sl={sl}  lock_tag={lock_tag or '—'}  study={study_name} ==="
    )
    tuner.run_tuning(
        eval_data_files=eval_data_files,
        model_files=model_files,
        config=config,
        n_trials=n_trials,
        live_logging=True,
        experiment_version="v6",
        study_name=study_name,
        seed=seed,
        staged=False,
    )

    best_path = os.path.join(
        BASE_DIR, "shared", "agents", "optuna_studies", f"{study_name}_best_params.json",
    )
    with open(best_path, encoding="utf-8") as f:
        payload = json.load(f)
    lock_params = dict(payload.get("best_params") or {})
    lock_params["max_wallet_percentage_crypto"] = mw
    lock_params["stop_loss"] = sl
    lock_name = _lock_basename(seed, mw, sl, lock_tag=lock_tag)
    # twarda ochrona: lock_tag nigdy nie celuje w baseline r1_lock_seed{S}.json
    if lock_tag and lock_name == f"r1_lock_seed{seed}.json":
        raise RuntimeError(f"odmowa nadpisania baseline: {lock_name}")
    lock_path = os.path.join(
        BASE_DIR, "shared", "agents", "optuna_studies", lock_name,
    )
    lock_artifact = make_optuna_artifact(
        seed=seed,
        study_name=study_name,
        best_params=lock_params,
        best_metric=payload.get("best_metric"),
        best_trial=payload.get("best_trial"),
        static_params=payload.get("static_params") or {},
        experiment_version="v6",
        window="train",
        algorithm="simulation",
        extra={
            "role": "R1_LOCK_FINAL" if (lock_tag or "").lower() == "final" else "R1_LOCK",
            "lock_tag": lock_tag,
            "optuna_space": "flat_6f",
            "source_study": study_name,
            "best_params_path": best_path,
            "max_wallet_percentage_crypto": mw,
            "stop_loss": sl,
            "exp_suffix": suffix or None,
        },
    )
    write_json(lock_path, lock_artifact)
    print(f"\n=== R1 LOCK (seed={seed} maxc={mw} sl={sl} tag={lock_tag or '—'}) → {lock_path} ===")
    print(
        f"  by_seed.{seed}.best_params_path = "
        f"shared/agents/optuna_studies/{lock_name}"
    )
    return lock_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R1 LOCK — Optuna MP flat 6F")
    parser.add_argument(
        "--seed", type=int, action="append", default=None,
        help="Seed study (można powtórzyć). Domyślnie: pierwszy z THESIS_SEEDS",
    )
    parser.add_argument(
        "--all-seeds", action="store_true",
        help=f"Pętla po THESIS_SEEDS {list(THESIS_SEEDS)}",
    )
    parser.add_argument("--n-trials", type=int, default=150)
    parser.add_argument(
        "--max-wallet",
        type=float,
        default=float(MP_CONST["max_wallet_percentage_crypto"]),
        help=(
            "max_wallet_percentage_crypto (CONST). "
            f"Domyślnie {MP_CONST['max_wallet_percentage_crypto']}."
        ),
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=float(MP_CONST["stop_loss"]),
        help=(
            "stop_loss (CONST). "
            f"Domyślnie {MP_CONST['stop_loss']}. "
            "Z --lock-tag final nie rusza r1_lock_seed{{S}}.json."
        ),
    )
    parser.add_argument(
        "--lock-tag",
        default=None,
        help=(
            "Osobny plik locka: r1_lock_{tag}_seed{S}[_sl…].json. "
            "Np. --lock-tag final → r1_lock_final_seed396379_sl003.json"
        ),
    )
    args = parser.parse_args()

    if args.all_seeds:
        seeds = list(THESIS_SEEDS)
    elif args.seed:
        seeds = list(args.seed)
    else:
        seeds = [THESIS_SEEDS[0]]

    for i, seed in enumerate(seeds):
        print(
            f"\n######## R1 LOCK  seed={seed}  maxc={args.max_wallet}  "
            f"sl={args.stop_loss}  tag={args.lock_tag or '—'}  "
            f"({i + 1}/{len(seeds)}) ########"
        )
        run_r1_lock(
            seed=int(seed),
            n_trials=int(args.n_trials),
            max_wallet=float(args.max_wallet),
            stop_loss=float(args.stop_loss),
            lock_tag=args.lock_tag,
        )
