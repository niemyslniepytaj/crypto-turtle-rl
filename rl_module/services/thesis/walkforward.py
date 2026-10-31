"""
Walk-forward orchestrator — ramiona tezy R0–M3.

API: jedno ramię na wywołanie (arm=R0|R1|R2|M1|M2|M3), zakres okien
start_window..end_window. Bez sim_mode w API — mapuje się z ramienia.

  R0  HODL
  R1  v6+mp, knoby LOCK (Optuna na train poza sekwencją)
  R2  v6+rl, zamrożony agent
  M1  v6+mp, Optuna rolling 3 okna przed każdym foldem
  M2  v6+rl, continue-learn na poprzednim oknie przed następnym eval
  M3  v6+rlul, continue-learn + UL k-means (reżim one-hot)
"""

from __future__ import annotations

import gzip
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

import time

from test_module.services.core_logic.windows import load_windows, parse_window_ids, slice_windows
from test_module.services.simulation_service import SimulationService
from test_module.services.test.rating import Rating
from test_module.services.test.statistics_service import StatisticsService
from rl_module.services.thesis.export import (
    b_folds_only,
    copy_gzip_to_run,
    make_run_id,
    sequence_b_gzip_basename,
    stitch_sequence_payload,
    window_gzip_basename,
    write_viewer_gzip,
)
from rl_module.services.thesis.arms import (
    ARM_META,
    CANONICAL_ARMS,
    IMPLEMENTED_ARMS,
    PLACEHOLDER_ARMS,
    normalize_arm,
)
from rl_module.services.thesis.knobs import (
    CLAMP_WALLET_CAP_TO_1N,
    HODL_ENTRY_FEE_DEFAULT,
    HODL_ENTRY_FEE_RATE,
    HODL_WARMUP_TICKS,
    MP_CONST as _KC,
    PPO_HPARAMS,
    V5RL_KNOB_HOLD_BARS,
    V5RL_TRADE_LOCK_BARS,
    apply_tuner_settings,
    build_optuna_config,
    default_mp_params,
    tuner_settings_snapshot,
)
from rl_module.services.thesis import console as thesis_console

_SIM_KWARGS = {
    "balance_start", "add_fee", "slippage", "wallet_percentage_limit",
    "max_wallet_percentage_crypto", "stop_loss", "stop_loss_cooldown",
    "threshold_buy", "threshold_sell", "mincost", "tx_fraction",
    "max_fraction_when_confident", "mp_alloc_bias", "mp_alloc_scale",
    "mp_confidence", "mp_urgency", "mp_signal_deadzone", "mp_rebalance_deadzone",
    "mp_ema_alpha_min", "mp_ema_alpha_span", "rlagent_deterministic",
    "v5rl_knob_hold_bars", "v5rl_trade_lock_bars",
    "clamp_wallet_cap_to_1n",
}

def _filter_sim_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in (params or {}).items() if k in _SIM_KWARGS}


def expanding_train_spec(windows: dict, eval_fold: str) -> str:
    """All windows in sequence strictly before eval_fold, joined with '+'."""
    seq = list(windows.get("sequence") or [])
    if eval_fold not in seq:
        raise KeyError(f"Fold {eval_fold} not in sequence={seq}")
    idx = seq.index(eval_fold)
    if idx == 0:
        raise ValueError(f"Cannot build train window before first fold {eval_fold}")
    return "+".join(seq[:idx])


def rolling_train_spec(windows: dict, eval_fold: str, n: int = 3) -> str:
    """Last n windows in sequence strictly before eval_fold (M1 ≈ 3 months)."""
    seq = list(windows.get("sequence") or [])
    if eval_fold not in seq:
        raise KeyError(f"Fold {eval_fold} not in sequence={seq}")
    idx = seq.index(eval_fold)
    prior = seq[:idx]
    if not prior:
        raise ValueError(f"Cannot build rolling train before first fold {eval_fold}")
    roll = prior[-int(n):] if n > 0 else prior
    return "+".join(roll)


def resolve_folds_range(
    windows: dict,
    start_window: Optional[str] = None,
    end_window: Optional[str] = None,
    eval_folds: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Resolve eval folds from start/end (inclusive) on sequence, or explicit list.
    Default thesis OOS: B1..B10.
    """
    seq = list(windows.get("sequence") or [])
    if eval_folds is not None:
        out: List[str] = []
        for spec in eval_folds:
            for wid in parse_window_ids(spec) or []:
                if wid not in windows["windows"]:
                    raise KeyError(f"Unknown eval fold: {wid}")
                out.append(wid)
        return out

    if start_window is None and end_window is None:
        return [w for w in seq if str(w).startswith("B")]

    if start_window is None or end_window is None:
        raise ValueError("Provide both start_window and end_window, or neither")
    if start_window not in seq:
        raise KeyError(f"start_window {start_window} not in sequence")
    if end_window not in seq:
        raise KeyError(f"end_window {end_window} not in sequence")
    i0, i1 = seq.index(start_window), seq.index(end_window)
    if i1 < i0:
        raise ValueError(f"end_window {end_window} before start_window {start_window}")
    return seq[i0 : i1 + 1]


def resolve_eval_folds(windows: dict, eval_folds: Optional[Sequence[str]] = None) -> List[str]:
    """Backward-compat wrapper."""
    return resolve_folds_range(windows, eval_folds=eval_folds)


class WalkForwardService:
    """Orkiestracja walk-forward tezy: foldy, ramiona, artefakty i preflight."""
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.windows = load_windows(base_dir)
        self.data_dir = os.path.join(base_dir, "shared", "files", "processed_simulation_data")
        self.results_root = os.path.join(base_dir, "shared", "simulation", "thesis_results")
        self.rating = Rating()
        self.stats = StatisticsService()

    # ------------------------------------------------------------------ public

    def run_from_config(
        self,
        config_path: str,
        **overrides,
    ) -> List[Dict[str, Any]]:
        """
        Odpal sekwencję z JSON (seeds + by_seed.best_params_path / agent_rl).

        Dla każdego seeda osobny run_id: `{prefix}_seed{seed}_{stamp}`.
        """
        from rl_module.services.thesis_artifacts import (
            load_sequence_config,
            seed_bundle,
            write_json,
        )

        cfg = load_sequence_config(config_path, self.base_dir)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = overrides.pop("run_id_prefix", None) or cfg.get("run_id_prefix") or "wf"
        summaries: List[Dict[str, Any]] = []

        common = {
            "experiment_version": cfg.get("experiment_version", "v6"),
            "start_window": cfg.get("start_window"),
            "end_window": cfg.get("end_window"),
            "eval_folds": cfg.get("eval_folds"),
            "data_files": cfg.get("data_files"),
            "model_paths": cfg.get("model_paths"),
            "balance_start": float(cfg.get("balance_start", 1000.0)),
            "optuna_n_trials": int(cfg.get("optuna_n_trials", 50)),
            "optuna_rolling_windows": int(cfg.get("optuna_rolling_windows", 3)),
            "optuna_config": cfg.get("optuna_config") or {},
            "continue_timesteps": int(cfg.get("continue_timesteps", 200_000)),
            # M2/M3 continue: stałe 1e-4 (config) — świadomie w przedziale
            # pretrainu PPO_HPARAMS learning_rate_start→end (2e-4→4e-5); nie schedule.
            "learning_rate": float(cfg.get("learning_rate", 1e-4)),
            "plots": bool(cfg.get("plots", True)),
            "logs": bool(cfg.get("logs", False)),
            "sim_results": bool(cfg.get("sim_results", True)),
            "quiet": bool(cfg.get("quiet", True)),
            "resume": bool(cfg.get("resume", True)),
            "num_envs": int(cfg.get("num_envs", 4)),
            "skip_placeholders": bool(cfg.get("skip_placeholders", True)),
            "run_label": cfg.get("run_label") or cfg.get("label") or "test",
            "portfolio_mode": cfg.get("portfolio_mode") or "reset",
        }
        # drop Nones so run() defaults apply
        common = {k: v for k, v in common.items() if v is not None}
        common.update({k: v for k, v in overrides.items() if v is not None})
        common.pop("seeds", None)

        arms = cfg.get("arms")
        arm = cfg.get("arm", "R1")
        if "arms" in overrides:
            arms = overrides["arms"]
        if "arm" in overrides:
            arm = overrides["arm"]
        arms_list = list(arms) if arms else [arm]

        seeds_list = overrides.get("seeds") or cfg["seeds"]
        for seed in seeds_list:
            bundle = seed_bundle(cfg, seed)
            for arm_id in arms_list:
                # R0 = HODL, nie bot: deterministyczny, liczony raz (pierwszy seed z configu)
                if str(arm_id).upper() == "R0" and int(seed) != int(cfg["seeds"][0]):
                    print(f"[WF] pomijam R0 dla seed={seed} (HODL liczony raz)")
                    continue
                print(f"\n[WF] === sequence arm={arm_id} seed={seed} ===")
                run_kwargs = dict(common)
                run_kwargs["arm"] = arm_id
                run_kwargs.pop("arms", None)
                summary = self.run(
                    seed=int(seed),
                    best_params_path=bundle.get("best_params_path"),
                    agent_rl=bundle.get("agent_rl"),
                    agent_rlul=bundle.get("agent_rlul") or cfg.get("agent_rlul"),
                    sequence_config_path=config_path,
                    **run_kwargs,
                )
                try:
                    run_dir = summary.get("paths", {}).get("run_dir")
                    if run_dir:
                        write_json(
                            os.path.join(run_dir, "sequence_config.json"),
                            {
                                **cfg,
                                "_resolved_seed": int(seed),
                                "_resolved_arm": arm_id,
                                "_bundle": bundle,
                            },
                        )
                except Exception as e:
                    print(f"[WF] warn: could not copy sequence_config: {e}")
                summaries.append(summary)
        return summaries

    def reaggregate(self, run_id: str) -> Dict[str, Any]:
        run_dir = os.path.join(self.results_root, run_id)
        manifest_path = os.path.join(run_dir, "manifest.json")
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(run_dir)
        manifest = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        fold_records: List[Dict[str, Any]] = []
        # rekordy okien zapisuje run() w windows/fold_{id}.json (arms/ ma tylko metadane)
        windows_dir = os.path.join(run_dir, "windows")
        if os.path.isdir(windows_dir):
            for name in sorted(os.listdir(windows_dir)):
                if not (name.startswith("fold_") and name.endswith(".json")):
                    continue
                with open(os.path.join(windows_dir, name), "r", encoding="utf-8") as f:
                    rec = json.load(f)
                if isinstance(rec, dict) and rec.get("arm") and rec.get("fold_id"):
                    fold_records.append(rec)
        return self.stats.build_and_save(
            fold_records=fold_records,
            run_dir=run_dir,
            manifest=manifest,
            baseline_arm="R0",
        )

    def run(
        self,
        *,
        arm: str = "R1",
        arms: Optional[Sequence[str]] = None,
        experiment_version: str = "v6",
        start_window: Optional[str] = None,
        end_window: Optional[str] = None,
        eval_folds: Optional[Sequence[str]] = None,
        data_files: Dict[str, str],
        model_paths: Dict[str, str],
        agent_rl: Optional[str] = None,
        agent_rlul: Optional[str] = None,
        balance_start: float = 1000.0,
        best_params_path: Optional[str] = None,
        mp_params: Optional[Dict[str, Any]] = None,
        v5mp_params: Optional[Dict[str, Any]] = None,  # legacy alias
        optuna_config: Optional[Dict[str, Any]] = None,
        optuna_n_trials: int = 50,
        optuna_rolling_windows: int = 3,
        continue_timesteps: int = 200_000,
        learning_rate: float = 1e-4,  # M2 continue: stałe LR (patrz run_from_config)
        seed: int = 0,
        run_id: Optional[str] = None,
        run_label: str = "test",
        portfolio_mode: str = "reset",  # reset | continuous
        plots: bool = True,
        logs: bool = False,
        sim_results: bool = True,
        quiet: bool = True,
        skip_placeholders: bool = True,
        resume: bool = True,
        num_envs: int = 4,
        sequence_config_path: Optional[str] = None,
        **knob_kwargs,
    ) -> Dict[str, Any]:
        """
        Jedno ramię → jeden folder wyników.
        Nazwa: {ARM}_{label}_{seed}_{YYYYMMDD}_{start}_{end}
        portfolio_mode: reset (1000$ co okno) | continuous (ten sam portfel).
        """
        from rl_module.services.thesis_artifacts import extract_seed, load_agent_artifact

        if arms is None:
            arms_list = [arm]
        else:
            arms_list = list(arms)

        arms_norm = self._normalize_arms(arms_list, skip_placeholders=skip_placeholders)
        # Wiele ramion → osobne foldery (spójne porównania)
        if len(arms_norm) > 1:
            outs = []
            for a in arms_norm:
                outs.append(
                    self.run(
                        arm=a,
                        arms=None,
                        experiment_version=experiment_version,
                        start_window=start_window,
                        end_window=end_window,
                        eval_folds=eval_folds,
                        data_files=data_files,
                        model_paths=model_paths,
                        agent_rl=agent_rl,
                        agent_rlul=agent_rlul,
                        balance_start=balance_start,
                        best_params_path=best_params_path,
                        mp_params=mp_params,
                        v5mp_params=v5mp_params,
                        optuna_config=optuna_config,
                        optuna_n_trials=optuna_n_trials,
                        optuna_rolling_windows=optuna_rolling_windows,
                        continue_timesteps=continue_timesteps,
                        learning_rate=learning_rate,
                        seed=seed,
                        run_id=None,
                        run_label=run_label,
                        portfolio_mode=portfolio_mode,
                        plots=plots,
                        logs=logs,
                        sim_results=sim_results,
                        quiet=quiet,
                        skip_placeholders=skip_placeholders,
                        resume=resume,
                        num_envs=num_envs,
                        sequence_config_path=sequence_config_path,
                        **knob_kwargs,
                    )
                )
            return {"multi_arm_runs": outs}

        arm_id = arms_norm[0]
        if arm_id in ("R3", "M3"):
            # R3/M3 używają agenta rlul (obs 20 wymiarów) - nigdy agenta R2 (18)
            if not agent_rlul:
                raise ValueError(f"{arm_id} wymaga agent_rlul (agent z etykietą UL)")
            agent_rl = agent_rlul
        folds = resolve_folds_range(
            self.windows,
            start_window=start_window,
            end_window=end_window,
            eval_folds=eval_folds,
        )
        if not folds:
            raise ValueError("No eval folds resolved")
        if not data_files:
            raise ValueError("data_files is required")

        portfolio_mode = str(portfolio_mode or "reset").lower().strip()
        _pm_aliases = {
            "reset": "reset",
            "reset_1000": "reset",
            "per_window": "reset",
            "continuous": "continuous",
            "carry": "continuous",
            "same_portfolio": "continuous",
        }
        portfolio_mode = _pm_aliases.get(portfolio_mode, portfolio_mode)
        if portfolio_mode not in ("reset", "continuous"):
            raise ValueError(
                "portfolio_mode must be 'reset' (1000$ co okno) "
                "lub 'continuous' (ten sam portfel)"
            )

        start_w = start_window or folds[0]
        end_w = end_window or folds[-1]
        run_id = run_id or make_run_id(
            arm=arm_id,
            label=run_label,
            seed=int(seed),
            start_window=start_w,
            end_window=end_w,
        )
        if resume:
            # wznowienie po zmianie daty: ten sam arm/label/seed/okna, dowolna data
            import glob as _glob
            _pat = run_id.replace(
                f"_{int(seed)}_", f"_{int(seed)}_*_", 1
            )
            _old = [
                d for d in _glob.glob(os.path.join(self.results_root, _pat))
                if os.path.isdir(os.path.join(d, "windows"))
            ]
            if _old and not os.path.isdir(os.path.join(self.results_root, run_id)):
                run_id = os.path.basename(max(_old, key=os.path.getmtime))
                print(f"[WF] wznawiam istniejacy run: {run_id}")
        run_dir = os.path.join(self.results_root, run_id)
        windows_dir = os.path.join(run_dir, "windows")
        os.makedirs(windows_dir, exist_ok=True)

        lock_params, lock_meta = self._resolve_lock_params(
            balance_start=balance_start,
            best_params_path=best_params_path,
            mp_params=mp_params,
            v5mp_params=v5mp_params,
            knob_kwargs=knob_kwargs,
            expected_seed=seed,
        )
        optuna_config = dict(optuna_config or {})

        agent_meta = None
        if agent_rl:
            agent_meta = load_agent_artifact(
                agent_rl,
                base_dir=self.base_dir,
                agents_dir=os.path.join(self.base_dir, "shared", "agents", "ppo"),
            )
            if agent_meta is not None:
                aseed = extract_seed(agent_meta)
                if aseed is not None and int(aseed) != int(seed):
                    print(
                        f"[WF] WARN agent seed={aseed} != run seed={seed} "
                        f"({agent_rl})"
                    )

        from rl_module.services.thesis.repro import build_repro_block
        from rl_module.services.thesis.knobs import (
            HODL_ENTRY_FEE_DEFAULT,
            HODL_ENTRY_FEE_RATE,
            REGIME_K_DEFAULT,
        )

        manifest = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "arm": arm_id,
            "arms": [arm_id],
            "run_label": run_label,
            "portfolio_mode": portfolio_mode,
            "experiment_version": experiment_version,
            "start_window": start_w,
            "end_window": end_w,
            "eval_folds": list(folds),
            "data_files": data_files,
            "model_paths": model_paths,
            "agent_rl": agent_rl,
            "agent_rlul": agent_rlul,
            "agent_meta_seed": extract_seed(agent_meta) if agent_meta else None,
            "balance_start": balance_start,
            "seed": seed,
            "lock_artifact_seed": lock_meta.get("seed"),
            "optuna_n_trials": optuna_n_trials,
            "optuna_rolling_windows": optuna_rolling_windows,
            "continue_timesteps": continue_timesteps,
            "learning_rate": learning_rate,
            "best_params_path": best_params_path,
            "sequence_config_path": sequence_config_path,
            "arm_meta": {arm_id: ARM_META.get(arm_id, {})},
            "windows_sequence": self.windows.get("sequence"),
            "tuner_settings": tuner_settings_snapshot(),
            "hodl_entry_fee": HODL_ENTRY_FEE_DEFAULT,
            "hodl_entry_fee_rate": HODL_ENTRY_FEE_RATE,
            "regime_k_default": REGIME_K_DEFAULT,
            "regime_lock": "shared/files/regime_lock.json",
            "regime_model_train": "shared/files/regime_model_train.joblib",
            "repro": build_repro_block(
                base_dir=self.base_dir,
                data_files=data_files,
                model_paths=model_paths,
            ),
        }
        with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        fold_records: List[Dict[str, Any]] = []
        current_agent = agent_rl
        portfolio_state = None
        sim_payloads_by_fold: Dict[str, Dict[str, Any]] = {}

        for i, fold_id in enumerate(folds):
            out_path = os.path.join(windows_dir, f"fold_{fold_id}.json")
            if resume and os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8") as f:
                    record = json.load(f)
                fold_records.append(record)
                if arm_id in ("M2", "M3"):
                    nxt = (record.get("artifacts") or {}).get("agent_after_continue")
                    if nxt:
                        current_agent = nxt
                    elif (record.get("artifacts") or {}).get("agent_used"):
                        current_agent = record["artifacts"]["agent_used"]
                if portfolio_mode == "continuous":
                    portfolio_state = (record.get("artifacts") or {}).get("portfolio_state")
                gz = (record.get("artifacts") or {}).get("gzip_path")
                if gz and os.path.isfile(gz):
                    try:
                        with gzip.open(gz, "rt", encoding="utf-8") as f:
                            sim_payloads_by_fold[fold_id] = json.load(f)
                    except Exception:
                        pass
                print(f"[WF] resume skip {arm_id}/{fold_id}")
                continue

            thesis_console.print_kv(
                "WF",
                f"arm={arm_id} fold={fold_id} mode={portfolio_mode} seed={seed}",
                kind="info",
            )
            fold_balance = balance_start
            fold_state = None
            if portfolio_mode == "continuous" and portfolio_state:
                fold_state = portfolio_state
                # R0 continuous: kontynuuj kapitał jako nowy balance_start
                if arm_id == "R0" and portfolio_state.get("balance") is not None:
                    fold_balance = float(portfolio_state["balance"])

            gzip_base = window_gzip_basename(
                arm=arm_id, label=run_label, seed=int(seed), fold_id=fold_id,
            )
            fold_sim_dir = os.path.join(windows_dir, fold_id)

            t0 = time.time()
            record, current_agent = self._run_arm_fold(
                arm=arm_id,
                fold_id=fold_id,
                fold_index=i,
                folds=folds,
                data_files=data_files,
                model_paths=model_paths,
                balance_start=fold_balance,
                experiment_version=experiment_version,
                lock_params=lock_params,
                optuna_config=optuna_config,
                optuna_n_trials=optuna_n_trials,
                optuna_rolling_windows=optuna_rolling_windows,
                agent_rl=current_agent,
                continue_timesteps=continue_timesteps,
                learning_rate=learning_rate,
                seed=seed,
                plots=plots,
                logs=logs,
                # zawsze zapisuj gzip — viewer + sequence_B
                sim_results=True,
                quiet=quiet,
                run_dir=run_dir,
                num_envs=num_envs,
                portfolio_state=fold_state,
                gzip_basename=gzip_base,
                sim_dir=fold_sim_dir,
                run_label=run_label,
            )
            wall_s = time.time() - t0
            art = record.setdefault("artifacts", {})
            art["wall_time_s"] = round(wall_s, 2)
            art["wall_time_s_total"] = round(wall_s, 2)
            # I1: jawny koszt adaptacji
            if arm_id in ("R0", "R1", "R2"):
                art.setdefault("optuna_time_s", 0.0)
                art.setdefault("train_time_s", 0.0)
                art.setdefault("ul_fit_time_s", 0.0)
                art.setdefault("adaptation_cost", 0)
            if arm_id == "M1":
                art.setdefault("optuna_trials", optuna_n_trials)
                art.setdefault("train_time_s", 0.0)
                art.setdefault("ul_fit_time_s", 0.0)
            if arm_id in ("M2", "M3"):
                art.setdefault("continue_timesteps", continue_timesteps)
                art.setdefault("optuna_time_s", 0.0)
            art.setdefault("num_envs", num_envs)
            try:
                from rl_module.services.thesis.repro import hardware_snapshot
                art.setdefault("hardware", hardware_snapshot())
            except Exception:
                pass
            # suma składowych ≈ wall (tolerancja — brakujące = 0)
            parts = [
                float(art.get("optuna_time_s") or 0),
                float(art.get("train_time_s") or 0),
                float(art.get("ul_fit_time_s") or 0),
            ]
            art["wall_time_parts_sum"] = round(sum(parts), 2)
            thesis_console.print_kv(
                "WF",
                f"done {arm_id}/{fold_id} status={record.get('status')} "
                f"wall_time_s={wall_s:.1f} excess={record.get('excess_return')}",
                kind="ok" if record.get("status") == "ok" else "warn",
            )
            # continuous: zapamiętaj stan
            if portfolio_mode == "continuous":
                portfolio_state = (record.get("artifacts") or {}).get("portfolio_state")
                if arm_id == "R0" and record.get("metrics"):
                    # HODL: następne okno startuje od końcowej wartości portfela
                    fv = record["metrics"].get("final_portfolio_value")
                    if fv is not None:
                        portfolio_state = {"balance": float(fv), "holdings": {}}

            # kopia gzip do windows/ + ścieżki w artifacts
            gz = (record.get("artifacts") or {}).get("gzip_path")
            if gz:
                copy_gzip_to_run(gz, windows_dir)
            raw_out = (record.get("artifacts") or {}).pop("sim_output", None)
            if raw_out:
                sim_payloads_by_fold[fold_id] = raw_out

            # nie dumpuj ogromnego sim_output do fold json
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            fold_records.append(record)
            status = record.get("status")
            exc = record.get("metrics", {}).get("total_return")
            print(f"[WF] done {arm_id}/{fold_id} status={status} total_return={exc}")

        # --- sekwencja B1+ (osobny plik ciągły; A1/A2 tylko jako foldy) ---
        self._export_sequence_b(
            arm_id=arm_id,
            run_label=run_label,
            seed=int(seed),
            folds=folds,
            fold_records=fold_records,
            sim_payloads_by_fold=sim_payloads_by_fold,
            run_dir=run_dir,
            portfolio_mode=portfolio_mode,
            plots=plots,
        )

        summary = self.stats.build_and_save(
            fold_records=fold_records,
            run_dir=run_dir,
            manifest=manifest,
            baseline_arm="R0",
        )
        summary["paths"] = {"run_dir": run_dir, "windows_dir": windows_dir}
        wall_sum = sum(
            float((r.get("artifacts") or {}).get("wall_time_s") or 0) for r in fold_records
        )
        status_counts = {}
        for r in fold_records:
            st = r.get("status") or "?"
            status_counts[st] = status_counts.get(st, 0) + 1
        thesis_console.print_run_summary(
            arm=arm_id,
            seed=seed,
            portfolio_mode=portfolio_mode,
            folds=folds,
            run_dir=run_dir,
            wall_times=wall_sum,
            status_counts=status_counts,
        )
        print(f"\n[WF] Artifacts written to {run_dir}")
        return summary

    # -------------------------------------------------------------- arm runners

    def _run_arm_fold(self, **kwargs) -> Tuple[Dict[str, Any], Optional[str]]:
        arm = kwargs["arm"]
        agent = kwargs.get("agent_rl")
        if arm == "R0":
            return self._run_hodl(**kwargs), agent
        if arm == "R1":
            return self._run_r1(**kwargs), agent
        if arm == "R2":
            return self._run_r2(**kwargs), agent
        if arm == "R3":
            return self._run_r3(**kwargs), agent
        if arm == "M1":
            if int(kwargs.get("fold_index", 0)) == 0:
                # A1: stan bazowy bez adaptacji (parametry R1 z lock, jak R2/R3 przed douczaniem)
                rec = self._run_r1(**kwargs)
                art = rec.setdefault("artifacts", {})
                art.update({
                    "optuna_time_s": 0.0, "optuna_trials": 0,
                    "train_time_s": 0.0, "ul_fit_time_s": 0.0,
                    "adaptation_cost": 0, "baseline_no_adaptation": True,
                })
                return rec, agent
            return self._run_m1(**kwargs), agent
        if arm == "M2":
            return self._run_m2(**kwargs)
        if arm == "M3":
            return self._run_m3(**kwargs)
        if arm in PLACEHOLDER_ARMS:
            return self._placeholder_record(
                arm=arm,
                fold_id=kwargs["fold_id"],
                train_window=None,
                reason="not_implemented",
            ), agent
        raise ValueError(f"Unknown arm: {arm}. Known: {CANONICAL_ARMS}")

    def _run_hodl(
        self,
        *,
        arm: str,
        fold_id: str,
        data_files: Dict[str, str],
        balance_start: float,
        plots: bool = True,
        sim_results: bool = True,
        gzip_basename=None,
        sim_dir=None,
        run_label: str = "test",
        seed: int = 0,
        **_kwargs,
    ) -> Dict[str, Any]:
        # Entry fee jak w wallet buy: cash_out = crypto_usd * (1 + fee)
        fee_on = bool(_kwargs.get("hodl_entry_fee", HODL_ENTRY_FEE_DEFAULT))
        fee_rate = float(_kwargs.get("hodl_entry_fee_rate", HODL_ENTRY_FEE_RATE)) if fee_on else 0.0

        closes_by_symbol: Dict[str, List[float]] = {}
        n_ticks = None
        for symbol, filename in data_files.items():
            df = self._load_sliced(filename, fold_id)
            if "close" not in df.columns:
                raise KeyError(f"{filename}: missing 'close' column")
            closes = df["close"].astype(float).tolist()
            if not closes:
                raise ValueError(f"Empty window {fold_id} for {symbol}")
            closes_by_symbol[symbol] = closes
            n_ticks = len(closes) if n_ticks is None else min(n_ticks, len(closes))

        # Ten sam zakres co w botach: bez rozbiegu cech modelu (ostatnie n świec okna).
        warmup = int(_kwargs.get("hodl_warmup_ticks", HODL_WARMUP_TICKS))
        n_window = int(n_ticks)
        n_ticks = max(n_window - warmup, 1)
        for sym in list(closes_by_symbol):
            closes_by_symbol[sym] = closes_by_symbol[sym][-n_ticks:]

        symbols = list(closes_by_symbol.keys())
        n_sym = max(len(symbols), 1)
        cash_per = float(balance_start) / float(n_sym)
        # qty po opłacie wejścia (reszta cashu = 0 — full deploy)
        qty_by_symbol: Dict[str, float] = {}
        refs: Dict[str, List[float]] = {}
        for sym in symbols:
            closes = closes_by_symbol[sym][:n_ticks]
            c0 = float(closes[0])
            crypto_usd = cash_per / (1.0 + fee_rate) if fee_rate > 0 else cash_per
            qty = crypto_usd / c0 if c0 > 0 else 0.0
            qty_by_symbol[sym] = qty
            # ref 100% w jednym coinie (z fee) — do viewer
            full_crypto = float(balance_start) / (1.0 + fee_rate) if fee_rate > 0 else float(balance_start)
            full_qty = full_crypto / c0 if c0 > 0 else 0.0
            refs[sym] = [full_qty * float(c) for c in closes]

        portfolio = [
            sum(qty_by_symbol[sym] * float(closes_by_symbol[sym][i]) for sym in symbols)
            for i in range(n_ticks)
        ]

        metrics = self.rating.rate(
            portfolio,
            transactions=1 if fee_on else 0,
            closed_trades=[],
            holdings_history=[[1.0] for _ in portfolio],
            balance_start=balance_start,
        )
        # jawny koszt wejścia (notional * fee na leg)
        entry_fee_paid = float(balance_start) * fee_rate if fee_on else 0.0
        metrics["entry_fee_paid"] = entry_fee_paid
        metrics["fee_paid"] = entry_fee_paid
        hodl_return = float(metrics.get("total_return") or 0.0)
        final_v = float(portfolio[-1]) if portfolio else float(balance_start)

        gzip_base = gzip_basename or window_gzip_basename(
            arm=arm, label=run_label, seed=int(seed), fold_id=fold_id,
        )
        safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(gzip_base))
        gzip_name = f"{safe}.json.gz"
        out_dir = os.path.join(self.base_dir, "shared", "simulation", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        gzip_path = os.path.join(out_dir, gzip_name)
        fold_sim_dir = sim_dir or os.path.join(
            self.base_dir, "shared", "simulation", "sims", safe,
        )
        os.makedirs(fold_sim_dir, exist_ok=True)

        payload = {
            "files": {"data": data_files, "models": {}, "agents": None},
            "config": {
                "arm": arm,
                "fold_id": fold_id,
                "balance_start": balance_start,
                "sim_mode": "hodl",
                "run_label": run_label,
                "seed": int(seed),
                "hodl_entry_fee": fee_on,
                "hodl_entry_fee_rate": fee_rate,
            },
            "model": {
                "metrics": metrics,
                "portfolio_values": portfolio,
                "holdings": [[1.0] for _ in portfolio],
                "timestamps": len(portfolio),
            },
            "hodl": {
                "crypto_ref_values": refs,
                "portfolio_values": portfolio,
            },
        }
        artifacts: Dict[str, Any] = {
            "portfolio_state": {"balance": final_v, "holdings": dict(qty_by_symbol)},
            "costs": {
                "entry_fee_paid": entry_fee_paid,
                "fee_paid": entry_fee_paid,
                "fee_rate": fee_rate,
            },
        }
        if sim_results or plots:
            write_viewer_gzip(gzip_path, payload)
            artifacts["gzip_path"] = gzip_path
            artifacts["gzip_name"] = gzip_name
            artifacts["sim_path"] = fold_sim_dir
            artifacts["sim_output"] = payload
            with open(os.path.join(fold_sim_dir, "sim_results.txt"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        if plots and artifacts.get("gzip_name"):
            try:
                from test_module.services.visualization_service import VisualizationService
                VisualizationService().visualize(
                    simulation_files=gzip_name,
                    sim_path=fold_sim_dir,
                )
            except Exception as e:
                print(f"[WF] HODL chart skip: {e}")

        return self._fold_record(
            arm=arm,
            fold_id=fold_id,
            train_window=None,
            eval_window=fold_id,
            params={
                "hodl_entry_fee": fee_on,
                "hodl_entry_fee_rate": fee_rate,
                "balance_start": balance_start,
            },
            metrics=metrics,
            hodl_return=hodl_return,
            excess_return=0.0,
            n_ticks=n_ticks,
            status="ok",
            artifacts=artifacts,
            extras={
                "portfolio_mode": "equal_weight_close",
                "hodl_warmup_ticks": warmup,
                "n_window_ticks": n_window,
                "hodl_entry_fee": fee_on,
                "entry_fee_paid": entry_fee_paid,
            },
        )

    def _run_r1(
        self,
        *,
        arm: str,
        fold_id: str,
        data_files: Dict[str, str],
        model_paths: Dict[str, str],
        balance_start: float,
        experiment_version: str,
        lock_params: Dict[str, Any],
        plots: bool,
        logs: bool,
        sim_results: bool,
        quiet: bool,
        portfolio_state=None,
        gzip_basename=None,
        sim_dir=None,
        **_kwargs,
    ) -> Dict[str, Any]:
        params = dict(lock_params)
        params.setdefault("balance_start", balance_start)
        sim = SimulationService(self.base_dir)
        result = sim.run_simulation(
            model_paths=model_paths,
            data_files=data_files,
            agent_rl=None,
            sim_mode="mp",
            experiment_version=experiment_version or "v6",
            window_id=fold_id,
            plots=plots,
            logs=logs,
            results=sim_results,
            quiet=quiet,
            name_tag=f"wf_R1_{fold_id}",
            portfolio_state=portfolio_state,
            gzip_basename=gzip_basename,
            sim_dir=sim_dir,
            **_filter_sim_params(params),
        )
        return self._sim_to_fold_record(
            arm=arm,
            fold_id=fold_id,
            train_window=None,
            params=params,
            sim_result=result,
            artifacts={},
        )

    def _run_r2(
        self,
        *,
        arm: str,
        fold_id: str,
        data_files: Dict[str, str],
        model_paths: Dict[str, str],
        balance_start: float,
        experiment_version: str,
        lock_params: Dict[str, Any],
        agent_rl: Optional[str],
        plots: bool,
        logs: bool,
        sim_results: bool,
        quiet: bool,
        portfolio_state=None,
        gzip_basename=None,
        sim_dir=None,
        **_kwargs,
    ) -> Dict[str, Any]:
        if not agent_rl:
            raise ValueError("R2 requires agent_rl= (pretrained frozen agent)")
        from rl_module.services.rl.obs_rms_utils import resolve_obs_rms_beside_agent

        params = dict(lock_params)
        params.setdefault("balance_start", balance_start)
        params.setdefault("rlagent_deterministic", True)
        obs_used = resolve_obs_rms_beside_agent(
            agent_rl,
            agents_dir=os.path.join(self.base_dir, "shared", "agents", "ppo"),
        )
        sim = SimulationService(self.base_dir)
        result = sim.run_simulation(
            model_paths=model_paths,
            data_files=data_files,
            agent_rl=agent_rl,
            sim_mode="rl",
            experiment_version=experiment_version or "v6",
            window_id=fold_id,
            plots=plots,
            logs=logs,
            results=sim_results,
            quiet=quiet,
            name_tag=f"wf_R2_{fold_id}",
            portfolio_state=portfolio_state,
            gzip_basename=gzip_basename,
            sim_dir=sim_dir,
            obs_rms_file=obs_used,
            **_filter_sim_params(params),
        )
        return self._sim_to_fold_record(
            arm=arm,
            fold_id=fold_id,
            train_window=None,
            params=params,
            sim_result=result,
            artifacts={
                "agent_used": agent_rl,
                "obs_rms_file": obs_used,
                "regime_k": 0,
                "adaptation_cost": 0,
                "adapt_core_hours": 0.0,
                "adapt_wall_time_s": 0.0,
            },
        )

    def _run_r3(
        self,
        *,
        arm: str,
        fold_id: str,
        data_files: Dict[str, str],
        model_paths: Dict[str, str],
        balance_start: float,
        experiment_version: str,
        lock_params: Dict[str, Any],
        agent_rl: Optional[str],
        plots: bool,
        logs: bool,
        sim_results: bool,
        quiet: bool,
        portfolio_state=None,
        gzip_basename=None,
        sim_dir=None,
        **_kwargs,
    ) -> Dict[str, Any]:
        """R3: zamrożony agent rlul + zamrożony model UL (T1–T6), bez refitu/douczenia."""
        agent = agent_rl or _kwargs.get("agent_rlul")
        if not agent:
            raise ValueError("R3 requires agent_rlul= / agent_rl= (pretrained rlul agent)")
        from rl_module.services.thesis.regime_labels import build_onehot_for_window
        from rl_module.services.rl.obs_rms_utils import resolve_obs_rms_beside_agent

        params = dict(lock_params)
        params.setdefault("balance_start", balance_start)
        params.setdefault("rlagent_deterministic", True)
        onehot, k, ul_meta = build_onehot_for_window(
            self.base_dir, data_files, fold_id, refit=False, seed=0,
        )
        obs_used = resolve_obs_rms_beside_agent(
            agent,
            agents_dir=os.path.join(self.base_dir, "shared", "agents", "ppo"),
        )
        sim = SimulationService(self.base_dir)
        result = sim.run_simulation(
            model_paths=model_paths,
            data_files=data_files,
            agent_rl=agent,
            sim_mode="rlul",
            experiment_version=experiment_version or "v6",
            window_id=fold_id,
            plots=plots,
            logs=logs,
            results=sim_results,
            quiet=quiet,
            name_tag=f"wf_R3_{fold_id}",
            portfolio_state=portfolio_state,
            gzip_basename=gzip_basename,
            sim_dir=sim_dir,
            obs_rms_file=obs_used,
            regime_k=k,
            regime_onehot_series=onehot,
            **_filter_sim_params(params),
        )
        return self._sim_to_fold_record(
            arm=arm,
            fold_id=fold_id,
            train_window="train",
            params=params,
            sim_result=result,
            artifacts={
                "agent_used": agent,
                "obs_rms_file": obs_used,
                "ul_model_path": ul_meta.get("model_path"),
                "ul_k": k,
                "ul_fit_time_s": 0.0,
                "ul_refit": False,
                "adaptation_cost": 0,
                "adapt_core_hours": 0.0,
                "adapt_wall_time_s": 0.0,
            },
        )

    def _run_m1(
        self,
        *,
        arm: str,
        fold_id: str,
        data_files: Dict[str, str],
        model_paths: Dict[str, str],
        balance_start: float,
        experiment_version: str,
        lock_params: Dict[str, Any],
        optuna_config: Dict[str, Any],
        optuna_n_trials: int,
        optuna_rolling_windows: int,
        plots: bool,
        logs: bool,
        sim_results: bool,
        quiet: bool,
        run_dir: str,
        seed: int,
        **_kwargs,
    ) -> Dict[str, Any]:
        from rl_module.services.simulation_tuner_service import SimulationTunerService

        train_window = rolling_train_spec(
            self.windows, fold_id, n=int(optuna_rolling_windows or 3),
        )
        config = self._default_optuna_config(
            balance_start, train_window, n_symbols=len(data_files or {}),
        )
        config.update(optuna_config or {})
        config["window"] = train_window

        study_name = f"wf_M1_{fold_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tuner = SimulationTunerService(self.base_dir)
        apply_tuner_settings(tuner)

        t_opt = time.time()
        study = tuner.run_tuning(
            eval_data_files=data_files,
            model_files=model_paths,
            config=config,
            n_trials=int(optuna_n_trials),
            live_logging=not quiet,
            experiment_version=experiment_version or "v6",
            study_name=study_name,
            seed=int(seed),
        )
        optuna_time_s = time.time() - t_opt

        best = dict(study.best_params)
        for key, spec in config.items():
            if key == "window" or key in best:
                continue
            if not isinstance(spec, (list, tuple)):
                best[key] = spec
            elif len(spec) >= 2 and spec[0] == "C" and len(spec) == 2:
                best[key] = spec[1]

        eval_params = dict(lock_params)
        eval_params.setdefault("balance_start", balance_start)
        eval_params.update(best)

        sim = SimulationService(self.base_dir)
        result = sim.run_simulation(
            model_paths=model_paths,
            data_files=data_files,
            agent_rl=None,
            sim_mode="mp",
            experiment_version=experiment_version or "v6",
            window_id=fold_id,
            plots=plots,
            logs=logs,
            results=sim_results,
            quiet=quiet,
            name_tag=f"wf_M1_{fold_id}",
            portfolio_state=_kwargs.get("portfolio_state"),
            gzip_basename=_kwargs.get("gzip_basename"),
            sim_dir=_kwargs.get("sim_dir"),
            **_filter_sim_params(eval_params),
        )

        record = self._sim_to_fold_record(
            arm=arm,
            fold_id=fold_id,
            train_window=train_window,
            params=eval_params,
            sim_result=result,
            artifacts={
                "optuna_study": study_name,
                "optuna_best_params_file": getattr(tuner, "best_params_file", None),
                "optuna_best_metric": study.best_value,
                "optuna_best_trial": study.best_trial.number,
                "optuna_trials": int(optuna_n_trials),
                "optuna_space": "flat_6f",
                "optuna_time_s": round(optuna_time_s, 2),
                "optuna_cores": float(getattr(tuner, "n_jobs", None) or 1),
                "train_time_s": 0.0,
                "ul_fit_time_s": 0.0,
            },
        )
        from rl_module.services.thesis.costs import attach_adapt_costs
        attach_adapt_costs(record["artifacts"], arm="M1")
        record["extras"] = {
            "optuna_rolling_windows": optuna_rolling_windows,
            "optuna_n_trials": optuna_n_trials,
            "seed": seed,
        }
        arm_folds = os.path.join(run_dir, "arms", arm)
        os.makedirs(arm_folds, exist_ok=True)
        with open(os.path.join(arm_folds, f"fold_{fold_id}_best_params.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "train_window": train_window,
                    "eval_window": fold_id,
                    "best_params": best,
                    "best_metric": study.best_value,
                    "study_name": study_name,
                },
                f,
                indent=2,
            )
        return record

    def _run_m2(
        self,
        *,
        arm: str,
        fold_id: str,
        fold_index: int,
        folds: Sequence[str],
        data_files: Dict[str, str],
        model_paths: Dict[str, str],
        balance_start: float,
        experiment_version: str,
        lock_params: Dict[str, Any],
        agent_rl: Optional[str],
        continue_timesteps: int,
        learning_rate: float,
        seed: int,
        plots: bool,
        logs: bool,
        sim_results: bool,
        quiet: bool,
        run_dir: str,
        num_envs: int,
        **_kwargs,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Eval on fold with current agent; if not last fold, continue-learn on this
        fold before returning updated agent for the next eval.
        """
        if not agent_rl:
            raise ValueError("M2 requires agent_rl= (pretrained agent to continue)")

        params = dict(lock_params)
        params.setdefault("balance_start", balance_start)
        params.setdefault("rlagent_deterministic", True)

        sim = SimulationService(self.base_dir)
        result = sim.run_simulation(
            model_paths=model_paths,
            data_files=data_files,
            agent_rl=agent_rl,
            sim_mode="rl",
            experiment_version=experiment_version or "v6",
            window_id=fold_id,
            plots=plots,
            logs=logs,
            results=sim_results,
            quiet=quiet,
            name_tag=f"wf_M2_{fold_id}",
            portfolio_state=_kwargs.get("portfolio_state"),
            gzip_basename=_kwargs.get("gzip_basename"),
            sim_dir=_kwargs.get("sim_dir"),
            **_filter_sim_params(params),
        )
        from rl_module.services.rl.obs_rms_utils import resolve_obs_rms_beside_agent

        obs_used = resolve_obs_rms_beside_agent(
            agent_rl,
            agents_dir=os.path.join(self.base_dir, "shared", "agents", "ppo"),
        )
        record = self._sim_to_fold_record(
            arm=arm,
            fold_id=fold_id,
            train_window=None,
            params=params,
            sim_result=result,
            artifacts={
                "agent_used": agent_rl,
                "obs_rms_file": obs_used,
                "optuna_time_s": 0.0,
                "ul_fit_time_s": 0.0,
            },
        )

        next_agent = agent_rl
        is_last = fold_index >= len(folds) - 1
        if not is_last and record.get("status") == "ok":
            # Continue on just-evaluated window before next fold
            t_train = time.time()
            next_agent = self._continue_ppo(
                continue_from=agent_rl,
                window_id=fold_id,
                data_files=data_files,
                model_paths=model_paths,
                experiment_version=experiment_version or "v6",
                continue_timesteps=continue_timesteps,
                learning_rate=learning_rate,
                lock_params=params,
                num_envs=num_envs,
                seed=seed,
                run_dir=run_dir,
                fold_id=fold_id,
                sim_mode="rl",
                arm_name="M2",
            )
            train_s = time.time() - t_train
            from rl_module.services.thesis.costs import attach_adapt_costs
            art = record.setdefault("artifacts", {})
            art["agent_after_continue"] = next_agent
            art["train_time_s"] = round(train_s, 2)
            art["train_cores"] = int(num_envs)
            art["env_steps"] = int(continue_timesteps)
            art["obs_rms_after_continue"] = resolve_obs_rms_beside_agent(
                next_agent,
                agents_dir=os.path.join(self.base_dir, "shared", "agents", "ppo"),
            )
            attach_adapt_costs(art, arm="M2")
            record.setdefault("extras", {})["continue_on"] = fold_id
            record["extras"]["continue_timesteps"] = continue_timesteps
        else:
            from rl_module.services.thesis.costs import attach_adapt_costs
            attach_adapt_costs(record.setdefault("artifacts", {}), arm="M2")

        return record, next_agent

    def _run_m3(
        self,
        *,
        arm: str,
        fold_id: str,
        fold_index: int,
        folds: Sequence[str],
        data_files: Dict[str, str],
        model_paths: Dict[str, str],
        balance_start: float,
        experiment_version: str,
        lock_params: Dict[str, Any],
        agent_rl: Optional[str],
        continue_timesteps: int,
        learning_rate: float,
        seed: int,
        plots: bool,
        logs: bool,
        sim_results: bool,
        quiet: bool,
        run_dir: str,
        num_envs: int,
        optuna_rolling_windows: int = 3,
        **_kwargs,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """M3 = continue-learn + UL refit rolling (K z regime_lock.json)."""
        agent = agent_rl or _kwargs.get("agent_rlul")
        if not agent:
            raise ValueError("M3 requires agent_rlul= / agent_rl= (pretrained rlul agent)")

        from rl_module.services.rl.obs_rms_utils import resolve_obs_rms_beside_agent
        from rl_module.services.thesis.knobs import REGIME_FIT_ROLLING_WINDOWS
        from rl_module.services.thesis.regime_labels import build_onehot_for_window
        from rl_module.services.thesis.costs import attach_adapt_costs

        params = dict(lock_params)
        params.setdefault("balance_start", balance_start)
        params.setdefault("rlagent_deterministic", True)

        fit_spec = rolling_train_spec(
            self.windows, fold_id, n=int(optuna_rolling_windows or REGIME_FIT_ROLLING_WINDOWS),
        )
        onehot, k, ul_meta = build_onehot_for_window(
            self.base_dir,
            data_files,
            fold_id,
            refit=True,
            fit_window_spec=fit_spec,
            seed=int(seed),
        )
        ul_fit_s = float(ul_meta.get("ul_fit_time_s") or 0.0)

        arm_dir = os.path.join(run_dir, "arms", "M3")
        os.makedirs(arm_dir, exist_ok=True)
        # zapisz refit model per fold (opcjonalnie z serwisu — reload z fit)
        from rl_module.services.regime_service import RegimeConfig, RegimeService, require_regime_k
        require_regime_k(self.base_dir)  # wymuś obecność locka
        model_path = os.path.join(arm_dir, f"fold_{fold_id}_ul.joblib")
        # przebuduj i zapisz model refit (build_onehot już fitował wewnętrznie —
        # powtórz fit do artefaktu)
        from rl_module.services.regime_service import load_regime_lock
        lock = load_regime_lock(self.base_dir)
        svc = RegimeService(RegimeConfig(k_locked=int(k), seed=int(lock.get("seed") or 0)))
        closes_fit = {}
        for symbol, filename in data_files.items():
            df = self._load_sliced(filename, fit_spec)
            closes_fit[symbol] = df["close"].astype(float).to_numpy()
        svc.fit_from_closes(closes_fit, seed=int(seed), k=int(k))
        svc.save(model_path)

        obs_used = resolve_obs_rms_beside_agent(
            agent,
            agents_dir=os.path.join(self.base_dir, "shared", "agents", "ppo"),
        )
        sim = SimulationService(self.base_dir)
        result = sim.run_simulation(
            model_paths=model_paths,
            data_files=data_files,
            agent_rl=agent,
            sim_mode="rlul",
            experiment_version=experiment_version or "v6",
            window_id=fold_id,
            plots=plots,
            logs=logs,
            results=sim_results,
            quiet=quiet,
            name_tag=f"wf_M3_{fold_id}",
            portfolio_state=_kwargs.get("portfolio_state"),
            gzip_basename=_kwargs.get("gzip_basename"),
            sim_dir=_kwargs.get("sim_dir"),
            obs_rms_file=obs_used,
            regime_k=k,
            regime_onehot_series=onehot,
            **_filter_sim_params(params),
        )
        record = self._sim_to_fold_record(
            arm=arm,
            fold_id=fold_id,
            train_window=fit_spec,
            params=params,
            sim_result=result,
            artifacts={
                "agent_used": agent,
                "obs_rms_file": obs_used,
                "ul_model_path": model_path,
                "ul_fit_time_s": round(ul_fit_s, 2),
                "ul_cores": 1,
                "ul_k": k,
                "ul_fit_windows": fit_spec,
                "ul_profile": svc.profile_,
                "ul_refit": True,
                "optuna_time_s": 0.0,
            },
        )

        next_agent = agent
        is_last = fold_index >= len(folds) - 1
        if not is_last and record.get("status") == "ok":
            t_train = time.time()
            next_agent = self._continue_ppo(
                continue_from=agent,
                window_id=fold_id,
                data_files=data_files,
                model_paths=model_paths,
                experiment_version=experiment_version or "v6",
                continue_timesteps=continue_timesteps,
                learning_rate=learning_rate,
                lock_params=params,
                num_envs=num_envs,
                seed=seed,
                run_dir=run_dir,
                fold_id=fold_id,
                sim_mode="rlul",
                regime_onehot_series=onehot,
                arm_name="M3",
            )
            train_s = time.time() - t_train
            art = record.setdefault("artifacts", {})
            art["agent_after_continue"] = next_agent
            art["train_time_s"] = round(train_s, 2)
            art["train_cores"] = int(num_envs)
            art["obs_rms_after_continue"] = resolve_obs_rms_beside_agent(
                next_agent,
                agents_dir=os.path.join(self.base_dir, "shared", "agents", "ppo"),
            )
            art["env_steps"] = int(continue_timesteps)
            attach_adapt_costs(art, arm="M3")
            record.setdefault("extras", {})["continue_on"] = fold_id
            record["extras"]["continue_timesteps"] = continue_timesteps
        else:
            attach_adapt_costs(record.setdefault("artifacts", {}), arm="M3")

        return record, next_agent

    def _continue_ppo(
        self,
        *,
        continue_from: str,
        window_id: str,
        data_files: Dict[str, str],
        model_paths: Dict[str, str],
        experiment_version: str,
        continue_timesteps: int,
        learning_rate: float,
        lock_params: Dict[str, Any],
        num_envs: int,
        seed: int,
        run_dir: str,
        fold_id: str,
        sim_mode: str = "rl",
        regime_onehot_series=None,
        arm_name: Optional[str] = None,
    ) -> str:
        from rl_module.services.train_agent_ppo_service import TrainAgentPPOService

        arm_name = arm_name or ("M3" if sim_mode == "rlul" else "M2")
        tag = f"WF-{arm_name}"
        thesis_console.print_card(
            f"Douczanie PPO ({arm_name})",
            [
                ("fold", fold_id),
                ("window_id", window_id),
                ("continue_from", continue_from),
                ("timesteps", f"{int(continue_timesteps):,}"),
                ("seed", seed),
                ("learning_rate", learning_rate),
                ("sim_mode", sim_mode),
            ],
            tag=tag,
        )
        series = regime_onehot_series
        if sim_mode == "rlul" and series is None:
            from rl_module.services.thesis.regime_labels import build_onehot_for_window
            from rl_module.services.thesis.knobs import REGIME_FIT_ROLLING_WINDOWS
            fit_spec = rolling_train_spec(
                self.windows, window_id, n=REGIME_FIT_ROLLING_WINDOWS,
            )
            series, _k, _m = build_onehot_for_window(
                self.base_dir,
                data_files,
                window_id,
                refit=True,
                fit_window_spec=fit_spec,
                seed=int(seed),
            )
        trainer = TrainAgentPPOService(self.base_dir)
        # force reload for this window
        trainer.models = {}
        result = trainer.train_agent_ppo(
            model_files=model_paths,
            data_files=data_files,
            total_timesteps=int(continue_timesteps),
            learning_rate=float(learning_rate),
            # douczanie: te same wymiary co pretrain (n_steps/batch/gamma/gae/clip/epochs
            # wczytuje PPO.load z agenta); entropia stała = wartość końcowa pretrainu,
            # epizod 10080 jak w treningu (domyślne 1024 zmieniałoby skalę nagrody).
            ent_coef=float(PPO_HPARAMS["ent_coef_end"]),
            ent_coef_end=float(PPO_HPARAMS["ent_coef_end"]),
            episode_timesteps=int(PPO_HPARAMS["episode_timesteps"]),
            live_logging=False,
            experiment_version=experiment_version,
            sim_mode=sim_mode,
            regime_onehot_series=series,
            regime_zeros_only=(sim_mode != "rlul"),
            window_id=window_id,
            eval_window_id=None,
            stop_loss=float(lock_params.get("stop_loss", _KC["stop_loss"])),
            stop_loss_cooldown=int(lock_params.get("stop_loss_cooldown", _KC["stop_loss_cooldown"])),
            wallet_percentage_limit=float(
                lock_params.get("wallet_percentage_limit", _KC["wallet_percentage_limit"])
            ),
            max_wallet_percentage_crypto=float(
                lock_params.get("max_wallet_percentage_crypto", _KC["max_wallet_percentage_crypto"])
            ),
            mincost=float(lock_params.get("mincost", _KC["mincost"])),
            clamp_wallet_cap_to_1n=bool(
                lock_params.get("clamp_wallet_cap_to_1n", CLAMP_WALLET_CAP_TO_1N)
            ),
            mp_alloc_bias=lock_params.get("mp_alloc_bias"),
            mp_alloc_scale=lock_params.get("mp_alloc_scale"),
            mp_confidence=lock_params.get("mp_confidence"),
            mp_urgency=lock_params.get("mp_urgency"),
            mp_signal_deadzone=lock_params.get("mp_signal_deadzone"),
            mp_rebalance_deadzone=lock_params.get("mp_rebalance_deadzone"),
            v5rl_knob_hold_bars=int(
                lock_params.get("v5rl_knob_hold_bars", V5RL_KNOB_HOLD_BARS)
            ),
            v5rl_trade_lock_bars=int(
                lock_params.get("v5rl_trade_lock_bars", V5RL_TRADE_LOCK_BARS)
            ),
            continue_from=continue_from,
            reset_num_timesteps=False,
            num_envs=int(num_envs),
            early_stop_patience=0,
            early_stop_min_dumps=0,
            seed=int(seed),
        )
        agent_file = result.get("agent_file") or result.get("agent_path")
        if not agent_file:
            raise RuntimeError(
                f"{arm_name} continue-learn failed on {window_id}: {result}"
            )
        # Prefer basename for SimulationService lookup
        if os.path.isabs(str(agent_file)):
            agent_file = os.path.basename(agent_file)
        meta_path = os.path.join(
            run_dir, "arms", arm_name, f"fold_{fold_id}_continue.json",
        )
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "arm": arm_name,
                    "continue_on": window_id,
                    "continue_from": continue_from,
                    "agent_file": agent_file,
                    "sim_mode": sim_mode,
                    "result": result.get("result"),
                    "seed": seed,
                    "regime_labels": series is not None and sim_mode == "rlul",
                },
                f,
                indent=2,
            )
        return agent_file

    # --------------------------------------------------------------- helpers

    def _normalize_arms(self, arms: Sequence[str], skip_placeholders: bool) -> List[str]:
        out: List[str] = []
        for a in arms:
            key = normalize_arm(a)
            if skip_placeholders and key in PLACEHOLDER_ARMS:
                thesis_console.print_kv("WF", f"pomijam placeholder: {key}", kind="warn")
                continue
            if key not in out:
                out.append(key)
        if not out:
            raise ValueError("Brak uruchamialnych ramion po filtrze placeholderów")
        return out

    def _resolve_lock_params(
        self,
        *,
        balance_start: float,
        best_params_path: Optional[str],
        mp_params: Optional[Dict[str, Any]],
        v5mp_params: Optional[Dict[str, Any]],
        knob_kwargs: Dict[str, Any],
        expected_seed: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        from rl_module.services.thesis_artifacts import (
            extract_best_params,
            extract_seed,
            read_json,
        )

        params = default_mp_params(balance_start)
        meta: Dict[str, Any] = {"seed": None, "path": best_params_path}
        if best_params_path:
            loaded = read_json(best_params_path, self.base_dir)
            art_seed = extract_seed(loaded)
            meta["seed"] = art_seed
            if (
                expected_seed is not None
                and art_seed is not None
                and int(art_seed) != int(expected_seed)
            ):
                print(
                    f"[WF] WARN best_params seed={art_seed} != run seed={expected_seed} "
                    f"({best_params_path})"
                )
            knobs = extract_best_params(loaded)
            params.update(knobs)
        if v5mp_params:
            params.update(v5mp_params)
        if mp_params:
            params.update(mp_params)
        for k, v in (knob_kwargs or {}).items():
            if k in _SIM_KWARGS and v is not None:
                params[k] = v
        params["balance_start"] = balance_start
        # hold/lock bars — te same co pretrain (nie z LOCK Optuny MP)
        params.setdefault("v5rl_knob_hold_bars", V5RL_KNOB_HOLD_BARS)
        params.setdefault("v5rl_trade_lock_bars", V5RL_TRADE_LOCK_BARS)
        params.setdefault("clamp_wallet_cap_to_1n", CLAMP_WALLET_CAP_TO_1N)
        return params, meta

    def _load_sliced(self, filename: str, window_spec: str) -> pd.DataFrame:
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df.dropna(inplace=True)
        return slice_windows(df, self.windows, window_spec)

    @staticmethod
    def _default_mp_params(balance_start: float) -> Dict[str, Any]:
        return default_mp_params(balance_start)

    @staticmethod
    def _default_optuna_config(
        balance_start: float, train_window: str, n_symbols: int = 2,
    ) -> Dict[str, Any]:
        """Flat 6F SHARED — ta sama przestrzeń co LOCK / plgrid_simulation_tuner."""
        return build_optuna_config(balance_start, train_window, n_symbols=n_symbols)

    def _export_sequence_b(
        self,
        *,
        arm_id: str,
        run_label: str,
        seed: int,
        folds: Sequence[str],
        fold_records: List[Dict[str, Any]],
        sim_payloads_by_fold: Dict[str, Dict[str, Any]],
        run_dir: str,
        portfolio_mode: str,
        plots: bool,
    ) -> None:
        """Plik ciągły od B1 (A1/A2 tylko jako osobne foldy)."""
        b_ids = b_folds_only(folds)
        if not b_ids:
            return
        payloads = []
        used = []
        for fid in b_ids:
            p = sim_payloads_by_fold.get(fid)
            if not p:
                # spróbuj wczytać gzip z windows/
                art = next(
                    (
                        (r.get("artifacts") or {})
                        for r in fold_records
                        if r.get("fold_id") == fid
                    ),
                    {},
                )
                gz = art.get("gzip_path")
                if gz and os.path.isfile(gz):
                    with gzip.open(gz, "rt", encoding="utf-8") as f:
                        p = json.load(f)
            if not p:
                continue
            payloads.append(p)
            used.append(fid)
        if not payloads:
            print("[WF] sequence_B: brak payloadów B — skip")
            return

        stitched = stitch_sequence_payload(
            fold_payloads=payloads,
            fold_ids=used,
            arm=arm_id,
            seed=seed,
            label=run_label,
            portfolio_mode=portfolio_mode,
        )
        # metryki na sklejonej krzywej
        mode = "agent" if "agent" in stitched else "model"
        pv = list((stitched.get(mode) or {}).get("portfolio_values") or [])
        if pv:
            bal0 = float(pv[0]) if pv[0] else 1000.0
            metrics = self.rating.rate(
                pv,
                transactions=0,
                closed_trades=[],
                holdings_history=[[1.0] for _ in pv],
                balance_start=bal0,
            )
            stitched[mode]["metrics"] = metrics

        seq_dir = os.path.join(run_dir, "sequence_B")
        os.makedirs(seq_dir, exist_ok=True)
        base = sequence_b_gzip_basename(
            arm=arm_id,
            label=run_label,
            seed=seed,
            start_b=used[0],
            end_b=used[-1],
        )
        out_gz = os.path.join(
            self.base_dir, "shared", "simulation", "outputs", f"{base}.json.gz"
        )
        write_viewer_gzip(out_gz, stitched)
        copy_gzip_to_run(out_gz, seq_dir)
        with open(os.path.join(seq_dir, "sequence_B_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "folds": used,
                    "gzip_name": f"{base}.json.gz",
                    "gzip_path": out_gz,
                    "portfolio_mode": portfolio_mode,
                    "metrics": (stitched.get(mode) or {}).get("metrics"),
                },
                f,
                indent=2,
            )
        if plots:
            try:
                from test_module.services.visualization_service import VisualizationService
                VisualizationService().visualize(
                    simulation_files=f"{base}.json.gz",
                    sim_path=seq_dir,
                )
            except Exception as e:
                print(f"[WF] sequence_B chart skip: {e}")
        # skrót w root run_dir (łatwe porównanie ramion)
        with open(os.path.join(run_dir, "sequence_B.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "folds": used,
                    "gzip_name": f"{base}.json.gz",
                    "gzip_path": out_gz,
                    "portfolio_mode": portfolio_mode,
                    "metrics": (stitched.get(mode) or {}).get("metrics"),
                },
                f,
                indent=2,
            )
        print(f"[WF] sequence_B → {out_gz}")

    def _sim_to_fold_record(
        self,
        *,
        arm: str,
        fold_id: str,
        train_window: Optional[str],
        params: Dict[str, Any],
        sim_result: Dict[str, Any],
        artifacts: Dict[str, Any],
    ) -> Dict[str, Any]:
        artifacts = dict(artifacts or {})
        if sim_result.get("portfolio_state"):
            artifacts["portfolio_state"] = sim_result["portfolio_state"]
        if sim_result.get("gzip_path"):
            artifacts["gzip_path"] = sim_result["gzip_path"]
            artifacts["gzip_name"] = sim_result.get("gzip_name")
        if sim_result.get("sim_path"):
            artifacts["sim_path"] = sim_result["sim_path"]
        if sim_result.get("sim_output"):
            artifacts["sim_output"] = sim_result["sim_output"]

        if sim_result.get("result") != "success":
            return self._fold_record(
                arm=arm,
                fold_id=fold_id,
                train_window=train_window,
                eval_window=fold_id,
                params=params,
                metrics={},
                hodl_return=None,
                excess_return=None,
                n_ticks=sim_result.get("n_ticks"),
                status="failed",
                artifacts=artifacts,
                extras={"error": sim_result.get("message")},
            )

        metrics = dict(sim_result.get("metrics") or {})
        hodl_return = sim_result.get("hodl_return")
        total_return = metrics.get("total_return")
        if total_return is None:
            final_v = sim_result.get("final_portfolio_value")
            bal = float(params.get("balance_start") or 1000.0)
            if final_v is not None and bal > 0:
                total_return = (float(final_v) / bal - 1.0) * 100.0
                metrics["total_return"] = total_return

        excess = None
        if total_return is not None and hodl_return is not None:
            excess = float(total_return) - float(hodl_return)

        return self._fold_record(
            arm=arm,
            fold_id=fold_id,
            train_window=train_window,
            eval_window=fold_id,
            params=params,
            metrics=metrics,
            hodl_return=hodl_return,
            excess_return=excess,
            n_ticks=sim_result.get("n_ticks"),
            status="ok",
            artifacts=artifacts,
        )

    def _placeholder_record(
        self,
        *,
        arm: str,
        fold_id: str,
        train_window: Optional[str],
        reason: str,
    ) -> Dict[str, Any]:
        meta = ARM_META.get(arm, {})
        return self._fold_record(
            arm=arm,
            fold_id=fold_id,
            train_window=train_window,
            eval_window=fold_id,
            params={},
            metrics={},
            hodl_return=None,
            excess_return=None,
            n_ticks=None,
            status="placeholder",
            artifacts={},
            extras={"reason": reason, "plan_id": meta.get("plan_id")},
        )

    def _fold_record(
        self,
        *,
        arm: str,
        fold_id: str,
        train_window: Optional[str],
        eval_window: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        hodl_return: Optional[float],
        excess_return: Optional[float],
        n_ticks: Optional[int],
        status: str,
        artifacts: Dict[str, Any],
        extras: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta = ARM_META.get(arm, {})
        return {
            "schema_version": 2,
            "arm": arm,
            "plan_id": meta.get("plan_id"),
            "arm_label": meta.get("label"),
            "fold_id": fold_id,
            "train_window": train_window,
            "eval_window": eval_window,
            "experiment_version": meta.get("experiment_version"),
            "sim_mode": meta.get("sim_mode"),
            "params": params,
            "metrics": metrics,
            "hodl_return": hodl_return,
            "excess_return": excess_return,
            "n_ticks": n_ticks,
            "artifacts": artifacts,
            "status": status,
            "extras": extras or {},
            "timestamp": datetime.now().isoformat(),
        }
