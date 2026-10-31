import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from datetime import datetime, timedelta
from typing import List

from test_module.services.core_logic.mp_signal import ModelSignal
from test_module.services.core_logic.decision import DecisionEngine
from test_module.services.core_logic.structures import CoreConfig


class ObsImportanceService:
    """Analiza ważności wymiarów przestrzeni obserwacji agenta."""
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir   = base_dir
        self.DATA_DIR   = os.path.join(base_dir, "shared", "files", "processed_simulation_data")
        self.OUTPUT_DIR = os.path.join(base_dir, "shared", "simulation", "obs_importance")
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        self.num_ticks = 0
        self.models    = {}
        self.symbols   = []
        self.engine    = None


    def run_analysis(
        self,
        rabbit_msg_id=None,
        model_paths=None,
        data_files=None,
        agent_rl=None,
        # CoreConfig params
        balance_start=1000.0,
        add_fee=True,
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
        experiment_version="v3",
        # Analiza params
        t_start=0,
        t_end=5000,
        step=50,
        method="both",       # "perturbation" | "gradient" | "both"
        save_plot=True,
        save_json=True,
    ):
        try:
            print("[ObsImportance] Start")
            start_time = time.time()
            data_files  = data_files  or {}
            model_paths = model_paths or {}

            # --- Ładowanie danych i modeli ---
            symbols = []
            for symbol, filename in data_files.items():
                filepath = os.path.join(self.DATA_DIR, filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Brak pliku: {filepath}")
                df = pd.read_csv(filepath)
                df.dropna(inplace=True)
                len_df = len(df)
                if self.num_ticks == 0: self.num_ticks = len_df
                if len_df < self.num_ticks: self.num_ticks = len_df

                self.models[symbol] = ModelSignal(
                    model_name=model_paths[symbol],
                    data=df.copy(),
                    base_dir=self.base_dir
                )
                symbols.append(symbol)

                len_pred = len(self.models[symbol].predictions)
                if len_pred < self.num_ticks: self.num_ticks = len_pred

            for symbol in symbols:
                self.models[symbol].process_models_data(self.num_ticks)
            self.symbols = symbols

            # --- CoreConfig ---
            core_config = CoreConfig(
                sim_mode="rl",
                balance_start=balance_start,
                add_fee=add_fee,
                symbols=self.symbols,
                rlagent=agent_rl,
                rlagent_deterministic=rlagent_deterministic,
                slippage=slippage,
                wallet_percentage_limit=wallet_percentage_limit,
                max_wallet_percentage_crypto=max_wallet_percentage_crypto,
                stop_loss=stop_loss,
                threshold_buy=threshold_buy,
                threshold_sell=threshold_sell,
                mincost=mincost,
                tx_fraction=tx_fraction,
                max_fraction_when_confident=max_fraction_when_confident,
                experiment_version=experiment_version,
            )

            # --- DecisionEngine z załadowanym agentem ---
            self.engine = DecisionEngine(
                agent_name=agent_rl,
                models=self.models,
                base_dir=self.base_dir,
                core_mode="sim_rl",
                core_config=core_config,
            )

            if self.engine.agent is None:
                raise ValueError("Agent nie został załadowany — sprawdź agent_rl path.")

            t_end_clamped = min(t_end, self.num_ticks - 1)
            results = self._analyze(t_start, t_end_clamped, step, method)

            output_name = self._build_filename(experiment_version)
            if save_json:
                self._save_json(results, output_name)
            if save_plot:
                self._save_plot(results, output_name)

            self._print_report(results)

            print(f"[ObsImportance] Zakończono w {timedelta(seconds=time.time() - start_time)}")
            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "success",
                "output_dir": self.OUTPUT_DIR,
                "output_name": output_name,
                "feature_names": results["feature_names"],
                "perturbation_mean": results.get("perturbation_mean", []),
                "gradient_mean": results.get("gradient_mean", []),
            }

        except Exception as e:
            import traceback
            print("[ObsImportance][ERROR]", e)
            traceback.print_exc()
            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "failed",
                "message": str(e),
            }


    def _analyze(self, t_start: int, t_end: int, step: int, method: str) -> dict:
        feature_names = self._build_feature_names()
        perturb_all   = []
        gradient_all  = []
        t_range       = range(t_start, t_end, step)

        print(f"[ObsImportance] Próbkowanie {len(t_range)} kroków ({method}), step={step}...")

        for t in t_range:
            self.engine.t_data(t)
            obs = self.engine.build_obs(t)

            if method in ("perturbation", "both"):
                perturb_all.append(self._perturbation(obs))
            if method in ("gradient", "both"):
                gradient_all.append(self._gradient(obs))

        results = {
            "feature_names": feature_names,
            "t_start": t_start,
            "t_end": t_end,
            "step": step,
            "method": method,
        }

        if perturb_all:
            arr = np.stack(perturb_all)
            results["perturbation_mean"] = arr.mean(axis=0).tolist()
            results["perturbation_std"]  = arr.std(axis=0).tolist()

        if gradient_all:
            arr = np.stack(gradient_all)
            results["gradient_mean"] = arr.mean(axis=0).tolist()
            results["gradient_std"]  = arr.std(axis=0).tolist()

        return results

    def _perturbation(self, obs: np.ndarray) -> np.ndarray:
        base_action, _ = self.engine.agent.predict(obs, deterministic=True)
        base_action    = np.asarray(base_action, dtype=np.float32).flatten()
        importances    = np.zeros(len(obs), dtype=np.float32)

        for i in range(len(obs)):
            perturbed    = obs.copy()
            perturbed[i] = 0.0
            action, _    = self.engine.agent.predict(perturbed, deterministic=True)
            action       = np.asarray(action, dtype=np.float32).flatten()
            importances[i] = np.abs(base_action - action).mean()

        return importances

    def _gradient(self, obs: np.ndarray) -> np.ndarray:
        policy     = self.engine.agent.policy
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).requires_grad_(True)
        try:
            features = policy.mlp_extractor(policy.extract_features(obs_tensor))
            latent_pi = features[0] if isinstance(features, tuple) else features
            action    = policy.action_net(latent_pi)
            action.sum().backward()
            return obs_tensor.grad.abs().squeeze().detach().numpy().astype(np.float32)
        except Exception as e:
            print(f"[ObsImportance] gradient fallback → perturbation ({e})")
            return self._perturbation(obs)


    def _print_report(self, results: dict):
        names = results["feature_names"]

        for key, label in [
            ("perturbation_mean", "PERTURBATION"),
            ("gradient_mean",     "GRADIENT SALIENCY"),
        ]:
            if key not in results:
                continue
            values = np.array(results[key])
            std    = np.array(results.get(key.replace("mean", "std"), np.zeros_like(values)))
            order  = np.argsort(values)[::-1]
            total  = values.sum() if values.sum() > 0 else 1.0

            print(f"\n{'─'*52}")
            print(f"  {label}")
            print(f"{'─'*52}")
            print(f"  {'Feature':<30} {'Imp':>8}  {'%':>6}  {'Std':>8}")
            print(f"  {'─'*28} {'─'*8}  {'─'*6}  {'─'*8}")
            for i in order:
                print(f"  {names[i]:<30} {values[i]:>8.5f}  "
                      f"{100*values[i]/total:>5.1f}%  {std[i]:>8.5f}")

        key = "perturbation_mean"
        if key in results:
            values = np.array(results[key])
            print(f"\n{'─'*52}")
            print("  WAŻNOŚĆ PER SYMBOL")
            print(f"{'─'*52}")
            print(f"  {'cash_ratio':<20}  {values[0]:.5f}")
            for i, s in enumerate(self.symbols):
                b = values[1 + i*5 : 1 + (i+1)*5]
                print(f"  {s:<20}  sum={b.sum():.5f}  "
                      f"[mu={b[0]:.4f} sig={b[1]:.4f} "
                      f"mom={b[2]:.4f} pos={b[3]:.4f} pnl={b[4]:.4f}]")


    def _save_json(self, results: dict, name: str):
        path = os.path.join(self.OUTPUT_DIR, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[ObsImportance] JSON zapisany: {path}")

    def _save_plot(self, results: dict, name: str):
        names  = results["feature_names"]
        has_p  = "perturbation_mean" in results
        has_g  = "gradient_mean"     in results
        n_cols = int(has_p) + int(has_g)
        if n_cols == 0:
            return

        colors = ["#00d4ff", "#ff6b6b", "#ffd700", "#7bed9f",
                  "#a29bfe", "#fd79a8", "#fdcb6e"]

        fig = plt.figure(
            figsize=(7 * n_cols, max(5, len(names) * 0.4 + 2)),
            facecolor="#1a1a2e"
        )
        gs = gridspec.GridSpec(1, n_cols, figure=fig)

        def _bar(ax, values, std, title):
            values = np.array(values)
            std    = np.array(std)
            order  = np.argsort(values)
            y_pos  = np.arange(len(order))
            clrs   = [colors[i % len(colors)] for i in range(len(order))]
            ax.barh(y_pos, values[order], xerr=std[order],
                    color=clrs, alpha=0.85, height=0.6,
                    error_kw=dict(ecolor="#ffffff44", capsize=3))
            ax.set_yticks(y_pos)
            ax.set_yticklabels([names[i] for i in order], fontsize=9, color="white")
            ax.set_xlabel("Importance", color="white", fontsize=10)
            ax.set_title(title, color="white", fontsize=12, pad=10)
            ax.set_facecolor("#0f0f23")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")

        col = 0
        if has_p:
            _bar(fig.add_subplot(gs[0, col]),
                 results["perturbation_mean"],
                 results["perturbation_std"],
                 "Perturbation Importance")
            col += 1
        if has_g:
            _bar(fig.add_subplot(gs[0, col]),
                 results["gradient_mean"],
                 results["gradient_std"],
                 "Gradient Saliency")

        fig.suptitle("Observation Importance Analysis",
                     color="white", fontsize=14, y=1.01)
        plt.tight_layout()

        path = os.path.join(self.OUTPUT_DIR, f"{name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[ObsImportance] Wykres zapisany: {path}")

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _build_feature_names(self) -> List[str]:
        names = ["cash_ratio"]
        for s in self.symbols:
            names += [f"{s}_mu", f"{s}_sigma", f"{s}_momentum",
                      f"{s}_position", f"{s}_pnl"]
        return names

    def _build_filename(self, experiment_version: str) -> str:
        date_str = datetime.now().strftime("%d_%m_%H-%M")
        return f"obs_importance_{experiment_version}_{date_str}"
