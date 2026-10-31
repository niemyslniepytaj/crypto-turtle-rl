import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Sequence, Tuple
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from test_module.services.simulation_service import SimulationService
from test_module.services.core_logic.windows import load_windows, expand_window_ids

class BaseTunerService:
    """Wspólna warstwa Optuna dla treningu i symulacji (staged E1–E3, scoring)."""

    def __init__(self, base_dir: str = None):

        self.base_dir = base_dir
        self.model_files = {}
        self.train_data_files = {}
        self.eval_data_files = {}
        self.log_path = os.path.join(self.base_dir, "shared", "agents", "optuna_studies")

        self.sim = SimulationService(self.base_dir)

        self.agent_type = None
        self.live_logging = False
        self.experiment_version = None
        self.config = None
        self.defaults = None
        self.study_name = None
        self.storage = None
        self.results_file = None
        self.best_params_file = None
        self.optuna_seed = 42

    def _parse_config(self, config: Dict[str, List], defaults: Dict[str, Any] = None):
        """Dynamiczne tworzenie funkcji dla parametrów"""
        self.config = config
        self.defaults = defaults or {}
        known_types = {"C", "F", "FL", "I"}

        # Create getters for parameters in config
        for param_name, spec in config.items():
            # skróty: "train" / ["train"] / ["A1","A2"] == typ C bez prefiksu
            if not isinstance(spec, (list, tuple)):
                param_type, values = "C", [spec]
            elif len(spec) == 0:
                raise ValueError(f"Pusta specyfikacja parametru: {param_name}")
            elif spec[0] not in known_types:
                param_type, values = "C", list(spec)
            else:
                param_type, values = spec[0], list(spec[1:])

            if param_type == "C":
                # Categorical or Fixed
                if len(values) == 1:
                    # Fixed value
                    def getter(trial, val=values[0]):
                        return val
                else:
                    # Categorical
                    def getter(trial, name=param_name, choices=values):
                        return trial.suggest_categorical(name, choices)

            elif param_type == "F":
                # Float (linear scale)
                low, high = values[0], values[1]

                def getter(trial, name=param_name, l=low, h=high):
                    return trial.suggest_float(name, l, h, log=False)

            elif param_type == "FL":
                # Float (log scale)
                low, high = values[0], values[1]

                def getter(trial, name=param_name, l=low, h=high):
                    return trial.suggest_float(name, l, h, log=True)

            elif param_type == "I":
                # Integer
                low, high = values[0], values[1]

                def getter(trial, name=param_name, l=low, h=high):
                    return trial.suggest_int(name, l, h)

            else:
                raise ValueError(f"Nieznany typ parametru: {param_type} dla parametru: {param_name}")

            # Dynamically set getter method
            setattr(self, f"get_{param_name}", getter)

        # Create static getters for defaults
        for param_name, default_value in self.defaults.items():
            def getter(trial, val=default_value):
                return val

            setattr(self, f"get_{param_name}", getter)

    def _get_param(self, trial: optuna.Trial, param_name: str, fallback_default: Any = None):
        """Wyciąganie parametrów ze słownika (mp_* z fallbackiem na legacy ml_*)."""
        getter_name = f"get_{param_name}"
        if hasattr(self, getter_name):
            return getattr(self, getter_name)(trial)
        if param_name.startswith("mp_"):
            legacy = "ml_" + param_name[3:]
            legacy_getter = f"get_{legacy}"
            if hasattr(self, legacy_getter):
                return getattr(self, legacy_getter)(trial)
        if fallback_default is not None:
            return fallback_default
        raise ValueError(f"Nie ma takiego parametru: {param_name}")

    @staticmethod
    def _hodl_return_corr(portfolio_values, hodl_refs, step: int = 60) -> float:
        """Korelacja zwrotów strategii vs HODL (1.0 ≈ ta sama krzywa). step=60 → ~1h z 1m."""
        if not portfolio_values or not hodl_refs:
            return 1.0
        series = list(hodl_refs.values())
        if not series or not series[0]:
            return 1.0
        n = min(len(portfolio_values), min(len(s) for s in series))
        if n < 3:
            return 1.0
        step = max(int(step), 1)
        idx = list(range(0, n, step))
        if idx[-1] != n - 1:
            idx.append(n - 1)
        sp = [float(portfolio_values[i]) for i in idx]
        hp = [sum(float(s[i]) for s in series) / len(series) for i in idx]
        sr, hr = [], []
        for i in range(1, len(sp)):
            if sp[i - 1] > 0 and hp[i - 1] > 0:
                sr.append(sp[i] / sp[i - 1] - 1.0)
                hr.append(hp[i] / hp[i - 1] - 1.0)
        m = len(sr)
        if m < 2:
            return 1.0
        mean_s = sum(sr) / m
        mean_h = sum(hr) / m
        var_s = sum((x - mean_s) ** 2 for x in sr)
        var_h = sum((x - mean_h) ** 2 for x in hr)
        # brak zmienności strategii (płaski cash / 0 tx) ≠ HODL — nie karz corr=1.0
        if var_s <= 1e-18:
            return 0.0
        if var_h <= 1e-18:
            return 0.0
        cov = sum((sr[i] - mean_s) * (hr[i] - mean_h) for i in range(m))
        return max(-1.0, min(1.0, cov / ((var_s * var_h) ** 0.5)))

    def _anti_hodl_penalties(self, tx_count, closed_trades, days, portfolio_values, hodl_refs):
        """
        Feasibility (hard):
          min_tx <= tx, closed >= min_closed, corr < hard,
          tx <= max_tx * overtrade_hard_mult
        Lekki overtrade (max..hard_mult) = soft penalty, nie INV.
        """
        days = max(float(days), 1e-9)
        min_tx_per_day = float(getattr(self, 'min_tx_per_day', 1.5))
        min_tx_floor = int(getattr(self, 'min_tx_floor', 20))
        if getattr(self, 'min_transactions', None) is not None:
            min_tx = int(self.min_transactions)
        else:
            min_tx = max(min_tx_floor, int(round(days * min_tx_per_day)))

        min_closed_per_day = float(getattr(self, 'min_closed_per_day', 0.4))
        min_closed_floor = int(getattr(self, 'min_closed_floor', 10))
        if getattr(self, 'min_closed_trades', None) is not None:
            min_closed = int(self.min_closed_trades)
        else:
            min_closed = max(min_closed_floor, int(round(days * min_closed_per_day)))

        max_tx_per_day = float(getattr(self, 'max_tx_per_day', 4.0))
        max_tx = max(min_tx, int(round(days * max_tx_per_day)))
        overtrade_hard_mult = float(getattr(self, 'overtrade_hard_mult', 1.75))
        max_tx_hard = max(max_tx, int(round(max_tx * overtrade_hard_mult)))

        corr_soft = float(getattr(self, 'hodl_corr_soft', 0.88))
        corr_hard = float(getattr(self, 'hodl_corr_hard', 0.97))
        soft_shape_penalty = float(getattr(self, 'shape_penalty', 40.0))
        invalid_floor = float(getattr(self, 'invalid_floor', -400.0))
        corr_weight = float(getattr(self, 'corr_weight', 80.0))
        overtrade_soft_weight = float(getattr(self, 'overtrade_soft_weight', 25.0))

        hodl_corr = self._hodl_return_corr(portfolio_values, hodl_refs)

        reasons = []
        if tx_count < min_tx:
            reasons.append("undertrade")
        if tx_count > max_tx_hard:
            reasons.append("overtrade")
        if closed_trades < min_closed:
            reasons.append("few_closed")
        if hodl_corr >= corr_hard:
            reasons.append("hodl_corr")

        valid = len(reasons) == 0

        if hodl_corr <= corr_soft:
            shape_penalty = 0.0
        elif hodl_corr >= corr_hard:
            shape_penalty = soft_shape_penalty
        else:
            shape_penalty = (
                (hodl_corr - corr_soft) / max(corr_hard - corr_soft, 1e-9)
            ) * soft_shape_penalty

        # ciągła premia za niższy corr (nawet poniżej hard) — kluczowe przy bull-train
        corr_term = (corr_hard - float(hodl_corr)) * corr_weight

        if tx_count > max_tx and tx_count <= max_tx_hard and max_tx > 0:
            overtrade_penalty = (
                (tx_count - max_tx) / float(max_tx)
            ) * overtrade_soft_weight
        else:
            overtrade_penalty = 0.0

        return {
            'min_tx': min_tx,
            'max_tx': max_tx,
            'max_tx_hard': max_tx_hard,
            'min_closed': min_closed,
            'hodl_corr': hodl_corr,
            'valid': valid,
            'invalid_reasons': reasons,
            'invalid_floor': invalid_floor,
            'shape_penalty': shape_penalty,
            'overtrade_penalty': overtrade_penalty,
            'corr_term': corr_term,
        }

    @staticmethod
    def _enable_windows_ansi():
        if os.name != "nt":
            return
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass

    def _color(self, enabled=True):
        if not enabled or os.environ.get("NO_COLOR"):
            class _No:
                """Puste kody ANSI (wyłączone kolorowanie)."""
                reset = bold = dim = red = green = yellow = blue = magenta = cyan = ""
            return _No()

        class _Ansi:
            """Kody ANSI do kolorowego wyjścia konsoli."""
            reset = "\033[0m"
            bold = "\033[1m"
            dim = "\033[2m"
            red = "\033[31m"
            green = "\033[32m"
            yellow = "\033[33m"
            blue = "\033[34m"
            magenta = "\033[35m"
            cyan = "\033[36m"

        return _Ansi()

    def _evaluate_agent(self,
                        trial: optuna.Trial,
                        agent_filename: str,
                        mp_threshold: float) -> float:
        """Ewaluacja na zbiorze testowym."""
        try:
            print(f"[{self.agent_type.upper()} Trial {trial.number}] Ewaluacja...")

            sim_result = self.sim.run_simulation(
                rabbit_msg_id=None,
                model_paths=self.model_files,
                data_files=self.eval_data_files,
                agent_rl=agent_filename,
                balance_start=1000.0,
                add_fee=True,
                sim_mode="rl",
                rlagent_deterministic=True,
                slippage=0,
                wallet_percentage_limit=0.9,
                max_wallet_percentage_crypto=1.0,
                stop_loss=1.0,
                threshold_buy=mp_threshold,
                threshold_sell=mp_threshold,
                mincost=1.0,
                tx_fraction=0.01,
                max_fraction_when_confident=0.5,
                plots=False,
                logs=False,
                results=True,
                experiment_version=self.experiment_version
            )

            final_value = sim_result.get('final_portfolio_value', 1000.0)
            sharpe = sim_result.get('sharpe_ratio', 0.0)
            max_dd = sim_result.get('max_drawdown', -100.0)
            metric = (final_value - 1000.0) / 1000.0 * 100

            print(f"\n[{self.agent_type.upper()} Trial {trial.number}] - wyniki:")
            print(f"  Final Value: ${final_value:.2f}")
            print(f"  Return: {metric:.2f}%")
            print(f"  Sharpe: {sharpe:.2f}")
            print(f"  Max DD: {max_dd:.2f}%\n")

            trial.set_user_attr('final_value', final_value)
            trial.set_user_attr('sharpe_ratio', sharpe)
            trial.set_user_attr('max_drawdown', max_dd)
            trial.set_user_attr('agent_file', agent_filename)

            # Save trial result
            self._save_trial_result(trial, metric, final_value, sharpe, max_dd, agent_filename)

            return metric

        except Exception as e:
            print(f"\n[{self.agent_type.upper()} Trial {trial.number}] Evaluation FAILED: {e}")
            import traceback
            traceback.print_exc()
            return -1000.0

    def _evaluate_mp_simulation(self,
                                trial: optuna.Trial,
                                threshold_buy: float,
                                threshold_sell: float,
                                wallet_percentage_limit: float,
                                max_wallet_percentage_crypto: float,
                                stop_loss: float,
                                tx_fraction: float,
                                max_fraction_when_confident: float,
                                slippage: float,
                                mincost: float,
                                balance_start: float,
                                mp_alloc_bias: float = 0.5,
                                mp_alloc_scale: float = 0.5,
                                mp_confidence: float = 0.5,
                                mp_urgency: float = 0.5,
                                mp_signal_deadzone: float = 0.15,
                                mp_rebalance_deadzone: float = 0.25,
                                mp_ema_alpha_min: float = 0.02,
                                mp_ema_alpha_span: float = 0.08,
                                stop_loss_cooldown: int = 120,
                                clamp_wallet_cap_to_1n: bool = True,
                                window=None) -> float:
        """Evaluate simulation using only MP models (no RL agent)."""
        try:
            # predykcje cache'owane w SimulationService (reload tylko przy zmianie window)
            sim_result = self.sim.run_simulation(
                rabbit_msg_id=None,
                model_paths=self.model_files,
                data_files=self.eval_data_files,
                agent_rl=None,  # NO RL AGENT - MP models only!
                balance_start=balance_start,
                add_fee=True,
                sim_mode="mp",
                rlagent_deterministic=False,
                slippage=slippage,
                wallet_percentage_limit=wallet_percentage_limit,
                max_wallet_percentage_crypto=max_wallet_percentage_crypto,
                stop_loss=stop_loss,
                stop_loss_cooldown=stop_loss_cooldown,
                threshold_buy=threshold_buy,
                threshold_sell=threshold_sell,
                mincost=mincost,
                tx_fraction=tx_fraction,
                max_fraction_when_confident=max_fraction_when_confident,
                mp_alloc_bias=mp_alloc_bias,
                mp_alloc_scale=mp_alloc_scale,
                mp_confidence=mp_confidence,
                mp_urgency=mp_urgency,
                mp_signal_deadzone=mp_signal_deadzone,
                mp_rebalance_deadzone=mp_rebalance_deadzone,
                mp_ema_alpha_min=mp_ema_alpha_min,
                mp_ema_alpha_span=mp_ema_alpha_span,
                clamp_wallet_cap_to_1n=bool(clamp_wallet_cap_to_1n),
                plots=False,
                logs=False,
                results=False,
                experiment_version=self.experiment_version,
                window_id=window,
                quiet=True,
            )

            import math

            def _finite(x, default=0.0):
                try:
                    v = float(x)
                except (TypeError, ValueError):
                    return default
                return v if math.isfinite(v) else default

            final_value = _finite(sim_result.get('final_portfolio_value', balance_start), balance_start)
            sharpe = _finite(sim_result.get('sharpe_ratio', 0.0), 0.0)
            max_dd = _finite(sim_result.get('max_drawdown', -100.0), 0.0)
            tx_count = int(sim_result.get('transaction_count') or 0)
            turnover = sim_result.get('turnover')
            closed_trades = int(sim_result.get('closed_trades_count') or 0)

            strat_return = (final_value - balance_start) / balance_start * 100.0
            hodl_return = sim_result.get('hodl_return')
            if hodl_return is None:
                hodl_return = 0.0
            excess = strat_return - float(hodl_return)

            n_ticks = int(sim_result.get('n_ticks') or getattr(self.sim, 'num_ticks', 0) or 0)
            periods_per_year = float(getattr(self, 'periods_per_year', 525_600))  # 1m
            days = n_ticks / (periods_per_year / 365.0) if n_ticks > 0 else 0.0

            # Constraints (hard INV): tx / closed / corr≈HODL / overtrade-hard
            score_bits = self._anti_hodl_penalties(
                tx_count=tx_count,
                closed_trades=closed_trades,
                days=days,
                portfolio_values=sim_result.get('portfolio_values'),
                hodl_refs=sim_result.get('hodl_crypto_ref_values'),
            )
            overtrade_penalty = score_bits['overtrade_penalty']
            hodl_corr = score_bits['hodl_corr']
            min_tx = score_bits['min_tx']
            max_tx = score_bits['max_tx']
            max_tx_hard = score_bits.get('max_tx_hard', max_tx)
            min_closed = score_bits['min_closed']
            valid = score_bits['valid']
            invalid_reasons = score_bits.get('invalid_reasons') or []
            # shape/corr_term zostają tylko diagnostyką (nie wchodzą do metric)
            shape_penalty = score_bits['shape_penalty']
            corr_term = score_bits['corr_term']

            time_in_cash = float(sim_result.get('time_in_cash') or 0.0)

            ruin_floor = float(getattr(self, 'ruin_return_floor', -20.0))
            ruin_w = float(getattr(self, 'ruin_penalty_weight', 2.0))
            ruin_penalty = max(0.0, ruin_floor - strat_return) * ruin_w

            # Metryka OOS-first: to co widać na testach (excess, Sharpe, DD).
            # corr/tx/closed = bramki valid, nie premia za „bycie nie-HODLem”.
            # cash_term wyłączony — nagradzanie gotówki pchało w 0 tx.
            sharpe_w = float(getattr(self, 'sharpe_weight', 5.0))
            dd_w = float(getattr(self, 'dd_weight', 0.75))
            sharpe_term = sharpe_w * sharpe
            dd_abs = max(0.0, -max_dd)          # max_dd zwykle ujemne, np. -6.8
            dd_penalty = dd_w * dd_abs
            inv_excess_w = float(getattr(self, 'invalid_excess_weight', 0.05))
            if valid:
                metric = (
                    excess + sharpe_term
                    - dd_penalty
                    - overtrade_penalty
                    - ruin_penalty
                )
            else:
                metric = (
                    score_bits['invalid_floor']
                    + inv_excess_w * excess
                    + 0.25 * sharpe_term
                    - 0.5 * dd_penalty
                    - overtrade_penalty
                    - ruin_penalty
                )
            metric = _finite(metric, score_bits['invalid_floor'])
            cash_term = 0.0  # legacy attr w logach / JSON
            trial.set_user_attr('final_value', final_value)
            trial.set_user_attr('sharpe_ratio', sharpe)
            trial.set_user_attr('max_drawdown', max_dd)
            trial.set_user_attr('sim_mode', 'mp')
            trial.set_user_attr('window', window)
            trial.set_user_attr('strat_return', strat_return)
            trial.set_user_attr('hodl_return', float(hodl_return))
            trial.set_user_attr('excess_return', excess)
            trial.set_user_attr('transaction_count', tx_count)
            trial.set_user_attr('min_transactions', min_tx)
            trial.set_user_attr('max_transactions', max_tx)
            trial.set_user_attr('max_transactions_hard', max_tx_hard)
            trial.set_user_attr('closed_trades_count', closed_trades)
            trial.set_user_attr('min_closed_trades', min_closed)
            trial.set_user_attr('hodl_corr', hodl_corr)
            trial.set_user_attr('valid_trial', valid)
            trial.set_user_attr('invalid_reasons', invalid_reasons)
            trial.set_user_attr('window_days', days)
            trial.set_user_attr('shape_penalty', shape_penalty)
            trial.set_user_attr('overtrade_penalty', overtrade_penalty)
            trial.set_user_attr('corr_term', corr_term)
            trial.set_user_attr('cash_term', cash_term)
            trial.set_user_attr('ruin_penalty', ruin_penalty)
            trial.set_user_attr('time_in_cash', time_in_cash)
            trial.set_user_attr('sharpe_term', sharpe_term)
            trial.set_user_attr('dd_penalty', dd_penalty)
            if turnover is not None:
                trial.set_user_attr('turnover', turnover)

            self._save_trial_result(
                trial, metric, final_value, sharpe, max_dd, agent_path=None,
                extra={
                    'strat_return': strat_return,
                    'hodl_return': float(hodl_return),
                    'excess_return': excess,
                    'transaction_count': tx_count,
                    'min_transactions': min_tx,
                    'max_transactions': max_tx,
                    'max_transactions_hard': max_tx_hard,
                    'closed_trades_count': closed_trades,
                    'min_closed_trades': min_closed,
                    'hodl_corr': hodl_corr,
                    'valid_trial': valid,
                    'invalid_reasons': invalid_reasons,
                    'window_days': days,
                    'shape_penalty': shape_penalty,
                    'overtrade_penalty': overtrade_penalty,
                    'corr_term': corr_term,
                    'cash_term': cash_term,
                    'ruin_penalty': ruin_penalty,
                    'time_in_cash': time_in_cash,
                    'sharpe_term': sharpe_term,
                    'dd_penalty': dd_penalty,
                    'turnover': turnover,
                },
            )

            if getattr(self, 'live_logging', True):
                self._log_trial_compact(
                    trial=trial,
                    metric=metric,
                    final_value=final_value,
                    strat_return=strat_return,
                    hodl_return=float(hodl_return),
                    excess=excess,
                    tx_count=tx_count,
                    min_tx=min_tx,
                    max_tx=max_tx,
                    closed_trades=closed_trades,
                    min_closed=min_closed,
                    hodl_corr=hodl_corr,
                    valid=valid,
                    invalid_reasons=invalid_reasons,
                    shape_penalty=shape_penalty,
                    overtrade_penalty=overtrade_penalty,
                    sharpe=sharpe,
                    sharpe_term=sharpe_term,
                    max_dd=max_dd,
                    window=window,
                    corr_term=corr_term,
                    cash_term=cash_term,
                    ruin_penalty=ruin_penalty,
                    dd_penalty=dd_penalty,
                )

            return metric

        except Exception as e:
            print(f"[Optuna] Trial {trial.number} FAILED: {e}")
            import traceback
            traceback.print_exc()
            return -1000.0

    # legacy alias (old name)
    _evaluate_ml_simulation = _evaluate_mp_simulation

    def _log_trial_compact(
        self,
        trial,
        metric,
        final_value,
        strat_return,
        hodl_return,
        excess,
        tx_count,
        min_tx,
        max_tx,
        closed_trades,
        min_closed,
        hodl_corr,
        valid,
        invalid_reasons,
        shape_penalty,
        overtrade_penalty,
        sharpe,
        sharpe_term,
        max_dd,
        window,
        corr_term=0.0,
        cash_term=0.0,
        ruin_penalty=0.0,
        dd_penalty=0.0,
    ):
        """Jedna zwarta, kolorowa karta wyniku trialu."""
        c = self._color(True)
        n_total = getattr(self, 'n_trials_total', None)
        stage = getattr(self, '_stage_label', None)
        stage_txt = f"{c.yellow}{c.bold}{stage}{c.reset} " if stage else ""
        head = f"{stage_txt}Trial {trial.number}" + (f"/{n_total}" if n_total else "")
        if valid:
            flag = f"{c.green}{c.bold}OK{c.reset}"
        else:
            why = ",".join(invalid_reasons) if invalid_reasons else "?"
            flag = f"{c.red}{c.bold}INV{c.reset}{c.dim}({why}){c.reset}"

        bias = trial.user_attrs.get('log_bias', trial.params.get('mp_alloc_bias'))
        scale = trial.user_attrs.get('log_scale', trial.params.get('mp_alloc_scale'))
        conf = trial.user_attrs.get('log_conf', trial.params.get('mp_confidence'))
        urg = trial.user_attrs.get('log_urg', trial.params.get('mp_urgency'))
        minc = trial.user_attrs.get('log_mincost', trial.params.get('mincost'))
        wlim = trial.user_attrs.get('log_wlim', trial.params.get('wallet_percentage_limit'))
        maxc = trial.user_attrs.get('log_maxcrypto', trial.params.get('max_wallet_percentage_crypto'))
        sl = trial.user_attrs.get('log_sl', trial.params.get('stop_loss'))
        sl_cd = trial.user_attrs.get('log_sl_cd', trial.params.get('stop_loss_cooldown'))

        def _f(v, nd=3):
            try:
                return f"{float(v):.{nd}f}"
            except (TypeError, ValueError):
                return str(v)

        def _signed(v, nd=2, good_positive=True):
            try:
                x = float(v)
            except (TypeError, ValueError):
                return str(v)
            ok = (x >= 0) if good_positive else (x <= 0)
            col = c.green if ok else c.red
            return f"{col}{x:+.{nd}f}{c.reset}"

        best = getattr(self, '_live_best', None)
        is_new_best = best is None or metric > best.get('metric', float('-inf'))
        if is_new_best:
            self._live_best = {
                'trial': trial.number,
                'metric': metric,
                'final_value': final_value,
                'excess': excess,
                'corr': hodl_corr,
                'valid': valid,
            }
            best_txt = f" {c.yellow}{c.bold}★ NEW BEST{c.reset}"
        else:
            b = self._live_best
            best_txt = (
                f" {c.dim}| best t{b['trial']}: metric={b['metric']:.2f} "
                f"excess={b.get('excess', float('nan')):+.2f} ${b['final_value']:.0f}{c.reset}"
            )

        max_tx_hard = int(trial.user_attrs.get('max_transactions_hard') or max_tx)
        tx_ok = min_tx <= tx_count <= max_tx
        tx_soft = (not tx_ok) and (tx_count <= max_tx_hard) and (tx_count >= min_tx)
        tx_col = c.green if tx_ok else (c.yellow if tx_soft else c.red)
        corr_hard = float(getattr(self, 'hodl_corr_hard', 0.97))
        corr_col = c.green if hodl_corr < corr_hard else c.red

        print(
            f"\n{c.cyan}{c.bold}── {head}{c.reset} [{flag}] "
            f"{c.blue}window={window}{c.reset}{best_txt}\n"
            f"   {c.dim}params{c.reset}  bias={_f(bias)} scale={_f(scale)} conf={_f(conf)} "
            f"urg={_f(urg)} minc={_f(minc,1)} wlim={_f(wlim)} maxc={_f(maxc)} "
            f"sl={_f(sl)} cd={sl_cd}\n"
            f"   {c.dim}result{c.reset}  {c.bold}${final_value:.0f}{c.reset}  "
            f"ret={_signed(strat_return)}%  hodl={hodl_return:+.2f}%  "
            f"excess={_signed(excess)}%\n"
            f"   {c.dim}gate{c.reset}    {tx_col}tx={tx_count}∈[{min_tx},{max_tx}]"
            f"..{max_tx_hard}{c.reset}  "
            f"closed={closed_trades}/{min_closed}  "
            f"{corr_col}corr={hodl_corr:.3f}{c.reset}  "
            f"ot={overtrade_penalty:.1f}\n"
            f"   {c.magenta}{c.bold}score{c.reset}   metric={c.bold}{metric:.2f}{c.reset}  "
            f"excess={excess:+.2f}  sharpe={_signed(sharpe)} (w·s={sharpe_term:+.1f})  "
            f"dd={_signed(max_dd, good_positive=False)}% (−{dd_penalty:.1f})  "
            f"ruin={ruin_penalty:.1f}\n"
        )

    # ------------------------------------------------------------------ staged

    def _print_stage_banner(self, stage: str, detail: str = ""):
        c = self._color(True)
        colors = {"E1": c.cyan, "E2": c.magenta, "E3": c.yellow, "PICK": c.green, "DONE": c.green}
        col = colors.get(stage, c.blue)
        line = f"{'═' * 70}"
        print(f"\n{col}{c.bold}{line}{c.reset}")
        print(f"{col}{c.bold}  Optuna {stage}{c.reset}" + (f"  {c.dim}{detail}{c.reset}" if detail else ""))
        print(f"{col}{c.bold}{line}{c.reset}\n")

    def _config_window_spec(self, config: Dict[str, Any]) -> Optional[str]:
        if not config or "window" not in config:
            return None
        spec = config["window"]
        if isinstance(spec, (list, tuple)):
            if len(spec) >= 2 and spec[0] == "C":
                return str(spec[1])
            if len(spec) == 1:
                return str(spec[0])
            return "+".join(str(x) for x in spec)
        return str(spec)

    def _with_window(self, config: Dict[str, Any], window_spec: str) -> Dict[str, Any]:
        out = dict(config or {})
        out["window"] = ["C", window_spec]
        return out

    def _expand_ids(self, window_spec: Optional[str]) -> List[str]:
        if not window_spec:
            return []
        try:
            windows = load_windows(self.base_dir)
            return expand_window_ids(windows, window_spec) or []
        except Exception:
            if "+" in str(window_spec):
                return [p.strip() for p in str(window_spec).split("+") if p.strip()]
            return [str(window_spec)]

    def _tail_window_spec(self, window_spec: Optional[str], n_tail: int = 2) -> str:
        ids = self._expand_ids(window_spec)
        if not ids:
            return window_spec or "train"
        n = max(1, min(int(n_tail), len(ids)))
        return "+".join(ids[-n:])

    def _mid_window_spec(self, window_spec: Optional[str]) -> str:
        """Środek bloku (nie ogon) — do arbitrażu E1/E2/E3."""
        ids = self._expand_ids(window_spec)
        if not ids:
            return window_spec or "train"
        if len(ids) == 1:
            return ids[0]
        if len(ids) == 2:
            return ids[0]
        mid = ids[len(ids) // 2 - 1]  # bliżej środka, nie ostatni
        return mid

    def _apply_range_overrides(self, config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Nałóż warm-start / zawężone zakresy na F/FL/I; reszta bez zmian."""
        out = dict(config or {})
        for key, ov in (overrides or {}).items():
            if key not in out:
                continue
            spec = out[key]
            if not isinstance(spec, (list, tuple)) or len(spec) < 3:
                continue
            ptype = spec[0]
            if ptype not in ("F", "FL", "I"):
                continue
            if isinstance(ov, (list, tuple)) and len(ov) >= 2:
                lo, hi = float(ov[0]), float(ov[1])
            elif isinstance(ov, dict):
                lo, hi = float(ov["low"]), float(ov["high"])
            else:
                continue
            if lo > hi:
                lo, hi = hi, lo
            # nie wychodź poza oryginalny box
            orig_lo, orig_hi = float(spec[1]), float(spec[2])
            lo = max(orig_lo, lo)
            hi = min(orig_hi, hi)
            if lo >= hi:
                mid = 0.5 * (orig_lo + orig_hi)
                lo, hi = max(orig_lo, mid * 0.98), min(orig_hi, mid * 1.02)
            out[key] = [ptype, lo, hi]
        return out

    def _narrow_config_from_study(
        self,
        base_config: Dict[str, Any],
        study: optuna.Study,
        top_k: int = 15,
        q_lo: float = 0.05,
        q_hi: float = 0.95,
        include_params: Optional[Dict[str, Any]] = None,
        pad_frac: float = 0.02,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Zawęż F/FL/I do percentyli top‑K (prefer valid).

        Zawsze rozszerza box tak, by objąć ``include_params`` (zwykle E1-best),
        z drobnym pad_frac względem oryginalnego zakresu — żeby narrowing nie
        wycinał championa E1.
        """
        import numpy as np

        trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        valid = [t for t in trials if t.user_attrs.get("valid_trial") is True]
        pool = valid if valid else trials
        pool = sorted(pool, key=lambda t: t.value, reverse=True)[: max(1, int(top_k))]
        if not pool:
            return dict(base_config), {}

        ranges_info = {}
        out = dict(base_config)
        cover = include_params or {}
        for key, spec in base_config.items():
            if not isinstance(spec, (list, tuple)) or len(spec) < 3:
                continue
            ptype = spec[0]
            if ptype not in ("F", "FL", "I"):
                continue
            vals = [t.params[key] for t in pool if key in t.params]
            if len(vals) < 2:
                continue
            lo = float(np.quantile(vals, q_lo))
            hi = float(np.quantile(vals, q_hi))
            orig_lo, orig_hi = float(spec[1]), float(spec[2])
            span = max(1e-9, orig_hi - orig_lo)
            # zawsze pokryj championa (+ pad)
            if key in cover and cover[key] is not None:
                try:
                    cv = float(cover[key])
                    pad = float(pad_frac) * span
                    lo = min(lo, cv - pad)
                    hi = max(hi, cv + pad)
                except (TypeError, ValueError):
                    pass
            lo = max(orig_lo, lo)
            hi = min(orig_hi, hi)
            if lo >= hi:
                mid = 0.5 * (lo + hi) if vals else 0.5 * (orig_lo + orig_hi)
                lo = max(orig_lo, mid - 0.05 * span)
                hi = min(orig_hi, mid + 0.05 * span)
            if ptype == "I":
                lo, hi = int(np.floor(lo)), int(np.ceil(hi))
                if lo >= hi:
                    hi = lo + 1
                out[key] = [ptype, lo, hi]
            else:
                out[key] = [ptype, lo, hi]
            ranges_info[key] = {"low": lo, "high": hi, "n": len(vals)}
        return out, ranges_info

    def _clip_params_to_config(
        self, params: Dict[str, Any], config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Przytnij tunable knoby do aktualnego search boxu (pod enqueue)."""
        out = {}
        for key, val in (params or {}).items():
            spec = (config or {}).get(key)
            if not isinstance(spec, (list, tuple)) or len(spec) < 2:
                continue
            ptype = spec[0]
            if ptype == "C":
                out[key] = spec[1]
                continue
            if ptype not in ("F", "FL", "I") or len(spec) < 3:
                continue
            lo, hi = float(spec[1]), float(spec[2])
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            v = min(hi, max(lo, v))
            out[key] = int(round(v)) if ptype == "I" else v
        return out

    def _pick_best_trial(self, study: optuna.Study) -> Optional[optuna.trial.FrozenTrial]:
        tops = self._pick_top_trials(study, k=1)
        return tops[0] if tops else None

    def _pick_top_trials(
        self, study: Optional[optuna.Study], k: int = 3,
    ) -> List[optuna.trial.FrozenTrial]:
        """Top‑K complete trials (prefer valid), metric desc."""
        if study is None:
            return []
        trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        if not trials:
            return []
        valid = [t for t in trials if t.user_attrs.get("valid_trial") is True]
        pool = valid if valid else trials
        pool = sorted(pool, key=lambda t: t.value, reverse=True)
        return pool[: max(1, int(k))]

    @staticmethod
    def _params_fingerprint(params: Dict[str, Any]) -> tuple:
        """Stable key for dedupe przy picku E1/E2 top‑K."""
        items = []
        for key in sorted((params or {}).keys()):
            if key in ("balance_start", "window", "threshold_buy", "threshold_sell",
                       "stop_loss_cooldown", "slippage", "mp_ema_alpha_min", "mp_ema_alpha_span"):
                continue
            val = params[key]
            try:
                items.append((key, round(float(val), 8)))
            except (TypeError, ValueError):
                items.append((key, val))
        return tuple(items)

    def _merge_fixed_params(self, config: Dict[str, Any], tuned: Dict[str, Any]) -> Dict[str, Any]:
        """Dołącz stałe C z configu do best_params."""
        out = dict(tuned or {})
        for key, spec in (config or {}).items():
            if key in out or key == "window":
                continue
            if isinstance(spec, (list, tuple)) and len(spec) >= 2 and spec[0] == "C" and len(spec) == 2:
                out[key] = spec[1]
            elif not isinstance(spec, (list, tuple)):
                out[key] = spec
        return out

    def _score_params_on_window(self, params: Dict[str, Any], window_spec: str) -> Dict[str, Any]:
        """Jednorazowa ewaluacja knóbów na zadanym oknie (arbitraż E2/E3)."""
        p = dict(params or {})
        balance_start = float(p.get("balance_start", 1000.0))
        sim_result = self.sim.run_simulation(
            rabbit_msg_id=None,
            model_paths=self.model_files,
            data_files=self.eval_data_files,
            agent_rl=None,
            balance_start=balance_start,
            add_fee=True,
            sim_mode="mp",
            rlagent_deterministic=False,
            slippage=float(p.get("slippage", 0.0) or 0.0),
            wallet_percentage_limit=float(p.get("wallet_percentage_limit", 0.9)),
            max_wallet_percentage_crypto=float(p.get("max_wallet_percentage_crypto", 1.0)),
            stop_loss=float(p.get("stop_loss", 1.0)),
            stop_loss_cooldown=int(p.get("stop_loss_cooldown", 120)),
            threshold_buy=float(p.get("threshold_buy", 0.5)),
            threshold_sell=float(p.get("threshold_sell", 0.5)),
            mincost=float(p.get("mincost", 1.0)),
            tx_fraction=float(p.get("tx_fraction", 0.01)),
            max_fraction_when_confident=float(p.get("max_fraction_when_confident", 0.5)),
            mp_alloc_bias=float(p.get("mp_alloc_bias", 0.5)),
            mp_alloc_scale=float(p.get("mp_alloc_scale", 0.5)),
            mp_confidence=float(p.get("mp_confidence", 0.5)),
            mp_urgency=float(p.get("mp_urgency", 0.5)),
            mp_signal_deadzone=float(p.get("mp_signal_deadzone", 0.15)),
            mp_rebalance_deadzone=float(p.get("mp_rebalance_deadzone", 0.25)),
            mp_ema_alpha_min=float(p.get("mp_ema_alpha_min", 0.02)),
            mp_ema_alpha_span=float(p.get("mp_ema_alpha_span", 0.08)),
            plots=False,
            logs=False,
            results=False,
            experiment_version=self.experiment_version,
            window_id=window_spec,
            quiet=True,
        )
        final = float(sim_result.get("final_portfolio_value", balance_start) or balance_start)
        sharpe = float(sim_result.get("sharpe_ratio", 0.0) or 0.0)
        max_dd = float(sim_result.get("max_drawdown", 0.0) or 0.0)
        strat = (final - balance_start) / balance_start * 100.0
        hodl = float(sim_result.get("hodl_return", 0.0) or 0.0)
        excess = strat - hodl
        sharpe_w = float(getattr(self, "sharpe_weight", 5.0))
        dd_w = float(getattr(self, "dd_weight", 0.75))
        metric = excess + sharpe_w * sharpe - dd_w * max(0.0, -max_dd)
        return {
            "metric": metric,
            "final_value": final,
            "excess_return": excess,
            "strat_return": strat,
            "hodl_return": hodl,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "window": window_spec,
            "valid_trial": True,  # soft score; gates nie blokują picka mid
        }

    def _run_stage_study(
        self,
        *,
        stage: str,
        study_name: str,
        config: Dict[str, Any],
        defaults: Dict[str, Any],
        n_trials: int,
        n_jobs: int,
        timeout,
        seed: int = 42,
        enqueue_trials: Optional[List[Dict[str, Any]]] = None,
    ) -> optuna.Study:
        c = self._color(True)
        self._stage_label = stage
        self.n_trials_total = n_trials
        self._live_best = None
        self._parse_config(config, defaults)

        self.study_name = study_name
        self.storage = f"sqlite:///{self.log_path}/{study_name}.db"
        self.results_file = os.path.join(self.log_path, f"{study_name}_results.json")

        win = self._config_window_spec(config)
        self._print_stage_banner(
            stage,
            f"trials={n_trials}  window={win}  study={study_name}",
        )
        # pokaż zawężone F jeśli nie pełne
        float_keys = [
            k for k, s in config.items()
            if isinstance(s, (list, tuple)) and len(s) >= 3 and s[0] in ("F", "FL", "I")
        ]
        if float_keys and stage in ("E2", "E3"):
            print(f"  {c.dim}search box:{c.reset}")
            for k in float_keys:
                s = config[k]
                print(f"    {c.blue}{k}{c.reset} ∈ [{s[1]:.4g}, {s[2]:.4g}]")
            print()

        study = optuna.create_study(
            study_name=study_name,
            storage=self.storage,
            load_if_exists=True,
            direction="maximize",
            sampler=TPESampler(seed=seed),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )
        for raw in enqueue_trials or []:
            clipped = self._clip_params_to_config(raw, config)
            if not clipped:
                continue
            try:
                study.enqueue_trial(clipped, skip_if_exists=True)
                print(
                    f"  {c.green}enqueue{c.reset}  {len(clipped)} knobs "
                    f"(E1-best → {stage})"
                )
            except Exception as e:
                print(f"  {c.yellow}enqueue skip{c.reset}: {e}")
        study.optimize(
            self._objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=not self.live_logging,
        )
        best = self._pick_best_trial(study)
        if best is not None:
            ua = best.user_attrs
            vcol = c.green if ua.get("valid_trial") else c.red
            print(
                f"\n  {c.bold}{stage} best{c.reset}  t{best.number}  "
                f"metric={c.bold}{best.value:.2f}{c.reset}  "
                f"{vcol}valid={ua.get('valid_trial')}{c.reset}  "
                f"final=${ua.get('final_value')}  excess={ua.get('excess_return')}"
            )
        return study

    def run_tuning(self,
                   model_files: Dict[str, str],
                   eval_data_files: Dict[str, str],
                   train_data_files: Dict[str, str] = None,
                   config: Dict[str, List] = None,
                   defaults: Dict[str, Any] = None,
                   n_trials: int = 50,
                   n_jobs: int = 1,
                   timeout: int = None,
                   study_name: str = None,
                   live_logging: bool = True,
                   experiment_version: str = "v0",
                   seed: int = 42,
                   # --- staged Optuna ---
                   staged: bool = False,
                   e1_trials: int = 50,
                   e2_trials: int = 100,
                   e3_trials: int = 0,  # legacy; E3 nie robi już Optuny
                   e1_full: bool = True,
                   warm_start_ranges: Dict[str, Any] = None,
                   e2_top_k: int = 15,
                   e2_q_low: float = 0.05,
                   e2_q_high: float = 0.95,
                   e3_tail_windows: int = 2,
                   pick_top_k: int = 3,
                   pick_train_blend: float = 0.0,  # legacy unused
                   ) -> optuna.Study:
        """
        Run hyperparameter tuning.

        staged=True → E1 (full box) → E2 (narrow, covers E1-best + enqueue)
        → E3 = score top‑K z E1 i E2 na ogonie bloku (bez osobnego Optuny).
        e1_full=True (default): E1 na pełnych zakresach z config.
        e1_full=False: pomiń E1; E2 startuje z warm_start_ranges (albo pełnym boxem).
        pick_top_k: ile bestów z E1 i z E2 brać do scoringu na ogonie (2 lub 3).
        """

        self.model_files = model_files
        self.train_data_files = train_data_files
        self.eval_data_files = eval_data_files
        self.live_logging = live_logging
        self.experiment_version = experiment_version
        self.optuna_seed = int(seed)
        self._live_best = None
        self._stage_label = None
        if live_logging:
            self._enable_windows_ansi()
        if config is None:
            config = {}

        if study_name is None:
            timestamp = datetime.now().strftime("%d-%m__%H-%M-%S")
            study_name = f"{self.agent_type}_tuning_{timestamp}"

        self.study_name = study_name
        os.makedirs(self.log_path, exist_ok=True)
        self.best_params_file = os.path.join(self.log_path, f"{study_name}_best_params.json")
        # aggregated results path (stages append with suffix)
        self.results_file = os.path.join(self.log_path, f"{study_name}_results.json")

        if live_logging:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        c = self._color(True)
        base_window = self._config_window_spec(config)
        print(f"\n{c.cyan}{c.bold}{'═' * 70}{c.reset}")
        print(f"{c.cyan}{c.bold}  Optuna: {study_name}{c.reset}")
        print(f"{c.cyan}{c.bold}{'═' * 70}{c.reset}")
        print(
            f"  algo={self.agent_type}  version={experiment_version}  "
            f"seed={self.optuna_seed}  "
            f"staged={c.bold}{staged}{c.reset}  jobs={n_jobs}"
        )
        if staged:
            print(
                f"  stages   E1={e1_trials} (full={e1_full})  "
                f"E2={e2_trials} (narrow top_k={e2_top_k}, q={e2_q_low:.0%}-{e2_q_high:.0%})  "
                f"E3=score top‑{int(pick_top_k)}×E1/E2 on last {e3_tail_windows} windows"
            )
            print(f"  window   train_block={base_window}")
        else:
            print(f"  trials={n_trials}  window={base_window}")
        print(
            f"  metric(OK)  = excess + {float(getattr(self, 'sharpe_weight', 5.0)):.0f}·Sharpe "
            f"− {float(getattr(self, 'dd_weight', 0.75)):.2f}·|DD| − ot − ruin"
        )
        print(
            f"  gates: tx∈[min,max]  closed≥min  "
            f"corr<{float(getattr(self, 'hodl_corr_hard', 0.97)):.3f}"
        )
        print(f"{c.cyan}{'═' * 70}{c.reset}\n")

        if not staged:
            self._stage_label = None
            self.n_trials_total = n_trials
            self._parse_config(config, defaults)
            self.storage = f"sqlite:///{self.log_path}/{study_name}.db"
            study = optuna.create_study(
                study_name=study_name,
                storage=self.storage,
                load_if_exists=True,
                direction="maximize",
                sampler=TPESampler(seed=self.optuna_seed),
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            )
            study.set_user_attr("seed", self.optuna_seed)
            study.optimize(
                self._objective,
                n_trials=n_trials,
                n_jobs=n_jobs,
                timeout=timeout,
                show_progress_bar=not live_logging,
            )
            self._finalize_study(study, config, defaults, stage_meta={"staged": False})
            return study

        # ---------- STAGED ----------
        # E1 search → E2 narrow search → E3 = score top‑K(E1)+top‑K(E2) on tail
        k_pick = max(1, int(pick_top_k))
        stage_meta: Dict[str, Any] = {
            "staged": True,
            "e1_full": bool(e1_full),
            "e1_trials": int(e1_trials),
            "e2_trials": int(e2_trials),
            "e3_mode": "score_tail",
            "e3_tail_windows": int(e3_tail_windows),
            "pick_top_k": k_pick,
            "base_window": base_window,
        }
        study_e1 = study_e2 = None
        ranges_e2 = {}
        cfg_e2 = dict(config)
        enqueue_e2: List[Dict[str, Any]] = []

        # E1
        if e1_full:
            study_e1 = self._run_stage_study(
                stage="E1",
                study_name=f"{study_name}_e1",
                config=config,
                defaults=defaults,
                n_trials=int(e1_trials),
                n_jobs=n_jobs,
                timeout=timeout,
                seed=self.optuna_seed,
            )
            best_e1 = self._pick_best_trial(study_e1)
            cfg_e2, ranges_e2 = self._narrow_config_from_study(
                config,
                study_e1,
                top_k=e2_top_k,
                q_lo=e2_q_low,
                q_hi=e2_q_high,
                include_params=dict(best_e1.params) if best_e1 else None,
            )
            stage_meta["e1_study"] = f"{study_name}_e1"
            stage_meta["e2_ranges"] = ranges_e2
            if best_e1 is not None:
                enqueue_e2.append(dict(best_e1.params))
            print(
                f"  {c.green}E1→E2{c.reset} narrowed {len(ranges_e2)} knobs "
                f"from top‑{e2_top_k} (covers E1-best)"
            )
        else:
            self._print_stage_banner("E1", "SKIPPED (e1_full=False)")
            if warm_start_ranges:
                cfg_e2 = self._apply_range_overrides(config, warm_start_ranges)
                ranges_e2 = warm_start_ranges
                print(f"  {c.yellow}warm-start{c.reset}  applied {len(warm_start_ranges)} range overrides → E2")
            else:
                print(f"  {c.yellow}warn{c.reset}  brak warm_start_ranges — E2 na pełnym boxie")
                cfg_e2 = dict(config)
            stage_meta["e1_skipped"] = True
            stage_meta["e2_ranges"] = ranges_e2

        # E2
        study_e2 = self._run_stage_study(
            stage="E2",
            study_name=f"{study_name}_e2",
            config=cfg_e2,
            defaults=defaults,
            n_trials=int(e2_trials),
            n_jobs=n_jobs,
            timeout=timeout,
            seed=self.optuna_seed + 1,
            enqueue_trials=enqueue_e2,
        )
        stage_meta["e2_study"] = f"{study_name}_e2"

        # E3 — bez Optuny: top‑K z E1 i E2 na ogonie bloku
        tail = self._tail_window_spec(base_window, e3_tail_windows)
        stage_meta["e3_window"] = tail
        self._print_stage_banner(
            "E3",
            f"score top‑{k_pick} E1 + top‑{k_pick} E2 on tail={tail}  (no Optuna)",
        )

        candidates = []  # (label, params, stage_metric, trial_no, stage)
        seen_fp = set()
        for stage_label, study, merge_cfg, color_name in (
            ("E1", study_e1, config, "cyan"),
            ("E2", study_e2, cfg_e2, "magenta"),
        ):
            tops = self._pick_top_trials(study, k=k_pick)
            for rank, trial in enumerate(tops, start=1):
                params = self._merge_fixed_params(merge_cfg, dict(trial.params))
                fp = self._params_fingerprint(params)
                if fp in seen_fp:
                    print(
                        f"  {c.dim}skip {stage_label}#{rank} t{trial.number} "
                        f"(dup params){c.reset}"
                    )
                    continue
                seen_fp.add(fp)
                candidates.append((
                    f"{stage_label}#{rank}",
                    params,
                    float(trial.value),
                    trial.number,
                    stage_label,
                ))

        if not candidates:
            print(f"  {c.red}brak kandydatów do E3 — fallback E2/E1 best{c.reset}")
            fallback = self._pick_best_trial(study_e2) or self._pick_best_trial(study_e1)
            if fallback is None:
                raise RuntimeError("Staged tuning: no complete trials in E1/E2")
            pick_params = self._merge_fixed_params(
                cfg_e2 if study_e2 is not None else config,
                dict(fallback.params),
            )
            pick_source = "E2" if study_e2 is not None else "E1"
            pick_metric = float(fallback.value)
            pick_trial_no = fallback.number
            best_pick_score = pick_metric
            pick_rows = {}
        else:
            pick_rows = {}
            best_pick_score = float("-inf")
            pick_source = candidates[0][0]
            pick_params = candidates[0][1]
            pick_metric = candidates[0][2]
            pick_trial_no = candidates[0][3]

            for label, params, stage_m, trial_no, stage_label in candidates:
                color = c.cyan if stage_label == "E1" else c.magenta
                try:
                    scored = self._score_params_on_window(params, tail)
                except Exception as e:
                    print(f"  {c.red}{label} t{trial_no} FAILED{c.reset}: {e}")
                    continue
                score = float(scored["metric"])
                pick_rows[label] = {
                    "tail": scored,
                    "pick_score": score,
                    "stage_metric": stage_m,
                    "trial": trial_no,
                    "stage": stage_label,
                }
                print(
                    f"  {color}{label}{c.reset} t{trial_no}  "
                    f"tail_metric={score:.2f}  stage={stage_m:.2f}  "
                    f"final=${scored.get('final_value'):.1f}  "
                    f"exc={scored.get('excess_return'):+.2f}%  "
                    f"sh={scored.get('sharpe_ratio')}  dd={scored.get('max_drawdown')}"
                )
                if score > best_pick_score:
                    best_pick_score = score
                    pick_source = label
                    pick_params = params
                    pick_metric = stage_m
                    pick_trial_no = trial_no

        stage_meta["pick_source"] = pick_source
        stage_meta["pick_score"] = best_pick_score
        stage_meta["pick_rows"] = pick_rows
        print(
            f"\n  {c.green}{c.bold}▶ FINAL ← {pick_source}{c.reset}  "
            f"tail_metric={best_pick_score:.2f}  stage_metric={pick_metric:.2f}  "
            f"tail={tail}"
        )

        final_study = study_e2 or study_e1
        self.study_name = study_name
        self.results_file = os.path.join(self.log_path, f"{study_name}_results.json")
        self._finalize_study(
            final_study,
            config,
            defaults,
            stage_meta=stage_meta,
            override_best_params=pick_params,
            override_best_metric=pick_metric,
            override_best_trial=pick_trial_no,
        )
        self._stage_label = None
        return final_study

    def _finalize_study(
        self,
        study: optuna.Study,
        config: Dict[str, Any],
        defaults: Dict[str, Any],
        stage_meta: Dict[str, Any] = None,
        override_best_params: Dict[str, Any] = None,
        override_best_metric: float = None,
        override_best_trial: int = None,
    ):
        c = self._color(True)
        self._print_stage_banner("DONE", self.study_name)

        if override_best_params is not None:
            best_params_tuned = override_best_params
            best_metric = override_best_metric
            best_trial_no = override_best_trial
            best_agent = None
        else:
            best = self._pick_best_trial(study) or study.best_trial
            best_params_tuned = dict(best.params)
            best_metric = best.value
            best_trial_no = best.number
            best_agent = best.user_attrs.get("agent_file")
            ua = best.user_attrs
            if ua:
                print(
                    f"  excess={ua.get('excess_return')}  final=${ua.get('final_value')}  "
                    f"valid={ua.get('valid_trial')}  corr={ua.get('hodl_corr')}  "
                    f"tx={ua.get('transaction_count')}  closed={ua.get('closed_trades_count')}"
                )

        best_params_tuned = self._merge_fixed_params(config, best_params_tuned)

        print(f"  {c.bold}Best trial:{c.reset} {best_trial_no}")
        print(f"  {c.bold}Best metric:{c.reset} {best_metric}")
        if stage_meta and stage_meta.get("staged"):
            print(
                f"  {c.bold}Pick:{c.reset} {stage_meta.get('pick_source')}  "
                f"tail={stage_meta.get('e3_window')}  "
                f"top_k={stage_meta.get('pick_top_k')}  "
                f"mode={stage_meta.get('e3_mode')}"
            )
        print(f"\n{c.bold}Best parameters:{c.reset}")
        for key, value in best_params_tuned.items():
            print(f"  {c.blue}{key}{c.reset}= {value},")

        if defaults:
            print(f"\n{c.dim}Static parameters (not tuned):{c.reset}")
            for key, value in defaults.items():
                print(f"  {key}= {value},")

        from rl_module.services.thesis_artifacts import make_optuna_artifact

        window_spec = None
        try:
            window_spec = self._config_window_spec(config) if config else None
        except Exception:
            window_spec = None
        payload = make_optuna_artifact(
            seed=int(getattr(self, "optuna_seed", 42) or 42),
            study_name=str(self.study_name),
            best_params=best_params_tuned,
            best_metric=best_metric,
            best_trial=best_trial_no,
            static_params=defaults or {},
            config=config,
            experiment_version=getattr(self, "experiment_version", None),
            window=window_spec,
            best_agent=best_agent,
            stage_meta=stage_meta or {},
            algorithm=self.agent_type,
        )
        with open(self.best_params_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(
            f"\n{c.green}Best parameters saved to:{c.reset} {self.best_params_file} "
            f"(seed={payload.get('seed')})"
        )
        try:
            self._generate_report(study)
        except Exception as e:
            print(f"{c.yellow}report skipped:{c.reset} {e}")

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function - must be implemented by subclass.
        """
        raise NotImplementedError("Subclass must implement _objective()")

    def _save_trial_result(self, trial, metric, final_value, sharpe, max_dd, agent_path, extra=None):
        """Save individual trial result to JSON"""
        if not getattr(self, "results_file", None):
            return
        result = {
            'algorithm': self.agent_type,
            'trial_number': trial.number,
            'params': trial.params,
            'static_params': self.defaults or {},
            'metric': metric,
            'final_value': final_value,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'agent_file': agent_path,
            'timestamp': datetime.now().isoformat()
        }
        if extra:
            result.update(extra)

        results = []
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                results = json.load(f)

        results.append(result)

        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)

    def _generate_report(self, study):
        """Generate CSV report"""
        trials_df = study.trials_dataframe()

        report_filename = f"{self.study_name}_report.csv"
        report_path = os.path.join(self.log_path, report_filename)
        trials_df.to_csv(report_path, index=False)

        print(f"Full report saved to: {report_path}")

        print("\nTop 5 trials:")
        top_trials = trials_df.nlargest(5, 'value')
        print(top_trials[['number', 'value']])