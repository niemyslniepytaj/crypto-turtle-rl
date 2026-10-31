import os
import json
import gzip
import time
import sys
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from .core_logic.mp_signal import ModelSignal
from .core_logic.decision import DecisionEngine
from .core_logic.windows import load_windows, slice_windows
from .test.rating import Rating
from .visualization_service import VisualizationService
from .core_logic.structures import (
    CoreConfig, coerce_sim_mode, coerce_experiment_pair, normalize_mp_kwargs,
)


class SimulationService:
    """Backtester: sygnał MP, DecisionEngine, zapis wyników i metryk symulacji."""

    def __init__(self, base_dir = None):
        if base_dir is None: base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        self.DATA_DIR = os.path.join(base_dir, "shared", "files", "processed_simulation_data")
        self.SIM_DIR = os.path.join(base_dir, "shared", "simulation","simulations")
        os.makedirs(self.SIM_DIR, exist_ok=True)
        self.OUTPUT_DIR = os.path.join(base_dir, "shared", "simulation", "outputs")
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        self.rating = Rating()
        self.core_config = CoreConfig()
        self.num_ticks = 0
        self.models = {}
        self.symbols = []
        self._models_window_id = None  # okno, dla którego policzono predykcje (cache)
        self.engine = None
        self.output = {}

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
            """Kody ANSI do kolorowej karty wyników w konsoli."""
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

    def _print_results_card(
        self,
        *,
        sim_name,
        sim_mode,
        metrics,
        hodl_return,
        window_id,
        elapsed_s,
        model_paths=None,
        data_files=None,
    ):
        """Kolorowa karta wyników na koniec symulacji (nie w quiet/Optuna)."""
        self._enable_windows_ansi()
        c = self._color(True)
        cfg = self.core_config
        bal0 = float(getattr(cfg, "balance_start", 1000.0) or 1000.0)
        final = float(metrics.get("final_portfolio_value") or bal0)
        ret = float(metrics.get("total_return") or 0.0)
        sharpe = metrics.get("sharpe_ratio")
        sortino = metrics.get("sortino_ratio")
        dd = float(metrics.get("max_drawdown") or 0.0)
        tx = int(metrics.get("transaction_count") or 0)
        closed = int(metrics.get("closed_trades_count") or 0)
        tx_day = metrics.get("avg_transactions_per_day")
        cash = metrics.get("time_in_cash")
        wr = metrics.get("win_rate")
        pf = metrics.get("profit_factor")
        turnover = metrics.get("turnover")
        score = metrics.get("score")
        cagr = metrics.get("cagr")

        hodl = float(hodl_return) if hodl_return is not None else None
        excess = (ret - hodl) if hodl is not None else None
        hodl_final = None
        if hodl is not None:
            hodl_final = bal0 * (1.0 + hodl / 100.0)

        def _signed(v, nd=2, good_positive=True):
            if v is None:
                return "—"
            try:
                x = float(v)
            except (TypeError, ValueError):
                return str(v)
            ok = (x >= 0) if good_positive else (x <= 0)
            col = c.green if ok else c.red
            return f"{col}{x:+.{nd}f}{c.reset}"

        def _f(v, nd=2):
            if v is None:
                return "—"
            try:
                return f"{float(v):.{nd}f}"
            except (TypeError, ValueError):
                return str(v)

        beat = excess is not None and excess > 0
        verdict = (
            f"{c.green}{c.bold}BEAT HODL{c.reset}"
            if beat
            else (
                f"{c.red}{c.bold}vs HODL{c.reset}"
                if excess is not None
                else f"{c.dim}no HODL{c.reset}"
            )
        )

        sl = float(getattr(cfg, "stop_loss", 0.0) or 0.0)
        sl_on = 0.0 < sl < 1.0
        sl_txt = f"{sl:.3f}" if sl_on else f"{c.dim}off({sl:g}){c.reset}"
        symbols = ",".join(getattr(cfg, "symbols", None) or self.symbols or [])
        models = model_paths or {}
        model_txt = ",".join(f"{k}:{v}" for k, v in models.items()) if models else "—"

        print(
            f"\n{c.cyan}{c.bold}{'─' * 66}{c.reset}\n"
            f"{c.cyan}{c.bold}  SIM RESULT{c.reset}  {c.bold}{sim_name}{c.reset}  [{verdict}]\n"
            f"{c.cyan}{c.bold}{'─' * 66}{c.reset}\n"
            f"   {c.dim}setup{c.reset}   mode={c.blue}{sim_mode}{c.reset}  "
            f"ver={c.blue}{getattr(cfg, 'experiment_version', '?')}{c.reset}  "
            f"window={c.blue}{window_id or 'full'}{c.reset}  "
            f"sym={symbols}  ticks={self.num_ticks}  "
            f"t={timedelta(seconds=int(elapsed_s))}\n"
            f"   {c.dim}risk{c.reset}    sl={sl_txt}  "
            f"cd={getattr(cfg, 'stop_loss_cooldown', '—')}  "
            f"fee={'on' if getattr(cfg, 'add_fee', True) else 'off'}  "
            f"minc={_f(getattr(cfg, 'mincost', None), 1)}  "
            f"wlim={_f(getattr(cfg, 'wallet_percentage_limit', None), 3)}  "
            f"maxc={_f(getattr(cfg, 'max_wallet_percentage_crypto', None), 3)}\n"
            f"   {c.dim}alloc{c.reset}   bias={_f(getattr(cfg, 'mp_alloc_bias', None), 3)}  "
            f"scale={_f(getattr(cfg, 'mp_alloc_scale', None), 3)}  "
            f"conf={_f(getattr(cfg, 'mp_confidence', None), 3)}  "
            f"urg={_f(getattr(cfg, 'mp_urgency', None), 3)}  "
            f"sigDZ={_f(getattr(cfg, 'mp_signal_deadzone', None), 3)}  "
            f"rebDZ={_f(getattr(cfg, 'mp_rebalance_deadzone', None), 3)}\n"
            f"   {c.dim}model{c.reset}   {c.dim}{model_txt}{c.reset}\n"
            f"   {c.dim}equity{c.reset}  strat={c.bold}${final:.1f}{c.reset}  "
            f"ret={_signed(ret)}%  "
            + (
                f"hodl={c.bold}${hodl_final:.1f}{c.reset} ({hodl:+.2f}%)  "
                f"excess={_signed(excess)}%  "
                if hodl is not None and hodl_final is not None
                else ""
            )
            + f"cagr={_signed(cagr)}%\n"
            f"   {c.dim}riskΔ{c.reset}   sharpe={_signed(sharpe)}  sortino={_signed(sortino)}  "
            f"dd={_signed(dd, good_positive=False)}%  "
            f"calmar={_f(metrics.get('calmar_ratio'), 3)}\n"
            f"   {c.dim}trade{c.reset}   {c.bold}tx={tx}{c.reset}  "
            f"closed={closed}  "
            f"tx/d={_f(tx_day, 2)}  "
            f"cash={_f(float(cash) * 100 if cash is not None else None, 2)}"
            f"{'%' if cash is not None else ''}  "
            f"WR={_f(float(wr) * 100 if wr is not None else None, 1)}"
            f"{'%' if wr is not None else ''}  "
            f"PF={_f(pf, 2)}  turn={_f(turnover, 2)}\n"
            f"   {c.magenta}{c.bold}score{c.reset}   {c.bold}{score if score is not None else '—'}{c.reset}"
            f"   peak={_signed(metrics.get('max_portfolio_growth'))}%  "
            f"trough={_signed(metrics.get('max_portfolio_drop'), good_positive=False)}%\n"
            f"{c.cyan}{c.bold}{'─' * 66}{c.reset}\n"
        )

    def run_simulation(
        self,
        rabbit_msg_id=None,
        balance_start=1000.0,
        add_fee=True,
        sim_mode="rl",
        rlagent_deterministic=False,
        slippage=0,
        wallet_percentage_limit=0.9,
        max_wallet_percentage_crypto=1.0,
        stop_loss=1.0,
        stop_loss_cooldown=120,
        threshold_buy=0.5,
        threshold_sell=0.5,
        mincost=1.0,
        tx_fraction=0.01,
        max_fraction_when_confident=0.5,
        mp_alloc_bias=0.5,
        mp_alloc_scale=0.5,
        mp_confidence=0.5,
        mp_urgency=0.5,
        mp_signal_deadzone=0.15,
        mp_rebalance_deadzone=0.25,
        mp_ema_alpha_min=0.02,
        mp_ema_alpha_span=0.08,
        v5rl_knob_hold_bars=1,
        v5rl_trade_lock_bars=15,
        clamp_wallet_cap_to_1n=True,
        data_files=None,
        model_paths=None,
        agent_rl=None,
        plots=True,
        logs=True,
        results=True,
        experiment_version="v0",
        window_id=None,
        quiet=False,
        name_tag=None,
        obs_rms_file=None,
        portfolio_state=None,
        gzip_basename=None,
        sim_dir=None,
        **legacy_kwargs,
    ):
        """
        Uruchamia backtest na zadanym oknie i konfiguracji CoreConfig.

        portfolio_state — stan ciągły z poprzedniego okna (balance, holdings, avg/last buy).
        gzip_basename — nazwa pliku w outputs/ (bez .json.gz).
        sim_dir — katalog artefaktów lokalnych; domyślnie SIM_DIR/<basename>.
        """
        try:
            self.output, self.engine,  symbols = {}, None,  []
            self._quiet = bool(quiet)

            _knobs = normalize_mp_kwargs({
                "sim_mode": sim_mode,
                "mp_alloc_bias": mp_alloc_bias,
                "mp_alloc_scale": mp_alloc_scale,
                "mp_confidence": mp_confidence,
                "mp_urgency": mp_urgency,
                "mp_signal_deadzone": mp_signal_deadzone,
                "mp_rebalance_deadzone": mp_rebalance_deadzone,
                "mp_ema_alpha_min": mp_ema_alpha_min,
                "mp_ema_alpha_span": mp_ema_alpha_span,
                **{
                    k: v for k, v in (legacy_kwargs or {}).items()
                    if str(k).startswith("ml_") or k in ("sim_mode",)
                },
            })
            sim_mode = coerce_sim_mode(_knobs.get("sim_mode", sim_mode))
            experiment_version, sim_mode = coerce_experiment_pair(
                experiment_version, sim_mode,
            )
            mp_alloc_bias = _knobs["mp_alloc_bias"]
            mp_alloc_scale = _knobs["mp_alloc_scale"]
            mp_confidence = _knobs["mp_confidence"]
            mp_urgency = _knobs["mp_urgency"]
            mp_signal_deadzone = _knobs["mp_signal_deadzone"]
            mp_rebalance_deadzone = _knobs["mp_rebalance_deadzone"]
            mp_ema_alpha_min = _knobs["mp_ema_alpha_min"]
            mp_ema_alpha_span = _knobs["mp_ema_alpha_span"]

            if sim_mode not in ("mp", "rl", "rlul"):
                raise ValueError("[SIMULATION ERROR] Invalid simulation mode")
            # rlul: meta-knoby jak rl + kanał reżimu (nie spłaszczaj do rl)
            if not self._quiet:
                print("[Simulation] Start")
            start_time = time.time()
            data_files = data_files or {}
            model_paths = model_paths or {}
            windows = load_windows(self.base_dir) if window_id else None

            # --- Ładowanie danych / predykcje (raz na okno; przy zmianie window_id reload) ---
            cache_key = str(window_id) if window_id is not None else None
            if self.models and self._models_window_id != cache_key:
                self.models = {}
                self.num_ticks = 0
                self.symbols = []

            if self.models == {}:
                for symbol, filename in data_files.items():
                    filepath = os.path.join(self.DATA_DIR, filename)
                    if not os.path.exists(filepath):
                        raise FileNotFoundError(f"Brak pliku: {filepath}")
                    df = pd.read_csv(filepath)
                    df.dropna(inplace=True)
                    if window_id:
                        df = slice_windows(df, windows, window_id)
                        if not self._quiet:
                            print(f"[Simulation] window={window_id} {symbol}: {len(df)} rows")
                        if len(df) == 0:
                            raise ValueError(f"[SIMULATION ERROR] Puste okno {window_id} dla {symbol}")
                    len_df = len(df)
                    if self.num_ticks == 0: self.num_ticks = len_df
                    if len_df < self.num_ticks: self.num_ticks = len_df

                    # --- Inicjalizacja modeli ---
                    self.models[symbol] = ModelSignal(model_name=model_paths[symbol], data=df.copy(),base_dir=self.base_dir)
                    symbols.append(symbol)

                    len_pred = len(self.models[symbol].predictions)

                    if len_pred < self.num_ticks: self.num_ticks = len_pred

                for symbol in symbols: self.models[symbol].process_models_data(self.num_ticks)
                self.symbols = symbols
                self._models_window_id = cache_key
                print(
                    f"[Optuna] Predykcje cache (window={cache_key}, ticks={self.num_ticks})"
                    if self._quiet
                    else f"[Simulation] Predykcje policzone i cache'owane (window={cache_key})"
                )

            self.core_config = CoreConfig(
                sim_mode=sim_mode,
                balance_start=balance_start,
                add_fee=add_fee,
                symbols=self.symbols,
                rlagent=sim_mode,
                rlagent_deterministic=rlagent_deterministic,
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
                v5rl_knob_hold_bars=int(v5rl_knob_hold_bars if v5rl_knob_hold_bars is not None else 1),
                v5rl_trade_lock_bars=int(v5rl_trade_lock_bars if v5rl_trade_lock_bars is not None else 15),
                clamp_wallet_cap_to_1n=bool(clamp_wallet_cap_to_1n),
                experiment_version=experiment_version
            )

            # --- Inicjalizacja Portfela i Decision Engine---
            # mp → sim_mp (mp_config); rl → sim_rl (meta-knoby + agent)
            _core_mode = "sim_mp" if sim_mode == "mp" else "sim_rl"
            self.engine = DecisionEngine(
                agent_name=agent_rl,
                models=self.models,
                base_dir=self.base_dir,
                core_mode=_core_mode,
                core_config=self.core_config,
            )
            if not obs_rms_file and agent_rl:
                try:
                    from rl_module.services.rl.obs_rms_utils import resolve_obs_rms_beside_agent
                    agents_dir = os.path.join(self.base_dir, "shared", "agents", "ppo")
                    obs_rms_file = resolve_obs_rms_beside_agent(agent_rl, agents_dir=agents_dir)
                except Exception:
                    obs_rms_file = None
            if obs_rms_file:
                self.engine.load_obs_rms(obs_rms_file)
            # kanał reżimu: K z agenta (meta / obs_space − market), nie z locka
            if sim_mode in ("rl", "rlul", "mprl"):
                try:
                    rk_kw = legacy_kwargs.pop("regime_k", None)
                    series = legacy_kwargs.pop("regime_onehot_series", None)
                    onehot = legacy_kwargs.pop("regime_onehot", None)
                    if rk_kw is not None:
                        rk = int(rk_kw)
                    else:
                        rk = int(self.engine.infer_regime_k_from_agent())
                    self.engine.set_regime_channel(k=rk, onehot_series=series, onehot=onehot)
                    if not self._quiet:
                        print(f"[Simulation] regime_k={rk} (z agenta/kwargs)")
                except Exception as _reg_err:
                    if not self._quiet:
                        print(f"[Simulation] regime channel skip: {_reg_err}")
                    try:
                        self.engine.set_regime_channel(k=0)
                    except Exception:
                        pass

            if portfolio_state:
                self._apply_portfolio_state(portfolio_state)

            # --- Symulacja ---
            match sim_mode:
                case "mp":
                    sim_mode = "model"
                    self.simulation("model")
                case "rl" | "rlul" | "mprl":
                    sim_mode = "agent"
                    self.engine.core_mode = "sim_rl"
                    self.simulation("agent")

            if not self._quiet:
                print(f"[Simulation] Zakończono pętlę symulacyjną w {timedelta(seconds=time.time() - start_time)}")
            # --- Zapis wyników ---

            self.output["files"] = {
                "models": model_paths,
                "agents": agent_rl,
                "data": data_files
            }

            self.output["config"] = self.core_config.to_dict()
            if window_id is not None:
                self.output["config"]["window_id"] = str(window_id)

            #TODO: mode simulation

            gzip_name, sim_path, gzip_path = self.build_filename(
                name_tag=name_tag,
                gzip_basename=gzip_basename,
                sim_dir=sim_dir,
                window_id=window_id,
            )

            if logs or plots or results:
                if not self._quiet:
                    print(f"[Simulation] Zapisywanie do plików")
                os.makedirs(sim_path, exist_ok=True)
                os.makedirs(self.OUTPUT_DIR, exist_ok=True)

            if logs:
                self.save_logs(sim_mode, sim_path)
            # gzip potrzebny do viewer + wykresów (VisualizationService czyta z outputs/)
            if results or plots:
                self.save_results(sim_mode, sim_path)
                with gzip.open(gzip_path, "wt", encoding="utf-8") as f:
                    json.dump(self.output, f, indent=2)

            chart_path = None
            if plots:
                chart_path = VisualizationService().visualize(
                    simulation_files=gzip_name,
                    sim_path=sim_path,
                )
                if not self._quiet and chart_path:
                    print(f"[Simulation] Wykres: {chart_path}")

            metrics = self.output[sim_mode]["metrics"]
            hodl_return = None
            hodl_block = self.output.get("hodl") or {}
            hodl_refs = hodl_block.get("crypto_ref_values") or {}
            if hodl_refs:
                # equal-weight portfel (= średnia returnów per symbol)
                from rl_module.services.thesis.assets import hodl_return_pct_from_refs
                hodl_return = hodl_return_pct_from_refs(hodl_refs)

            if not self._quiet:
                try:
                    self._print_results_card(
                        sim_name=os.path.splitext(os.path.splitext(gzip_name)[0])[0],
                        sim_mode=sim_mode,
                        metrics=metrics,
                        hodl_return=hodl_return,
                        window_id=window_id,
                        elapsed_s=time.time() - start_time,
                        model_paths=model_paths,
                        data_files=data_files,
                    )
                except UnicodeEncodeError:
                    # Windows cp1250 konsola — wynik symulacji i tak jest gotowy
                    pass

            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "success",
                "message": "Successfully retrieved simulation data",
                #"model_logs_file": model_log_path,
                #"agent_logs_file": agent_log_path,
                "metrics": metrics,
                "final_portfolio_value": metrics["final_portfolio_value"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "transaction_count": metrics.get("transaction_count", 0),
                "turnover": metrics.get("turnover"),
                "closed_trades_count": metrics.get("closed_trades_count"),
                "avg_transactions_per_day": metrics.get("avg_transactions_per_day"),
                "time_in_cash": metrics.get("time_in_cash"),
                "n_ticks": self.num_ticks,
                "hodl_return": hodl_return,
                "portfolio_values": self.output[sim_mode].get("portfolio_values"),
                "hodl_crypto_ref_values": hodl_refs,
                "portfolio_state": self._capture_portfolio_state(),
                "gzip_name": gzip_name,
                "gzip_path": gzip_path,
                "sim_path": sim_path,
                "chart_path": chart_path,
                "sim_output": self.output,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            import traceback
            print("[Simulation][ERROR]", e)
            traceback.print_exc()
            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "failed",
                "message": str(e),
                "logs_file": None,
                "plot_file": None,
                "portfolio_values": [],
                "timestamp": datetime.now().isoformat()
            }


    def simulation(self, mode):
        quiet = bool(getattr(self, "_quiet", False))
        if not quiet:
            print("[Simulation] Liczba timestampów: ", self.num_ticks)

        tick_iter = range(self.num_ticks)
        if not quiet:
            tick_iter = tqdm(tick_iter, file=sys.stdout, dynamic_ncols=True, leave=False)
        for t in tick_iter:
            self.engine.generate_decisions(t)

        crypto_ref_values = {}

        for symbol in self.core_config.symbols:
            # Mnożenie wektora przez skalar -> super szybkie w numpy
            ref_vector = (self.models[symbol].close_prices / self.models[symbol].close_prices[0]) * self.core_config.balance_start
            crypto_ref_values[symbol] = ref_vector.tolist()

        from rl_module.services.thesis.assets import equal_weight_hodl_curve
        hodl_portfolio = equal_weight_hodl_curve(crypto_ref_values)

        metrics = self.rating.rate(
            self.engine.wallet.portfolio_values_history,
            self.engine.wallet.transaction_count,
            closed_trades=self.engine.wallet.closed_trades,
            holdings_history=self.engine.wallet.portfolio_holdings_history,
            balance_start=self.core_config.balance_start,
        )

        # quiet (Optuna): nie serializuj setek tys. orderów
        if quiet:
            self.engine.wallet.logs = []
        else:
            self.engine.wallet.logs = [order.to_dict() for order in self.engine.wallet.logs]
        self.output["hodl"] = {
            "crypto_ref_values": crypto_ref_values,
            "portfolio_values": hodl_portfolio,
        }

        self.output[mode] = {
            "timestamps": self.num_ticks,
            "metrics": metrics,
            "portfolio_values": self.engine.wallet.portfolio_values_history,
            "holdings": self.engine.wallet.portfolio_holdings_history
        }
        return

    def save_logs(self, mode, sim_path):
        log_path = os.path.join(sim_path, f"{mode}_transaction_logs.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            for line in self.engine.wallet.logs:
                f.write(json.dumps(line) + "\n")


    def save_results(self, mode, sim_path):
        results_path = os.path.join(sim_path, "sim_results.txt")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(self.output[mode]["metrics"], f, indent=2)

    def _apply_portfolio_state(self, state: dict):
        """Przywróć cash/holdings z poprzedniego okna (portfolio_mode=continuous)."""
        if not state or not self.engine or not getattr(self.engine, "wallet", None):
            return
        w = self.engine.wallet
        if state.get("balance") is not None:
            w.balance = float(state["balance"])
        for sym, qty in (state.get("holdings") or {}).items():
            if sym in w.holdings:
                w.holdings[sym] = float(qty)
        for sym, px in (state.get("avg_buy_price") or {}).items():
            if sym in w.avg_buy_price:
                w.avg_buy_price[sym] = float(px)
        for sym, px in (state.get("last_buy_price") or {}).items():
            if sym in w.last_buy_price:
                w.last_buy_price[sym] = float(px)

    def _capture_portfolio_state(self) -> dict:
        if not self.engine or not getattr(self.engine, "wallet", None):
            return {}
        w = self.engine.wallet
        return {
            "balance": float(w.balance),
            "holdings": {k: float(v) for k, v in (w.holdings or {}).items()},
            "avg_buy_price": {k: float(v) for k, v in (w.avg_buy_price or {}).items()},
            "last_buy_price": {k: float(v) for k, v in (w.last_buy_price or {}).items()},
        }

    def build_filename(self, name_tag=None, gzip_basename=None, sim_dir=None, window_id=None):
        # Thesis / sequence: nazwa już zawiera okno (np. seq_R1_..._A1)
        if gzip_basename:
            safe = "".join(
                ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(gzip_basename)
            )
            g_n = f"{safe}.json.gz"
            sim_path = sim_dir or os.path.join(self.SIM_DIR, safe)
            gzip_path = os.path.join(self.OUTPUT_DIR, g_n)
            return g_n, sim_path, gzip_path

        date_str = datetime.now().strftime("%d_%m")
        existing_files = [
            f for f in os.listdir(self.OUTPUT_DIR)
            if f.startswith("sim_") and f"_{date_str}_" in f and f.endswith(".gz")
        ]

        version_numbers = []
        for f in existing_files:
            try:
                base = os.path.splitext(os.path.splitext(f)[0])[0]
                v_part = base.split("_v")[-1].split("_")[0]
                version_numbers.append(int(v_part))
            except ValueError:
                continue

        next_version = max(version_numbers, default=0) + 1
        tag = ""
        if name_tag:
            # bezpieczny dopisek do nazwy (np. rltrain_es42)
            safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(name_tag))
            tag = f"_{safe}"
        win = ""
        if window_id is not None and str(window_id).strip():
            safe_w = "".join(
                ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(window_id).strip()
            )
            # nie dubluj, jeśli name_tag już kończy się na _A1 / A1
            if safe_w and not tag.endswith(f"_{safe_w}"):
                win = f"_{safe_w}"
        mode = getattr(self.core_config, "sim_mode", None) or "mp"
        s_p = f"sim_{date_str}_v{next_version}_{mode}{tag}{win}"
        g_n = f"{s_p}.json.gz"
        sim_path = sim_dir or os.path.join(self.SIM_DIR, s_p)
        gzip_path = os.path.join(self.OUTPUT_DIR, g_n)
        return g_n, sim_path, gzip_path