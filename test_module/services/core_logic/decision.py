import os
import numpy as np
import types
from stable_baselines3 import PPO, SAC
from .version_registry import EXPERIMENTS
from .structures import (
    Order, DecisionConfig, CoreConfig, WalletConfig,
    coerce_experiment_pair, coerce_sim_mode, is_meta_knob_mode,
)
from .wallet import Wallet

class DecisionEngine:
    """Silnik decyzji handlowych: obserwacja, sygnał MP lub agent RL oraz powiązanie z Wallet."""

    def __init__(self, base_dir, core_mode="sim_mp", models=None, agent_name=None, core_config: CoreConfig = None):
        self.BASE_MODEL_DIR = os.path.join(base_dir, "shared", "agents")
        self.agent_name = agent_name
        self.agent_path = ""
        self.core_mode = core_mode
        self.core_config = core_config

        self.symbols = self.core_config.symbols
        self.config = {}
        for symbol in self.symbols:
            self.config[symbol] = DecisionConfig()
            if core_mode in ("sim_rl", "sim_mp", "sim_ml", "train_rl"):
                self.config[symbol].set(core_config)

        self.models = models

        self.current_prices = {}
        self.current_probs = {}
        self.current_value = 0
        self.prev_value = 0
        ver, mode = coerce_experiment_pair(
            core_config.experiment_version, getattr(core_config, "sim_mode", None),
        )
        core_config.experiment_version = ver
        core_config.sim_mode = mode or core_config.sim_mode
        self.experiment_version = ver
        self.n_params_per_symbol = 0
        self.wallet = Wallet(self.core_mode, core_config)

        self.rlagent_deterministic = core_config.rlagent_deterministic
        self.agent = None
        self._obs_rms = None  # opcjonalna normalizacja obs (z VecNormalize treningu)
        self.BUY_IDX = 0  #TODO: np. pobrane z class_map
        self.HOLD_IDX = 1
        self.SELL_IDX = 2
        if self.agent_name:
            agent_type = self.check_agent_type()
            match agent_type:
                case "ppo":
                    self.agent = PPO.load(self.agent_path, env=None)

                case "sac":
                    self.agent = SAC.load(self.agent_path, env=None)
                case _:
                    raise ValueError(f"Unsupported agent type: {agent_type}")
        self.apply_experiment_version()
        self._v5rl_trade_lock_until = {}
        # sim: hold/lock z CoreConfig (train ustawia to w env.reset)
        if is_meta_knob_mode(self.experiment_version, getattr(core_config, "sim_mode", None)):
            self._v5rl_hold_bars = int(
                getattr(core_config, "v5rl_knob_hold_bars", 1) if core_config else 1
            )
            self._v5rl_trade_lock_bars = int(
                getattr(core_config, "v5rl_trade_lock_bars", 15) if core_config else 15
            )

    def init_local_configs(self, symbol):
        self.config[symbol] = DecisionConfig()
        self.config[symbol].set(self.core_config)
        self.wallet.config[symbol] = WalletConfig()
        self.wallet.config[symbol].set(self.core_config)

    def reset(self):
        """Przywraca konfiguracje per symbol, portfel oraz stan epizodu do wartości początkowych."""
        self.config = {}
        for symbol in self.symbols:
            self.config[symbol] = DecisionConfig()
            if self.core_mode in ("sim_rl", "sim_mp", "sim_ml"):
                self.config[symbol].set(self.core_config)
        self.prev_value = self.core_config.balance_start
        self.current_value = self.core_config.balance_start
        self._prev_target = {}
        self._mp_prev_target_raw = {}
        self._sl_cooldown_until = {}
        self._v5rl_trade_lock_until = {}
        self._peak_portfolio = float(self.core_config.balance_start)
        self._t = 0

    def _in_sl_cooldown(self, symbol, t) -> bool:
        until = getattr(self, "_sl_cooldown_until", {}).get(symbol, -1)
        return t < until

    def _in_trade_lock(self, symbol, t) -> bool:
        """v6/mprl (ex-v5rl): po accepted buy/sell blokuj kolejne tx (SL omija)."""
        if not is_meta_knob_mode(
            self.experiment_version, getattr(self.core_config, "sim_mode", None)
        ):
            return False
        until = getattr(self, "_v5rl_trade_lock_until", {}).get(symbol, -1)
        return t < until

    def _trade_lock_bars(self) -> int:
        bars = getattr(self, "_v5rl_trade_lock_bars", None)
        if bars is None:
            bars = getattr(self.core_config, "v5rl_trade_lock_bars", None)
        if bars is None:
            from .version_registry import V5RL_TRADE_LOCK_BARS
            bars = V5RL_TRADE_LOCK_BARS
        return int(bars)

    def _arm_trade_lock_from_executed(self, t):
        """Po egzekucji: accepted non-SL → lock portfela na N barów."""
        lock_bars = self._trade_lock_bars()
        if lock_bars <= 0 or not is_meta_knob_mode(
            self.experiment_version, getattr(self.core_config, "sim_mode", None)
        ):
            return
        if not hasattr(self, "_v5rl_trade_lock_until"):
            self._v5rl_trade_lock_until = {}
        slip = int(getattr(self.core_config, "slippage", 0) or 0)
        for order in self.wallet.orders_by_t.get(t - slip, []):
            if getattr(order, "status", "") != "accepted":
                continue
            if getattr(order, "force_exit", False) or getattr(order, "reason", "") == "stop_loss":
                continue
            if order.action not in ("buy", "sell"):
                continue
            self._v5rl_trade_lock_until[order.symbol] = int(t) + lock_bars

    def _build_stop_loss_orders(self, t):
        """
        v24 risk exits:
          1) SL pozycji od avg_buy
          2) SL portfela od peak equity (ten sam próg stop_loss)
        Po wyjściu: targety=0, cooldown bez buy, peak reset po egzekucji.
        """
        sl = float(getattr(self.core_config, "stop_loss", 0.0) or 0.0)
        if not Wallet.stop_loss_enabled(sl):
            return [], set()

        if not hasattr(self, "_sl_cooldown_until"):
            self._sl_cooldown_until = {}
        if not hasattr(self, "_peak_portfolio"):
            self._peak_portfolio = float(self.core_config.balance_start)

        cooldown = int(getattr(self.core_config, "stop_loss_cooldown", 120) or 120)
        pv = float(self.wallet.total_value(self.current_prices))
        self._peak_portfolio = max(float(self._peak_portfolio), pv)

        risk_symbols = set()
        peak = float(self._peak_portfolio)
        if peak > 0.0 and (peak - pv) / peak >= sl:
            for symbol in self.symbols:
                if float(self.wallet.holdings.get(symbol, 0.0) or 0.0) > 1e-12:
                    risk_symbols.add(symbol)

        for symbol in self.symbols:
            price = float(self.current_prices.get(symbol, 0.0) or 0.0)
            if price > 0.0 and self.wallet.check_stop_loss(symbol, price):
                risk_symbols.add(symbol)

        orders = []
        for symbol in risk_symbols:
            orders.append(Order(
                timestamp=t,
                symbol=symbol,
                action="sell",
                value=1.0,
                buy_p=0.0,
                hold_p=0.0,
                sell_p=1.0,
                price=float(self.current_prices[symbol]),
                rl_config="stop_loss",
                fraction=1.0,
                force_exit=True,
                reason="stop_loss",
            ))
            if hasattr(self, "_prev_target"):
                self._prev_target[symbol] = 0.0
            if hasattr(self, "_mp_prev_target_raw"):
                self._mp_prev_target_raw[symbol] = 0.0
            self._sl_cooldown_until[symbol] = int(t) + max(0, cooldown)
            # SL przebija trade-lock — nie trzymaj starego locka po flatten
            if hasattr(self, "_v5rl_trade_lock_until"):
                self._v5rl_trade_lock_until[symbol] = -1

        return orders, risk_symbols

    def apply_experiment_version(self):
        """Przypina metody decyzji, obserwacji i nagrody z rejestru EXPERIMENTS."""

        ver, mode = coerce_experiment_pair(
            self.experiment_version,
            getattr(self.core_config, "sim_mode", None),
        )
        self.experiment_version = ver
        if self.core_config is not None:
            self.core_config.experiment_version = ver
            if mode:
                self.core_config.sim_mode = mode

        if self.experiment_version not in EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment: {self.experiment_version}. "
                f"Available: {list(EXPERIMENTS.keys())}"
            )

        exp = EXPERIMENTS[self.experiment_version]
        mode = coerce_sim_mode(getattr(self.core_config, "sim_mode", None)) or "mp"
        overlay = (exp.get("by_sim_mode") or {}).get(mode) or {}

        # Override n_params_per_symbol (v6: zależy od sim_mode)
        self.n_params_per_symbol = int(
            overlay.get("n_params_per_symbol", exp["n_params_per_symbol"])
        )
        self.obs_len_global = int(
            overlay.get("obs_len_global", exp.get("obs_len_global", 0) or 0)
        )
        self.obs_len_for_symbol = int(
            overlay.get("obs_len_for_symbol", exp.get("obs_len_for_symbol", 0) or 0)
        )

        # Bind Decision methods
        for method_name, method_func in exp["decision_methods"].items():
            bound_method = types.MethodType(method_func, self)
            setattr(self, method_name, bound_method)

        # Legacy alias: ml_config → mp_config
        if hasattr(self, "mp_config") and not hasattr(self, "ml_config"):
            self.ml_config = self.mp_config

        # Bind Wallet methods
        for method_name, method_func in exp["wallet_methods"].items():
            bound_method = types.MethodType(method_func, self.wallet)
            setattr(self.wallet, method_name, bound_method)

    # def _normalize_model_value(self, mv):
    #     if mv is None:
    #         return 0.0
    #     mv = float(mv)
    #     if self.config["ETHUSDT"].clamp_model_value:
    #         mv = max(0.0, min(1.0, mv))
    #     return mv

    def check_agent_type(self):
        for subfolder in os.listdir(self.BASE_MODEL_DIR):
            subfolder_path = os.path.join(self.BASE_MODEL_DIR, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            candidate_path = os.path.join(subfolder_path, self.agent_name)
            if os.path.exists(candidate_path):
                self.agent_path = candidate_path
                return subfolder.lower()

        raise ValueError(f"Nie znaleziono agenta '{self.agent_name}' w żadnym z podfolderów {os.listdir(self.BASE_MODEL_DIR)}")

    def rl_config(self, scaled_dict):
        for symbol in self.symbols:
            scaled = scaled_dict[symbol]
            thr_low, thr_high = 0.25, 0.85
            threshold_buy = thr_low + scaled[0] * (thr_high - thr_low)
            threshold_sell = thr_low + scaled[1] * (thr_high - thr_low)

            tx_min, tx_max = 0.01, 1.0
            tx_fraction = tx_min + scaled[2] * (tx_max - tx_min)

            tx_min2, tx_max2 = 0.45, 1.0
            tx_max_fraction = tx_min2 + scaled[3] * (tx_max2 - tx_min2)

            self.config[symbol].threshold_buy = float(round(threshold_buy,2))
            self.config[symbol].threshold_sell = float(round(threshold_sell,2))
            self.config[symbol].tx_fraction = float(round(tx_fraction,2))
            self.config[symbol].max_fraction_when_confident = float(round(tx_max_fraction,2))

    def t_data(self, t):
        self._t = int(t)
        for symbol in self.symbols:
            self.current_probs[symbol] = self.models[symbol].predictions[t]
            self.current_prices[symbol] = self.models[symbol].close_prices[t]

        self.prev_value = self.current_value
        self.current_value = self.wallet.total_value(self.current_prices)
        # peak equity do soft risk-off (nie mylić z peak pod hard SL)
        if not hasattr(self, "_peak_portfolio"):
            self._peak_portfolio = float(self.core_config.balance_start)
        self._peak_portfolio = max(float(self._peak_portfolio), float(self.current_value))

    def generate_decisions(self, t, scaled = None):
        if self.core_mode in ("sim_rl", "sim_mp", "sim_ml"):
            self.t_data(t)

        if self.core_mode == "sim_rl":
            scaled = self.get_obs(t)
        elif self.core_mode in ("sim_mp", "sim_ml") and (
            hasattr(self, "mp_config") or hasattr(self, "ml_config")
        ):
            # v5mp: sygnał MP -> te same 3 parametry alokacji co PPO
            cfg_fn = getattr(self, "mp_config", None) or getattr(self, "ml_config", None)
            scaled = cfg_fn()
        elif self.core_mode == "train_rl":
            pass

        # cooldown: trzymaj target=0 zanim rl_config zaktualizuje EMA / frakcje
        # mprl: scaled[0] to bias mapowania, nie position_target — nie zeruj akcji
        meta_knobs = is_meta_knob_mode(
            self.experiment_version, getattr(self.core_config, "sim_mode", None)
        )
        for symbol in self.symbols:
            if not self._in_sl_cooldown(symbol, t):
                continue
            if hasattr(self, "_mp_prev_target_raw"):
                self._mp_prev_target_raw[symbol] = 0.0
            if hasattr(self, "_prev_target"):
                self._prev_target[symbol] = 0.0
            if meta_knobs:
                continue
            if scaled is not None and symbol in scaled:
                s = list(scaled[symbol])
                if len(s) >= 1:
                    s[0] = 0.0
                scaled[symbol] = s

        if self.core_mode in ("sim_rl", "sim_mp", "sim_ml") and scaled is not None:
            self.rl_config(scaled)
        elif self.core_mode == "train_rl":
            self.rl_config(scaled)

        risk_orders, risk_symbols = self._build_stop_loss_orders(t)
        candidates = list(risk_orders)

        for symbol in self.symbols:
            if symbol in risk_symbols:
                continue  # już mamy force sell ze stop-loss

            # Pobieramy prawdopodobieństwa z modelu ML
            probs = self.current_probs[symbol]
            buy_prob = probs[self.BUY_IDX]
            sell_prob = probs[self.SELL_IDX]


            buy_diff_prob = buy_prob - self.config[symbol].threshold_buy
            sell_diff_prob = sell_prob - self.config[symbol].threshold_sell

            if sell_diff_prob >= buy_diff_prob and sell_prob >= self.config[symbol].threshold_sell:
                action = "sell"
                value = sell_prob
            elif buy_diff_prob > sell_diff_prob and buy_prob >= self.config[symbol].threshold_buy:
                action = "buy"
                value = buy_prob
            else:
                action = "hold"
                value = 0.0
                fraction = 0.0

            # po SL — bez dokupywania przez cooldown (cash w DD → niższy hodl_corr)
            if action == "buy" and self._in_sl_cooldown(symbol, t):
                action = "hold"
                value = 0.0

            # v5rl: decyzja co 1m, ale accepted tx blokuje portfel (oprócz SL)
            trade_locked = False
            if action in ("buy", "sell") and self._in_trade_lock(symbol, t):
                action = "hold"
                value = 0.0
                trade_locked = True

            order = Order(
                timestamp = t,
                symbol = symbol,
                action = action,
                value  = round(value,4),
                buy_p  = round(buy_prob,4),
                hold_p = round(probs[self.HOLD_IDX],4) if self.HOLD_IDX else 0.0,
                sell_p = round(sell_prob,4),
                price  = self.current_prices[symbol],
                rl_config = scaled[symbol] if scaled is not None else "-",
            )
            if order.action == "hold":
                if trade_locked:
                    reason = "trade_lock"
                elif self._in_sl_cooldown(symbol, t):
                    reason = "stop_loss_cooldown"
                else:
                    reason = "lower_than_thresholds"
                order.rejected(reason)
            else:
                order = self.calculate_fraction(order)
            candidates.append(order)

        sells = [o for o in candidates if o.action == 'sell']
        buys =  [o for o in candidates if o.action == 'buy']
        holds = [o for o in candidates if o.action == 'hold']

        # Sortujemy wewnątrz grup malejąco po pewności (najpierw najpewniejsze ruchy)
        sells.sort(key=lambda x: x.value, reverse=True)
        buys.sort(key=lambda x: x.value, reverse=True)

        final_orders = sells + buys + holds
        self.wallet.orders.extend(final_orders)
        for order in final_orders:
            self.wallet.orders_by_t[order.timestamp].append(order)  # dodaj też tu
        self.wallet.execute_orders(t, self.current_prices)
        self._arm_trade_lock_from_executed(t)

        # po flatten: peak = bieżący equity (żeby SL portfela nie wisiał w nieskończoność)
        if risk_orders:
            self._peak_portfolio = float(self.wallet.total_value(self.current_prices))

    def calculate_fraction(self, order):
        symbol = order.symbol
        if order.action == "buy":
              threshold = self.config[symbol].threshold_buy
        else: threshold = self.config[symbol].threshold_sell

        denom = max(1.0 - threshold, 1e-12)
        norm_signal = max(0.0, min(1.0, (order.value - threshold) / denom))

        frac = self.config[symbol].tx_fraction + norm_signal * (self.config[symbol].max_fraction_when_confident - self.config[symbol].tx_fraction)
        frac = max(0.0, min(1.0, frac))
        order.fraction = round(frac,4)

        return order

    def load_obs_rms(self, filename):
        """Ładuje mean/var z treningowego VecNormalize (plik .npz w katalogu agenta)."""
        path = filename
        if not os.path.isabs(path):
            # szukaj obok załadowanego agenta / w shared/agents/*
            candidates = []
            if self.agent_path:
                candidates.append(os.path.join(os.path.dirname(self.agent_path), filename))
            candidates.append(os.path.join(self.BASE_MODEL_DIR, filename))
            for sub in os.listdir(self.BASE_MODEL_DIR) if os.path.isdir(self.BASE_MODEL_DIR) else []:
                candidates.append(os.path.join(self.BASE_MODEL_DIR, sub, filename))
            path = next((p for p in candidates if os.path.isfile(p)), filename)
        data = np.load(path)
        self._obs_rms = {
            "mean": np.asarray(data["mean"], dtype=np.float64),
            "var": np.asarray(data["var"], dtype=np.float64),
            "clip_obs": float(data["clip_obs"]) if "clip_obs" in data else 10.0,
            "epsilon": float(data["epsilon"]) if "epsilon" in data else 1e-8,
        }

    def _normalize_obs(self, obs):
        if not self._obs_rms:
            return obs
        x = np.asarray(obs, dtype=np.float64)
        mean = self._obs_rms["mean"]
        var = self._obs_rms["var"]
        # dopasuj długość gdy mismatch (tylko część rynkowa — reżim doklejany potem)
        n = min(x.shape[0], mean.shape[0])
        x = x.copy()
        x[:n] = (x[:n] - mean[:n]) / np.sqrt(var[:n] + self._obs_rms["epsilon"])
        clip = self._obs_rms["clip_obs"]
        x[:n] = np.clip(x[:n], -clip, clip)
        return x.astype(np.float32)

    def market_obs_dim(self) -> int:
        """Długość obserwacji rynkowej (bez kanału reżimu) z overlay eksperymentu."""
        g = int(getattr(self, "obs_len_global", 0) or 0)
        per = int(getattr(self, "obs_len_for_symbol", 0) or 0)
        n = max(len(getattr(self, "symbols", []) or []), 1)
        if g <= 0 and per <= 0:
            return 0
        return int(g + per * n)

    def infer_regime_k_from_agent(self) -> int:
        """
        K z metadanych agenta albo observation_space.shape − market_obs_dim.
        Bez locka / REGIME_K_DEFAULT. Brak kanału → 0.
        """
        # 1) sidecar JSON (train_meta.regime_k)
        try:
            from rl_module.services.thesis_artifacts import load_agent_artifact
            base = os.path.dirname(os.path.dirname(self.BASE_MODEL_DIR))
            agents_dir = os.path.join(base, "shared", "agents", "ppo")
            meta = load_agent_artifact(
                self.agent_name or self.agent_path or "",
                base_dir=base,
                agents_dir=agents_dir,
            )
            if isinstance(meta, dict):
                if meta.get("regime_k") is not None:
                    return max(0, int(meta["regime_k"]))
                tm = meta.get("train_meta") or {}
                if isinstance(tm, dict) and tm.get("regime_k") is not None:
                    return max(0, int(tm["regime_k"]))
        except Exception:
            pass
        # 2) observation_space agenta
        if self.agent is not None and getattr(self.agent, "observation_space", None) is not None:
            shape = getattr(self.agent.observation_space, "shape", None) or ()
            agent_dim = int(shape[0]) if len(shape) else 0
            market = self.market_obs_dim()
            if market > 0 and agent_dim >= market:
                return int(agent_dim - market)
        return 0

    def set_regime_channel(self, *, k: int = 0, onehot_series=None, onehot=None):
        """
        Kanał reżimu PO normalizacji (I2 wariant B).
        onehot_series: (T, K); onehot: (K,) stały; k=0 → brak doklejania.
        """
        self._regime_k = int(k or 0)
        self._regime_onehot_series = None if onehot_series is None else np.asarray(onehot_series, dtype=np.float32)
        self._regime_onehot = None if onehot is None else np.asarray(onehot, dtype=np.float32).ravel()

    def _append_regime(self, obs):
        k = int(getattr(self, "_regime_k", 0) or 0)
        if k <= 0:
            return obs
        series = getattr(self, "_regime_onehot_series", None)
        if series is not None and len(series):
            ti = int(getattr(self, "t", 0) or 0)
            ti = min(max(ti, 0), len(series) - 1)
            vec = series[ti]
        elif getattr(self, "_regime_onehot", None) is not None:
            vec = self._regime_onehot
        else:
            vec = np.zeros(k, dtype=np.float32)
        vec = np.asarray(vec, dtype=np.float32).ravel()
        if vec.shape[0] != k:
            out = np.zeros(k, dtype=np.float32)
            m = min(k, vec.shape[0])
            out[:m] = vec[:m]
            vec = out
        return np.concatenate([np.asarray(obs, dtype=np.float32).ravel(), vec])

    def get_obs(self, t):
        """Predykcja agenta RL na znormalizowanej obserwacji; zwraca akcje w [0, 1] per symbol."""
        scaled_dict = {}

        obs = self.build_obs(t)
        obs = self._normalize_obs(obs)
        obs = self._append_regime(obs)
        action, _ = self.agent.predict(obs, deterministic=self.rlagent_deterministic)
        action = np.asarray(action, dtype=np.float32).flatten()

        scaled = (action + 1.0) / 2.0

        for i, symbol in enumerate(self.symbols):
            start = i * self.n_params_per_symbol
            end = start + self.n_params_per_symbol
            scaled_dict[symbol] = scaled[start:end]
        return scaled_dict

    def build_obs(self, t):
        """Buduje wektor obserwacji dla chwili t (domyślna implementacja; warianty w version_registry)."""
        self.t_data(t)

        current_portfolio_value = self.wallet.total_value(self.current_prices)
        cash_ratio = self.wallet.balance / current_portfolio_value if current_portfolio_value > 0 else 0.0
        obs_parts = [np.array([float(cash_ratio)], dtype=np.float32)]

        for symbol in self.symbols:
            # A. Pewność modelu ML (Entropy)
            # To kluczowe dla Gatekeepera! Jeśli model ML waha się (0.4 buy, 0.4 sell),
            # to entropia jest wysoka -> RL powinien podnieść progi (nie handlować).
            probs = self.current_probs[symbol]
            buy_p = probs[self.BUY_IDX]
            sell_p = probs[self.SELL_IDX]

            # B. Zmienność rynku (Log Returns + Volatility)
            # Zamiast różnicy cen, weź logarytmiczną stopę zwrotu (znormalizowaną)
            price_t = self.models[symbol].close[t]
            price_t_minus_1 = self.models[symbol].close[t - 1] if t > 0 else price_t

            # Log return: niezależny od rzędu wielkości (działa tak samo dla BTC i SHIB)
            log_ret = np.log(price_t / price_t_minus_1) if price_t_minus_1 > 0 else 0.0

            # "Skalowanie" dla sieci neuronowej (log returny są małe, np. 0.001, mnożymy x100)
            market_momentum = log_ret * 100.0

            # C. Pozycja i PnL (bez zmian, jest OK)
            holding_amount = float(self.wallet.holdings.get(symbol, 0.0))
            holding_value = holding_amount * float(self.current_prices[symbol])
            position_ratio = holding_value / current_portfolio_value if current_portfolio_value > 0 else 0.0

            # Zamiast surowego PnL, który może być > 1000%, spłaszczamy go funkcją tanh
            avg_buy_price = self.wallet.avg_buy_price.get(symbol, 0.0)
            unrealized_pnl = 0.0
            if holding_amount > 0 and avg_buy_price > 0:
                raw_pnl = (self.current_prices[symbol] - avg_buy_price) / avg_buy_price
                unrealized_pnl = np.tanh(raw_pnl)  # Zawsze w zakresie [-1, 1]

            symbol_obs = np.array([
                buy_p,  # Co mówi ML?
                sell_p,  # Co mówi ML?
                # self.models[symbol].volatility[t],
                market_momentum,  # Czy rynek zapierdziela? (zastępuje różnicę cen)
                position_ratio,  # Ile tego mamy?
                unrealized_pnl  # Czy zarabiamy na tym?
            ], dtype=np.float32)

            obs_parts.append(symbol_obs)

        # Składamy i przycinamy
        obs = np.concatenate(obs_parts)
        return np.clip(obs, -5.0, 5.0)