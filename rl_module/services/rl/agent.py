import gymnasium as gymnasium
from gymnasium import spaces
import numpy as np
import types
from test_module.services.core_logic.decision import DecisionEngine
from test_module.services.core_logic.structures import (
    PreviousEpoch, CoreConfig, coerce_experiment_pair, coerce_sim_mode, is_meta_knob_mode,
)
from test_module.services.core_logic.version_registry import EXPERIMENTS, pick_episode_start_t

class TradingEnv(gymnasium.Env):
    """Środowisko Gymnasium do treningu RL: krok barowy, nagroda i obserwacja przez DecisionEngine."""
    def __init__(self, models, num_ticks, base_dir, core_config: CoreConfig = None, episode_timesteps = 10240, agent_type=None, episode_sync=None):
        super().__init__()

        #TODO: przystosowanie pod dqn

        self.symbols = core_config.symbols
        self.core_mode = "train_rl"
        self.base_dir = base_dir
        ver, mode = coerce_experiment_pair(
            core_config.experiment_version, getattr(core_config, "sim_mode", None),
        )
        core_config.experiment_version = ver
        if mode:
            core_config.sim_mode = mode
        self.core_config = core_config
        self.experiment_version = ver
        # shared across SubprocVecEnv workers (None = niezależne starty)
        self._episode_sync = episode_sync

        # Iteration and callback config
        self.t = 0
        self.end_step = 0
        self.max_steps = num_ticks
        self.episode_length = episode_timesteps
        self.epoch_counter = 0
        self.action_space_shape = 0
        self.obs_len_for_symbol = 5
        self.prev_epoch = PreviousEpoch()
        self.last_reward = 0
        self.rewards = []
        self.initial_balance = core_config.balance_start
        self._apply_experiment_version()

        self.engine = DecisionEngine(core_mode=self.core_mode,models=models, base_dir=self.base_dir,
                                     core_config=core_config)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_space_shape * len(self.symbols),), dtype=np.float32)

        obs_len = self.obs_len_global + (self.obs_len_for_symbol * len(self.symbols))
        low = np.full(obs_len, -np.inf, dtype=np.float32)
        high = np.full(obs_len, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._ep_start_t = 0
        self._ep_last_t = 0
        self._knob_keys = (
            "mp_alloc_bias", "mp_alloc_scale", "mp_confidence",
            "mp_urgency", "mp_signal_deadzone", "mp_rebalance_deadzone",
        )
        self._begin_episode_trackers()

        self.reset()

    def _begin_episode_trackers(self):
        self._ep_start_t = int(getattr(self, "t", 0) or 0)
        self._ep_last_t = self._ep_start_t
        self._knob_sums = {k: 0.0 for k in self._knob_keys}
        self._knob_count = 0

    def _track_step_meta(self):
        """Zbiera knoby meta (v6/mprl) + ostatni tick epizodu (do excess/HODL)."""
        self._ep_last_t = int(self.t)
        if not is_meta_knob_mode(
            self.experiment_version, getattr(self.core_config, "sim_mode", None)
        ):
            return
        for symbol in self.symbols:
            cfg = self.engine.config.get(symbol)
            if cfg is None:
                continue
            for k in self._knob_keys:
                self._knob_sums[k] += float(getattr(cfg, k, 0.0))
            self._knob_count += 1

    def _apply_experiment_version(self):
        """Binduje custom methods z experiment registry"""

        ver, mode = coerce_experiment_pair(
            self.experiment_version,
            getattr(self.core_config, "sim_mode", None),
        )
        self.experiment_version = ver
        if self.core_config is not None and mode:
            self.core_config.sim_mode = mode
            self.core_config.experiment_version = ver

        if self.experiment_version not in EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment: {self.experiment_version}. "
                f"Available: {list(EXPERIMENTS.keys())}"
            )

        exp = EXPERIMENTS[self.experiment_version]
        mode = coerce_sim_mode(getattr(self.core_config, "sim_mode", None)) or "rl"
        overlay = (exp.get("by_sim_mode") or {}).get(mode) or {}

        # Override n_params / obs (v6: zależy od sim_mode)
        self.action_space_shape = int(
            overlay.get("n_params_per_symbol", exp["n_params_per_symbol"])
        )
        self.obs_len_for_symbol = int(
            overlay.get("obs_len_for_symbol", exp["obs_len_for_symbol"])
        )
        self.obs_len_global = int(
            overlay.get("obs_len_global", exp.get("obs_len_global", 1))
        )

        # Bind env methods
        for method_name, method_func in exp["env_methods"].items():
            bound_method = types.MethodType(method_func, self)
            setattr(self, method_name, bound_method)

    def prepare_callback(self):
        # Zawsze zapisuj metryki epizodu (także 0 tx / cash) — inaczej dashboard kłamie.
        self.prev_epoch.buys = self.engine.wallet.buys_count
        self.prev_epoch.sells = self.engine.wallet.sells_count
        self.prev_epoch.transactions = self.engine.wallet.transaction_count
        self.prev_epoch.closed_trades = len(getattr(self.engine.wallet, "closed_trades", []) or [])
        self.prev_epoch.portfolio_values = list(self.engine.wallet.portfolio_values_history)
        self.prev_epoch.rewards = list(self.rewards) if self.rewards else [0.0]

        ep_bars = max(1, int(self._ep_last_t) - int(self._ep_start_t))
        self.prev_epoch.episode_bars = ep_bars
        days = ep_bars / 1440.0  # 1m bars
        self.prev_epoch.tx_per_day = float(self.prev_epoch.transactions) / max(days, 1e-9)

        start_bal = float(self.initial_balance) or 1000.0
        if self.prev_epoch.portfolio_values:
            end_p = float(self.prev_epoch.portfolio_values[-1])
        else:
            end_p = start_bal
            self.prev_epoch.portfolio_values = [start_bal]
        self.prev_epoch.strat_return_pct = (end_p / start_bal - 1.0) * 100.0

        # HODL equal-weight na odcinku epizodu — pełna krzywa przez [t0, t1],
        # zsamplowana do długości portfela (potem ::50 w calculate_history).
        # Ważne: mapuj i → t liniowo na cały okres; NIE t0+i (przy dłuższej
        # historii portfela ogon HODL robił się płaski na t1).
        hodl_rets = []
        t0 = max(0, int(self._ep_start_t))
        t1 = max(t0, int(self._ep_last_t))
        price_series = []
        for symbol in self.symbols:
            prices = self.engine.models[symbol].close_prices
            if t1 >= len(prices):
                t1 = len(prices) - 1
            if t0 < len(prices) and prices[t0] > 0 and prices[t1] > 0:
                hodl_rets.append((prices[t1] / prices[t0] - 1.0) * 100.0)
                price_series.append(prices)
        self.prev_epoch.hodl_return_pct = float(np.mean(hodl_rets)) if hodl_rets else 0.0
        self.prev_epoch.excess_pct = self.prev_epoch.strat_return_pct - self.prev_epoch.hodl_return_pct

        n_p = len(self.prev_epoch.portfolio_values)
        span = max(int(t1) - int(t0), 0)
        hodl_curve = []
        if price_series and n_p > 0:
            for i in range(n_p):
                if n_p == 1 or span == 0:
                    t = int(t1)
                else:
                    t = int(t0) + int(round(i * span / (n_p - 1)))
                t = min(max(t, int(t0)), int(t1))
                ratios = []
                for prices in price_series:
                    t_i = min(t, len(prices) - 1)
                    p0 = float(prices[t0])
                    if p0 > 0:
                        ratios.append(float(prices[t_i]) / p0)
                hodl_curve.append(
                    start_bal * float(np.mean(ratios)) if ratios else start_bal
                )
        self.prev_epoch.hodl_values = hodl_curve

        # max drawdown na historii portfela epizodu
        peak = start_bal
        max_dd = 0.0
        for v in self.prev_epoch.portfolio_values:
            fv = float(v)
            if fv > peak:
                peak = fv
            if peak > 0:
                max_dd = max(max_dd, (peak - fv) / peak)
        self.prev_epoch.max_dd_pct = float(max_dd * 100.0)

        if self._knob_count > 0:
            self.prev_epoch.knobs = {
                k: float(self._knob_sums[k] / self._knob_count) for k in self._knob_keys
            }
        else:
            self.prev_epoch.knobs = {}

        self.prev_epoch.calculate_history()
        self.epoch_counter += 1
        self.prev_epoch.epoch = self.epoch_counter


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start_index = self.max_steps - self.episode_length - 1
        # --- STARY PODZIAŁ: każdy env losuje własny start ---
        # if max_start_index > 0:
        #     self.t = self.np_random.integers(0, max_start_index)
        # else:
        #     self.t = 0
        # --- SYNC: wspólny t0 w fali VecEnv (gdy _episode_sync ustawione) ---
        self.t = pick_episode_start_t(self, max_start_index, fallback=0)

        self.end_step = self.t + self.episode_length

        # ------------Reset Env

        self.prepare_callback()
        self.engine.wallet.reset(self.initial_balance)
        self.engine.reset()
        self.last_reward = 0
        self.rewards = []
        self.engine.t_data(self.t)
        self._begin_episode_trackers()
        return self._get_obs(), {}

    def _get_obs(self):
        obs = self.engine.build_obs(self.t)
        return obs

    def _calculate_reward(self):
        """
        Nagroda kroku.
        - legacy (domyślnie): log-return portfela ×100; strata ×2 (stare zachowanie).
        - excess: (portfolio_return − bench_return) ×100; bez ×2.
          bench = równa waga log(close[t]/close[t−1]) po symbolach.
        """
        portfolio_return = 0.0
        if self.engine.prev_value > 0:
            portfolio_return = np.log(self.engine.current_value / self.engine.prev_value)

        mode = str(
            getattr(self.core_config, "reward_mode", None) or "legacy"
        ).strip().lower()

        if mode == "excess":
            benchmark_return = 0.0
            n = len(self.symbols) or 1
            for symbol in self.symbols:
                prices = self.engine.models[symbol].close_prices
                if self.t > 0 and len(prices) > self.t and prices[self.t - 1] > 0:
                    benchmark_return += np.log(prices[self.t] / prices[self.t - 1]) / n
            return float((portfolio_return - benchmark_return) * 100.0)

        # legacy
        reward = portfolio_return * 100.0
        if reward < 0:
            reward *= 2  # Pain is stronger than pleasure
        return reward
    def step(self, action):

        action = np.asarray(action, dtype=np.float32).flatten()
        scaled = (action + 1.0) / 2.0
        # mapowanie na słownik
        scaled_dict= {}
        n_params = self.action_space_shape
        for i, symbol in enumerate(self.symbols):
            idx = i * n_params
            scaled_dict[symbol] = scaled[idx: idx + n_params]


        self.engine.generate_decisions(self.t, scaled=scaled_dict)
        self._track_step_meta()
        self.t += 1
        # Pobranie NOWYCH danych rynkowych (dla t+1)

        self.engine.t_data(self.t)

        # 5. OBLICZENIE NAGRODY (Wydzielona funkcja)
        reward = self._calculate_reward()

        self.engine.wallet.portfolio_values_history.append(self.engine.current_value)

        cum_reward = self.last_reward + reward
        self.rewards.append(round(cum_reward, 4))
        self.last_reward = cum_reward

        terminated = (self.t >= self.end_step) or (self.t >= self.max_steps - 1)
        truncated = False
        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        observation = obs.copy()
        info = {}
        # for symbol in self.symbols:
        #     info[f"{symbol}_portfolio_value"] = float(portfolio_value)
        #     info[f"{symbol}_threshold_buy"] = float(scaled_dict[symbol][0])
        #     info[f"{symbol}_threshold_sell"] = float(scaled_dict[symbol][1])
        #     info[f"{symbol}_tx_fraction"] = float(scaled_dict[symbol][2])
        #     info[f"{symbol}_max_fraction_when_confident"] = float(scaled_dict[symbol][3])
        #     info[f"{symbol}_mincost"] = float(scaled_dict[symbol][4])
        #     info[f"{symbol}_max_wallet_percentage_crypto"] = float(scaled_dict[symbol][5])

        return observation, reward, terminated, truncated, info

