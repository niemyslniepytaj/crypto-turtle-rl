"""Rejestr wariantów eksperymentu: metody decyzji, obserwacji i nagrody per version/sim_mode."""

from .binance_integration import (
    testnet_wallet_init,
    testnet_wallet_buy,
    testnet_wallet_sell,
    testnet_check_pending_orders,
    testnet_sync_after_orders,
)
from .structures import mp_attr

# ============================================================================
# DECISION ENGINE METHODS
# ============================================================================

def v1_decision_rl_config(self, scaled):
    """Ustawia fraction_buy/sell z akcji RL (wariant v1)."""
    for symbol in self.symbols:
        fr_min, fr_max = 0.01, 1.0
        fraction_buy = fr_min + scaled[symbol][0] * (fr_max - fr_min)
        fraction_sell = fr_min + scaled[symbol][1] * (fr_max - fr_min)

        self.config[symbol].fraction_buy = float(fraction_buy)
        self.config[symbol].fraction_sell = float(fraction_sell)


def decision_calculate_fraction(self, order):
    """Przypisuje order.fraction z konfiguracji buy/sell agenta (v1–v3)."""

    symbol = order.symbol
    if order.action == "buy":
        order.fraction = self.config[symbol].fraction_buy
    else:
        order.fraction = self.config[symbol].fraction_sell
    return order


def decision_calculate_signals(self):
    pass


def v3_decision_rl_config(self, scaled_dict):
    """
    Wariant v3: agent ustawia buy/sell/hold oraz fraction per symbol (4 parametry w [0, 1]).
    Predykcje ML są pomijane; progi threshold pozostają 0.5.
    """
    for symbol in self.symbols:
        scaled = scaled_dict[symbol]

        buy_value  = float(scaled[0])
        sell_value = float(scaled[1])
        hold_value = float(max(0.0, 1.0 - buy_value - sell_value))

        self.current_probs[symbol] = [buy_value, hold_value, sell_value]

        fr_min, fr_max = 0.01, 1.0
        self.config[symbol].fraction_buy  = float(fr_min + scaled[2] * (fr_max - fr_min))
        self.config[symbol].fraction_sell = float(fr_min + scaled[3] * (fr_max - fr_min))


# ============================================================================
# TRADING ENV METHODS
# ============================================================================

def v3_decision_build_obs(self, t):
    """
    Obserwacja v3: cash_ratio oraz per symbol (mu, sigma, momentum, position, PnL).
    Wspólna dla train_rl i sim_rl przez engine.build_obs.
    """
    import numpy as np

    current_portfolio_value = self.wallet.total_value(self.current_prices)
    cash_ratio = (
        self.wallet.balance / current_portfolio_value
        if current_portfolio_value > 0 else 0.0
    )

    obs_parts = [np.array([float(cash_ratio)], dtype=np.float32)]

    for symbol in self.symbols:
        model = self.models[symbol]

        mu    = float(model.xgb_mu[t])    if (model.xgb_mu    is not None and len(model.xgb_mu)    > t) else 0.0
        sigma = float(model.xgb_sigma[t]) if (model.xgb_sigma is not None and len(model.xgb_sigma) > t) else 0.0

        price_t      = model.close[t]
        price_t_prev = model.close[t - 1] if t > 0 else price_t
        log_ret         = np.log(price_t / price_t_prev) if price_t_prev > 0 else 0.0
        market_momentum = log_ret * 100.0
        holding_amount = float(self.wallet.holdings.get(symbol, 0.0))
        holding_value  = holding_amount * float(self.current_prices[symbol])
        position_ratio = holding_value / current_portfolio_value if current_portfolio_value > 0 else 0.0

        avg_buy_price  = self.wallet.avg_buy_price.get(symbol, 0.0)
        unrealized_pnl = 0.0
        if holding_amount > 0 and avg_buy_price > 0:
            raw_pnl        = (self.current_prices[symbol] - avg_buy_price) / avg_buy_price
            unrealized_pnl = np.tanh(raw_pnl)
        # Średni log-return z ostatnich 20 kroków
        lookback = 20
        returns = [np.log(model.close[t - i] / model.close[t - i - 1])
                   for i in range(min(lookback, t))]
        trend = np.mean(returns) * 100.0 if returns else 0.0

        symbol_obs = np.array([
            mu,              # Przewidywana zmiana ceny (XGB mu)
            sigma,           # Niepewność predykcji (XGB sigma)
            market_momentum, # Momentum rynku
            position_ratio,  # Udział symbolu w portfelu
            unrealized_pnl,  # Niezrealizowany P&L
            trend,
        ], dtype=np.float32)

        obs_parts.append(symbol_obs)

    obs = np.concatenate(obs_parts)
    if not np.isfinite(obs).all():
        bad = np.where(~np.isfinite(obs))[0]
        for i in bad:
            print(f"[NaN] t={t} feature={obs[i]} val={obs[i]}")
    obs_min, obs_max = obs.min(), obs.max()
    obs = 2 * (obs - obs_min) / (obs_max - obs_min + 1e-8) - 1
    return np.clip(obs, -5.0, 5.0)


def v3_env_calculate_reward(self):
    """
    Funkcja nagrody odporna na strategię 'nic nie rób'.

    Składowe:
      1. ALPHA          — log-return portfela minus benchmark buy&hold * 100.
                          Agent musi bić rynek, nie tylko trzymać cash.
      2. IDLE PENALTY   — kara rosnąca liniowo po grace period bez transakcji.
                          Pierwsze `idle_grace` kroków są darmowe (czekanie na sygnał).
                          Potem kara rośnie do max `idle_max_pen` na krok.
      3. BONUS          — jeśli transakcja i alpha > 0, bonus = alpha * 0.5.
                          Nagradza trafne decyzje, nie samą aktywność.
      4. ASYMETRIA STRAT — alpha < 0 → *1.5 (łagodniejsze niż poprzednie *2,
                           które czyniło inaction zawsze dominującą strategią).
    Usage: V3
    """
    import numpy as np

    t = self.t

    # ── 1. Log-return portfela ───────────────────────────────────────────────
    portfolio_return = 0.0
    if self.engine.prev_value > 0:
        portfolio_return = np.log(self.engine.current_value / self.engine.prev_value)

    # ── 2. Benchmark: equal-weight buy & hold ────────────────────────────────
    benchmark_return = 0.0
    n = len(self.symbols)
    for symbol in self.symbols:
        prices = self.engine.models[symbol].close_prices
        if t > 0 and prices[t - 1] > 0:
            benchmark_return += np.log(prices[t] / prices[t - 1]) / n

    alpha = (portfolio_return - benchmark_return) * 100.0

    # ── 3. Asymetria strat ───────────────────────────────────────────────────
    if alpha < 0:
        alpha *= 1.5

    # ── 4. Kara za bezczynność ───────────────────────────────────────────────
    accepted_this_step = sum(
        1 for o in self.engine.wallet.logs
        if o.timestamp == t - 1 and o.status == "accepted"
    )

    if accepted_this_step > 0:
        self.idle_steps = 0
    else:
        self.idle_steps += 1

    idle_grace = 20
    idle_max_pen = 0.5
    idle_penalty = idle_max_pen * (1 - np.exp(-0.01 * max(0, self.idle_steps - idle_grace)))

    # ── 5. Bonus za trafną transakcję ────────────────────────────────────────
    transaction_bonus = 0.0
    if accepted_this_step > 0 and alpha > 0:
        transaction_bonus = alpha * 0.5

    reward = alpha - idle_penalty + transaction_bonus
    return reward


def v3_env_reset(self, seed=None, options=None):
    """
    Nadpisuje reset w TradingEnv — inicjuje idle_steps dla v3.
    Usage: V3
    """
    import numpy as np

    super(self.__class__, self).reset(seed=seed)

    max_start_index = self.max_steps - self.episode_length - 1
    if max_start_index > 0:
        self.t = np.random.randint(0, max_start_index)
    else:
        self.t = 0

    self.end_step = self.t + self.episode_length

    self.prepare_callback()
    self.engine.wallet.reset(self.initial_balance)
    self.engine.reset()
    self.last_reward  = 0
    self.rewards      = []
    self.idle_steps   = 0          # ← reset licznika bezczynności
    self.engine.t_data(self.t)
    return self._get_obs(), {}


# ============================================================================
# V4 — Target-shaped reward + rozszerzona obserwacja
#
# Kluczowe zmiany względem v3:
#
#  NAGRODA:
#    Składowa alignment_bonus: nagradzamy agenta gdy jego akcja zgadza się
#    z tym co faktycznie zrobi cena za TARGET_HORIZON kroków.
#    Target NIE wchodzi do obserwacji — agent musi sam znaleźć pattern
#    w obs który koreluje z przyszłym ruchem.
#    lambda_align maleje w trakcie epizodu (curriculum) żeby agent
#    stopniowo uczył się bardziej z alpha niż z "podpowiedzi".
#
#  OBSERWACJA (9 featury per symbol + 1 globalny = 1 + 9*n):
#    mu, sigma          — predykcja XGBoost i jej niepewność
#    xgb_zscore         — mu/sigma: pewność sygnału (NOWE)
#    momentum_1         — log-return ostatniej świeczki * 100
#    momentum_5         — log-return 5 świeczek * 100 (NOWE)
#    trend_20           — średni log-return z 20 kroków * 100
#    volatility_ratio   — sigma_5 / sigma_50: czy rynek się rozgrzewa (NOWE)
#    position_ratio     — udział symbolu w portfelu
#    unrealized_pnl     — tanh(pnl%), zawsze w [-1, 1]
#    steps_since_trade  — znormalizowany czas od ostatniej transakcji (NOWE)
#
#  POPRAWKI z poprzedniej analizy:
#    - trend przy t=0 zwraca 0.0 zamiast nan
#    - usunięta błędna per-step min/max normalizacja obs
#    - steps_since_trade resetuje się przy każdej zaakceptowanej transakcji
# ============================================================================

TARGET_HORIZON = 15  # ile kroków w przód liczymy target (minuty dla danych 1m)
LAMBDA_ALIGN_START = 0.8  # waga alignment na początku epizodu
LAMBDA_ALIGN_END = 0.2  # waga alignment pod koniec epizodu (agent "wyrasta" z podpowiedzi)


def v4_decision_rl_config(self, scaled_dict):
    """
    Identyczny mechanizm jak v3: agent zwraca [buy, sell, fraction_buy, fraction_sell].
    Usage: V4
    """
    for symbol in self.symbols:
        scaled = scaled_dict[symbol]

        buy_value = float(scaled[0])
        sell_value = float(scaled[1])
        hold_value = float(max(0.0, 1.0 - buy_value - sell_value))

        self.current_probs[symbol] = [buy_value, hold_value, sell_value]

        fr_min, fr_max = 0.01, 1.0
        self.config[symbol].fraction_buy = float(fr_min + scaled[2] * (fr_max - fr_min))
        self.config[symbol].fraction_sell = float(fr_min + scaled[3] * (fr_max - fr_min))


def v4_decision_build_obs(self, t):
    """
    Rozszerzona obserwacja dla v4.
    Kształt: [cash_ratio, mu, sigma, xgb_zscore, momentum_1, momentum_5,
               trend_20, volatility_ratio, position_ratio, unrealized_pnl,
               steps_since_trade] * n_symbols
    Usage: V4
    """
    import numpy as np

    current_portfolio_value = self.wallet.total_value(self.current_prices)
    cash_ratio = (
        self.wallet.balance / current_portfolio_value
        if current_portfolio_value > 0 else 0.0
    )

    obs_parts = [np.array([float(cash_ratio)], dtype=np.float32)]

    for symbol in self.symbols:
        model = self.models[symbol]

        # ── XGBoost signal ───────────────────────────────────────────────────
        mu = float(model.xgb_mu[t]) if (model.xgb_mu is not None and len(model.xgb_mu) > t) else 0.0
        sigma = float(model.xgb_sigma[t]) if (model.xgb_sigma is not None and len(model.xgb_sigma) > t) else 0.0

        # Z-score: jak pewny jest sygnał względem szumu rynkowego
        # Duże |zscore| = model bardzo pewny kierunku
        xgb_zscore = np.tanh(mu / (sigma + 1e-8))  # tanh żeby zawrzeć w [-1, 1]

        # ── Momentum (kilka timeframe'ów) ────────────────────────────────────
        close = model.close
        price_t = close[t]

        momentum_1 = 0.0
        if t > 0 and close[t - 1] > 0:
            momentum_1 = np.log(price_t / close[t - 1]) * 100.0

        momentum_5 = 0.0
        if t >= 5 and close[t - 5] > 0:
            momentum_5 = np.log(price_t / close[t - 5]) * 100.0

        # ── Trend (średni log-return z 20 kroków) ────────────────────────────
        lookback_trend = 20
        n_avail = min(lookback_trend, t)
        if n_avail > 0:
            returns = [
                np.log(close[t - i] / close[t - i - 1])
                for i in range(n_avail)
                if close[t - i - 1] > 0
            ]
            trend_20 = float(np.mean(returns)) * 100.0 if returns else 0.0
        else:
            trend_20 = 0.0

        # ── Volatility ratio: sigma_5 / sigma_50 ─────────────────────────────
        # > 1.0: rynek się rozgrzewa (krótkoterm. zmienność > długoterm.)
        # < 1.0: rynek się uspokaja
        # Pozwala agentowi rozróżnić spokojny trend od nerwowego szumu
        vol_ratio = 1.0
        if t >= 50:
            rets_5 = [np.log(close[t - i] / close[t - i - 1]) for i in range(5) if close[t - i - 1] > 0]
            rets_50 = [np.log(close[t - i] / close[t - i - 1]) for i in range(50) if close[t - i - 1] > 0]
            s5 = float(np.std(rets_5)) if len(rets_5) > 1 else 1e-8
            s50 = float(np.std(rets_50)) if len(rets_50) > 1 else 1e-8
            vol_ratio = np.tanh(s5 / (s50 + 1e-8) - 1.0)  # tanh(ratio - 1): 0 gdy równe
        else:
            vol_ratio = 0.0

        # ── Pozycja i PnL ────────────────────────────────────────────────────
        holding_amount = float(self.wallet.holdings.get(symbol, 0.0))
        holding_value = holding_amount * float(self.current_prices[symbol])
        position_ratio = holding_value / current_portfolio_value if current_portfolio_value > 0 else 0.0

        avg_buy_price = self.wallet.avg_buy_price.get(symbol, 0.0)
        unrealized_pnl = 0.0
        if holding_amount > 0 and avg_buy_price > 0:
            raw_pnl = (self.current_prices[symbol] - avg_buy_price) / avg_buy_price
            unrealized_pnl = float(np.tanh(raw_pnl))

        # ── Czas od ostatniej transakcji (per symbol) ────────────────────────
        # Normalizowany: 0 = właśnie handlował, ~1 = dawno nie handlował
        # Zapobiega temu że agent nie wie czy "już może handlować znowu"
        steps_since = float(getattr(self, '_steps_since_trade', {}).get(symbol, 50))
        steps_since_norm = np.tanh(steps_since / 50.0)  # nasyca przy ~150 krokach

        symbol_obs = np.array([
            mu,  # Predykcja XGBoost (surowa, bez normalizacji)
            sigma,  # Niepewność XGBoost
            xgb_zscore,  # mu/sigma: pewność kierunku w [-1, 1]
            momentum_1,  # Log-return ostatniej świeczki * 100
            momentum_5,  # Log-return 5 świeczek * 100
            trend_20,  # Średni trend 20 kroków * 100
            vol_ratio,  # Volatility ratio w [-1, 1]
            position_ratio,  # Ile portfela w tym symbolu
            unrealized_pnl,  # Niezrealizowany PnL w [-1, 1]
            steps_since_norm,  # Czas od ostatniej transakcji w [0, 1]
        ], dtype=np.float32)

        obs_parts.append(symbol_obs)

    obs = np.concatenate(obs_parts)

    # Walidacja NaN/Inf — zastępuj zerem zamiast crashować
    if not np.isfinite(obs).all():
        bad_idx = np.where(~np.isfinite(obs))[0]
        for i in bad_idx:
            print(f"[v4_obs][NaN/Inf] t={t} idx={i} val={obs[i]}")
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    return np.clip(obs, -5.0, 5.0)


def v4_env_calculate_reward(self):
    """
    Nagroda v4 z target-shaped alignment.

    Składowe:
      1. ALPHA          — log-return portfela minus benchmark * 100.
                          Podstawowy sygnał finansowy.

      2. ALIGNMENT BONUS — agent dostaje bonus gdy jego akcja zgadza się
                          z faktycznym ruchem ceny za TARGET_HORIZON kroków.
                          Target = tanh(future_log_return * 200):
                            ~+1 gdy cena mocno wzrośnie
                            ~-1 gdy cena mocno spadnie
                            ~0  gdy cena stoi w miejscu
                          action_signal:
                            +1 = buy (scaled[0] > scaled[1])
                             0 = hold
                            -1 = sell
                          alignment = target * action_signal
                          lambda maleje liniowo przez epizod (curriculum):
                            na początku agent korzysta z "podpowiedzi"
                            pod koniec musi sam oceniać sytuację

      3. LOSS ASYMMETRY  — alpha < 0 → kara * 1.5 (zniechęca do strat)

      4. IDLE PENALTY    — rośnie gdy agent nie handluje za długo

    Usage: V4
    """
    import numpy as np

    t = self.t

    # ── 1. Alpha (portfolio vs benchmark) ────────────────────────────────────
    portfolio_return = 0.0
    if self.engine.prev_value > 0:
        portfolio_return = np.log(self.engine.current_value / self.engine.prev_value)

    benchmark_return = 0.0
    n_sym = len(self.symbols)
    for symbol in self.symbols:
        prices = self.engine.models[symbol].close_prices
        if t > 0 and prices[t - 1] > 0:
            benchmark_return += np.log(prices[t] / prices[t - 1]) / n_sym

    alpha = (portfolio_return - benchmark_return) * 100.0
    if alpha < 0:
        alpha *= 1.5

    # ── 2. Alignment bonus ───────────────────────────────────────────────────
    alignment_bonus = 0.0
    horizon = TARGET_HORIZON

    # Sprawdzamy czy mamy dane wystarczająco daleko w przód
    # (epizod powinien być dłuższy niż horizon — sprawdź przy konfiguracji)
    for symbol in self.symbols:
        prices = self.engine.models[symbol].close_prices
        t_future = t + horizon

        if t_future < len(prices) and prices[t] > 0:
            # Target: jak zmieni się cena za `horizon` kroków
            # tanh * 200 żeby mała zmiana (0.5%) dawała ~tanh(1) ≈ 0.76
            # czyli już wyraźny sygnał, nie tylko przy dużych ruchach
            future_log_ret = np.log(prices[t_future] / prices[t])
            target = float(np.tanh(future_log_ret * 200.0))

            # Akcja agenta w tym kroku (pobrana z last_rl_config jeśli dostępna)
            # Używamy current_probs ustawionych przez rl_config w tym kroku
            probs = self.engine.current_probs.get(symbol, [0.33, 0.34, 0.33])
            buy_p = probs[0]
            sell_p = probs[2]

            # action_signal: ciągły, nie binarny
            # +1 gdy agent mocno buy, -1 gdy mocno sell, 0 gdy hold/niezdecydowany
            action_signal = float(np.tanh((buy_p - sell_p) * 4.0))
            # *4 żeby różnica 0.25 już dawała ~tanh(1) ≈ 0.76

            # Alignment: +1 gdy trafne, -1 gdy błędne, 0 gdy hold lub brak ruchu
            alignment = target * action_signal
            alignment_bonus += alignment / n_sym

    # Curriculum: lambda maleje liniowo od START do END przez cały epizod
    episode_progress = float(t - self.t_start) / max(self.episode_length, 1)
    episode_progress = float(np.clip(episode_progress, 0.0, 1.0))
    lambda_align = LAMBDA_ALIGN_START + episode_progress * (LAMBDA_ALIGN_END - LAMBDA_ALIGN_START)

    # ── 3. Idle penalty ──────────────────────────────────────────────────────
    accepted_this_step = sum(
        1 for o in self.engine.wallet.logs
        if o.timestamp == t - 1 and o.status == "accepted"
    )

    if accepted_this_step > 0:
        self.idle_steps = 0
        # Aktualizuj steps_since_trade per symbol
        for o in self.engine.wallet.logs:
            if o.timestamp == t - 1 and o.status == "accepted":
                if not hasattr(self.engine, '_steps_since_trade'):
                    self.engine._steps_since_trade = {}
                self.engine._steps_since_trade[o.symbol] = 0
    else:
        self.idle_steps += 1
        # Inkrementuj steps_since_trade
        if not hasattr(self.engine, '_steps_since_trade'):
            self.engine._steps_since_trade = {}
        for symbol in self.symbols:
            self.engine._steps_since_trade[symbol] = (
                    self.engine._steps_since_trade.get(symbol, 0) + 1
            )

    idle_grace = 20
    idle_max_pen = 0.4
    idle_penalty = idle_max_pen * (1 - np.exp(-0.01 * max(0, self.idle_steps - idle_grace)))

    # ── Składamy nagrodę ─────────────────────────────────────────────────────
    reward = alpha + lambda_align * alignment_bonus - idle_penalty
    return float(reward)


def v4_env_reset(self, seed=None, options=None):
    """
    Reset dla v4 — inicjuje idle_steps, t_start i steps_since_trade.
    Usage: V4
    """
    import numpy as np

    super(self.__class__, self).reset(seed=seed)

    max_start_index = self.max_steps - self.episode_length - TARGET_HORIZON - 1
    if max_start_index > 0:
        self.t = np.random.randint(0, max_start_index)
    else:
        self.t = 0

    self.end_step = self.t + self.episode_length
    self.t_start = self.t  # potrzebne do curriculum w calculate_reward

    self.prepare_callback()
    self.engine.wallet.reset(self.initial_balance)
    self.engine.reset()
    self.last_reward = 0
    self.rewards = []
    self.idle_steps = 0
    self.engine._steps_since_trade = {symbol: 50 for symbol in self.symbols}
    self.engine.t_data(self.t)
    return self._get_obs(), {}
# ============================================================================
# WALLET METHODS
# ============================================================================
# ============================================================================
# V5 — Risk-Adjusted Portfolio Management
#
# Filozofia: agent nie próbuje "przewidzieć rynku" — zarządza RYZYKIEM.
#
# KLUCZOWE ZMIANY względem v4:
#
#  ARCHITEKTURA DZIAŁANIA:
#    Agent ma 3 parametry per symbol: [position_target, confidence, urgency]
#    - position_target [0,1]: docelowy % portfela w tym symbolu (0=0%, 1=100%)
#    - confidence [0,1]:      pewność sygnału — skaluje wielkość kroku
#    - urgency [0,1]:         jak szybko rebalansować (małe = cierpliwie)
#    Rebalansowanie: jeśli current_pos < target → buy, jeśli > → sell
#    Fraction = |current_pos - target| * confidence * urgency
#    → Agent myśli w kategoriach PORTFELA, nie pojedynczych transakcji
#
#  OBSERWACJA (13 featury per symbol + 3 globalne):
#    Globalne:
#      cash_ratio          — % gotówki w portfelu
#      portfolio_return_5  — zwrot portfela z ostatnich 5 kroków
#      drawdown            — aktualny drawdown od lokalnego szczytu
#    Per symbol:
#      mu, sigma           — predykcja XGBoost i jej niepewność
#      xgb_zscore          — mu/sigma: pewność sygnału
#      momentum_1          — log-return ostatniej świeczki * 100
#      momentum_5          — log-return 5 świeczek * 100
#      momentum_20         — log-return 20 świeczek * 100
#      vol_regime          — volatility ratio (short/long), tanh-scaled
#      rsi_proxy           — upward moves / total moves, 14-krokowy
#      position_ratio      — aktualny % portfela w tym symbolu
#      unrealized_pnl      — tanh(pnl%), zawsze w [-1, 1]
#      time_in_position    — jak długo trzymamy pozycję (nasycenie przy 200 krokach)
#      xgb_signal_1h       — średnia predykcja mu z ostatnich 60 kroków
#
#  NAGRODA (Sharpe-inspired, multi-component):
#    1. SHARPE PROXY       — alpha (vs benchmark) podzielone przez lokalną
#                            zmienność zwrotów portfela z ostatnich 20 kroków.
#                            Nagradza stabilne zyski, karze za nerwowe ruchy.
#    2. DRAWDOWN PENALTY   — kara proporcjonalna do aktualnego drawdownu.
#                            Silnie zniechęca do wielkich strat.
#    3. EFFICIENCY BONUS   — bonus za transakcje które przyniosły zysk
#                            (zrealizowany lub niezrealizowany).
#    4. OVERTRADING PENALTY— kara za nadmierną aktywność (transaction_cost proxy).
#                            Każda transakcja kosztuje 0.001 (jak fee).
#    5. CONCENTRATION PENALTY — kara za trzymanie >80% portfela w jednym symbolu.
#                            Wymusza dywersyfikację gdy jest wiele symboli.
#
#  CURRICULUM:
#    - Pierwsze 30% epizodu: lambda_drawdown=2.0 (naucz się nie tracić),
#      lambda_sharpe=0.5 (nie za dużo presji na zysk)
#    - Ostatnie 30%: lambda_drawdown=0.5, lambda_sharpe=2.0 (zarabiaj!)
#    - Środek: liniowe przejście
#
#  POPRAWKI TECHNICZNE:
#    - Brak per-step normalizacji obs (błąd z v3, naprawiony w v4, utrzymany)
#    - nan_to_num wszędzie gdzie dzielimy przez sigma/vol
#    - steps_since_trade zastąpione przez time_in_position (per symbol)
#    - max_start_index uwzględnia TARGET_HORIZON_V5 (brak OOB przy align bonus)
# ============================================================================

# --------------------------------------------------------------------------
# Stałe dla v5
# --------------------------------------------------------------------------
TARGET_HORIZON_V5    = 20    # ile kroków w przód liczymy alignment target
RSI_WINDOW_V5        = 14    # okno RSI proxy
VOL_SHORT_V5         = 10    # okno krótkoterminalowej zmienności
VOL_LONG_V5          = 100   # okno długoterminowej zmienności
SIGNAL_LOOKBACK_V5   = 60    # okno średniej predykcji mu XGBoost

# Curriculum lambdy
CURRICULUM_RAMPUP    = 0.30  # % epizodu — faza "nie trać"
CURRICULUM_RAMPDOWN  = 0.70  # % epizodu — od tu faza "zarabiaj"
LAMBDA_SHARPE_START  = 0.5
LAMBDA_SHARPE_END    = 2.0
LAMBDA_DRAW_START    = 2.0
LAMBDA_DRAW_END      = 0.5

# Kary
OVERTRADE_COST       = 0.002  # koszt jednej transakcji (2x fee = przybliżenie round-trip)
CONCENTRATION_THR    = 0.80   # powyżej tej % portfela w jednym symbolu -> kara
CONCENTRATION_PEN    = 0.3    # maks kara za koncentrację per krok

# --------------------------------------------------------------------------
# v5rl meta-kontroler — pełny zakres fizyczny u∈[0,1]
#   bias/scale/conf/urgency → [0, 1]
#   signal_dz / rebalance_dz → [0, 0.5]  (= 0.5 * u)
# hold + Δknob penalty ograniczają jitter (nie pas wokół v5mp).
# --------------------------------------------------------------------------
# centrum = locked v5mp (tylko seed przy reset / obs reference)
V5MP_KNOB_CENTER = (
    0.4976812781670948,   # mp_alloc_bias
    0.8716159315285962,   # mp_alloc_scale
    0.7813473534299133,   # mp_confidence
    0.7238180061686769,   # mp_urgency
    0.05599558165565585,  # mp_signal_deadzone (physical)
    0.21236378896795072,  # mp_rebalance_deadzone (physical)
)
V5RL_KNOB_HOLD_BARS = 1           # decyzja meta-knobów co 1 bar (1m)
V5RL_TRADE_LOCK_BARS = 15         # po accepted buy/sell: lock portfela (SL omija)
V5RL_DKNOB_PENALTY = 0.20         # kara * mean(|Δknob|) przy odświeżeniu holda
V5RL_TX_RATE_WINDOW = 60          # rolling tx density do obs
# podłoga deadzone — u=0 nie może oznaczać "rebalance co bar"
V5RL_SIGNAL_DZ_FLOOR = 0.05
V5RL_REBALANCE_DZ_FLOOR = 0.12
# waga krokowego excess vs HODL w nagrodzie
# (2026-10-07) mocniejszy anti-cash + głośniejszy terminal — poprzedni run
# uczył „mniej boli w bear”, cash-drag w bull
V5RL_EXCESS_SCALE = 4.5
V5RL_TX_STEP_PEN = 0.05
V5RL_CASH_PEN = 0.40
V5RL_CASH_RATIO_THR = 0.50          # było 0.80 — kara wcześniej
V5RL_SIGNAL_ABS_THR = 0.08          # było 0.12
V5RL_ALIGN_SCALE = 0.10
V5RL_TERMINAL_EXCESS = 1.0



def pick_episode_start_t(env, max_start_index, fallback=0):
    """
    Wspólny start epizodu dla wszystkich workerów VecEnv.

    Sync (domyślnie ON): deterministyczny t0 z (salt + epoch_counter).
    Przy stałej długości epizodu envy są w lockstep → ten sam wave → ten sam t0.
    Nie używamy multiprocessing.Value: przy spawn na Windows każdy worker
    dostaje własną kopię i „sync” zamienia się w powtarzanie tego samego
    okna przez num_envs epizodów z rzędu.

    Bez sync / enabled=False → niezależny start per env (stary tryb).

    Stary niezależny los (przywróć zamiast bloku sync):
      # if max_start_index > 0:
      #     return int(env.np_random.integers(0, max_start_index))
      # return int(fallback)

    Stary (zepsuty) sync przez shared Value+Lock — zostawione dla historii:
      # with lock:
      #     count_v.value += 1
      #     if (int(count_v.value) - 1) % n == 0:
      #         start_v.value = int(env.np_random.integers(0, max_start_index))
      #     return int(start_v.value)
    """
    import numpy as np

    if max_start_index <= 0:
        return int(fallback)

    sync = getattr(env, "_episode_sync", None)
    if not sync or not sync.get("enabled", False):
        # --- STARY PODZIAŁ: każdy env losuje własny start ---
        return int(env.np_random.integers(0, max_start_index))

    # wave = ile epizodów już domkniętych (prepare_callback ++ po pick,
    # więc przy resecie epoch_counter = index kolejnego epizodu)
    wave = int(getattr(env, "epoch_counter", 0) or 0)
    salt = int(sync.get("salt", 0) or 0)
    rng = np.random.default_rng((salt + 1) * 1_000_003 + wave * 97_411)
    return int(rng.integers(0, max_start_index))


# ============================================================================
# DECISION METHODS — V5
# ============================================================================

def v5_decision_rl_config(self, scaled_dict):
    """
    Agent zarządza portfelem przez wskazanie DOCELOWEJ POZYCJI.

    Parametry per symbol (3):
      scaled[0] -> position_target  [0, 1]  — docelowy udział w portfelu
      scaled[1] -> confidence       [0, 1]  — jak bardzo agent jest pewny
      scaled[2] -> urgency          [0, 1]  — jak szybko rebalansować

    Mechanizm:
      current_ratio = wartosc_krypto[symbol] / total_portfolio
      delta = position_target - current_ratio

      delta > 0  -> buy  (jesteśmy poniżej targetu)
      delta < 0  -> sell (jesteśmy powyżej targetu)
      |delta| mały -> hold (różnica zbyt mała by opłacało się handlować)

      fraction = clip(|delta| * confidence * urgency_scale, min, max)

    Dlaczego to działa lepiej niż buy/sell/hold:
      - Agent myśli w kategoriach alokacji portfela (jak prawdziwy PM)
      - Naturalne wygaszanie: im bliżej targetu, tym mniejsza transakcja
      - Confidence skaluje agresywność -> agent może być ostrożny
      - Urgency pozwala agentowi decydować o tempie rebalansowania

    Usage: V5
    """
    import numpy as np

    current_prices = self.current_prices
    total_value = self.wallet.total_value(current_prices)

    for symbol in self.symbols:
        scaled = scaled_dict[symbol]

        position_target_raw = float(scaled[0])     # [0, 1]
        confidence      = float(scaled[1])     # [0, 1]
        urgency         = float(scaled[2])     # [0, 1]

        # EMA wygładzanie (anti-noise); v5mp/v5rl ustawiają wolniejsze alpha_* na DecisionConfig
        if not hasattr(self, '_prev_target'): self._prev_target = {}
        if symbol not in self._prev_target:
            h_val = self.wallet.holdings.get(symbol, 0.0) * current_prices.get(symbol, 0.0)
            self._prev_target[symbol] = h_val / total_value if total_value > 0 else 0.0

        alpha_min = float(getattr(self.config[symbol], "ema_alpha_min", 0.05))
        alpha_span = float(getattr(self.config[symbol], "ema_alpha_span", 0.20))
        alpha_ema = alpha_min + alpha_span * urgency
        position_target = alpha_ema * position_target_raw + (1.0 - alpha_ema) * self._prev_target[symbol]
        self._prev_target[symbol] = position_target

        # Aktualna pozycja
        holding_value   = self.wallet.holdings.get(symbol, 0.0) * current_prices.get(symbol, 0.0)
        current_ratio   = holding_value / total_value if total_value > 0 else 0.0

        delta = position_target - current_ratio

        # Minimalna delta żeby handlować (v5mp/v5rl: większy dead-zone z configu)
        dead_zone = float(getattr(self.config[symbol], "rebalance_dead_zone", 0.15))

        cash = self.wallet.balance
        
        if abs(delta) < dead_zone:
            action      = "hold"
            buy_signal  = 0.33
            sell_signal = 0.33
            fraction    = 0.0
        elif delta > 0:
            action      = "buy"
            buy_signal  = 0.5 + delta * 0.5          # [0.5, 1.0] proporcjonalnie do delty
            sell_signal = 0.0
            
            # Aby osiągnąć target, musimy kupić za: delta * total_value
            target_buy_value = delta * total_value
            urgency_scale = 0.3 + urgency * 0.7      # [0.3, 1.0]
            
            if cash > 0:
                exact_fraction = target_buy_value / cash
            else:
                exact_fraction = 0.0
                
            fraction = float(np.clip(exact_fraction * confidence * urgency_scale, 0.05, 1.0))
        else:
            action      = "sell"
            sell_signal = 0.5 + abs(delta) * 0.5     # [0.5, 1.0]
            buy_signal  = 0.0
            
            # Aby osiągnąć target, musimy sprzedać za: abs(delta) * total_value
            target_sell_value = abs(delta) * total_value
            urgency_scale = 0.3 + urgency * 0.7
            
            if holding_value > 0:
                exact_fraction = target_sell_value / holding_value
            else:
                exact_fraction = 0.0
                
            fraction = float(np.clip(exact_fraction * confidence * urgency_scale, 0.05, 1.0))

        hold_signal = float(max(0.0, 1.0 - buy_signal - sell_signal))
        self.current_probs[symbol] = [buy_signal, hold_signal, sell_signal]

        self.config[symbol].fraction_buy  = fraction
        self.config[symbol].fraction_sell = fraction


def v5_decision_calculate_fraction(self, order):
    """
    Fraction już wyliczona w rl_config przez mechanizm delta-rebalansowania.
    Tutaj tylko przepisujemy wartość z config.

    Usage: V5
    """
    symbol = order.symbol
    if order.action == "buy":
        order.fraction = self.config[symbol].fraction_buy
    else:
        order.fraction = self.config[symbol].fraction_sell
    return order


def v5mp_decision_mp_config(self):
    """
    Czyste ML na silniku alokacji v5 (anty-churn) — locked path v5mp:
      - |signal| < mp_signal_deadzone → nie ruszaj raw targetu
      - confidence = |signal| * mp_confidence (nie stała)
      - wolniejsze EMA + większy rebalance dead-zone niż RL/v5
    Zwraca [target, confidence, urgency] jak PPO.
    Usage: V5MP (sim_ml); też mapper wewnętrzny dla v5rl
    """
    if not hasattr(self, "_mp_prev_target_raw"):
        self._mp_prev_target_raw = {}

    scaled_dict = {}
    for symbol in self.symbols:
        probs = self.current_probs.get(symbol, [0.33, 0.34, 0.33])
        buy_p = float(probs[self.BUY_IDX])
        sell_p = float(probs[self.SELL_IDX])
        signal = buy_p - sell_p  # ~[-1, 1]

        cfg = self.config[symbol]
        bias = float(mp_attr(cfg, "mp_alloc_bias", 0.5))
        scale = float(mp_attr(cfg, "mp_alloc_scale", 0.5))
        sig_dz = float(mp_attr(cfg, "mp_signal_deadzone", 0.15))
        conf_gain = float(mp_attr(cfg, "mp_confidence", 0.5))
        urgency = float(mp_attr(cfg, "mp_urgency", 0.5))

        prev_raw = self._mp_prev_target_raw.get(symbol, bias)
        if abs(signal) < sig_dz:
            target = prev_raw
        else:
            target = max(0.0, min(1.0, bias + scale * signal))
            self._mp_prev_target_raw[symbol] = target

        # słaby sygnał → mała pewność → mniejsze rebalance
        confidence = max(0.05, min(1.0, abs(signal) * (0.5 + conf_gain)))

        # przełącz silnik v5 na tryb MP (wolniejsze EMA, szerszy dead-zone)
        cfg.rebalance_dead_zone = float(mp_attr(cfg, "mp_rebalance_deadzone", 0.25))
        cfg.ema_alpha_min = float(mp_attr(cfg, "mp_ema_alpha_min", 0.02))
        cfg.ema_alpha_span = float(mp_attr(cfg, "mp_ema_alpha_span", 0.08))

        scaled_dict[symbol] = [target, confidence, urgency]
    return scaled_dict


def v5rl_decision_rl_config(self, scaled_dict):
    """
    Meta-kontroler v5rl:
      - akcja u∈[0,1] → pełny zakres fizyczny
          [0..3] bias/scale/conf/urgency → [0, 1]
          [4..5] signal/rebalance deadzone → [0, 0.5]
      - knoby odświeżane co V5RL_KNOB_HOLD_BARS barów (domyślnie 1 = co minutę)
      - anti-churn: po accepted buy/sell portfel zablokowany
        V5RL_TRADE_LOCK_BARS barów (stop_loss omija lock)
      - Δknob przy odświeżeniu → self._v5rl_dknob_l1 (kara w reward)

    Potem ten sam pipeline co v5mp (ml_config → v5 rebalance).
    Usage: V5RL (train_rl / sim_rl)
    """
    import numpy as np

    t = int(getattr(self, "_t", 0) or 0)
    hold_bars = getattr(self, "_v5rl_hold_bars", None)
    if hold_bars is None:
        hold_bars = getattr(getattr(self, "core_config", None), "v5rl_knob_hold_bars", None)
    hold_bars = int(hold_bars if hold_bars is not None else V5RL_KNOB_HOLD_BARS)
    if not hasattr(self, "_v5rl_held_u"):
        self._v5rl_held_u = {}
        self._v5rl_hold_until = {}
        self._v5rl_prev_physical = {}
        self._v5rl_dknob_l1 = 0.0

    self._v5rl_dknob_l1 = 0.0
    dknob_acc = []

    for symbol in self.symbols:
        raw = np.asarray(scaled_dict[symbol], dtype=np.float64).reshape(-1)
        if len(raw) < 6:
            raw = np.pad(raw, (0, 6 - len(raw)), constant_values=0.5)
        u = np.clip(raw[:6], 0.0, 1.0)

        until = int(self._v5rl_hold_until.get(symbol, -1))
        if t < until and symbol in self._v5rl_held_u:
            u = np.asarray(self._v5rl_held_u[symbol], dtype=np.float64)
            refresh = False
        else:
            self._v5rl_held_u[symbol] = u.copy()
            self._v5rl_hold_until[symbol] = t + max(1, hold_bars)
            refresh = True

        physical = [
            float(u[0]),
            float(u[1]),
            float(u[2]),
            float(u[3]),
            float(max(V5RL_SIGNAL_DZ_FLOOR, 0.5 * float(u[4]))),
            float(max(V5RL_REBALANCE_DZ_FLOOR, 0.5 * float(u[5]))),
        ]

        if refresh:
            prev = self._v5rl_prev_physical.get(symbol)
            if prev is not None:
                dknob_acc.append(float(np.mean(np.abs(np.asarray(physical) - np.asarray(prev)))))
            self._v5rl_prev_physical[symbol] = list(physical)

        cfg = self.config[symbol]
        cfg.mp_alloc_bias = physical[0]
        cfg.mp_alloc_scale = physical[1]
        cfg.mp_confidence = physical[2]
        cfg.mp_urgency = physical[3]
        cfg.mp_signal_deadzone = physical[4]
        cfg.mp_rebalance_deadzone = physical[5]

    if dknob_acc:
        self._v5rl_dknob_l1 = float(np.mean(dknob_acc))

    triples = v5mp_decision_mp_config(self)
    v5_decision_rl_config(self, triples)


def _v5rl_knob_obs_vector(self):
    """Aktualne knoby znormalizowane do ~[-1, 1] na pełnym zakresie fizycznym."""
    import numpy as np

    symbol = self.symbols[0] if self.symbols else None
    if symbol is None:
        return np.zeros(6, dtype=np.float32)
    cfg = self.config.get(symbol)
    if cfg is None:
        return np.zeros(6, dtype=np.float32)
    vals = [
        float(getattr(cfg, "mp_alloc_bias", 0.5)),
        float(getattr(cfg, "mp_alloc_scale", 0.5)),
        float(getattr(cfg, "mp_confidence", 0.5)),
        float(getattr(cfg, "mp_urgency", 0.5)),
        float(getattr(cfg, "mp_signal_deadzone", 0.15)),
        float(getattr(cfg, "mp_rebalance_deadzone", 0.25)),
    ]
    out = [
        float(np.clip(2.0 * vals[0] - 1.0, -1.0, 1.0)),
        float(np.clip(2.0 * vals[1] - 1.0, -1.0, 1.0)),
        float(np.clip(2.0 * vals[2] - 1.0, -1.0, 1.0)),
        float(np.clip(2.0 * vals[3] - 1.0, -1.0, 1.0)),
        float(np.clip(2.0 * (vals[4] / 0.5) - 1.0, -1.0, 1.0)),
        float(np.clip(2.0 * (vals[5] / 0.5) - 1.0, -1.0, 1.0)),
    ]
    return np.asarray(out, dtype=np.float32)


def v5rl_decision_build_obs(self, t):
    """
    v5 obs + knoby t−1 (6) + rolling tx density (1).
    Global: 3 (v5) + 1 tx + 6 knobs = 10; per symbol: 13 (jak v5).
    """
    import numpy as np

    base = v5_decision_build_obs(self, t)
    # base = [cash, port_ret5, dd] + symbol feats...
    n_sym = max(len(self.symbols), 1)
    g = 3
    per = 13
    expected = g + per * n_sym
    if base.shape[0] < expected:
        base = np.pad(base, (0, expected - base.shape[0]))

    # rolling tx density (ostatnie V5RL_TX_RATE_WINDOW barów)
    tx_rate = 0.0
    logs = getattr(self.wallet, "logs", None) or []
    if logs:
        window = int(V5RL_TX_RATE_WINDOW)
        t0 = max(0, int(t) - window)
        accepted = sum(
            1 for o in logs
            if getattr(o, "status", "") == "accepted"
            and t0 <= int(getattr(o, "timestamp", -1)) <= int(t)
        )
        tx_rate = float(np.clip(accepted / float(window), 0.0, 1.0))

    knobs = _v5rl_knob_obs_vector(self)
    extra = np.concatenate([np.asarray([tx_rate], dtype=np.float32), knobs])
    # wstaw extra zaraz po global_obs v5
    obs = np.concatenate([base[:g], extra, base[g:]])
    if not np.isfinite(obs).all():
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
    return np.clip(obs, -5.0, 5.0).astype(np.float32)


def v5rl_env_calculate_reward(self):
    """
    Nagroda meta-kontrolera — prosto i głośno vs HODL.

    Stary v5 (Sharpe+DD+overtrade) dawał R≈−70k i uczył cash/HODL-shadow.
    Tu:
      1. excess kroku (port − HODL) * V5RL_EXCESS_SCALE
      2. lekka kara za tx
      3. kara za cash przy wyraźnym sygnale ML
      4. soft alignment pozycja ↔ znak sygnału
      5. terminal bonus = excess% epizodu
      6. Δknob penalty
    """
    import numpy as np

    t = int(self.t)
    n_sym = max(len(self.symbols), 1)
    eng = self.engine

    port_r = 0.0
    if eng.prev_value > 0:
        port_r = float(np.log(eng.current_value / eng.prev_value))

    hodl_r = 0.0
    for symbol in self.symbols:
        prices = eng.models[symbol].close_prices
        if t > 0 and len(prices) > t and prices[t - 1] > 0:
            hodl_r += float(np.log(prices[t] / prices[t - 1])) / n_sym

    excess = (port_r - hodl_r) * 100.0

    accepted = 0
    for o in eng.wallet.logs:
        if getattr(o, "timestamp", None) == t - 1 and getattr(o, "status", "") == "accepted":
            accepted += 1
    tx_pen = float(accepted) * float(V5RL_TX_STEP_PEN)

    tot = float(eng.current_value) or 1.0
    cash_ratio = float(eng.wallet.balance) / tot
    signal_pen = 0.0
    align = 0.0
    buy_idx = int(getattr(eng, "BUY_IDX", 0))
    sell_idx = int(getattr(eng, "SELL_IDX", 2))
    for symbol in self.symbols:
        probs = eng.current_probs.get(symbol, [0.33, 0.34, 0.33])
        buy_p = float(probs[buy_idx]) if len(probs) > buy_idx else 0.33
        sell_p = float(probs[sell_idx]) if len(probs) > sell_idx else 0.33
        sig = buy_p - sell_p
        abs_sig = abs(sig)
        cash_thr = float(V5RL_CASH_RATIO_THR)
        sig_thr = float(V5RL_SIGNAL_ABS_THR)
        if abs_sig > sig_thr and cash_ratio > cash_thr:
            signal_pen += float(V5RL_CASH_PEN) * abs_sig * (cash_ratio - cash_thr)

        holding = float(eng.wallet.holdings.get(symbol, 0.0))
        price = float(eng.current_prices.get(symbol, 0.0))
        pos = (holding * price / tot) if tot > 0 else 0.0
        # long przy buy_signal / short-cash przy sell → dodatnie
        align += float(V5RL_ALIGN_SCALE) * sig * (pos - 0.35)

    reward = (
        float(V5RL_EXCESS_SCALE) * excess
        - tx_pen
        - signal_pen
        + align
    )

    dknob = float(getattr(eng, "_v5rl_dknob_l1", 0.0) or 0.0)
    if dknob > 0.0:
        reward -= float(V5RL_DKNOB_PENALTY) * dknob
        eng._v5rl_dknob_l1 = 0.0

    # terminal: excess % całego epizodu
    ep_len = max(int(getattr(self, "episode_length", 1)), 1)
    t_start = int(getattr(self, "t_start", 0))
    near_end = (t - t_start) >= (ep_len - 1) or t >= int(getattr(self, "end_step", t + 1)) - 1
    if near_end:
        start_bal = float(getattr(self, "initial_balance", 1000.0) or 1000.0)
        strat_pct = (float(eng.current_value) / start_bal - 1.0) * 100.0
        hodl_pcts = []
        for symbol in self.symbols:
            prices = eng.models[symbol].close_prices
            if t_start < len(prices) and t < len(prices) and prices[t_start] > 0:
                hodl_pcts.append((float(prices[t]) / float(prices[t_start]) - 1.0) * 100.0)
        hodl_pct = float(np.mean(hodl_pcts)) if hodl_pcts else 0.0
        reward += float(V5RL_TERMINAL_EXCESS) * (strat_pct - hodl_pct)

    return float(np.clip(reward, -10.0, 10.0))


def v5rl_env_reset(self, seed=None, options=None):
    """v5 reset + czyszczenie stanu hold/Δknob/trade-lock meta-kontrolera."""
    obs_info = v5_env_reset(self, seed=seed, options=options)
    eng = self.engine
    eng._v5rl_held_u = {}
    eng._v5rl_hold_until = {}
    eng._v5rl_prev_physical = {}
    eng._v5rl_dknob_l1 = 0.0
    eng._v5rl_trade_lock_until = {}
    hold = getattr(self, "v5rl_knob_hold_bars", None)
    if hold is None:
        hold = getattr(getattr(eng, "core_config", None), "v5rl_knob_hold_bars", None)
    eng._v5rl_hold_bars = int(hold if hold is not None else V5RL_KNOB_HOLD_BARS)
    lock = getattr(self, "v5rl_trade_lock_bars", None)
    if lock is None:
        lock = getattr(getattr(eng, "core_config", None), "v5rl_trade_lock_bars", None)
    eng._v5rl_trade_lock_bars = int(lock if lock is not None else V5RL_TRADE_LOCK_BARS)
    # seed DecisionConfig knoby na centrum v5mp (obs start)
    for symbol in self.symbols:
        cfg = eng.config.get(symbol)
        if cfg is None:
            continue
        cfg.mp_alloc_bias = float(V5MP_KNOB_CENTER[0])
        cfg.mp_alloc_scale = float(V5MP_KNOB_CENTER[1])
        cfg.mp_confidence = float(V5MP_KNOB_CENTER[2])
        cfg.mp_urgency = float(V5MP_KNOB_CENTER[3])
        cfg.mp_signal_deadzone = float(V5MP_KNOB_CENTER[4])
        cfg.mp_rebalance_deadzone = float(V5MP_KNOB_CENTER[5])
    return obs_info


# ============================================================================
# V6 — jeden silnik; przełącznik = sim_mode (mp | mprl | rl | rlul)
# ============================================================================

def _v6_sim_mode(self) -> str:
    from .structures import coerce_sim_mode, coerce_experiment_pair
    raw_ver = getattr(self, "experiment_version", None) or getattr(
        getattr(self, "core_config", None), "experiment_version", "v6"
    )
    raw_mode = getattr(getattr(self, "core_config", None), "sim_mode", None)
    if not raw_mode:
        raw_mode = getattr(self, "sim_mode", None)
    _ver, mode = coerce_experiment_pair(raw_ver, raw_mode)
    mode = coerce_sim_mode(mode) or "mp"
    return mode


def _v6_is_meta(mode: str) -> bool:
    """v6+rl (=ex-mprl) i rlul → meta-knoby; mp → static."""
    return mode in ("rl", "rlul", "mprl")


def v6_decision_rl_config(self, scaled_dict):
    """Dispatch konfiguracji RL/MP dla v6 (meta-knoby rl/rlul vs ścieżka mp)."""
    mode = _v6_sim_mode(self)
    if _v6_is_meta(mode):
        return v5rl_decision_rl_config(self, scaled_dict)
    return v5_decision_rl_config(self, scaled_dict)


# Rozszerzone obs meta-kontrolera (v6 rl/rlul): cechy wielohoryzontowe + wiarygodnosc LSTM.
# False = obs bazowe (13 cech/symbol) - powrot do konfiguracji bazowej.
V6RL_EXTRA_OBS = False
V6RL_EXTRA_PER_SYMBOL = 10
# Minimalna obs (ablacja 25.10): 10 global + 4/symbol = 18 dla 2 aktywow.
# Per symbol z 13 cech bazowych: zscore(2), xgb_signal_1h(11), position_ratio(8), distance_to_target(12).
V6RL_OBS_MIN = True
V6RL_OBS_MIN_IDX = (2, 11, 8, 12)
V6RL_PRED_HORIZON = 12  # horyzont predykcji LSTM (model_config: horizon=12 barow)
_V6X_4H = 240
_V6X_24H = 1440
_V6X_7D = 10080
if V6RL_OBS_MIN:
    _V6_OBS_PER = len(V6RL_OBS_MIN_IDX)
else:
    _V6_OBS_PER = 13 + (V6RL_EXTRA_PER_SYMBOL if V6RL_EXTRA_OBS else 0)


def _v6_hit_rate(mu, close, t, window, horizon=V6RL_PRED_HORIZON):
    """Trafnosc kierunku mu vs zrealizowany zwrot, tylko z przeszlosci (s+h <= t)."""
    import numpy as np

    b = int(t) - int(horizon)
    a = max(0, b - int(window) + 1)
    if b - a < 30 or mu is None or len(mu) <= b:
        return 0.0
    m = np.asarray(mu[a:b + 1], dtype=np.float64)
    c0 = np.asarray(close[a:b + 1], dtype=np.float64)
    c1 = np.asarray(close[a + horizon:b + horizon + 1], dtype=np.float64)
    if c0.shape != c1.shape or c0.size == 0:
        return 0.0
    r = np.log(np.maximum(c1, 1e-12) / np.maximum(c0, 1e-12))
    hit = float(np.mean(np.sign(m) == np.sign(r))) - 0.5
    return float(np.clip(hit * 4.0, -2.0, 2.0))


_V6X_CACHE = {}


def _v6x_arrays(model):
    """Skumulowane tablice per model (1x; klucz: id+dlugosc). Wszystko prefiksowe => przyczynowe."""
    import numpy as np

    close_raw = model.close
    mu_raw = getattr(model, "xgb_mu", None)
    key = (id(close_raw), len(close_raw), id(mu_raw), 0 if mu_raw is None else len(mu_raw))
    hit = _V6X_CACHE.get(id(model))
    if hit is not None and hit[0] == key:
        return hit[1]

    close = np.asarray(close_raw, dtype=np.float64)
    n = close.shape[0]
    logc = np.log(np.maximum(close, 1e-12))
    r1 = np.zeros(n)
    r1[1:] = np.diff(logc)
    cs1 = np.concatenate([[0.0], np.cumsum(r1)])
    cs2 = np.concatenate([[0.0], np.cumsum(r1 * r1)])
    try:
        import pandas as pd
        mx7 = pd.Series(close).rolling(_V6X_7D + 1, min_periods=1).max().to_numpy()
    except Exception:
        mx7 = np.array([close[max(0, i - _V6X_7D): i + 1].max() for i in range(n)])

    cmu = None
    chit = None
    if mu_raw is not None and len(mu_raw) > 0:
        mu = np.asarray(mu_raw, dtype=np.float64)
        m = min(mu.shape[0], n)
        mu = mu[:m]
        cmu = np.concatenate([[0.0], np.cumsum(mu)])
        h = V6RL_PRED_HORIZON
        hitv = np.zeros(m)
        k = max(0, min(m, n - h))
        if k > 0:
            fut = logc[h:h + k] - logc[:k]
            hitv[:k] = (np.sign(mu[:k]) == np.sign(fut)).astype(np.float64)
        chit = np.concatenate([[0.0], np.cumsum(hitv)])
    arrs = dict(n=n, logc=logc, cs1=cs1, cs2=cs2, mx7=mx7, close=close, cmu=cmu, chit=chit)
    _V6X_CACHE[id(model)] = (key, arrs)
    return arrs


def v6rl_extra_features(self, t):
    """V6RL_EXTRA_PER_SYMBOL cech na symbol, wszystkie przyczynowe (dane <= t), O(1)/krok."""
    import numpy as np

    t = int(t)
    out = []
    buy_idx = int(getattr(self, "BUY_IDX", 0))
    sell_idx = int(getattr(self, "SELL_IDX", 2))
    h = V6RL_PRED_HORIZON
    for symbol in self.symbols:
        model = self.models[symbol]
        A = _v6x_arrays(model)
        n = A["n"]
        tt = min(max(t, 0), n - 1)
        probs = self.current_probs[symbol] if symbol in self.current_probs else [0.33, 0.34, 0.33]
        p_buy = float(probs[buy_idx]) if len(probs) > buy_idx else 0.0
        p_sell = float(probs[sell_idx]) if len(probs) > sell_idx else 0.0

        cmu, chit = A["cmu"], A["chit"]

        def _mu_mean(w):
            if cmu is None or tt < 1 or tt + 1 >= len(cmu):
                return 0.0
            lo = max(0, tt - w + 1)
            return float(np.clip((cmu[tt + 1] - cmu[lo]) / (tt + 1 - lo) * 100.0, -5.0, 5.0))

        def _hit(w):
            if chit is None:
                return 0.0
            b = tt - h
            a = max(0, b - w + 1)
            if b - a < 30 or b + 1 >= len(chit):
                return 0.0
            rate = (chit[b + 1] - chit[a]) / float(b - a + 1) - 0.5
            return float(np.clip(rate * 4.0, -2.0, 2.0))

        def _ret(w):
            if tt < w:
                return 0.0
            return float((A["logc"][tt] - A["logc"][tt - w]) * 100.0)

        def _std(lo, hi):  # std zwrotow 1-baro r1[lo+1..hi]
            cnt = hi - lo
            if cnt < 2:
                return 0.0
            s1 = A["cs1"][hi + 1] - A["cs1"][lo + 1]
            s2 = A["cs2"][hi + 1] - A["cs2"][lo + 1]
            var = s2 / cnt - (s1 / cnt) ** 2
            return float(np.sqrt(max(var, 0.0)))

        ret_4h = float(np.tanh(_ret(_V6X_4H) / 1.5))
        ret_24h = float(np.tanh(_ret(_V6X_24H) / 3.0))

        vol_ratio = 0.0
        if tt >= _V6X_24H + 1:
            vd = _std(tt - _V6X_24H, tt)
            vw = _std(max(0, tt - _V6X_7D), tt)
            vol_ratio = float(np.tanh(vd / (vw + 1e-8) - 1.0))

        peak = float(A["mx7"][tt]) if tt >= 1 else 0.0
        dd7 = 0.0
        if peak > 0:
            dd7 = float(np.tanh((float(A["close"][tt]) / peak - 1.0) * 100.0 / 5.0))

        out.extend([
            p_buy, p_sell,
            _mu_mean(_V6X_4H), _mu_mean(_V6X_24H),
            _hit(_V6X_4H), _hit(_V6X_24H),
            ret_4h, ret_24h, vol_ratio, dd7,
        ])
    return np.asarray(out, dtype=np.float32)


def v6_decision_build_obs(self, t):
    """Obserwacja v6: meta (v5rl ± redukcja cech) albo klasyczna ścieżka v5."""
    import numpy as np

    mode = _v6_sim_mode(self)
    if _v6_is_meta(mode):
        base = v5rl_decision_build_obs(self, t)
        if V6RL_OBS_MIN:
            n_sym = max(len(self.symbols), 1)
            g = 10
            parts = [base[:g]]
            for i in range(n_sym):
                s = base[g + 13 * i: g + 13 * (i + 1)]
                parts.append(s[list(V6RL_OBS_MIN_IDX)])
            return np.concatenate(parts).astype(np.float32)
        if not V6RL_EXTRA_OBS:
            return base
        obs = np.concatenate([base, v6rl_extra_features(self, t)])
        if not np.isfinite(obs).all():
            obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        return np.clip(obs, -5.0, 5.0).astype(np.float32)
    return v5_decision_build_obs(self, t)


def v6_env_calculate_reward(self):
    """Nagroda v6: delegacja do v5rl (meta) lub v5."""
    mode = _v6_sim_mode(self)
    if _v6_is_meta(mode):
        return v5rl_env_calculate_reward(self)
    return v5_env_calculate_reward(self)


def v6_env_reset(self, seed=None, options=None):
    """Reset środowiska v6: delegacja do v5rl (meta) lub v5."""
    mode = _v6_sim_mode(self)
    if _v6_is_meta(mode):
        return v5rl_env_reset(self, seed=seed, options=options)
    return v5_env_reset(self, seed=seed, options=options)


# ============================================================================
# DECISION / ENV — V5 (build_obs + reward + reset)
# ============================================================================
def v5_decision_build_obs(self, t):
    """Obserwacja v5: cechy globalne portfela oraz cechy per symbol (sygnał, pozycja)."""
    import numpy as np

    current_portfolio_value = self.wallet.total_value(self.current_prices)
    cash_ratio = (self.wallet.balance / current_portfolio_value
                  if current_portfolio_value > 0 else 1.0)

    hist = self.wallet.portfolio_values_history

    # FIX 1: nie twórz nowej listy – porównaj osobno
    portfolio_return_5 = 0.0
    if len(hist) >= 5 and hist[-5] > 0:
        portfolio_return_5 = np.log(current_portfolio_value / hist[-5]) * 100.0
    elif len(hist) >= 1 and hist[-1] > 0:
        portfolio_return_5 = np.log(current_portfolio_value / hist[-1]) * 100.0

    # FIX 1: peak trzymaj jako atrybut, nie licz z całej listy każdy krok
    # (wymaga ustawienia self._peak_value w reset – już to masz w v5_env_reset)
    drawdown = 0.0
    if hasattr(self, '_peak_value') and self._peak_value > 0:
        if current_portfolio_value > self._peak_value:
            self._peak_value = current_portfolio_value
        dd = (current_portfolio_value - self._peak_value) / self._peak_value
        drawdown = float(np.tanh(dd * 10.0))
    elif hist:
        # fallback przy braku _peak_value
        peak = max(hist)  # bez concat
        peak = max(peak, current_portfolio_value)
        if peak > 0:
            drawdown = float(np.tanh((current_portfolio_value - peak) / peak * 10.0))

    global_obs = np.array([
        float(cash_ratio),
        float(np.clip(portfolio_return_5, -5.0, 5.0)),
        float(drawdown),
    ], dtype=np.float32)

    obs_parts = [global_obs]

    for symbol in self.symbols:
        model = self.models[symbol]
        close = model.close

        mu    = float(model.xgb_mu[t])    if (model.xgb_mu    is not None and len(model.xgb_mu)    > t) else 0.0
        sigma = float(model.xgb_sigma[t]) if (model.xgb_sigma is not None and len(model.xgb_sigma) > t) else 0.0
        xgb_zscore = float(np.tanh(mu / (sigma + 1e-8)))

        price_t = close[t]
        momentum_1  = np.log(price_t / close[t-1])  * 100.0 if t > 0  and close[t-1]  > 0 else 0.0
        momentum_5  = np.log(price_t / close[t-5])  * 100.0 if t >= 5 and close[t-5]  > 0 else 0.0
        momentum_20 = np.log(price_t / close[t-20]) * 100.0 if t >= 20 and close[t-20] > 0 else 0.0

        # FIX 2: numpy slice zamiast list comprehension
        vol_regime = 0.0
        if t >= VOL_LONG_V5:
            # close jest numpy array – slice jest O(1), std na array jest szybkie
            rets_short = np.diff(np.log(close[t - VOL_SHORT_V5: t + 1]))
            rets_long  = np.diff(np.log(close[t - VOL_LONG_V5:  t + 1]))
            vs = float(np.std(rets_short)) if len(rets_short) > 1 else 1e-8
            vl = float(np.std(rets_long))  if len(rets_long)  > 1 else 1e-8
            vol_regime = float(np.tanh(vs / (vl + 1e-8) - 1.0))

        # FIX 2: numpy slice zamiast list comprehension
        rsi_proxy = 0.5
        if t >= RSI_WINDOW_V5:
            window = close[t - RSI_WINDOW_V5: t + 1]
            diffs = np.diff(window)
            up_moves = int(np.sum(diffs > 0))
            rsi_proxy = float(up_moves / len(diffs)) if len(diffs) > 0 else 0.5

        holding_amount = float(self.wallet.holdings.get(symbol, 0.0))
        holding_value  = holding_amount * float(self.current_prices[symbol])
        position_ratio = holding_value / current_portfolio_value if current_portfolio_value > 0 else 0.0

        avg_buy_price  = self.wallet.avg_buy_price.get(symbol, 0.0)
        unrealized_pnl = 0.0
        if holding_amount > 0 and avg_buy_price > 0:
            unrealized_pnl = float(np.tanh(
                (self.current_prices[symbol] - avg_buy_price) / avg_buy_price * 5.0
            ))

        time_in = float(getattr(self, '_time_in_position', {}).get(symbol, 0))
        time_in_norm = float(np.tanh(time_in / 200.0))

        xgb_signal_1h = 0.0
        lookback_mu = min(SIGNAL_LOOKBACK_V5, t)
        if lookback_mu > 0 and model.xgb_mu is not None and len(model.xgb_mu) > t:
            # FIX 2: numpy slice zamiast list comprehension
            xgb_signal_1h = float(np.mean(model.xgb_mu[t - lookback_mu: t]) * 100.0)

        target = getattr(self, "_prev_target", {}).get(symbol, position_ratio)
        distance_to_target = float(target - position_ratio)

        symbol_obs = np.array([
            mu, sigma, xgb_zscore,
            float(np.clip(momentum_1,  -5.0, 5.0)),
            float(np.clip(momentum_5,  -5.0, 5.0)),
            float(np.clip(momentum_20, -5.0, 5.0)),
            vol_regime, rsi_proxy, position_ratio, unrealized_pnl,
            time_in_norm,
            float(np.clip(xgb_signal_1h, -5.0, 5.0)),
            float(np.clip(distance_to_target, -1.0, 1.0)),
        ], dtype=np.float32)
        obs_parts.append(symbol_obs)

    obs = np.concatenate(obs_parts)
    if not np.isfinite(obs).all():
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
    return np.clip(obs, -5.0, 5.0)


# ============================================================================
# ENV METHODS — V5
# ============================================================================

def v5_env_calculate_reward(self):
    """
    Nagroda v5 — Sharpe-inspired + drawdown penalty + curriculum.

    Składowe:
      1. SHARPE PROXY       — alpha / rolling_std_20(portfolio_returns).
                              Nagradza stabilne zyski, karze nerwowe ruchy.
                              Gdy historia zbyt krótka -> czyste alpha.
      2. DRAWDOWN PENALTY   — kara liniowa proporcjonalna do drawdownu.
                              lambda skaluje się z curriculum (duże na starcie).
      3. EFFICIENCY BONUS   — +bonus gdy transakcja i unrealized_pnl > 0.
                              Nagradza wchodzenie w zyski, nie samo handlowanie.
      4. OVERTRADING PENALTY— kara za każdą transakcję = OVERTRADE_COST * 100.
                              Zmusza agenta do liczyć się z kosztami.
      5. CONCENTRATION PEN  — kara gdy > CONCENTRATION_THR portfela w jednym symbolu.

    Curriculum (liniowe skalowanie lambdy):
      - 0-30% epizodu:    lambda_sharpe=0.5, lambda_draw=2.0  -> "nie trać!"
      - 30-70% epizodu:   przejście
      - 70-100% epizodu:  lambda_sharpe=2.0, lambda_draw=0.5  -> "zarabiaj!"

    Usage: V5
    """
    import numpy as np

    t = self.t
    n_sym = max(len(self.symbols), 1)

    # -- 1. Alpha: portfolio vs equal-weight benchmark ------------------------
    portfolio_return = 0.0
    if self.engine.prev_value > 0:
        portfolio_return = np.log(self.engine.current_value / self.engine.prev_value)

    benchmark_return = 0.0
    for symbol in self.symbols:
        prices = self.engine.models[symbol].close_prices
        if t > 0 and len(prices) > t and prices[t-1] > 0:
            benchmark_return += np.log(prices[t] / prices[t-1]) / n_sym

    alpha = (portfolio_return - benchmark_return) * 100.0

    # -- 2. Sharpe proxy: alpha / rolling_std ---------------------------------
    self._reward_history.append(portfolio_return * 100.0)
    if len(self._reward_history) > 20:
        self._reward_history.pop(0)

    if len(self._reward_history) >= 5:
        rolling_std = float(np.std(self._reward_history))
        sharpe_proxy = alpha / (rolling_std + 0.1)  # +0.1 żeby nie dzielić przez 0
    else:
        sharpe_proxy = alpha

    sharpe_proxy = float(np.clip(sharpe_proxy, -10.0, 10.0))

    # -- 3. Drawdown penalty --------------------------------------------------
    current_value = self.engine.current_value
    if current_value > self._peak_value:
        self._peak_value = current_value

    drawdown = 0.0
    if self._peak_value > 0 and current_value < self._peak_value:
        drawdown = (self._peak_value - current_value) / self._peak_value

    drawdown_pen = float(drawdown * 100.0)   # 10% drawdown = -10 nagrody

    # -- 4. Curriculum: skalowanie lambd --------------------------------------
    episode_progress = float(t - self.t_start) / max(self.episode_length, 1)
    episode_progress = float(np.clip(episode_progress, 0.0, 1.0))

    if episode_progress <= CURRICULUM_RAMPUP:
        phase = 0.0
    elif episode_progress >= CURRICULUM_RAMPDOWN:
        phase = 1.0
    else:
        phase = (episode_progress - CURRICULUM_RAMPUP) / (CURRICULUM_RAMPDOWN - CURRICULUM_RAMPUP)

    lambda_sharpe = LAMBDA_SHARPE_START + phase * (LAMBDA_SHARPE_END - LAMBDA_SHARPE_START)
    lambda_draw   = LAMBDA_DRAW_START   + phase * (LAMBDA_DRAW_END   - LAMBDA_DRAW_START)

    # -- 5 & 6: jeden scan zamiast dwóch -----------------------------------------
    efficiency_bonus = 0.0
    accepted_this_step = 0
    for o in self.engine.wallet.logs:
        if o.timestamp == t - 1:
            if o.status == "accepted":
                accepted_this_step += 1
                if o.action == "sell":
                    avg_p = self.engine.wallet.avg_buy_price.get(o.symbol, 0.0)
                    curr_p = self.engine.current_prices.get(o.symbol, 0.0)
                    if avg_p > 0 and curr_p > avg_p:
                        efficiency_bonus += float(np.tanh((curr_p - avg_p) / avg_p * 20.0)) * 2.0
    overtrading_pen = float(accepted_this_step) * OVERTRADE_COST * 100.0

    # -- 7. Concentration penalty ---------------------------------------------
    concentration_pen = 0.0
    if len(self.symbols) > 1:
        total_val = self.engine.wallet.total_value(self.engine.current_prices)
        for symbol in self.symbols:
            holding_val = (
                self.engine.wallet.holdings.get(symbol, 0.0) *
                self.engine.current_prices.get(symbol, 0.0)
            )
            if total_val > 0:
                ratio = holding_val / total_val
                if ratio > CONCENTRATION_THR:
                    excess = ratio - CONCENTRATION_THR
                    concentration_pen += float(excess / (1.0 - CONCENTRATION_THR)) * CONCENTRATION_PEN

    # -- Składamy nagrodę -----------------------------------------------------
    reward = (
          lambda_sharpe * sharpe_proxy
        - lambda_draw   * drawdown_pen
        + efficiency_bonus
        - overtrading_pen
        - concentration_pen
    )

    # Aktualizuj time_in_position po każdym kroku
    for symbol in self.symbols:
        if self.engine.wallet.holdings.get(symbol, 0.0) > 0:
            self.engine._time_in_position[symbol] = (
                self.engine._time_in_position.get(symbol, 0) + 1
            )
        else:
            self.engine._time_in_position[symbol] = 0

    return float(np.clip(reward, -20.0, 20.0))


def v5_env_reset(self, seed=None, options=None):
    """
    Reset dla v5.

    Inicjuje:
      - idle_steps, t_start (jak v3/v4)
      - _time_in_position per symbol
      - _reward_history dla rolling Sharpe
      - _peak_value dla drawdown
      - losowy start z marginesem (TARGET_HORIZON + SIGNAL_LOOKBACK + VOL_LONG)

    Usage: V5
    """
    import numpy as np

    super(self.__class__, self).reset(seed=seed)

    margin = TARGET_HORIZON_V5 + SIGNAL_LOOKBACK_V5 + VOL_LONG_V5 + 5
    max_start_index = self.max_steps - self.episode_length - margin
    # --- STARY PODZIAŁ: każdy env losuje własny start (duży spread HODL w dumpie) ---
    # if max_start_index > 0:
    #     self.t = self.np_random.integers(0, max_start_index)
    # else:
    #     self.t = margin
    # --- SYNC: wszystkie envy w tej samej fali → ten sam t0 ---
    self.t = pick_episode_start_t(self, max_start_index, fallback=margin)

    self.end_step  = self.t + self.episode_length
    self.t_start   = self.t

    self.prepare_callback()
    self.engine.wallet.reset(self.initial_balance)
    self.engine.reset()

    self.last_reward = 0
    self.rewards     = []
    self.idle_steps  = 0

    # V5-specific state
    self.engine._time_in_position = {symbol: 0 for symbol in self.symbols}
    self._reward_history          = []
    self._peak_value              = self.initial_balance

    self.engine.t_data(self.t)
    if hasattr(self, "_begin_episode_trackers"):
        self._begin_episode_trackers()
    return self._get_obs(), {}


# ============================================================================
# REJESTRACJA EKSPERYMENTU V5
# ============================================================================

# V5_EXPERIMENT = {
#     "description": (
#         "Risk-adjusted portfolio management. "
#         "Agent wskazuje docelową alokację portfela (position_target) "
#         "zamiast buy/sell/hold. "
#         "Nagroda Sharpe-inspired z drawdown penalty i curriculum. "
#         "3 params per symbol: [position_target, confidence, urgency]. "
#         "Obs: 3 globalne + 13 per symbol."
#     ),
#     "n_params_per_symbol": 3,
#     "obs_len_global": 3,
#     "obs_len_per_symbol": 13,
#
#     "decision_methods": {
#         "rl_config":          v5_decision_rl_config,
#         "calculate_fraction": v5_decision_calculate_fraction,
#         "build_obs":          v5_decision_build_obs,
#     },
#     "wallet_methods": {},
#     "env_methods": {
#         "_calculate_reward": v5_env_calculate_reward,
#         "reset":             v5_env_reset,
#     },
# }

# ============================================================================
# JAK DODAĆ DO version_registry.py:
#
# 1. Skopiuj stałe i funkcje z tego pliku do version_registry.py
#    (przed słownikiem EXPERIMENTS).
#
# 2. W słowniku EXPERIMENTS dodaj:
#
#    "v5": V5_EXPERIMENT,
#
# 3. Opcjonalne: zaktualizuj hiperparametry treningu:
#
#    train_agent_ppo(
#        ...,
#        experiment_version  = "v5",
#        episode_timesteps   = 2048,   # dłuższe epizody -> lepszy Sharpe
#        total_timesteps     = 5_000_000,
#        n_steps             = 4096,
#        batch_size          = 256,
#        ent_coef            = 0.005,  # mniejszy entropy -> agent mniej losowy
#        gamma               = 0.995,  # wyższe gamma -> myśli długoterminowo
#    )
#
# 4. Rozmiar obserwacji:
#    obs_size = 3 + 13 * n_symbols
#    Dla 1 symbolu:  16
#    Dla 3 symboli:  42
#    Dla 5 symboli:  68
#
# 5. Rozmiar akcji:
#    action_size = 3 * n_symbols
#    Dla 1 symbolu:  3
#    Dla 3 symboli:  9
#    Dla 5 symboli:  15
# ============================================================================



# ============================================================================
# EXPERIMENT REGISTRY
# ============================================================================

EXPERIMENTS = {
    "v0": {
        "description": "Podstawowa wersja core_logic",
        "n_params_per_symbol": 4,

        "decision_methods": {},

        "wallet_methods": {},

        "env_methods": {}
    },
    "v1": {
        "description": "Fraction liczone przez agenta",
        "n_params_per_symbol": 2,

        "decision_methods": {
            "calculate_fraction": decision_calculate_fraction,
            "rl_config": v1_decision_rl_config,
        },

        "wallet_methods": {},

        "env_methods": {}
    },
    "v2": {
        "description": "Agent liczy sygnały, thresholdy pozostają statyczne",  # TODO:
        "n_params_per_symbol": 2,

        "decision_methods": {
            "calculate_fraction": decision_calculate_fraction,
            "rl_config": v1_decision_rl_config,
        },

        "wallet_methods": {},

        "env_methods": {}
    },
    "v3": {
        "description": (
            "Agent bezpośrednio zwraca buy/sell value [0,1] zamiast progów. "
            "Progi threshold_buy/sell stałe = 0.5 (z DecisionConfig default). "
            "Obserwacja: xgb_mu + xgb_sigma zamiast buy_p/sell_p z ML. "
            "Fraction sterowane przez agenta (jak v1). "
            "4 params per symbol: [buy_value, sell_value, fraction_buy, fraction_sell]."
        ),
        "n_params_per_symbol": 4,
        "obs_len_for_symbol": 6,
        "decision_methods": {
            "rl_config":        v3_decision_rl_config,
            "calculate_fraction": decision_calculate_fraction,
            "build_obs":        v3_decision_build_obs,
        },

        "wallet_methods": {},

        "env_methods": {
            "_calculate_reward": v3_env_calculate_reward,
            "reset":             v3_env_reset,
        }
    },
    "v4": {
            "description": (
                "Target-shaped reward: alignment_bonus nagradza gdy akcja agenta "
                "zgadza się z faktycznym ruchem ceny za TARGET_HORIZON kroków. "
                "Target NIE jest w obserwacji — agent musi sam znaleźć pattern. "
                "Curriculum: lambda_align maleje przez epizod (0.8 → 0.2). "
                "Rozszerzona obs: mu, sigma, xgb_zscore, momentum_1/5, trend_20, "
                "vol_ratio, position_ratio, unrealized_pnl, steps_since_trade. "
                "4 params per symbol: [buy_value, sell_value, fraction_buy, fraction_sell]."
            ),
            "n_params_per_symbol": 4,
            "obs_len_for_symbol": 10,   # 9 featury per symbol + 1 globalny (cash_ratio)
            "decision_methods": {
                "rl_config":          v4_decision_rl_config,
                "calculate_fraction": decision_calculate_fraction,
                "build_obs":          v4_decision_build_obs,
            },
            "wallet_methods": {},
            "env_methods": {
                "_calculate_reward": v4_env_calculate_reward,
                "reset":             v4_env_reset,
            }
        },
    "v5":{
        "description": (
            "Risk-adjusted portfolio management. "
            "Agent wskazuje docelową alokację portfela (position_target) "
            "zamiast buy/sell/hold. "
            "Nagroda Sharpe-inspired z drawdown penalty i curriculum. "
            "3 params per symbol: [position_target, confidence, urgency]. "
            "Obs: 3 globalne + 13 per symbol."
        ),
        "n_params_per_symbol": 3,
        "obs_len_global": 3,
        "obs_len_for_symbol": 13,

        "decision_methods": {
            "rl_config": v5_decision_rl_config,
            "calculate_fraction": v5_decision_calculate_fraction,
            "build_obs": v5_decision_build_obs,
        },
        "wallet_methods": {},
        "env_methods": {
            "_calculate_reward": v5_env_calculate_reward,
            "reset": v5_env_reset,
        },
    },
    # Locked ML path — LEGACY; używaj v6 + sim_mode=mp
    "v5mp": {
        "description": (
            "LEGACY alias → v6+mp. Locked MP path. v5 allocation + MP mapping. "
            "signal→target with dead-zone, confidence from |signal|, slow EMA, "
            "larger rebalance dead-zone. Risk: stop_loss (+ cooldown). "
            "Same rebalance engine as v5. Knobs static from CoreConfig/Optuna."
        ),
        "n_params_per_symbol": 3,
        "obs_len_global": 3,
        "obs_len_for_symbol": 13,
        "decision_methods": {
            "rl_config": v5_decision_rl_config,
            "mp_config": v5mp_decision_mp_config,
            "ml_config": v5mp_decision_mp_config,  # legacy alias
            "calculate_fraction": v5_decision_calculate_fraction,
            "build_obs": v5_decision_build_obs,
        },
        "wallet_methods": {},
        "env_methods": {
            "_calculate_reward": v5_env_calculate_reward,
            "reset": v5_env_reset,
        },
    },
    # Meta-kontroler: RL zarządza knoby mapowania MP w czasie rzeczywistym
    # LEGACY — używaj v6 + sim_mode=rl (mprl to alias → rl).
    "v5rl": {
        "description": (
            "LEGACY alias → v6+rl. Meta-controller on locked MP mapper. "
            "RL action = 6 mapping knobs over full physical range "
            "(bias/scale/conf/urg ∈[0,1], deadzones ∈[0,0.5]); "
            "knobs every 1 bar; after accepted tx portfolio locked "
            "~15 bars (stop_loss bypasses); obs adds prev knobs + tx density; "
            "same v5 rebalance + Δknob penalty."
        ),
        "n_params_per_symbol": 6,
        # global: cash, port_ret5, dd, tx_rate, 6×knob_norm = 10
        "obs_len_global": 10,
        "obs_len_for_symbol": 13,
        "decision_methods": {
            "rl_config": v5rl_decision_rl_config,
            "mp_config": v5mp_decision_mp_config,
            "ml_config": v5mp_decision_mp_config,  # legacy alias
            "calculate_fraction": v5_decision_calculate_fraction,
            "build_obs": v5rl_decision_build_obs,
        },
        "wallet_methods": {},
        "env_methods": {
            "_calculate_reward": v5rl_env_calculate_reward,
            "reset": v5rl_env_reset,
        },
    },
    # v6 = silnik alokacji. Przełącznik: sim_mode.
    #   mp   → stałe knoby MP
    #   rl   → meta-RL nad knoby MP (ex-mprl / ex-v5rl)
    #   mprl → legacy alias → rl (coerce)
    #   rlul → jak rl + UL (stub; na razie = rl)
    # Klasyczny position_target: tylko experiment_version=v5 (nie v6+rl).
    "v6": {
        "description": (
            "Unified allocation engine. "
            "Switch with sim_mode: mp | rl | rlul (mprl=legacy→rl). "
            "mp=static MP knobs; rl=meta-RL over knobs; rlul=reserved(+UL)."
        ),
        "n_params_per_symbol": 6,
        "obs_len_global": 10,
        "obs_len_for_symbol": _V6_OBS_PER,
        "by_sim_mode": {
            "mp": {
                "n_params_per_symbol": 3,
                "obs_len_global": 3,
                "obs_len_for_symbol": 13,
            },
            "rl": {
                "n_params_per_symbol": 6,
                "obs_len_global": 10,
                "obs_len_for_symbol": _V6_OBS_PER,
            },
            "mprl": {
                "n_params_per_symbol": 6,
                "obs_len_global": 10,
                "obs_len_for_symbol": _V6_OBS_PER,
            },
            "rlul": {
                "n_params_per_symbol": 6,
                "obs_len_global": 10,
                "obs_len_for_symbol": _V6_OBS_PER,
            },
        },
        "decision_methods": {
            "rl_config": v6_decision_rl_config,
            "mp_config": v5mp_decision_mp_config,
            "ml_config": v5mp_decision_mp_config,
            "calculate_fraction": v5_decision_calculate_fraction,
            "build_obs": v6_decision_build_obs,
        },
        "wallet_methods": {},
        "env_methods": {
            "_calculate_reward": v6_env_calculate_reward,
            "reset": v6_env_reset,
        },
    },
    "testnet_v0": {
        "description": "Lokalna logika + Binance Testnet market orders z reconciliation i pending tracking.",
        "n_params_per_symbol": 4,
        "decision_methods": {},
        "wallet_methods": {
            "buy": testnet_wallet_buy,
            "sell": testnet_wallet_sell,
            "testnet_wallet_init": testnet_wallet_init,
            "testnet_check_pending_orders": testnet_check_pending_orders,
            "testnet_sync_after_orders": testnet_sync_after_orders,
        },
        "env_methods": {}
    },
}

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================