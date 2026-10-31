"""
Universal Feature Engineering for Cryptocurrency OHLCV Data
============================================================
Supports:
  - LSTM price prediction  (default, backward-compatible)
  - LSTM log-return prediction  (future)
  - XGBoost log-return prediction
  - Model 3  (placeholder — extend get_feature_sets() when ready)

Feature sets
------------
Universal (shared across models):
  full          – all generated features
  technical     – universal optimum (no model-specific features)
  technical_bp  – technical + buy_pressure (requires taker_buy column)
  minimal       – smallest universal subset

LSTM-dedicated:
  lstm_momentum
  lstm_volatility
  lstm_price     – price-specific (SMA ratios, BB — suited for price target)

XGBoost-dedicated:
  xgb_momentum
  xgb_volatility
  xgb_returns    – pure return-based (no price-level features)

Placeholders (add your own features):
  k_set1
  k_set2
  k_set3
"""

import numpy as np
import pandas as pd


# ============================================================================
# CORE FEATURE GENERATION
# ============================================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all technical features from raw OHLCV data.

    Args:
        df: DataFrame with columns: open_time, open, high, low, close, volume
            Optionally: taker_buy_base_asset_volume  (for buy_pressure)

    Returns:
        DataFrame with additional feature columns appended.
        Original columns are preserved unchanged.
    """
    df = df.copy()

    # -------------------------------------------------------------------------
    # LOG-RETURNS  (multiple timeframes)
    # -------------------------------------------------------------------------
    for window in [1, 5, 15, 60]:
        df[f'return_{window}'] = np.log(df['close'] / df['close'].shift(window))

    # -------------------------------------------------------------------------
    # SMA RATIOS  (close / SMA — normalised price position)
    # -------------------------------------------------------------------------
    for window in [7, 14, 30]:
        sma = df['close'].rolling(window).mean()
        df[f'close_sma_{window}'] = df['close'] / sma

    # -------------------------------------------------------------------------
    # MACD
    # -------------------------------------------------------------------------
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # -------------------------------------------------------------------------
    # RSI  (bugfix: clip loss to avoid Inf/NaN propagation when loss == 0)
    # -------------------------------------------------------------------------
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0).rolling(14).mean()).clip(lower=1e-10)
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # -------------------------------------------------------------------------
    # VOLATILITY  (rolling std of return_1)
    # -------------------------------------------------------------------------
    for window in [7, 14, 30]:
        df[f'volatility_{window}'] = df['return_1'].rolling(window).std()

    # -------------------------------------------------------------------------
    # VOLATILITY REGIME
    # z-score of 30-bar realised vol over a 500-bar rolling window.
    # Tells the model where current volatility sits on its historical
    # distribution — value 0.0 = median regime, +2.0 = elevated, -1.0 = calm.
    # Avoids raw magnitude differences across market regimes.
    # Note: requires ~530 bars to be non-NaN; NaN rows are dropped by callers.
    # -------------------------------------------------------------------------
    realized_vol = df['return_1'].rolling(30).std()
    vol_roll_mean = realized_vol.rolling(500).mean()
    vol_roll_std = realized_vol.rolling(500).std().clip(lower=1e-10)
    df['vol_regime'] = (realized_vol - vol_roll_mean) / vol_roll_std

    # -------------------------------------------------------------------------
    # VOLUME
    # -------------------------------------------------------------------------
    volume_sma = df['volume'].rolling(20).mean()
    df['relative_volume'] = df['volume'] / volume_sma.clip(lower=1e-10)

    # -------------------------------------------------------------------------
    # BUY PRESSURE  (order flow imbalance)
    # taker_buy_base_asset_volume / total_volume  in [0, 1].
    # One of the strongest short-term predictors in crypto microstructure.
    # Gracefully skipped when the column is absent (non-Binance sources).
    # -------------------------------------------------------------------------
    if 'taker_buy_base_asset_volume' in df.columns:
        df['buy_pressure'] = (
            df['taker_buy_base_asset_volume'] / df['volume'].clip(lower=1e-10)
        )

    # -------------------------------------------------------------------------
    # BOLLINGER BANDS
    # bugfix: clip denominator to avoid 0/0 = NaN when all 20 candles identical
    # -------------------------------------------------------------------------
    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    df['bb_position'] = (df['close'] - bb_lower) / (
        (bb_upper - bb_lower).clip(lower=1e-10)
    )

    # -------------------------------------------------------------------------
    # HIGH-LOW RANGE
    # -------------------------------------------------------------------------
    df['hl_range'] = (df['high'] - df['low']) / df['close'].clip(lower=1e-10)

    # -------------------------------------------------------------------------
    # TIME FEATURES  (cyclical encoding)
    # -------------------------------------------------------------------------
    dt = pd.to_datetime(df['open_time'])
    hour = dt.dt.hour
    dow = dt.dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    return df


# ============================================================================
# FEATURE COLUMN REGISTRY
# ============================================================================

def get_feature_columns(include_buy_pressure: bool = False) -> list:
    """
    Canonical ordered list of all generated feature names.

    Args:
        include_buy_pressure: Include 'buy_pressure' only when the source data
            contained taker_buy_base_asset_volume.  Default False keeps the
            list deterministic for callers that do not have that column.

    Returns:
        list of feature names (raw OHLCV / Binance passthrough columns excluded).
    """
    features = []

    # Returns
    features.extend([f'return_{w}' for w in [1, 5, 15, 60]])

    # Trend / momentum
    features.extend([f'close_sma_{w}' for w in [7, 14, 30]])
    features.extend(['macd', 'macd_signal', 'macd_diff', 'rsi'])

    # Volatility
    features.extend([f'volatility_{w}' for w in [7, 14, 30]])
    features.append('vol_regime')

    # Volume / microstructure
    features.append('relative_volume')
    if include_buy_pressure:
        features.append('buy_pressure')

    # Price pattern
    features.extend(['bb_position', 'hl_range'])

    # Time
    features.extend(['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'])

    return features


# ============================================================================
# FEATURE SETS
# ============================================================================

def get_feature_sets(include_buy_pressure: bool = False) -> dict:
    """
    All named feature sets.

    Universal sets are designed to work across LSTM-price, LSTM-logreturn
    and XGBoost without model-specific tuning.

    Args:
        include_buy_pressure: When True, 'buy_pressure' is included in sets
            that benefit from microstructure signals (full, technical_bp,
            xgb_momentum, lstm_momentum).
            Pass True only when raw data contains taker_buy_base_asset_volume.

    Returns:
        dict  set_name -> list[str]
    """
    all_features = get_feature_columns(include_buy_pressure=include_buy_pressure)

    # ------------------------------------------------------------------
    # UNIVERSAL SETS
    # ------------------------------------------------------------------

    # full — every generated feature (canonical reference set)
    full = all_features

    # technical — balanced universal set
    #   Excludes:
    #     • time features (hour/dow) — not always useful; add via lstm_* sets
    #     • vol_regime — 500-bar warm-up window; excluded to keep 'technical'
    #       safe for shorter prepared-data windows
    #   Includes: returns, SMA ratios, MACD, RSI, volatility, volume, BB, HL
    technical = [
        'return_1', 'return_5', 'return_15',
        'close_sma_7', 'close_sma_14', 'close_sma_30',
        'macd', 'macd_signal', 'macd_diff',
        'rsi',
        'volatility_7', 'volatility_14',
        'relative_volume',
        'bb_position',
        'hl_range',
    ]

    # technical_bp — technical + buy_pressure (order-flow imbalance)
    # Requires taker_buy_base_asset_volume in raw data (Binance full export).
    technical_bp = technical.copy()
    if include_buy_pressure:
        technical_bp.append('buy_pressure')

    # minimal — smallest viable universal subset
    minimal = [
        'return_1', 'return_5',
        'close_sma_14',
        'rsi',
        'volatility_14',
        'relative_volume',
    ]

    # ------------------------------------------------------------------
    # LSTM-DEDICATED SETS
    # ------------------------------------------------------------------

    # lstm_momentum — trend/momentum focus with intraday time patterns
    # Time features (hour/dow) help LSTM capture recurring intraday cycles
    # that matter for price-level prediction.
    lstm_momentum = [
        'return_1', 'return_5', 'return_15', 'return_60',
        'macd', 'macd_signal', 'macd_diff',
        'rsi',
        'close_sma_7', 'close_sma_14', 'close_sma_30',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    ]
    if include_buy_pressure:
        lstm_momentum.append('buy_pressure')

    # lstm_volatility — volatility regime focus with time context
    lstm_volatility = [
        'return_1', 'return_5',
        'volatility_7', 'volatility_14', 'volatility_30',
        'vol_regime',
        'bb_position',
        'hl_range',
        'relative_volume',
        'hour_sin', 'hour_cos',
    ]

    # lstm_price — features best suited for absolute-price prediction target
    # SMA ratios and BB are in price-space; they are more interpretable when
    # the LSTM output is a price rather than a log-return.
    lstm_price = [
        'return_1', 'return_5', 'return_15',
        'close_sma_7', 'close_sma_14', 'close_sma_30',
        'bb_position',
        'rsi',
        'macd', 'macd_diff',
        'volatility_14',
        'relative_volume',
        'hl_range',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    ]

    # ------------------------------------------------------------------
    # XGBOOST-DEDICATED SETS
    # ------------------------------------------------------------------

    # xgb_momentum — strong short-term predictors for log-return target;
    # vol_regime added to handle regime-dependent signal strength
    xgb_momentum = [
        'return_1', 'return_5', 'return_15', 'return_60',
        'macd', 'macd_signal', 'macd_diff',
        'rsi',
        'close_sma_7', 'close_sma_14', 'close_sma_30',
        'vol_regime',
    ]
    if include_buy_pressure:
        xgb_momentum.append('buy_pressure')

    # xgb_volatility — regime + spread features for log-return prediction
    xgb_volatility = [
        'return_1', 'return_5',
        'volatility_7', 'volatility_14', 'volatility_30',
        'vol_regime',
        'bb_position',
        'hl_range',
        'relative_volume',
    ]

    # xgb_returns — pure return-based; maximally portable across data sources
    xgb_returns = [f'return_{w}' for w in [1, 5, 15, 60]]

    # ------------------------------------------------------------------
    # PLACEHOLDERS
    # Fill in your own feature lists below without modifying anything else.
    # ------------------------------------------------------------------
    k_set1: list = []
    k_set2: list = []
    k_set3: list = []

    return {
        # Universal
        'full':             full,
        'technical':        technical,
        'technical_bp':     technical_bp,
        'minimal':          minimal,

        # LSTM-dedicated
        'lstm_momentum':    lstm_momentum,
        'lstm_volatility':  lstm_volatility,
        'lstm_price':       lstm_price,

        # XGBoost-dedicated
        'xgb_momentum':     xgb_momentum,
        'xgb_volatility':   xgb_volatility,
        'xgb_returns':      xgb_returns,

        # Placeholders
        'k_set1':           k_set1,
        'k_set2':           k_set2,
        'k_set3':           k_set3,
    }


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_features(
    df: pd.DataFrame,
    feature_set: str = None,
    custom_features: list = None,
    validate: bool = True,
    include_buy_pressure: bool = False,
) -> list:
    """
    Select features from a DataFrame.

    Priority order: custom_features > feature_set > default (full).

    Args:
        df:                  DataFrame after prepare_features() has been called.
        feature_set:         Name of a predefined set.  See get_feature_sets().
        custom_features:     Explicit list — overrides feature_set entirely.
        validate:            True  → raise ValueError for any missing columns.
                             False → silently drop missing columns.
        include_buy_pressure: Forwarded to get_feature_sets() /
                              get_feature_columns() when building the candidate
                              list.  Ignored when custom_features is provided.

    Returns:
        list of feature column names.
    """
    if custom_features is not None:
        if validate:
            missing = [f for f in custom_features if f not in df.columns]
            if missing:
                raise ValueError(
                    f"Missing custom features in DataFrame: {missing}"
                )
            return list(custom_features)
        return [f for f in custom_features if f in df.columns]

    if feature_set is not None:
        sets = get_feature_sets(include_buy_pressure=include_buy_pressure)
        if feature_set not in sets:
            raise ValueError(
                f"Unknown feature set: '{feature_set}'. "
                f"Available: {sorted(sets.keys())}"
            )
        features = sets[feature_set]
        if not features:
            # Placeholder set (k_set*) — return empty list without raising
            return []
        if validate:
            missing = [f for f in features if f not in df.columns]
            if missing:
                raise ValueError(
                    f"Features missing from DataFrame for set "
                    f"'{feature_set}': {missing}"
                )
            return list(features)
        return [f for f in features if f in df.columns]

    # Default: full feature list
    features = get_feature_columns(include_buy_pressure=include_buy_pressure)
    if validate:
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing standard features: {missing}")
        return list(features)
    return [f for f in features if f in df.columns]


# ============================================================================
# BINANCE COLUMN REGISTRY
# ============================================================================

def get_binance_columns() -> list:
    """
    Binance API passthrough columns that are NOT features and should be
    excluded from feature DataFrames.

    Design note: taker_buy_base_asset_volume is intentionally kept in this
    list so it is never accidentally used as a raw feature.  prepare_features()
    consumes it to produce the derived 'buy_pressure' feature first; callers
    that strip Binance columns afterwards will then drop the raw source column
    while keeping the engineered one.
    """
    return [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
        'ignore',
        # Alternative capitalisation variants (some Binance export formats)
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume',
        'Ignore',
    ]
