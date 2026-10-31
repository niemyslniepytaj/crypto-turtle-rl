"""
XGBModelPackage — standalone inference package for XGBoost crypto price models.

Produces probabilistic BUY / SELL signals from a trained XGBoost log-return model.

Signal model (Approach A — two independent thresholds, normal distribution):
    mu    = model prediction (log-return, after inverse_transform)
    sigma = rolling std of log-returns from close prices (market volatility)
    T     = threshold (minimum economically meaningful move)

    signal_buy  = P(r >  T) = (1 - Phi((T  - mu) / sigma)) * 100   [0-100]
    signal_sell = P(r < -T) = Phi((-T - mu) / sigma) * 100          [0-100]

    BUY + SELL < 100 always (gap = P(-T <= r <= T) = "hold zone").
    The simulation module decides action based on min_signal threshold.

Inputs accepted:
    'raw'      — raw OHLCV from Binance (open_time, open, high, low, close, volume)
    'prepared' — prepared CSV from data_preparation (features already scaled,
                 close is raw price, see prepared data format)
    'auto'     — auto-detect based on column presence (default)

Usage:
    from xgb_model_package import XGBModelPackage, SignalConfig

    cfg = SignalConfig(threshold=0.001, window_sigma=100)
    pkg = XGBModelPackage('xgb_models/X060_eth_xgb_model_E001_0803', signal_config=cfg)

    # Batch (simulation / backtest)
    df_signals = pkg.predict_batch(df)
    # columns: timestamp, mu, sigma, signal_buy, signal_sell

    # Live (one signal for latest candle)
    signal = pkg.predict_live(df_buffer)
    # dict: {timestamp, mu, sigma, signal_buy, signal_sell}

    # Model info
    info = pkg.get_model_info()

    # Minimum rows needed for predict_live
    n = pkg.min_rows_required
"""

import os
import sys
import json
import dataclasses
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, RobustScaler

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Feature engineering bridge (resolves root feature_engineering.py)
# ---------------------------------------------------------------------------
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from feature_engineering import prepare_features, select_features  # noqa: E402


# ===========================================================================
# CONFIG DATACLASS
# ===========================================================================

@dataclasses.dataclass
class SignalConfig:
    """
    Parameters for the probabilistic signal generator.

    threshold:    Minimum log-return to trigger a signal.
                  Should cover trading fees + spread.
                  Typical range for BTC/ETH 1m: 0.0005 – 0.005
                  Default 0.001 ≈ 0.1% (covers ~0.1% fee round-trip).

    window_sigma: Rolling window (in candles) for estimating market volatility.
                  Larger = more stable but slower to react.
                  Recommended for 1m data: 50 – 200.
                  Default 100 = ~100 minutes of history.
    """
    threshold:    float = 0.001
    window_sigma: int   = 100


# ===========================================================================
# FLATTENING  (copied from xgb_model/common/data_utils.py — no dependency)
# ===========================================================================

def _create_aggregated_features(df: pd.DataFrame, features: list, lookback: int):
    """Aggregate temporal features (mean, std, min, max, last) over lookback window."""
    agg_list = []
    names = []
    for feat in features:
        rolled = df[feat].shift(1).rolling(window=lookback, min_periods=1)
        agg_list += [rolled.mean(), rolled.std(), rolled.min(), rolled.max(), df[feat].shift(1)]
        names += [
            f'{feat}_mean_{lookback}', f'{feat}_std_{lookback}',
            f'{feat}_min_{lookback}',  f'{feat}_max_{lookback}',
            f'{feat}_last',
        ]
    flat = pd.concat(agg_list, axis=1)
    flat.columns = names
    return flat, names


def _create_lag_features(df: pd.DataFrame, features: list, lookback: int):
    """Create simple lag features t-1 … t-lookback."""
    lag_list = []
    names = []
    for feat in features:
        for lag in range(1, lookback + 1):
            lag_list.append(df[feat].shift(lag))
            names.append(f'{feat}_lag_{lag}')
    flat = pd.concat(lag_list, axis=1)
    flat.columns = names
    return flat, names


def _create_hybrid_features(df: pd.DataFrame, features: list, lookback: int,
                             lag_points=None, windows=None):
    """Strategic lags + multi-window rolling stats + diffs."""
    if lag_points is None:
        lag_points = [1, 2, 3, 5, 10, 24]
    if windows is None:
        windows = [6, 12, 24, 48]
    lag_points = [l for l in lag_points if l <= lookback]
    windows    = [w for w in windows    if w <= lookback]

    parts = []
    names = []
    for feat in features:
        for lag in lag_points:
            parts.append(df[feat].shift(lag))
            names.append(f'{feat}_lag_{lag}')
        for w in windows:
            rolled = df[feat].shift(1).rolling(window=w, min_periods=1)
            parts += [rolled.mean(), rolled.std()]
            names += [f'{feat}_mean_{w}', f'{feat}_std_{w}']
        parts.append(df[feat].shift(1).diff(1))
        names.append(f'{feat}_diff_1')
        if lookback >= 5:
            parts.append(df[feat].shift(1).diff(5))
            names.append(f'{feat}_diff_5')

    flat = pd.concat(parts, axis=1)
    flat.columns = names
    return flat, names


def _flatten_features(df: pd.DataFrame, features: list, lookback: int, method: str):
    """Dispatch to the correct flattening method."""
    if method == 'agg':
        return _create_aggregated_features(df, features, lookback)
    elif method == 'lag':
        return _create_lag_features(df, features, lookback)
    elif method == 'hybrid':
        return _create_hybrid_features(df, features, lookback)
    else:
        raise ValueError(f"Unknown flattening method: '{method}'. "
                         f"Expected 'agg', 'lag', or 'hybrid'.")


# ===========================================================================
# TARGET SCALER  (standalone, no xgb_model dependency)
# ===========================================================================

class _TargetScaler:
    """Thin wrapper around sklearn scalers for 1-D log-return arrays."""

    def __init__(self, method: str = 'robust'):
        self.method = method
        if method == 'robust':
            self._scaler = RobustScaler()
        elif method == 'standard':
            self._scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler method: '{method}'")

    @classmethod
    def from_params(cls, params: dict) -> '_TargetScaler':
        """Reconstruct scaler from target_scaler_params stored in model_config.json."""
        obj = cls(method=params['method'])
        if params['method'] == 'robust':
            obj._scaler.center_ = np.array([params['center']])
            obj._scaler.scale_  = np.array([params['scale']])
        elif params['method'] == 'standard':
            obj._scaler.mean_   = np.array([params['mean']])
            obj._scaler.scale_  = np.array([params['std']])
            obj._scaler.var_    = np.array([params['std'] ** 2])
            obj._scaler.n_features_in_ = 1
        return obj

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return self._scaler.inverse_transform(y.reshape(-1, 1)).flatten()


# ===========================================================================
# VALIDATION HELPERS
# ===========================================================================

_RAW_REQUIRED   = {'open_time', 'open', 'high', 'low', 'close', 'volume'}
_PREP_REQUIRED  = {'open_time', 'close'}
# Minimum warmup rows for prepare_features (return_60 needs 60, EWM ~3×26=78)
_FE_WARMUP = 100


def _detect_mode(df: pd.DataFrame, config: dict) -> str:
    """Return 'prepared' if feature columns already present, else 'raw'."""
    first_feat = config['features'][0]
    return 'prepared' if first_feat in df.columns else 'raw'


def _validate(df: pd.DataFrame, mode: str, config: dict, window_sigma: int) -> None:
    """
    Raise ValueError with a descriptive message if df is not suitable.
    Checks are intentionally fast (no heavy iteration).
    """
    lookback = config['lookback']

    if mode == 'raw':
        missing = _RAW_REQUIRED - set(df.columns)
        if missing:
            raise ValueError(
                f"[raw mode] Missing required columns: {sorted(missing)}\n"
                f"Expected: {sorted(_RAW_REQUIRED)}"
            )
        min_rows = lookback + window_sigma + _FE_WARMUP
        if len(df) < min_rows:
            raise ValueError(
                f"[raw mode] Insufficient rows: got {len(df)}, need >= {min_rows} "
                f"(lookback={lookback} + window_sigma={window_sigma} + warmup={_FE_WARMUP})"
            )

    elif mode == 'prepared':
        missing_base = _PREP_REQUIRED - set(df.columns)
        if missing_base:
            raise ValueError(
                f"[prepared mode] Missing base columns: {sorted(missing_base)}"
            )
        missing_feats = [f for f in config['features'] if f not in df.columns]
        if missing_feats:
            preview = missing_feats[:5]
            extra   = f' … (+{len(missing_feats)-5} more)' if len(missing_feats) > 5 else ''
            raise ValueError(
                f"[prepared mode] Missing feature columns: {preview}{extra}\n"
                f"Model expects {len(config['features'])} features from config."
            )
        min_rows = lookback + window_sigma
        if len(df) < min_rows:
            raise ValueError(
                f"[prepared mode] Insufficient rows: got {len(df)}, need >= {min_rows} "
                f"(lookback={lookback} + window_sigma={window_sigma})"
            )
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Use 'raw', 'prepared', or 'auto'.")

    # Shared checks
    if df['close'].isna().any():
        raise ValueError("Column 'close' contains NaN values.")
    if (df['close'] <= 0).any():
        raise ValueError("Column 'close' contains zero or negative values.")


# ===========================================================================
# SIGNAL CALCULATION  (vectorised)
# ===========================================================================

def _compute_sigma(close: pd.Series, window: int) -> pd.Series:
    """Rolling std of log-returns. First (window-1) values are NaN."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=window, min_periods=window).std()


def _compute_signals(mu: np.ndarray, sigma: np.ndarray,
                     threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised signal calculation.

    signal_buy  = P(r >  T | mu, sigma) * 100
    signal_sell = P(r < -T | mu, sigma) * 100

    Values outside [0, 100] are clipped (numerical safety).
    """
    sigma_safe  = np.maximum(sigma, 1e-8)
    signal_buy  = (1.0 - norm.cdf((threshold  - mu) / sigma_safe)) * 100.0
    signal_sell = norm.cdf((-threshold - mu) / sigma_safe) * 100.0
    signal_buy  = np.clip(signal_buy,  0.0, 100.0)
    signal_sell = np.clip(signal_sell, 0.0, 100.0)
    return signal_buy, signal_sell


# ===========================================================================
# MAIN CLASS
# ===========================================================================

class XGBModelPackage:
    """
    Standalone inference + signal package for a trained XGBoost log-return model.

    Parameters
    ----------
    model_dir : str
        Path to model directory produced by xgb_train.py.
        Must contain: xgb_model.json, feature_scaler.pkl, model_config.json
        Optional:     target_scaler.pkl (reconstructed from model_config if absent)

    signal_config : SignalConfig, optional
        Signal generation parameters. Uses defaults if None.
    """

    def __init__(self, model_dir: str, signal_config: Optional[SignalConfig] = None):
        self._model_dir = model_dir
        self._sig_cfg   = signal_config or SignalConfig()

        self._config              = self._load_config()
        self._model               = self._load_model()
        self._feature_scaler      = self._load_feature_scaler()
        self._target_scaler       = self._load_target_scaler()
        self._iter_range          = self._resolve_iter_range()
        self._scaler_feature_cols = self._resolve_scaler_feature_cols()

    # ------------------------------------------------------------------
    # LOADING
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        path = os.path.join(self._model_dir, 'model_config.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"model_config.json not found in: {self._model_dir}")
        with open(path, 'r') as f:
            return json.load(f)

    def _load_model(self) -> xgb.Booster:
        path = os.path.join(self._model_dir, 'xgb_model.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"xgb_model.json not found in: {self._model_dir}")
        model = xgb.Booster()
        model.load_model(path)
        return model

    def _load_feature_scaler(self):
        # Prepared data uses 'scaler_features.pkl'; training saves 'feature_scaler.pkl'
        for name in ('feature_scaler.pkl', 'scaler_features.pkl'):
            path = os.path.join(self._model_dir, name)
            if os.path.exists(path):
                return joblib.load(path)
        raise FileNotFoundError(
            f"Feature scaler not found in {self._model_dir} "
            f"(tried feature_scaler.pkl and scaler_features.pkl)"
        )

    def _load_target_scaler(self) -> Optional['_TargetScaler']:
        """
        Load target scaler.

        Priority:
          1. target_scaler_params from model_config.json — always preferred,
             requires no external module, immune to pickle compatibility issues.
          2. target_scaler.pkl — attempted only when params are absent;
             pkl files serialised with common.data_utils.TargetScaler in sys.path
             will raise ModuleNotFoundError, which is caught and re-raised with
             a clear message.
        """
        if not self._config.get('scale_target', False):
            return None

        # Priority 1: reconstruct from JSON params — no pickle, no module deps
        params = self._config.get('target_scaler_params')
        if params:
            return _TargetScaler.from_params(params)

        # Priority 2: load .pkl as fallback (models without params in config)
        pkl_path = os.path.join(self._model_dir, 'target_scaler.pkl')
        if os.path.exists(pkl_path):
            try:
                raw = joblib.load(pkl_path)
            except (ModuleNotFoundError, ImportError) as exc:
                raise RuntimeError(
                    f"Cannot load target_scaler.pkl ({exc}) and "
                    f"'target_scaler_params' is missing from model_config.json. "
                    f"Re-save the model with xgb_train.py to embed scaler params."
                ) from exc
            if isinstance(raw, _TargetScaler):
                return raw
            # Wrap a bare sklearn scaler
            wrapper = _TargetScaler.__new__(_TargetScaler)
            wrapper.method  = self._config.get('target_scaler_type', 'robust')
            wrapper._scaler = raw
            return wrapper

        return None

    def _resolve_iter_range(self) -> tuple:
        """
        Determine iteration range for model.predict().

        XGBoost iteration_range is [start, end) — end is exclusive.
        The constraint is:  end <= model.BoostedRounds().

        We always use num_boosted_rounds() as the authoritative upper bound
        so we never generate an out-of-range error regardless of what values
        were persisted in model_config.json.
        """
        actual_rounds = self._model.num_boosted_rounds()  # authoritative count

        saved = self._model.attr('saved_iteration')
        if saved is not None:
            # saved_iteration is 0-based index of last tree → need index+1 trees,
            # but clamp to actual_rounds to avoid XGBoost out-of-range error.
            n = min(int(saved) + 1, actual_rounds)
        else:
            # Fallback: use all trees physically in the file
            n = actual_rounds

        return (0, max(1, n))

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    @property
    def min_rows_required(self) -> int:
        """
        Minimum number of rows needed in the input DataFrame for predict_live.
        For 'prepared' mode: lookback + window_sigma.
        For 'raw' mode:      lookback + window_sigma + FE warmup (100).
        Returns the larger value (safe for both modes).
        """
        lb = self._config['lookback']
        ws = self._sig_cfg.window_sigma
        return lb + ws + _FE_WARMUP

    def get_model_info(self) -> dict:
        """Return key model metadata."""
        c = self._config
        return {
            'model_id':          c.get('model_id', 'unknown'),
            'data_id':           c.get('data_id', 'unknown'),
            'symbol':            c.get('symbol', 'unknown'),
            'lookback':          c.get('lookback'),
            'horizon':           c.get('horizon'),
            'n_features':        c.get('n_features'),
            'n_flat_features':   c.get('n_flat_features'),
            'flattening_method': c.get('flattening_method'),
            'feature_set':       c.get('feature_set'),
            'scale_target':      c.get('scale_target'),
            'target_scaler_type':c.get('target_scaler_type'),
            'train_metrics':     c.get('train_metrics'),
            'val_metrics':       c.get('val_metrics'),
            'timestamp':         c.get('timestamp'),
            'signal_threshold':  self._sig_cfg.threshold,
            'signal_window_sigma': self._sig_cfg.window_sigma,
            'iter_range':        self._iter_range,
        }

    def predict_batch(self, df: pd.DataFrame,
                      data_mode: str = 'auto') -> pd.DataFrame:
        """
        Process an entire DataFrame and return signals for every valid sample.

        Parameters
        ----------
        df : pd.DataFrame
            Input data. See module docstring for column requirements.
        data_mode : str
            'raw' | 'prepared' | 'auto' (default).

        Returns
        -------
        pd.DataFrame with columns:
            timestamp    — open_time of the current candle (t)
            mu           — predicted log-return (after inverse_transform)
            sigma        — rolling volatility at candle t
            signal_buy   — P(r > threshold) * 100  [0-100]
            signal_sell  — P(r < -threshold) * 100 [0-100]

        Rows where sigma is NaN (first window_sigma-1 candles) are dropped.
        """
        mode = _detect_mode(df, self._config) if data_mode == 'auto' else data_mode
        _validate(df, mode, self._config, self._sig_cfg.window_sigma)

        df_work = self._prepare_df(df, mode)
        mu      = self._predict_mu(df_work)
        sigma_aligned, timestamps = self._compute_aligned_sigma(df_work, len(mu))
        return self._build_output(mu, sigma_aligned, timestamps)

    def predict_live(self, df: pd.DataFrame,
                     data_mode: str = 'auto') -> dict:
        """
        Produce a single signal for the latest candle (live / streaming mode).

        Internally calls predict_batch and returns the last row as a dict.
        The caller must maintain a rolling buffer of at least min_rows_required rows.

        Parameters
        ----------
        df : pd.DataFrame
            Rolling buffer (latest candles), same format as predict_batch.
        data_mode : str
            'raw' | 'prepared' | 'auto' (default).

        Returns
        -------
        dict:
            {timestamp, mu, sigma, signal_buy, signal_sell}
        """
        result_df = self.predict_batch(df, data_mode=data_mode)
        if result_df.empty:
            raise RuntimeError(
                "predict_live: no valid signals produced. "
                f"Provide at least {self.min_rows_required} rows."
            )
        row = result_df.iloc[-1]
        return {
            'timestamp':   row['timestamp'],
            'mu':          float(row['mu']),
            'sigma':       float(row['sigma']),
            'signal_buy':  float(row['signal_buy']),
            'signal_sell': float(row['signal_sell']),
        }

    # ------------------------------------------------------------------
    # INTERNAL PIPELINE
    # ------------------------------------------------------------------

    def _resolve_scaler_feature_cols(self) -> Optional[list]:
        """
        Determine the ordered list of feature columns that feature_scaler was
        fitted on. Required in 'raw' mode to build the full input matrix before
        calling scaler.transform().

        Resolution order:
          1. prepared data metadata.json  (same file used at training time)
          2. model_config 'features'      (only when count matches scaler)
          3. None                         (raw mode will warn + best-effort scale)
        """
        n = self._feature_scaler.n_features_in_

        # 1. Try prepared data metadata.json
        train_path = self._config.get('train_path', '')
        if train_path:
            # train_path is stored relative to the model's working directory
            # (i.e. relative to model_dir/../..)
            candidates = [
                train_path,
                os.path.join(self._model_dir, '..', '..', train_path),
                os.path.join(self._model_dir, '..', train_path),
                os.path.join(os.path.dirname(self._model_dir), train_path),
                os.path.join(self._model_dir),
            ]
            for base in candidates:
                meta_path = os.path.normpath(os.path.join(base, 'metadata.json'))
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                        cols = meta.get('features', [])
                        if len(cols) == n:
                            return cols
                    except Exception:
                        pass

        # 2. Fallback: model config features (if count matches scaler)
        model_feats = self._config.get('features', [])
        if len(model_feats) == n:
            return model_feats
        print("[XGBModelPackage][ERROR] w modelu jest inna liczba features niż w skalerze oraz nie znaleziono zapasowego pliku metadata.json")
        # 3. Cannot determine
        return None

    def _prepare_df(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        """
        Return a clean DataFrame with scaled feature columns and raw close prices.
        The returned df has a clean integer index (reset_index).
        """
        features = self._config['features']

        if mode == 'raw':
            df_feat = prepare_features(df.copy())
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
            # Drop rows where any model feature is NaN.
            # Only consider features that prepare_features actually produced
            # (buy_pressure requires taker_buy columns, absent in plain OHLCV).
            available_model_feats = [f for f in features if f in df_feat.columns]
            df_feat = df_feat.dropna(subset=available_model_feats).reset_index(drop=True)
            df_feat = df_feat.copy()

            # Scale features.
            # feature_scaler was fit on ALL features from prepared data (e.g. 23),
            # which may be more than the model uses (e.g. 16). We must pass the
            # full column set in the exact order the scaler expects.
            scaler_cols = self._scaler_feature_cols
            if scaler_cols is not None:
                # Fill any column the scaler needs but OHLCV input cannot produce
                for col in scaler_cols:
                    if col not in df_feat.columns:
                        df_feat[col] = 0.0
                df_feat[scaler_cols] = self._feature_scaler.transform(df_feat[scaler_cols])
            else:
                # Cannot determine scaler column order — scale only available feats
                warnings.warn(
                    "Could not resolve scaler column order from prepared data metadata. "
                    "Scaling only available model features — predictions may be inaccurate.",
                    RuntimeWarning
                )
                if available_model_feats:
                    df_feat[available_model_feats] = self._feature_scaler.transform(
                        df_feat[available_model_feats]
                    )

        else:  # prepared
            # Features already scaled; close is raw price
            df_feat = df.copy().reset_index(drop=True)

        return df_feat

    def _predict_mu(self, df_work: pd.DataFrame) -> np.ndarray:
        """Run flattening → XGBoost predict → inverse_transform → mu array."""
        features  = self._config['features']
        lookback  = self._config['lookback']
        method    = self._config['flattening_method']

        X_flat, _ = _flatten_features(df_work, features, lookback, method)

        # First `lookback` rows of X_flat have no complete window — drop them
        X_flat = X_flat.iloc[lookback:].reset_index(drop=True)

        # Also align with the number of targets (horizon may trim the tail)
        # We predict on ALL X_flat rows; caller aligns with sigma
        X_flat = X_flat.dropna()
        X_np   = X_flat.values.astype(np.float32)

        dmat   = xgb.DMatrix(X_np)
        iter_range = self._iter_range if self._iter_range[1] > 0 else None

        if iter_range:
            y_scaled = self._model.predict(dmat, iteration_range=iter_range)
        else:
            y_scaled = self._model.predict(dmat)

        # Inverse-transform to raw log-return space
        if self._target_scaler is not None:
            mu = self._target_scaler.inverse_transform(y_scaled)
        else:
            mu = y_scaled

        return mu.astype(np.float64)

    def _compute_aligned_sigma(self, df_work: pd.DataFrame,
                                n_mu: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute sigma aligned with mu predictions.

        For sample i (0-indexed from predict), the XGB window covers
        df_work rows [i .. i+lookback-1], so the "current candle" index
        in df_work is i + lookback - 1.
        Therefore sigma_aligned[i] = sigma_series.iloc[i + lookback - 1].

        Returns (sigma_aligned, timestamps), both length == n_mu
        (NaN sigma values are preserved; caller drops them via build_output).
        """
        window   = self._sig_cfg.window_sigma
        lookback = self._config['lookback']

        sigma_series = _compute_sigma(df_work['close'], window)
        timestamps   = df_work['open_time'] if 'open_time' in df_work.columns \
                       else pd.RangeIndex(len(df_work))

        # Map sample index → df_work row index
        current_candle_indices = np.arange(n_mu) + lookback

        # Clip to valid range (guard against edge cases)
        max_idx = len(df_work) - 1
        current_candle_indices = np.clip(current_candle_indices, 0, max_idx)

        sigma_aligned = sigma_series.iloc[current_candle_indices].values
        ts_aligned    = timestamps.iloc[current_candle_indices].values

        return sigma_aligned, ts_aligned

    def _build_output(self, mu: np.ndarray, sigma: np.ndarray,
                      timestamps: np.ndarray) -> pd.DataFrame:
        """Compute signals and return clean output DataFrame (NaN sigma rows dropped)."""
        signal_buy, signal_sell = _compute_signals(mu, sigma, self._sig_cfg.threshold)

        out = pd.DataFrame({
            'timestamp':   timestamps,
            'mu':          mu,
            'sigma':       sigma,
            'signal_buy':  signal_buy,
            'signal_sell': signal_sell,
        })

        # Drop rows where sigma could not be computed (start of series)
        out = out.dropna(subset=['sigma']).reset_index(drop=True)
        return out
