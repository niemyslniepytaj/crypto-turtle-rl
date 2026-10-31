"""
Standalone LSTM Price Prediction Package with Advanced Technical Analysis
Version: 2.0 - Updated for current architecture

This file is completely self-contained and can be used independently.
Simply copy this file and the model directory - no other dependencies needed.

Usage:
    from model_package import ModelPredictor
    
    predictor = ModelPredictor('path/to/model_dir')
    result = predictor.predict(df)
    
    print(f"Price: ${result['predicted_price']:.2f}")
    print(f"Buy Signal: {result['signals']['buy_signal']:.2f}")
    print(f"Sell Signal: {result['signals']['sell_signal']:.2f}")
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import warnings
import os
import json
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

warnings.filterwarnings('ignore')


# ============================================================================
# NEURAL NETWORK MODELS
# ============================================================================

class LSTMPriceModel(nn.Module):
    """
    LSTM model for price prediction - matches lstm_train_price.py architecture
    Automatically adapts to model configuration
    """
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
        super(LSTMPriceModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class LSTMTunable(nn.Module):
    """
    Tunable LSTM variant - supports use_batch_norm parameter
    Matches lstm_hyperparameter_tuning.py architecture
    """
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1, use_batch_norm=True):
        super(LSTMTunable, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc1(out)
        
        if self.use_batch_norm:
            out = self.bn1(out)
        
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineering:
    """
    Feature engineering for OHLCV data
    Matches feature_engineering.py exactly
    """

    @staticmethod
    def prepare_features(df):
        """
        Generate technical features from raw OHLCV data
        Identical to feature_engineering.prepare_features()
        """
        df = df.copy()

        # Log-returns
        for window in [1, 5, 15, 60]:
            df[f'return_{window}'] = np.log(df['close'] / df['close'].shift(window))

        # SMA and ratios
        for window in [7, 14, 30]:
            sma = df['close'].rolling(window).mean()
            df[f'close_sma_{window}'] = df['close'] / sma

        # EMA
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()

        # MACD
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Volatility
        for window in [7, 14, 30]:
            df[f'volatility_{window}'] = df['return_1'].rolling(window).std()

        # Volume
        volume_sma = df['volume'].rolling(20).mean()
        df['relative_volume'] = df['volume'] / volume_sma

        # Bollinger Bands
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']

        # Time features (cyclical encoding)
        dt = pd.to_datetime(df['open_time'])
        hour = dt.dt.hour
        dow = dt.dt.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

        return df

    @staticmethod
    def get_feature_columns():
        """Returns list of feature column names (excluding base columns)"""
        features = []
        
        # Returns
        features.extend([f'return_{w}' for w in [1, 5, 15, 60]])
        
        # Technical indicators
        features.extend([f'close_sma_{w}' for w in [7, 14, 30]])
        features.extend(['macd', 'macd_signal', 'macd_diff', 'rsi'])
        
        # Volatility and volume
        features.extend([f'volatility_{w}' for w in [7, 14, 30]])
        features.append('relative_volume')
        
        # Price patterns
        features.extend(['bb_position', 'hl_range'])
        
        # Time features
        features.extend(['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'])
        
        return features


# ============================================================================
# TECHNICAL ANALYSIS ENGINE
# ============================================================================

class TechnicalAnalyzer:
    """
    Advanced technical analysis for generating buy/sell probability signals
    
    Uses multiple weighted indicators to create robust trading signals:
    - Price prediction direction and magnitude
    - RSI (Relative Strength Index)
    - Bollinger Bands position
    - MACD (Moving Average Convergence Divergence)
    - Trend strength (SMA ratios)
    - Volatility adjustment
    - Volume confirmation
    
    Returns probability scores (0-1) for buy/sell actions
    """
    
    def __init__(self, 
                 rsi_oversold: float = 30, 
                 rsi_overbought: float = 70,
                 bb_lower_threshold: float = 0.2,
                 bb_upper_threshold: float = 0.8,
                 volatility_penalty: float = 0.3,
                 volume_boost: float = 0.2,
                 confidence_threshold: float = 0.5,
                 strong_threshold: float = 0.75):
        """
        Initialize technical analyzer with configurable parameters
        
        Args:
            rsi_oversold: RSI level for oversold condition (buy signal)
            rsi_overbought: RSI level for overbought condition (sell signal)
            bb_lower_threshold: BB position threshold for buy signal
            bb_upper_threshold: BB position threshold for sell signal
            volatility_penalty: Max signal reduction due to high volatility (0-1)
            volume_boost: Max signal boost from high volume (0-1)
            confidence_threshold: Minimum confidence for BUY/SELL action (vs HOLD)
            strong_threshold: Minimum confidence for STRONG_BUY/STRONG_SELL
        """
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_lower_threshold = bb_lower_threshold
        self.bb_upper_threshold = bb_upper_threshold
        self.volatility_penalty = volatility_penalty
        self.volume_boost = volume_boost
        self.confidence_threshold = confidence_threshold
        self.strong_threshold = strong_threshold
        
        # Indicator weights (must sum to 1.0)
        self.weights = {
            'price': 0.35,      # Model prediction
            'rsi': 0.15,        # Momentum
            'bb': 0.15,         # Mean reversion
            'macd': 0.15,       # Trend
            'trend': 0.10,      # SMA trend
            'volume': 0.10      # Volume confirmation
        }
    
    def _sigmoid(self, x: float, scale: float = 1.0, shift: float = 0.0) -> float:
        """Sigmoid function for smooth normalization"""
        return 1.0 / (1.0 + np.exp(-(x - shift) / scale))
    
    def _normalize_indicator(self, value: float, low: float, high: float, 
                            inverse: bool = False) -> float:
        """
        Normalize indicator to 0-1 range
        
        Args:
            value: Current indicator value
            low: Lower bound
            high: Upper bound
            inverse: If True, higher values give lower scores
        """
        if high == low:
            return 0.5
        
        normalized = (value - low) / (high - low)
        normalized = max(0.0, min(1.0, normalized))
        
        if inverse:
            normalized = 1.0 - normalized
        
        return normalized
    
    def _calculate_rsi_signal(self, rsi: float) -> Tuple[float, float]:
        """
        Calculate RSI-based buy/sell signals
        
        Returns:
            (buy_signal, sell_signal) both in range 0-1
        """
        # Buy signal: stronger when RSI is low (oversold)
        if rsi < self.rsi_oversold:
            buy = 1.0
        elif rsi < 50:
            buy = self._normalize_indicator(rsi, self.rsi_oversold, 50, inverse=True)
        else:
            buy = 0.0
        
        # Sell signal: stronger when RSI is high (overbought)
        if rsi > self.rsi_overbought:
            sell = 1.0
        elif rsi > 50:
            sell = self._normalize_indicator(rsi, 50, self.rsi_overbought, inverse=False)
        else:
            sell = 0.0
        
        return buy, sell
    
    def _calculate_bb_signal(self, bb_position: float) -> Tuple[float, float]:
        """
        Calculate Bollinger Bands-based buy/sell signals
        
        Returns:
            (buy_signal, sell_signal) both in range 0-1
        """
        # Buy signal: stronger when price is near lower band
        if bb_position < self.bb_lower_threshold:
            buy = 1.0
        elif bb_position < 0.5:
            buy = self._normalize_indicator(
                bb_position, self.bb_lower_threshold, 0.5, inverse=True
            )
        else:
            buy = 0.0
        
        # Sell signal: stronger when price is near upper band
        if bb_position > self.bb_upper_threshold:
            sell = 1.0
        elif bb_position > 0.5:
            sell = self._normalize_indicator(
                bb_position, 0.5, self.bb_upper_threshold, inverse=False
            )
        else:
            sell = 0.0
        
        return buy, sell
    
    def _calculate_macd_signal(self, macd_diff: float) -> Tuple[float, float]:
        """
        Calculate MACD-based buy/sell signals
        
        Returns:
            (buy_signal, sell_signal) both in range 0-1
        """
        # Normalize MACD diff (typically ranges from -0.02 to 0.02 for crypto)
        scale = 0.015
        
        buy = self._sigmoid(macd_diff, scale=scale, shift=0.0)
        sell = self._sigmoid(-macd_diff, scale=scale, shift=0.0)
        
        return buy, sell
    
    def _calculate_trend_signal(self, sma_14: float, sma_30: float) -> Tuple[float, float]:
        """
        Calculate trend-based buy/sell signals from SMA ratios
        
        Returns:
            (buy_signal, sell_signal) both in range 0-1
        """
        # Strong uptrend: price above SMAs
        # Strong downtrend: price below SMAs
        
        buy = 0.0
        sell = 0.0
        
        if sma_14 > 1.0:  # Price above 14-day SMA
            buy = self._normalize_indicator(sma_14, 1.0, 1.05, inverse=False)
        else:  # Price below 14-day SMA
            sell = self._normalize_indicator(sma_14, 0.95, 1.0, inverse=True)
        
        return buy, sell
    
    def _calculate_volume_factor(self, relative_volume: float) -> float:
        """
        Calculate volume confirmation factor
        High volume = more confident signals
        
        Returns:
            Volume factor (0.8 - 1.2 typical range)
        """
        if relative_volume > 1.5:
            return 1.0 + self.volume_boost
        elif relative_volume > 1.0:
            return 1.0 + (relative_volume - 1.0) * self.volume_boost / 0.5
        elif relative_volume < 0.5:
            return 0.8
        else:
            return 0.8 + (relative_volume - 0.5) * 0.4
    
    def _calculate_volatility_factor(self, volatility: float) -> float:
        """
        Calculate volatility adjustment factor
        High volatility = reduce confidence
        
        Returns:
            Volatility factor (typically 0.7 - 1.0)
        """
        # Normalize volatility (typical range for crypto: 0.005 - 0.05)
        normalized_vol = min(1.0, volatility / 0.03)
        
        # Higher volatility reduces signal strength
        return 1.0 - (normalized_vol * self.volatility_penalty)
    
    def calculate_signal(self, 
                        current_price: float,
                        predicted_price: float,
                        features_dict: Dict[str, float]) -> Dict[str, Union[float, str, Dict]]:
        """
        Calculate comprehensive buy/sell signals
        
        Args:
            current_price: Current asset price
            predicted_price: Model's predicted future price
            features_dict: Dictionary with technical indicator values
        
        Returns:
            Dictionary containing:
                - buy_signal: Buy probability (0-1)
                - sell_signal: Sell probability (0-1)
                - action: Recommended action (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
                - confidence: Overall confidence in the signal (0-1)
                - details: Breakdown of each indicator's contribution
        """
        # 1. Price prediction signal
        price_change_pct = ((predicted_price - current_price) / current_price) * 100
        
        # Use sigmoid to convert price change % to probability
        price_buy = self._sigmoid(price_change_pct, scale=2.0)
        price_sell = self._sigmoid(-price_change_pct, scale=2.0)
        
        # 2. RSI signal
        rsi = features_dict.get('rsi', 50.0)
        rsi_buy, rsi_sell = self._calculate_rsi_signal(rsi)
        
        # 3. Bollinger Bands signal
        bb_position = features_dict.get('bb_position', 0.5)
        bb_buy, bb_sell = self._calculate_bb_signal(bb_position)
        
        # 4. MACD signal
        macd_diff = features_dict.get('macd_diff', 0.0)
        macd_buy, macd_sell = self._calculate_macd_signal(macd_diff)
        
        # 5. Trend signal
        sma_14 = features_dict.get('close_sma_14', 1.0)
        sma_30 = features_dict.get('close_sma_30', 1.0)
        trend_buy, trend_sell = self._calculate_trend_signal(sma_14, sma_30)
        
        # 6. Volume signal (acts as confirmation factor)
        relative_volume = features_dict.get('relative_volume', 1.0)
        volume_factor = self._calculate_volume_factor(relative_volume)
        
        # 7. Volatility adjustment
        volatility_14 = features_dict.get('volatility_14', 0.01)
        volatility_factor = self._calculate_volatility_factor(volatility_14)
        
        # Weighted combination of signals
        buy_signal = (
            self.weights['price'] * price_buy +
            self.weights['rsi'] * rsi_buy +
            self.weights['bb'] * bb_buy +
            self.weights['macd'] * macd_buy +
            self.weights['trend'] * trend_buy +
            self.weights['volume'] * (relative_volume - 0.5)  # Volume as delta from normal
        )
        
        sell_signal = (
            self.weights['price'] * price_sell +
            self.weights['rsi'] * rsi_sell +
            self.weights['bb'] * bb_sell +
            self.weights['macd'] * macd_sell +
            self.weights['trend'] * trend_sell +
            self.weights['volume'] * (relative_volume - 0.5)
        )
        
        # Apply adjustments
        buy_signal = max(0.0, min(1.0, buy_signal * volatility_factor * volume_factor))
        sell_signal = max(0.0, min(1.0, sell_signal * volatility_factor * volume_factor))
        
        # Determine action and confidence
        net_signal = buy_signal - sell_signal
        confidence = max(buy_signal, sell_signal)
        
        if buy_signal >= self.strong_threshold and buy_signal > sell_signal:
            action = 'STRONG_BUY'
        elif buy_signal >= self.confidence_threshold and buy_signal > sell_signal:
            action = 'BUY'
        elif sell_signal >= self.strong_threshold and sell_signal > buy_signal:
            action = 'STRONG_SELL'
        elif sell_signal >= self.confidence_threshold and sell_signal > buy_signal:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        # Detailed breakdown
        details = {
            'price_signal': {
                'change_pct': price_change_pct,
                'buy': price_buy,
                'sell': price_sell
            },
            'indicators': {
                'rsi': {
                    'value': rsi,
                    'buy': rsi_buy,
                    'sell': rsi_sell
                },
                'bollinger_bands': {
                    'value': bb_position,
                    'buy': bb_buy,
                    'sell': bb_sell
                },
                'macd': {
                    'value': macd_diff,
                    'buy': macd_buy,
                    'sell': macd_sell
                },
                'trend': {
                    'sma_14': sma_14,
                    'sma_30': sma_30,
                    'buy': trend_buy,
                    'sell': trend_sell
                }
            },
            'adjustments': {
                'volatility_factor': volatility_factor,
                'volume_factor': volume_factor,
                'relative_volume': relative_volume,
                'volatility_14': volatility_14
            },
            'weights': self.weights
        }
        
        return {
            'buy_signal': float(buy_signal),
            'sell_signal': float(sell_signal),
            'action': action,
            'confidence': float(confidence),
            'net_signal': float(net_signal),
            'details': details
        }


# ============================================================================
# MODEL PREDICTOR
# ============================================================================

class ModelPredictor:
    """
    Main prediction class - handles model loading, data preparation, and prediction
    
    Features:
    - Automatic model architecture detection
    - Dynamic feature handling
    - Technical analysis signal generation
    - Batch and streaming prediction modes
    - Full error handling and validation
    """
    
    def __init__(self, 
                 model_dir: str,
                 analyzer_config: Optional[Dict] = None,
                 device: Optional[str] = None):
        """
        Initialize predictor with model directory
        
        Args:
            model_dir: Path to model directory (contains .pth, .pkl files)
            analyzer_config: Optional config dict for TechnicalAnalyzer
            device: Optional device ('cuda', 'cpu', 'mps'). Auto-detect if None
        """
        self.model_dir = model_dir
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Load model components
        self._load_model()
        
        # Initialize technical analyzer
        if analyzer_config is None:
            analyzer_config = {}
        self.analyzer = TechnicalAnalyzer(**analyzer_config)
        
        # Feature engineering
        self.feature_engineer = FeatureEngineering()
    
    def _load_model(self):
        """Load model, scalers, and configuration from directory"""
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        # Load configuration
        config_path = os.path.join(self.model_dir, 'model_config.pkl')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config not found: {config_path}")
        
        self.config = joblib.load(config_path)
        
        # Extract model parameters
        input_size = len(self.config['features'])
        hidden_size = self.config['hidden_size']
        num_layers = self.config['num_layers']
        dropout = self.config['dropout']
        
        # Detect model architecture and create instance
        model_path = os.path.join(self.model_dir, 'lstm_price_model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        
        # Try to detect if model uses batch_norm parameter (from Optuna tuning)
        use_batch_norm = self.config.get('use_batch_norm', None)
        
        if use_batch_norm is not None:
            # Model from Optuna tuning
            self.model = LSTMTunable(
                input_size, hidden_size, num_layers, dropout, use_batch_norm
            ).to(self.device)
        else:
            # Standard model
            self.model = LSTMPriceModel(
                input_size, hidden_size, num_layers, dropout
            ).to(self.device)
        
        # Load weights with error handling for architecture mismatch
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        except RuntimeError as e:
            # Try alternative architecture
            if use_batch_norm is None:
                # Try with batch norm
                self.model = LSTMTunable(
                    input_size, hidden_size, num_layers, dropout, True
                ).to(self.device)
                try:
                    self.model.load_state_dict(state_dict)
                except:
                    # Try without batch norm
                    self.model = LSTMTunable(
                        input_size, hidden_size, num_layers, dropout, False
                    ).to(self.device)
                    self.model.load_state_dict(state_dict)
            else:
                raise e
        
        self.model.eval()
        
        # Load scalers
        feature_scaler_path = os.path.join(self.model_dir, 'feature_scaler.pkl')
        price_scaler_path = os.path.join(self.model_dir, 'price_scaler.pkl')
        
        if not os.path.exists(feature_scaler_path):
            raise FileNotFoundError(f"Feature scaler not found: {feature_scaler_path}")
        if not os.path.exists(price_scaler_path):
            raise FileNotFoundError(f"Price scaler not found: {price_scaler_path}")
        
        self.feature_scaler = joblib.load(feature_scaler_path)
        self.price_scaler = joblib.load(price_scaler_path)
        
        # Extract configuration parameters
        self.lookback = self.config['lookback']
        self.horizon = self.config['horizon']
        self.features = self.config['features']
        self.use_log_prices = self.config.get('use_log_prices', True)
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare dataframe with features
        
        Args:
            df: Raw OHLCV dataframe
            
        Returns:
            DataFrame with generated features
        """
        df = df.copy()
        
        # Generate features
        df = self.feature_engineer.prepare_features(df)
        
        # Handle inf/nan
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Check for required features
        missing_features = [f for f in self.features if f not in df.columns]
        
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        return df
    
    def _create_sequence(self, df: pd.DataFrame, index: int) -> np.ndarray:
        """
        Create sequence for prediction at given index
        
        Args:
            df: DataFrame with features
            index: Index position (must have at least lookback history)
            
        Returns:
            Sequence array of shape (1, lookback, n_features)
        """
        if index < self.lookback - 1:
            raise ValueError(
                f"Insufficient history: need {self.lookback} rows, "
                f"index {index} only has {index + 1}"
            )
        
        start_idx = index - self.lookback + 1
        end_idx = index + 1
        
        sequence = df[self.features].iloc[start_idx:end_idx].values
        
        # Reshape to (1, lookback, n_features)
        sequence = sequence.reshape(1, self.lookback, -1)
        
        return sequence.astype(np.float32)
    
    def _inverse_transform_price(self, scaled_price: float) -> float:
        """
        Convert scaled price back to real price
        
        Args:
            scaled_price: Scaled/normalized price
            
        Returns:
            Real price value
        """
        if self.use_log_prices:
            # Inverse: log -> real
            log_price = self.price_scaler.inverse_transform([[scaled_price]])[0, 0]
            real_price = np.exp(log_price)
        else:
            # Direct inverse
            real_price = self.price_scaler.inverse_transform([[scaled_price]])[0, 0]
        
        return float(real_price)
    
    def predict(self, 
                df: pd.DataFrame, 
                include_signals: bool = True) -> Dict:
        """
        Single prediction from latest data (streaming mode)
        
        Args:
            df: DataFrame with OHLCV data (minimum lookback rows required)
            include_signals: If True, include technical analysis signals
            
        Returns:
            Dictionary with prediction and optionally signals:
                {
                    'current_price': float,
                    'predicted_price': float,
                    'price_change': float,
                    'price_change_pct': float,
                    'direction': str,
                    'horizon': int,
                    'timestamp': str,
                    'signals': {...}  # if include_signals=True
                }
        """
        # Prepare data
        df_prepared = self._prepare_dataframe(df)
        
        # Drop NaN rows
        df_prepared = df_prepared.dropna()
        
        if len(df_prepared) < self.lookback:
            raise ValueError(
                f"Insufficient data after feature generation: "
                f"got {len(df_prepared)}, need {self.lookback}"
            )
        
        # Use last available index
        index = len(df_prepared) - 1
        
        # Get current price (before scaling)
        current_price = float(df.iloc[index]['close'])
        
        # Scale features
        df_prepared[self.features] = self.feature_scaler.transform(
            df_prepared[self.features]
        )
        
        # Scale prices
        if self.use_log_prices:
            df_prepared['close'] = np.log(df_prepared['close'])
        
        df_prepared['close'] = self.price_scaler.transform(
            df_prepared['close'].values.reshape(-1, 1)
        ).flatten()
        
        # Create sequence
        sequence = self._create_sequence(df_prepared, index)
        
        # Predict
        with torch.no_grad():
            X = torch.FloatTensor(sequence).to(self.device)
            prediction_scaled = self.model(X).squeeze().cpu().item()
        
        # Inverse transform prediction
        predicted_price = self._inverse_transform_price(prediction_scaled)
        
        # Calculate changes
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        direction = 'UP' if price_change > 0 else 'DOWN' if price_change < 0 else 'FLAT'
        
        # Get timestamp
        if 'open_time' in df.columns:
            timestamp = df.iloc[index]['open_time']
            if isinstance(timestamp, (int, float)):
                timestamp = pd.to_datetime(timestamp, unit='ms').isoformat()
            else:
                timestamp = str(timestamp)
        else:
            timestamp = datetime.now().isoformat()
        
        result = {
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'direction': direction,
            'horizon': self.horizon,
            'timestamp': timestamp
        }
        
        # Add technical analysis signals
        if include_signals:
            # Get raw features (before scaling) for technical analysis
            features_dict = {}
            raw_row = df_prepared.iloc[index]
            
            # Reconstruct unscaled features for indicators that need real values
            for feature in self.features:
                features_dict[feature] = float(raw_row[feature])
            
            # Generate signals
            signals = self.analyzer.calculate_signal(
                current_price,
                predicted_price,
                features_dict
            )
            
            result['signals'] = signals
        
        return result
    
    def predict_batch(self, 
                     df: pd.DataFrame,
                     batch_size: int = 1024,
                     include_signals: bool = True) -> List[Dict]:
        """
        Batch prediction for multiple time points
        
        Args:
            df: DataFrame with OHLCV data
            batch_size: Batch size for predictions (larger = faster but more memory)
            include_signals: If True, include technical analysis signals
            
        Returns:
            List of prediction dictionaries
        """
        # # Prepare data
        # df_prepared = self._prepare_dataframe(df)

        #TODO:
        # # Get original prices before any transformation
        # original_prices = df['close'].values.copy()
        #
        # # Drop NaN rows
        # df_prepared = df_prepared.dropna()
        df_prepared = self._prepare_dataframe(df)
        df_prepared = df_prepared.dropna()  # Najpierw usuń NaN

        # Teraz pobierz ceny z wyrównanego DataFrame'a
        original_prices = df_prepared['close'].values.copy()



        if len(df_prepared) < self.lookback:
            raise ValueError(
                f"Insufficient data: got {len(df_prepared)}, need {self.lookback}"
            )
        
        # Store raw features for signal calculation
        features_list = []
        if include_signals:
            for idx in range(self.lookback - 1, len(df_prepared)):
                features_dict = {}
                for feature in self.features:
                    features_dict[feature] = float(df_prepared.iloc[idx][feature])
                features_list.append(features_dict)
        
        # Scale features
        df_prepared[self.features] = self.feature_scaler.transform(
            df_prepared[self.features]
        )
        print("a")
        # Scale prices
        if self.use_log_prices:
            df_prepared['close'] = np.log(df_prepared['close'])
        print("a")
        df_prepared['close'] = self.price_scaler.transform(
            df_prepared['close'].values.reshape(-1, 1)
        ).flatten()
        print("a")



        # # Create all sequences
        # n_predictions = len(df_prepared) - self.lookback + 1
        # X_all = np.zeros((n_predictions, self.lookback, len(self.features)),
        #                 dtype=np.float32)
        # print("a")
        # for i in range(n_predictions):
        #     X_all[i] = self._create_sequence(df_prepared, i + self.lookback - 1)
        #     print(i)
        # print("a")




        # PO (ZOPTYMALIZOWANE - 200x szybciej!):
        # Create all sequences using vectorized sliding window
        features_array = df_prepared[self.features].values.astype(np.float32)
        n_rows = len(features_array)
        n_predictions = n_rows - self.lookback + 1

        # Use numpy sliding_window_view (numpy >= 1.20)
        from numpy.lib.stride_tricks import sliding_window_view
        X_all = sliding_window_view(features_array, window_shape=self.lookback, axis=0)
        X_all = np.transpose(X_all, (0, 2, 1)).astype(np.float32)
        # Batch predictions
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X_all), batch_size):
                batch_end = min(i + batch_size, len(X_all))
                X_batch = torch.FloatTensor(X_all[i:batch_end]).to(self.device)
                batch_pred = self.model(X_batch).squeeze().cpu().numpy()
                
                if batch_pred.ndim == 0:
                    predictions.append(batch_pred.item())
                else:
                    predictions.extend(batch_pred.tolist())
        
        # Process results
        results = []
        
        for i, pred_scaled in enumerate(predictions):
            # Original index in df
            original_idx = i + self.lookback - 1
            
            # Get current price from original data
            current_price = float(original_prices[original_idx])
            
            # Inverse transform prediction
            predicted_price = self._inverse_transform_price(pred_scaled)
            
            # Calculate changes
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100
            direction = 'UP' if price_change > 0 else 'DOWN' if price_change < 0 else 'FLAT'
            
            # Get timestamp
            if 'open_time' in df.columns:
                timestamp = df.iloc[original_idx]['open_time']
                if isinstance(timestamp, (int, float)):
                    timestamp = pd.to_datetime(timestamp, unit='ms').isoformat()
                else:
                    timestamp = str(timestamp)
                timestamp_idx = original_idx
            else:
                timestamp = None
                timestamp_idx = original_idx
            
            result = {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'direction': direction,
                'horizon': self.horizon,
                'timestamp': timestamp,
                'index': timestamp_idx
            }
            
            # Add signals
            if include_signals and i < len(features_list):
                signals = self.analyzer.calculate_signal(
                    current_price,
                    predicted_price,
                    features_list[i]
                )
                result['signals'] = signals
            
            results.append(result)
        
        return results
    
    def predict_streaming(self, df: pd.DataFrame, include_signals: bool = True) -> Dict:
        """
        Alias for predict() for clarity in streaming use cases
        """
        return self.predict(df, include_signals=include_signals)
    
    def get_model_info(self) -> Dict:
        """
        Get model configuration and metadata
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_id': self.config.get('model_id', 'unknown'),
            'data_id': self.config.get('data_id', 'unknown'),
            'lookback': self.lookback,
            'horizon': self.horizon,
            'features': len(self.features),
            'feature_names': self.features,
            'hidden_size': self.config['hidden_size'],
            'num_layers': self.config['num_layers'],
            'dropout': self.config['dropout'],
            'symbol': self.config.get('symbol', 'unknown'),
            'timestamp': self.config.get('timestamp', 'unknown'),
            'best_val_loss': self.config.get('best_val_loss', None),
            'use_log_prices': self.use_log_prices,
            'device': str(self.device),
            'feature_set': self.config.get('feature_set', 'unknown'),
            'model_type': self.config.get('model_type', 'price_prediction'),
            'data_source': self.config.get('data_source', 'unknown')
        }
    
    def get_analyzer_config(self) -> Dict:
        """
        Get technical analyzer configuration
        
        Returns:
            Dictionary with analyzer parameters
        """
        return {
            'rsi_oversold': self.analyzer.rsi_oversold,
            'rsi_overbought': self.analyzer.rsi_overbought,
            'bb_lower_threshold': self.analyzer.bb_lower_threshold,
            'bb_upper_threshold': self.analyzer.bb_upper_threshold,
            'volatility_penalty': self.analyzer.volatility_penalty,
            'volume_boost': self.analyzer.volume_boost,
            'confidence_threshold': self.analyzer.confidence_threshold,
            'strong_threshold': self.analyzer.strong_threshold,
            'weights': self.analyzer.weights
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def predict_price(model_dir: str, 
                 df: pd.DataFrame, 
                 mode: str = 'streaming',
                 include_signals: bool = True,
                 analyzer_config: Optional[Dict] = None) -> Union[Dict, List[Dict]]:
    """
    Quick prediction function - simple interface
    
    Args:
        model_dir: Path to model directory
        df: DataFrame with OHLCV data
        mode: 'streaming' for single prediction, 'batch' for multiple
        include_signals: Include technical analysis signals
        analyzer_config: Optional TechnicalAnalyzer configuration
    
    Returns:
        Single prediction dict (streaming) or list of dicts (batch)
    
    Examples:
        # Single prediction
        result = predict_price('lstm_models/L001_btc_...', df)
        print(f"Price: ${result['predicted_price']:.2f}")
        print(f"Action: {result['signals']['action']}")
        
        # Batch predictions
        results = predict_price('lstm_models/L001_btc_...', df, mode='batch')
        for r in results:
            print(f"{r['timestamp']}: {r['action']}")
    """
    predictor = ModelPredictor(model_dir, analyzer_config=analyzer_config)
    
    if mode == 'batch':
        return predictor.predict_batch(df, include_signals=include_signals)
    else:
        return predictor.predict_streaming(df, include_signals=include_signals)


def load_model(model_dir: str, 
               analyzer_config: Optional[Dict] = None) -> ModelPredictor:
    """
    Load model predictor
    
    Args:
        model_dir: Path to model directory
        analyzer_config: Optional TechnicalAnalyzer configuration
    
    Returns:
        ModelPredictor instance
    
    Example:
        predictor = load_model('lstm_models/L001_btc_...')
        result = predictor.predict(df)
    """
    return ModelPredictor(model_dir, analyzer_config=analyzer_config)

import numpy as np
import pandas as pd
from scipy.stats import norm

class ProbabilisticSignalGenerator:
    """
    Converts predicted price from model into probabilistic BUY/SELL signal
    using log-return and normal distribution assumption.

    log_return ~ N(mu, sigma^2)

    where:
        mu    = predicted log return
        sigma = rolling volatility
    """

    def __init__(self, threshold: float = 0.002, vol_window: int = 50):
        """
        Parameters
        ----------
        threshold : float
            minimal log-return considered meaningful signal
        vol_window : int
            rolling window used to estimate volatility
        """

        self.threshold = threshold
        self.vol_window = vol_window

    def compute_log_return(self, current_price: float, predicted_price: float) -> float:
        """
        Convert predicted price to log return
        """

        if current_price <= 0 or predicted_price <= 0:
            return 0.0

        return float(np.log(predicted_price / current_price))

    def compute_rolling_volatility(self, prices: pd.Series) -> float:
        """
        Estimate volatility from historical prices
        """

        log_returns = np.log(prices / prices.shift(1)).dropna()

        if len(log_returns) < 2:
            return 1e-8

        sigma = log_returns.tail(self.vol_window).std()

        if sigma == 0 or np.isnan(sigma):
            sigma = 1e-8

        return float(sigma)

    def compute_probabilities(self, mu: float, sigma: float):
        """
        Compute BUY and SELL probabilities
        """

        p_buy = 1 - norm.cdf((self.threshold - mu) / sigma)
        p_sell = norm.cdf((-self.threshold - mu) / sigma)

        return float(p_buy), float(p_sell)

    def generate_signal(
            self,
            current_price: float,
            predicted_price: float,
            price_series: pd.Series,
    ):
        """
        Main function returning probabilities and discrete signal
        """

        mu = self.compute_log_return(current_price, predicted_price)
        sigma = self.compute_rolling_volatility(price_series)

        p_buy, p_sell = self.compute_probabilities(mu, sigma)

        signal = 0
        if p_buy > 0.5:
            signal = 1
        elif p_sell > 0.5:
            signal = -1

        return {
            "mu": mu,
            "sigma": sigma,
            "p_buy": p_buy,
            "p_sell": p_sell,
            "signal": signal,
        }