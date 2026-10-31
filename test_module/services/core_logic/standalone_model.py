"""
Standalone LSTM Price Prediction Package
This file is auto-generated and saved with each trained model
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import warnings
import os

warnings.filterwarnings('ignore')


class LSTMPriceModel(nn.Module):
    """LSTM model for price prediction"""
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


class FeatureEngineering:
    """Feature engineering for OHLCV data"""

    @staticmethod
    def prepare_features(df):
        """Generate technical features from OHLCV data"""
        df = df.copy()

        # Returns
        for window in [1, 5, 15, 60]:
            df[f'return_{window}'] = np.log(df['close'] / df['close'].shift(window))

        # SMA ratios
        for window in [7, 14, 30]:
            sma = df['close'].rolling(window).mean()
            df[f'close_sma_{window}'] = df['close'] / sma

        # EMA and MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
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

        # Time features
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
        """Get list of feature column names"""
        features = []
        features.extend([f'return_{w}' for w in [1, 5, 15, 60]])
        features.extend([f'close_sma_{w}' for w in [7, 14, 30]])
        features.extend(['macd', 'macd_signal', 'macd_diff', 'rsi'])
        features.extend([f'volatility_{w}' for w in [7, 14, 30]])
        features.append('relative_volume')
        features.extend(['bb_position', 'hl_range'])
        features.extend(['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'])
        return features


class ModelPredictor:
    """Standalone model predictor - no external dependencies"""

    def __init__(self, model_dir):
        """
        Load trained model package

        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load config
        config_path = os.path.join(model_dir, 'model_config.pkl')
        self.config = joblib.load(config_path)

        # Load scalers
        self.feature_scaler = joblib.load(os.path.join(model_dir, 'feature_scaler.pkl'))
        self.price_scaler = joblib.load(os.path.join(model_dir, 'price_scaler.pkl'))

        # Load model
        self.model = LSTMPriceModel(
            len(self.config['features']),
            self.config['hidden_size'],
            self.config['num_layers'],
            self.config['dropout']
        ).to(self.device)

        self.model.load_state_dict(torch.load(
            os.path.join(model_dir, 'lstm_price_model.pth'),
            map_location=self.device
        ))
        self.model.eval()

    def predict(self, df, return_sequence=False):
        """
        Predict future price from recent OHLCV data

        Args:
            df: DataFrame with recent OHLCV data (must have at least lookback rows)
            return_sequence: If True, return all predictions for sequence

        Returns:
            dict with predictions and metadata
        """
        lookback = self.config['lookback']
        horizon = self.config['horizon']
        features = self.config['features']
        use_precomputed = self.config.get('use_precomputed_features', False)
        use_log_prices = self.config.get('use_log_prices', True)

        # Validate input
        if len(df) < lookback:
            raise ValueError(f"Need at least {lookback} rows, got {len(df)}")

        # Prepare features
        if not use_precomputed:
            df = FeatureEngineering.prepare_features(df)

        # Take last lookback rows
        df = df.tail(lookback + horizon).copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df) < lookback:
            raise ValueError(f"Insufficient data after cleaning")

        # Scale features
        df[features] = self.feature_scaler.transform(df[features])

        # Current price (before scaling)
        current_price = df['close'].iloc[-1]

        # Scale price
        if use_log_prices:
            df['close'] = np.log(df['close'])
        df['close'] = self.price_scaler.transform(df['close'].values.reshape(-1, 1)).flatten()

        # Create sequence
        X = df[features].tail(lookback).values.astype(np.float32)
        X = X.reshape(1, lookback, len(features))

        # Predict
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            pred_scaled = self.model(X_tensor).squeeze().cpu().numpy()

        # Denormalize
        if use_log_prices:
            pred_log = self.price_scaler.inverse_transform([[pred_scaled]])[0][0]
            predicted_price = np.exp(pred_log)
        else:
            predicted_price = self.price_scaler.inverse_transform([[pred_scaled]])[0][0]

        # Calculate change
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100

        result = {
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'price_change': float(price_change),
            'price_change_pct': float(price_change_pct),
            'direction': 'UP' if price_change > 0 else 'DOWN',
            'horizon': horizon,
            'confidence': abs(price_change_pct),
            'symbol': self.config.get('symbol', 'unknown'),
            'timestamp': df.index[-1] if hasattr(df.index[-1], 'isoformat') else str(df.index[-1])
        }

        return result

    def predict_batch(self, df, batch_size=2048):
        """
        FAST batch prediction for historical data

        Args:
            df: DataFrame with OHLCV data
            batch_size: Batch size for predictions

        Returns:
            list of prediction dicts
        """
        lookback = self.config['lookback']
        horizon = self.config['horizon']
        features = self.config['features']
        use_precomputed = self.config.get('use_precomputed_features', False)
        use_log_prices = self.config.get('use_log_prices', True)

        if len(df) < lookback + horizon:
            raise ValueError(f"Need at least {lookback + horizon} rows")

        # Prepare features
        if not use_precomputed:
            df = FeatureEngineering.prepare_features(df)

        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # Store original prices
        original_prices = df['close'].copy()

        # Scale features
        df[features] = self.feature_scaler.transform(df[features])

        # Scale prices
        if use_log_prices:
            df['close'] = np.log(df['close'])
        df['close'] = self.price_scaler.transform(df['close'].values.reshape(-1, 1)).flatten()

        # Create all sequences at once
        data_array = df[features].values.astype(np.float32)
        prices = df['close'].values.astype(np.float32)

        n_samples = len(df) - lookback - horizon + 1
        X = np.zeros((n_samples, lookback, len(features)), dtype=np.float32)
        current_prices_scaled = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            X[i] = data_array[i:i + lookback]
            current_prices_scaled[i] = prices[i + lookback - 1]

        # Batch predictions
        predictions_scaled = []
        print("Rozpoczynam predykcję.....")
        with torch.no_grad():
            try:
                for i in range(0, len(X), batch_size):
                    batch_end = min(i + batch_size, len(X))
                    X_batch = torch.FloatTensor(X[i:batch_end]).to(self.device)

                    batch_pred = self.model(X_batch).squeeze().cpu().numpy()

                    if batch_pred.ndim == 0:
                        predictions_scaled.append(batch_pred.item())
                    else:
                        predictions_scaled.extend(batch_pred.tolist())
            except:
                print("Błąd")

        predictions_scaled = np.array(predictions_scaled)

        # Denormalize predictions
        if use_log_prices:
            pred_log = self.price_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
            predicted_prices = np.exp(pred_log)
            current_log = self.price_scaler.inverse_transform(current_prices_scaled.reshape(-1, 1)).flatten()
            current_prices_real = np.exp(current_log)
        else:
            predicted_prices = self.price_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
            current_prices_real = self.price_scaler.inverse_transform(current_prices_scaled.reshape(-1, 1)).flatten()

        # Build results
        results = []
        for i in range(len(predictions_scaled)):
            current_price = current_prices_real[i]
            predicted_price = predicted_prices[i]
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100

            # Get timestamp from original dataframe
            timestamp_idx = i + lookback - 1
            if 'open_time' in df.columns:
                timestamp = df['open_time'].iloc[timestamp_idx]
            else:
                timestamp = timestamp_idx

            results.append({
                'current_price': float(current_price),
                'predicted_price': float(predicted_price),
                'price_change': float(price_change),
                'price_change_pct': float(price_change_pct),
                'direction': 'UP' if price_change > 0 else 'DOWN',
                'horizon': horizon,
                'timestamp': timestamp,
                'index': timestamp_idx
            })

        return results

    def predict_streaming(self, df):
        """
        Single prediction (for live/streaming data)
        Alias for predict() for clarity
        """
        return self.predict(df)

    def get_model_info(self):
        """Get model configuration and metadata"""
        return {
            'lookback': self.config['lookback'],
            'horizon': self.config['horizon'],
            'features': len(self.config['features']),
            'hidden_size': self.config['hidden_size'],
            'num_layers': self.config['num_layers'],
            'symbol': self.config.get('symbol', 'unknown'),
            'timestamp': self.config.get('timestamp', 'unknown'),
            'best_val_loss': self.config.get('best_val_loss', None),
            'use_log_prices': self.config.get('use_log_prices', True),
            'device': str(self.device)
        }


# Convenience function for quick predictions
def predict_price(model_dir, df, mode='streaming'):
    """
    Quick prediction function

    Args:
        model_dir: Path to model directory
        df: DataFrame with recent OHLCV data
        mode: 'streaming' for single prediction, 'batch' for multiple predictions

    Returns:
        Prediction dict (streaming) or list of dicts (batch)
    """
    predictor = ModelPredictor(model_dir)

    if mode == 'batch':
        return predictor.predict_batch(df)
    else:
        return predictor.predict_streaming(df)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python model_package.py <model_dir> [csv_file] [--batch]")
        print("\nExamples:")
        print("  Single prediction:")
        print("    python model_package.py train_results_price_btc/run_20241110_123456 test.csv")
        print("\n  Batch predictions (fast):")
        print("    python model_package.py train_results_price_btc/run_20241110_123456 test.csv --batch")
        print("\n  Model info only:")
        print("    python model_package.py train_results_price_btc/run_20241110_123456")
        sys.exit(1)

    model_dir = sys.argv[1]
    batch_mode = '--batch' in sys.argv

    # Load predictor
    predictor = ModelPredictor(model_dir)

    # Show model info
    print("\n" + "=" * 70)
    print("MODEL INFO")
    print("=" * 70)
    info = predictor.get_model_info()
    for key, value in info.items():
        print(f"  {key:20s}: {value}")

    # Test prediction if CSV provided
    if len(sys.argv) >= 3 and not sys.argv[2].startswith('--'):
        csv_file = sys.argv[2]
        print("\n" + "=" * 70)
        print(f"PREDICTION TEST - {'BATCH MODE' if batch_mode else 'STREAMING MODE'}")
        print("=" * 70)
        print(f"Loading: {csv_file}")

        df = pd.read_csv(csv_file)
        print(f"Data: {len(df)} rows")

        if batch_mode:
            import time
            start = time.time()
            results = predictor.predict_batch(df, batch_size=1024)
            elapsed = time.time() - start

            print(f"\nGenerated {len(results)} predictions in {elapsed:.2f}s")
            print(f"Speed: {len(results)/elapsed:.0f} predictions/sec")

            print("\nFirst 5 predictions:")
            for i, result in enumerate(results[:5], 1):
                print(f"{i}. Current: ${result['current_price']:,.2f} → "
                      f"Predicted: ${result['predicted_price']:,.2f} "
                      f"({result['price_change_pct']:+.2f}%) {result['direction']}")

            # Calculate accuracy
            correct = sum(1 for r in results if (r['direction'] == 'UP' and r['price_change'] > 0) or
                         (r['direction'] == 'DOWN' and r['price_change'] < 0))
            accuracy = (correct / len(results)) * 100
            print(f"\nDirectional accuracy: {accuracy:.2f}%")
        else:
            result = predictor.predict_streaming(df)

            print("\nRESULT:")
            print(f"  Current:    ${result['current_price']:,.2f}")
            print(f"  Predicted:  ${result['predicted_price']:,.2f}")
            print(f"  Change:     ${result['price_change']:+,.2f} ({result['price_change_pct']:+.2f}%)")
            print(f"  Direction:  {result['direction']}")
            print(f"  Horizon:    t+{result['horizon']}")

        print("=" * 70)