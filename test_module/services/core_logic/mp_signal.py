import joblib
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

#from .standalone_model import ModelPredictor
from .model_package import ModelPredictor, ProbabilisticSignalGenerator
# from .xgb_model_package import XGBModelPackage, SignalConfig
from shared.models.xgb_model_package import XGBModelPackage,SignalConfig
try:
    from shared.models.lstm_model_package_latest import (
        LSTMModelPackage as LSTMModelPackageLatest,
        SignalConfig as LSTMSignalConfigLatest,
    )
except ImportError:
    LSTMModelPackageLatest = None
    LSTMSignalConfigLatest = None
try:
    from shared.models.lstm_model_package2 import LSTMModelPackage as LSTMModelPackage2, SignalConfig as LSTMSignalConfig2
except ImportError:
    LSTMModelPackage2 = None
    LSTMSignalConfig2 = None

from shared.models.lstm_model_package import LSTMModelPackage as LSTMModelPackage1, SignalConfig as LSTMSignalConfig1

class ModelSignal:
    """Ładuje pakiet modelu MP i udostępnia predykcje / prawdopodobieństwa na szeregu OHLCV."""

    def __init__(self, model_name, data, base_dir, realtime = False):
        self.BASE_MODEL_DIR = os.path.join(base_dir, "shared", "models")
        os.makedirs(os.path.join(self.BASE_MODEL_DIR, "sklearn"), exist_ok=True)
        os.makedirs(os.path.join(self.BASE_MODEL_DIR, "lstm"), exist_ok=True)
        os.makedirs(os.path.join(self.BASE_MODEL_DIR, "xgboost"), exist_ok=True)
        os.makedirs(os.path.join(self.BASE_MODEL_DIR, "lstm_model_package"), exist_ok=True)

        self.data = data
        self.close_prices = []
        self.open_times   = []
        self.volatility = []

        self.open = []
        self.high = []
        self.low = []
        self.close = []
        self.predicted_prices = []
        self.xgb_sigma = []
        self.xgb_mu = []

        self.xgb_timestamps = None
        self.model_name = model_name
        self.model_path : str = ""
        self.model_type = ""
        print(f"[Core_Logic] Obliczanie predykcji.")



        match self.check_model_type():
            case "sklearn":
                self.model = joblib.load(self.model_path)

                if realtime:
                    self.features = self.model.feature_names_in_
                    self.predictions = []
                    self.class_map = ["buy", "hold", "sell"]
                else:
                    self.predictions, self.class_map = self.sklearn()

            case "lstm":
                self.model = ModelPredictor(self.model_path)

                if realtime:
                    print("[ModelSignal][ERROR] Brak obsługi tego typu modelu w realtime.")
                else:
                    self.predictions, self.class_map = self.lstm_price_predictor()



            case "xgboost":
                # self.model = joblib.load(self.model_path)

                if realtime:
                    print("[ModelSignal][ERROR] Brak obsługi tego typu modelu w realtime.")
                else:
                    self.predictions, self.class_map = self.xgboost()

            case "lstm_model_package":
                if realtime:
                    print("[ModelSignal][ERROR] Brak obsługi tego typu modelu w realtime.")
                else:
                    self.predictions, self.class_map = self.lstm_model_package()


            case _:
                print("[ModelSignal][ERROR] Brak obsługi tego typu modelu.")




    def process_models_data(self, num_ticks):
        print(f"[Core_Logic] Ekstrakcja danych.")

        if self.model_type in ("xgboost", "lstm_model_package"):
            valid_timestamps = self.xgb_timestamps[-num_ticks:]
            mask = self.data['open_time'].isin(valid_timestamps)
            self.data = self.data[mask].reset_index(drop=True)
        else:
            self.data = self.data.iloc[-num_ticks:].reset_index(drop=True)

        self.open_times = self.data["open_time"].to_numpy()
        self.close_prices = self.data["close"].to_numpy()

        #TODO: range dzielenia per symbol
        self.close = self.data["close"].to_numpy()
        self.high = self.data["high"].to_numpy()
        self.low = self.data["low"].to_numpy()
        self.open = self.data["open"].to_numpy()



        previous_close = self.data['close'].shift(1)
        true_range_pct = np.maximum(
            (self.data['high'] - self.data['low']) / self.data['close'],
            np.abs(self.data['high'] - previous_close) / previous_close
        )

        # Krok 2: Średnia z ostatnich 20 minut
        self.data['volatility'] = true_range_pct.rolling(window=20).mean()

        # Krok 3: Wypełnienie NaN (bo rolling robi dziury na początku)
        self.volatility = self.data["volatility"].to_numpy()
        self.data = None


    def generate_prediction(self, data):
        match self.model_type:
            case "sklearn":
                self.realtime_sklearn(data)
            case _:
                print("[ModelSignal][ERROR] Brak obsługi tego typu modelu.")


    def sklearn(self):
        features = self.model.feature_names_in_
        # print(features)
        # self.data["target"] = 0
        x = self.data[features].values

        probs = self.model.predict_proba(x)
        classes = [str(c) for c in self.model.classes_]
        class_map = {label: idx for idx, label in enumerate(classes)}
        return probs, class_map

    def realtime_sklearn(self,df):

        self.data.append(df)
        features = self.model.feature_names_in_

        df_row = pd.DataFrame([df])

        x = df_row[features].values

        prob = self.model.predict_proba(x)

        self.predictions.append(prob[0])
        self.open_times.append(df["open_time"])
        self.close_prices.append(df["close"])

        self.close.append(df["close"])
        self.high.append(df["high"])
        self.low.append(df["low"])
        self.open.append(df["open"])


    def lstm_price_predictor(self):
        preds = self.model.predict_batch(
            self.data,
            batch_size=256,
            include_signals=True # ← Wyłącz sygnały techniczne!
        )
        signal_model = ProbabilisticSignalGenerator(threshold=0.002)


        # Prosty fallback - oblicz z price_change
        signals = []
        prices = []
        for p in preds:

            result = signal_model.generate_signal(
            current_price=p["current_price"],
            predicted_price=p["predicted_price"],
            price_series=self.data["close"]
            )
            buy = result["p_buy"]
            hold = 0.0
            sell = result["p_sell"]
            signals.append([buy, hold, sell])
            # log_ret = np.log(predicted / current)
            # conf = min(abs(log_ret) * 50, 1.0)
            # if log_ret > 0:
            #     signals.append([conf, 1 - conf, 0.0])
            # elif log_ret < 0:
            #     signals.append([0.0, 1 - conf, conf])
            # else:
            #     signals.append([0.0, 1.0, 0.0])
            prices.append(p["predicted_price"])

        classes = ["buy", "hold", "sell"]
        self.predicted_prices = prices
        return np.array(signals), classes

    def xgboost(self):

        cfg = SignalConfig(threshold=0.0011, window_sigma=200)
        pkg = XGBModelPackage(self.model_path, signal_config=cfg)
        df_signals = pkg.predict_batch(df=self.data,data_mode='auto')

        self.xgb_timestamps = df_signals['timestamp'].to_numpy()
        self.xgb_mu = df_signals['mu'].to_numpy()
        self.xgb_sigma = df_signals['sigma'].to_numpy()
        p_buy = (df_signals['signal_buy'] / 100.0).to_numpy()
        p_sell = (df_signals['signal_sell'] / 100.0).to_numpy()
        p_hold = np.clip(1.0 - p_buy - p_sell, 0.0, 1.0)

        predictions = np.column_stack([p_buy, p_hold, p_sell])
        class_map = {"buy": 0, "hold": 1, "sell": 2}
        return predictions, class_map

    def lstm_model_package(self):
        threshold = 0.001
        config_path = os.path.join(self.model_path, 'model_config.json')
        if os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    cfg_json = json.load(f)
                threshold = cfg_json.get('signal_threshold', threshold)
            except Exception:
                pass

        # Prefer newest package from crypto_zulf; keep v2/v1 as fallbacks.
        if LSTMModelPackageLatest is not None:
            try:
                cfg = LSTMSignalConfigLatest(threshold=threshold, window_sigma=100)
                pkg = LSTMModelPackageLatest(self.model_path, signal_config=cfg)
                df_signals = pkg.predict_batch(df=self.data, data_mode='auto')
                return self._predictions_from_signal_df(
                    df_signals, package_name="lstm_model_package_latest"
                )
            except Exception as e:
                print(
                    f"[ModelSignal][WARNING] lstm_model_package_latest failed for "
                    f"{self.model_name}: {e}. Falling back."
                )

        if LSTMModelPackage2 is not None:
            try:
                cfg = LSTMSignalConfig2(threshold=threshold, window_sigma=100)
                pkg = LSTMModelPackage2(self.model_path, signal_config=cfg)
                df_signals = pkg.predict_batch(df=self.data, data_mode='auto')
                return self._predictions_from_signal_df(
                    df_signals, package_name="lstm_model_package2"
                )
            except Exception as e:
                print(
                    f"[ModelSignal][WARNING] lstm_model_package2 failed for "
                    f"{self.model_name}: {e}. Falling back to lstm_model_package."
                )

        cfg = LSTMSignalConfig1(threshold=threshold, window_sigma=100)
        pkg = LSTMModelPackage1(self.model_path, signal_config=cfg)
        df_signals = pkg.predict_batch(df=self.data, data_mode='auto')
        return self._predictions_from_signal_df(
            df_signals, package_name="lstm_model_package"
        )

    def _predictions_from_signal_df(self, df_signals, package_name):
        self.xgb_timestamps = df_signals['timestamp'].to_numpy()
        self.xgb_mu = df_signals['mu'].to_numpy()
        self.xgb_sigma = df_signals['sigma'].to_numpy()
        p_buy = (df_signals['signal_buy'] / 100.0).to_numpy()
        p_sell = (df_signals['signal_sell'] / 100.0).to_numpy()
        p_hold = np.clip(1.0 - p_buy - p_sell, 0.0, 1.0)
        predictions = np.column_stack([p_buy, p_hold, p_sell])
        class_map = {"buy": 0, "hold": 1, "sell": 2}
        print(f"[ModelSignal] Successfully loaded model {self.model_name} with {package_name}")
        return predictions, class_map

    def check_model_type(self):
        """ Wyszukuje model w podfolderach katalogu models i zwraca typ modelu oraz pełną ścieżkę."""

        for subfolder in os.listdir(self.BASE_MODEL_DIR):
            subfolder_path = os.path.join(self.BASE_MODEL_DIR, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            candidate_path = os.path.join(subfolder_path, self.model_name)
            if os.path.exists(candidate_path):
                self.model_path = candidate_path
                if subfolder.lower() in ("lstm", "lstm_lr_models", "lstm_models") and os.path.exists(os.path.join(candidate_path, "model_config.json")):
                    self.model_type = "lstm_model_package"
                    return "lstm_model_package"
                self.model_type = subfolder.lower()
                return subfolder.lower()  # 'sklearn' lub 'lstm' lub 'xgboost'

        raise ValueError(f"Nie znaleziono modelu '{self.model_name}' w żadnym z podfolderów {os.listdir(self.BASE_MODEL_DIR)}")

    def __getstate__(self):
        pickle_keys = {
            'model_name', 'model_path', 'model_type', 'class_map',
            'predictions', 'close_prices', 'close',
            'xgb_mu', 'xgb_sigma', 'xgb_timestamps', 'volatility'
        }
        state = {k: v for k, v in self.__dict__.items() if k in pickle_keys}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for key in ['open_times', 'open', 'high', 'low', 'predicted_prices', 'data', 'model']:
            if key not in self.__dict__:
                self.__dict__[key] = None
