import json
import os
import pandas as pd
from datetime import datetime, timedelta
from .core_logic.mp_signal import ModelSignal
from .core_logic.decision import DecisionEngine
from .core_logic.structures import CoreConfig, Order, DecisionConfig, WalletConfig

class RealtimeGenerateDecisionService:
    """Generowanie decyzji handlowych w trybie realtime na podstawie configu."""
    def __init__(self, base_dir = None):
        if base_dir is None: base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        self.symbols = []
        self.model_paths = {}
        self.core_config = CoreConfig()

        self.config_path = "/app/configuration/realtime_simulation_config.json"
        self._last_config_mtime = 0

        current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.data_dir = os.path.join(self.base_dir,"shared", "realtime_simulation", f"{current_time}")
        os.makedirs(self.data_dir, exist_ok=True)
        self.data_file_path = os.path.join(self.data_dir, f"realtime_data.csv")
        self.orders_file_path = os.path.join(self.data_dir, f"realtime_orders_logs.csv")

        self.models = {}
        self.core_config.symbols = self.symbols
        self.engine = DecisionEngine(
            agent_name=None,
            models=self.models,
            base_dir=self.base_dir,
            core_mode="sim_mp",
            core_config=self.core_config
        )

    def live_decision(
        self,
        rabbit_msg_id=None,
        results=None,
    ):
        try:

            self._check_and_load_config()
            for result in results:
                data = result["data"]["realtime_data"][-1]
                symbol = result["symbol"].upper()
                print(symbol, "czas: ",datetime.now().isoformat())
                data["open_time"] = pd.to_datetime(data["open_time"], unit="ms")
                data["close_time"] = pd.to_datetime(data["close_time"], unit="ms").round("min")
                print(self.symbols, "przed if ",datetime.now().isoformat())
                if symbol not in self.symbols:
                    self.symbols.append(symbol)
                    self.models[symbol] = ModelSignal(
                        model_name=self.model_paths[symbol],
                        data=[],
                        base_dir=self.base_dir,
                        realtime=True
                    )

                    self.engine.init_local_configs(symbol)

                    with open(self.config_path, "r") as f:
                        symbols_cfg = json.load(f).get("symbols", {})
                    self.load_modified_params(symbols_cfg)

                    self.engine.wallet.initialize_symbol_and_holdings(symbol)

                self.models[symbol].generate_prediction(data)

            if hasattr(self.engine.wallet, "testnet_check_pending_orders"):
                self.engine.wallet.testnet_check_pending_orders()

            self.engine.generate_decisions(t=-1)

            if hasattr(self.engine.wallet, "testnet_sync_after_orders"):
                self.engine.wallet.testnet_sync_after_orders(
                    wait_seconds= 10.0
                )

            current_ts = results[0]["data"]["realtime_data"][-1]["close_time"]

            last_t_orders = self.engine.wallet.get_last_t_logs(close_time=current_ts)

            for symbol in self.symbols:

                last_row = self.models[symbol].data[-1]

                if last_row["close_time"] == current_ts:
                    symbol_file_path = os.path.join(self.data_dir, f"{symbol}.csv")
                    df_to_save = pd.DataFrame([last_row])

                    file_exists = os.path.isfile(symbol_file_path)
                    df_to_save.to_csv(symbol_file_path, mode='a', index=False, header=not file_exists)

            if last_t_orders:
                orders_df = pd.DataFrame([o for o in last_t_orders])
                orders_file_exists = os.path.isfile(self.orders_file_path)
                orders_df.to_csv(self.orders_file_path, mode='a', index=False, header=not orders_file_exists)

            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "success",
                "message": f"Successfully generate decision for timestamp: {datetime.now().isoformat()}",
                "order_logs": last_t_orders,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            import traceback
            print("[Realtime Generate Decision][ERROR]", e)
            traceback.print_exc()
            return {
                "rabbit_msg_id": rabbit_msg_id,
                "result": "failed",
                "message": str(e),
                "order_logs": None,
                "timestamp": datetime.now().isoformat()
            }

    def _check_and_load_config(self):
        """Sprawdza metadane pliku i przeładowuje konfigurację gdy nastąpiła zmiana."""
        try:
            if not os.path.exists(self.config_path):
                return

            current_mtime = os.path.getmtime(self.config_path)
            if current_mtime <= self._last_config_mtime:
                return

            print(f"[RealtimeGenerateDecision] Wykryto zmianę w konfiguracji. Przeładowuję...")
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)

            self.model_paths = {
                symbol.upper(): cfg.get("model")
                for symbol, cfg in config_data.get("symbols", {}).items()
                if cfg.get("model")
            }
            self.load_models()

            raw_core = config_data.get("core_config", {})
            if raw_core:
                self.load_core_config(raw_core)
                self.load_modified_params(config_data.get("symbols", {}))

            self._last_config_mtime = current_mtime

        except Exception as e:
            print(f"[RealtimeGenerateDecision] Błąd sprawdzania pliku: {e}")

    def load_models(self):
        for symbol in self.symbols:
            symbol_upper = symbol.upper()
            model_name = self.model_paths.get(symbol_upper)
            if not model_name:
                print(f"[RealtimeGenerateDecision] Brak modelu dla {symbol_upper} w configu — pomijam")
                continue
            self.models[symbol].model_name = model_name
            self.models[symbol].check_model_type()

    def load_core_config(self, raw_core):
        for field_name, value in raw_core.items():
            if field_name != "symbols" and hasattr(self.core_config, field_name):
                setattr(self.core_config, field_name, value)

        if self.engine:
            self.engine.experiment_version = self.core_config.experiment_version
            self.engine.apply_experiment_version()
            #Szybka łatka - do poprawy
            if self.engine.wallet.balance == 0.0:
                self.engine.wallet.balance = self.core_config.balance_start

    def load_modified_params(self,symbols_cfg: dict):
        for symbol in self.symbols:
            overrides = symbols_cfg.get(symbol, {}).get("modified_params", {})
            if overrides is not {}:
                for field_name, value in overrides.items():
                    if hasattr(self.engine.config[symbol], field_name):
                        setattr(self.engine.config[symbol], field_name, value)
                    if hasattr(self.engine.wallet.config[symbol], field_name):
                        setattr(self.engine.wallet.config[symbol], field_name, value)
                print(f"[RealtimeGenerateDecision] Załadowano modyfikacje DecisionConfig i WalletConfig")