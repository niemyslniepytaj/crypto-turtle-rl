"""Przyczynowosc sygnalu modelu predykcyjnego: zmiana swiec po chwili t nie moze zmienic sygnalu do t.

Test podmienia wszystkie swiece po punkcie CUT (ceny, wolumen, wolumen kupujacych) i sprawdza,
ze mu, sigma oraz sygnaly kupna i sprzedazy dla swiec do CUT sa identyczne jak dla oryginalnych danych.
Dla kontroli sprawdza tez, ze sygnal po CUT sie zmienia (test jest czuly).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

DATA_DIR = os.path.join(BASE, "shared", "files", "processed_simulation_data")
MODELS_DIR = os.path.join(BASE, "shared", "models", "lstm_lr_models")
CASES = {
    "ETH": ("ETHUSDT_1m_20250511_20260831.csv", "LR1449_eth_model_E114_2709"),
    "BTC": ("BTCUSDT_1m_20250511_20260831.csv", "LR1414_btc_model_B104_2209"),
}
N_ROWS = 9000
CUT = 6000


@pytest.mark.parametrize("symbol", sorted(CASES))
def test_signal_does_not_depend_on_future_candles(symbol):
    csv_name, model_name = CASES[symbol]
    csv_path = os.path.join(DATA_DIR, csv_name)
    model_path = os.path.join(MODELS_DIR, model_name)
    if not (os.path.exists(csv_path) and os.path.isdir(model_path)):
        pytest.skip("brak danych lub modelu")
    try:
        from shared.models.lstm_model_package_latest import LSTMModelPackage, SignalConfig
    except ImportError:
        pytest.skip("brak pakietu lstm_model_package_latest")

    df = pd.read_csv(csv_path, nrows=N_ROWS)
    pkg = LSTMModelPackage(model_path, signal_config=SignalConfig(threshold=0.001, window_sigma=100))
    base = pkg.predict_batch(df=df.copy(), data_mode="auto")

    rng = np.random.default_rng(0)
    future = df.copy()
    for col in ("open", "high", "low", "close"):
        future.loc[CUT:, col] = future.loc[CUT:, col].values * (1 + rng.normal(0, 0.02, N_ROWS - CUT))
    future.loc[CUT:, "volume"] = future.loc[CUT:, "volume"].values * 3
    future.loc[CUT:, "taker_buy_base_asset_volume"] = future.loc[CUT:, "taker_buy_base_asset_volume"].values * 0.2
    changed = pkg.predict_batch(df=future.copy(), data_mode="auto")

    cut_ts = pd.to_datetime(df["open_time"].iloc[CUT - 1], utc=True)
    t_base = pd.to_datetime(base["timestamp"], utc=True)
    t_chg = pd.to_datetime(changed["timestamp"], utc=True)
    before_base = base[t_base <= cut_ts].reset_index(drop=True)
    before_chg = changed[t_chg <= cut_ts].reset_index(drop=True)

    assert len(before_base) == len(before_chg) > 1000
    for col in ("mu", "sigma", "signal_buy", "signal_sell"):
        assert np.array_equal(before_base[col].values, before_chg[col].values, equal_nan=True), col

    after_base = base[t_base > cut_ts]["mu"].values
    after_chg = changed[t_chg > cut_ts]["mu"].values
    n = min(len(after_base), len(after_chg))
    assert np.nanmax(np.abs(after_base[:n] - after_chg[:n])) > 0.0
