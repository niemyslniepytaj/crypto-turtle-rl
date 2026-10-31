"""
Quick smoke verification and benchmark for lstm_model_package.
Tests loading, validation, raw mode, prepared mode, calibration, and throughput.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Ensure package is importable directly from any working directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PARENT = os.path.dirname(CURRENT_DIR)
while CURRENT_DIR in sys.path:
    sys.path.remove(CURRENT_DIR)
if PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, PACKAGE_PARENT)

from lstm_model_package import LSTMModelPackage, SignalConfig
REPO_ROOT = Path(PACKAGE_PARENT)


def check_dependencies() -> bool:
    required = ['torch', 'numpy', 'pandas', 'scipy', 'sklearn', 'joblib']
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[FAIL] Missing package: {pkg}")
            return False
    return True


def _generate_synthetic_ohlcv(n_rows: int = 1500) -> pd.DataFrame:
    base_time = 1735689600000
    times = [base_time + i * 60000 for i in range(n_rows)]
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.001, size=n_rows)
    close = 2500.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + rng.uniform(0, 0.002, size=n_rows))
    low = close * (1.0 - rng.uniform(0, 0.002, size=n_rows))
    open_p = close * (1.0 + rng.normal(0, 0.0005, size=n_rows))
    volume = rng.uniform(10, 100, size=n_rows)
    taker_vol = volume * rng.uniform(0.4, 0.6, size=n_rows)
    return pd.DataFrame({
        'open_time': times,
        'open': open_p,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'taker_buy_base_asset_volume': taker_vol,
    })


def run_smoke_tests(model_path: str | None = None):
    print("=== LSTMModelPackage Smoke Verification ===")
    assert check_dependencies(), "Dependency check failed."
    print("  [PASS] All required packages present.")

    models_dir = REPO_ROOT / 'lstm_logreturn_model' / 'lstm_lr_models'
    e104_dir = REPO_ROOT / 'data_preparation' / 'prepared_data' / 'E104_prep_eth_universal_full_1m'
    raw_csv = REPO_ROOT / 'data_preparation' / 'eth_1m_BP.csv'

    test_model = Path(model_path) if model_path is not None else (models_dir / 'LR1107_eth_model_E104_1107')
    assert test_model.exists(), f"Model {test_model} not found."

    pkg = LSTMModelPackage(str(test_model), signal_config=SignalConfig(threshold=0.001, window_sigma=100))
    info = pkg.get_model_info()
    print(f"  [PASS] Loaded model {info['model_id']} (arch: {info['rnn_type']}, calibrated: {info['calibrated']})")

    # 1. Prepared mode test (or fallback to synthetic raw if outside repo)
    if (e104_dir / 'test.csv').exists():
        df_test = pd.read_csv(e104_dir / 'test.csv', nrows=500)
        out_prep = pkg.predict_batch(df_test, data_mode='prepared')
        assert len(out_prep) > 0, "Empty prepared batch output"
        assert {'timestamp', 'close', 'mu', 'sigma', 'signal_buy', 'signal_sell'}.issubset(out_prep.columns)
        print(f"  [PASS] predict_batch (prepared): {len(out_prep)} rows generated.")

        # 2. Live signal test on prepared
        live_sig = pkg.predict_live(df_test, data_mode='prepared')
        assert 0.0 <= live_sig['signal_buy'] <= 100.0
        assert 0.0 <= live_sig['signal_sell'] <= 100.0
        assert live_sig['timestamp'] == df_test['open_time'].iloc[-1], "Timestamp mismatch with latest candle"
        assert abs(live_sig['close'] - float(df_test['close'].iloc[-1])) < 1e-4, "Close price mismatch with latest candle"
        print(f"  [PASS] predict_live: buy={live_sig['signal_buy']:.1f}%, sell={live_sig['signal_sell']:.1f}% (aligned with latest candle)")
    else:
        print("  [INFO] Prepared test.csv not found; running in standalone mode with synthetic OHLCV data.")

    # 3. Raw OHLCV mode test
    if raw_csv.exists():
        df_raw = pd.read_csv(raw_csv, nrows=1200)
    else:
        df_raw = _generate_synthetic_ohlcv(1500)

    # Test return_df=False fast path
    out_dict = pkg.predict_batch(df_raw, data_mode='raw', return_df=False)
    assert isinstance(out_dict, dict) and 'mu' in out_dict and isinstance(out_dict['mu'], np.ndarray)
    assert len(out_dict['mu']) > 0
    print(f"  [PASS] return_df=False fast dict path verified ({len(out_dict['mu'])} samples).")

    out_raw = pkg.predict_batch(df_raw, data_mode='raw')
    assert len(out_raw) > 0, "Empty raw batch output"
    live_raw = pkg.predict_live(df_raw, data_mode='raw')
    assert live_raw['timestamp'] == df_raw['open_time'].iloc[-1], "Raw live timestamp mismatch"
    assert abs(live_raw['close'] - float(df_raw['close'].iloc[-1])) < 1e-4, "Raw live close mismatch"
    print(f"  [PASS] predict_batch & live (raw): {len(out_raw)} rows generated, latest candle aligned.")

    # 4. SWA and from_directory factory test (LR1016 if present)
    m1016 = models_dir / 'LR1016_eth_model_E104_0907'
    if m1016.exists() and (m1016 / 'lstm_model_swa.pth').exists():
        pkg_swa = LSTMModelPackage.from_directory(str(m1016), checkpoint='swa')
        out_swa = pkg_swa.predict_batch(df_raw.head(350), data_mode='raw')
        assert len(out_swa) > 0
        print(f"  [PASS] from_directory factory and checkpoint='swa' verified on LR1016.")

    # 5. Vol-norm model test (LR1364) if present
    m1364 = models_dir / 'LR1364_eth_model_E109_2009'
    e109_test = REPO_ROOT / 'data_preparation' / 'prepared_data' / 'E109_prep_eth_universal_max_1m' / 'test.csv'
    if m1364.exists() and e109_test.exists():
        pkg1364 = LSTMModelPackage.from_directory(str(m1364))
        df_e109 = pd.read_csv(e109_test, nrows=300)
        out1364 = pkg1364.predict_batch(df_e109, data_mode='prepared')
        assert len(out1364) > 0 and not out1364['mu'].isna().any()
        print(f"  [PASS] vol_norm model LR1364 denormalized mu successfully.")

    # 6. Throughput benchmark (prepared inference vs raw end-to-end)
    if (e104_dir / 'test.csv').exists():
        repeats = int(np.ceil(200000 / len(df_test)))
        df_large_prep = pd.concat([df_test] * repeats, ignore_index=True).iloc[:200000].copy()
        t0 = time.perf_counter()
        out_prep_bm = pkg.predict_batch(df_large_prep, data_mode='prepared', batch_size=4096)
        el_prep = time.perf_counter() - t0
        tput_prep = len(out_prep_bm) / el_prep
        print(f"  [PASS] Prepared Inference Throughput: {len(out_prep_bm):,} samples in {el_prep:.2f}s ({tput_prep:,.0f} samples/s).")

    repeats_raw = int(np.ceil(50000 / len(df_raw)))
    df_large_raw = pd.concat([df_raw] * repeats_raw, ignore_index=True).iloc[:50000].copy()
    t0 = time.perf_counter()
    out_raw_bm = pkg.predict_batch(df_large_raw, data_mode='raw', batch_size=4096)
    el_raw = time.perf_counter() - t0
    tput_raw = len(out_raw_bm) / el_raw
    print(f"  [PASS] Raw End-to-End Throughput (FE 3.0 + Inference): {len(out_raw_bm):,} samples in {el_raw:.2f}s ({tput_raw:,.0f} samples/s).")

    print("\n[V] ALL SMOKE TESTS AND BENCHMARKS PASSED SUCCESSFULLY!")


if __name__ == '__main__':
    custom_model = sys.argv[1] if len(sys.argv) > 1 else None
    run_smoke_tests(custom_model)
