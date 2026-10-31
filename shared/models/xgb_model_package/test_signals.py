"""
Quick test for XGBModelPackage signal generation.

Tests four input scenarios against model X064 (E101 prepared data):
  1. Prepared CSV  — test.csv   (data_mode='prepared')
  2. Prepared CSV  — sym.csv    (data_mode='prepared', different time split)
  3. Auto-detect   — test.csv   (data_mode='auto', must resolve to 'prepared')
  4. Raw OHLCV     — reconstructed from test.csv close + OHLCV cols (data_mode='raw')

Run from the xgb_model directory:
    cd claude_xgb/xgb_model
    python ../xgb_model_package/test_signals.py
"""

import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allow running from xgb_model/ or project root
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))          # xgb_model_package/
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)                         # claude_xgb/
_XGB_MODEL    = os.path.join(_PROJECT_ROOT, 'xgb_model')            # xgb_model/

# Add package dir to path so 'from xgb_model_package import ...' works
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PREPARED_DIR = os.path.join(_XGB_MODEL, 'xgb_prepared_data', 'E101_prep_eth_universal_full_1m')
MODEL_DIR    = os.path.join(_XGB_MODEL, 'xgb_models', 'X064_eth_xgb_model_E101_0903')

TEST_CSV     = os.path.join(PREPARED_DIR, 'test.csv')
SYM_CSV      = os.path.join(PREPARED_DIR, 'sym.csv')

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from xgb_model_package import XGBModelPackage, SignalConfig


# ===========================================================================
# HELPERS
# ===========================================================================

SEP  = '=' * 72
SEP2 = '-' * 72

def header(text: str):
    print(f'\n{SEP}')
    print(f'  {text}')
    print(SEP)

def subheader(text: str):
    print(f'\n{SEP2}')
    print(f'  {text}')
    print(SEP2)

def ok(msg):  print(f'  [OK]   {msg}')
def err(msg): print(f'  [FAIL] {msg}')
def info(msg): print(f'         {msg}')


def summarise_signals(df: pd.DataFrame, label: str):
    """Print compact statistics for a signals DataFrame."""
    subheader(f'Signal summary — {label}')
    n = len(df)
    print(f'  Rows:           {n}')
    if n == 0:
        err('Empty output — no signals produced.')
        return

    # mu
    print(f'\n  mu  (log-return prediction):')
    print(f'    mean   {df["mu"].mean():.6f}')
    print(f'    std    {df["mu"].std():.6f}')
    print(f'    min    {df["mu"].min():.6f}')
    print(f'    max    {df["mu"].max():.6f}')

    # sigma
    print(f'\n  sigma (rolling volatility):')
    print(f'    mean   {df["sigma"].mean():.6f}')
    print(f'    std    {df["sigma"].std():.6f}')
    print(f'    NaN    {df["sigma"].isna().sum()}')

    # signals
    print(f'\n  signal_buy  [0-100]:')
    print(f'    mean   {df["signal_buy"].mean():.2f}')
    print(f'    median {df["signal_buy"].median():.2f}')
    print(f'    >50%%  {(df["signal_buy"] > 50).sum()} ({(df["signal_buy"] > 50).mean()*100:.1f}%%)')
    print(f'    >70%%  {(df["signal_buy"] > 70).sum()} ({(df["signal_buy"] > 70).mean()*100:.1f}%%)')

    print(f'\n  signal_sell [0-100]:')
    print(f'    mean   {df["signal_sell"].mean():.2f}')
    print(f'    median {df["signal_sell"].median():.2f}')
    print(f'    >50%%  {(df["signal_sell"] > 50).sum()} ({(df["signal_sell"] > 50).mean()*100:.1f}%%)')
    print(f'    >70%%  {(df["signal_sell"] > 70).sum()} ({(df["signal_sell"] > 70).mean()*100:.1f}%%)')

    # Independence check: sum should be < 100 for all rows (hold zone exists)
    sums = df['signal_buy'] + df['signal_sell']
    violations = (sums > 100.01).sum()
    print(f'\n  BUY+SELL independence:')
    print(f'    mean sum  {sums.mean():.2f}  (expected < 100)')
    print(f'    >100 rows {violations}  (expected 0)')
    if violations == 0:
        ok('BUY + SELL < 100 for all rows')
    else:
        err(f'{violations} rows where BUY+SELL > 100 — signal formula issue!')

    # Timestamp sanity
    if 'timestamp' in df.columns:
        print(f'\n  timestamp range:')
        print(f'    first  {df["timestamp"].iloc[0]}')
        print(f'    last   {df["timestamp"].iloc[-1]}')

    # First 3 rows
    print(f'\n  First 3 rows:')
    cols = ['timestamp', 'mu', 'sigma', 'signal_buy', 'signal_sell']
    cols = [c for c in cols if c in df.columns]
    with pd.option_context('display.float_format', '{:.6f}'.format,
                           'display.max_columns', 10, 'display.width', 120):
        print(df[cols].head(3).to_string(index=False, justify='left'))


def check_live(pkg: XGBModelPackage, df_slice: pd.DataFrame,
               mode: str, label: str):
    """Test predict_live on the last portion of df_slice."""
    subheader(f'predict_live — {label}')
    min_rows = pkg.min_rows_required
    tail     = df_slice.tail(min_rows + 20).copy()
    print(f'  Buffer size: {len(tail)} rows (min_rows_required={min_rows})')

    t0     = time.perf_counter()
    result = pkg.predict_live(tail, data_mode=mode)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f'  timestamp:   {result["timestamp"]}')
    print(f'  mu:          {result["mu"]:.6f}')
    print(f'  sigma:       {result["sigma"]:.6f}')
    print(f'  signal_buy:  {result["signal_buy"]:.2f}')
    print(f'  signal_sell: {result["signal_sell"]:.2f}')
    print(f'  Time:        {elapsed:.1f} ms')
    ok('predict_live returned a valid dict')


# ===========================================================================
# TEST CASES
# ===========================================================================

def test_model_loading():
    """Verify model loads cleanly and get_model_info works."""
    header('TEST 0 — Model loading & info')
    cfg = SignalConfig(threshold=0.001, window_sigma=100)
    pkg = XGBModelPackage(MODEL_DIR, signal_config=cfg)
    info_d = pkg.get_model_info()
    for k, v in info_d.items():
        if k not in ('train_metrics', 'val_metrics'):
            print(f'  {k:<22} {v}')
    ok('Model loaded successfully')
    return pkg


def test_prepared_test_csv(pkg: XGBModelPackage):
    """Test 1: prepared test.csv — explicit mode='prepared', full dataset."""
    header('TEST 1 — Prepared data: test.csv  (data_mode="prepared")')

    print(f'  Loading {TEST_CSV} ...')
    t0 = time.perf_counter()
    df = pd.read_csv(TEST_CSV)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f'  Rows: {len(df):,}   Load time: {load_ms:.0f} ms')

    t0 = time.perf_counter()
    result = pkg.predict_batch(df, data_mode='prepared')
    batch_ms = (time.perf_counter() - t0) * 1000

    ok(f'predict_batch completed in {batch_ms:.0f} ms')
    print(f'  Speed: {len(result) / (batch_ms/1000):,.0f} signals/sec')
    summarise_signals(result, 'test.csv prepared')
    check_live(pkg, df, mode='prepared', label='test.csv prepared')
    return result


def test_prepared_sym_csv(pkg: XGBModelPackage):
    """Test 2: prepared sym.csv — different date range (most recent data)."""
    header('TEST 2 — Prepared data: sym.csv  (data_mode="prepared")')

    print(f'  Loading {SYM_CSV} ...')
    t0 = time.perf_counter()
    df = pd.read_csv(SYM_CSV)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f'  Rows: {len(df):,}   Load time: {load_ms:.0f} ms')

    t0 = time.perf_counter()
    result = pkg.predict_batch(df, data_mode='prepared')
    batch_ms = (time.perf_counter() - t0) * 1000

    ok(f'predict_batch completed in {batch_ms:.0f} ms')
    print(f'  Speed: {len(result) / (batch_ms/1000):,.0f} signals/sec')
    summarise_signals(result, 'sym.csv prepared')
    check_live(pkg, df, mode='prepared', label='sym.csv prepared')
    return result


def test_auto_detect(pkg: XGBModelPackage):
    """Test 3: data_mode='auto' on prepared test.csv — must resolve to 'prepared'."""
    header('TEST 3 — Auto-detect on prepared test.csv  (data_mode="auto")')

    df = pd.read_csv(TEST_CSV)
    print(f'  Rows: {len(df):,}')

    t0 = time.perf_counter()
    result = pkg.predict_batch(df, data_mode='auto')
    batch_ms = (time.perf_counter() - t0) * 1000

    ok(f'Auto-detect resolved correctly, {batch_ms:.0f} ms')
    print(f'  Rows out: {len(result):,}')
    if len(result) > 0:
        ok('Signals produced — auto resolved to prepared mode')
    else:
        err('No signals produced in auto mode')
    return result


def test_raw_ohlcv(pkg: XGBModelPackage):
    """
    Test 4: raw OHLCV mode.
    Reconstruct raw OHLCV by keeping only [open_time, open, high, low, close, volume]
    from test.csv. The prepared CSV contains these cols with raw (unscaled) prices,
    so this is a valid raw OHLCV feed.
    """
    header('TEST 4 — Raw OHLCV mode  (data_mode="raw")')

    df_full = pd.read_csv(TEST_CSV)
    raw_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    missing_raw = [c for c in raw_cols if c not in df_full.columns]
    if missing_raw:
        err(f'Skipped: raw OHLCV columns missing from test.csv: {missing_raw}')
        return None

    df_raw = df_full[raw_cols].copy()
    # Use a manageable slice to keep the test fast (~5 000 rows)
    # Still needs min_rows_required + FE warmup
    n_slice = min(5000, len(df_raw))
    df_slice = df_raw.tail(n_slice).reset_index(drop=True)
    print(f'  Using last {n_slice} rows of test.csv as raw OHLCV')

    t0 = time.perf_counter()
    result = pkg.predict_batch(df_slice, data_mode='raw')
    batch_ms = (time.perf_counter() - t0) * 1000

    ok(f'predict_batch (raw) completed in {batch_ms:.0f} ms')
    print(f'  Rows out: {len(result):,}  Speed: {len(result) / max(batch_ms/1000, 0.001):,.0f} signals/sec')
    summarise_signals(result, 'test.csv raw OHLCV slice')
    check_live(pkg, df_slice, mode='raw', label='raw OHLCV')
    return result


def test_validation_errors(pkg: XGBModelPackage):
    """
    Test 5: Verify validation raises meaningful errors for bad inputs.
    Each sub-test expects a ValueError.
    """
    header('TEST 5 — Validation error handling')

    df_ok = pd.read_csv(TEST_CSV)

    # 5a: Too few rows (prepared)
    subheader('5a — Too few rows (prepared mode)')
    try:
        pkg.predict_batch(df_ok.head(10), data_mode='prepared')
        err('Should have raised ValueError')
    except ValueError as e:
        ok(f'ValueError raised: {str(e)[:80]}')

    # 5b: Missing feature column (prepared)
    subheader('5b — Missing feature column (prepared mode)')
    df_missing = df_ok.drop(columns=['return_1'])
    try:
        pkg.predict_batch(df_missing, data_mode='prepared')
        err('Should have raised ValueError')
    except ValueError as e:
        ok(f'ValueError raised: {str(e)[:80]}')

    # 5c: Too few rows (raw mode)
    subheader('5c — Too few rows (raw mode)')
    raw_cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    df_raw_tiny = df_ok[raw_cols].head(50)
    try:
        pkg.predict_batch(df_raw_tiny, data_mode='raw')
        err('Should have raised ValueError')
    except ValueError as e:
        ok(f'ValueError raised: {str(e)[:80]}')

    # 5d: Missing OHLCV column (raw mode)
    subheader('5d — Missing OHLCV column (raw mode)')
    df_no_volume = df_ok[['open_time', 'open', 'high', 'low', 'close']].copy()
    try:
        pkg.predict_batch(df_no_volume, data_mode='raw')
        err('Should have raised ValueError')
    except ValueError as e:
        ok(f'ValueError raised: {str(e)[:80]}')

    # 5e: NaN in close
    subheader('5e — NaN in close column')
    df_nan = df_ok.copy()
    df_nan.loc[100, 'close'] = np.nan
    try:
        pkg.predict_batch(df_nan, data_mode='prepared')
        err('Should have raised ValueError')
    except ValueError as e:
        ok(f'ValueError raised: {str(e)[:80]}')

    # 5f: Negative close value
    subheader('5f — Negative close value')
    df_neg = df_ok.copy()
    df_neg.loc[100, 'close'] = -1.0
    try:
        pkg.predict_batch(df_neg, data_mode='prepared')
        err('Should have raised ValueError')
    except ValueError as e:
        ok(f'ValueError raised: {str(e)[:80]}')


def test_signal_config_variants(pkg_default: XGBModelPackage):
    """
    Test 6: Different SignalConfig values — verify signals change meaningfully.
    Uses a fixed 500-row prepared slice for speed.
    """
    header('TEST 6 — SignalConfig variants')

    df = pd.read_csv(TEST_CSV, nrows=2000)

    configs = [
        SignalConfig(threshold=0.0005, window_sigma=50),
        SignalConfig(threshold=0.001,  window_sigma=100),   # default
        SignalConfig(threshold=0.002,  window_sigma=100),
        SignalConfig(threshold=0.003,  window_sigma=200),
    ]

    for cfg in configs:
        pkg = XGBModelPackage(MODEL_DIR, signal_config=cfg)
        result = pkg.predict_batch(df, data_mode='prepared')
        subheader(f'threshold={cfg.threshold}  window_sigma={cfg.window_sigma}')
        if len(result) == 0:
            err('No signals produced')
            continue
        print(f'  Rows out:        {len(result)}')
        print(f'  avg buy signal:  {result["signal_buy"].mean():.2f}')
        print(f'  avg sell signal: {result["signal_sell"].mean():.2f}')
        print(f'  avg sigma:       {result["sigma"].mean():.6f}')
        ok('Config applied correctly')


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print(SEP)
    print('  XGBModelPackage — Signal generation tests')
    print(f'  Prepared data : {PREPARED_DIR}')
    print(f'  Model         : {MODEL_DIR}')
    print(SEP)

    # Pre-flight checks
    for path, label in [
        (PREPARED_DIR, 'Prepared data dir'),
        (MODEL_DIR,    'Model dir'),
        (TEST_CSV,     'test.csv'),
        (SYM_CSV,      'sym.csv'),
    ]:
        if not os.path.exists(path):
            print(f'[ABORT] {label} not found: {path}')
            sys.exit(1)
        ok(f'{label} found')

    results = {}
    errors  = []

    # --- Test 0: loading ---
    try:
        pkg = test_model_loading()
    except Exception as e:
        print(f'[ABORT] Model failed to load: {e}')
        traceback.print_exc()
        sys.exit(1)

    # --- Tests 1-6 ---
    for name, fn, args in [
        ('prepared_test',   test_prepared_test_csv,   (pkg,)),
        ('prepared_sym',    test_prepared_sym_csv,    (pkg,)),
        ('auto_detect',     test_auto_detect,         (pkg,)),
        ('raw_ohlcv',       test_raw_ohlcv,           (pkg,)),
        ('validation',      test_validation_errors,   (pkg,)),
        ('signal_configs',  test_signal_config_variants, (pkg,)),
    ]:
        try:
            res = fn(*args)
            results[name] = res
        except Exception as e:
            errors.append((name, e))
            err(f'Test "{name}" raised unexpected exception: {e}')
            traceback.print_exc()

    # --- Final summary ---
    header('SUMMARY')
    passed = len(results)
    failed = len(errors)
    total  = passed + failed
    print(f'  Passed: {passed} / {total}')
    if errors:
        print(f'\n  Failed tests:')
        for name, exc in errors:
            print(f'    {name}: {type(exc).__name__}: {exc}')
    else:
        ok('All tests passed')
    print(SEP + '\n')


if __name__ == '__main__':
    main()
